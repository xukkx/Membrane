import logging
import re
import time
from typing import List, Dict, Any, Optional
from membrane.models import Scope, Event, KBItem
from membrane.storage.sqlite import SQLiteStorage
from membrane.storage.jsonl import JSONLStorage
from membrane.storage.ram import RAMStorage
from membrane.storage.policy import load_policy
from membrane.logic.canonicalize import generate_kb_item_id, generate_embedding_text

logger = logging.getLogger(__name__)

class IngestionEngine:
    def __init__(self, sqlite: SQLiteStorage, jsonl: JSONLStorage, ram: RAMStorage):
        self.sqlite = sqlite
        self.jsonl = jsonl
        self.ram = ram
        self.policy = load_policy()

        # Pre-compile regexes
        self.regex_camel = re.compile(r'\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b')
        self.regex_snake = re.compile(r'\b[a-z0-9]+(?:_[a-z0-9]+){1,}\b')
        self.regex_caps = re.compile(r'\b[A-Z0-9]{3,}(?:_[A-Z0-9]{2,})*\b')
        self.regex_path = re.compile(r'(?:[A-Za-z]:\\|/)[^\s\]\)\}\'"]+')
        self.regex_hex = re.compile(r'\b0x[0-9A-Fa-f]+\b')

    def add_event(self, event: Event):
        # 1. Add to SQLite
        # Derive search text: normalize(text) -> strip whitespace
        search_text = " ".join(event.text.split())

        self.sqlite.add_event(event.model_dump(), search_text=search_text)

        # 2. Append to JSONL (cold log)
        self.jsonl.append({
            "tool": "temp.add_event",
            "payload": event.model_dump(),
            "ts": event.ts_epoch_ms
        })

        # 3. Update RAM index (incremental)
        self.ram.add_temp_event(event.event_id, search_text)

    def promote(self, project_id: str, source_event_ids: List[str]) -> List[KBItem]:
        # 1. Fetch events
        events = []
        for eid in source_event_ids:
            edata = self.sqlite.get_event(eid)
            if edata:
                events.append(edata)

        if not events:
            return []

        # 2. Generate Summary Candidate
        boilerplate_prefixes = [
            "note:",
            "summary:",
            "observation:",
            "context:",
            "assistant:",
            "user:",
            "system:",
        ]
        parts = []
        for e in events:
            part = e['text'].lstrip()
            lowered = part.lower()
            for prefix in boilerplate_prefixes:
                if lowered.startswith(prefix):
                    part = part[len(prefix):].lstrip()
                    break
            normalized = " ".join(part.split())
            if normalized:
                parts.append(normalized)

        summary_text = " ".join(parts)[:500]

        base_ts = max(e["ts_epoch_ms"] for e in events)

        candidates = []

        # Summary item
        summary_content = {"scope": "project", "text": summary_text}
        summary_id = generate_kb_item_id(project_id, "summary", summary_content)
        candidates.append(KBItem(
            kb_item_id=summary_id,
            project_id=project_id,
            kind="summary",
            content=summary_content,
            provenance_event_ids=sorted(source_event_ids),
            confidence=0.6,
            status="proposed",
            ts_epoch_ms=base_ts
        ))

        # 3. Entity Candidates
        seen_entities = set()
        entity_candidates = []

        def add_candidate(name: str, type_hint: str):
            # Normalize: strip punctuation (regex handles boundaries), keep case
            # Dedupe case-insensitively
            normalized = name.strip(".,;:?!'\"")
            key = normalized.lower()
            if key not in seen_entities and len(normalized) > 0:
                seen_entities.add(key)
                content = {"name": normalized, "type": type_hint}
                eid = generate_kb_item_id(project_id, "entity", content)
                entity_candidates.append(KBItem(
                    kb_item_id=eid,
                    project_id=project_id,
                    kind="entity",
                    content=content,
                    provenance_event_ids=sorted(source_event_ids),
                    confidence=0.55,
                    status="proposed",
                    ts_epoch_ms=base_ts
                ))

        for e in events:
            text = e['text']

            for m in self.regex_camel.finditer(text): add_candidate(m.group(), "identifier")
            for m in self.regex_snake.finditer(text): add_candidate(m.group(), "identifier")
            for m in self.regex_caps.finditer(text): add_candidate(m.group(), "code")
            for m in self.regex_path.finditer(text): add_candidate(m.group(), "file")
            for m in self.regex_hex.finditer(text): add_candidate(m.group(), "hex")

        # Cap entities at 20
        candidates.extend(entity_candidates[:20])

        return candidates

    def commit(self, project_id: str, commits: List[KBItem]) -> Dict[str, Any]:
        results = {"upserted": 0, "conflicts": 0, "superseded": 0}

        for item in commits:
            existing_data = self.sqlite.get_kb_item(item.kb_item_id)

            final_item = item
            if existing_data:
                # Deserialize existing KBItem (partially done by get_kb_item returning dict)
                # We reconstruct KBItem to manipulate
                existing = KBItem(**existing_data)

                # Merge Provenance
                new_prov = sorted(list(set(existing.provenance_event_ids + item.provenance_event_ids)))
                final_item.provenance_event_ids = new_prov

                # Conflict Logic
                conflict_threshold = self.policy.conflict_threshold

                if existing.status == "verified":
                    if item.status == "verified":
                        # Verified vs Verified
                        if item.confidence >= existing.confidence + conflict_threshold:
                            # Supersede
                            final_item.supersedes = sorted(list(set(existing.supersedes + [existing.kb_item_id])))
                            results["superseded"] += 1
                        else:
                            # Conflict: keep existing, mark conflict on existing?
                            # Spec: "otherwise mark conflict and keep both."
                            # But kb_item_id is stable hash of CONTENT.
                            # If content is same (same ID), they are the SAME item.
                            # So "keep both" applies if IDs differ but they semantically conflict.
                            # BUT here we are upserting by ID. Same ID = Same Content.
                            # So there is no "conflict" in content, only metadata updates (confidence/status).
                            # If status/confidence update is rejected, we just update provenance and keep old status/conf.

                            # Wait, "committing same KB item twice results in one row".
                            # If content is identical, ID is identical.
                            # So we merge metadata.

                            # "Verified facts must not be overwritten silently."
                            # If proposed tries to overwrite verified -> Ignore status update.
                            # If verified tries to overwrite verified -> Check confidence.

                            if item.confidence < existing.confidence + conflict_threshold:
                                # Keep existing status/confidence
                                final_item.status = existing.status
                                final_item.confidence = existing.confidence
                                # Merge conflicts/supersedes
                                final_item.conflicts_with = sorted(list(set(existing.conflicts_with + item.conflicts_with)))
                                final_item.supersedes = sorted(list(set(existing.supersedes + item.supersedes)))
                                results["conflicts"] += 1 # Recorded as conflict/no-op update
                            else:
                                # Update accepted
                                results["upserted"] += 1

                    elif item.status == "proposed":
                        # Verified supersedes Proposed (ignore update to status/conf)
                        final_item.status = existing.status
                        final_item.confidence = existing.confidence
                        # Merge lists
                        final_item.conflicts_with = sorted(list(set(existing.conflicts_with + item.conflicts_with)))
                        final_item.supersedes = sorted(list(set(existing.supersedes + item.supersedes)))
                        # No count change, just provenance merge
                else:
                    # Existing is proposed.
                    if item.status == "verified":
                        # Verified supersedes proposed
                        results["upserted"] += 1
                    else:
                        # Proposed vs Proposed: Take max confidence? or New?
                        # Usually max confidence or just update.
                        # Spec doesn't strictly say. "Verified > Proposed".
                        if item.confidence > existing.confidence:
                             results["upserted"] += 1
                        else:
                             final_item.confidence = existing.confidence
                             results["upserted"] += 1

            else:
                results["upserted"] += 1

            # Execute Upsert
            search_text = generate_embedding_text(final_item.kind, final_item.content)
            self.sqlite.upsert_kb_item(final_item.model_dump(), search_text)
            self.ram.upsert_kb_item(final_item.kb_item_id, search_text)

            # Log commit
            self.jsonl.append({
                "tool": "proj_kb.commit",
                "payload": final_item.model_dump(),
                "ts": final_item.ts_epoch_ms
            })

        return results
