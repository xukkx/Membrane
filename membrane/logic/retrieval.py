import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from membrane.models import Scope, SearchResult, SearchFilters
from membrane.storage.sqlite import SQLiteStorage
from membrane.storage.ram import RAMStorage
from membrane.storage.policy import load_policy

logger = logging.getLogger(__name__)

class RetrievalEngine:
    def __init__(self, sqlite: SQLiteStorage, ram: RAMStorage):
        self.sqlite = sqlite
        self.ram = ram
        self.policy = load_policy()

    def _fusion_score(self, lex_norm: float, vec_score: float, item: Dict[str, Any]) -> float:
        w = self.policy.weights

        # Hybrid base score: lexical + vector.
        base = (w.get('w_lex', 0.3) * lex_norm) + (w.get('w_vec', 0.7) * vec_score)

        # Bonuses
        bonus = 0.0
        status = item.get('status')
        if status == 'verified':
            bonus += w.get('w_status', 0.1)

        conf = item.get('confidence', 0.0)
        bonus += w.get('w_conf', 0.1) * conf

        # Recency (Linear decay over 30 days)
        now = int(time.time() * 1000)
        ts = item.get('ts_epoch_ms', 0)
        age_ms = max(0, now - ts)
        max_age = 30 * 24 * 3600 * 1000 # 30 days
        recency = max(0.0, 1.0 - (age_ms / max_age))
        bonus += w.get('w_recency', 0.1) * recency

        return base + bonus

    def temp_search(self, scope: Scope, query: str, k: int = 20, filters: Optional[SearchFilters] = None) -> List[SearchResult]:
        # Cap k
        k = min(k, self.policy.retrieval_caps.get('max_k', 20))

        # 1. FTS Candidates
        candidates = self.sqlite.search_temp_events(
            query,
            scope.project_id,
            scope.agent_id,
            scope.session_id,
            limit=self.policy.retrieval_caps.get('fts_candidates', 200)
        )

        if not candidates:
            return []

        # 2. Vector Rerank
        query_vec = self.ram.encode([query])

        filtered = []
        for cand in candidates:
            if filters:
                if filters.type and cand['type'] != filters.type:
                    continue
                if filters.tags and not any(t in cand['tags'] for t in filters.tags):
                    continue
                if filters.entities and not any(e in cand['entities'] for e in filters.entities):
                    continue
                if filters.thread_id and cand['scope'].get('thread_id') != filters.thread_id:
                    continue
                if filters.task_id and cand['scope'].get('task_id') != filters.task_id:
                    continue
            filtered.append(cand)

        scored = []
        rank_pos = -1
        last_fts_rank = None
        for cand in filtered:
            fts_rank = float(cand.get('fts_rank', 0.0))
            if last_fts_rank is None or fts_rank != last_fts_rank:
                rank_pos += 1
                last_fts_rank = fts_rank

            vec = self.ram.get_temp_vector(cand['event_id'])
            cosine = 0.0
            if vec is not None and query_vec.size > 0:
                cosine = float(np.dot(query_vec, vec).flatten()[0])

            lex_norm = 1.0 / (1.0 + rank_pos)
            score = self._fusion_score(lex_norm, cosine, cand)
            scored.append((score, cand, cosine))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, cand, cosine in scored[:k]:
            results.append(SearchResult(
                id=cand['event_id'],
                source="temp",
                snippet=cand['text'][:200],
                score=score,
                why={
                    "fts_rank": cand.get('fts_rank'),
                    "cosine": cosine,
                    "status": "N/A",
                    "confidence": 1.0,
                    "scope": cand['scope']
                }
            ))

        return results

    def proj_kb_search(self, project_id: str, query: str, k: int = 20, filters: Optional[SearchFilters] = None) -> List[SearchResult]:
        k = min(k, self.policy.retrieval_caps.get('max_k', 20))

        candidates = self.sqlite.search_kb_items(
            query,
            project_id,
            limit=self.policy.retrieval_caps.get('fts_candidates', 200)
        )

        if not candidates:
            return []

        query_vec = self.ram.encode([query])

        filtered = []
        for cand in candidates:
            if filters and filters.status and cand['status'] != filters.status:
                continue
            filtered.append(cand)

        scored = []
        rank_pos = -1
        last_fts_rank = None
        for cand in filtered:
            fts_rank = float(cand.get('fts_rank', 0.0))
            if last_fts_rank is None or fts_rank != last_fts_rank:
                rank_pos += 1
                last_fts_rank = fts_rank

            vec = self.ram.get_kb_vector(cand['kb_item_id'])
            cosine = 0.0
            if vec is not None and query_vec.size > 0:
                cosine = float(np.dot(query_vec, vec).flatten()[0])

            lex_norm = 1.0 / (1.0 + rank_pos)
            score = self._fusion_score(lex_norm, cosine, cand)
            scored.append((score, cand, cosine))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, cand, cosine in scored[:k]:
            results.append(SearchResult(
                id=cand['kb_item_id'],
                source="proj_kb",
                snippet=str(cand['content'])[:200],
                score=score,
                why={
                    "fts_rank": cand.get('fts_rank'),
                    "cosine": cosine,
                    "status": cand['status'],
                    "confidence": cand['confidence'],
                    "scope": {"project_id": cand['project_id']}
                }
            ))

        return results
