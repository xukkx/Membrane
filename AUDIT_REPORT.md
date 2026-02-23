# Membrane v0 Audit Report

## Scope
This report covers structural, storage, behavioral, and adversarial invariants requested for Membrane v0.

## PASS/FAIL Matrix

| Invariant | Status | Evidence |
|---|---|---|
| Required top-level paths (`membrane/storage`, `membrane/models`, `membrane/logic`, `membrane/server`, `verify_server.py`) | **FAIL (partial)** | `membrane/models` is a file (`membrane/models.py`), not a directory. |
| `Scope` requires `project_id`, `agent_id`, `session_id` | **PASS** | Required fields are declared in model. |
| `SearchFilters` rejects unknown keys | **PASS** | `model_config = {"extra": "forbid"}` and runtime validation fails on unknown keys. |
| `KBItem` includes required fields | **PASS** | Model contains all required fields. |
| `kb_item_id = sha256(canonical_json({project_id, kind, content}))` | **PASS** | Implemented in canonicalization helpers. |
| Canonical JSON deterministic (sorted keys, no whitespace) | **PASS** | `json.dumps(... sort_keys=True, separators=(',', ':'))`. |
| Identity payload excludes provenance/confidence/timestamps | **PASS** | ID generation uses only `project_id`, `kind`, normalized `content`. |
| Deterministic embedding templates exist | **PASS** | Template builder is deterministic by kind and normalized content. |
| SQLite tables exist (`temp_events`, `kb_items`, `temp_events_fts`, `kb_items_fts`) | **PASS** | Created in `_init_db`. |
| `search_text` column exists | **PASS** | Present in both `temp_events` and `kb_items`. |
| `UNIQUE(kb_item_id)` enforced | **PASS** | `kb_item_id TEXT PRIMARY KEY`. |
| FTS5 fail-fast behavior | **PASS** | `_ensure_fts5()` raises `RuntimeError` if unavailable. |
| RAM vectors rebuild from SQLite `search_text` | **PASS** | `rebuild_index()` uses `fetch_all_temp_events()` / `fetch_all_kb_items()` and `search_text`. |
| Positive flow (add→promote→commit→search) | **PASS** | Verified in expanded local harness. |
| Isolation (agent/project) | **PASS** | Scoped FTS query filters by project+agent+session. |
| Idempotency (same item twice, merged provenance) | **PASS** | Single row preserved and provenance union merged. |
| Conflict logic: proposed→verified supersede | **PASS** | Existing proposed updated by verified commit path. |
| Conflict logic: verified vs verified threshold behavior | **PASS (same-ID path)** | `< +0.1` treated as conflict/no-op metadata update; `>= +0.1` accepted and supersede counter increments. |
| Persistence (restart + reindex keeps search parity) | **PASS** | Equivalent result IDs before/after rebuild in harness. |
| Retrieval bounds (FTS<=200, final<=20) | **PASS** | Enforced via policy caps in retrieval/storage calls. |
| `temp_search` and `proj_kb_search` result domains never merged | **PASS** | Separate APIs and fixed source tags (`temp` vs `proj_kb`). |
| Malformed `SearchFilters` unknown keys | **PASS** | Validation rejects unknown keys. |
| Scope bypass attempt on `temp_search` | **PASS** | Results empty outside exact scope triplet. |
| Canonicalization ordering attack | **PASS** | Different dict key order yields same ID. |
| Same content + different provenance => same `kb_item_id` | **PASS** | ID insensitive to provenance metadata. |
| Entity extraction cap (>20) | **PASS** | Promote caps entities at 20. |

## Determinism Review
- Deterministic ID generation and deterministic embedding templates are implemented as required.
- Content normalization strips/collapses string whitespace before ID and embedding text generation.
- No non-deterministic fields are used in identity payload.

## Scope Leakage Risk Review
- `temp_search` scope constraints are enforced in SQL (`project_id`, `agent_id`, `session_id`).
- No cross-source merge path observed between temp and KB search APIs.
- Residual risk: if future code exposes raw FTS methods without scope checks, leakage risk could reappear.

## Idempotency Review
- Same `kb_item_id` upserts remain single-row due to PK.
- Provenance merges as set-union (deduplicated).
- Observed subtlety: conflict accounting uses counters, but true “keep both” for verified-vs-verified conflicts cannot occur for identical-content same-ID records by design.

## Retrieval Stability Review
- Retrieval caps are policy-driven and applied (`fts_candidates`, `max_k`).
- Results remain stable across RAM rebuilds in tested harness.
- `verify_server.py` is not reproducible in this environment without model artifact availability (Hugging Face download blocked), but logic-level tests were executed with deterministic local embedding monkeypatch.

## Hardening Targets (Minimal)
1. **Spec alignment:** add `membrane/models/` package wrapper (or update spec/docs) to resolve structure mismatch.
2. **Explicit DB uniqueness declaration:** current PK is sufficient, but adding explicit `UNIQUE(kb_item_id)` could improve schema readability.
3. **Verified-vs-verified semantics:** document same-ID behavior (metadata conflict/no-op) vs different-ID semantic conflicts to avoid ambiguity.
4. **Offline testability:** provide deterministic local embedding fallback for CI environments without model network access.

## Minimal Patches Suggested
- No runtime invariant break requiring immediate architecture refactor was found.
- Suggested patches are documentation/schema clarity and testability improvements only.
