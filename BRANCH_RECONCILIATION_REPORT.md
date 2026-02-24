# BRANCH_RECONCILIATION_REPORT

## A) Source-of-Truth & Version Identity

### A1. Exact codebase under test
- Repository type: **git repo present** (`.git` exists).
- Current branch: `work`.
- Current HEAD: `c3179a7cf0b1ef4322c27141f4ff5e1ccb99eb5d`.
- `git status` summary: clean (`## work`).

Evidence commands run:
- `git branch --show-current`
- `git rev-parse HEAD`
- `git status --short --branch`

### A2. Local artifact inventory (zips/folders)
Searched for alternate local artifacts (zip/folder copies) with:
- `find / -maxdepth 4 -iname 'Membrane-*.zip'`
- `find /workspace -maxdepth 3 ... (init|submit|review|branch)`

Result:
- **No `Membrane-*.zip` artifacts found on disk in this environment.**
- No extracted alternate Membrane copies were found under `/workspace`.
- Therefore: only one physical local artifact exists for audit: the git working tree at `/workspace/Membrane`.

### A3. Mapping “Codex-submit” vs “v0-init”
Because no zip artifacts exist locally, mapping is done against reachable commits in this repo:

| ArtifactName | Identity | sha256/commit | has_git | server entrypoint | tool names | fts query style |
|---|---|---|---|---|---|---|
| Current working tree (“Codex-submit” equivalent) | HEAD on `work` | `c3179a7cf0b1ef4322c27141f4ff5e1ccb99eb5d` | Yes | `membrane/server/main.py` | dotted + underscore alias tools | `bm25(...) AS fts_rank` |
| Baseline “v0-init” equivalent | earliest v0 implementation commit in this repo | `76f9fe6911012ccd20606100b400a95a0f1aa112` | Yes | `membrane/server/main.py` | underscore-only tools | `SELECT rank ... ORDER BY rank` |

> Note: exact zip labels (`Membrane-codex-submit-...zip`, etc.) are not present locally, so commit IDs are the only verifiable identity source in this environment.

---

## B) Diff Map (Where They Diverge)

Compared: `76f9fe6..c3179a7`.

### B4. Diff heatmap for key files

| File | Diff? | High-level delta |
|---|---|---|
| `membrane/storage/sqlite.py` | Yes | Switched FTS ranking from `rank` to `bm25(... ) AS fts_rank`; removed dead placeholder method. |
| `membrane/logic/retrieval.py` | Yes | Fusion now includes lexical term; added thread/task filters; passes `fts_rank` into fusion for temp+KB. |
| `membrane/logic/ingestion.py` | Yes | Added deterministic boilerplate-prefix stripping in summary promotion. |
| `membrane/storage/ram.py` | Yes | Removed fixed 384-dim assumption; infer embedding dimension dynamically. |
| `membrane/logic/canonicalize.py` | No | No divergence detected between compared artifacts. |
| `membrane/server/main.py` | Yes | Added dotted tool names + underscore aliases; strict JSON returns for add_event/maintenance; promote accepts `candidates_schema`. |
| `verify_server.py` | Yes | Persistence test now reuses chosen mock/real embedding mode from setup. |

### B4a. Important unified diff snippets + impact

#### 1) `membrane/storage/sqlite.py`
```diff
-                SELECT rank, rowid, * FROM temp_events_fts
+                SELECT bm25(temp_events_fts) AS fts_rank, rowid, * FROM temp_events_fts
...
-                ORDER BY rank
+                ORDER BY fts_rank ASC
...
-                    d['fts_rank'] = row['rank']
+                    d['fts_rank'] = row['fts_rank']
```
Impact: **bug/spec drift fix** — old `rank`-based query style is invalid/non-portable; bm25 gives deterministic lexical rank semantics.

#### 2) `membrane/logic/retrieval.py`
```diff
-        base = w.get('w_vec', 0.7) * vec_score
+        lex = 1.0 / (1.0 + max(0.0, lex_score))
+        base = (w.get('w_lex', 0.3) * lex) + (w.get('w_vec', 0.7) * vec_score)
...
-            score = self._fusion_score(0.0, cosine, cand)
+            score = self._fusion_score(float(cand.get('fts_rank', 0.0)), cosine, cand)
```
Impact: **ranking correctness** — converts candidate generation + vector-only rerank into actual hybrid rerank.

#### 3) `membrane/logic/ingestion.py`
```diff
-        cleaned_text = " ".join(full_text.split())
+        boilerplate_prefixes = ["note:", "summary:", "observation:", "context:", "assistant:", "user:", "system:"]
+        stripped_parts = []
+        for raw_part in full_text.split(" ... "):
+            part = raw_part.lstrip()
+            lowered = part.lower()
+            for prefix in boilerplate_prefixes:
+                if lowered.startswith(prefix):
+                    part = part[len(prefix):].lstrip()
+                    break
+            stripped_parts.append(part)
+        cleaned_text = " ".join(" ... ".join(stripped_parts).split())
```
Impact: **spec-quality alignment** — reduces summary noise and aligns with “strip boilerplate prefixes” intent.

#### 4) `membrane/storage/ram.py`
```diff
-        self.temp_vectors = np.empty((0, 384))
+        self._vector_dim: Optional[int] = None
+        self.temp_vectors = self._empty_vectors()
...
-        if not texts:
-            return np.empty((0, 384))
+        if not texts:
+            return self._empty_vectors()
+        embeddings = np.asarray(self.model.encode(texts), dtype=np.float32)
+        if self._vector_dim is None:
+            self._vector_dim = embeddings.shape[1]
```
Impact: **robustness/perf correctness** — avoids model-dimension mismatch failures.

#### 5) `membrane/server/main.py`
```diff
-@mcp.tool(name="temp_add_event")
-async def add_event(...) -> str:
+@mcp.tool(name="temp.add_event")
+async def add_event(...) -> Dict[str, Any]:
...
-    return "Event added."
+    return {"ok": True, "event_id": event.event_id}
...
+@mcp.tool(name="temp_add_event")
+async def add_event_alias(...):
+    return await add_event(...)
```
Impact: **tool contract stability** — adds spec-name compatibility while preserving legacy aliases.

#### 6) `verify_server.py`
```diff
-    return ingestion, retrieval, ram, sqlite
+    return ingestion, retrieval, ram, sqlite, use_mock
...
-def test_persistence():
+def test_persistence(use_mock: bool = False):
...
-    use_mock = os.environ.get("MEMBRANE_TEST_MOCK_EMBEDDING", "0") == "1"
+    # Reuse whichever embedding mode setup() selected
```
Impact: **test reliability** — removes false failures in offline/proxy-restricted environments.

### B5. Frozen v0 spec compliance matrix

| Requirement | `76f9fe6` (v0-init equivalent) | `c3179a7` (current/Codex-submit equivalent) | Notes |
|---|---|---|---|
| FTS query uses `bm25()` (not `rank`) | FAIL | PASS | init uses `SELECT rank`; current uses `bm25(... ) AS fts_rank`. |
| `SearchResult.why` includes `fts_rank` and `cosine` | PASS | PASS | Both include these fields in `why`. |
| `temp_search` and `proj_kb_search` separate | PASS | PASS | Distinct retrieval methods in both artifacts. |
| `kb_item_id` hashing = sha256(canonical_json({project_id,kind,content})) | PASS | PASS | canonicalize unchanged and deterministic. |
| ALL_CAPS regex uses `(?:_[A-Z0-9]{2,})*` | PASS | PASS | Present in ingestion regex. |
| Tool names: dotted + underscore alias | FAIL | PASS | init has underscore only; current has both. |
| `k` capped to `min(k,20)` via policy max | PASS | PASS | Implemented in both temp and KB searches. |

---

## C) Root Cause of Inconsistency

### C6. Branch mismatch evidence
- Branch list (`git branch -vv`) shows only one branch: `work`, at `c3179a7`.
- Recent history (`git log --oneline --decorate -n 20`) includes multiple incremental “initial implementation” commits and later fix commits.
- No remotes configured (`git remote -v` returned empty).
- No uncommitted local edits at audit start (`git status` clean).

Interpretation:
- Inconsistency is **not from active local branch divergence** right now.
- Main confusion source is likely **external artifacts (online repo / downloaded zips) not present in this environment**, plus multiple successive commits representing evolving states.

### C7. Multiple entrypoints / duplicated servers
- Found server entrypoint file: `membrane/server/main.py`.
- No `membrane/server/__main__.py` and no root-level `server.py` detected.
- `verify_server.py` does **not** call MCP server entrypoint; it directly instantiates storage/logic classes (`SQLiteStorage`, `RAMStorage`, `IngestionEngine`, `RetrievalEngine`).

Implication:
- Verification results primarily validate logic/storage behavior, not networked MCP registration plumbing.

---

## D) Decision — What is “Latest”?

### D8. Canonical artifact selection
**Canonical latest for review: `c3179a7` (HEAD on `work`).**

Justification:
1. **Spec compliance:** passes critical frozen-v0 items that v0-init fails (bm25 ranking, dotted+alias tool surface).  
2. **Test pass likelihood:** verifier passes in both forced-mock and fallback modes in this environment.  
3. **Tool contract stability:** supports both dotted spec names and underscore legacy aliases.  
4. **Minimal bug risk:** removes known rank-query drift and embedding-dimension fragility.

### D9. Must-have merges from non-canonical artifacts (max 5)
Given no alternate local artifact beyond historical commits, must-have items are already present in canonical HEAD. Keep these anchored capabilities:
1. `membrane/storage/sqlite.py::search_temp_events/search_kb_items` — keep bm25 rank + `fts_rank` propagation.
2. `membrane/logic/retrieval.py::_fusion_score/temp_search/proj_kb_search` — keep lexical+vector hybrid fusion and `fts_rank` handoff.
3. `membrane/storage/ram.py::encode/_empty_vectors/rebuild_index` — keep dynamic embedding dimension inference.
4. `membrane/server/main.py` tool registrations — keep dotted names + underscore aliases + strict JSON for `temp.add_event` and `maintenance.*`.
5. `verify_server.py::setup/test_persistence` — keep consistent mock fallback mode across persistence test.

### D10. Changes to explicitly discard
Do **not** reintroduce:
- `SELECT rank ... ORDER BY rank` FTS query style.
- Vector-only scoring that ignores lexical rank in hybrid fusion.
- Fixed `384`-dim vector allocation assumptions.
- Underscore-only MCP tool naming without dotted spec names.
- Plain-string responses for `temp.add_event` / `maintenance.*` where strict JSON object is expected.

---

## E) Minimal Reconciliation Plan (No code changes yet)

1. Treat commit `c3179a7` on branch `work` as review baseline.
2. Reject any external zip/repo snapshot that lacks either:
   - bm25-based FTS queries, or
   - dotted+alias tool contracts.
3. If external reviewer uses zip artifacts, require they provide:
   - artifact sha256,
   - top-level tree listing,
   - and commit or patch provenance before accepting feedback.
4. Only after provenance alignment, open targeted code-change PRs (if new defects are proven).

---

## Next action (exact commands, no patches applied now)

### Commands to run on canonical artifact
```bash
cd /workspace/Membrane
git checkout work
git rev-parse HEAD
python3 -m py_compile membrane/storage/sqlite.py membrane/logic/retrieval.py membrane/logic/ingestion.py membrane/storage/ram.py membrane/server/main.py verify_server.py
MEMBRANE_TEST_MOCK_EMBEDDING=1 python3 verify_server.py
python3 verify_server.py
```

### Exact patches needed (if and only if a non-canonical artifact is selected)
- Patch set A: `membrane/storage/sqlite.py` — replace rank-based queries with `bm25(...) AS fts_rank`, order ascending, propagate `fts_rank`.
- Patch set B: `membrane/server/main.py` — add dotted tool names while retaining underscore aliases; enforce JSON-object responses for add_event/maintenance.
- Patch set C: `membrane/storage/ram.py` — infer vector dimension from embeddings; remove fixed `(0, 384)` allocations.
- Patch set D: `membrane/logic/retrieval.py` — include lexical rank in fusion and pass candidate `fts_rank` in scoring calls.
- Patch set E: `verify_server.py` — persist chosen embedding mode into persistence test.

