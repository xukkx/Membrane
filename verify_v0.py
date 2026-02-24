import json
import hashlib
import os
import platform
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from membrane.logic.canonicalize import generate_kb_item_id
from membrane.logic.ingestion import IngestionEngine
from membrane.logic.retrieval import RetrievalEngine
from membrane.models import Event, KBItem, Scope, SearchFilters
from membrane.storage.jsonl import JSONLStorage
from membrane.storage.ram import RAMStorage
from membrane.storage.sqlite import SQLiteStorage


@dataclass
class TestResult:
    test_name: str
    invariant: str
    phase: str
    passed: bool
    failure_message_if_any: Optional[str] = None


class DeterministicMockEmbeddingModel:
    """Deterministic mock embedding model per test plan."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.constant = np.ones(self.dim, dtype=np.float32)
        self.constant /= (np.linalg.norm(self.constant) + 1e-9)

    def _seed_vec(self, seed: str) -> np.ndarray:
        h = abs(hash(seed)) % (2**32)
        rng = np.random.default_rng(h)
        v = rng.random(self.dim).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v

    def encode(self, texts: List[str]) -> np.ndarray:
        out = []
        for t in texts:
            m = re.search(r"VECTOR_SEED:([^\s]+)", t)
            if m:
                out.append(self._seed_vec(m.group(1)))
            else:
                out.append(self.constant.copy())
        return np.vstack(out)


class Harness:
    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="membrane_v0_verify_")
        self.db_path = os.path.join(self.tmp, "membrane.db")
        self.jsonl_path = os.path.join(self.tmp, "events.jsonl")
        self.sqlite = SQLiteStorage(self.db_path)
        self.jsonl = JSONLStorage(self.jsonl_path)
        self.ram = RAMStorage(self.sqlite)
        self.ram._model = DeterministicMockEmbeddingModel()
        self.ram.rebuild_index()
        self.ingestion = IngestionEngine(self.sqlite, self.jsonl, self.ram)
        self.retrieval = RetrievalEngine(self.sqlite, self.ram)

    def restart(self):
        self.sqlite = SQLiteStorage(self.db_path)
        self.ram = RAMStorage(self.sqlite)
        self.ram._model = DeterministicMockEmbeddingModel()
        self.ram.rebuild_index()
        self.ingestion = IngestionEngine(self.sqlite, self.jsonl, self.ram)
        self.retrieval = RetrievalEngine(self.sqlite, self.ram)

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _mk_event(scope: Scope, text: str, type_: str = "obs", tags=None, entities=None):
    return Event(
        event_id=str(uuid.uuid4()),
        scope=scope,
        type=type_,
        text=text,
        tags=tags or [],
        entities=entities or [],
        ts_epoch_ms=int(time.time() * 1000),
    )




def _tree_fingerprint() -> str:
    """Deterministic fingerprint immune to CRLF/LF and zip metadata drift."""
    tracked = [
        "membrane/storage/sqlite.py",
        "membrane/logic/retrieval.py",
        "membrane/logic/ingestion.py",
        "membrane/storage/ram.py",
        "membrane/server/main.py",
        "verify_v0.py",
        "PATCH_NOTES.md",
    ]
    # Sort to guarantee stable iteration order
    tracked.sort()

    h = hashlib.sha256()
    for rel in tracked:
        if not os.path.exists(rel):
            continue

        # 1. Read as text with explicit UTF-8 encoding
        with open(rel, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        # 2. Normalize Windows CRLF to Unix LF to prevent OS-level hash drift
        normalized_text = raw_text.replace("\r\n", "\n")
        data_bytes = normalized_text.encode("utf-8")

        # 3. Hash path, exact normalized length, and the ENTIRE file content
        h.update(rel.encode("utf-8"))
        h.update(str(len(data_bytes)).encode("utf-8"))
        h.update(data_bytes)

    return h.hexdigest()


def print_env_header():
    import subprocess

    branch = "UNKNOWN"
    head = "UNKNOWN"
    status = "UNKNOWN (non-git artifact or git unavailable)"
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() or "DETACHED"
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--short", "--branch"], text=True).strip()
    except Exception:
        pass

    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    sqlite_ver = sqlite3.sqlite_version
    try:
        cur.execute("SELECT sqlite_compileoption_used('ENABLE_FTS5')")
        fts5_enabled = cur.fetchone()[0]
    except Exception:
        fts5_enabled = None
    fts5_probe_ok = True
    try:
        cur.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(content)")
    except Exception:
        fts5_probe_ok = False
    conn.close()

    print(f"BRANCH={branch}")
    print(f"HEAD={head}")
    print(f"STATUS={status}")
    print(f"PYTHON={sys.version.split()[0]}")
    print(f"OS={platform.platform()}")
    print(f"SQLITE_VERSION={sqlite_ver}")
    print(f"SQLITE_ENABLE_FTS5={fts5_enabled}")
    print(f"SQLITE_FTS5_PROBE={'PASS' if fts5_probe_ok else 'FAIL'}")
    print(f"ARTIFACT_FINGERPRINT={_tree_fingerprint()}")
    print("EMBEDDING_MODE=mock")


# ---- tests ----

def t_i1_agent_temp_isolation(h: Harness):
    s1 = Scope(project_id="p", agent_id="a1", session_id="s")
    s2 = Scope(project_id="p", agent_id="a2", session_id="s")
    h.ingestion.add_event(_mk_event(s1, "secret alpha"))
    h.ingestion.add_event(_mk_event(s2, "secret beta"))
    r = h.retrieval.temp_search(s1, "secret")
    _assert(len(r) == 1 and "alpha" in r[0].snippet, "agent isolation failed")


def t_i1_project_temp_isolation(h: Harness):
    s1 = Scope(project_id="p1", agent_id="a", session_id="s")
    s2 = Scope(project_id="p2", agent_id="a", session_id="s")
    h.ingestion.add_event(_mk_event(s1, "proj1 token"))
    h.ingestion.add_event(_mk_event(s2, "proj2 token"))
    r = h.retrieval.temp_search(s1, "token")
    _assert(len(r) == 1 and "proj1" in r[0].snippet, "project isolation failed")


def t_i1_session_boundary(h: Harness):
    s1 = Scope(project_id="p", agent_id="a", session_id="s1")
    s2 = Scope(project_id="p", agent_id="a", session_id="s2")
    h.ingestion.add_event(_mk_event(s1, "sess one"))
    h.ingestion.add_event(_mk_event(s2, "sess two"))
    r = h.retrieval.temp_search(s1, "sess")
    _assert(len(r) == 1 and "one" in r[0].snippet, "session boundary failed")


def t_i1_thread_task_and(h: Harness):
    s = Scope(project_id="p", agent_id="a", session_id="s", thread_id="t1", task_id="k1")
    h.ingestion.add_event(_mk_event(s, "thread task hit", tags=["x"], entities=["e"]))
    s2 = Scope(project_id="p", agent_id="a", session_id="s", thread_id="t1", task_id="k2")
    h.ingestion.add_event(_mk_event(s2, "thread only"))
    filters = SearchFilters(thread_id="t1", task_id="k1")
    r = h.retrieval.temp_search(Scope(project_id="p", agent_id="a", session_id="s"), "thread", filters=filters)
    _assert(len(r) == 1 and "hit" in r[0].snippet, "thread/task AND semantics failed")


def t_i2_intra_project_kb_visibility(h: Harness):
    item = KBItem(
        kb_item_id="k1", project_id="p", kind="summary", content={"scope": "project", "text": "shared fact"},
        provenance_event_ids=["e1"], confidence=0.6, status="verified", ts_epoch_ms=1
    )
    h.ingestion.commit("p", [item])
    r = h.retrieval.proj_kb_search("p", "shared")
    _assert(len(r) >= 1 and r[0].source == "proj_kb", "intra project kb visibility failed")


def t_i2_inter_project_kb_isolation(h: Harness):
    a = KBItem(kb_item_id="ka", project_id="p1", kind="summary", content={"scope": "project", "text": "isolate me"}, provenance_event_ids=["e1"], confidence=0.5, status="verified", ts_epoch_ms=1)
    b = KBItem(kb_item_id="kb", project_id="p2", kind="summary", content={"scope": "project", "text": "isolate me"}, provenance_event_ids=["e2"], confidence=0.5, status="verified", ts_epoch_ms=1)
    h.ingestion.commit("p1", [a]); h.ingestion.commit("p2", [b])
    r = h.retrieval.proj_kb_search("p1", "isolate")
    _assert(all(x.id != "kb" for x in r), "inter-project kb isolation failed")


def t_i3_no_implicit_merge(h: Harness):
    scope = Scope(project_id="p", agent_id="a", session_id="s")
    h.ingestion.add_event(_mk_event(scope, "temp only marker"))
    item = KBItem(kb_item_id="kid", project_id="p", kind="summary", content={"scope": "project", "text": "kb only marker"}, provenance_event_ids=["e"], confidence=0.7, status="verified", ts_epoch_ms=1)
    h.ingestion.commit("p", [item])
    rt = h.retrieval.temp_search(scope, "marker")
    rk = h.retrieval.proj_kb_search("p", "marker")
    _assert(all(x.source == "temp" for x in rt), "temp search mixed sources")
    _assert(all(x.source == "proj_kb" for x in rk), "kb search mixed sources")


def t_i4_stable_kb_item_id_restart(h: Harness):
    pid, kind, content = "p", "fact", {"x": " y z "}
    a = generate_kb_item_id(pid, kind, content)
    h.restart()
    b = generate_kb_item_id(pid, kind, content)
    _assert(a == b, "stable kb_item_id failed")


def t_i4_canonical_invariant(h: Harness):
    id1 = generate_kb_item_id("p", "fact", {"a": "  hello   world ", "b": 1})
    id2 = generate_kb_item_id("p", "fact", {"b": 1, "a": "hello world"})
    _assert(id1 == id2, "canonical hash invariant failed")


def t_i4_deterministic_promotion(h: Harness):
    scope = Scope(project_id="p", agent_id="a", session_id="s")
    e1 = _mk_event(scope, "Summary: Deterministic text")
    h.ingestion.add_event(e1)
    p1 = h.ingestion.promote("p", [e1.event_id])
    p2 = h.ingestion.promote("p", [e1.event_id])
    _assert([x.model_dump() for x in p1] == [x.model_dump() for x in p2], "promotion not deterministic")


def t_i4_deterministic_entity_extraction(h: Harness):
    scope = Scope(project_id="p", agent_id="a", session_id="s")
    e = _mk_event(scope, "Use API_GATEWAY and CamelCaseToken and 0xABCD")
    h.ingestion.add_event(e)
    p1 = [x.model_dump() for x in h.ingestion.promote("p", [e.event_id]) if x.kind == "entity"]
    p2 = [x.model_dump() for x in h.ingestion.promote("p", [e.event_id]) if x.kind == "entity"]
    _assert(p1 == p2, "entity extraction not deterministic")


def t_i4_adversarial_delimiter_injection(h: Harness):
    scope = Scope(project_id="p", agent_id="a", session_id="s")
    e = _mk_event(scope, "User text naturally contains ... delimiter in body")
    h.ingestion.add_event(e)
    cands = h.ingestion.promote("p", [e.event_id])
    summary = next(c for c in cands if c.kind == "summary")
    _assert("delimiter in body" in summary.content["text"], "delimiter injection fragmented summary")


def t_i5_idempotent_commit_no_duplicate_row(h: Harness):
    item = KBItem(kb_item_id="iid", project_id="p", kind="fact", content={"text": "x"}, provenance_event_ids=["e1"], confidence=0.5, status="proposed", ts_epoch_ms=1)
    h.ingestion.commit("p", [item]); h.ingestion.commit("p", [item])
    with h.sqlite.get_connection() as conn:
        cnt = conn.execute("SELECT COUNT(*) FROM kb_items WHERE kb_item_id='iid'").fetchone()[0]
    _assert(cnt == 1, "idempotent commit produced duplicate row")


def t_i5_provenance_sorted_union(h: Harness):
    base = KBItem(kb_item_id="prov", project_id="p", kind="fact", content={"text": "x"}, provenance_event_ids=["e2"], confidence=0.5, status="verified", ts_epoch_ms=1)
    h.ingestion.commit("p", [base])
    inc = KBItem(kb_item_id="prov", project_id="p", kind="fact", content={"text": "x"}, provenance_event_ids=["e1","e2"], confidence=0.45, status="verified", ts_epoch_ms=2)
    h.ingestion.commit("p", [inc])
    got = h.sqlite.get_kb_item("prov")
    _assert(got["provenance_event_ids"] == ["e1", "e2"], f"provenance union wrong: {got['provenance_event_ids']}")


def t_i5_conflicts_supersedes_sorted_union(h: Harness):
    i1 = KBItem(kb_item_id="merge1", project_id="p", kind="fact", content={"text": "x"}, provenance_event_ids=["e"], confidence=0.7, status="verified", conflicts_with=["a"], supersedes=["s2"], ts_epoch_ms=1)
    i2 = KBItem(kb_item_id="merge1", project_id="p", kind="fact", content={"text": "x"}, provenance_event_ids=["e"], confidence=0.75, status="verified", conflicts_with=["b","a"], supersedes=["s1"], ts_epoch_ms=2)
    h.ingestion.commit("p", [i1]); h.ingestion.commit("p", [i2])
    got = h.sqlite.get_kb_item("merge1")
    _assert(got["conflicts_with"] == ["a", "b"], "conflicts union wrong")
    _assert(got["supersedes"] == ["s1", "s2"], "supersedes union wrong")


def t_i6_verified_supersedes_proposed(h: Harness):
    p = KBItem(kb_item_id="c1", project_id="p", kind="fact", content={"text":"x"}, provenance_event_ids=["e"], confidence=0.4, status="proposed", ts_epoch_ms=1)
    v = p.model_copy(); v.status = "verified"; v.confidence = 0.5
    h.ingestion.commit("p", [p]); h.ingestion.commit("p", [v])
    got = h.sqlite.get_kb_item("c1")
    _assert(got["status"] == "verified", "verified did not supersede proposed")


def t_i6_verified_vs_verified_low_delta_conflict(h: Harness):
    a = KBItem(kb_item_id="c2", project_id="p", kind="fact", content={"text":"x"}, provenance_event_ids=["e1"], confidence=0.6, status="verified", ts_epoch_ms=1)
    b = a.model_copy(); b.confidence = 0.65
    h.ingestion.commit("p", [a]); res = h.ingestion.commit("p", [b])
    got = h.sqlite.get_kb_item("c2")
    _assert(res["conflicts"] == 1 and abs(got["confidence"] - 0.6) < 1e-9, "low-delta conflict logic failed")


def t_i6_verified_vs_verified_high_delta_replace(h: Harness):
    a = KBItem(kb_item_id="c3", project_id="p", kind="fact", content={"text":"x"}, provenance_event_ids=["e1"], confidence=0.6, status="verified", ts_epoch_ms=1)
    b = a.model_copy(); b.confidence = 0.71
    h.ingestion.commit("p", [a]); h.ingestion.commit("p", [b])
    got = h.sqlite.get_kb_item("c3")
    _assert(abs(got["confidence"] - 0.71) < 1e-9, "high-delta replace failed")


def t_i6_conflict_record_bidirectional_diff_ids(h: Harness):
    a = KBItem(kb_item_id="confA", project_id="p", kind="fact", content={"text":"A"}, provenance_event_ids=["e1"], confidence=0.7, status="verified", conflicts_with=["confB"], ts_epoch_ms=1)
    b = KBItem(kb_item_id="confB", project_id="p", kind="fact", content={"text":"B"}, provenance_event_ids=["e2"], confidence=0.7, status="verified", conflicts_with=["confA"], ts_epoch_ms=1)
    h.ingestion.commit("p", [a, b])
    ga = h.sqlite.get_kb_item("confA"); gb = h.sqlite.get_kb_item("confB")
    _assert("confB" in ga["conflicts_with"] and "confA" in gb["conflicts_with"], "bi-directional conflict record failed")


def t_i7_lexical_rank_dominance(h: Harness):
    iA = KBItem(kb_item_id="lexA", project_id="p", kind="summary", content={"scope": "project", "text": "RARETERM_ZXQ apple"}, provenance_event_ids=["e"], confidence=0.5, status="verified", ts_epoch_ms=1)
    iB = KBItem(kb_item_id="lexB", project_id="p", kind="summary", content={"scope": "project", "text": "apple"}, provenance_event_ids=["e"], confidence=0.5, status="verified", ts_epoch_ms=1)
    h.ingestion.commit("p", [iA, iB])
    r = h.retrieval.proj_kb_search("p", "RARETERM_ZXQ", k=2)
    _assert(len(r) >= 1 and r[0].id == "lexA", "lexical dominance failed")


def t_i7_vector_rank_dominance(h: Harness):
    iA = KBItem(kb_item_id="vecA", project_id="p", kind="summary", content={"scope": "project", "text": "apple VECTOR_SEED:foo"}, provenance_event_ids=["e"], confidence=0.5, status="verified", ts_epoch_ms=1)
    iB = KBItem(kb_item_id="vecB", project_id="p", kind="summary", content={"scope": "project", "text": "apple VECTOR_SEED:bar"}, provenance_event_ids=["e"], confidence=0.5, status="verified", ts_epoch_ms=1)
    h.ingestion.commit("p", [iA, iB])
    r = h.retrieval.proj_kb_search("p", "apple", k=2)
    q = h.ram.encode(["apple"])[0]
    a = h.ram.get_kb_vector("vecA"); b = h.ram.get_kb_vector("vecB")
    expected = "vecA" if float(np.dot(q, a)) >= float(np.dot(q, b)) else "vecB"
    _assert(r[0].id == expected, f"vector dominance failed: expected {expected}, got {r[0].id}")


def t_i7_fusion_explainability(h: Harness):
    i = KBItem(kb_item_id="why1", project_id="p", kind="summary", content={"scope": "project", "text":"explain apple"}, provenance_event_ids=["e"], confidence=0.9, status="verified", ts_epoch_ms=1)
    h.ingestion.commit("p", [i])
    r = h.retrieval.proj_kb_search("p", "apple", k=1)[0]
    why = r.why
    for k in ["fts_rank", "cosine", "status", "confidence", "scope"]:
        _assert(k in why, f"missing why.{k}")


def t_i7_negative_lexical_score_handling(h: Harness):
    iA = KBItem(kb_item_id="negA", project_id="p", kind="summary", content={"scope": "project", "text":"apple"}, provenance_event_ids=["e"], confidence=0.5, status="verified", ts_epoch_ms=1)
    iB = KBItem(kb_item_id="negB", project_id="p", kind="summary", content={"scope": "project", "text":"apple"}, provenance_event_ids=["e"], confidence=0.5, status="verified", ts_epoch_ms=1)
    h.ingestion.commit("p", [iA, iB])

    original = h.sqlite.search_kb_items
    def fake(query, project_id, limit=200):
        return [
            {**h.sqlite.get_kb_item("negA"), "fts_rank": -10.0},
            {**h.sqlite.get_kb_item("negB"), "fts_rank": 0.0},
        ]
    h.sqlite.search_kb_items = fake
    try:
        r = h.retrieval.proj_kb_search("p", "apple", k=2)
    finally:
        h.sqlite.search_kb_items = original
    _assert(r[0].id == "negA", "negative lexical score handling failed")


def t_i8_schema_external_content_and_triggers(h: Harness):
    with h.sqlite.get_connection() as conn:
        rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table','trigger')").fetchall()
    d = {r[0]: (r[1] or "") for r in rows}
    te_sql = d.get("temp_events_fts", "")
    kb_sql = d.get("kb_items_fts", "")
    _assert("content='temp_events'" in te_sql and "content_rowid='rowid'" in te_sql, "temp_events_fts not external content")
    _assert("content='kb_items'" in kb_sql and "content_rowid='rowid'" in kb_sql, "kb_items_fts not external content")
    required = [
        "trg_temp_events_ai", "trg_temp_events_au", "trg_temp_events_ad",
        "trg_kb_items_ai", "trg_kb_items_au", "trg_kb_items_ad",
    ]
    for t in required:
        _assert(t in d, f"missing trigger {t}")


def t_i8_fts_sync_insert_update_delete(h: Harness):
    scope = Scope(project_id="p", agent_id="a", session_id="s")
    ev = _mk_event(scope, "fts initial value")
    h.ingestion.add_event(ev)
    _assert(len(h.retrieval.temp_search(scope, "initial")) >= 1, "fts insert sync failed")
    with h.sqlite.get_connection() as conn:
        conn.execute("UPDATE temp_events SET search_text=? WHERE event_id=?", ("fts updated value", ev.event_id))
        conn.commit()
    _assert(len(h.retrieval.temp_search(scope, "updated")) >= 1, "fts update sync failed")
    with h.sqlite.get_connection() as conn:
        conn.execute("DELETE FROM temp_events WHERE event_id=?", (ev.event_id,))
        conn.commit()
    _assert(len(h.retrieval.temp_search(scope, "updated")) == 0, "fts delete sync failed")


def t_i8_no_manual_fts_ops(h: Harness):
    root = os.getcwd()
    forbidden = [
        "DELETE FROM kb_items_fts",
        "INSERT INTO kb_items_fts",
        "DELETE FROM temp_events_fts",
        "INSERT INTO temp_events_fts",
    ]
    bad = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            if fn == 'verify_v0.py':
                continue
            path = os.path.join(dirpath, fn)
            txt = open(path, 'r', encoding='utf-8').read()
            for p in forbidden:
                if p in txt:
                    bad.append((path, p))
    _assert(not bad, f"forbidden manual FTS ops found: {bad}")


def t_i9_ram_rebuild_and_consistency(h: Harness):
    scope = Scope(project_id="p", agent_id="a", session_id="s")
    h.ingestion.add_event(_mk_event(scope, "restart parity secret"))
    pre = [x.id for x in h.retrieval.temp_search(scope, "secret")]
    h.restart()
    post = [x.id for x in h.retrieval.temp_search(scope, "secret")]
    _assert(pre == post and len(post) > 0, "ram rebuild/search consistency failed")


def t_i9_post_sqlite_crash_recovery(h: Harness):
    scope = Scope(project_id="p", agent_id="a", session_id="s")
    orig = h.ram.add_temp_event
    def boom(*args, **kwargs):
        raise RuntimeError("simulated ram failure")
    h.ram.add_temp_event = boom
    ev = _mk_event(scope, "sqlite survives ram crash")
    failed = False
    try:
        h.ingestion.add_event(ev)
    except RuntimeError:
        failed = True
    finally:
        h.ram.add_temp_event = orig
    _assert(failed, "failure injection did not trigger")
    h.restart()
    r = h.retrieval.temp_search(scope, "survives")
    _assert(any(x.id == ev.event_id for x in r), "post-crash recovery failed")


def t_i9_jsonl_lag_tolerance(h: Harness):
    scope = Scope(project_id="p", agent_id="a", session_id="s")
    orig = h.jsonl.append
    def boom(*args, **kwargs):
        raise RuntimeError("jsonl down")
    h.jsonl.append = boom
    ev = _mk_event(scope, "jsonl fail but sqlite persists")
    failed = False
    try:
        h.ingestion.add_event(ev)
    except RuntimeError:
        failed = True
    finally:
        h.jsonl.append = orig
    _assert(failed, "jsonl failure injection did not trigger")
    r = h.retrieval.temp_search(scope, "sqlite persists")
    _assert(any(x.id == ev.event_id for x in r), "jsonl lag tolerance failed")


def t_i10_fts5_missing_probe(h: Harness):
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    try:
        cur.execute("CREATE VIRTUAL TABLE fts_check_x USING fts5(content)")
    except sqlite3.OperationalError as e:
        raise AssertionError(f"fts5 missing: {e}")
    finally:
        conn.close()


def t_i10_invalid_policy_failfast(h: Harness):
    from membrane.storage import policy as policy_mod
    p = policy_mod.POLICY_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    backup = None
    if os.path.exists(p):
        backup = p + ".bak_v0"
        shutil.copyfile(p, backup)
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write("invalid = [\n")
        try:
            _ = RetrievalEngine(h.sqlite, h.ram)
            raise AssertionError("expected invalid policy fail-fast")
        except RuntimeError:
            pass
    finally:
        if backup and os.path.exists(backup):
            shutil.move(backup, p)
        elif os.path.exists(p):
            os.remove(p)


def t_i10_unknown_filter_key(h: Harness):
    from pydantic import ValidationError
    try:
        SearchFilters(unknown="x")
    except ValidationError:
        return
    raise AssertionError("unknown filter key was accepted")


def build_tests():
    return [
        ("agent_temp_isolation", "I1 Scope isolation (Temp)", "BASELINE", t_i1_agent_temp_isolation),
        ("project_temp_isolation", "I1 Scope isolation (Temp)", "BASELINE", t_i1_project_temp_isolation),
        ("session_boundary", "I1 Scope isolation (Temp)", "BASELINE", t_i1_session_boundary),
        ("thread_task_and_semantics", "I1 Scope isolation (Temp)", "BASELINE", t_i1_thread_task_and),
        ("intra_project_kb_visibility", "I2 Project KB sharing", "BASELINE", t_i2_intra_project_kb_visibility),
        ("inter_project_kb_isolation", "I2 Project KB sharing", "BASELINE", t_i2_inter_project_kb_isolation),
        ("no_implicit_merge_sources", "I3 No implicit merge", "BASELINE", t_i3_no_implicit_merge),
        ("stable_kb_item_id_restart", "I4 Determinism", "BASELINE", t_i4_stable_kb_item_id_restart),
        ("canonical_hashing_invariant", "I4 Determinism", "BASELINE", t_i4_canonical_invariant),
        ("deterministic_promotion", "I4 Determinism", "BASELINE", t_i4_deterministic_promotion),
        ("deterministic_entity_extraction", "I4 Determinism", "BASELINE", t_i4_deterministic_entity_extraction),
        ("adversarial_delimiter_injection", "I4 Determinism", "POST_PATCH", t_i4_adversarial_delimiter_injection),
        ("idempotent_commit_no_duplicate_row", "I5 Idempotency & merge semantics", "BASELINE", t_i5_idempotent_commit_no_duplicate_row),
        ("provenance_sorted_union", "I5 Idempotency & merge semantics", "BASELINE", t_i5_provenance_sorted_union),
        ("conflicts_supersedes_sorted_union", "I5 Idempotency & merge semantics", "BASELINE", t_i5_conflicts_supersedes_sorted_union),
        ("verified_supersedes_proposed", "I6 Conflict resolution", "BASELINE", t_i6_verified_supersedes_proposed),
        ("verified_vs_verified_delta_005", "I6 Conflict resolution", "BASELINE", t_i6_verified_vs_verified_low_delta_conflict),
        ("verified_vs_verified_delta_011", "I6 Conflict resolution", "BASELINE", t_i6_verified_vs_verified_high_delta_replace),
        ("conflict_record_bidirectional_diff_ids", "I6 Conflict resolution", "BASELINE", t_i6_conflict_record_bidirectional_diff_ids),
        ("lexical_rank_dominance", "I7 Hybrid retrieval correctness", "BASELINE", t_i7_lexical_rank_dominance),
        ("vector_rank_dominance", "I7 Hybrid retrieval correctness", "BASELINE", t_i7_vector_rank_dominance),
        ("fusion_score_explainability", "I7 Hybrid retrieval correctness", "BASELINE", t_i7_fusion_explainability),
        ("negative_lexical_score_handling", "I7 Hybrid retrieval correctness", "POST_PATCH", t_i7_negative_lexical_score_handling),
        ("fts_external_content_schema", "I8 FTS external content correctness", "POST_PATCH", t_i8_schema_external_content_and_triggers),
        ("fts_sync_insert_update_delete", "I8 FTS external content correctness", "POST_PATCH", t_i8_fts_sync_insert_update_delete),
        ("no_manual_fts_ops", "I8 FTS external content correctness", "POST_PATCH", t_i8_no_manual_fts_ops),
        ("ram_rebuild_consistency", "I9 Persistence / recovery", "BASELINE", t_i9_ram_rebuild_and_consistency),
        ("post_sqlite_crash_recovery", "I9 Persistence / recovery", "POST_PATCH", t_i9_post_sqlite_crash_recovery),
        ("jsonl_lag_tolerance", "I9 Persistence / recovery", "BASELINE", t_i9_jsonl_lag_tolerance),
        ("fts5_missing_probe", "I10 Fail-fast checks", "BASELINE", t_i10_fts5_missing_probe),
        ("invalid_policy_failfast", "I10 Fail-fast checks", "BASELINE", t_i10_invalid_policy_failfast),
        ("unknown_filter_key", "I10 Fail-fast checks", "BASELINE", t_i10_unknown_filter_key),
    ]


def run_suite(run_phase: str) -> int:
    print_env_header()
    results: List[TestResult] = []
    tests = build_tests()

    for name, invariant, phase, fn in tests:
        if run_phase == "baseline" and phase != "BASELINE":
            continue

        h = Harness()
        try:
            fn(h)
            print(f"PASS {phase} {name}")
            results.append(TestResult(name, invariant, phase, True, None))
        except Exception as e:
            print(f"FAIL {phase} {name}: {e}")
            results.append(TestResult(name, invariant, phase, False, str(e)))
        finally:
            h.cleanup()

    summary = [
        {
            "test_name": r.test_name,
            "invariant": r.invariant,
            "phase": r.phase,
            "pass": r.passed,
            "failure_message_if_any": r.failure_message_if_any,
        }
        for r in results
    ]
    print("JSON_SUMMARY=" + json.dumps(summary, indent=2))

    failed = any(not r.passed for r in results)
    if run_phase == "full" and failed:
        return 1
    return 0


if __name__ == "__main__":
    phase = "full"
    if len(sys.argv) > 1:
        phase = sys.argv[1].strip().lower()
    if phase not in {"baseline", "full"}:
        print("Usage: python3 verify_v0.py [baseline|full]")
        sys.exit(2)
    sys.exit(run_suite(phase))
