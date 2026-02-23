import os
import shutil
import time
import uuid
import logging
import numpy as np
from membrane.models import Scope, Event, KBItem, SearchFilters
from membrane.storage.sqlite import SQLiteStorage
from membrane.storage.jsonl import JSONLStorage
from membrane.storage.ram import RAMStorage
from membrane.logic.ingestion import IngestionEngine
from membrane.logic.retrieval import RetrievalEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify")

TEST_DIR = "./test_membrane_data"

class MockEmbeddingModel:
    """Deterministic mock embedding model for offline/CI verification."""
    def encode(self, texts):
        res = []
        for t in texts:
            # Deterministic seed from text hash
            seed = abs(hash(t)) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.random(384).astype(np.float32)
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            res.append(vec)
        return np.array(res)

def setup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)

    db_path = os.path.join(TEST_DIR, "membrane.db")
    jsonl_path = os.path.join(TEST_DIR, "events.jsonl")

    sqlite = SQLiteStorage(db_path)
    jsonl = JSONLStorage(jsonl_path)
    ram = RAMStorage(sqlite)

    # Try using real model, fallback to mock if download fails or env var set
    use_mock = os.environ.get("MEMBRANE_TEST_MOCK_EMBEDDING", "0") == "1"

    if not use_mock:
        try:
            # Attempt to load real model (triggers download)
            _ = ram.model
        except Exception as e:
            logger.warning(f"Failed to load real embedding model ({e}). Falling back to MockEmbeddingModel.")
            use_mock = True

    if use_mock:
        logger.info("Using MockEmbeddingModel for verification.")
        ram._model = MockEmbeddingModel()

    ram.rebuild_index()

    ingestion = IngestionEngine(sqlite, jsonl, ram)
    retrieval = RetrievalEngine(sqlite, ram)

    return ingestion, retrieval, ram, sqlite

def test_positive_flow(ingestion, retrieval):
    logger.info("Testing Positive Flow...")

    scope = Scope(project_id="proj1", agent_id="agent1", session_id="sess1")
    event_id = str(uuid.uuid4())
    event = Event(
        event_id=event_id,
        scope=scope,
        type="observation",
        text="The sky is blue today.",
        ts_epoch_ms=int(time.time() * 1000)
    )

    ingestion.add_event(event)

    # Search Temp
    results = retrieval.temp_search(scope, "blue", k=5)
    assert len(results) > 0, "Should find the event"
    assert results[0].id == event_id

    # Promote
    candidates = ingestion.promote("proj1", [event_id])
    assert len(candidates) > 0, "Should generate candidates"
    # Expect summary at least
    summary = next((c for c in candidates if c.kind == "summary"), None)
    assert summary is not None
    assert "blue" in summary.content["text"]

    # Commit
    summary.status = "verified"
    res = ingestion.commit("proj1", [summary])
    assert res["upserted"] == 1

    # Search KB
    kb_res = retrieval.proj_kb_search("proj1", "blue", k=5)
    assert len(kb_res) > 0
    assert kb_res[0].id == summary.kb_item_id

def test_isolation(ingestion, retrieval):
    logger.info("Testing Isolation...")

    scope1 = Scope(project_id="proj1", agent_id="agent1", session_id="sess1")
    scope2 = Scope(project_id="proj1", agent_id="agent2", session_id="sess2") # Different agent
    scope3 = Scope(project_id="proj2", agent_id="agent1", session_id="sess1") # Different project

    e1 = Event(event_id=str(uuid.uuid4()), scope=scope1, type="obs", text="Secret 1", ts_epoch_ms=1000)
    e2 = Event(event_id=str(uuid.uuid4()), scope=scope2, type="obs", text="Secret 2", ts_epoch_ms=1000)
    e3 = Event(event_id=str(uuid.uuid4()), scope=scope3, type="obs", text="Secret 3", ts_epoch_ms=1000)

    ingestion.add_event(e1)
    ingestion.add_event(e2)
    ingestion.add_event(e3)

    # Agent 1 should only see Secret 1
    res1 = retrieval.temp_search(scope1, "Secret")
    # Should see Secret 1
    assert len(res1) == 1
    assert "Secret 1" in res1[0].snippet

    # Agent 2 should only see Secret 2
    res2 = retrieval.temp_search(scope2, "Secret")
    assert len(res2) == 1
    assert "Secret 2" in res2[0].snippet

    # Project 2 should only see Secret 3
    res3 = retrieval.temp_search(scope3, "Secret")
    assert len(res3) == 1
    assert "Secret 3" in res3[0].snippet

def test_idempotency_and_conflict(ingestion, retrieval):
    logger.info("Testing Idempotency & Conflict...")

    # Commit item
    item = KBItem(
        kb_item_id="item1", project_id="proj1", kind="fact",
        content={"text": "fact"}, provenance_event_ids=["e1"],
        confidence=0.5, status="proposed", ts_epoch_ms=1000
    )
    ingestion.commit("proj1", [item])

    # Commit same item again
    res = ingestion.commit("proj1", [item])
    # Expect 1 upsert (idempotent update logic returns upserted=1 even if no change?
    # Logic: results["upserted"] += 1 unless conflict logic rejects it.
    # Same item = no conflict = upserted.
    assert res["upserted"] == 1

    # Test Verified > Proposed
    item_v = item.model_copy()
    item_v.status = "verified"
    item_v.confidence = 0.6
    res = ingestion.commit("proj1", [item_v])
    assert res["upserted"] == 1

    curr = ingestion.sqlite.get_kb_item("item1")
    assert curr["status"] == "verified"

    # Test Verified vs Verified (low confidence delta)
    item_v2 = item_v.model_copy()
    item_v2.confidence = 0.65 # +0.05, less than 0.1 threshold
    item_v2.provenance_event_ids = ["e2"] # New provenance

    res = ingestion.commit("proj1", [item_v2])
    # Should be conflict (no upsert of status/conf, but merge prov)
    # My logic counts "conflicts"
    assert res["conflicts"] == 1

    curr = ingestion.sqlite.get_kb_item("item1")
    assert curr["confidence"] == 0.6 # Unchanged
    assert "e2" in curr["provenance_event_ids"] # Merged

def test_persistence():
    logger.info("Testing Persistence...")
    # Teardown and Re-init
    # Assuming setup() created DB on disk.

    db_path = os.path.join(TEST_DIR, "membrane.db")
    sqlite = SQLiteStorage(db_path)
    ram = RAMStorage(sqlite)

    # Use Mock if needed (should match what setup used if we want exact results but mock uses hash(text) seed so deterministic)
    # If environment variable set, use it.
    use_mock = os.environ.get("MEMBRANE_TEST_MOCK_EMBEDDING", "0") == "1"
    if use_mock:
        ram._model = MockEmbeddingModel()

    ram.rebuild_index()

    retrieval = RetrievalEngine(sqlite, ram)

    # Search for items added in previous tests
    # scope1 from test_isolation
    scope1 = Scope(project_id="proj1", agent_id="agent1", session_id="sess1")
    res = retrieval.temp_search(scope1, "Secret")
    assert len(res) > 0
    assert "Secret 1" in res[0].snippet

if __name__ == "__main__":
    try:
        ingestion, retrieval, ram, sqlite = setup()
        test_positive_flow(ingestion, retrieval)
        test_isolation(ingestion, retrieval)
        test_idempotency_and_conflict(ingestion, retrieval)

        # Persistence test requires re-init
        test_persistence()

        print("ALL TESTS PASSED")
    except Exception as e:
        logger.error(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)
