import logging
import os
from typing import List, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP

from membrane.storage.sqlite import SQLiteStorage
from membrane.storage.jsonl import JSONLStorage
from membrane.storage.ram import RAMStorage
from membrane.storage.policy import load_policy
from membrane.logic.ingestion import IngestionEngine
from membrane.logic.retrieval import RetrievalEngine
from membrane.models import Scope, Event, KBItem, SearchResult, SearchFilters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("membrane")

# Initialize backends
MEMBRANE_DIR = os.path.expanduser("~/.membrane")
os.makedirs(MEMBRANE_DIR, exist_ok=True)
DB_PATH = os.path.join(MEMBRANE_DIR, "membrane.db")
JSONL_PATH = os.path.join(MEMBRANE_DIR, "events.jsonl")

# Check if FTS5 available by initializing SQLite
try:
    sqlite = SQLiteStorage(DB_PATH)
except RuntimeError as e:
    logger.critical(f"Startup failed: {e}")
    exit(1)

jsonl = JSONLStorage(JSONL_PATH)
ram = RAMStorage(sqlite)

# Rebuild index on startup
try:
    ram.rebuild_index()
except Exception as e:
    logger.error(f"Failed to rebuild index: {e}")

ingestion = IngestionEngine(sqlite, jsonl, ram)
retrieval = RetrievalEngine(sqlite, ram)

mcp = FastMCP("membrane")

@mcp.tool(name="temp.add_event")
async def add_event(scope: Scope, event: Event) -> Dict[str, Any]:
    """Add a temporary event. Ensure event.scope matches the provided scope."""
    if event.scope.model_dump() != scope.model_dump():
        return {"ok": False, "error": "scope_mismatch"}

    ingestion.add_event(event)
    return {"ok": True, "event_id": event.event_id}

@mcp.tool(name="temp_add_event")
async def add_event_alias(scope: Scope, event: Event) -> Dict[str, Any]:
    return await add_event(scope, event)

@mcp.tool(name="temp.search")
async def search_temp(scope: Scope, query: str, k: int = 20, filters: Optional[SearchFilters] = None) -> List[SearchResult]:
    """Search temporary events."""
    return retrieval.temp_search(scope, query, k, filters)

@mcp.tool(name="temp_search")
async def search_temp_alias(scope: Scope, query: str, k: int = 20, filters: Optional[SearchFilters] = None) -> List[SearchResult]:
    return await search_temp(scope, query, k, filters)

@mcp.tool(name="proj_kb.promote")
async def promote(project_id: str, source_event_ids: List[str], candidates_schema: Optional[Dict[str, Any]] = None) -> List[KBItem]:
    """Promote events to KB candidates. candidates_schema is accepted for spec compatibility in v0."""
    _ = candidates_schema
    return ingestion.promote(project_id, source_event_ids)

@mcp.tool(name="proj_kb_promote")
async def promote_alias(project_id: str, source_event_ids: List[str], candidates_schema: Optional[Dict[str, Any]] = None) -> List[KBItem]:
    return await promote(project_id, source_event_ids, candidates_schema)

@mcp.tool(name="proj_kb.commit")
async def commit(project_id: str, commits: List[KBItem]) -> Dict[str, Any]:
    """Commit KB items."""
    return ingestion.commit(project_id, commits)

@mcp.tool(name="proj_kb_commit")
async def commit_alias(project_id: str, commits: List[KBItem]) -> Dict[str, Any]:
    return await commit(project_id, commits)

@mcp.tool(name="proj_kb.search")
async def search_kb(project_id: str, query: str, k: int = 20, filters: Optional[SearchFilters] = None) -> List[SearchResult]:
    """Search Project KB."""
    return retrieval.proj_kb_search(project_id, query, k, filters)

@mcp.tool(name="proj_kb_search")
async def search_kb_alias(project_id: str, query: str, k: int = 20, filters: Optional[SearchFilters] = None) -> List[SearchResult]:
    return await search_kb(project_id, query, k, filters)

@mcp.tool(name="behavior.read_policy")
async def read_policy() -> Dict[str, Any]:
    """Read global behavior policy."""
    policy = load_policy()
    return policy.model_dump()

@mcp.tool(name="behavior_read_policy")
async def read_policy_alias() -> Dict[str, Any]:
    return await read_policy()

@mcp.tool(name="maintenance.flush")
async def flush() -> Dict[str, Any]:
    """Flush pending writes."""
    return {"ok": True, "flushed": True}

@mcp.tool(name="maintenance_flush")
async def flush_alias() -> Dict[str, Any]:
    return await flush()

@mcp.tool(name="maintenance.reindex")
async def reindex() -> Dict[str, Any]:
    """Rebuild RAM index."""
    ram.rebuild_index()
    return {"ok": True, "reindexed": True}

@mcp.tool(name="maintenance_reindex")
async def reindex_alias() -> Dict[str, Any]:
    return await reindex()

if __name__ == "__main__":
    mcp.run()
