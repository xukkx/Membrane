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

@mcp.tool(name="temp_add_event")
async def add_event(scope: Scope, event: Event) -> str:
    """Add a temporary event. Ensure event.scope matches the provided scope."""
    if event.scope.model_dump() != scope.model_dump():
        # Allow if event.scope is missing? No, it's required.
        # Strict consistency check
        return "Error: Scope mismatch between argument and event data."

    ingestion.add_event(event)
    return "Event added."

@mcp.tool(name="temp_search")
async def search_temp(scope: Scope, query: str, k: int = 20, filters: Optional[SearchFilters] = None) -> List[SearchResult]:
    """Search temporary events."""
    return retrieval.temp_search(scope, query, k, filters)

@mcp.tool(name="proj_kb_promote")
async def promote(project_id: str, source_event_ids: List[str]) -> List[KBItem]:
    """Promote events to KB candidates."""
    return ingestion.promote(project_id, source_event_ids)

@mcp.tool(name="proj_kb_commit")
async def commit(project_id: str, commits: List[KBItem]) -> Dict[str, Any]:
    """Commit KB items."""
    return ingestion.commit(project_id, commits)

@mcp.tool(name="proj_kb_search")
async def search_kb(project_id: str, query: str, k: int = 20, filters: Optional[SearchFilters] = None) -> List[SearchResult]:
    """Search Project KB."""
    return retrieval.proj_kb_search(project_id, query, k, filters)

@mcp.tool(name="behavior_read_policy")
async def read_policy() -> Dict[str, Any]:
    """Read global behavior policy."""
    policy = load_policy()
    return policy.model_dump()

@mcp.tool(name="maintenance_flush")
async def flush() -> str:
    """Flush pending writes."""
    return "Flushed."

@mcp.tool(name="maintenance_reindex")
async def reindex() -> str:
    """Rebuild RAM index."""
    ram.rebuild_index()
    return "Reindexed."

if __name__ == "__main__":
    mcp.run()
