from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, Extra

# Scope
class Scope(BaseModel):
    project_id: str
    agent_id: str
    session_id: str
    thread_id: Optional[str] = None
    task_id: Optional[str] = None

# SearchFilters
class SearchFilters(BaseModel):
    thread_id: Optional[str] = None
    task_id: Optional[str] = None
    type: Optional[str] = None
    status: Optional[Literal["proposed", "verified"]] = None
    tags: Optional[List[str]] = None
    entities: Optional[List[str]] = None

    model_config = {"extra": "forbid"}

# Event
class Event(BaseModel):
    event_id: str
    scope: Scope
    type: str
    text: str
    tags: List[str] = []
    entities: List[str] = []
    ts_epoch_ms: int
    ts_iso_utc: Optional[str] = None

# KBItem
class KBItem(BaseModel):
    kb_item_id: str
    project_id: str
    kind: Literal["fact", "summary", "entity", "relation"]
    content: Dict[str, Any]
    provenance_event_ids: List[str]
    confidence: float
    status: Literal["proposed", "verified"]
    conflicts_with: List[str] = []
    supersedes: List[str] = []
    ts_epoch_ms: int
    ts_iso_utc: Optional[str] = None

# PolicyModel
class PolicyModel(BaseModel):
    retrieval_caps: Dict[str, int] = Field(default_factory=lambda: {"fts_candidates": 200, "rerank_k": 20, "max_k": 20})
    weights: Dict[str, float] = Field(default_factory=lambda: {"w_lex": 0.3, "w_vec": 0.7, "w_status": 0.1, "w_conf": 0.1, "w_recency": 0.1})
    conflict_threshold: float = 0.1
    embedding_model: str = "all-MiniLM-L6-v2"
    text_limits: Dict[str, int] = Field(default_factory=lambda: {"embed_text_max_chars": 1000})

# SearchResult
class SearchResult(BaseModel):
    id: str
    source: Literal["temp", "proj_kb"]
    snippet: str
    score: float
    why: Dict[str, Any]
    # why must include: fts_rank, cosine, status, confidence, scope
