# Membrane

Membrane is a local MCP server that provides:
- Temp store (scoped short-term memory)
- Project KB (structured, graph-like, indexed knowledge base)
- Global behavior policy
- Hybrid retrieval (FTS5 + Vector)
- Strict JSON tool contracts

## Structure
- `membrane/models.py`: Pydantic models.
- `membrane/storage`: Storage implementations (SQLite, JSONL, RAM, Policy).
- `membrane/logic`: Core logic (Identity, Canonicalization, Ingestion, Retrieval).
- `membrane/server`: MCP server implementation.
