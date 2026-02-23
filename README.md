# Membrane

Membrane is a local MCP server that provides:
- Temp store (scoped short-term memory)
- Project KB (structured, graph-like, indexed knowledge base)
- Global behavior policy
- Hybrid retrieval (FTS5 + Vector)
- Strict JSON tool contracts

## Structure
- `membrane/models/`: Pydantic data models (`Scope`, `Event`, `KBItem`, `SearchResult`, etc.).
- `membrane/storage`: Storage implementations (SQLite/FTS5, JSONL, RAM vector index, Policy).
- `membrane/logic`: Core logic (Identity, Canonicalization, Ingestion, Retrieval).
- `membrane/server`: MCP server implementation.
- `verify_server.py`: End-to-end verification harness for positive/negative flows.

## Run locally
- Start server: `python3 membrane/server/main.py`
- Run verification (offline-safe): `MEMBRANE_TEST_MOCK_EMBEDDING=1 python3 verify_server.py`

## How to test further
- Run with real embeddings (network/model availability required): `python3 verify_server.py`
- Run offline deterministic verification (recommended for CI): `MEMBRANE_TEST_MOCK_EMBEDDING=1 python3 verify_server.py`
- Confirm clean repeatability by running the verifier twice in a row and checking both runs print `ALL TESTS PASSED`.
- Start the MCP server to ensure it boots without runtime errors: `python3 membrane/server/main.py`
- While server smoke-testing, verify persistence artifacts are created under the default local storage directory (`~/.membrane/`) and no startup exceptions are printed.
