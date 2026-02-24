# PATCH_NOTES

## What failed
- `verify_v0.py full` could fail determinism checks (`deterministic_promotion` / `deterministic_entity_extraction`) because `promote()` stamped candidates with wall-clock `time.time()`.

## Root cause
- Promotion output included non-deterministic `ts_epoch_ms` values, so repeated calls with identical inputs produced structurally different KBItem payloads.

## What changed
- `membrane/logic/ingestion.py::promote()` now derives a deterministic timestamp anchor:
  - `base_ts = max(e["ts_epoch_ms"] for e in events)`
- All promoted candidates in a call now use `ts_epoch_ms=base_ts` (summary + entities).
- `verify_v0.py` deterministic promotion test no longer monkeypatches time and validates deterministic behavior directly.

## Why this is safe for v0
- Preserves existing identity/canonicalization rules (`kb_item_id` unchanged).
- Keeps deterministic promotion invariant aligned with the adversarial runner.
- No networking/model behavior added; storage and retrieval invariants remain unchanged.
