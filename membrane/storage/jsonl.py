import json
import os
from typing import Dict, Any

class JSONLStorage:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._ensure_dir()

    def _ensure_dir(self):
        dir_path = os.path.dirname(self.filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def append(self, data: Dict[str, Any]):
        """Appends a dictionary as a JSON line."""
        try:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            # Fallback/Audit logging failure should not crash the main flow?
            # Spec says "Immediate write path for all events (temp + tool IO) as raw logs."
            # If this fails, it's bad. I'll let the exception propagate for now or log to stderr.
            # Given it's a critical audit log, maybe propagating is safer.
            print(f"Error writing to JSONL: {e}")
            raise e
