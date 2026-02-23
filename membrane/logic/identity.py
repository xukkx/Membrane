import hashlib
import os
import json

def resolve_project_id(project_path: str = None) -> str:
    """Resolve project ID from folder path."""
    if project_path is None:
        project_path = os.getcwd()

    # Normalize path
    # Resolve symlinks, normalize separators, no trailing slash
    abs_path = os.path.abspath(os.path.realpath(project_path))
    normalized_path = abs_path.rstrip(os.sep)

    # Check for .membrane/project.json override (optional but recommended)
    override_path = os.path.join(normalized_path, ".membrane", "project.json")
    if os.path.exists(override_path):
        try:
            with open(override_path, "r") as f:
                data = json.load(f)
                if "project_id" in data:
                    return data["project_id"]
        except Exception:
            pass # Fallback to hash

    # Canonical hash
    return hashlib.sha256(normalized_path.encode('utf-8')).hexdigest()
