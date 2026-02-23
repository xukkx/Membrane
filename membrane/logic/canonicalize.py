import hashlib
import json
from typing import Dict, Any

def jcs_canonical(data: Dict[str, Any]) -> bytes:
    """Produces JCS compliant canonical JSON (RFC 8785)."""
    # ensure sorted keys
    # ensure no whitespace
    # ensure UTF-8
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

def normalize_content_values(content: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes string values: strip(), collapse internal whitespace."""
    normalized_content = {}
    for k, v in content.items():
        if isinstance(v, str):
            normalized_content[k] = " ".join(v.strip().split())
        else:
            normalized_content[k] = v
    return normalized_content

def generate_kb_item_id(project_id: str, kind: str, content: Dict[str, Any]) -> str:
    """Generates stable ID for KB items."""
    normalized_content = normalize_content_values(content)

    identity_payload = {
        "project_id": project_id,
        "kind": kind,
        "content": normalized_content
    }
    canonical_bytes = jcs_canonical(identity_payload)
    return hashlib.sha256(canonical_bytes).hexdigest()

def generate_embedding_text(kind: str, content: Dict[str, Any]) -> str:
    """Generates deterministic text for embedding."""
    normalized_content = normalize_content_values(content)

    if kind == "fact":
        return f"FACT | subject={normalized_content.get('subject', '')} | predicate={normalized_content.get('predicate', '')} | object={normalized_content.get('object', '')}"
    elif kind == "summary":
        return f"SUMMARY | scope={normalized_content.get('scope', '')} | text={normalized_content.get('text', '')}"
    elif kind == "entity":
        return f"ENTITY | type={normalized_content.get('type', '')} | name={normalized_content.get('name', '')}"
    elif kind == "relation":
        return f"REL | src={normalized_content.get('src', '')} | rel={normalized_content.get('rel', '')} | dst={normalized_content.get('dst', '')}"
    else:
        # Fallback to JCS dump if kind is unknown (though models enforce Literal)
        return json.dumps(normalized_content, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
