import os
import tomli
from membrane.models import PolicyModel
import logging

logger = logging.getLogger(__name__)

POLICY_PATH = os.path.expanduser("~/.membrane/behavior_policy.toml")

def load_policy() -> PolicyModel:
    if not os.path.exists(POLICY_PATH):
        # Create default
        dir_path = os.path.dirname(POLICY_PATH)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Write default TOML
        default_toml = """
conflict_threshold = 0.1
embedding_model = "all-MiniLM-L6-v2"

[retrieval_caps]
fts_candidates = 200
rerank_k = 20
max_k = 20

[weights]
w_lex = 0.3
w_vec = 0.7
w_status = 0.1
w_conf = 0.1
w_recency = 0.1

[text_limits]
embed_text_max_chars = 1000
"""
        with open(POLICY_PATH, "w") as f:
            f.write(default_toml)
        logger.info(f"Created default policy at {POLICY_PATH}")

    try:
        with open(POLICY_PATH, "rb") as f:
            data = tomli.load(f)
        return PolicyModel(**data)
    except Exception as e:
        raise RuntimeError(f"Failed to load or validate policy from {POLICY_PATH}: {e}")
