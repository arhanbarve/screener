import os
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_env(key: str) -> str:
    val = os.environ.get(key)
    if val is None:
        raise KeyError(f"Missing required env var: {key}")
    return val
