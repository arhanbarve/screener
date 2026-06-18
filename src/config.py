import os
import yaml
from dotenv import load_dotenv

load_dotenv()

def load_config(path: str = "config.yaml") -> dict:
    f = open(path, "r")
    cfg = yaml.safe_load(f)
    f.close()
    return cfg

def get_env(key: str) -> str:
    val = os.environ.get(key)
    if val is None:
        raise KeyError(f"Missing required env var: {key}")
    return val
