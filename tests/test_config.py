# tests/test_config.py
import os
import pytest
from src.config import load_config, get_env

def test_load_config_returns_dict():
    cfg = load_config("config.yaml")
    assert isinstance(cfg, dict)
    assert "universe" in cfg
    assert "liquidity_gate" in cfg
    assert "factors" in cfg

def test_load_config_has_weights():
    cfg = load_config("config.yaml")
    weights = cfg["factors"]["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9

def test_get_env_missing_key_raises():
    with pytest.raises(KeyError):
        get_env("NONEXISTENT_KEY_XYZ")

def test_config_has_sizing_block():
    cfg = load_config()
    sizing = cfg["sizing"]
    assert sizing["enabled"] is True
    assert sizing["vol_window"] == 63
    assert sizing["name_cap"] == 0.10
    assert sizing["sector_cap"] == 0.25
