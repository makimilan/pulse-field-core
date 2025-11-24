"""
Config management for Pulse-Field thresholds and seeds.
"""

import os
import yaml
from typing import Any, Dict, Optional


class Config:
    """
    Central configuration management.
    
    Loads from YAML files with defaults and environment overrides.
    """
    
    DEFAULT_CONFIG = {
        "seed": {"global": 42},
        "thresholds": {
            "tau_compat": 0.7,
            "tau_lambda": 0.001,
            "tau_cos": 0.25,
            "tau_amp": 1.0,
            "tau_energy": 0.05,
            "tau_l2": 1.0,
            "k_min": 2,
            "n_max": 16,
        },
        "energy": {
            "e_max": 1.5,
            "e_init_range": [0.8, 1.2],
        },
        "router": {
            "l_max": 10,
            "cache_ttl": 600,
        },
        "archive": {
            "k": 5,
            "ttl": 600,
        },
        "evolution": {
            "invariants_pass_rate_commit": 0.95,
            "invariants_pass_rate_cache": 0.99,
        },
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config.
        
        Args:
            config_path: Path to YAML config file.
        """
        self.config = dict(self.DEFAULT_CONFIG)
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self._merge_config(self.config, user_config)
    
    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge override into base."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value via dot notation.
        
        Args:
            key: Key path (e.g., "thresholds.tau_compat").
            default: Default if not found.
        
        Returns:
            Config value.
        """
        parts = key.split(".")
        current = self.config
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return default
            else:
                return default
        return current
    
    def set(self, key: str, value: Any):
        """
        Set config value via dot notation.
        
        Args:
            key: Key path.
            value: Value to set.
        """
        parts = key.split(".")
        current = self.config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    
    def to_dict(self) -> Dict:
        """Export config as dict."""
        return dict(self.config)
