"""
配置加载——ConfigLoader

从 YAML 文件加载配置，提供类型安全的链式访问。
"""

from pathlib import Path


class ConfigLoader:
    """加载 YAML 配置，提供类型安全的访问"""

    def __init__(self, path: str = ""):
        self.path = path
        self._data: dict = {}
        if path:
            self.load(path)

    def load(self, path: str | None = None) -> dict:
        p = path or self.path
        if not p:
            return self._data
        p = Path(p)
        if not p.exists():
            self._data = {}
            return self._data
        try:
            import yaml

            with open(p, encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        except ImportError:
            import json

            with open(p, encoding="utf-8") as f:
                self._data = json.load(f)
        return self._data

    def get(self, *keys: str, default=None):
        val: dict | None = self._data
        for k in keys:
            if not isinstance(val, dict):
                return default
            val = val.get(k)
            if val is None:
                return default
        return val

    def as_dict(self) -> dict:
        return self._data
