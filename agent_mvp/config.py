"""配置层：YAML 配置加载，支持 ${ENV_VAR} 占位符替换。"""

import os
import re
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

SYSTEM_PROMPT = "你是一个可以调用工具来帮助用户的智能助手，请用中文回答。"


def load_config(path: str | os.PathLike = DEFAULT_CONFIG_PATH) -> dict:
    """从 YAML 加载配置，并支持 ${ENV_VAR} 占位符替换。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    def _sub(m):
        return os.environ.get(m.group(1), "")

    text = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", _sub, text)
    cfg = yaml.safe_load(text) or {}
    if not cfg.get("agent", {}).get("system_prompt"):
        cfg.setdefault("agent", {})["system_prompt"] = SYSTEM_PROMPT
    return cfg
