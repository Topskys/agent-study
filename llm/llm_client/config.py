"""配置加载：环境变量 + YAML，与 agent_mvp/config.py 风格对齐。

支持 ${ENV_VAR} 占位符替换；.env 自动加载（override=False，不覆盖已有环境变量）。
"""

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class LLMConfig:
    """LLM 模块配置。对应 YAML 里的 llm: 段。"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        provider: str = "openai_compat",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        timeout: float = 120,
        max_retries: int = 3,
        enable_cache: bool = False,
        task_models: dict[str, str] | None = None,
        extra: dict | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_cache = enable_cache
        self.task_models = task_models or {}
        self.extra = extra or {}

    # ---------- 构造便捷入口 ----------

    @classmethod
    def from_dict(cls, d: dict) -> "LLMConfig":
        models = d.get("models") or {}
        return cls(
            api_key=d.get("api_key", ""),
            base_url=d.get("base_url", ""),
            model=models.get("default") or d.get("model", ""),
            provider=d.get("provider", "openai_compat"),
            temperature=float(d.get("temperature", 0.7)),
            top_p=float(d.get("top_p", 0.95)),
            max_tokens=int(d.get("max_tokens", 2048)),
            timeout=float(d.get("timeout", 120)),
            max_retries=int(d.get("max_retries", 3)),
            enable_cache=bool(d.get("enable_cache", False)),
            task_models={k: v for k, v in models.items() if k != "default"} or None,
            extra=d.get("extra") or {},
        )


def load_config(path: str | os.PathLike = DEFAULT_CONFIG_PATH) -> LLMConfig:
    """加载 YAML 配置为 LLMConfig。YAML 形如：

    llm:
      api_key: ${API_KEY}
      base_url: ${BASE_URL}
      models:
        default: ...
        chat: ...
        intent: ...
    """
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    def _sub(m):
        return os.environ.get(m.group(1), "")

    text = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", _sub, text)
    cfg = yaml.safe_load(text) or {}
    llm = cfg.get("llm", cfg)
    return LLMConfig.from_dict(llm)
