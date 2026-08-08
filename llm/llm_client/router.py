"""任务路由：按 task 标签选择模型。

设计：route 表（task -> model），未命中 task 时用 default_model。
请求携带 model 时优先请求显式指定，绕过路由。
"""

from dataclasses import dataclass, field

from .types import LLMRequest


@dataclass
class RouteRule:
    """一条路由规则：task 标签 → 模型。"""

    task: str
    model: str


@dataclass
class LLMRouter:
    """模型路由器：resolve(req) 返回模型名。"""

    default_model: str = ""
    rules: dict[str, str] = field(default_factory=dict)

    def resolve(self, req: LLMRequest) -> str:
        if req.model:
            return req.model
        model = self.rules.get(req.task)
        if model:
            return model
        return self.default_model

    def register(self, task: str, model: str) -> None:
        self.rules[task] = model

    @classmethod
    def from_mapping(
        cls, default_model: str, mapping: dict[str, str] | None
    ) -> "LLMRouter":
        return cls(default_model=default_model, rules=mapping or {})


# 便捷构造：config 里 llm.models 形如 {"default": "...", "chat": "...", "intent": "..."}
def router_from_config(models: dict[str, str]) -> LLMRouter:
    default = models.get("default", "")
    rules = {k: v for k, v in models.items() if k != "default"}
    return LLMRouter(default_model=default, rules=rules)
