"""任务调度中心：并行 / 串行任务编排执行。

对齐 v3 类图 TaskScheduleService：
- 组内任务并行（ThreadPoolExecutor 并发执行）；
- 组间按依赖顺序串行（后序组依赖前序组结果）。
executor 为注入式执行回调；不注入则不实际执行（仅返回 pending 状态）。
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .models import TaskGroup


class TaskScheduleService:
    """并行 / 串行任务调度。"""

    def schedule(
        self,
        groups: list[TaskGroup],
        executor: Callable[[str, dict], Any] | None = None,
        slots: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """按依赖顺序调度执行，返回 {intent_id: 执行结果}。

        executor 为 None 时只返回分组结构，不实际执行（结果为 pending）。
        """
        results: dict[str, Any] = {}
        slots = slots or {}
        ordered = self._topological_order(groups)

        for group in ordered:
            tasks: list[tuple[str, dict]] = [
                (it.intent_id, slots.get(it.intent_id, {})) for it in group.intents
            ]
            if executor is None:
                for intent_id, _ in tasks:
                    results[intent_id] = {"status": "pending"}
                continue
            if len(tasks) <= 1:
                for intent_id, kv in tasks:
                    results[intent_id] = self._run(intent_id, kv, executor)
                continue
            batch = self.run_parallel(tasks, executor)
            for intent_id, _ in tasks:
                results[intent_id] = batch.get(intent_id)
        return results

    def run_parallel(
        self, tasks: list[tuple[str, dict]], executor: Callable
    ) -> dict[str, Any]:
        """组内任务并发执行（保序收集结果）。"""

        def _do(item):
            intent_id, kv = item
            return intent_id, self._run(intent_id, kv, executor)

        with ThreadPoolExecutor(max_workers=max(2, len(tasks))) as pool:
            return dict(pool.map(_do, tasks))

    def run_serial(
        self, tasks: list[tuple[str, dict]], executor: Callable
    ) -> list[Any]:
        """按顺序依次执行（供外部直接调用 / 测试）。"""
        return [executor(i, kv) for i, kv in tasks]

    def _run(self, intent_id: str, kv: dict, executor: Callable) -> Any:
        try:
            return executor(intent_id, kv)
        except Exception as e:  # noqa: BLE001 - 调度层兜底，不让单个任务炸掉整批
            return {"error": str(e)}

    @staticmethod
    def _topological_order(groups: list[TaskGroup]) -> list[TaskGroup]:
        """按 group_id 升序稳定返回（依赖组的 group_id 更小）。"""
        return sorted(groups, key=lambda g: g.group_id)
