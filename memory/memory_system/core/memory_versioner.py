"""记忆版本管理（MemoryVersioner）治理模块。

负责为记忆内容的变化记录版本历史，支持版本查询、回滚
以及两个版本之间的内容对比，为记忆的可追溯性提供保障。
"""

from datetime import datetime
from typing import Dict, List, Optional

from ..models.memory_item import MemoryItem
from ..models.version import MemoryVersion
from ..utils.id_generator import generate_id


class MemoryVersioner:
    """记忆版本器：以内存字典维护每条记忆的版本历史。"""

    def __init__(self):
        # 版本历史索引：memory_id -> 该记忆的版本列表（按创建先后排列）
        self.versions: Dict[str, List[MemoryVersion]] = {}

    def create_version(self, memory: MemoryItem) -> MemoryVersion:
        """为记忆创建一条新版本快照。

        生成新的版本记录并追加到该记忆的版本列表，
        同时将该记忆的 version 字段更新为最新版本序号。

        :param memory: 待创建版本快照的记忆条目
        :return: 新创建的版本记录（MemoryVersion）
        """
        version = MemoryVersion(
            version_id=generate_id(),
            memory_id=memory.memory_id,
            content=memory.content,
            created_at=datetime.now(),
        )
        # 首次记录该记忆时初始化其版本列表
        if memory.memory_id not in self.versions:
            self.versions[memory.memory_id] = []
        self.versions[memory.memory_id].append(version)
        # 以版本列表长度作为最新版本号并回写到记忆条目
        memory.version = len(self.versions[memory.memory_id])
        return version

    def get_versions(self, memory_id: str) -> List[MemoryVersion]:
        """查询指定记忆的全部版本历史。

        :param memory_id: 记忆唯一标识
        :return: 版本列表（不存在时返回空列表）
        """
        return list(self.versions.get(memory_id, []))

    def rollback(self, memory: MemoryItem, version_id: str) -> Optional[MemoryItem]:
        """将记忆回滚到指定版本的内容。

        :param memory: 待回滚的记忆条目
        :param version_id: 目标版本 ID
        :return: 回滚成功返回更新后的记忆条目；版本不存在返回 None
        """
        versions = self.versions.get(memory.memory_id, [])
        for v in versions:
            if v.version_id == version_id:
                # 用历史版本内容覆盖当前记忆并刷新更新时间
                memory.content = v.content
                memory.updated_at = datetime.now()
                return memory
        return None

    def diff(
        self, memory_id: str, version_a: str, version_b: str
    ) -> Optional[Dict[str, str]]:
        """对比同一记忆的两个版本内容。

        :param memory_id: 记忆唯一标识
        :param version_a: 旧版本 ID
        :param version_b: 新版本 ID
        :return: {"old": 旧内容, "new": 新内容}；任一版本不存在时返回 None
        """
        versions = self.versions.get(memory_id, [])
        va = next((v for v in versions if v.version_id == version_a), None)
        vb = next((v for v in versions if v.version_id == version_b), None)
        if not va or not vb:
            return None
        return {"old": va.content, "new": vb.content}
