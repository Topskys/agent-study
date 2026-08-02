"""用户画像（UserProfile）数据模型定义。

定义用户画像的数据结构，用于持久化用户的基础信息、偏好、
分场景画像、配置、锁定字段以及版本与向量等信息。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UserProfile:
    """用户画像数据结构。

    聚合一个用户的多方面画像数据，包括基础信息、偏好设置、
    不同场景下的画像、配置项，以及锁定、版本、更新时间和向量表示。
    """

    user_id: str  # 用户唯一标识
    base_info: Dict[str, Any] = field(default_factory=dict)  # 用户基础信息
    preferences: Dict[str, Any] = field(default_factory=dict)  # 用户偏好设置
    scene_profiles: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # 分场景画像，键为场景名
    config: Dict[str, Any] = field(default_factory=dict)  # 用户配置项
    lock_keys: List[str] = field(
        default_factory=list
    )  # 被锁定、不允许被自动覆盖的字段名列表
    current_version: int = 1  # 画像当前版本号
    last_updated: Optional[float] = None  # 最近更新时间（时间戳）
    embedding: Optional[List[float]] = None  # 用户画像的向量表示（用于相似检索）
