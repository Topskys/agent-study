"""记忆图谱（Graph）数据模型定义。

定义图谱中的节点（GraphNode）与边（GraphEdge）的数据结构，
用于以图的形式组织记忆之间的关联关系。
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class GraphNode:
    """图谱节点，表示记忆或实体。"""

    node_id: str  # 节点唯一标识
    label: str  # 节点标签
    properties: Dict[str, Any] = field(default_factory=dict)  # 节点附加属性


@dataclass
class GraphEdge:
    """图谱边，表示两个节点之间的关联关系。"""

    edge_id: str  # 边唯一标识
    from_node_id: str  # 起点节点 ID
    to_node_id: str  # 终点节点 ID
    relationship: str  # 关联关系类型
    properties: Dict[str, Any] = field(default_factory=dict)  # 边的附加属性
