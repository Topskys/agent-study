import json
import sqlite3
from typing import Any, Dict, List, Optional

from ..models.graph import GraphEdge, GraphNode

"""图存储引擎：基于内存（字典 + 列表）实现的无向图存储，
维护节点（GraphNode）与边（GraphEdge），支持按属性查询节点、
以及通过深度优先搜索（DFS）查找两点之间的所有可达路径。

传入 db_path 时启用 SQLite 持久化（写穿透）：节点/边在内存与
磁盘间保持同步，重启后可恢复；未传 db_path 则纯内存运行。
"""


class GraphStore:
    """图结构存储：节点存于 dict（以 node_id 为键），边存于 list，
    全部驻留内存；提供节点/边的增删查、属性过滤查询，
    以及基于 DFS 的路径搜索能力。"""

    def __init__(self, store_id: str = "default_graph", db_path: Optional[str] = None):
        """初始化图存储：记录 store_id 并创建空的节点与边容器。"""
        self.store_id = store_id
        self.db_path = db_path
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        if self.db_path:
            self._init_db()
            self._load_all()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """创建持久化所需的图节点与图边数据表（若不存在）。"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}'
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                edge_id TEXT PRIMARY KEY,
                from_node_id TEXT NOT NULL,
                to_node_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.commit()
        conn.close()

    def _load_all(self):
        """启动时从 SQLite 将全部节点与边载入内存。"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT node_id, label, properties FROM graph_nodes")
        for node_id, label, props in cursor.fetchall():
            self.nodes[node_id] = GraphNode(
                node_id=node_id, label=label, properties=json.loads(props)
            )
        cursor.execute(
            "SELECT edge_id, from_node_id, to_node_id, relationship, properties "
            "FROM graph_edges"
        )
        for edge_id, frm, to, rel, props in cursor.fetchall():
            self.edges.append(
                GraphEdge(
                    edge_id=edge_id,
                    from_node_id=frm,
                    to_node_id=to,
                    relationship=rel,
                    properties=json.loads(props),
                )
            )
        conn.close()

    def add_node(self, node: GraphNode):
        """添加一个节点，若 node_id 已存在则覆盖。"""
        self.nodes[node.node_id] = node
        if self.db_path:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO graph_nodes (node_id, label, properties) "
                "VALUES (?, ?, ?)",
                (node.node_id, node.label, json.dumps(node.properties)),
            )
            conn.commit()
            conn.close()

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """按 node_id 获取节点；不存在时返回 None。"""
        return self.nodes.get(node_id)

    def delete_node(self, node_id: str):
        """删除节点，并同时删除与该节点相关的所有边。"""
        self.nodes.pop(node_id, None)
        self.edges = [
            e
            for e in self.edges
            if e.from_node_id != node_id and e.to_node_id != node_id
        ]
        if self.db_path:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM graph_nodes WHERE node_id = ?", (node_id,))
            cursor.execute(
                "DELETE FROM graph_edges WHERE from_node_id = ? OR to_node_id = ?",
                (node_id, node_id),
            )
            conn.commit()
            conn.close()

    def add_edge(self, edge: GraphEdge):
        """添加一条边。"""
        self.edges.append(edge)
        if self.db_path:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO graph_edges "
                "(edge_id, from_node_id, to_node_id, relationship, properties) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    edge.edge_id,
                    edge.from_node_id,
                    edge.to_node_id,
                    edge.relationship,
                    json.dumps(edge.properties),
                ),
            )
            conn.commit()
            conn.close()

    def get_edges(self, node_id: str) -> List[GraphEdge]:
        """返回与指定节点相连的所有边（作为起点或终点）。"""
        return [
            e
            for e in self.edges
            if e.from_node_id == node_id or e.to_node_id == node_id
        ]

    def delete_edge(self, edge_id: str):
        """按 edge_id 删除一条边。"""
        self.edges = [e for e in self.edges if e.edge_id != edge_id]
        if self.db_path:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM graph_edges WHERE edge_id = ?", (edge_id,))
            conn.commit()
            conn.close()

    def query_nodes(self, label: Optional[str] = None, **kwargs) -> List[GraphNode]:
        """按标签（label）和/或属性（kwargs 中的键值对）过滤查询节点，
        同时满足所有过滤条件才返回。"""
        results = list(self.nodes.values())
        if label:
            results = [n for n in results if n.label == label]
        for key, value in kwargs.items():
            results = [n for n in results if n.properties.get(key) == value]
        return results

    def find_path(
        self, from_node_id: str, to_node_id: str, max_depth: int = 5
    ) -> List[List[str]]:
        """使用深度优先搜索（DFS）查找从 from_node_id 到 to_node_id 的
        所有可达路径（路径为节点 id 列表），最大深度为 max_depth；
        返回路径列表，无路径时为空列表。"""
        paths = []

        # 内部 DFS 递归函数：visited 记录当前路径已访问的节点以防环路
        def dfs(current: str, target: str, visited: set, path: List[str], depth: int):
            # 超过最大深度时停止继续向下搜索
            if depth > max_depth:
                return
            # 找到目标节点，记录当前路径
            if current == target:
                paths.append(path + [current])
                return
            visited.add(current)
            # 遍历所有边，向未访问过的邻居节点继续搜索（无向图，双向可达）
            for edge in self.edges:
                next_node = None
                if edge.from_node_id == current and edge.to_node_id not in visited:
                    next_node = edge.to_node_id
                elif edge.to_node_id == current and edge.from_node_id not in visited:
                    next_node = edge.from_node_id
                if next_node:
                    dfs(next_node, target, visited, path + [current], depth + 1)
            # 回溯：将当前节点从 visited 中移除，允许其他路径经过
            visited.discard(current)

        dfs(from_node_id, to_node_id, set(), [], 0)
        return paths

    def clear(self):
        """清空全部节点与边。"""
        self.nodes.clear()
        self.edges.clear()
        if self.db_path:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM graph_nodes")
            cursor.execute("DELETE FROM graph_edges")
            conn.commit()
            conn.close()
