"""ID 生成工具。

提供全局唯一的 ID 生成能力，用于创建记忆、版本、事件等的唯一标识。
"""

import uuid


def generate_id() -> str:
    """生成一个全局唯一的 UUID 字符串 ID。

    返回:
        形如标准 UUID 的字符串，例如 "550e8400-e29b-41d4-a716-446655440000"。
    """
    return str(uuid.uuid4())
