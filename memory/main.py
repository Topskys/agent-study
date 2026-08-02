from memory_system.core.memory_system import MemorySystem


def main():
    memory_system = MemorySystem(db_path="memory_system.db")

    user_memory = memory_system.get_user_memory("user_123")

    user_memory.add_working_memory("用户说他喜欢吃川菜")

    user_memory.start_session("session_456")
    user_memory.add_session_memory(
        "用户询问了关于Python的问题", {"session_id": "session_456"}
    )

    user_memory.add_long_term_memory("用户是一名后端开发工程师", {"importance": 0.8})

    memories = user_memory.retrieve_relevant_memories("用户的职业是什么？")
    print("相关记忆：")
    for memory in memories:
        print(f"  [{memory.memory_type.value}] {memory.content}")

    user_memory.end_session()

    user_memory.update_profile(
        {
            "base_info": {
                "name": "张三",
                "age": 25,
                "occupation": "后端开发工程师",
            },
            "preferences": {
                "food": "川菜",
                "hobby": "编程",
            },
        }
    )

    print("\n用户画像：")
    if user_memory.profile:
        print(f"  base_info: {user_memory.profile.base_info}")
        print(f"  preferences: {user_memory.profile.preferences}")

    user_memory.flush_working_memory()


if __name__ == "__main__":
    main()
