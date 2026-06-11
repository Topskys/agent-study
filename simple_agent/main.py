import os
from dotenv import load_dotenv
from agent import SimpleAgent, get_weather, get_time

load_dotenv()


def main():
    agent = SimpleAgent(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        model=os.getenv("LLM_MODEL", "deepseek-ai/deepseek-v4-flash"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "1")),
        top_p=float(os.getenv("LLM_TOP_P", "0.95")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
        timeout=int(os.getenv("LLM_TIMEOUT", "120")),
    )
    agent.register_tool("get_weather", get_weather, "查询当前天气")
    agent.register_tool("get_time", get_time, "获取当前时间")

    print("Simple Agent (输入 /exit 退出)")
    while True:
        user_input = input("\n你: ")
        if user_input.strip().lower() in ("/exit", "/quit"):
            break
        try:
            reply = agent.run(user_input)
            print(f"助手: {reply}")
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
