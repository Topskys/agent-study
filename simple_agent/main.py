import os

from dotenv import load_dotenv

from agent import SimpleAgent, get_weather, get_time

load_dotenv()


def _env_bool(key: str, default: str = "false") -> bool:
    return os.getenv(key, default).strip().lower() in ("true", "1", "yes")


def main():
    agent = SimpleAgent(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        model=os.getenv("LLM_MODEL", "deepseek-ai/deepseek-v4-pro"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "1")),
        top_p=float(os.getenv("LLM_TOP_P", "0.95")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
        stream=_env_bool("LLM_STREAM"),
        timeout=int(os.getenv("LLM_TIMEOUT", "120")),
        extra_body={"chat_template_kwargs": {"thinking": _env_bool("LLM_THINKING")}},
    )
    agent.register_tool("get_weather", get_weather, "查询当前天气")
    agent.register_tool("get_time", get_time, "获取当前时间")
    try:
        print(agent.run("今天天气怎么样？"))
        print(agent.run("现在几点了？"))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
