from datetime import datetime


def get_weather(location: str = ""):
    return '{"city": "北京", "temperature": "25°C", "condition": "晴"}'


def get_time(tz: str = ""):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
