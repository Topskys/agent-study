from datetime import datetime


def get_weather():
    """查询当前天气"""
    return '{"city": "北京", "temperature": "25°C", "condition": "晴"}'


def get_time():
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
