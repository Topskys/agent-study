"""
文档加载器——视频加载器（暂未实现）

视频处理拆分为两条独立管线：
  画面帧管线: ffmpeg 抽帧 → PaddleOCR 提取画面文字
  音频管线: ffmpeg 提取音频 → Whisper 转录
"""

from rag.datatypes import RawDocument
from .base import BaseLoader


class VideoLoader(BaseLoader):
    """视频加载器（暂未实现）"""

    def load(self, file_path: str) -> list[RawDocument]:
        raise NotImplementedError(
            "VideoLoader 暂未实现。需要 ffmpeg + PaddleOCR + Whisper。"
        )

    def supported_extensions(self) -> list[str]:
        return [".mp4", ".avi", ".mov", ".mkv"]
