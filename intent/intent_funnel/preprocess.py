"""文本预处理（轻量，V3 §3.1 清洗步骤）。

- 全角转半角 / 去除首尾空白
- 敏感词过滤由 RuleMatcher 高危清单负责（不进本层）
实际语义层直接消费清洗后的文本。
"""

import re


def _full_to_half(text: str) -> str:
    result = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            result.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        else:
            result.append(ch)
    return "".join(result)


def normalize(text: str) -> str:
    if text is None:
        return ""
    text = _full_to_half(str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text