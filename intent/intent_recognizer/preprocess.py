"""预处理服务：错别字修正 / 代词消解 / 短提问扩写。

对齐 v3 类图 TextPreprocessService。职责：净化输入，保留用户核心意图，
不添加无关信息；任一子步骤判定信息缺失 → 返回 ambiguous 标志，
交由调度器走主动交互补全。

设计要点：
- 代词只消解指物代词（它/这个/那个/这份/这些/它们/此），不动人称（你/我）；
- 短提问扩写由注入式 llm_expand 回调完成，无回调不扩写；
- 带"完整短指令守卫"：动作+目标齐全的短指令不扩写，避免破坏确定性规则识别。
"""

import json
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
_DEFAULT_VOCAB_PATH = _RESOURCE_DIR / "business_vocab.json"

# 指物代词（参与替换）；人称代词不参与消解
_REFERENTIAL_PRONOUNS = ("这些", "这个", "那个", "这份", "它们", "它", "此")

# 完整短指令的动作前缀（查/发/打开/关闭/算/记住+目标 视为信息完整）
_COMPLETE_ACTION_PREFIXES = (
    "查一下",
    "查",
    "发",
    "打开",
    "关闭",
    "算一下",
    "算",
    "记住",
    "读取",
    "看",
)

# 泛化动词：动作不明确，需要扩写/补全
_GENERIC_VERBS = re.compile(r"(处理|弄一下|弄|搞一下|搞|看一下|看看|安排一下)")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_entities_from_text(text: str, business_vocab: List[str]) -> List[str]:
    """从单段文本提取实体：业务词命中 + 至多前 4 字修饰语。

    供 TextPreprocessService 与外部共用（轻量信号词，复杂指代靠 LLM 兜底）。
    """
    entities: List[str] = []
    for word in sorted(business_vocab, key=len, reverse=True):
        idx = text.rfind(word)
        if idx == -1:
            continue
        start = max(0, idx - 4)
        entity = text[start : idx + len(word)]
        # 去掉前导的动词/介词等噪声字
        entity = entity.lstrip("把的在与和为从看算发")
        if entity:
            entities.append(entity)
    return entities


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein 编辑距离（短词场景够用）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


class TextPreprocessService:
    """预处理：错别字修正 → 代词消解 → 短提问扩写。

    llm_expand: 可选回调 llm_expand(text, history) -> str | None，
    用于短提问扩写；不注入则跳过扩写。
    """

    def __init__(
        self,
        business_vocab: Optional[List[str]] = None,
        vocab_path: Path = _DEFAULT_VOCAB_PATH,
        llm_expand: Optional[Callable[[str, List[str]], Optional[str]]] = None,
    ):
        if business_vocab is None:
            data = _load_json(vocab_path)
            business_vocab = data.get("business_vocab", [])
        self.business_vocab: List[str] = list(business_vocab)
        self._llm_expand = llm_expand

    # ---------- 主流程 ----------

    def process(
        self, text: str, history: Optional[List[str]] = None
    ) -> Tuple[str, bool]:
        """预处理净化，返回 (净化后文本, 是否信息缺失/歧义)。"""
        history = history or []
        ambiguous = False

        text = self.correct_typos(text)

        text, amb = self.resolve_pronouns(text, history)
        ambiguous = ambiguous or amb

        text, amb = self.expand_short_query(text, history)
        ambiguous = ambiguous or amb

        return text, ambiguous

    # ---------- ① 错别字修正 ----------

    def correct_typos(self, text: str) -> str:
        """滑窗 + 编辑距离 ≤1 匹配业务词库，命中替换（"周抱"→"周报"）。"""
        corrected = text
        # 长词优先，避免短词先替换破坏长词匹配
        for word in sorted(self.business_vocab, key=len, reverse=True):
            corrected = self._fix_typo(corrected, word)
        return corrected

    def _fix_typo(self, text: str, word: str) -> str:
        """在 text 中用 word 替换与它编辑距离 ≤1 的滑窗片段。"""
        length = len(word)
        if length == 0:
            return text
        result: List[str] = []
        i = 0
        n = len(text)
        while i <= n - length:
            window = text[i : i + length]
            if window != word and _edit_distance(window, word) <= 1:
                result.append(word)
                i += length
                continue
            result.append(text[i])
            i += 1
        while i < n:
            result.append(text[i])
            i += 1
        return "".join(result)

    # ---------- ② 代词消解 ----------

    def resolve_pronouns(
        self, text: str, history: Optional[List[str]] = None
    ) -> Tuple[str, bool]:
        """消解指物代词：从历史提取最近实体替换；无实体可消解 → 歧义。"""
        history = history or []
        entities = self.extract_entities(history)

        if not entities:
            if any(p in text for p in _REFERENTIAL_PRONOUNS):
                return text, True
            return text, False

        resolved = text
        for pronoun in _REFERENTIAL_PRONOUNS:
            resolved = self._replace_pronoun(resolved, pronoun, entities[0])
        return resolved, False

    def extract_entities(self, history: Optional[List[str]] = None) -> List[str]:
        """从历史文本提取最近实体：业务词库命中 + 前缀修饰语。"""
        history = history or []
        entities: List[str] = []
        for text in reversed(history):
            if not text:
                continue
            found = extract_entities_from_text(text, self.business_vocab)
            if found:
                entities.extend(found)
        return entities

    def _replace_pronoun(self, text: str, pronoun: str, entity: str) -> str:
        """替换代词为实体。代词后紧跟业务名词时不替换（避免"这个合同"变"合同合同"）。"""
        result: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            if text.startswith(pronoun, i):
                end = i + len(pronoun)
                nxt = text[end:]
                followed_by_noun = any(nxt.startswith(w) for w in self.business_vocab)
                if not followed_by_noun:
                    result.append(entity)
                    i = end
                    continue
            result.append(text[i])
            i += 1
        return "".join(result)

    # ---------- ③ 短提问扩写 ----------

    def expand_short_query(
        self, text: str, history: Optional[List[str]] = None
    ) -> Tuple[str, bool]:
        """短提问扩写。守卫：len<5 且不匹配动作模式且无上下文时才扩写。

        返回 (扩写后文本, 是否仍歧义)。
        """
        history = history or []
        if not self._llm_expand:
            return text, self._is_ambiguous(text) and not history

        # 有可用上下文时交由规则/代词消解处理，不强制扩写
        if history:
            return text, False

        if not self._is_ambiguous(text):
            return text, False

        try:
            expanded = self._llm_expand(text, history)
        except Exception:
            expanded = None
        if expanded and expanded != text:
            return expanded, False
        return text, True

    def _is_ambiguous(self, text: str) -> bool:
        """判断文本是否信息缺失/动作不明。"""
        text = text.strip()
        if not text:
            return True
        if len(text) < 5:
            # 动作+目标齐全的短指令（如"查周报""算3+5"）视为完整
            if any(text.startswith(p) for p in _COMPLETE_ACTION_PREFIXES):
                return False
            return True
        # 长提问：含泛化动词（处理/弄/搞/看看）→ 动作不明
        if _GENERIC_VERBS.search(text):
            return True
        return False


# 兼容旧命名：v2 的 Preprocessor 即 v3 的 TextPreprocessService
Preprocessor = TextPreprocessService
