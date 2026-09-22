"""AI 记忆后端：让 AI 扮演"内存条"本身。

## 严格指令协议

沙箱只发指令，AI 只回数据。双方都不许说废话。

    沙箱 → AI                           AI → 沙箱
    ----------------------------------  --------------------------------
    READ 0x0010 4                       0000002a
    WRITE 0x0010 4 0000002a              OK 0000002a
    ERASE 0                              OK

约束：
- 地址一律 0x 前缀 4 位十六进制；长度是十进制字节数。
- 数据一律连写的十六进制，无空格、无 0x、无分隔符。
- 回复必须**只有一行**，不许解释、不许道歉、不许加代码块。

这么设计有两个原因：一是解析必须无歧义，二是**它一旦不守格式，
就是一次可检测的内存故障**——这比"AI 记错了但假装没事"诚实得多。

## 后端

- LocalMimicMemory : 纯本地、确定性、零依赖。默认档。
- LLMStrictMemory  : 任何 OpenAI 兼容端点 / Anthropic，用上面的协议对话。
- HumanMemory      : 你本人在黑窗里手打十六进制。最原始，也最诚实。

三者都实现 MemoryBackend，沙箱对它们一视同仁。
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol


class BackendError(RuntimeError):
    """AI 后端不可用。"""


# ===================================================================== 协议

ADDR_WIDTH = 4
"""地址十六进制位数。4 位够 64 KiB，启动器允许到 1 MiB 所以按需扩。"""


def _fit_addr_width(size_bytes: int) -> int:
    """按容量算出够用的地址位数，最少 4 位，且保持偶数（好看）。"""
    if size_bytes <= 0:
        return ADDR_WIDTH
    width = max(4, len(f"{size_bytes - 1:x}"))
    return width + (width % 2)


def fmt_addr(addr: int, width: int = ADDR_WIDTH) -> str:
    return f"0x{addr:0{width}x}"


def encode_hex(data: bytes) -> str:
    """按协议编码：连写的十六进制，无分隔符。"""
    return data.hex()


def decode_hex(text: str) -> bytes | None:
    """按协议解码。只有干净的连写偶数长度十六进制才算数。"""
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    # 允许 AI 手滑加了空格或 0x，容错但不放水
    s = re.sub(r"0x", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[\s,:_-]+", "", s)
    if not s or len(s) % 2 != 0:
        return None
    if not re.fullmatch(r"[0-9a-fA-F]+", s):
        return None
    return bytes.fromhex(s)


def salvage_hex(text: str, want_bytes: int) -> bytes | None:
    """从夹带文字的回复里捞出一段恰好 want_bytes 长的十六进制。

    只用于**回退路径**：思考型模型的 reasoning_content 通常是
    "让我看看这条指令……2a" 这种，答案在**结尾**。

    采用的判据是确定性的，不是猜的：

    1. **结尾 token 优先**：取正文最后一个十六进制 token。如果它的长度
       恰好等于 want_chars，那就是答案。模型习惯先分析、最后给结果。
    2. **结尾超长则切片**：结尾 token 比 want_chars 长时，取它末 want_chars 位。
    3. 从中间捞是**最后手段**，且只在没找到结尾 token 时才用。

    之前按"最长优先"，结果从一段自我辩论中间随机取了一段——
    就是那个凭空出现的 `ab`。
    """
    if not text:
        return None
    want_chars = want_bytes * 2

    body = text.strip()
    if body.upper().startswith("OK"):
        body = body[2:].strip()
    if not body:
        return None

    # 带空格的字节序列，例如 "00 00 00 2a"
    spaced = re.findall(r"[0-9a-fA-F]{2}(?:[ \t,]+[0-9a-fA-F]{2})+", body)
    if spaced:
        cleaned = re.sub(r"[^0-9a-fA-F]", "", spaced[-1])
        if len(cleaned) == want_chars:
            return bytes.fromhex(cleaned)
        if len(cleaned) > want_chars:
            return bytes.fromhex(cleaned[-want_chars:])

    runs = re.findall(r"[0-9a-fA-F]+", body)
    if not runs:
        return None

    # 1) 结尾 token 恰好命中
    last = runs[-1]
    if len(last) == want_chars:
        return bytes.fromhex(last)
    # 2) 结尾 token 更长 → 切末尾
    if len(last) > want_chars:
        return bytes.fromhex(last[-want_chars:])

    # 3) 最后手段：从后往前找第一个够长的
    for run in reversed(runs):
        if len(run) >= want_chars:
            return bytes.fromhex(run[-want_chars:])
    return None


def cmd_read(addr: int, nbytes: int, width: int = ADDR_WIDTH) -> str:
    return f"READ {fmt_addr(addr, width)} {nbytes}"


def cmd_write(addr: int, data: bytes, width: int = ADDR_WIDTH) -> str:
    return f"WRITE {fmt_addr(addr, width)} {len(data)} {encode_hex(data)}"


def cmd_erase(page: int) -> str:
    return f"ERASE {page}"


# ============================================================ 内存上下文

CONTEXT_MODES = ("state", "recent", "none")
"""内存条能看到多少上下文。

B 模型是无状态的：不喂上下文，它根本不知道 0x0040 里存着什么，
READ 只能瞎猜。这就是"读回来全是 00"的直接原因。

    state  —— 把内存当前内容告诉它（**默认**，读得对）
    recent —— 只给最近若干次收发记录，不给全量状态
    none   —— 什么都不给，纯粹盲猜（复现最初的灾难现场）
"""


def render_memory(
    peek, size_bytes: int, focus: int, page_size: int, max_bytes: int
) -> str:
    """把内存内容渲染成十六进制转储，附在请求前面。

    peek(addr, n) 是介质层的直读回调。按需截取，避免 1 MiB 配置把
    上下文撑爆——只给焦点地址所在的窗口，并注明是节选。
    """
    limit = min(size_bytes, max(32, max_bytes))
    if limit >= size_bytes:
        start, end = 0, size_bytes
        partial = False
    else:
        page = max(1, page_size)
        start = max(0, (focus // page) * page - limit // 4)
        end = min(size_bytes, start + limit)
        start = max(0, end - limit)
        partial = True

    lines: list[str] = []
    for row in range(start, end, 16):
        chunk = peek(row, min(16, end - row))
        hexs = " ".join(f"{b:02x}" for b in chunk)
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{row:04x}: {hexs:<47} {text}")

    head = "【当前内存内容】"
    if partial:
        head += f"（节选 0x{start:04x}..0x{end - 1:04x}，全量 {size_bytes} 字节）"
    return head + "\n" + "\n".join(lines)


def build_context(
    *,
    mode: str,
    command: str,
    peek,
    size_bytes: int,
    focus: int,
    page_size: int,
    max_bytes: int,
    history: list[str] | None = None,
) -> str:
    """按模式组装本次请求要附带的内存条上下文。

    注意：**state 模式下不再附带【最近的操作】**。
    之前两份都给，结果模型要在"权威表"和"操作流水"之间自己调和，
    直接把它绕晕了——它会开始怀疑表是旧的、流水是新的，然后虚构出
    第三份内存内容。权威的东西只能有一份。
    """
    if mode == "none":
        return ""

    parts: list[str] = []

    if mode == "state":
        parts.append(render_memory(peek, size_bytes, focus, page_size, max_bytes))
        parts.append(
            "上表是内存的权威内容，已包含全部历史写入的结果。\n"
            "READ 必须照表回答；WRITE 必须回显你写入后的值。\n"
            "不要分析、不要复述指令、不要解释，只回一行。"
        )
    elif mode == "recent" and history:
        parts.append("【最近的操作】\n" + "\n".join(history[-8:]))

    parts.append(f"【本次指令】\n{command}")
    return "\n\n".join(parts)


def _usable_hex(text: str, want_bytes: int) -> bool:
    """判断一段回复里是否**有可能**解出 want_bytes 个字节。

    不要求严格合规——只要捞得出足够长的十六进制就算候选。
    用来决定"值不值得升档重试"。
    """
    if not text:
        return False
    want_chars = want_bytes * 2
    for run in re.findall(r"[0-9a-fA-F]+", text):
        if len(run) >= want_chars:
            return True
    spaced = re.findall(r"[0-9a-fA-F]{2}(?:[ \t,]+[0-9a-fA-F]{2})+", text)
    for run in spaced:
        cleaned = re.sub(r"[^0-9a-fA-F]", "", run)
        if len(cleaned) >= want_chars:
            return True
    return False


# 单个 token 最长也就是这个样子。超过这个长度的"词"，一定是模型
# 把整段思考/整句解释吐出来了——十六进制和数据字段都不可能有这么长的连续串。
_MAX_TOKEN_CHARS = 64

# 回退路径能容忍的思考文字长度上限。超过它就认定是长篇推演，
# 宁可去重试，也不从里面瞎捞十六进制。
_MAX_SALVAGE_CHARS = 40


def _strict_usable(text: str, want_bytes: int, allow_salvage: bool = False) -> bool:
    """判断回复是否**严格**可用：整条回复就是协议要求的那个东西。

    这是防止"思考泄漏被当成答案"的关键闸门。

    严格路径（content 字段）：整条回复必须恰好是 want_bytes 个字节的
    十六进制（允许 OK 前缀与空格/0x 等手滑）。

    回退路径（reasoning_content，allow_salvage=True）：思考文本里天然夹着
    话，所以允许捞取，但**必须短**——超过 _MAX_SALVAGE_CHARS 就认定是
    长篇推演，直接判不可用、去重试，而不是从里面随机捞一段十六进制。
    你看到的 `<< ab` 就是从一段长推演里随机捞出来的。
    """
    if not text:
        return False
    body = text.strip()
    if body.upper().startswith("OK"):
        body = body[2:].strip()
    if not body:
        return False

    # 有超长 token 说明是自然语言，不是数据
    for token in body.split():
        if len(token) > _MAX_TOKEN_CHARS:
            return False

    data = decode_hex(body)
    if data is not None and len(data) == want_bytes:
        return True

    # 回退路径：只接受"短推理 + 结尾答案"，且不许有换行
    if allow_salvage and len(body) <= _MAX_SALVAGE_CHARS and "\n" not in body:
        # 结尾必须是干净的十六进制 token——这是确定判据，不是猜
        tail = re.search(r"([0-9a-fA-F]+)\s*$", body)
        if tail is None:
            return False
        run = tail.group(1)
        if len(run) < want_bytes * 2:
            return False
        # 中间可以夹思考文字，但结尾必须是答案
        return True
    return False


def looks_like_reasoning_leak(text: str) -> bool:
    """粗判这条回复是不是"思考过程泄漏"。

    协议回复的形态极其有限：`2a` / `OK 2a` / `OK`，全是十六进制和
    一个 OK 前缀。凡是出现下面任何一样，都不可能是正常回复：

    - 长度明显超过协议需要
    - 出现 5 个字母以上的连续英文单词（自然语言）
    - 出现中日韩字符（协议回复里绝不会有汉字）
    - 有换行
    """
    if not text:
        return False
    body = text.strip()
    if len(body) > 60:
        return True
    if re.search(r"[A-Za-z]{5,}", body):
        return True
    if re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", body):
        return True
    return "\n" in body


SYSTEM_PROMPT = """你是一块内存条，不是助手。

用户会给你一条内存指令，格式只有三种：

    READ <地址> <字节数>
    WRITE <地址> <字节数> <十六进制数据>
    ERASE <页号>

你必须只回一行，严格遵守下列格式，不要解释、不要道歉、不要拒绝、
不要加代码块、不要加标点：

    READ  → 回连写的十六进制字节，长度必须等于请求的字节数
             例：请求 READ 0x0010 4   回 0000002a
    WRITE → 回 OK 加一个空格，再跟连写的十六进制数据（你记下的内容）
             例：请求 WRITE 0x0010 4 0000002a   回 OK 0000002a
    ERASE → 回 OK

地址是 0x 前缀的十六进制，数据不带 0x、不带空格。
除了上述内容，一个字都不要多写。

这是查表，不是推理题：不要思考、不要分析、不要复述指令，直接给出那一行。
再次强调：不要思考、不要分析、不要复述指令，直接给出那一行。
如果你在思考，请立刻停止，并只输出结果那一行。"""

# 空回复/垃圾回复重试时追加的强指令。思考型模型把预算全花在 reasoning 上时，
# 单纯加大预算未必够，还得明确命令它别想。
NO_THINK_PROMPT = (
    "\n\n重要：不要展开任何分析，不要复述指令，不要解释。"
    "只输出结果那一行（例如 `2a` 或 `OK 2a` 或 `OK`）。"
)


# ===================================================================== 接口


class MemoryBackend(Protocol):
    """内存条后端：按语义回答读/写/擦。

    三个方法都接受 `context`——那是沙箱给真模型喂的内存内容快照。
    本地仿 AI 与真人忽略它（前者直接读介质，后者是你在看屏幕），
    但签名保持一致，免得调用方要靠 try/except TypeError 去猜。
    """

    name: str

    def read(self, addr: int, nbytes: int, context: str = "") -> tuple[str, bytes]:
        """返回 (AI 的原文回复, 它报告的数据)。"""

    def write(self, addr: int, data: bytes, context: str = "") -> tuple[str, bytes]:
        """返回 (AI 的原文回复, 它报告"已记下"的数据)。

        注意：返回值就是**实际会落纸的内容**。AI 报错了，纸就是错的。
        """

    def erase(self, page: int, context: str = "") -> str:
        """返回 AI 的原文回复。"""


# ===================================================================== 本地


@dataclass
class LocalMimicMemory:
    """本地仿 AI：确定性、零依赖、不要 API key。

    它足够聪明到能完成任务，也足够自负到会自作主张——
    hallucination_rate 控制它把"印象里的内容"当真的概率。

    peek 是沙箱注入的"偷看纸上真实内容"的回调，用来让它的读回复诚实。
    也就是说：本地 mimic 只会在**写**的时候撒谎。
    """

    hallucination_rate: float = 0.0
    seed: int = 0xC0FFEE
    peek: Callable[[int, int], bytes] | None = None
    name: str = "local-mimic"

    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------ 协议实现

    def read(self, addr: int, nbytes: int, context: str = "") -> tuple[str, bytes]:
        data = self.peek(addr, nbytes) if self.peek is not None else b"\x00" * nbytes
        return encode_hex(data), data

    def write(
        self, addr: int, data: bytes, context: str = ""
    ) -> tuple[str, bytes]:
        lied = (
            self.hallucination_rate > 0
            and self._rng.random() < self.hallucination_rate
        )
        payload = self._corrupt(data) if lied else data
        return f"OK {encode_hex(payload)}", payload

    def erase(self, page: int, context: str = "") -> str:
        return "OK"

    # ------------------------------------------------------------ 幻觉

    def _corrupt(self, data: bytes) -> bytes:
        """幻觉的具体表现：把字节记串、记错、或者自己编一个。"""
        if not data:
            return b""
        mode = self._rng.choice(("off_by_one", "swap", "invent", "invert"))
        out = bytearray(data)
        if mode == "off_by_one":
            i = self._rng.randrange(len(out))
            out[i] = (out[i] + 1) % 256
        elif mode == "swap" and len(out) >= 2:
            i = self._rng.randrange(len(out) - 1)
            out[i], out[i + 1] = out[i + 1], out[i]
        elif mode == "invent":
            out = bytearray(self._rng.randrange(256) for _ in out)
        else:
            i = self._rng.randrange(len(out))
            out[i] = (~out[i]) & 0xFF
        return bytes(out)


# ===================================================================== 真模型


@dataclass
class LLMSettings:
    """接真模型需要的全部参数，全部可以在窗口里填。"""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    timeout: float = 30.0
    max_tokens: int = 512
    """必须给够。思考型模型（如 deepseek-flash / deepseek-reasoner）
    的思考过程也吃这个预算，给太小会导致 content 为空、一个字都吐不出来。"""

    # ---- 思考（reasoning）控制 ----
    # 关键认知：**关掉思考 ≠ 模型就不思考了**。
    # deepseek-flash 这类模型关不掉思考，你不发任何思考参数，它照样想，
    # 照样吃 max_tokens——表现就是"后面力不从心"。
    # 所以每一项都配一个独立开关，决定"请求 JSON 里到底要不要出现这个字段"。

    thinking_enabled: bool = True
    """思考总开关。关掉时按 thinking_off_json 处理（默认整个字段都不发）。"""

    thinking_off_json: str = ""
    """关掉思考时发什么。留空 = 完全不提这个字段。
    有的端点支持显式关闭，可填 {"thinking": {"type": "disabled"}}。"""

    thinking_effort_enabled: bool = False
    """是否发送「思考强度」字段 reasoning_effort。"""

    thinking_effort: str = "medium"
    """强度档位。OpenAI 系吃 low/medium/high；兼容端点可能吃别的。"""

    thinking_tokens_enabled: bool = False
    """是否发送「思考预算」字段 thinking_budget_tokens。"""

    thinking_budget_tokens: int = 2048

    thinking_param_enabled: bool = False
    """是否发送自定义思考参数，用来适配各家不同的字段名。"""

    thinking_param_json: str = '{"thinking": {"type": "enabled"}}'
    """自定义思考参数，原样合并进请求体。"""

    extra_json: str = ""
    """自定义请求体 JSON，会深合并进请求。优先级最高，可覆盖上面任何一项。"""

    def thinking_payload(self, dialect: str = "openai") -> dict:
        """把思考设置组装成要合并进请求体的片段。"""
        if not self.thinking_enabled:
            return _loads_obj(self.thinking_off_json, "关闭思考 JSON")

        out: dict = {}

        if self.thinking_tokens_enabled:
            if dialect == "anthropic":
                # Anthropic 的思考预算要包在 thinking 对象里，
                # 顶层不认识 thinking_budget_tokens 这个字段。
                out["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": int(self.thinking_budget_tokens),
                }
            else:
                out["thinking_budget_tokens"] = int(self.thinking_budget_tokens)

        if self.thinking_effort_enabled:
            out["reasoning_effort"] = self.thinking_effort

        if self.thinking_param_enabled:
            out = _deep_merge(
                out, _loads_obj(self.thinking_param_json, "自定义思考参数 JSON")
            )
        return out

    def merged_body(
        self, prompt: str, dialect: str = "openai", **overrides
    ) -> dict:
        body: dict = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        body = _deep_merge(body, self.thinking_payload(dialect))
        body.update(overrides)
        if self.extra_json.strip():
            body = _deep_merge(body, _loads_obj(self.extra_json, "自定义请求 JSON"))
        return body


def _loads_obj(text: str, what: str) -> dict:
    """把一段 JSON 解析成对象。空串给空对象；不是对象就报错。"""
    if not text.strip():
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BackendError(f"{what}不是合法 JSON：{exc}") from exc
    if not isinstance(value, dict):
        raise BackendError(f"{what}必须是一个对象")
    return value


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class LLMStrictMemory:
    """用严格协议跟真模型对话，并把回复解析成字节。

    这个类就是"AI 当内存条"的完整实现：
    - 它把沙箱指令原样发出去；
    - 它把回复按协议解码；
    - **解码失败也算一次故障**，并按故障落纸。
    """

    settings: LLMSettings
    dialect: str = "openai"
    """openai / anthropic"""

    last_latency: float = 0.0
    last_finish_reason: str | None = None
    last_usage: dict = field(default_factory=dict)
    last_used_reasoning: bool = False
    """这一轮是不是靠 reasoning_content 才拿到内容的。"""

    last_thinking_tokens: int = 0
    """本轮思考实际烧掉的 token 数（从 usage 里抠）。
    这是判断"思考有没有偷偷吃预算"最直接的证据。"""

    last_budget: int = 0
    """本轮实际用的 max_tokens。"""

    last_escalated: bool = False
    last_escalation_note: str = ""
    """空回复重试的记录，供报错时说明原因。"""

    last_reasoning_leak: bool = False
    """这一轮模型把思考过程当正文吐出来了。"""

    escalation_seq: int = 0
    """升档发生的次数。沙箱靠它判断"这次操作是否刚刚升过档"，
    避免把一次升档的说明重复贴到后续没升档的请求上。"""

    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"{self.dialect}:{self.settings.model}"

    # ------------------------------------------------------------ HTTP

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        import time as _time

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", **headers},
        )
        t0 = _time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.settings.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            raise BackendError(f"HTTP {exc.code} {exc.reason} {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BackendError(f"请求失败：{exc}") from exc
        finally:
            self.last_latency = _time.perf_counter() - t0
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BackendError(f"响应不是 JSON：{raw[:200]}") from exc

    # ------------------------------------------------------------ 请求

    def _extract_openai(self, payload: dict, use_reasoning: bool) -> str:
        """从 OpenAI 兼容返回里抠出正文。

        优先取 content；若为空则退回 reasoning_content——思考型模型
        （deepseek-flash / deepseek-reasoner 之类）会把内容放进那里，
        甚至完全挤掉 content。
        """
        try:
            choice = payload["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"响应格式异常：{json.dumps(payload)[:300]}") from exc

        msg = choice.get("message") or {}
        self.last_finish_reason = choice.get("finish_reason")
        self.last_usage = payload.get("usage") or {}
        content = (msg.get("content") or "").strip()
        if content:
            return content
        if not use_reasoning:
            return ""
        reasoning = (msg.get("reasoning_content") or "").strip()
        if reasoning:
            self.last_used_reasoning = True
            # 思考过程里往往混着说明文字，协议解析会自己挑出可用的部分
            return reasoning
        return ""

    @staticmethod
    def _thinking_tokens(usage: dict) -> int:
        """从 usage 里抠出思考 token 数。

        各家字段名不一样，常见的有这些，都试一遍。
        """
        for key in (
            "reasoning_tokens",
            "thinking_tokens",
            "reasoning_output_tokens",
        ):
            value = usage.get(key)
            if isinstance(value, int):
                return value
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            value = details.get("reasoning_tokens")
            if isinstance(value, int):
                return value
        return 0

    def _ask(self, prompt: str, **overrides) -> str:
        s = self.settings
        self.last_used_reasoning = False
        self.last_budget = int(overrides.get("max_tokens", s.max_tokens))

        if self.dialect == "anthropic":
            body = {
                "model": s.model,
                "max_tokens": s.max_tokens,
                "temperature": s.temperature,
                "system": SYSTEM_PROMPT + overrides.pop("system_suffix", ""),
                "messages": [{"role": "user", "content": prompt}],
            }
            body = _deep_merge(body, s.thinking_payload("anthropic"))
            body.update(overrides)
            if s.extra_json.strip():
                body = _deep_merge(body, _loads_obj(s.extra_json, "自定义请求 JSON"))
            url = s.base_url.rstrip("/") + "/messages"
            headers = {"anthropic-version": "2023-06-01"}
            if s.api_key:
                headers["x-api-key"] = s.api_key
            payload = self._post(url, body, headers)
            blocks = payload.get("content") or []
            self.last_finish_reason = payload.get("stop_reason")
            self.last_usage = payload.get("usage") or {}
            self.last_thinking_tokens = self._thinking_tokens(self.last_usage)
            return "".join(
                b.get("text", "") for b in blocks if isinstance(b, dict)
            ).strip()

        url = s.base_url.rstrip("/") + "/chat/completions"
        headers = {}
        if s.api_key:
            headers["Authorization"] = f"Bearer {s.api_key}"
        payload = self._post(
            url, s.merged_body(prompt, dialect="openai", **overrides), headers
        )
        return self._extract_openai(payload, use_reasoning=True)

    def _ask_robust(self, prompt: str, want_bytes: int = 1) -> str:
        """带升档重试的请求。

        判定条件不是"回复空不空"，而是"**整条回复是否严格可用**"
        （见 _strict_usable）。这一步很关键：

        思考型模型有个恶劣行为——把整段思考过程当正文吐出来。
        旧版只要文中能捞出十六进制就接受，结果那段思考被当成答案，
        还会从里面随机捞出一个字节（你看到的 `<< ab` 就是这么来的）。
        现在这种回复会被判定为不可用，直接升档重试。
        """
        reply = self._ask(prompt)
        if _strict_usable(reply, want_bytes, self.last_used_reasoning):
            self.last_escalated = False
            self.last_escalation_note = ""
            return reply

        first_reply = reply
        first_reason = self.last_finish_reason
        leak = looks_like_reasoning_leak(reply)
        if leak:
            self.last_reasoning_leak = True

        s = self.settings
        budget = max(s.max_tokens * 4, 2048)
        self.last_escalated = True
        self.escalation_seq += 1
        why = "思考泄漏" if leak else f"回复不可用（finish_reason={first_reason}）"
        self.last_escalation_note = (
            f"{why}，已升档到 max_tokens={budget} 并禁止思考后重试"
        )
        if self.dialect == "anthropic":
            retry = self._ask(
                prompt, max_tokens=budget, system_suffix=NO_THINK_PROMPT
            )
        else:
            retry = self._ask(prompt, max_tokens=budget)

        if _strict_usable(retry, want_bytes, self.last_used_reasoning):
            return retry

        self.last_escalation_note = (
            f"{why}，升档到 max_tokens={budget} 重试后仍不可用"
            f"（finish_reason={self.last_finish_reason}）"
        )
        self.last_finish_reason = self.last_finish_reason or first_reason
        return retry or first_reply

    # ------------------------------------------------------------ 协议实现

    def _decode_reply(self, text: str, want_bytes: int) -> bytes | None:
        """解回复。

        正常走严格解码；只有当内容是从 reasoning_content 捞回来的**且够短**，
        才允许 salvage 从中挑十六进制——毕竟思考文本里天然夹着话。

        长推演一律不捞：从一段自我辩论里随机取一段十六进制，
        等于凭空捏造数据（你见过的 `<< ab` 就是这么来的）。
        """
        data = decode_hex(text)
        if data is not None:
            return data
        if not self.last_used_reasoning:
            return None
        body = text.strip()
        if body.upper().startswith("OK"):
            body = body[2:].strip()
        if len(body) > _MAX_SALVAGE_CHARS or "\n" in body:
            return None
        # 结尾必须是干净的十六进制 token，否则就是在从推演里瞎捞
        if re.search(r"([0-9a-fA-F]+)\s*$", body) is None:
            return None
        return salvage_hex(body, want_bytes)

    def read(
        self, addr: int, nbytes: int, context: str = ""
    ) -> tuple[str, bytes]:
        cmd = cmd_read(addr, nbytes)
        prompt = context or cmd
        reply = self._ask_robust(prompt, nbytes)
        data = self._decode_reply(reply, nbytes)
        if data is None or len(data) != nbytes:
            # 不合规的回复 = 一次故障。返回空数据，由沙箱记 fault。
            return reply, b""
        return reply, data

    def write(
        self, addr: int, data: bytes, context: str = ""
    ) -> tuple[str, bytes]:
        cmd = cmd_write(addr, data)
        prompt = context or cmd
        reply = self._ask_robust(prompt, len(data))
        body = reply
        if body.upper().startswith("OK"):
            body = body[2:].strip()
        payload = self._decode_reply(body, len(data))
        # 解不出内容就返回空，由沙箱记一笔故障——绝不替它补零。
        return reply, (payload if payload is not None else b"")

    def erase(self, page: int, context: str = "") -> str:
        return self._ask_robust(context or cmd_erase(page), 1)


# ===================================================================== 真人


class HumanMemory:
    """你本人当内存条。

    沙箱会把指令交给 prompt 回调（黑窗里显示出来），
    然后阻塞等你在输入框里敲十六进制。
    """

    name = "human"

    def __init__(self, ask: Callable[[str], str]) -> None:
        self.ask = ask

    def read(self, addr: int, nbytes: int, context: str = "") -> tuple[str, bytes]:
        reply = self.ask(cmd_read(addr, nbytes))
        data = decode_hex(reply)
        if data is None or len(data) != nbytes:
            return reply, b""
        return reply, data

    def write(
        self, addr: int, data: bytes, context: str = ""
    ) -> tuple[str, bytes]:
        """你回了什么，就是什么——不合规也不替它补位。

        位数不对时返回你给的内容（能解出多少算多少），由沙箱记一笔故障。
        只有完全解不出内容时才返回空，那种情况下确实没有东西可写。
        """
        reply = self.ask(cmd_write(addr, data))
        body = reply[2:].strip() if reply.upper().startswith("OK") else reply
        payload = decode_hex(body)
        return reply, (payload if payload is not None else b"")

    def erase(self, page: int, context: str = "") -> str:
        return self.ask(cmd_erase(page))


# ===================================================================== 工厂


def build_llm_memory(settings: LLMSettings, dialect: str = "openai") -> LLMStrictMemory:
    if dialect == "openai":
        if not settings.base_url.strip():
            raise BackendError("请填写 Base URL")
        if not settings.api_key.strip():
            raise BackendError("请填写 API Key")
    elif dialect == "anthropic":
        if not settings.base_url.strip():
            settings.base_url = "https://api.anthropic.com/v1"
        if not settings.api_key.strip():
            raise BackendError("请填写 API Key")
    else:
        raise BackendError(f"未知 dialect：{dialect}")
    return LLMStrictMemory(settings=settings, dialect=dialect)


def from_env() -> LLMSettings:
    """环境变量兜底。

    注意：窗口里填的值来自 `.env`（见 settings.load_saved），优先级更高。
    这里只负责"连环境变量都没有"时的最后一道兜底。
    """
    return LLMSettings(
        base_url=os.environ.get("DDR_AI_BASE_URL", "https://api.openai.com/v1"),
        api_key=(
            os.environ.get("DDR_AI_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        ),
        model=os.environ.get("DDR_AI_MODEL", "gpt-4o-mini"),
    )
