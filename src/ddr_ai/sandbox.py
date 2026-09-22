"""沙箱：把客户程序关起来，只给它一条通往"纸 + AI"的内存通路。

数据流是这样的：

    客户程序(guest.CPU)
        │  load(addr) / store(addr, val)
        ▼
    沙箱(Sandbox)  ←── 你在这里看到黑窗里的每一行
        │  发一条严格指令，如 READ 0x0010 1 / WRITE 0x0010 1 2a
        ▼
    总线(PaperBus) ── 记账：这一趟该花多久
        │
        ▼
    内存条(ai_backend) ── 它回的数据会被无条件当真
        │
        ▼
    纸(PaperMemory) ── 真的把字节写进去

沙箱只做三件事：转发、记账、把过程打印出来。
它**不纠正**内存条：以回复为准，回复错了就让它错下去、并如实记为一次故障。
报错只是通知你出事了，绝不替它补数据、不重试、不用纸上的内容覆盖。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import ai_backend as ab
from .bus import PROFILES, AccessKind, LatencyProfile, PaperBus, Timing
from .guest import CPU, GuestError, GuestProgram
from .paper_mem import MemoryError_, PaperMemory, Stopwatch, human_bytes, human_time


# --------------------------------------------------------------------- 事件


@dataclass
class SandboxEvent:
    """沙箱往黑窗里吐的一行。"""

    kind: str
    """bus / ai / sys / io / err / fyi / ask"""

    text: str


@dataclass
class RunResult:
    ok: bool
    output: bytes
    steps: int
    program: str
    message: str
    sim_seconds: float = 0.0
    real_seconds: float = 0.0
    accesses: int = 0
    faults: int = 0
    """故障总数 = 写侧（写坏/写数不合规）+ 读侧（读不符/读数不合规）。
    与 Sandbox.hallucinations 同口径。"""

    read_mismatches: int = 0
    """其中读侧那一部分，方便单独看"模型记不记得住"。"""

    effective_bps: float = 0.0
    memory: PaperMemory | None = None


EventSink = Callable[[SandboxEvent], None]
AskHuman = Callable[[str], str]
"""真人模式：把指令显示出来，阻塞等待用户敲十六进制。"""


# --------------------------------------------------------------------- 配置


@dataclass
class SandboxConfig:
    # ---- 内存 ----
    memory_bytes: int = 4096
    page_size: int = 256

    # ---- 介质 / 机构 ----
    profile: LatencyProfile = LatencyProfile.HUMAN_PEN
    timing: Timing | None = None

    # ---- 谁来当内存条 ----
    backend_spec: str = "local"
    """local / openai-compat / anthropic / human"""

    llm: ab.LLMSettings = field(default_factory=ab.LLMSettings)
    hallucination_rate: float = 0.0
    """仅本地仿 AI 使用。"""

    context_mode: str = "state"
    """给真模型多少上下文：state（给内存内容，默认）/ recent / none。

    给 none 就是让一个无状态模型盲猜内存内容——READ 必然全错。
    state 才是能真正工作起来的配置。
    """

    context_max_bytes: int = 2048
    """上下文里最多渲染多少字节的内存，防止大容量把请求撑爆。"""

    # ---- 节流 ----
    pacing: float = 0.06
    """把"仿真耗时"折算成黑窗里真实等待的系数。0 = 全速不等待。"""

    pace_cap: float = 0.35
    """单次访问真实等待的上限（秒）。"""

    # ---- 杂项 ----
    output_style: str = "ascii"
    """ascii / dec / hex"""


# --------------------------------------------------------------------- 沙箱


class Sandbox:
    """一次运行 = 一个沙箱实例。不重用，不共享状态。"""

    def __init__(
        self,
        config: SandboxConfig,
        emit: EventSink | None = None,
        ask_human: AskHuman | None = None,
    ) -> None:
        self.config = config
        self.emit = emit or (lambda ev: None)
        self.ask_human = ask_human
        self.memory = PaperMemory(config.memory_bytes, config.page_size)
        self._addr_width = ab._fit_addr_width(config.memory_bytes)
        timing = config.timing or PROFILES[config.profile]
        self.bus = PaperBus(timing, config.profile)
        self.mem_ops = 0
        self.hallucinations = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._escalation_seen = 0
        """已经播报过的升档序号。升档只提示一次，免得刷屏。"""
        self._history: list[str] = []
        """最近若干次收发记录，供 recent 上下文模式使用。"""
        self._thinking_tokens = 0
        """累计思考 token 数。"""
        self._thinking_calls = 0
        """有多少次请求产生了思考 token。"""
        self._thinking_last = 0
        self._out_buf = bytearray()
        """程序输出的暂存。攒到换行或一批够多才刷一次，避免逐字节刷屏。"""
        self.backend = self._build_backend()

    # ------------------------------------------------------------ 后端装配

    def _build_backend(self):
        spec = self.config.backend_spec
        if spec == "local":
            return ab.LocalMimicMemory(
                hallucination_rate=self.config.hallucination_rate,
                peek=lambda addr, n: self.memory.peek(addr, n),
            )
        if spec in ("openai-compat", "openai"):
            return ab.build_llm_memory(self.config.llm, "openai")
        if spec == "anthropic":
            return ab.build_llm_memory(self.config.llm, "anthropic")
        if spec == "human":
            if self.ask_human is None:
                raise ab.BackendError(
                    "选择了「能工智人 · 真实」模式，但没有接入输入通道"
                )
            return ab.HumanMemory(self.ask_human)
        raise ab.BackendError(f"未知后端：{spec}")

    # ------------------------------------------------------------ 输出

    def _say(self, kind: str, text: str) -> None:
        self.emit(SandboxEvent(kind, text))

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def _sleep(self, sim_seconds: float) -> None:
        """把仿真时间折算成真实等待，让你看得见。"""
        if self.config.pacing <= 0 or self.stopped:
            return
        real = min(sim_seconds * self.config.pacing, self.config.pace_cap)
        if real > 0.001:
            time.sleep(real)

    def _backend_diag(self, empty: bool = False) -> str:
        """把内存条"为什么答成这样"翻译成一句人话，附在报错后面。

        空回复尤其需要解释：多半是思考型模型把 max_tokens 全花在
        思考过程上，导致正式回答一个字都没有。
        """
        b = self.backend
        if not hasattr(b, "last_finish_reason"):
            return ""

        bits: list[str] = []
        if b.last_finish_reason:
            bits.append(f"finish_reason={b.last_finish_reason}")
        if b.last_budget:
            bits.append(f"max_tokens={b.last_budget}")
        if b.last_used_reasoning:
            bits.append("内容取自 reasoning_content")
        if getattr(b, "last_thinking_tokens", 0):
            bits.append(f"思考烧了 {b.last_thinking_tokens} token")
        if b.last_escalation_note:
            bits.append(b.last_escalation_note)

        if not bits:
            return ""
        diag = "；".join(bits)
        if getattr(b, "last_reasoning_leak", False):
            diag += "。它把思考过程当正文吐出来了，建议关掉思考或调小 max_tokens"
        if empty and b.last_finish_reason == "length":
            diag += "。多半是思考型模型把预算吃完了，把 max_tokens 调大试试"
        return f"（{diag}）"

    def _note_thinking(self) -> None:
        """累计思考 token 消耗。

        这是"关掉思考却依然力不从心"的直接证据：
        模型在暗处把预算烧掉了，你看不见，但 max_tokens 确实被吃掉。
        """
        b = self.backend
        n = getattr(b, "last_thinking_tokens", 0) or 0
        if n:
            self._thinking_tokens += n
            self._thinking_calls += 1
            self._thinking_last = n

    def _note_escalation(self) -> None:
        """升档救回时给一条可见提示。

        静默恢复虽然不报错，但用户有权知道"这条回复是重试两次才拿到的"——
        它意味着延迟翻倍，也意味着 max_tokens 该调大了。

        只报首次：靠 escalation_seq 判断这次操作是否刚刚升过档，
        免得把一次升档的说明重复贴到后面没升档的请求上。
        """
        b = self.backend
        seq = getattr(b, "escalation_seq", 0)
        if seq <= self._escalation_seen:
            return
        self._escalation_seen = seq
        note = getattr(b, "last_escalation_note", "")
        if note:
            self._say("fyi", f"[提示] {note}。建议把 max_tokens 调大。")

    # ------------------------------------------------------------ 上下文

    def _build_context(self, command: str, focus: int) -> str:
        """给真模型组装上下文。

        本地仿 AI 与真人不需要——前者直接从介质读，后者是你在看屏幕。
        返回空串时后端会退回发裸指令（即最初那种"盲猜"行为）。
        """
        if self.config.backend_spec not in ("openai-compat", "anthropic"):
            return ""
        return ab.build_context(
            mode=self.config.context_mode,
            command=command,
            peek=self.memory.peek,
            size_bytes=self.config.memory_bytes,
            focus=focus,
            page_size=self.config.page_size,
            max_bytes=self.config.context_max_bytes,
            history=self._history,
        )

    def _remember(self, command: str, reply: str) -> None:
        self._history.append(f">> {command}\n<< {reply}")
        if len(self._history) > 16:
            del self._history[:-16]

    # ------------------------------------------------------------ 内存通路

    def load(self, addr: int) -> int:
        """CPU 的一次读。走总线 → 内存条。

        **以内存条的回复为准，不做任何纠正。** 它说读到什么，客户程序就拿到什么。
        纸上的内容只用来判断它有没有说错，绝不回填、不覆盖、不重试。
        """
        if self.stopped:
            raise GuestError("用户中止了运行")

        self.mem_ops += 1
        cmd = ab.cmd_read(addr, 1, self._addr_width)
        self._say("bus", f">> {cmd}")

        context = self._build_context(cmd, addr)
        try:
            reply, reported = self.backend.read(addr, 1, context=context)
        except ab.BackendError as exc:
            # 它完全没给内容。这不是"以它为准"，这是无内容可依据——既定行为：按 0 读。
            self._say("err", f"[!!] 内存条无响应：{exc}；该次读按 0 返回。")
            self._say("ai", "<< (无回复)")
            self.hallucinations += 1
            self.memory.stats.read_mismatches += 1
            cost = self.bus.charge(AccessKind.READ, 1)
            self._sleep(cost)
            return 0

        self._remember(cmd, reply)
        self._say("ai", f"<< {reply}")
        self._note_escalation()
        self._note_thinking()

        if len(reported) != 1:
            # 它给了内容，但位数不对。这是它的错，照它给的用，只记一笔。
            self.hallucinations += 1
            self.memory.stats.read_mismatches += 1
            how = ab.encode_hex(reported) if reported else "(空)"
            self._say(
                "err",
                f"[!!] 读数不合规：请求 1 字节，它给了 {how}。"
                f"{self._backend_diag(empty=not reported)}",
            )
        else:
            truth = self.memory.peek(addr, 1)
            if reported != truth:
                self.hallucinations += 1
                self.memory.stats.read_mismatches += 1
                self._say(
                    "err",
                    f"[!!] 读错了：0x{addr:04x} 它说 "
                    f"{ab.encode_hex(reported)}，实际记的是 "
                    f"{ab.encode_hex(truth)}。以它说的为准。",
                )

        value = reported[0] if reported else 0

        # 记账：读回来的字节也算搬过的比特
        self.memory.stats.reads += 1
        self.memory.stats.bits_moved += len(reported) * 8

        cost = self.bus.charge(AccessKind.READ, 1)
        self._sleep(cost)
        return value

    def store(self, addr: int, value: int) -> None:
        """CPU 的一次写。内存条说记下了什么，纸上就是什么——**原样落纸**。

        它给的数据位数不对也不替它补。位数不对只是记一笔错，照写不误。
        """
        if self.stopped:
            raise GuestError("用户中止了运行")

        value &= 0xFF
        truth = bytes([value])
        self.mem_ops += 1
        cmd = ab.cmd_write(addr, truth, self._addr_width)
        self._say("bus", f">> {cmd}")

        context = self._build_context(cmd, addr)
        try:
            reply, payload = self.backend.write(addr, truth, context=context)
        except ab.BackendError as exc:
            # 它没说话。没说话就没有内容可写——这是故障，不是我们代它决定写什么。
            self._say("err", f"[!!] 内存条无响应：{exc}；该次写丢弃。")
            self.hallucinations += 1
            self.memory.stats.faults += 1
            self._say("ai", "<< (无回复)")
            cost = self.bus.charge(AccessKind.WRITE, 1, page_rewrite=True)
            self._sleep(cost)
            return

        self._remember(cmd, reply)
        self._say("ai", f"<< {reply}")
        self._note_escalation()
        self._note_thinking()

        if len(payload) != len(truth):
            self.hallucinations += 1
            self.memory.stats.faults += 1
            how = ab.encode_hex(payload) if payload else "(空)"
            self._say(
                "err",
                f"[!!] 写数不合规：指令给 1 字节，它回了 {how}。"
                f"{self._backend_diag(empty=not payload)}",
            )

        # 不管它合规与否，都按它给的落纸（多字节时截到本地址的 1 个字节）
        written = payload[:1] if payload else b""
        if written:
            self.memory.raw_write(addr, written)

        if written and written != truth:
            # 内存条把内容记错了：这正是"幻觉"在物理层的形态
            self.hallucinations += 1
            self.memory.stats.faults += 1
            self._say(
                "err",
                f"[!!] 写错了：0x{addr:04x} 它记成 "
                f"{ab.encode_hex(written)}，指令要求 {ab.encode_hex(truth)}。",
            )

        cost = self.bus.charge(AccessKind.WRITE, 1, page_rewrite=True)
        self._sleep(cost)

    def erase(self, page: int) -> None:
        if self.stopped:
            raise GuestError("用户中止了运行")

        cmd = ab.cmd_erase(page)
        self._say("bus", f">> {cmd}")
        self.memory.erase_page(page)
        cost = self.bus.charge(AccessKind.ERASE, self.memory.page_size)
        try:
            reply = self.backend.erase(
                page, context=self._build_context(cmd, page * self.memory.page_size)
            )
            self._remember(cmd, reply)
            self._say("ai", f"<< {reply}")
        except ab.BackendError as exc:
            self._say("err", f"[!!] 内存条无响应：{exc}")
        self._sleep(cost)

    # ------------------------------------------------------------ 程序 IO

    def _on_program_out(self, value: int) -> None:
        """CPU 每执行一次 OUT 就调这里一次。

        这就是"交互"的来源：输出在产生的**当下**就进黑窗，
        而不是等程序跑完再一次性倒出来。
        """
        self._out_buf.append(value)
        # 换行立刻刷；否则攒够 16 字节再刷，避免逐字节刷屏
        if value == 0x0A or len(self._out_buf) >= 16:
            self._flush_out()

    def _flush_out(self) -> None:
        if not self._out_buf:
            return
        chunk = bytes(self._out_buf)
        self._out_buf.clear()
        self._say("io", f"   程序输出 ▸ {self._format_output(chunk)}")

    # ------------------------------------------------------------ 运行

    def run(self, program: GuestProgram) -> RunResult:
        cfg = self.config
        watch = Stopwatch()

        if cfg.backend_spec == "human":
            self._say(
                "sys",
                f"[提示] {('你' if cfg.pacing <= 0 else '仿真')}要亲自扮演内存条："
                "沙箱每发一条指令，都要有人按协议回一行十六进制。",
            )

        self._banner(program)

        power_on = self.bus.power_on()
        self._say("sys", f"[上电] 纸张装载完成，自检耗时 {human_time(power_on)}。")
        self._say(
            "sys",
            f"[配置] 容量 {human_bytes(cfg.memory_bytes)} / 页大小 {cfg.page_size} B "
            f"/ 机构 {cfg.profile.describe()}",
        )
        self._say("sys", f"[配置] 内存条 {getattr(self.backend, 'name', '?')}")
        if cfg.backend_spec == "local":
            self._say("sys", f"[配置] 幻觉率 {cfg.hallucination_rate:.0%}")
        self._say("sys", "-" * 62)

        cpu = CPU(memory=self, program=program, on_out=self._on_program_out)
        ok = True
        message = "正常停机"

        try:
            cpu.run()
        except MemoryError_ as exc:
            ok, message = False, f"内存错误：{exc}"
        except GuestError as exc:
            ok, message = False, f"客户程序错误：{exc}"
        except ab.BackendError as exc:
            ok, message = False, f"内存条失败：{exc}"
        except Exception as exc:  # noqa: BLE001  沙箱要兜住一切，别把 GUI 带崩
            ok, message = False, f"沙箱内部异常：{exc!r}"

        self._flush_out()  # 把还没刷出去的尾巴补上
        real = watch.stop()
        self._report(cpu, ok, message, real)
        return RunResult(
            ok=ok,
            output=bytes(cpu.output),
            steps=cpu.steps,
            program=program.name,
            message=message,
            sim_seconds=self.bus.ledger.clock.elapsed,
            real_seconds=real,
            accesses=self.bus.ledger.accesses,
            # 口径统一：写侧 + 读侧，与 hallucinations 一致
            faults=self.memory.stats.faults + self.memory.stats.read_mismatches,
            read_mismatches=self.memory.stats.read_mismatches,
            effective_bps=self.bus.ledger.effective_bps,
            memory=self.memory,
        )

    # ------------------------------------------------------------ 打印

    def _banner(self, program: GuestProgram) -> None:
        self._say("sys", "=" * 62)
        self._say("sys", "  DDR-AI · 纸介质内存沙箱 v0.2")
        self._say("sys", "=" * 62)
        self._say("sys", f"[装载] {program.name}（{len(program.code)} 条指令）")
        if program.data:
            pretty = ", ".join(f"0x{a:04x}={v}" for a, v in sorted(program.data.items()))
            self._say("sys", f"[数据] {pretty}")

    def _format_output(self, output: bytes) -> str:
        style = self.config.output_style
        if not output:
            return "（无输出）"
        if style == "dec":
            return " ".join(str(b) for b in output)
        if style == "hex":
            return " ".join(f"{b:02x}" for b in output)
        return "".join(chr(b) if 32 <= b < 127 else f"\\x{b:02x}" for b in output)

    def _report(self, cpu: CPU, ok: bool, message: str, real: float) -> None:
        led = self.bus.ledger
        self._say("sys", "-" * 62)
        # 程序输出不在这里打——它已经在 OUT 执行的当下逐段刷进黑窗了。
        # 这一行只交代结果，免得同一份输出出现两遍。
        self._say("sys", f"[执行结果] {message}（{'成功' if ok else '失败'}）")
        if not cpu.output:
            self._say("sys", "  （本程序没有产生输出）")
        self._say("sys", "-" * 62)
        for line in self.bus.summary_lines():
            self._say("sys", f"  {line}")
        self._say("sys", f"  纸介质写入次数 : {self.memory.stats.writes:,}")
        self._say("sys", f"  纸介质擦除次数 : {self.memory.stats.erases:,}")
        self._say("sys", f"  写坏（幻觉）   : {self.memory.stats.faults:,}")
        self._say("sys", f"  读不符         : {self.memory.stats.read_mismatches:,}")
        if self._thinking_calls:
            avg = self._thinking_tokens / self._thinking_calls
            self._say(
                "sys",
                f"  思考 token     : {self._thinking_tokens:,}"
                f"（{self._thinking_calls} 次请求，平均 {avg:.0f}/次）",
            )
            if self._thinking_tokens > 0 and self.mem_ops:
                self._say(
                    "sys",
                    "  注意：思考是隐形开销——它吃 max_tokens 但不产生回复。"
                    "把它关掉或调小，能明显缓解「跑着跑着力不从心」。",
                )
        self._say("sys", f"  真实耗时       : {real:.2f} s")
        if led.clock.elapsed > 0 and real > 0:
            self._say(
                "sys",
                f"  （仿真世界跑了 {human_time(led.clock.elapsed)}，"
                f"现实只花了 {real:.1f} 秒——加速比 {led.clock.elapsed / real:,.0f}×）",
            )
        self._say("sys", "-" * 62)
        self._say("sys", "  内存转储：")
        for line in self.memory.dump(0, min(64, self.config.memory_bytes)).splitlines():
            self._say("sys", f"  {line}")
        self._say("sys", "=" * 62)


# ------------------------------------------------------------------ 线程封装

_ACTIVE: list[Sandbox] = []


def run_in_thread(
    config: SandboxConfig,
    program: GuestProgram,
    emit: EventSink,
    done: Callable[[RunResult], None],
    ask_human: AskHuman | None = None,
) -> Sandbox:
    """把一次运行丢进后台线程，好让 GUI 不卡。

    返回 Sandbox 句柄，供"中止"按钮调用 stop()。
    """
    holder: list[Sandbox] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            sandbox = Sandbox(config, emit, ask_human)
        except BaseException as exc:  # noqa: BLE001 装配失败也要回报，别静默死线程
            error.append(exc)
            emit(SandboxEvent("err", f"[!!] 沙箱装配失败：{exc}"))
            done(
                RunResult(
                    ok=False,
                    output=b"",
                    steps=0,
                    program=program.name,
                    message=f"沙箱装配失败：{exc}",
                )
            )
            return
        holder.append(sandbox)
        _ACTIVE.append(sandbox)
        try:
            result = sandbox.run(program)
        finally:
            if sandbox in _ACTIVE:
                _ACTIVE.remove(sandbox)
        done(result)

    th = threading.Thread(target=worker, name="ddr-ai-sandbox", daemon=True)
    th.start()

    deadline = time.time() + 1.0
    while not holder and not error and time.time() < deadline:
        time.sleep(0.005)
    if not holder:
        # 装配失败或线程还没起来。给一个占位句柄用于中止。
        return _placeholder(config, emit)
    return holder[0]


def _placeholder(config: SandboxConfig, emit: EventSink) -> Sandbox:
    """极端情况下的占位句柄：stop() 要能用，所以得是合法实例。"""
    box = Sandbox.__new__(Sandbox)
    box.config = config
    box.emit = emit
    box._stop = threading.Event()
    return box


def stop_all() -> None:
    for s in list(_ACTIVE):
        s.stop()
