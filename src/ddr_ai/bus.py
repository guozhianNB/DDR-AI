"""总线与延迟记账。

这个模块干的事很单纯：**把每一次内存访问翻译成物理时间**。

它定义了整套 demo 的世界观：

    用户 ↔ 沙箱 ↔ 总线 ↔ AI ↔ 纸介质

其中"快"和"慢"全部由 LatencyProfile 决定。默认档位基于一个
网上大佬实测的 0.8 bit/s 手写速度，往上按不同读写机构递增。

注意：仿真时间是"这段代码在真实硬件上应该花多久"，与实际执行
这个 Python 脚本用了多久无关。两者都会被记录，摆在一起看就是笑话。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .paper_mem import human_time


class AccessKind(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    ERASE = "ERASE"
    FETCH = "FETCH"


class LatencyProfile(str, Enum):
    """不同的"读写机构"档位。数值单位：秒/单位操作。"""

    HUMAN_PEN = "human_pen"
    """原始方案：人拿笔在纸上写。已按"打印机能跟上"的量级标定。"""

    LITERAL_0_8 = "literal_0_8"
    """严格按大佬实测的 0.8 bit/s。选了这个，你就要有等下去的觉悟。"""

    PAPER_ROBOT = "paper_robot"
    """纸 + 机械臂：比人手快，但每次改字节仍要整页重写。"""

    AI_API = "ai_api"
    """用 AI 直接记忆内容：一次往返 = 一次推理请求。"""

    AI_CONTEXT = "ai_context"
    """用 AI + 超长上下文记忆：写一次 = 把整页塞进上下文窗口。"""

    def describe(self) -> str:
        return _PROFILE_DOC[self]


_PROFILE_DOC: dict[LatencyProfile, str] = {
    LatencyProfile.HUMAN_PEN: "人手 + 纸（誊写 ~2 b/s 量级）",
    LatencyProfile.LITERAL_0_8: "人手 + 纸（严格 0.8 b/s）",
    LatencyProfile.PAPER_ROBOT: "机械臂 + 纸（快 100 倍，但整页重写）",
    LatencyProfile.AI_API: "AI 记忆（一次推理往返 ≈ 50 ms）",
    LatencyProfile.AI_CONTEXT: "AI 长上下文记忆（整页入窗 ≈ 1.2 s）",
}


@dataclass
class Timing:
    """一种机构的四项基本开销（秒）。"""

    per_bit: float = 0.0
    """每个比特的搬运时间（顺序写）。"""

    per_access_fixed: float = 0.0
    """一次访问的固定开销（寻址、握手、开机）。"""

    per_page_rewrite: float = 0.0
    """改 1 字节时，整页重写的额外代价。"""

    setup: float = 0.0
    """上电自检：把整块纸搬进屋。"""


PROFILES: dict[LatencyProfile, Timing] = {
    # 誊写加寻址，单次访问约 2.2 s。比手写快，但仍属于"人在弄纸"。
    LatencyProfile.HUMAN_PEN: Timing(
        per_bit=0.05,
        per_access_fixed=1.2,
        per_page_rewrite=1.8,
        setup=1800.0,
    ),
    # 严格照搬大佬实测：0.8 bit/s。一次 8 bit 访问 = 10 s。
    LatencyProfile.LITERAL_0_8: Timing(
        per_bit=1.25,
        per_access_fixed=0.0,
        per_page_rewrite=180.0,
        setup=3600.0,
    ),
    # 机械臂快 100 倍：单次访问约 25 ms
    LatencyProfile.PAPER_ROBOT: Timing(
        per_bit=0.0005,
        per_access_fixed=0.012,
        per_page_rewrite=0.45,
        setup=120.0,
    ),
    # AI 往返 ≈ 50 ms/次
    LatencyProfile.AI_API: Timing(
        per_bit=0.0,
        per_access_fixed=0.05,
        per_page_rewrite=0.02,
        setup=2.0,
    ),
    # 整页塞进长上下文：一次写 ≈ 1.2 s
    LatencyProfile.AI_CONTEXT: Timing(
        per_bit=0.0,
        per_access_fixed=0.08,
        per_page_rewrite=1.2,
        setup=6.0,
    ),
}

# 参考基准：一条普通 DDR5 的实际数据（用于最后羞辱一下自己）
DRAM_REFERENCE = {
    "profile": "DDR5-5600 单通道",
    "read_latency_s": 12e-9,
    "bandwidth_bits_per_s": 44.8e9,
}


class VirtualClock:
    """仿真时钟。只累加，不睡眠（睡眠是 console 的事）。"""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def tick(self, seconds: float) -> None:
        self.elapsed += max(0.0, seconds)

    def advance(self, seconds: float) -> float:
        self.tick(seconds)
        return self.elapsed

    def format(self) -> str:
        return human_time(self.elapsed)


@dataclass
class BusLedger:
    """总线记账：每一笔内存流量的明细。"""

    clock: VirtualClock = field(default_factory=VirtualClock)
    accesses: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    erases: int = 0
    bits_moved: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def record(self, kind: AccessKind, nbytes: int, cost: float) -> None:
        self.clock.tick(cost)
        self.accesses += 1
        self.bits_moved += nbytes * 8
        if kind is AccessKind.READ or kind is AccessKind.FETCH:
            self.read_bytes += nbytes
        elif kind is AccessKind.WRITE:
            self.write_bytes += nbytes
        elif kind is AccessKind.ERASE:
            self.erases += 1
        self.by_kind[kind.value] = self.by_kind.get(kind.value, 0) + 1

    @property
    def effective_bps(self) -> float:
        """等效读写速率（bit/s）——这个数字会非常难看。"""
        if self.clock.elapsed <= 0:
            return float("inf")
        return self.bits_moved / self.clock.elapsed


class PaperBus:
    """总线：所有内存访问都必须经过这里，并被明码标价。"""

    def __init__(
        self,
        timing: Timing,
        profile: LatencyProfile = LatencyProfile.HUMAN_PEN,
    ) -> None:
        self.timing = timing
        self.profile = profile
        self.ledger = BusLedger()

    # ------------------------------------------------------------ 计费规则

    def cost_of(self, kind: AccessKind, nbytes: int, page_rewrite: bool) -> float:
        bits = nbytes * 8
        cost = self.timing.per_access_fixed
        cost += bits * self.timing.per_bit
        if kind is AccessKind.WRITE and page_rewrite:
            # 改一个字节 = 重誊一整页。纸介质写放大的来源。
            cost += self.timing.per_page_rewrite
        if kind is AccessKind.ERASE:
            cost += self.timing.per_page_rewrite
        return cost

    def charge(self, kind: AccessKind, nbytes: int, page_rewrite: bool = False) -> float:
        cost = self.cost_of(kind, nbytes, page_rewrite)
        self.ledger.record(kind, nbytes, cost)
        return cost

    def power_on(self) -> float:
        return self.ledger.clock.advance(self.timing.setup)

    # ------------------------------------------------------------ 报告

    def summary_lines(self) -> list[str]:
        led = self.ledger
        ref = DRAM_REFERENCE
        ratio = (
            ref["bandwidth_bits_per_s"] / led.effective_bps
            if led.effective_bps > 0
            else float("inf")
        )
        return [
            f"总线访问次数   : {led.accesses:,}",
            f"其中 {led.by_kind if led.by_kind else '{}'}",
            f"搬运比特数     : {led.bits_moved:,} bit",
            f"仿真耗时       : {led.clock.format()}",
            f"等效读写速率   : {led.effective_bps:.4f} bit/s",
            f"{ref['profile']:<14} : {ref['bandwidth_bits_per_s'] / 1e9:.1f} Gbit/s",
            f"差距           : {ratio:,.0f} 倍",
        ]
