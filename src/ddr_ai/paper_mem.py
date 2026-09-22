"""PaperMemory —— 用"纸 + AI"实现的内存条。

本模块只负责介质本身：一个按字节寻址、按页擦写的存储体。
所有"慢"的部分由 bus.py 记账，本模块只做两件事：

1. 忠实模拟内存语义（地址、字节、页边界、容量越界）；
2. 记录指标（读次数、写次数、擦除次数、累计搬运的比特数）。

设计原则：介质必须是真的能存对数据。这样"它慢得离谱但仍然正确"
这件事才成立；至于 AI 什么时候把它写坏，那是 ai_backend 的自由。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class MemoryError_(Exception):
    """地址越界等内存级错误。"""


@dataclass
class MemStats:
    """存储体累计指标。"""

    reads: int = 0
    writes: int = 0
    erases: int = 0
    bits_moved: int = 0
    bytes_written: int = 0
    faults: int = 0
    """内存条写坏了次数：它声称写成功，实际落纸与指令不符。"""

    read_mismatches: int = 0
    """内存条读数与纸上不符的次数（读路径故障，与 faults 分开计）。"""

    dirty_pages: set[int] = field(default_factory=set)

    def snapshot(self) -> dict[str, int]:
        return {
            "reads": self.reads,
            "writes": self.writes,
            "erases": self.erases,
            "bits_moved": self.bits_moved,
            "bytes_written": self.bytes_written,
            "faults": self.faults,
            "read_mismatches": self.read_mismatches,
            "dirty_pages": len(self.dirty_pages),
        }


class PaperMemory:
    """一块"纸介质"内存。

    与真实 DRAM 的差别只有两点，但都是致命的：
    - 写入不是"电荷翻转"，而是"把这一页重新誊写一遍"，所以写一个字节
      的代价等于写整页；
    - 擦除是显式操作（橡皮），不是免费的。
    """

    def __init__(self, size_bytes: int, page_size: int = 256) -> None:
        if size_bytes <= 0:
            raise ValueError("内存大小必须为正数")
        if page_size <= 0 or page_size > size_bytes:
            raise ValueError("页大小必须为正且不超过总容量")
        self.size_bytes = size_bytes
        self.page_size = page_size
        self._cells = bytearray(size_bytes)
        self.stats = MemStats()

    # ---------------------------------------------------------------- 地址

    @property
    def pages(self) -> int:
        return (self.size_bytes + self.page_size - 1) // self.page_size

    def page_of(self, addr: int) -> int:
        return addr // self.page_size

    def page_range(self, page: int) -> tuple[int, int]:
        start = page * self.page_size
        return start, min(start + self.page_size, self.size_bytes)

    def _check(self, addr: int, length: int = 1) -> None:
        if addr < 0 or length < 0 or addr + length > self.size_bytes:
            self.stats.faults += 1
            raise MemoryError_(
                f"段错误: 0x{addr:04x}+{length} 越界 (容量 {self.size_bytes} 字节)"
            )

    # ---------------------------------------------------------------- 读写

    def peek(self, addr: int, length: int = 1) -> bytes:
        """不记账、不产生总线流量的直读。调试/自检用。"""
        self._check(addr, length)
        return bytes(self._cells[addr : addr + length])

    def raw_read(self, addr: int, length: int = 1) -> bytes:
        """介质层读：只负责把纸上的字节抄下来。"""
        self._check(addr, length)
        data = bytes(self._cells[addr : addr + length])
        self.stats.reads += 1
        self.stats.bits_moved += length * 8
        return data

    def raw_write(self, addr: int, data: bytes) -> None:
        """介质层写：真实的落纸动作。"""
        self._check(addr, len(data))
        self._cells[addr : addr + len(data)] = data
        self.stats.writes += 1
        self.stats.bytes_written += len(data)
        self.stats.bits_moved += len(data) * 8
        self.stats.dirty_pages.add(self.page_of(addr))

    def erase_page(self, page: int) -> None:
        """橡皮擦：把一页恢复成空白。"""
        if not 0 <= page < self.pages:
            raise MemoryError_(f"擦除越界: 页 {page}")
        start, end = self.page_range(page)
        self._cells[start:end] = b"\x00" * (end - start)
        self.stats.erases += 1
        self.stats.dirty_pages.discard(page)

    def rewrite_page(self, page: int, offset: int, data: bytes) -> None:
        """誊写一页：读出现有内容、打补丁、整页写回。

        这就是纸介质写放大的来源——改 1 个字节要动一整页。
        """
        start, end = self.page_range(page)
        page_bytes = bytearray(self._cells[start:end])
        lo = offset - start
        page_bytes[lo : lo + len(data)] = data
        self._cells[start:end] = page_bytes
        self.stats.writes += 1
        self.stats.bytes_written += end - start
        self.stats.bits_moved += (end - start) * 8
        self.stats.dirty_pages.add(page)

    # ---------------------------------------------------------------- 视图

    def dump(self, start: int = 0, length: int = 64) -> str:
        """十六进制转储，用于"肉眼检查这张纸上到底写了什么"。"""
        length = min(length, self.size_bytes - start)
        lines: list[str] = []
        for row in range(start, start + length, 16):
            chunk = bytes(self._cells[row : min(row + 16, start + length)])
            hexs = " ".join(f"{b:02x}" for b in chunk)
            text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"{row:08x}  {hexs:<47}  |{text}|")
        return "\n".join(lines)


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}" if unit != "B" else f"{n:.0f} B"
        n /= 1024
    return f"{n:.2f} PiB"


def human_time(seconds: float) -> str:
    """把仿真时间翻译成人能理解的尺度——这个函数是整活的精华。"""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} 分 {sec:.1f} 秒"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{int(hours)} 小时 {int(minutes)} 分"
    days, hours = divmod(hours, 24)
    if days < 365.25:
        return f"{int(days)} 天 {int(hours)} 小时"
    years = days / 365.25
    return f"{years:,.0f} 年"


class Stopwatch:
    """真实耗时统计（不参与仿真时间）。"""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.elapsed = 0.0

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self._t0
        return self.elapsed
