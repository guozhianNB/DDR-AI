"""客户程序：一个极小的 CPU 与它的汇编。

沙箱里跑的不是 Python、不是 C，而是一套**自己造的指令集**。
理由是：只有自造 CPU，才能把"每一次内存访问"都精确地摆到台面上，
让总线和 AI 有机会插进去、让黑窗有机会把每次访问都打出来。

指令集（17 条，够跑循环、分支、数组、排序和 IO）：

    MOVI r, imm          立即数送寄存器
    LOAD r, addr         从内存读一个字节
    LDI  r, ra           **间接寻址**：r = mem[ra]（数组遍历靠它）
    STORE addr, r        把寄存器写回内存一个字节
    STI  ra, r           间接写：mem[ra] = r
    SWAP a1, a2          交换两个内存地址的内容
    ADDI r, imm          r += imm
    ADD  rd, rs          rd += rs（寄存器相加）
    SUBI r, imm          r -= imm
    SUB  rd, rs          rd -= rs（寄存器相减，两个寄存器比较的前提）
    JMP  label           无条件跳转
    JZ   r, label        r == 0 则跳转
    JNZ  r, label        r != 0 则跳转
    JGT  r, imm, label   r > imm 则跳转
    JLT  r, imm, label   r < imm 则跳转
    OUT  r               把寄存器当 ASCII 打到屏幕（不占内存）
    HALT [code]          停机

寄存器有 r0 / r1 / r2 / r3 四个。故意不给多：寄存器越少，
程序越被迫频繁访问内存，黑窗就越热闹。排序程序刚好把四个用完。

字长与语义：所有寄存器与内存单元都是 **8 位无符号**（0..255），
加减法按 256 取模，没有符号位。`JGT`/`JLT` 和立即数比较时用的是
无符号值，所以"借位"要靠 0..255 的落点自己判断（见 sort.asm）。

程序通过 `load(addr)` / `store(addr, val)` 回调与外界交互，
这两个回调就是沙箱接总线的地方。CPU 本身对"内存有多慢"一无所知——
它甚至不知道自己插的是一张纸。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol


class GuestError(Exception):
    """客户程序自身的问题（语法、越界、除零之类）。"""


class HaltSignal(Exception):
    def __init__(self, code: int = 0) -> None:
        super().__init__(f"halt({code})")
        self.code = code


@dataclass
class Instruction:
    op: str
    args: tuple = ()
    line: int = 0

    def __str__(self) -> str:
        if not self.args:
            return self.op
        return f"{self.op} {', '.join(str(a) for a in self.args)}"


@dataclass
class GuestProgram:
    """一份可运行的客户程序。"""

    name: str
    source: str
    code: list[Instruction] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)
    data: dict[int, int] = field(default_factory=dict)
    """预置到内存里的字节（数据段）。"""

    @property
    def mem_footprint(self) -> int:
        if not self.data:
            return 16
        return max(self.data) + 1

    @classmethod
    def from_source(cls, name: str, source: str) -> "GuestProgram":
        prog = cls(name=name, source=source)
        prog._assemble()
        return prog

    @classmethod
    def from_file(cls, path: str | Path) -> "GuestProgram":
        p = Path(path)
        return cls.from_source(p.stem, p.read_text(encoding="utf-8"))

    def _assemble(self) -> None:
        for lineno, raw in enumerate(self.source.splitlines(), start=1):
            line = raw.split(";", 1)[0].split("#", 1)[0].strip()
            if not line:
                continue

            # 数据段：.data <addr> <v1> <v2> ...
            if line.startswith(".data"):
                parts = line.split()
                addr = int(parts[1], 0)
                for offset, tok in enumerate(parts[2:]):
                    self.data[addr + offset] = int(tok, 0) & 0xFF
                continue

            # 标签
            while ":" in line:
                label, _, line = line.partition(":")
                label = label.strip()
                if not label:
                    break
                self.labels[label] = len(self.code)
                line = line.strip()

            if not line:
                continue

            op, _, rest = line.partition(" ")
            op = op.upper()
            args: list[object] = []
            for tok in (a.strip() for a in rest.split(",")):
                if not tok:
                    continue
                args.append(tok)
            self.code.append(Instruction(op, tuple(args), lineno))


class MemoryPort(Protocol):
    """CPU 看内存的窗口。沙箱来实现它。"""

    def load(self, addr: int) -> int: ...
    def store(self, addr: int, value: int) -> None: ...


@dataclass
class CPU:
    """极简 CPU。四个寄存器，够用。"""

    memory: MemoryPort
    program: GuestProgram
    pc: int = 0
    regs: dict[str, int] = field(
        default_factory=lambda: {"r0": 0, "r1": 0, "r2": 0, "r3": 0}
    )
    steps: int = 0
    output: bytearray = field(default_factory=bytearray)
    max_steps: int = 100_000
    on_out: Callable[[int], None] | None = None
    """每产出一个字节就回调一次。沙箱靠它在 OUT 执行的当下刷新黑窗。

    注意 on_out 只在"内存不够快"之外还有意义：它是**交互性**的来源。
    """

    def __post_init__(self) -> None:
        for addr, value in self.program.data.items():
            self.memory.store(addr, value)

    # ------------------------------------------------------------------ 执行

    def run(self, on_step: Callable[[int, Instruction], None] | None = None) -> bytearray:
        try:
            for ins in self:
                if on_step is not None:
                    on_step(self.steps, ins)
        except HaltSignal:
            pass
        return self.output

    def __iter__(self) -> Iterator[Instruction]:
        while self.pc < len(self.program.code):
            if self.steps >= self.max_steps:
                raise GuestError(f"步数超过上限 {self.max_steps}，疑似死循环")
            ins = self.program.code[self.pc]
            self.pc += 1
            self.steps += 1
            self._exec(ins)
            yield ins

    def _resolve(self, label: str) -> int:
        if label in self.program.labels:
            return self.program.labels[label]
        if label.lstrip("-").isdigit():
            return int(label)
        raise GuestError(f"第 {self.pc} 行: 未知标签 {label}")

    def _reg(self, name: str) -> str:
        key = name.lower()
        if key not in self.regs:
            raise GuestError(f"未知寄存器 {name}（只有 r0/r1/r2/r3）")
        return key

    def _emit(self, value: int) -> None:
        """把一个字节送进程序输出。

        除了累积到 output（供最终指标使用），还会立刻通知 on_out——
        这样黑窗能在 OUT 执行的**当下**就显示出来，而不是等跑完一次性倒出来。
        """
        self.output.append(value & 0xFF)
        if self.on_out is not None:
            self.on_out(value & 0xFF)

    def _exec(self, ins: Instruction) -> None:
        op, a = ins.op, ins.args
        try:
            if op == "MOVI":
                self.regs[self._reg(str(a[0]))] = int(str(a[1]), 0) & 0xFF
            elif op == "LOAD":
                dest = self._reg(str(a[0]))
                self.regs[dest] = self.memory.load(int(str(a[1]), 0)) & 0xFF
            elif op == "LDI":
                dest = self._reg(str(a[0]))
                addr_reg = self._reg(str(a[1]))
                self.regs[dest] = self.memory.load(self.regs[addr_reg]) & 0xFF
            elif op == "STORE":
                src = self._reg(str(a[1]))
                self.memory.store(int(str(a[0]), 0), self.regs[src])
            elif op == "STI":
                addr_reg = self._reg(str(a[0]))
                src = self._reg(str(a[1]))
                self.memory.store(self.regs[addr_reg], self.regs[src])
            elif op == "SWAP":
                x, y = int(str(a[0]), 0), int(str(a[1]), 0)
                vx, vy = self.memory.load(x), self.memory.load(y)
                self.memory.store(x, vy)
                self.memory.store(y, vx)
            elif op == "ADDI":
                reg = self._reg(str(a[0]))
                self.regs[reg] = (self.regs[reg] + int(str(a[1]), 0)) & 0xFF
            elif op == "ADD":
                dst = self._reg(str(a[0]))
                src = self._reg(str(a[1]))
                self.regs[dst] = (self.regs[dst] + self.regs[src]) & 0xFF
            elif op == "SUBI":
                reg = self._reg(str(a[0]))
                self.regs[reg] = (self.regs[reg] - int(str(a[1]), 0)) & 0xFF
            elif op == "SUB":
                dst = self._reg(str(a[0]))
                src = self._reg(str(a[1]))
                self.regs[dst] = (self.regs[dst] - self.regs[src]) & 0xFF
            elif op == "JMP":
                self.pc = self._resolve(str(a[0]))
            elif op == "JZ":
                if self.regs[self._reg(str(a[0]))] == 0:
                    self.pc = self._resolve(str(a[1]))
            elif op == "JNZ":
                if self.regs[self._reg(str(a[0]))] != 0:
                    self.pc = self._resolve(str(a[1]))
            elif op == "JGT":
                if self.regs[self._reg(str(a[0]))] > int(str(a[1]), 0):
                    self.pc = self._resolve(str(a[2]))
            elif op == "JLT":
                if self.regs[self._reg(str(a[0]))] < int(str(a[1]), 0):
                    self.pc = self._resolve(str(a[2]))
            elif op == "OUT":
                self._emit(self.regs[self._reg(str(a[0]))] & 0xFF)
            elif op == "HALT":
                raise HaltSignal(int(str(a[0]), 0) if a else 0)
            else:
                raise GuestError(f"未知指令 {op}")
        except IndexError as exc:
            raise GuestError(f"指令参数不足: {ins}") from exc
        except ValueError as exc:
            raise GuestError(f"指令参数非法: {ins}") from exc


# ---------------------------------------------------------------- 示例程序

PROGRAM_DIR = Path(__file__).parent / "programs"


def available_programs() -> list[GuestProgram]:
    if not PROGRAM_DIR.is_dir():
        return []
    return [GuestProgram.from_file(p) for p in sorted(PROGRAM_DIR.glob("*.asm"))]


def load_program(name: str) -> GuestProgram:
    path = PROGRAM_DIR / f"{name}.asm"
    if not path.exists():
        raise GuestError(f"没有这个程序: {name}")
    return GuestProgram.from_file(path)


def search_paths(directory: str | Path) -> list[GuestProgram]:
    """从任意目录加载 .asm（支持用户在启动器里选外部程序）。"""
    d = Path(directory)
    if not d.is_dir():
        return []
    return [GuestProgram.from_file(p) for p in sorted(d.glob("*.asm"))]
