"""语义契约测试：以回复为准，不自动纠错。

    python tests/test_semantics.py

沙箱的核心契约有三条，任何一条被破坏，"幻觉"这个 demo 就不成立了：

  1. 正常路径只发指令、只收回复，不附带任何落纸确认行；
  2. 读失败时以内存条的回复为准，**不用纸上的内容覆盖**；
  3. 报错只是通知，不是恢复机制——不重试、不补位。
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ddr_ai.ai_backend import BackendError  # noqa: E402
from ddr_ai.guest import load_program  # noqa: E402
from ddr_ai.sandbox import Sandbox, SandboxConfig  # noqa: E402

FAILS = 0


def check(label: str, got, want) -> None:
    global FAILS
    ok = got == want
    if not ok:
        FAILS += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {label}: got={got!r} want={want!r}")


def make_box(**kw) -> tuple[Sandbox, list[tuple[str, str]]]:
    ev: list[tuple[str, str]] = []
    box = Sandbox(
        SandboxConfig(
            memory_bytes=kw.pop("memory_bytes", 256),
            page_size=kw.pop("page_size", 64),
            backend_spec="local",
            pacing=0.0,
            **kw,
        ),
        emit=lambda e: ev.append((e.kind, e.text)),
    )
    return box, ev


class LyingReader:
    """读永远乱报，写照实做。"""

    name = "lying-reader"

    def read(self, addr, n, context=""):
        return "ff", b"\xff"

    def write(self, addr, data, context=""):
        return "OK " + data.hex(), data

    def erase(self, page, context=""):
        return "OK"


class ShortWriter:
    """写回复多给字节（不合规），读正常。"""

    name = "short-writer"

    def read(self, addr, n, context=""):
        return "00" * n, b"\x00" * n

    def write(self, addr, data, context=""):
        return "OK 1234", bytes.fromhex("1234")

    def erase(self, page, context=""):
        return "OK"


class DeadBackend:
    """完全不响应。"""

    name = "dead"

    def _boom(self, *a, **k):
        raise BackendError("连接超时")

    read = _boom
    write = _boom
    erase = _boom


# ---------------------------------------------------------------- 1. 安静

print("=" * 64)
print("1. 正常路径：只有指令与回复")
print("=" * 64)
box, ev = make_box()
box.store(0x10, 0x2A)
box.load(0x10)
kinds = [k for k, _ in ev]
check("事件序列", kinds, ["bus", "ai", "bus", "ai"])
check("无 fyi 附注", "fyi" in kinds, False)
check("无 err", "err" in kinds, False)
check("无'落纸'字样", any("落纸" in t for _, t in ev), False)

# ---------------------------------------------------------------- 2. 以回复为准

print()
print("=" * 64)
print("2. 读：以回复为准，不用纸覆盖")
print("=" * 64)
box, ev = make_box()
box.memory.raw_write(0x10, b"\x2a")
box.backend = LyingReader()
value = box.load(0x10)
check("CPU 拿到回复值 0xFF", value, 0xFF)
check("纸未被覆盖（仍是 0x2a）", box.memory.peek(0x10, 1), b"\x2a")
check("读侧故障 +1", box.memory.stats.read_mismatches, 1)
check("幻觉 +1", box.hallucinations, 1)
check("有报错提示", any("读错了" in t for _, t in ev), True)
check("报错里说明以它为准", any("以它说的为准" in t for _, t in ev), True)

# ---------------------------------------------------------------- 3. 不补位

print()
print("=" * 64)
print("3. 写：位数不合规也不补位，照它给的写")
print("=" * 64)
box, ev = make_box()
box.backend = ShortWriter()
box.store(0x20, 0x05)
check("纸上取它给的第 1 字节 0x12", box.memory.peek(0x20, 1), b"\x12")
check("记了写数不合规", any("写数不合规" in t for _, t in ev), True)
check("写侧故障 ≥1", box.memory.stats.faults >= 1, True)

# ---------------------------------------------------------------- 4. 无响应

print()
print("=" * 64)
print("4. 无响应：读按 0、写丢弃，各记一次（不重复计数）")
print("=" * 64)
box, ev = make_box()
box.memory.raw_write(0x30, b"\x77")
box.backend = DeadBackend()
check("读按 0 返回（不是 0x77）", box.load(0x30), 0)
check("纸没变", box.memory.peek(0x30, 1), b"\x77")
check("读侧故障恰好 +1", box.memory.stats.read_mismatches, 1)

box.memory.raw_write(0x40, b"\x11")
box.store(0x40, 0x99)
check("写被丢弃，纸上仍是 0x11", box.memory.peek(0x40, 1), b"\x11")
check("写侧故障恰好 +1", box.memory.stats.faults, 1)
check("幻觉总数 2", box.hallucinations, 2)

# ---------------------------------------------------------------- 5. 端到端

print()
print("=" * 64)
print("5. 端到端：sort 跑通且黑窗干净")
print("=" * 64)
ev = []
box = Sandbox(
    SandboxConfig(
        memory_bytes=4096, page_size=128, backend_spec="local", pacing=0.0
    ),
    emit=lambda e: ev.append(e.text),
)
r = box.run(load_program("sort"))
check("排序正确", list(r.output), [1, 3, 5, 9])
check("无故障", r.faults, 0)
check("无'落纸'字样", any("落纸" in t for t in ev), False)
check("无'以纸上为准'", any("以纸上为准" in t for t in ev), False)
sends = sum(1 for t in ev if t.startswith(">>"))
recvs = sum(1 for t in ev if t.startswith("<<"))
check("指令与回复一一对应", sends, recvs)

print()
print("=" * 64)
print("6. 程序输出是实时的，且报告里不重复")
print("=" * 64)
ev = []
box = Sandbox(
    SandboxConfig(
        memory_bytes=4096, page_size=128, backend_spec="local", pacing=0.0
    ),
    emit=lambda e: ev.append((e.kind, e.text)),
)
r = box.run(load_program("hello"))
io_idx = [i for i, (k, _) in enumerate(ev) if k == "io"]
read_idx = [i for i, (k, t) in enumerate(ev) if t.startswith(">> READ")]
check("有 io 事件（数量 > 0）", len(io_idx) > 0, True)
check("有 READ 事件（数量 > 0）", len(read_idx) > 0, True)
if io_idx and read_idx:
    # 输出必须出现在读操作**之后**——证明是边跑边刷，不是跑完才倒
    check(
        "输出晚于读操作（边跑边刷）",
        io_idx[0] > read_idx[0],
        True,
    )
check(
    "报告里不再有 [程序输出]",
    any("[程序输出]" in t for _, t in ev),
    False,
)
print("  事件顺序（节选）：")
for k, t in ev:
    if k in ("bus", "io") or t.startswith("[执行结果]"):
        print(f"     [{k:>3}] {t}")

print()
print("=" * 64)
print("FAILS =", FAILS)
print("=" * 64)
if __name__ == "__main__":
    raise SystemExit(1 if FAILS else 0)
