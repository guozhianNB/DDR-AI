"""无头自检：不开 GUI，直接跑一次沙箱，验证端到端链路。

命令行：
    python -m ddr_ai --check              跑 sort
    python -m ddr_ai --check hello 0.3    指定程序与幻觉率
    python selfcheck.py sort --quiet      仓库根的同名脚本走这条路
"""

from __future__ import annotations

import sys
import time

from .bus import LatencyProfile
from .guest import load_program
from .sandbox import SandboxConfig, run_in_thread


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    args = [a for a in args if a != "--check"]
    quiet = "--quiet" in args
    args = [a for a in args if not a.startswith("--")]

    name = args[0] if args else "sort"
    hallucination = float(args[1]) if len(args) > 1 else 0.0

    try:
        program = load_program(name)
    except Exception as exc:  # noqa: BLE001
        print(f"加载程序失败：{exc}")
        return 1

    config = SandboxConfig(
        memory_bytes=4096,
        page_size=128,
        profile=LatencyProfile.HUMAN_PEN,
        hallucination_rate=hallucination,
        backend_spec="local",
        pacing=0.0,  # 自检不等真实时间
    )

    def emit(ev) -> None:
        if not quiet:
            print(f"{ev.kind:>3} | {ev.text}", flush=True)

    box: list = []
    run_in_thread(config, program, emit, box.append)

    deadline = time.time() + 120
    while not box and time.time() < deadline:
        time.sleep(0.02)

    if not box:
        print("自检失败：沙箱没有在 120 秒内返回")
        return 1

    r = box[0]
    print()
    print("=" * 64)
    print(f"程序        : {r.program}")
    print(f"是否成功    : {r.ok}")
    print(f"结论        : {r.message}")
    print(f"输出字节    : {list(r.output)}")
    print(f"CPU 步数    : {r.steps}")
    print(f"内存访问    : {r.accesses}")
    print(f"仿真耗时    : {r.sim_seconds:.3f} s")
    print(f"等效速率    : {r.effective_bps:.4f} bit/s")
    print(f"幻觉/写坏   : {r.faults}")
    print("=" * 64)
    return 0 if r.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
