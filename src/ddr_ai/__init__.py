"""DDR-AI —— 用 AI 和纸做的内存条。

    python -m ddr_ai          启动 GUI
    python -m ddr_ai --check  无头自检（不开窗口）

它当然不能真的当内存用。但如果你非要问"到底有多慢"，
这个包里每一个数字都是按真实物理量算出来的。
"""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> None:
    import sys

    if "--check" in sys.argv:
        from .selfcheck import main as check_main

        raise SystemExit(check_main())

    from .launcher import main as launcher_main

    launcher_main()


__all__ = ["main", "__version__"]
