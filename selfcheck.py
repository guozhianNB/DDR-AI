"""无头自检入口（仓库根便捷脚本）。

等价于 `python -m ddr_ai --check ...`。
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows 控制台默认不是 UTF-8，中文会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ddr_ai.selfcheck import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
