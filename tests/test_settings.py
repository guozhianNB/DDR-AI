"""`.env` 读写测试：解析、写回、往返、优先级。

    python tests/test_settings.py

注意：这些测试一律操作**临时目录里的** .env，
通过覆盖 settings.DEFAULT_ENV_PATH 实现，绝不碰项目里那份真的。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ddr_ai import settings as st  # noqa: E402

# 就地建临时目录，别写系统 TEMP——某些沙箱环境不允许写工作区之外
TMP_ROOT = Path(__file__).resolve().parent / "_tmp_settings"

FAILS = 0


def check(label: str, got, want) -> None:
    global FAILS
    ok = got == want
    if not ok:
        FAILS += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {label}: got={got!r} want={want!r}")


print("=" * 64)
print("1. 解析")
print("=" * 64)
text = """\
# 整行注释
DDR_AI_BASE_URL=https://api.openai.com/v1
DDR_AI_MODEL="gpt-4o-mini"
export DDR_AI_API_KEY='sk-abc'
DDR_AI_TEMPERATURE=0.7   # 行内注释
DDR_AI_EXTRA_JSON={"top_p": 0.9}
没有等号的行
= 空键
DDR_AI_EMPTY=
"""
d = st.parse_env_text(text)
check("普通值", d["DDR_AI_BASE_URL"], "https://api.openai.com/v1")
check("双引号剥掉", d["DDR_AI_MODEL"], "gpt-4o-mini")
check("export + 单引号", d["DDR_AI_API_KEY"], "sk-abc")
check("行内注释剥掉", d["DDR_AI_TEMPERATURE"], "0.7")
check("含引号的值保留", d["DDR_AI_EXTRA_JSON"], '{"top_p": 0.9}')
check("无等号行被忽略", "没有等号的行" in d, False)
check("空键被忽略", "" in d, False)
check("空值保留为空串", d["DDR_AI_EMPTY"], "")

print()
print("=" * 64)
print("2. 值格式化")
print("=" * 64)
check("空值加引号", st._fmt_value(""), '""')
check("普通值不加", st._fmt_value("sk-abc"), "sk-abc")
check("含 # 加引号", st._fmt_value("a#b"), '"a#b"')
check("首尾空白加引号", st._fmt_value("  x  "), '"  x  "')

print()
print("=" * 64)
print("3. 写回：保留注释与无关键")
print("=" * 64)
if TMP_ROOT.exists():
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
TMP_ROOT.mkdir(parents=True, exist_ok=True)

real_default = st.DEFAULT_ENV_PATH
try:
    envp = TMP_ROOT / ".env"
    st.DEFAULT_ENV_PATH = envp

    envp.write_text(
        "# 我手写的注释\n"
        "DDR_AI_BASE_URL=https://old.example/v1\n"
        "SOME_UNRELATED_KEY=keep-me\n"
        "DDR_AI_MODEL=old-model\n",
        encoding="utf-8",
    )
    st.save_saved(
        backend="openai-compat",
        base_url="https://new.example/v1",
        model="new-model",
        api_key="sk-new",
        temperature=0.3,
        extra_json="",
    )
    after = envp.read_text(encoding="utf-8")
    check("注释保留", "# 我手写的注释" in after, True)
    check("无关键保留", "SOME_UNRELATED_KEY=keep-me" in after, True)
    check("Base URL 已更新", "https://new.example/v1" in after, True)
    check("旧值已消失", "old.example" in after, False)
    check("模型已更新", "new-model" in after, True)
    check("key 已写入", "DDR_AI_API_KEY=sk-new" in after, True)

    print()
    print("=" * 64)
    print("4. 往返")
    print("=" * 64)
    got = st.load_saved()
    check("backend", got.backend, "openai-compat")
    check("base_url", got.base_url, "https://new.example/v1")
    check("model", got.model, "new-model")
    check("api_key", got.api_key, "sk-new")
    check("temperature", round(got.temperature, 3), 0.3)
    check("from_file", got.from_file, True)
    check("has_api_key", got.has_api_key(), True)

    print()
    print("=" * 64)
    print("5. 空文件 / 文件不存在")
    print("=" * 64)
    envp.unlink()
    got = st.load_saved()
    check("不存在时 from_file", got.from_file, False)
    check("不存在时 backend 兜底", got.backend, "local")

    # 不存在的路径也要能创建
    nested = TMP_ROOT / "sub" / "dir" / ".env"
    st.DEFAULT_ENV_PATH = nested
    st.save_saved(
        backend="anthropic",
        base_url="https://api.anthropic.com/v1",
        model="claude-3-5-haiku-latest",
        api_key="",
        temperature=0.0,
        extra_json="",
    )
    check("自动建目录", nested.is_file(), True)
    body = nested.read_text(encoding="utf-8")
    check("新文件带说明头", "由启动器自动读写" in body, True)
    check("空 key 写成空引号", 'DDR_AI_API_KEY=""' in body, True)
    got = st.load_saved()
    check("anthropic 往返", got.backend, "anthropic")
    check("空 key 读回为空", got.api_key, "")

    print()
    print("=" * 64)
    print("6. 温度钳制与坏值")
    print("=" * 64)
    st.DEFAULT_ENV_PATH = envp  # 第 5 节把它改到 nested 了，这里换回来
    envp.write_text("DDR_AI_TEMPERATURE=9.9\n", encoding="utf-8")
    check("温度上限钳到 2.0", st.load_saved().temperature, 2.0)
    envp.write_text("DDR_AI_TEMPERATURE=-3\n", encoding="utf-8")
    check("温度下限钳到 0.0", st.load_saved().temperature, 0.0)
    envp.write_text("DDR_AI_TEMPERATURE=abc\n", encoding="utf-8")
    check("坏值退回 0.0", st.load_saved().temperature, 0.0)
finally:
    st.DEFAULT_ENV_PATH = real_default
    shutil.rmtree(TMP_ROOT, ignore_errors=True)

print()
print("=" * 64)
print("7. 项目里的 .env 必须被 git 忽略")
print("=" * 64)
root = Path(__file__).resolve().parent.parent
gi = (root / ".gitignore").read_text(encoding="utf-8")
check(".gitignore 含 .env", any(
    ln.strip() == ".env" for ln in gi.splitlines()
), True)
check(".env.example 存在", (root / ".env.example").is_file(), True)

print()
print("=" * 64)
print("FAILS =", FAILS)
print("=" * 64)
if __name__ == "__main__":
    raise SystemExit(1 if FAILS else 0)
