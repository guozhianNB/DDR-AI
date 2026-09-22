"""思考控制测试：字段开关、方言差异、往返持久化。

    python tests/test_thinking.py

核心断言：**只有勾选的字段才会进请求 JSON**。
各家模型字段名不同，硬编码必然在某家上报"未知字段"。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ddr_ai import ai_backend as ab  # noqa: E402
from ddr_ai import settings as st  # noqa: E402

FAILS = 0


def check(label: str, got, want) -> None:
    global FAILS
    ok = got == want
    if not ok:
        FAILS += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {label}: got={got!r} want={want!r}")


BASE = dict(base_url="http://x/v1", api_key="k", model="m")
STRIP = ("messages", "model", "temperature", "max_tokens")


def think_keys(s: ab.LLMSettings, dialect: str = "openai") -> dict:
    body = s.merged_body("READ 0x0040 1", dialect=dialect)
    return {k: v for k, v in body.items() if k not in STRIP}


print("=" * 64)
print("1. 每个开关独立控制一个字段")
print("=" * 64)
check("默认：什么都不发", think_keys(ab.LLMSettings(**BASE)), {})
check(
    "只开强度",
    think_keys(ab.LLMSettings(**BASE, thinking_effort_enabled=True, thinking_effort="low")),
    {"reasoning_effort": "low"},
)
check(
    "只开预算",
    think_keys(ab.LLMSettings(**BASE, thinking_tokens_enabled=True, thinking_budget_tokens=4096)),
    {"thinking_budget_tokens": 4096},
)
check(
    "只开自定义参数",
    think_keys(
        ab.LLMSettings(
            **BASE,
            thinking_param_enabled=True,
            thinking_param_json='{"thinking": {"type": "enabled"}}',
        )
    ),
    {"thinking": {"type": "enabled"}},
)
check(
    "三个全开",
    think_keys(
        ab.LLMSettings(
            **BASE,
            thinking_effort_enabled=True,
            thinking_effort="high",
            thinking_tokens_enabled=True,
            thinking_budget_tokens=1024,
            thinking_param_enabled=True,
            thinking_param_json='{"foo": 1}',
        )
    ),
    {"reasoning_effort": "high", "thinking_budget_tokens": 1024, "foo": 1},
)

print()
print("=" * 64)
print("2. 方言差异：Anthropic 不接受顶层 thinking_budget_tokens")
print("=" * 64)
check(
    "anthropic 预算包成对象",
    think_keys(
        ab.LLMSettings(**BASE, thinking_tokens_enabled=True, thinking_budget_tokens=2048),
        "anthropic",
    ),
    {"thinking": {"type": "enabled", "budget_tokens": 2048}},
)
check(
    "openai 预算是顶层字段",
    think_keys(
        ab.LLMSettings(**BASE, thinking_tokens_enabled=True, thinking_budget_tokens=2048),
        "openai",
    ),
    {"thinking_budget_tokens": 2048},
)

print()
print("=" * 64)
print("3. 总开关关闭")
print("=" * 64)
check(
    "关闭 + 留空 = 完全不提",
    think_keys(
        ab.LLMSettings(
            **BASE,
            thinking_enabled=False,
            thinking_effort_enabled=True,
            thinking_tokens_enabled=True,
            thinking_param_enabled=True,
        )
    ),
    {},
)
check(
    "关闭 + 显式禁用 JSON",
    think_keys(
        ab.LLMSettings(
            **BASE,
            thinking_enabled=False,
            thinking_off_json='{"thinking": {"type": "disabled"}}',
            thinking_effort_enabled=True,
        )
    ),
    {"thinking": {"type": "disabled"}},
)

print()
print("=" * 64)
print("4. extra_json 优先级最高，可覆盖思考字段")
print("=" * 64)
check(
    "extra 覆盖 effort",
    think_keys(
        ab.LLMSettings(
            **BASE,
            thinking_effort_enabled=True,
            thinking_effort="low",
            extra_json='{"reasoning_effort": "high"}',
        )
    ),
    {"reasoning_effort": "high"},
)

print()
print("=" * 64)
print("5. 非法 JSON 要报错，不能静默吞掉")
print("=" * 64)
for field, kwargs in [
    ("自定义思考参数", dict(thinking_param_enabled=True, thinking_param_json="{不是json")),
    ("自定义请求 JSON", dict(extra_json="{不是json")),
    ("关闭思考 JSON", dict(thinking_enabled=False, thinking_off_json="[1,2]")),
]:
    try:
        ab.LLMSettings(**BASE, **kwargs).merged_body("READ 0x0040 1")
        check(f"{field} 非法时抛错", "没抛错", "BackendError")
    except ab.BackendError:
        check(f"{field} 非法时抛错", "BackendError", "BackendError")

print()
print("=" * 64)
print("6. 思考 token 统计（各家字段名不同）")
print("=" * 64)
for usage, want in [
    ({"reasoning_tokens": 7}, 7),
    ({"thinking_tokens": 8}, 8),
    ({"reasoning_output_tokens": 9}, 9),
    ({"completion_tokens_details": {"reasoning_tokens": 10}}, 10),
    ({"completion_tokens": 5}, 0),
    ({}, 0),
]:
    got = ab.LLMStrictMemory._thinking_tokens(usage)
    check(f"usage={usage}", got, want)

print()
print("=" * 64)
print("7. .env 往返（用临时目录，不碰你的 .env）")
print("=" * 64)
TMP = Path(__file__).resolve().parent / "_tmp_thinking"
if TMP.exists():
    shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True, exist_ok=True)
real = st.DEFAULT_ENV_PATH
try:
    st.DEFAULT_ENV_PATH = TMP / ".env"
    st.save_saved(
        backend="openai-compat",
        base_url="http://x/v1",
        model="m",
        api_key="k",
        temperature=0.3,
        extra_json="",
        thinking_enabled=False,
        thinking_off_json='{"thinking": {"type": "disabled"}}',
        thinking_effort_enabled=True,
        thinking_effort="high",
        thinking_tokens_enabled=True,
        thinking_budget_tokens=4096,
        thinking_param_enabled=True,
        thinking_param_json='{"a": 1}',
    )
    got = st.load_saved()
    check("thinking_enabled", got.thinking_enabled, False)
    check("thinking_off_json", got.thinking_off_json, '{"thinking": {"type": "disabled"}}')
    check("thinking_effort_enabled", got.thinking_effort_enabled, True)
    check("thinking_effort", got.thinking_effort, "high")
    check("thinking_tokens_enabled", got.thinking_tokens_enabled, True)
    check("thinking_budget_tokens", got.thinking_budget_tokens, 4096)
    check("thinking_param_enabled", got.thinking_param_enabled, True)
    check("thinking_param_json", got.thinking_param_json, '{"a": 1}')

    # 还原成默认再存一次，确认布尔能写回 0
    st.save_saved(
        backend="local", base_url="http://x/v1", model="m", api_key="",
        temperature=0.0, extra_json="", thinking_enabled=True,
        thinking_effort_enabled=False, thinking_tokens_enabled=False,
        thinking_param_enabled=False,
    )
    got2 = st.load_saved()
    check("布尔写回真", got2.thinking_enabled, True)
    check("布尔写回假", got2.thinking_effort_enabled, False)
finally:
    st.DEFAULT_ENV_PATH = real
    shutil.rmtree(TMP, ignore_errors=True)

print()
print("=" * 64)
print("FAILS =", FAILS)
print("=" * 64)
if __name__ == "__main__":
    raise SystemExit(1 if FAILS else 0)
