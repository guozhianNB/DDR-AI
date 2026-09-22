"""`.env` 读写：把接真模型要填的那几项记住，省得每次启动重填。

设计要点：

- **路径锚定项目根**，用 `__file__` 往上走，不依赖当前工作目录。
  否则从别的目录 `python -m ddr_ai` 就会读不到自己的配置。
- **不引入 python-dotenv**，自己写个小解析器：只需支持 `KEY=VALUE`、
  `#` 注释、引号包裹、`export` 前缀。少一个依赖就少一份风险。
- **保存时保留注释和无关键**，只覆盖我们管的那几个。
- `.env` 已在 `.gitignore` 里——里面是 API Key，绝不能进版本库。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 本文件在 <root>/src/ddr_ai/settings.py，所以项目根要往上走三层
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

# 本模块认的全部键
KEYS = (
    "DDR_AI_BACKEND",
    "DDR_AI_BASE_URL",
    "DDR_AI_MODEL",
    "DDR_AI_API_KEY",
    "DDR_AI_TEMPERATURE",
    "DDR_AI_MAX_TOKENS",
    "DDR_AI_CONTEXT_MODE",
    "DDR_AI_THINKING",
    "DDR_AI_THINKING_OFF_JSON",
    "DDR_AI_THINKING_EFFORT_ENABLED",
    "DDR_AI_THINKING_EFFORT",
    "DDR_AI_THINKING_TOKENS_ENABLED",
    "DDR_AI_THINKING_BUDGET_TOKENS",
    "DDR_AI_THINKING_PARAM_ENABLED",
    "DDR_AI_THINKING_PARAM_JSON",
    "DDR_AI_EXTRA_JSON",
)

_BOOL_KEYS = (
    "DDR_AI_THINKING",
    "DDR_AI_THINKING_EFFORT_ENABLED",
    "DDR_AI_THINKING_TOKENS_ENABLED",
    "DDR_AI_THINKING_PARAM_ENABLED",
)


def _as_bool(text: str, default: bool) -> bool:
    if not text:
        return default
    return text.strip().lower() in ("1", "true", "yes", "on", "是")

_HEADER = """\
# DDR-AI 本地配置 —— 由启动器自动读写。
# 这个文件已被 .gitignore 忽略，不会进版本库。
# 想手动改也完全可以，直接编辑下面的键值即可。
"""


# ===================================================================== 解析


def parse_env_text(text: str) -> dict[str, str]:
    """极简 dotenv 解析。

    支持：
        KEY=VALUE
        KEY="带空格的值"
        KEY='单引号值'
        export KEY=VALUE
        # 整行注释、行内 # 注释（仅在无引号时）

    不支持（也不打算支持）：多行值、变量插值、命令替换。
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            # 引号包裹：整段都是值，不做注释剥离
            value = value[1:-1]
        else:
            # 裸值：剥掉行内注释
            hash_at = value.find("#")
            if hash_at >= 0:
                value = value[:hash_at].rstrip()
        out[key] = value
    return out


def read_env(path: Path | None = None) -> dict[str, str]:
    """读 `.env`。文件不存在就返回空字典，不抛异常。"""
    p = Path(path) if path is not None else DEFAULT_ENV_PATH
    try:
        text = p.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return {}
    return parse_env_text(text)


def _fmt_value(value: str) -> str:
    """需要时加引号，保证读回来还是原样。"""
    if value == "":
        return '""'
    if value != value.strip() or "#" in value or "\n" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def write_env(updates: dict[str, str], path: Path | None = None) -> Path:
    """把 updates 合并进 `.env`。

    保留已有注释与无关键；只覆盖 updates 里出现的键，
    没有的键就追加到末尾。
    """
    p = Path(path) if path is not None else DEFAULT_ENV_PATH
    try:
        original = p.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        original = ""

    pending = dict(updates)
    lines_out: list[str] = []

    for raw in original.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            lines_out.append(raw)
            continue

        probe = stripped[len("export ") :] if stripped.startswith("export ") else stripped
        key = probe.split("=", 1)[0].strip()
        if key in pending:
            lines_out.append(f"{key}={_fmt_value(pending.pop(key))}")
        else:
            lines_out.append(raw)

    # 剩下的（新键）追加到末尾
    if pending:
        if lines_out and lines_out[-1].strip():
            lines_out.append("")
        for key in KEYS:
            if key in pending:
                lines_out.append(f"{key}={_fmt_value(pending.pop(key))}")
        # 不在 KEYS 里的也写出去，免得调用方给了却被吞掉
        for key, value in pending.items():
            lines_out.append(f"{key}={_fmt_value(value)}")

    if not original:
        body = _HEADER + "\n" + "\n".join(lines_out)
    else:
        body = "\n".join(lines_out)
    if not body.endswith("\n"):
        body += "\n"

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


# ===================================================================== 取值


@dataclass
class SavedConfig:
    """从 `.env` + 环境变量合并出来的配置。

    优先级：`.env` > 进程环境变量 > 调用方给的默认值。
    先 `.env` 是因为它是"用户在窗口里明确填过"的那一份，更贴近意图。
    """

    backend: str = "local"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 512
    context_mode: str = "state"
    """给真模型多少内存上下文：state / recent / none。
    不喂上下文，无状态模型读内存只能瞎猜——这是硬约束。"""

    # ---- 思考控制。每项都有开关，决定"请求 JSON 里要不要出现这个字段" ----
    thinking_enabled: bool = True
    thinking_off_json: str = ""
    thinking_effort_enabled: bool = False
    thinking_effort: str = "medium"
    thinking_tokens_enabled: bool = False
    thinking_budget_tokens: int = 2048
    thinking_param_enabled: bool = False
    thinking_param_json: str = '{"thinking": {"type": "enabled"}}'

    extra_json: str = ""
    env_path: Path = DEFAULT_ENV_PATH
    from_file: bool = False
    """`.env` 是否真实存在（用来在界面上提示用户）。"""

    def has_api_key(self) -> bool:
        return bool(self.api_key.strip())


def _first(*values: str | None) -> str:
    for v in values:
        if v is not None and v.strip():
            return v.strip()
    return ""


def load_saved(path: Path | None = None) -> SavedConfig:
    """读 `.env`，缺失的项退回环境变量与内置默认值。"""
    p = Path(path) if path is not None else DEFAULT_ENV_PATH
    file_vals = read_env(p)

    def pick(key: str, *fallbacks: str) -> str:
        return _first(file_vals.get(key), *fallbacks)

    # Anthropic 的默认地址跟 OpenAI 不一样，按后端给不同兜底
    backend = pick("DDR_AI_BACKEND", os.environ.get("DDR_AI_BACKEND")) or "local"
    default_base = (
        "https://api.anthropic.com/v1"
        if backend == "anthropic"
        else "https://api.openai.com/v1"
    )

    temp_raw = pick("DDR_AI_TEMPERATURE", os.environ.get("DDR_AI_TEMPERATURE"))
    try:
        temperature = float(temp_raw) if temp_raw else 0.0
    except ValueError:
        temperature = 0.0
    temperature = min(max(temperature, 0.0), 2.0)

    tokens_raw = pick("DDR_AI_MAX_TOKENS", os.environ.get("DDR_AI_MAX_TOKENS"))
    try:
        max_tokens = int(tokens_raw) if tokens_raw else 512
    except ValueError:
        max_tokens = 512
    max_tokens = min(max(max_tokens, 1), 32_768)

    api_key = pick(
        "DDR_AI_API_KEY",
        os.environ.get("DDR_AI_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("ANTHROPIC_API_KEY"),
    )

    ctx = pick("DDR_AI_CONTEXT_MODE", os.environ.get("DDR_AI_CONTEXT_MODE")) or "state"
    if ctx not in ("state", "recent", "none"):
        ctx = "state"

    budget_raw = pick(
        "DDR_AI_THINKING_BUDGET_TOKENS",
        os.environ.get("DDR_AI_THINKING_BUDGET_TOKENS"),
    )
    try:
        budget = int(budget_raw) if budget_raw else 2048
    except ValueError:
        budget = 2048
    budget = min(max(budget, 1), 200_000)

    return SavedConfig(
        backend=backend,
        base_url=pick("DDR_AI_BASE_URL", os.environ.get("DDR_AI_BASE_URL"), default_base),
        model=pick("DDR_AI_MODEL", os.environ.get("DDR_AI_MODEL")) or "gpt-4o-mini",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        context_mode=ctx,
        thinking_enabled=_as_bool(file_vals.get("DDR_AI_THINKING", ""), True),
        thinking_off_json=pick(
            "DDR_AI_THINKING_OFF_JSON", os.environ.get("DDR_AI_THINKING_OFF_JSON"), ""
        ),
        thinking_effort_enabled=_as_bool(
            file_vals.get("DDR_AI_THINKING_EFFORT_ENABLED", ""), False
        ),
        thinking_effort=pick(
            "DDR_AI_THINKING_EFFORT", os.environ.get("DDR_AI_THINKING_EFFORT")
        )
        or "medium",
        thinking_tokens_enabled=_as_bool(
            file_vals.get("DDR_AI_THINKING_TOKENS_ENABLED", ""), False
        ),
        thinking_budget_tokens=budget,
        thinking_param_enabled=_as_bool(
            file_vals.get("DDR_AI_THINKING_PARAM_ENABLED", ""), False
        ),
        thinking_param_json=pick(
            "DDR_AI_THINKING_PARAM_JSON",
            os.environ.get("DDR_AI_THINKING_PARAM_JSON"),
        )
        or '{"thinking": {"type": "enabled"}}',
        extra_json=pick("DDR_AI_EXTRA_JSON", os.environ.get("DDR_AI_EXTRA_JSON"), ""),
        env_path=p,
        from_file=p.is_file(),
    )


def save_saved(
    *,
    backend: str,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float,
    extra_json: str,
    max_tokens: int = 512,
    context_mode: str = "state",
    thinking_enabled: bool = True,
    thinking_off_json: str = "",
    thinking_effort_enabled: bool = False,
    thinking_effort: str = "medium",
    thinking_tokens_enabled: bool = False,
    thinking_budget_tokens: int = 2048,
    thinking_param_enabled: bool = False,
    thinking_param_json: str = '{"thinking": {"type": "enabled"}}',
    path: Path | None = None,
) -> Path:
    """把当前配置写回 `.env`。"""
    return write_env(
        {
            "DDR_AI_BACKEND": backend,
            "DDR_AI_BASE_URL": base_url,
            "DDR_AI_MODEL": model,
            "DDR_AI_API_KEY": api_key,
            "DDR_AI_TEMPERATURE": f"{temperature:g}",
            "DDR_AI_MAX_TOKENS": str(max_tokens),
            "DDR_AI_CONTEXT_MODE": context_mode,
            "DDR_AI_THINKING": "1" if thinking_enabled else "0",
            "DDR_AI_THINKING_OFF_JSON": thinking_off_json,
            "DDR_AI_THINKING_EFFORT_ENABLED": "1" if thinking_effort_enabled else "0",
            "DDR_AI_THINKING_EFFORT": thinking_effort,
            "DDR_AI_THINKING_TOKENS_ENABLED": "1" if thinking_tokens_enabled else "0",
            "DDR_AI_THINKING_BUDGET_TOKENS": str(thinking_budget_tokens),
            "DDR_AI_THINKING_PARAM_ENABLED": "1" if thinking_param_enabled else "0",
            "DDR_AI_THINKING_PARAM_JSON": thinking_param_json,
            "DDR_AI_EXTRA_JSON": extra_json,
        },
        path,
    )
