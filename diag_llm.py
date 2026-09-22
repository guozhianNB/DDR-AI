"""真模型后端诊断：打一次真实请求，把原始返回结构挖出来。

    python diag_llm.py
    python diag_llm.py "READ 0x0040 1"

什么时候用：
  - 黑窗里回复是空的（`<<` 后面什么都没有）；
  - 报错说"读数不合规：它给了 (空)"；
  - 想确认自家模型的返回字段长什么样（content？reasoning_content？）。

它会分别用 max_tokens=64 和 512 各打一次，对比差异——
思考型模型（deepseek-flash / deepseek-reasoner 等）在预算不足时
会把思考过程写进 reasoning_content，而 content 是空的。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ddr_ai.ai_backend import SYSTEM_PROMPT  # noqa: E402
from ddr_ai import settings as st  # noqa: E402

saved = st.load_saved()
prompt = sys.argv[1] if len(sys.argv) > 1 else "READ 0x0040 1"

print("=" * 70)
print("配置")
print("=" * 70)
print(f"  backend    : {saved.backend}")
print(f"  base_url   : {saved.base_url}")
print(f"  model      : {saved.model}")
print(f"  api_key    : {'已设置（' + str(len(saved.api_key)) + ' 字符）' if saved.api_key else '空！'}")
print(f"  temperature: {saved.temperature}")
print(f"  extra_json : {saved.extra_json!r}")
print(f"  max_tokens : {saved.max_tokens}")
print(f"  env 文件   : {saved.env_path}（{'存在' if saved.from_file else '不存在'}）")
print()
print(f"  测试指令   : {prompt}")

if not saved.api_key:
    print()
    print("没有 API Key。在启动器里填好并勾上「记住」，或直接编辑 .env，然后再跑。")
    raise SystemExit(1)

url = saved.base_url.rstrip("/") + "/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {saved.api_key}",
}

verdict_parts: list[str] = []

for max_tokens in (64, 512):
    body = {
        "model": saved.model,
        "temperature": saved.temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if saved.extra_json.strip():
        try:
            extra = json.loads(saved.extra_json)
            if isinstance(extra, dict):
                body.update(extra)
        except json.JSONDecodeError:
            pass

    print()
    print("=" * 70)
    print(f"请求 max_tokens={max_tokens}")
    print("=" * 70)
    print(f"  URL: {url}")

    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        status = exc.code
    except Exception as exc:  # noqa: BLE001
        print(f"  请求异常: {exc!r}")
        continue
    elapsed = time.perf_counter() - t0

    print(f"  HTTP {status}  耗时 {elapsed:.2f}s")
    print()
    print("  --- 原始返回 ---")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(raw[:2000])
        continue
    print(json.dumps(payload, ensure_ascii=False, indent=2)[:3000])

    print()
    print("  --- 关键字段探测 ---")
    try:
        choice = payload["choices"][0]
        msg = choice.get("message", {})
        content = msg.get("content")
        reasoning = msg.get("reasoning_content")
        print(f"  message.content           = {content!r}")
        print(f"  message.reasoning_content = {reasoning!r}")
        print(f"  finish_reason             = {choice.get('finish_reason')!r}")
        print(f"  usage                     = {payload.get('usage')}")
        print()
        if not (content or "").strip():
            if (reasoning or "").strip():
                print("  >>> content 为空，但 reasoning_content 有内容。")
                print("      思考型模型的典型形态；DDR-AI 会自动回退去读它。")
                verdict_parts.append("靠 reasoning_content 回退")
            else:
                print("  >>> content 与 reasoning_content 都空。")
                print("      大概率是 finish_reason=length：预算被思考吃完了。")
                verdict_parts.append("空回复，需调大 max_tokens")
        else:
            print("  >>> content 正常，可用。")
            verdict_parts.append("content 正常")
    except (KeyError, IndexError, TypeError) as exc:
        print(f"  解析结构失败: {exc}")

print()
print("=" * 70)
print("结论")
print("=" * 70)
print(f"  max_tokens=64 : {verdict_parts[0] if verdict_parts else '无结果'}")
if len(verdict_parts) > 1:
    print(f"  max_tokens=512: {verdict_parts[1]}")
    if "content 正常" in verdict_parts[1] and "空" in verdict_parts[0]:
        print()
        print("  → 建议：把界面上的 max_tokens 调到 512 以上。")
