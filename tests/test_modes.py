"""端到端测试：五种运行模式全跑一遍。

    python tests/test_modes.py

覆盖：
  1. 人工智能 · 本地仿 AI
  2. 人工智能 · OpenAI 兼容端点（起本地假服务器，验证 HTTP + JSON 合并 + 协议解析）
  3. 人工智能 · 真模型不守格式（验证故障判定）
  4. 能工智人 · 仿真
  5. 能工智人 · 真实（自动喂答案，验证输入桥）
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import tkinter as tk
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ddr_ai import ai_backend as ab  # noqa: E402
from ddr_ai.bus import LatencyProfile  # noqa: E402
from ddr_ai.guest import load_program  # noqa: E402
from ddr_ai.sandbox import SandboxConfig, run_in_thread  # noqa: E402

FAILS = 0
REQUESTS: list[dict] = []


def check(label: str, cond: bool, detail: object = "") -> None:
    global FAILS
    if not cond:
        FAILS += 1
    suffix = f" | {detail}" if detail != "" else ""
    print(f"  {'OK  ' if cond else 'FAIL'} {label}{suffix}")


# ============================================================ 假 LLM 服务器


class FakeLLM(BaseHTTPRequestHandler):
    """一个只会照协议回答的假 chat/completions 端点。

    mode:
      strict      —— 规矩回答
      sloppy      —— 夹带解释，不守格式
      thinking    —— 模拟思考型模型：content 空、内容塞 reasoning_content
      budget_dead —— 思考过程吃完预算，content 与 reasoning 都空
                     （只有更大的 max_tokens 才救得回来）
      leaky       —— 把整段思考过程当正文吐出来（真实发生过的事故）
    """

    mode = "strict"
    memory: dict[int, int] = {}
    calls = 0
    goldfish = False
    """True 时模拟**完全无状态**的模型：不留任何记忆，
    READ 只能靠 prompt 里的上下文回答。用来对照 none/state 的差别。"""

    def log_message(self, *args) -> None:  # 静音
        pass

    @staticmethod
    def _parse_memory_dump(text: str) -> dict[int, int]:
        """从 prompt 里的十六进制转储解析出内存内容。

        形如 `0040: 48 49 00 ...`。
        """
        mem: dict[int, int] = {}
        for line in text.splitlines():
            m = re.match(r"\s*([0-9a-fA-F]{4,}):\s+((?:[0-9a-fA-F]{2}\s*)+)", line)
            if not m:
                continue
            base = int(m.group(1), 16)
            for i, tok in enumerate(m.group(2).split()):
                mem[base + i] = int(tok, 16)
        return mem

    # 指令的真实形态：READ <0x…> <n> / WRITE <0x…> <n> <hex…> / ERASE <n>
    _CMD_RE = re.compile(
        r"^(READ\s+0x[0-9a-fA-F]+\s+\d+"
        r"|WRITE\s+0x[0-9a-fA-F]+\s+\d+\s+[0-9a-fA-F]+"
        r"|ERASE\s+\d+)$"
    )

    @classmethod
    def _extract_command(cls, text: str) -> str:
        """从 prompt 里挑出**真正的指令行**。

        注意不能只匹配行首的 READ/WRITE 关键字：上下文提示里有一句
        "WRITE 必须回显你写入后的值"，那会被误当成指令。
        所以要求整行严格符合指令语法。
        """
        # 优先在【本次指令】之后找
        marker = "【本次指令】"
        if marker in text:
            tail = text.split(marker, 1)[1]
            for line in tail.splitlines():
                s = line.strip()
                if cls._CMD_RE.match(s):
                    return s
        for line in text.splitlines():
            s = line.strip()
            if cls._CMD_RE.match(s):
                return s
        return text.strip()

    def _answer(self, user: str, cmd_line: str) -> str:
        parts = cmd_line.split()
        if not parts:
            return "OK"
        cmd = parts[0].upper()

        if cmd == "READ":
            addr = int(parts[1], 16)
            count = int(parts[2])
            # 有上下文就照上下文回答（这才是能工作的形态）；
            # 没有上下文时：金鱼模式只能瞎猜 0，普通模式靠自己的记忆。
            view = self._parse_memory_dump(user)
            if view:
                source: dict[int, int] = view
            elif FakeLLM.goldfish:
                source = {}
            else:
                source = FakeLLM.memory
            return bytes(source.get(addr + i, 0) for i in range(count)).hex()

        if cmd == "WRITE":
            addr = int(parts[1], 16)
            data = bytes.fromhex(parts[3])
            if not FakeLLM.goldfish:
                for i, b in enumerate(data):
                    FakeLLM.memory[addr + i] = b
            # 写照实回显——它记得自己刚写了什么，这只是回声，不代表它记住了状态
            return "OK " + data.hex()

        return "OK"

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        REQUESTS.append(body)
        FakeLLM.calls += 1

        user = body["messages"][-1]["content"]
        cmd = self._extract_command(user)
        content = self._answer(user, cmd)
        reasoning = ""
        finish = "stop"

        if FakeLLM.mode == "sloppy":
            # 故意不守格式：带解释、带空格
            content = f"好的，我记下了：{content} 😊"
        elif FakeLLM.mode == "thinking":
            # 思考型模型的典型形态：正文空，答案在 reasoning_content
            reasoning, content = f"让我看看这条指令……{content}", ""
        elif FakeLLM.mode == "budget_dead":
            # 思考吃完预算。只有 max_tokens 给够才会吐正文
            if body.get("max_tokens", 0) < 1024:
                reasoning, content, finish = "思考中……", "", "length"
        elif FakeLLM.mode == "leaky":
            # 把整段思考当正文吐出来——真实事故现场。
            # 只有升档（max_tokens 变大）后才老实回答。
            if body.get("max_tokens", 0) < 1024:
                content = (
                    "We need answer only one line. Need obey memory instruction. "
                    "Let's check the table: 0x0040 = 48. Recent ops show WRITE "
                    "0x0040 1 48. So the answer should be 48. Wait, let me "
                    "re-read the authoritative table again..."
                )
            else:
                content = self._answer(user, cmd)

        message: dict = {"content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        payload = json.dumps(
            {
                "choices": [{"message": message, "finish_reason": finish}],
                "usage": {"completion_tokens": len(reasoning) + len(content)},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_fake_llm(mode: str, goldfish: bool = False) -> tuple[HTTPServer, str]:
    FakeLLM.mode = mode
    FakeLLM.memory = {}
    FakeLLM.calls = 0
    FakeLLM.goldfish = goldfish
    srv = HTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


# ============================================================ 运行辅助


def run_box(cfg: SandboxConfig, name: str, ask_human=None, timeout=90.0):
    events: list[str] = []
    box: list = []
    prog = load_program(name)
    run_in_thread(cfg, prog, lambda ev: events.append(ev.text), box.append, ask_human)
    deadline = time.time() + timeout
    while not box and time.time() < deadline:
        time.sleep(0.02)
    return (box[0] if box else None), events


print("=" * 64)
print("模式 1：人工智能 · 本地仿 AI")
print("=" * 64)
cfg = SandboxConfig(
    memory_bytes=4096, page_size=128,
    backend_spec="local", hallucination_rate=0.0, pacing=0.0,
)
r, ev = run_box(cfg, "sort")
check("跑完并成功", r is not None and r.ok, r.message if r else "无结果")
check("排序正确", r is not None and list(r.output) == [1, 3, 5, 9], str(list(r.output)))
check("无故障", r is not None and r.faults == 0)
check("指令是严格格式", any(t.startswith(">> WRITE 0x") for t in ev))
check("回复是严格格式", any(t.startswith("<< OK ") for t in ev))

print()
print("=" * 64)
print("模式 2：人工智能 · OpenAI 兼容端点（本地假服务器）")
print("=" * 64)
srv, base = start_fake_llm("strict")
print("  假端点:", base)
cfg = SandboxConfig(
    memory_bytes=4096, page_size=128, backend_spec="openai-compat",
    pacing=0.0,
    llm=ab.LLMSettings(
        base_url=base, api_key="test-key", model="fake-model",
        temperature=0.7, extra_json='{"top_p": 0.9, "seed": 42}',
    ),
)
REQUESTS.clear()
r, ev = run_box(cfg, "sort")
check("跑完并成功", r is not None and r.ok, r.message if r else "无结果")
check("排序正确", r is not None and list(r.output) == [1, 3, 5, 9], str(list(r.output)))
check("无故障", r is not None and r.faults == 0, f"faults={r.faults if r else '?'}")
check("确实发了 HTTP 请求", len(REQUESTS) > 0, f"{len(REQUESTS)} 次")
if REQUESTS:
    b = REQUESTS[0]
    check("temperature 传对了", b.get("temperature") == 0.7, str(b.get("temperature")))
    check("model 传对了", b.get("model") == "fake-model")
    check("自定义 JSON 合并进去了", b.get("top_p") == 0.9 and b.get("seed") == 42,
          f"top_p={b.get('top_p')} seed={b.get('seed')}")
    check("system prompt 带协议", "READ" in b["messages"][0]["content"])
srv.shutdown()

print()
print("=" * 64)
print("模式 3：人工智能 · 真模型不守格式")
print("=" * 64)
srv, base = start_fake_llm("sloppy")
cfg = SandboxConfig(
    memory_bytes=4096, page_size=128, backend_spec="openai-compat",
    pacing=0.0,
    llm=ab.LLMSettings(base_url=base, api_key="k", model="fake-model"),
)
r, ev = run_box(cfg, "hello")
check("跑完了", r is not None)
check("被判定为故障", r is not None and r.faults > 0, f"faults={r.faults if r else '?'}")
check("红字里有故障提示", any("回复与纸上不符" in t or "幻觉" in t for t in ev))
srv.shutdown()

print()
print("=" * 64)
print("模式 3b：思考型模型 —— content 空、答案在 reasoning_content")
print("=" * 64)
srv, base = start_fake_llm("thinking")
cfg = SandboxConfig(
    memory_bytes=4096, page_size=128, backend_spec="openai-compat",
    pacing=0.0,
    llm=ab.LLMSettings(base_url=base, api_key="k", model="fake-thinking", max_tokens=512),
)
r, ev = run_box(cfg, "sort")
check("跑完了", r is not None, r.message if r else "无结果")
check("排序正确（靠 reasoning 回退）", r is not None and list(r.output) == [1, 3, 5, 9],
      str(list(r.output)) if r else "?")
check("不是靠升档救回来的", not any("升档" in t for t in ev))
check("没有空回复投诉", not any("(空)" in t for t in ev))

# 直接验后端层的回退标志
mem = ab.build_llm_memory(
    ab.LLMSettings(base_url=base, api_key="k", model="fake-thinking", max_tokens=512),
    "openai",
)
reply, data = mem.read(0x40, 1)
check("read 拿到字节", data == b"\x00", f"data={data!r}")
check("标记走了 reasoning", mem.last_used_reasoning, True)
check("预算记录", mem.last_budget, 512)
srv.shutdown()

print()
print("=" * 64)
print("模式 3c：思考吃完预算 —— 应自动升档救回")
print("=" * 64)
srv, base = start_fake_llm("budget_dead")
cfg = SandboxConfig(
    memory_bytes=4096, page_size=128, backend_spec="openai-compat",
    pacing=0.0,
    llm=ab.LLMSettings(base_url=base, api_key="k", model="fake-thinking", max_tokens=64),
)
r, ev = run_box(cfg, "hello")
check("跑完了", r is not None, r.message if r else "无结果")
check("靠升档救回来了", r is not None and r.faults == 0, f"faults={r.faults if r else '?'}")
check("升档有记录", any("升档" in t for t in ev), True)
check("提示了 max_tokens", any("max_tokens" in t for t in ev), True)
srv.shutdown()

print()
print("=" * 64)
print("模式 3d：上下文对照 —— none 盲猜全错 / state 照表全对")
print("=" * 64)

# 关键对照：用一个**无状态**模型——写照实回显（echo），但读没有记忆。
# 这正是"没给上下文"的真实形态：写看起来都对，读全错。
srv, base = start_fake_llm("strict", goldfish=True)


def run_hello_with_context(mode: str):
    cfg = SandboxConfig(
        memory_bytes=4096, page_size=128, backend_spec="openai-compat",
        pacing=0.0, context_mode=mode,
        llm=ab.LLMSettings(base_url=base, api_key="k", model="fake", max_tokens=512),
    )
    return run_box(cfg, "hello")


# none：模型看不到内存内容，READ 只能瞎猜 0
REQUESTS.clear()
r_none, ev_none = run_hello_with_context("none")
print(f"  context_mode=none  -> 输出={list(r_none.output)} 故障={r_none.faults}"
      f"（读侧 {r_none.read_mismatches}）")
for line in ev_none:
    if line.startswith("<<") or line.startswith(">>") or "读错了" in line:
        print(f"     {line}")
check("none 模式下读错", r_none.faults > 0, f"faults={r_none.faults}")
check("none 模式下 HI 读不回来", list(r_none.output) != [72, 73], str(list(r_none.output)))

# state：把内存内容喂给它，应该读到正确的值
r_state, ev_state = run_hello_with_context("state")
print(f"  context_mode=state -> 输出={list(r_state.output)} 故障={r_state.faults}")
check("state 模式下读对", r_state.faults == 0, f"faults={r_state.faults}")
check("state 模式下 HI 读回来了", list(r_state.output) == [72, 73], str(list(r_state.output)))

# 确认发出去的 prompt 里真的带了内存转储
with_dump = [r for r in REQUESTS if "【当前内存内容】" in r["messages"][-1]["content"]]
print(f"  带内存转储的请求：{len(with_dump)}/{len(REQUESTS)}")
check("prompt 里有内存转储", len(with_dump) > 0)
check("转储是可解析的行", "0040:" in with_dump[0]["messages"][-1]["content"])
srv.shutdown()

print()
print("=" * 64)
print("模式 3e：思考泄漏 —— 整段思考被当正文吐出")
print("=" * 64)
srv, base = start_fake_llm("leaky")
cfg = SandboxConfig(
    memory_bytes=4096, page_size=128, backend_spec="openai-compat",
    pacing=0.0,
    llm=ab.LLMSettings(base_url=base, api_key="k", model="fake-leaky", max_tokens=64),
)
r, ev = run_box(cfg, "hello")
check("跑完了", r is not None, r.message if r else "无结果")
check("泄漏被挡住、升档救回", r is not None and r.faults == 0,
      f"faults={r.faults if r else '?'}")
check("HI 读回来了", r is not None and list(r.output) == [72, 73],
      str(list(r.output)) if r else "?")
check("提示了思考泄漏", any("思考泄漏" in t for t in ev), True)
check("没有把思考文字当答案", not any("We need answer" in t for t in ev if t.startswith("<< OK")), True)
srv.shutdown()

print()
print("=" * 64)
print("模式 4：能工智人 · 仿真")
print("=" * 64)
cfg = SandboxConfig(
    memory_bytes=4096, page_size=128, backend_spec="human",
    profile=LatencyProfile.HUMAN_PEN, pacing=0.0,
)
answers = {"n": 0}


def auto_human(prompt: str) -> str:
    """替真人回答：按协议算出正确回复。"""
    answers["n"] += 1
    parts = prompt.split()
    cmd = parts[0].upper()
    if cmd == "READ":
        addr, count = int(parts[1], 16), int(parts[2])
        return bytes(0 for _ in range(count)).hex()
    if cmd == "WRITE":
        return "OK " + parts[3]
    return "OK"


r, ev = run_box(cfg, "hello", ask_human=auto_human)
check("跑完了", r is not None, r.message if r else "无结果")
check("确实问过人", answers["n"] > 0, f"{answers['n']} 次")
check("黑窗提示了这是人工模式", any("亲自扮演内存条" in t for t in ev))
check("用的是严格指令", any(t.startswith(">> READ 0x") for t in ev))

print()
print("=" * 64)
print("模式 5：能工智人 · 真实（输入桥 + 界面）")
print("=" * 64)
from ddr_ai.console import (  # noqa: E402
    EmbeddedConsole,
    EventHub,
    EventPump,
    HumanBridge,
    HumanPrompt,
)

root = tk.Tk()
root.withdraw()
hub = EventHub()
pump = EventPump(root, hub, interval_ms=10)
view = EmbeddedConsole(root)
view.pack()
view.attach(pump)
bridge = HumanBridge(hub)
active = {"manual": None}
box_done: list = []


def on_control(payload: object) -> None:
    if isinstance(payload, HumanPrompt):
        active["manual"] = view.attach_manual(bridge.submit)
        active["manual"].enable(payload.command)
    else:
        box_done.append(payload)


pump.on_control = on_control

cfg = SandboxConfig(
    memory_bytes=4096, page_size=128, backend_spec="human",
    profile=LatencyProfile.HUMAN_PEN, pacing=0.0,
)
prog = load_program("hello")
run_in_thread(cfg, prog, hub.emit, hub.push_control, ask_human=bridge.ask)

# 一边泵 UI，一边自动回答（模拟用户敲键盘）
fed = 0
deadline = time.time() + 60
while not box_done and time.time() < deadline:
    root.update()
    manual = active.get("manual")
    if manual is not None and manual._pending:
        prompt = manual.lbl.cget("text")
        p = prompt.split()
        if p and p[0].upper() == "READ":
            ans = "00" * int(p[2])
        elif p and p[0].upper() == "WRITE":
            ans = "OK " + p[3]
        else:
            ans = "OK"
        manual.entry.delete(0, "end")
        manual.entry.insert(0, ans)
        manual._submit()
        fed += 1
    time.sleep(0.01)

root.update()
text = view.text.get("1.0", "end")
check("跑完了", bool(box_done), box_done[0].message if box_done else "无结果")
check("用户确实敲了字", fed > 0, f"喂了 {fed} 次")
check("输入栏出现过", active.get("manual") is not None)
check("黑窗有对话内容", len(text.splitlines()) > 20)

root.destroy()

print()
print("=" * 64)
print("FAILS =", FAILS)
print("=" * 64)
if __name__ == "__main__":
    raise SystemExit(1 if FAILS else 0)
