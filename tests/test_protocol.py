"""协议编解码自测：直接在命令行跑，无需 pytest。

    python tests/test_protocol.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ddr_ai import ai_backend as ab  # noqa: E402

print("=== 指令生成 ===")
print("  ", ab.cmd_read(0x10, 4))
print("  ", ab.cmd_write(0x10, bytes([0, 0, 0, 0x2A])))
print("  ", ab.cmd_erase(0))

print()
print("=== 回复解码 ===")
cases = [
    ("0000002a", True),
    ("00 00 00 2a", True),
    ("0x0000002a", True),
    ("00-00-00-2a", True),
    ("00002a", True),          # 长度对不上，但解码本身合法；由调用方判长度
    ("0000002", False),        # 奇数长度
    ("zzzz", False),           # 非十六进制
    ("OK 0000002a", False),    # 带前缀，需先剥
    ("", False),
    (None, False),
    ("已记下 00 00 00 2a", False),
]
fails = 0
for text, want in cases:
    got = ab.decode_hex(text)
    ok = (got is not None) == want
    if not ok:
        fails += 1
    mark = "OK  " if ok else "FAIL"
    shown = got.hex() if got is not None else None
    print("  %s decode(%r) = %r" % (mark, text, shown))

print()
print("=== 长度校验（模拟 LLMStrictMemory / HumanMemory 的判定）===")
fails = 0
for reply, want_len, should_pass in [
    ("0000002a", 4, True),        # 纯数据：直接过
    ("00002a", 4, False),         # 长度不足：判故障
    ("OK 0000002a", 4, True),     # 带 OK 前缀：剥掉后过
    ("OK 00002a", 4, False),      # 带 OK 但长度不足：判故障
    ("OK", 4, False),             # 只有 OK 没数据：判故障
]:
    body = reply[2:].strip() if reply.upper().startswith("OK") else reply
    data = ab.decode_hex(body)
    got_len = len(data) if data is not None else None
    passes = data is not None and got_len == want_len
    ok = passes == should_pass
    if not ok:
        fails += 1
    mark = "OK  " if ok else "FAIL"
    print("  %s %-16r -> 解出长度 %-4s 结论=%s" % (
        mark, reply, got_len, "通过" if passes else "故障"
    ))

print()
print("=== salvage_hex：从夹带文字的思考内容里捞十六进制 ===")
print("（只用于 reasoning_content 回退路径，严格回复仍走 decode_hex）")
for text, want, expect in [
    ("让我看看这条指令……2a", 1, b"\x2a"),
    ("思考：地址 0x40 要写 48\n最终答案：48", 1, b"\x48"),
    ("分析了一下，答案是 00 00 00 2a", 4, b"\x00\x00\x00\x2a"),
    ("思考中……", 1, None),
    ("", 1, None),
    ("完全无关的中文", 1, None),
    ("前缀 0011223344 后缀", 2, b"\x33\x44"),   # 切末尾 2 字节
]:
    got = ab.salvage_hex(text, want)
    ok = got == expect
    if not ok:
        fails += 1
    print("  %s salvage(%r, %d) = %r" % (
        "OK  " if ok else "FAIL", text, want, got
    ))

print()
print("=== _strict_usable：思考泄漏闸门 ===")
print("防止模型把整段思考当正文吐出来、被当成答案（真实事故）")
for text, want, salvage, expect in [
    ("2a", 1, False, True),
    ("OK 2a", 1, False, True),
    ("00 00 00 2a", 4, False, True),
    ("OK", 1, False, False),
    ("OK 2a 2b", 1, False, False),
    ("We need answer only one line. Need obey memory instruction.", 1, False, False),
    ("我们只输出一行，格式 READ 回连写十六进制", 1, False, False),
    ("让我看看这条指令……2a", 1, False, False),   # 严格路径不救
    ("让我看看这条指令……2a", 1, True, True),    # 回退路径可救
    ("让我看看这条指令……但表里说 48，操作却说 01，我该信谁？", 1, True, False),
    ("OK 2a\n额外解释", 1, False, False),
    ("a" * 80, 1, False, False),
]:
    got = ab._strict_usable(text, want, allow_salvage=salvage)
    ok = got == expect
    if not ok:
        fails += 1
    print("  %s strict(%r, %d, salvage=%s) = %s" % (
        "OK  " if ok else "FAIL", text[:40], want, salvage, got
    ))

print()
print("=== looks_like_reasoning_leak ===")
for text, expect in [
    ("2a", False), ("OK 2a", False), ("ok", False),
    ("We need answer only one line.", True),
    ("我们需要回答用户最新指令", True),
    ("ok\nsecond line", True),
]:
    got = ab.looks_like_reasoning_leak(text)
    ok = got == expect
    if not ok:
        fails += 1
    print("  %s leak(%r) = %s" % ("OK  " if ok else "FAIL", text[:40], got))

print()
print("FAILS =", fails)
if __name__ == "__main__":
    raise SystemExit(1 if fails else 0)
