"""黑窗：把沙箱吐出来的对话流渲染成"终端"。

组成很简单：

    Sandbox ──emit──> EventHub（线程安全队列）──pump──> Renderer（tk.Text）

tkinter 不是线程安全的，所以沙箱线程只能往 EventHub 里丢事件，
由 UI 线程定时泵出来渲染。

"能工智人 · 真实"模式下，黑窗底部还会多一条输入栏：
沙箱发来一条 READ/WRITE 指令后阻塞等待，你在这里敲十六进制回给沙箱。
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from .sandbox import SandboxEvent

# 黑窗配色：越"出错"越刺眼
PALETTE = {
    "bus": "#4fc3f7",   # 总线指令：青
    "ai": "#e8e8e8",    # 内存条回复：白
    "sys": "#8d8d8d",   # 系统：灰
    "io": "#a5d6a7",    # IO：绿
    "fyi": "#6d6d6d",   # 附注：暗灰
    "err": "#ff5252",   # 故障/幻觉：红
    "ask": "#ffd54f",   # 等你输入：黄
}


class ControlItem:
    """队列里的非渲染项：沙箱线程用来把控制消息送回 UI 线程。

    为什么不用 root.after()？因为后台线程调 after() 会踩 tkinter
    的线程检查（main thread is not in main loop）。所以要一视同仁走队列。
    """

    __slots__ = ("payload",)

    def __init__(self, payload: object) -> None:
        self.payload = payload


class EventHub:
    """沙箱线程与 UI 线程之间的唯一通道。"""

    def __init__(self, maxsize: int = 200_000) -> None:
        self.q: queue.Queue[object] = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def _put(self, item: object) -> None:
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass
            self.dropped += 1

    def emit(self, ev: SandboxEvent) -> None:
        """由沙箱线程调用。队列满了就丢最旧的，绝不让沙箱卡住。"""
        self._put(ev)

    def push_control(self, payload: object) -> None:
        """由沙箱线程调用，投递一条控制消息（如运行结果）。"""
        self._put(ControlItem(payload))

    def drain(self, limit: int = 400) -> tuple[list[SandboxEvent], list[object]]:
        events: list[SandboxEvent] = []
        controls: list[object] = []
        for _ in range(limit):
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, ControlItem):
                controls.append(item.payload)
            else:
                events.append(item)
        return events, controls

    def empty(self) -> bool:
        return self.q.empty()


class Renderer:
    """把事件写进一个 tk.Text。只在 UI 线程调用。"""

    def __init__(self, text_widget: tk.Text, max_lines: int = 20_000) -> None:
        self.text = text_widget
        self.max_lines = max_lines
        self._lines = 0
        for kind, color in PALETTE.items():
            self.text.tag_configure(kind, foreground=color)
        self.text.tag_configure("bold", font=("Consolas", 10, "bold"))
        self.text.configure(state="disabled")

    def write_many(self, events: list[SandboxEvent]) -> None:
        if not events:
            return
        self.text.configure(state="normal")
        for ev in events:
            tag = ev.kind if ev.kind in PALETTE else "ai"
            self.text.insert("end", ev.text + "\n", tag)
        self._lines += len(events)
        if self._lines > self.max_lines:
            self.text.delete("1.0", f"{self.max_lines // 4}.0")
            self._lines -= self.max_lines // 4
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self._lines = 0


class EventPump:
    """UI 线程侧的定时泵：从 EventHub 取事件，交给当前激活的 Renderer。"""

    def __init__(self, root: tk.Misc, hub: EventHub, interval_ms: int = 40) -> None:
        self.root = root
        self.hub = hub
        self.interval_ms = interval_ms
        self.renderer: Renderer | None = None
        self.on_control: Callable[[object], None] | None = None
        self._running = False
        self._job: str | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tick()

    def stop(self) -> None:
        self._running = False
        if self._job is not None:
            try:
                self.root.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None

    def _tick(self) -> None:
        if not self._running:
            return
        try:
            events, controls = self.hub.drain()
            if events and self.renderer is not None:
                self.renderer.write_many(events)
            for payload in controls:
                cb = self.on_control
                if cb is not None:
                    try:
                        cb(payload)
                    except Exception:  # noqa: BLE001 控制回调崩了不该拖垮泵
                        pass
            self._job = self.root.after(self.interval_ms, self._tick)
        except tk.TclError:
            # 承载渲染的窗口被关掉了，体面收摊（顺手取消未决回调，
            # 免得 tkinter 在解释器退出时抱怨 invalid command name）
            self.stop()


# ---------------------------------------------------------------- 手工输入栏


class ManualInput(ttk.Frame):
    """真实模式下的输入栏：沙箱等你说"这一格是什么"。"""

    def __init__(self, master: tk.Misc, on_submit: Callable[[str], None]) -> None:
        super().__init__(master)
        self.on_submit = on_submit
        self._pending = False

        self.lbl = tk.Label(
            self,
            text="等待指令…",
            bg="#1a1a1a",
            fg="#ffd54f",
            font=("Consolas", 10),
            anchor="w",
            padx=8,
            pady=4,
        )
        self.lbl.pack(side="left", fill="x", expand=True)

        self.entry = tk.Entry(
            self,
            bg="#000000",
            fg="#e8e8e8",
            insertbackground="#ffd54f",
            font=("Consolas", 12),
            relief="flat",
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=4)
        self.entry.bind("<Return>", lambda _e: self._submit())

        self.btn = ttk.Button(self, text="提交", command=self._submit)
        self.btn.pack(side="right")

    def enable(self, prompt: str) -> None:
        """沙箱发来一条指令，进入等待状态。只在 UI 线程调用。"""
        self._pending = True
        self.lbl.configure(text=prompt)
        self.entry.configure(state="normal")
        self.btn.configure(state="normal")
        self.entry.focus_set()

    def _submit(self) -> None:
        if not self._pending:
            return
        text = self.entry.get()
        self.entry.delete(0, "end")
        self._pending = False
        self.entry.configure(state="disabled")
        self.btn.configure(state="disabled")
        self.lbl.configure(text="已提交，等待下一条指令…")
        self.on_submit(text)

    def disable(self) -> None:
        self._pending = False
        self.entry.configure(state="disabled")
        self.btn.configure(state="disabled")
        self.lbl.configure(text="（未启用）")


# ---------------------------------------------------------------- 黑窗


def make_text(parent: tk.Misc, height: int, width: int) -> ScrolledText:
    return ScrolledText(
        parent,
        height=height,
        width=width,
        bg="#0c0c0c",
        fg="#e8e8e8",
        insertbackground="#e8e8e8",
        font=("Consolas", 10),
        relief="flat",
        borderwidth=0,
        wrap="none",
    )


@dataclass
class ConsoleWindow:
    """一个独立弹出的黑窗。"""

    root: tk.Toplevel | tk.Tk
    text: ScrolledText
    renderer: Renderer
    pump: EventPump
    asks: "queue.Queue[str]" = field(default_factory=queue.Queue)
    manual: ManualInput | None = None
    _owns_pump: bool = True
    _topmost_job: str | None = None

    def focus(self) -> None:
        try:
            self.root.lift()
            self.root.attributes("-topmost", True)
        except tk.TclError:
            return
        root = self.root

        def release_topmost() -> None:
            self._topmost_job = None
            try:
                root.attributes("-topmost", False)
            except tk.TclError:
                pass  # 窗口已经没了

        try:
            self._topmost_job = root.after(300, release_topmost)
        except tk.TclError:
            self._topmost_job = None

    def attach_manual(self, on_submit: Callable[[str], None]) -> ManualInput:
        if self.manual is None:
            self.manual = ManualInput(self.root, on_submit)
            self.manual.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
        return self.manual

    def close(self) -> None:
        if self._topmost_job is not None:
            try:
                self.root.after_cancel(self._topmost_job)
            except tk.TclError:
                pass
            self._topmost_job = None
        if self._owns_pump:
            self.pump.stop()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def console_window(
    master: tk.Misc | None,
    hub: EventHub,
    pump: EventPump | None = None,
    title: str = "沙箱 · 纸介质内存",
    geometry: str = "1000x760",
    interval_ms: int = 40,
) -> ConsoleWindow:
    """弹出一个黑窗，并让它成为事件的渲染目标。

    传了 pump 就复用它（只把 renderer 换过去）。这一点很重要：
    如果另起一个泵，两个泵会抢同一个队列，事件被劈成两半。
    """
    win: tk.Toplevel | tk.Tk = (
        tk.Toplevel(master) if master is not None else tk.Tk()
    )
    win.title(title)
    win.geometry(geometry)
    win.configure(bg="#0c0c0c")
    if master is not None:
        try:
            win.transient(master)
        except tk.TclError:
            pass

    text = make_text(win, height=42, width=120)
    text.pack(fill="both", expand=True, padx=8, pady=8)
    renderer = Renderer(text)

    owns = pump is None
    if pump is None:
        pump = EventPump(win, hub, interval_ms=interval_ms)
    pump.renderer = renderer
    pump.start()

    console = ConsoleWindow(
        root=win, text=text, renderer=renderer, pump=pump, _owns_pump=owns
    )
    console.focus()
    return console


class EmbeddedConsole(ttk.Frame):
    """嵌在启动器里的黑窗视图（与独立黑窗共用同一份事件流）。"""

    def __init__(self, master: tk.Misc, height: int = 22) -> None:
        super().__init__(master)
        self.text = make_text(self, height=height, width=100)
        self.text.pack(fill="both", expand=True)
        self.renderer = Renderer(self.text)
        self.manual: ManualInput | None = None

    def attach(self, pump: EventPump) -> None:
        pump.renderer = self.renderer
        pump.start()

    def attach_manual(self, on_submit: Callable[[str], None]) -> ManualInput:
        if self.manual is None:
            self.manual = ManualInput(self, on_submit)
            self.manual.pack(side="bottom", fill="x", pady=(6, 0))
        return self.manual


# ---------------------------------------------------------------- 人机桥


@dataclass
class HumanPrompt:
    """走队列的"该你回答了"控制消息。"""

    command: str


@dataclass
class HumanBridge:
    """把"沙箱等人往黑窗里敲十六进制"这件事打包好。

    数据流（**全程不跨线程碰 tkinter**）：
        沙箱线程 ask()
            └─ hub.push_control(HumanPrompt)   ← 和"运行结果"同一套机制
                 └─ UI 线程: pump.on_control → 输入栏 enable
                      └─ 用户敲字 → submit(text) → queue
        沙箱线程在 queue 上阻塞，拿到答案再继续。

    为什么不用 root.after()？因为 root.after 底层要碰 Tcl 命令表，
    从非主线程调用会踩 tkinter 的线程检查。
    """

    hub: EventHub
    _asks: "queue.Queue[str]" = field(default_factory=queue.Queue)
    _waiting: bool = False

    def ask(self, prompt: str) -> str:
        """沙箱线程调用：把指令抛给界面，然后阻塞等答案。"""
        self._waiting = True
        self.hub.push_control(HumanPrompt(prompt))
        try:
            return self._asks.get(timeout=600.0)
        except queue.Empty:
            return ""
        finally:
            self._waiting = False

    def submit(self, text: str) -> None:
        """UI 线程调用：把用户敲的内容交回沙箱。"""
        self._asks.put(text)
