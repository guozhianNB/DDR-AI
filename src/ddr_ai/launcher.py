"""启动器：选程序、配内存、挑内存条，然后把程序关进沙箱。

界面按「谁当内存条」分两个模式：

    人工智能  ── 本地仿 AI，或接任意 OpenAI 兼容端点 / Anthropic
                （Base URL、API Key、模型、temperature、自定义请求 JSON 全在窗口里填）

    能工智人  ── 你本人当内存条
                ├─ 仿真：自动节流播放，看着它慢慢跑
                └─ 真实：黑窗弹出输入栏，你手敲十六进制

整个 GUI 只有一条线程（tkinter 主线程）碰控件；沙箱在后台线程跑，
事件经 EventHub 回流，再经 EventPump 渲染。
"""

from __future__ import annotations

import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import ai_backend as ab
from .bus import DRAM_REFERENCE, LatencyProfile
from .console import (
    ConsoleWindow,
    EmbeddedConsole,
    EventHub,
    EventPump,
    HumanBridge,
    HumanPrompt,
    console_window,
)
from .guest import GuestProgram, available_programs, search_paths
from .paper_mem import human_bytes, human_time
from .sandbox import RunResult, SandboxConfig, run_in_thread
from .settings import DEFAULT_ENV_PATH, SavedConfig, load_saved, save_saved

TITLE = "DDR-AI · 纸介质内存沙箱"

MEMORY_CHOICES = [
    ("256 B", 256),
    ("1 KiB", 1024),
    ("4 KiB", 4096),
    ("16 KiB", 16 * 1024),
    ("64 KiB", 64 * 1024),
    ("1 MiB（建议自备纸张）", 1024 * 1024),
]

PAGE_CHOICES = [("32 B", 32), ("64 B", 64), ("128 B", 128), ("256 B", 256), ("1 KiB", 1024)]

PROFILE_CHOICES = [
    (LatencyProfile.HUMAN_PEN, "人手 + 纸（誊写 ~2 bit/s）"),
    (LatencyProfile.LITERAL_0_8, "人手 + 纸（严格 0.8 bit/s，很慢）"),
    (LatencyProfile.PAPER_ROBOT, "机械臂 + 纸（快 100 倍）"),
    (LatencyProfile.AI_API, "AI 记忆（每次往返 50 ms）"),
    (LatencyProfile.AI_CONTEXT, "AI 长上下文记忆（整页入窗）"),
]

BACKEND_CHOICES = [
    ("local", "本地仿 AI（零依赖，推荐）"),
    ("openai-compat", "OpenAI 兼容端点（填 Base URL）"),
    ("anthropic", "Anthropic Claude"),
]

OUTPUT_CHOICES = [("ascii", "ASCII 文本"), ("dec", "十进制"), ("hex", "十六进制")]

PACE_CHOICES = [
    ("0.0", "全速（不等真实时间）"),
    ("0.02", "慢放 1/50"),
    ("0.06", "慢放 1/17（默认，肉眼可看）"),
    ("0.15", "慢放 ×0.15（很慢，适合演讲）"),
]

CONTEXT_CHOICES = [
    ("state", "给内存内容（读得对，推荐）"),
    ("recent", "只给最近操作记录"),
    ("none", "什么都不给（盲猜，复现灾难）"),
]

CONTEXT_SPECS = {label: spec for spec, label in CONTEXT_CHOICES}
CONTEXT_LABELS = {spec: label for spec, label in CONTEXT_CHOICES}

MODE_AI = "人工智能"
MODE_HUMAN = "能工智人"

# 后端标识 ↔ 界面文案。.env 里存的是标识。
BACKEND_SPECS = [spec for spec, _ in BACKEND_CHOICES]
BACKEND_LABELS = {spec: label for spec, label in BACKEND_CHOICES}


def backend_label(spec: str) -> str:
    return BACKEND_LABELS.get(spec, BACKEND_CHOICES[0][1])


def backend_spec(label: str) -> str:
    for spec, text in BACKEND_CHOICES:
        if text == label:
            return spec
    return "local"

CUSTOM_JSON_PLACEHOLDER = '{"top_p": 0.9, "seed": 42}'


class Launcher(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=10)
        self.master: tk.Tk = master
        self.pack(fill="both", expand=True)

        self.programs: list[GuestProgram] = available_programs()
        self.hub = EventHub()
        self.pump = EventPump(master, self.hub, interval_ms=40)
        self.bridge = HumanBridge(self.hub)
        self.sandbox = None
        self.running = False
        self.console_win: ConsoleWindow | None = None
        self.last_result: RunResult | None = None
        self.active_target = None
        """当前该往哪个视图弹输入栏（ConsoleWindow 或 EmbeddedConsole）。"""

        self._build_vars()
        self._build_widgets()
        self._refresh_program_list()
        self._apply_mode()
        self._load_saved_config()

    # ------------------------------------------------------------ 状态变量

    def _build_vars(self) -> None:
        saved = load_saved()
        self.saved: SavedConfig = saved

        self.var_program = tk.StringVar()
        self.var_memory = tk.StringVar(value=MEMORY_CHOICES[2][0])
        self.var_page = tk.StringVar(value=PAGE_CHOICES[2][0])

        self.var_mode = tk.StringVar(value=MODE_AI)

        # 人工智能 —— 初值来自 .env / 环境变量
        self.var_backend = tk.StringVar(value=backend_label(saved.backend))
        self.var_base_url = tk.StringVar(value=saved.base_url)
        self.var_api_key = tk.StringVar(value=saved.api_key)
        self.var_model = tk.StringVar(value=saved.model)
        self.var_temperature = tk.DoubleVar(value=saved.temperature)
        self.var_max_tokens = tk.StringVar(value=str(saved.max_tokens))
        self.var_context = tk.StringVar(
            value=CONTEXT_LABELS.get(saved.context_mode, CONTEXT_CHOICES[0][1])
        )
        self.var_extra_json = tk.StringVar(value=saved.extra_json)

        # 思考控制：每个字段配一个复选框，表示"要不要进请求 JSON"
        self.var_think_on = tk.BooleanVar(value=saved.thinking_enabled)
        self.var_think_off_json = tk.StringVar(value=saved.thinking_off_json)
        self.var_effort_on = tk.BooleanVar(value=saved.thinking_effort_enabled)
        self.var_effort = tk.StringVar(value=saved.thinking_effort)
        self.var_budget_on = tk.BooleanVar(value=saved.thinking_tokens_enabled)
        self.var_budget = tk.StringVar(value=str(saved.thinking_budget_tokens))
        self.var_thinkparam_on = tk.BooleanVar(value=saved.thinking_param_enabled)
        self.var_thinkparam = tk.StringVar(value=saved.thinking_param_json)
        self.var_halluc = tk.DoubleVar(value=0.0)
        self.var_save_key = tk.BooleanVar(value=True)
        """是否把 API Key 一起写进 .env。默认开——用户就是要省事。"""

        # 能工智人
        self.var_human_mode = tk.StringVar(value="仿真")
        self.var_profile = tk.StringVar(value=PROFILE_CHOICES[0][1])
        self.var_pace = tk.StringVar(value=PACE_CHOICES[2][0])

        # 通用
        self.var_output = tk.StringVar(value=OUTPUT_CHOICES[0][1])
        self.var_popup = tk.BooleanVar(value=True)
        self.var_status = tk.StringVar(value="就绪。选一个程序，然后点运行。")

    # ------------------------------------------------------------ 控件

    def _build_widgets(self) -> None:
        self._build_program_box()
        self._build_memory_box()
        self._build_mode_box()
        self._build_ai_box()
        self._build_human_box()
        self._build_common_box()
        self._build_action_bar()
        self._build_console_box()
        self._build_status_bar()

    # ---- 1. 程序 ----

    def _build_program_box(self) -> None:
        pad = {"padx": 4, "pady": 3}
        box = ttk.LabelFrame(self, text=" 1. 选择要运行的程序 ", padding=8)
        box.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        self.cmb_program = ttk.Combobox(
            box, textvariable=self.var_program, state="readonly", width=46
        )
        self.cmb_program.grid(row=0, column=0, sticky="w", **pad)
        self.cmb_program.bind("<<ComboboxSelected>>", lambda _e: self._show_program_info())
        ttk.Button(box, text="打开目录…", command=self._browse_dir).grid(row=0, column=1, **pad)
        ttk.Button(box, text="查看源码", command=self._show_source).grid(row=0, column=2, **pad)
        self.lbl_program_info = ttk.Label(box, text="", foreground="#555")
        self.lbl_program_info.grid(row=1, column=0, columnspan=4, sticky="w", **pad)

    # ---- 2. 内存 ----

    def _build_memory_box(self) -> None:
        pad = {"padx": 4, "pady": 3}
        box = ttk.LabelFrame(self, text=" 2. 给它多少内存 ", padding=8)
        box.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        ttk.Label(box, text="容量").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(
            box, textvariable=self.var_memory, values=[c[0] for c in MEMORY_CHOICES],
            state="readonly", width=18,
        ).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(box, text="页大小").grid(row=0, column=2, sticky="w", **pad)
        ttk.Combobox(
            box, textvariable=self.var_page, values=[c[0] for c in PAGE_CHOICES],
            state="readonly", width=10,
        ).grid(row=0, column=3, sticky="w", **pad)
        ttk.Label(
            box,
            text="提示：页大小决定「改 1 个字节要重誊多少纸」。页越大，写放大越狠。",
            foreground="#555",
        ).grid(row=1, column=0, columnspan=4, sticky="w", **pad)

    # ---- 3. 模式 ----

    def _build_mode_box(self) -> None:
        box = ttk.LabelFrame(self, text=" 3. 谁来当内存条 ", padding=8)
        box.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        ttk.Radiobutton(
            box, text="人工智能（让模型记忆内容）", value=MODE_AI,
            variable=self.var_mode, command=self._apply_mode,
        ).grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Radiobutton(
            box, text="能工智人（你自己上）", value=MODE_HUMAN,
            variable=self.var_mode, command=self._apply_mode,
        ).grid(row=0, column=1, sticky="w", padx=4, pady=3)

    # ---- 3a. 人工智能 ----

    def _build_ai_box(self) -> None:
        pad = {"padx": 4, "pady": 3}
        self.box_ai = ttk.LabelFrame(self, text=" 3a. 人工智能 · 接口配置 ", padding=8)
        self.box_ai.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(0, 6))

        ttk.Label(self.box_ai, text="后端").grid(row=0, column=0, sticky="w", **pad)
        self.cmb_backend = ttk.Combobox(
            self.box_ai, textvariable=self.var_backend,
            values=[c[1] for c in BACKEND_CHOICES], state="readonly", width=30,
        )
        self.cmb_backend.grid(row=0, column=1, sticky="w", **pad)
        self.cmb_backend.bind("<<ComboboxSelected>>", lambda _e: self._on_backend_change())

        ttk.Label(self.box_ai, text="模型").grid(row=0, column=2, sticky="w", **pad)
        self.ent_model = ttk.Entry(self.box_ai, textvariable=self.var_model, width=26)
        self.ent_model.grid(row=0, column=3, sticky="w", **pad)
        self.ent_model.bind("<FocusOut>", lambda _e: self._persist(quiet=True))

        ttk.Label(self.box_ai, text="Base URL").grid(row=1, column=0, sticky="w", **pad)
        self.ent_base = ttk.Entry(self.box_ai, textvariable=self.var_base_url, width=58)
        self.ent_base.grid(row=1, column=1, columnspan=3, sticky="we", **pad)
        self.ent_base.bind("<FocusOut>", lambda _e: self._persist(quiet=True))

        ttk.Label(self.box_ai, text="API Key").grid(row=2, column=0, sticky="w", **pad)
        self.ent_key = ttk.Entry(
            self.box_ai, textvariable=self.var_api_key, width=58, show="•"
        )
        self.ent_key.grid(row=2, column=1, sticky="we", **pad)
        self.ent_key.bind("<FocusOut>", lambda _e: self._persist(quiet=True))
        keybar = ttk.Frame(self.box_ai)
        keybar.grid(row=2, column=2, columnspan=2, sticky="w", **pad)
        self.chk_show_key = ttk.Checkbutton(
            keybar, text="显示", command=self._toggle_key
        )
        self.chk_show_key.pack(side="left", padx=(0, 8))
        self.chk_save_key = ttk.Checkbutton(
            keybar, text="记住", variable=self.var_save_key,
            command=lambda: self._persist(quiet=True),
        )
        self.chk_save_key.pack(side="left")

        ttk.Label(self.box_ai, text="temperature").grid(row=3, column=0, sticky="w", **pad)
        self.scale_temp = ttk.Scale(
            self.box_ai, from_=0.0, to=2.0, variable=self.var_temperature,
            orient="horizontal", length=180,
            command=lambda _v: self._update_temp_label(),
        )
        self.scale_temp.grid(row=3, column=1, sticky="w", **pad)
        self.lbl_temp = ttk.Label(self.box_ai, text="0.00")
        self.lbl_temp.grid(row=3, column=2, sticky="w", **pad)
        ttk.Label(self.box_ai, text="max_tokens").grid(
            row=3, column=3, sticky="e", **pad
        )
        self.ent_max_tokens = ttk.Entry(
            self.box_ai, textvariable=self.var_max_tokens, width=8
        )
        self.ent_max_tokens.grid(row=3, column=4, sticky="w", **pad)
        self.ent_max_tokens.bind("<FocusOut>", lambda _e: self._persist(quiet=True))

        ttk.Label(self.box_ai, text="自定义请求 JSON").grid(row=4, column=0, sticky="nw", **pad)
        self.ent_extra = ttk.Entry(self.box_ai, textvariable=self.var_extra_json, width=58)
        self.ent_extra.grid(row=4, column=1, columnspan=4, sticky="we", **pad)
        self.ent_extra.bind("<FocusOut>", lambda _e: self._persist(quiet=True))

        ttk.Label(self.box_ai, text="上下文").grid(row=6, column=0, sticky="w", **pad)
        self.cmb_context = ttk.Combobox(
            self.box_ai, textvariable=self.var_context,
            values=[c[1] for c in CONTEXT_CHOICES], state="readonly", width=32,
        )
        self.cmb_context.grid(row=6, column=1, columnspan=2, sticky="w", **pad)
        self.cmb_context.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_context_change()
        )
        self.lbl_context_hint = ttk.Label(
            self.box_ai, text="", foreground="#555"
        )
        self.lbl_context_hint.grid(row=6, column=3, columnspan=2, sticky="w", **pad)

        ttk.Label(self.box_ai, text="幻觉率").grid(row=7, column=0, sticky="w", **pad)
        self.scale_halluc = ttk.Scale(
            self.box_ai, from_=0.0, to=1.0, variable=self.var_halluc,
            orient="horizontal", length=220,
            command=lambda _v: self._update_halluc_label(),
        )
        self.scale_halluc.grid(row=7, column=1, sticky="w", **pad)
        self.lbl_halluc = ttk.Label(self.box_ai, text="")
        self.lbl_halluc.grid(row=7, column=2, columnspan=3, sticky="w", **pad)

        self._build_thinking_panel()

        self.lbl_ai_hint = ttk.Label(self.box_ai, text="", foreground="#555")
        self.lbl_ai_hint.grid(row=10, column=0, columnspan=4, sticky="w", **pad)
        ttk.Button(
            self.box_ai, text="打开配置(.env)", command=self._open_env_file
        ).grid(row=10, column=4, sticky="e", **pad)

        for c in range(5):
            self.box_ai.grid_columnconfigure(c, weight=1)

    # ---- 3a-2. 思考控制 ----

    def _build_thinking_panel(self) -> None:
        """每个字段前一个复选框：勾了才把该字段写进请求 JSON。

        这么设计是因为**各家模型的思考字段名、以及是否支持，都不一样**，
        硬编码必然在某个模型上报"未知字段"。
        """
        pad = {"padx": 4, "pady": 2}
        box = ttk.LabelFrame(
            self.box_ai, text=" 思考控制（勾选 = 该字段进入请求 JSON） ", padding=6
        )
        box.grid(row=9, column=0, columnspan=5, sticky="ew", pady=(4, 4))
        for c in range(6):
            box.grid_columnconfigure(c, weight=1)

        # 总开关
        ttk.Checkbutton(
            box, text="思考总开关", variable=self.var_think_on,
            command=self._on_thinking_change,
        ).grid(row=0, column=0, sticky="w", **pad)
        self.lbl_think_state = ttk.Label(box, text="", foreground="#555")
        self.lbl_think_state.grid(row=0, column=1, columnspan=5, sticky="w", **pad)

        # 关闭时发什么
        ttk.Label(box, text="关闭时发").grid(row=1, column=0, sticky="w", **pad)
        self.ent_off_json = ttk.Entry(
            box, textvariable=self.var_think_off_json, width=44
        )
        self.ent_off_json.grid(row=1, column=1, columnspan=4, sticky="we", **pad)
        self.ent_off_json.bind("<FocusOut>", lambda _e: self._persist(quiet=True))
        ttk.Label(
            box, text="留空 = 完全不提这个字段", foreground="#555"
        ).grid(row=1, column=5, sticky="w", **pad)

        # 强度
        self.chk_effort = ttk.Checkbutton(
            box, text="思考强度", variable=self.var_effort_on,
            command=self._on_thinking_change,
        )
        self.chk_effort.grid(row=2, column=0, sticky="w", **pad)
        self.cmb_effort = ttk.Combobox(
            box, textvariable=self.var_effort,
            values=["minimal", "low", "medium", "high", "xhigh"],
            width=10,
        )
        self.cmb_effort.grid(row=2, column=1, sticky="w", **pad)
        self.cmb_effort.bind("<<ComboboxSelected>>", lambda _e: self._persist(quiet=True))
        ttk.Label(
            box, text="字段 reasoning_effort；各家档位名可能不同，可手打",
            foreground="#555",
        ).grid(row=2, column=2, columnspan=4, sticky="w", **pad)

        # 预算
        self.chk_budget = ttk.Checkbutton(
            box, text="思考预算", variable=self.var_budget_on,
            command=self._on_thinking_change,
        )
        self.chk_budget.grid(row=3, column=0, sticky="w", **pad)
        self.ent_budget = ttk.Entry(box, textvariable=self.var_budget, width=10)
        self.ent_budget.grid(row=3, column=1, sticky="w", **pad)
        self.ent_budget.bind("<FocusOut>", lambda _e: self._persist(quiet=True))
        ttk.Label(
            box, text="token 数；Anthropic 会包成 thinking.budget_tokens",
            foreground="#555",
        ).grid(row=3, column=2, columnspan=4, sticky="w", **pad)

        # 自定义参数
        self.chk_thinkparam = ttk.Checkbutton(
            box, text="自定义思考参数", variable=self.var_thinkparam_on,
            command=self._on_thinking_change,
        )
        self.chk_thinkparam.grid(row=4, column=0, sticky="w", **pad)
        self.ent_thinkparam = ttk.Entry(
            box, textvariable=self.var_thinkparam, width=44
        )
        self.ent_thinkparam.grid(row=4, column=1, columnspan=4, sticky="we", **pad)
        self.ent_thinkparam.bind("<FocusOut>", lambda _e: self._persist(quiet=True))
        ttk.Label(
            box, text="适配各家不同字段名", foreground="#555"
        ).grid(row=4, column=5, sticky="w", **pad)

    def _on_thinking_change(self) -> None:
        self._update_thinking_state()
        self._persist(quiet=True)

    def _update_thinking_state(self) -> None:
        """总开关关掉时，下面三项的输入框变灰——因为整个字段都不会发。"""
        on = bool(self.var_think_on.get())
        sub = "normal" if on else "disabled"
        for w in (self.ent_budget, self.ent_thinkparam):
            w.configure(state=sub)
        self.cmb_effort.configure(state=sub)
        for w in (self.chk_effort, self.chk_budget, self.chk_thinkparam):
            w.configure(state=sub)

        if not on:
            what = self.var_think_off_json.get().strip()
            self.lbl_think_state.configure(
                text=(
                    f"已关闭：请求里带上 {what}" if what
                    else "已关闭：请求里完全不含思考字段（注意：有的模型照样会想）"
                )
            )
            return

        bits = []
        if self.var_effort_on.get():
            bits.append(f"reasoning_effort={self.var_effort.get()}")
        if self.var_budget_on.get():
            bits.append(f"thinking_budget_tokens={self.var_budget.get()}")
        if self.var_thinkparam_on.get():
            bits.append(self.var_thinkparam.get().strip() or "(空)")
        self.lbl_think_state.configure(
            text="将发送：" + ("、".join(bits) if bits else "无思考字段（可能被默认值接管）")
        )

    # ---- 3b. 能工智人 ----

    def _build_human_box(self) -> None:
        pad = {"padx": 4, "pady": 3}
        self.box_human = ttk.LabelFrame(self, text=" 3b. 能工智人 · 你来当内存条 ", padding=8)
        self.box_human.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(0, 6))

        ttk.Radiobutton(
            self.box_human, text="仿真（自动跑，我来看着）", value="仿真",
            variable=self.var_human_mode, command=self._apply_mode,
        ).grid(row=0, column=0, sticky="w", **pad)
        ttk.Radiobutton(
            self.box_human, text="真实（我自己手敲十六进制）", value="真实",
            variable=self.var_human_mode, command=self._apply_mode,
        ).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(self.box_human, text="机构档位").grid(row=1, column=0, sticky="w", **pad)
        self.cmb_profile = ttk.Combobox(
            self.box_human, textvariable=self.var_profile,
            values=[c[1] for c in PROFILE_CHOICES], state="readonly", width=34,
        )
        self.cmb_profile.grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(self.box_human, text="节流").grid(row=1, column=2, sticky="w", **pad)
        self.cmb_pace = ttk.Combobox(
            self.box_human, textvariable=self.var_pace,
            values=[c[0] for c in PACE_CHOICES], state="readonly", width=12,
        )
        self.cmb_pace.grid(row=1, column=3, sticky="w", **pad)

        self.lbl_human_hint = ttk.Label(self.box_human, text="", foreground="#555")
        self.lbl_human_hint.grid(row=2, column=0, columnspan=4, sticky="w", **pad)

    # ---- 4. 通用 ----

    def _build_common_box(self) -> None:
        pad = {"padx": 4, "pady": 3}
        box = ttk.LabelFrame(self, text=" 4. 输出与观感 ", padding=8)
        box.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        ttk.Label(box, text="输出格式").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(
            box, textvariable=self.var_output, values=[c[1] for c in OUTPUT_CHOICES],
            state="readonly", width=12,
        ).grid(row=0, column=1, sticky="w", **pad)
        ttk.Checkbutton(
            box, text="弹出独立黑窗（沙箱对话在那边刷）", variable=self.var_popup
        ).grid(row=0, column=2, columnspan=2, sticky="w", **pad)

    # ---- 操作条 ----

    def _build_action_bar(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        self.btn_run = ttk.Button(bar, text="▶  运行", command=self.on_run)
        self.btn_run.pack(side="left", padx=4)
        self.btn_stop = ttk.Button(bar, text="■  中止", command=self.on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        ttk.Button(bar, text="清空黑窗", command=self.on_clear).pack(side="left", padx=4)
        ttk.Button(bar, text="查看指令流", command=self.on_show_protocol).pack(side="left", padx=4)
        ttk.Button(bar, text="性能对比", command=self.on_summary).pack(side="left", padx=4)

    def _build_console_box(self) -> None:
        box = ttk.LabelFrame(self, text=" 沙箱对话（黑窗） ", padding=6)
        box.grid(row=7, column=0, columnspan=4, sticky="nsew", pady=(0, 6))
        self.embedded = EmbeddedConsole(box, height=18)
        self.embedded.pack(fill="both", expand=True)
        self.embedded.attach(self.pump)
        self.grid_rowconfigure(7, weight=1)

    def _build_status_bar(self) -> None:
        ttk.Separator(self, orient="horizontal").grid(
            row=8, column=0, columnspan=4, sticky="ew"
        )
        ttk.Label(self, textvariable=self.var_status, anchor="w").grid(
            row=9, column=0, columnspan=4, sticky="ew", pady=(4, 0)
        )
        for c in range(4):
            self.grid_columnconfigure(c, weight=1)

    # ------------------------------------------------------------ 模式切换

    def _current_backend(self) -> str:
        return self._lookup(BACKEND_CHOICES, self.var_backend.get(), "local")

    def _apply_mode(self) -> None:
        """按模式显隐对应的配置区，并把不适用的控件禁用掉。"""
        ai_mode = self.var_mode.get() == MODE_AI

        if ai_mode:
            self.box_ai.grid()
            self.box_human.grid_remove()
        else:
            self.box_ai.grid_remove()
            self.box_human.grid()

        if not ai_mode:
            human_real = self.var_human_mode.get() == "真实"
            # 真实模式下节流没意义（等人比节流慢多了）
            self.cmb_pace.configure(state="disabled" if human_real else "readonly")
            self.cmb_profile.configure(state="readonly")
            self.lbl_human_hint.configure(
                text=(
                    "真实模式：沙箱每发一条指令就停下来，黑窗底部会出现输入栏，"
                    "你把十六进制敲进去它才继续。没有节流——你的手速就是节流。"
                    if human_real
                    else "仿真模式：不需要你动手，按下面的节流档自动播放。"
                )
            )
            return

        backend = self._current_backend()
        uses_llm = backend in ("openai-compat", "anthropic")
        state = "normal" if uses_llm else "disabled"
        for w in (
            self.ent_base,
            self.ent_key,
            self.ent_model,
            self.ent_extra,
            self.ent_max_tokens,
        ):
            w.configure(state=state)
        self.scale_temp.configure(state=state)
        self.chk_show_key.configure(state=state)
        self.chk_save_key.configure(state=state)
        self.cmb_context.configure(state="readonly" if uses_llm else "disabled")
        self._update_context_hint()

        # 思考控制整体只在接了真模型时可用
        for w in (
            self.chk_effort, self.chk_budget, self.chk_thinkparam,
            self.ent_off_json,
        ):
            w.configure(state="normal" if uses_llm else "disabled")
        if not uses_llm:
            self.cmb_effort.configure(state="disabled")
            self.ent_budget.configure(state="disabled")
            self.ent_thinkparam.configure(state="disabled")
        else:
            self._update_thinking_state()

        # 幻觉率只对本地仿 AI 有意义
        halluc_state = "disabled" if uses_llm else "normal"
        self.scale_halluc.configure(state=halluc_state)

        if not uses_llm:
            self.lbl_ai_hint.configure(
                text="本地仿 AI 不需要任何配置，回答是确定性的。幻觉率在这里是有意义的。"
            )
        else:
            endpoint = "Anthropic Messages API" if backend == "anthropic" else "OpenAI 兼容"
            note = ""
            if self._looks_like_thinking_model(self.var_model.get()):
                note = "　⚠ 像是思考型模型：max_tokens 太小会让它想完没力气回答。"
            self.lbl_ai_hint.configure(
                text=(
                    f"将按 {endpoint} 发请求。真模型每次回答都可能不同，"
                    f"幻觉率滑杆对它无效。{note}"
                )
            )
        self._update_temp_label()

    @staticmethod
    def _looks_like_thinking_model(model: str) -> bool:
        """粗判思考型模型——它们最容易因 max_tokens 不足而返回空内容。"""
        m = model.lower()
        return any(k in m for k in ("reason", "thinking", "flash", "r1", "o1", "o3"))

    def _on_context_change(self) -> None:
        self._update_context_hint()
        self._persist(quiet=True)

    def _update_context_hint(self) -> None:
        spec = CONTEXT_SPECS.get(self.var_context.get(), "state")
        notes = {
            "state": "把内存内容随每次请求发给它——这是它能读对的前提",
            "recent": "只给最近几次收发，省 token 但读全量状态时仍可能错",
            "none": "什么都不给：无状态模型读内存只能瞎猜，READ 必然全错",
        }
        self.lbl_context_hint.configure(text=notes.get(spec, ""))

    def _toggle_key(self) -> None:
        self.ent_key.configure(show="" if self.chk_show_key.instate(["selected"]) else "•")

    def _on_backend_change(self) -> None:
        """切换后端时顺手带出该家的默认地址（只在用户还没改过时才换）。"""
        backend = self._current_backend()
        current = self.var_base_url.get().strip()
        known = {"https://api.openai.com/v1", "https://api.anthropic.com/v1"}
        if backend == "anthropic" and current in known:
            self.var_base_url.set("https://api.anthropic.com/v1")
        elif backend == "openai-compat" and current in known:
            self.var_base_url.set("https://api.openai.com/v1")
        self._apply_mode()
        self._persist(quiet=True)

    # ------------------------------------------------------------ 配置持久化

    def _load_saved_config(self) -> None:
        """启动时把 `.env` 里的值带出来，并提示来源。"""
        saved = self.saved
        self._update_thinking_state()
        if saved.from_file:
            origin = str(saved.env_path)
        else:
            origin = "未找到 .env，当前值来自环境变量或默认值"
        self.lbl_ai_hint.configure(text=f"配置来源：{origin}")

    def _persist(self, quiet: bool = False) -> bool:
        """把当前接口配置写回 `.env`。

        只记后端与连接参数；程序、内存、节流之类的每次运行本来就随手指拨，
        没必要持久化。
        """
        save_key = bool(self.var_save_key.get())
        api_key = self.var_api_key.get().strip() if save_key else ""
        llm = self._collect_llm()
        try:
            path = save_saved(
                backend=self._current_backend(),
                base_url=self.var_base_url.get().strip(),
                model=self.var_model.get().strip(),
                api_key=api_key,
                temperature=float(self.var_temperature.get()),
                extra_json=self.var_extra_json.get().strip(),
                max_tokens=llm.max_tokens,
                context_mode=CONTEXT_SPECS.get(self.var_context.get(), "state"),
                thinking_enabled=llm.thinking_enabled,
                thinking_off_json=llm.thinking_off_json,
                thinking_effort_enabled=llm.thinking_effort_enabled,
                thinking_effort=llm.thinking_effort,
                thinking_tokens_enabled=llm.thinking_tokens_enabled,
                thinking_budget_tokens=llm.thinking_budget_tokens,
                thinking_param_enabled=llm.thinking_param_enabled,
                thinking_param_json=llm.thinking_param_json,
            )
        except OSError as exc:
            if not quiet:
                self.var_status.set(f"配置保存失败：{exc}")
            return False
        self.saved = load_saved()
        self.lbl_ai_hint.configure(text=f"配置来源：{path}")
        if not quiet:
            self.var_status.set(f"配置已保存到 {path}")
        return True

    def _open_env_file(self) -> None:
        """用系统默认程序打开 `.env`（没有就先建一个）。"""
        path = DEFAULT_ENV_PATH
        if not path.exists():
            self._persist(quiet=True)
        if not path.exists():
            messagebox.showerror(TITLE, f"无法创建配置文件：{path}")
            return
        try:
            os.startfile(path)  # type: ignore[attr-defined]  Windows
        except AttributeError:
            import subprocess

            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(path)])
        except OSError as exc:
            messagebox.showerror(TITLE, f"打开失败：{exc}\n\n路径：{path}")

    def _update_temp_label(self) -> None:
        self.lbl_temp.configure(text=f"{self.var_temperature.get():.2f}")

    def _update_halluc_label(self) -> None:
        pct = self.var_halluc.get()
        if pct <= 0.001:
            note = "每次都老实回答"
        elif pct < 0.15:
            note = "偶尔记错，程序大概还能跑"
        elif pct < 0.4:
            note = "经常记错，结果基本不可信"
        else:
            note = "基本靠编，跑不跑得完看运气"
        self.lbl_halluc.configure(text=f"{pct:.0%}   —— {note}")

    # ------------------------------------------------------------ 程序列表

    def _refresh_program_list(self) -> None:
        names = [p.name for p in self.programs]
        self.cmb_program.configure(values=names)
        if names and not self.var_program.get():
            self.var_program.set(names[0])
        self._show_program_info()

    def _selected_program(self) -> GuestProgram | None:
        name = self.var_program.get()
        for p in self.programs:
            if p.name == name:
                return p
        return None

    def _show_program_info(self) -> None:
        p = self._selected_program()
        if p is None:
            self.lbl_program_info.configure(text="")
            return
        self.lbl_program_info.configure(
            text=f"{len(p.code)} 条指令 · 需要约 {human_bytes(p.mem_footprint)} 内存 · "
                 f"{len(p.data)} 个预置字节"
        )

    def _browse_dir(self) -> None:
        d = filedialog.askdirectory(title="选择包含 .asm 的目录")
        if not d:
            return
        extra = search_paths(d)
        if not extra:
            messagebox.showinfo(TITLE, "这个目录里没有 .asm 文件。")
            return
        existing = {p.name for p in self.programs}
        self.programs.extend(p for p in extra if p.name not in existing)
        self._refresh_program_list()
        self.var_status.set(f"已从 {d} 加载 {len(extra)} 个程序。")

    def _show_source(self) -> None:
        p = self._selected_program()
        if p is None:
            return
        win = tk.Toplevel(self.master)
        win.title(f"源码 · {p.name}.asm")
        win.geometry("760x560")
        txt = tk.Text(win, bg="#1e1e1e", fg="#dcdcdc", font=("Consolas", 10), wrap="none")
        txt.pack(fill="both", expand=True, padx=6, pady=6)
        txt.insert("1.0", p.source)
        txt.configure(state="disabled")

    # ------------------------------------------------------------ 配置收集

    def _lookup(self, choices, label: str, default):
        for key, text in choices:
            if text == label:
                return key
        return default

    def _collect_llm(self) -> ab.LLMSettings:
        try:
            max_tokens = int(self.var_max_tokens.get().strip() or "512")
        except ValueError:
            max_tokens = 512
        max_tokens = min(max(max_tokens, 1), 32_768)

        try:
            budget = int(self.var_budget.get().strip() or "2048")
        except ValueError:
            budget = 2048
        budget = min(max(budget, 1), 200_000)

        return ab.LLMSettings(
            base_url=self.var_base_url.get().strip(),
            api_key=self.var_api_key.get().strip(),
            model=self.var_model.get().strip() or "gpt-4o-mini",
            temperature=float(self.var_temperature.get()),
            max_tokens=max_tokens,
            thinking_enabled=bool(self.var_think_on.get()),
            thinking_off_json=self.var_think_off_json.get().strip(),
            thinking_effort_enabled=bool(self.var_effort_on.get()),
            thinking_effort=self.var_effort.get().strip() or "medium",
            thinking_tokens_enabled=bool(self.var_budget_on.get()),
            thinking_budget_tokens=budget,
            thinking_param_enabled=bool(self.var_thinkparam_on.get()),
            thinking_param_json=self.var_thinkparam.get().strip(),
            extra_json=self.var_extra_json.get().strip(),
        )

    def _collect_config(self) -> SandboxConfig:
        memory = self._lookup(MEMORY_CHOICES, self.var_memory.get(), 4096)
        page = self._lookup(PAGE_CHOICES, self.var_page.get(), 128)
        output = self._lookup(OUTPUT_CHOICES, self.var_output.get(), "ascii")

        if self.var_mode.get() == MODE_AI:
            backend = self._current_backend()
            profile = LatencyProfile.AI_API
            pacing = 0.06
            halluc = float(self.var_halluc.get())
            context = CONTEXT_SPECS.get(self.var_context.get(), "state")
        else:
            backend = "human"
            context = "none"
            profile = self._lookup(
                PROFILE_CHOICES, self.var_profile.get(), LatencyProfile.HUMAN_PEN
            )
            human_real = self.var_human_mode.get() == "真实"
            # 真实模式下人在等，节流毫无意义，直接给 0
            try:
                pacing = 0.0 if human_real else float(self.var_pace.get())
            except ValueError:
                pacing = 0.06
            halluc = 0.0

        return SandboxConfig(
            memory_bytes=memory,
            page_size=min(page, memory),
            profile=profile,
            backend_spec=backend,
            llm=self._collect_llm(),
            hallucination_rate=halluc,
            pacing=pacing,
            output_style=output,
            context_mode=context,
        )

    def _uses_human_real(self) -> bool:
        return (
            self.var_mode.get() == MODE_HUMAN
            and self.var_human_mode.get() == "真实"
        )

    def _validate(self, config: SandboxConfig) -> bool:
        """开跑前的拦截。返回 False 表示别跑。"""
        if config.backend_spec in ("openai-compat", "anthropic"):
            s = config.llm
            if not s.base_url:
                messagebox.showerror(TITLE, "请填写 Base URL。")
                return False
            if not s.api_key:
                messagebox.showerror(TITLE, "请填写 API Key。")
                return False
            if s.extra_json:
                try:
                    parsed = json.loads(s.extra_json)
                except json.JSONDecodeError as exc:
                    messagebox.showerror(TITLE, f"自定义请求 JSON 不是合法 JSON：\n{exc}")
                    return False
                if not isinstance(parsed, dict):
                    messagebox.showerror(TITLE, "自定义请求 JSON 必须是一个对象。")
                    return False

        program = self._selected_program()
        if program is not None and config.memory_bytes < program.mem_footprint:
            if not messagebox.askyesno(
                TITLE,
                f"这个程序大约需要 {human_bytes(program.mem_footprint)} 内存，"
                f"但你只给了 {human_bytes(config.memory_bytes)}。\n\n"
                "确定要让它撞段错误吗？（这也是个结果）",
            ):
                return False
        return True

    # ------------------------------------------------------------ 运行

    def on_run(self) -> None:
        if self.running:
            return
        program = self._selected_program()
        if program is None:
            messagebox.showwarning(TITLE, "先选一个程序。")
            return

        # 先落盘，再去校验/运行——这样即使校验拦下了，用户填的接口也记住了
        self._persist(quiet=True)

        config = self._collect_config()
        if not self._validate(config):
            return

        human_real = self._uses_human_real()
        if human_real:
            n = self._estimate_accesses(program)
            if not messagebox.askyesno(
                TITLE,
                f"真实模式下，沙箱每访问一次内存都会停下来等你输入。\n\n"
                f"大概会有 {n} 次左右（例如 sort 约 106 次）。\n"
                f"你要手敲 {n} 遍十六进制。确定开始吗？",
            ):
                return

        self.embedded.renderer.clear()
        self.pump.start()

        target = None
        if self.var_popup.get() or human_real:
            try:
                self.console_win = console_window(
                    self.master, self.hub, pump=self.pump, interval_ms=40
                )
                target = self.console_win
            except Exception as exc:  # noqa: BLE001 弹窗失败不该阻止运行
                self.var_status.set(f"独立黑窗打开失败（{exc}），改用内嵌视图。")
                self.console_win = None
                target = self.embedded
                self.embedded.attach(self.pump)
        else:
            if self.console_win is not None:
                self.console_win.close()
                self.console_win = None
            target = self.embedded
            self.embedded.attach(self.pump)

        self.active_target = target
        if human_real and target is not None:
            # 先把输入栏建出来，免得第一条指令到了才开始布局
            try:
                target.attach_manual(self.bridge.submit)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

        self.running = True
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.var_status.set(f"正在运行 {program.name}…（沙箱已启动，请稍候）")

        hub = self.hub
        hub.dropped = 0

        def done(result: RunResult) -> None:
            # 关键：这里还在沙箱线程里，不能碰任何 tkinter 控件，
            # 也不能调 root.after()（会踩线程检查）。走队列，由泵派发。
            hub.push_control(result)

        # 控制消息路由：既收运行结果，也收"该你回答了"
        self.pump.on_control = self._on_control
        self.sandbox = run_in_thread(
            config,
            program,
            hub.emit,
            done,
            ask_human=self.bridge.ask if human_real else None,
        )

    def _on_control(self, payload: object) -> None:
        """UI 线程：派发来自沙箱线程的控制消息。"""
        if isinstance(payload, RunResult):
            self._on_finished(payload)
        elif isinstance(payload, HumanPrompt):
            self._on_human_prompt(payload)

    def _on_human_prompt(self, prompt: HumanPrompt) -> None:
        target = self.active_target
        if target is None:
            return
        try:
            manual = target.attach_manual(self.bridge.submit)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return
        manual.enable(prompt.command)
        if self.console_win is not None:
            self.console_win.focus()

    def _estimate_accesses(self, program: GuestProgram) -> int:
        """粗估访问次数：按指令数乘个经验系数，给用户一个心理准备。"""
        loads = sum(1 for i in program.code if i.op in ("LOAD", "LDI"))
        stores = sum(1 for i in program.code if i.op in ("STORE", "STI", "SWAP"))
        data = len(program.data)
        return max(1, (loads + stores + data) * 3)

    def _on_finished(self, result: RunResult) -> None:
        self.running = False
        self.last_result = result
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        verdict = "成功" if result.ok else "失败"
        # 程序输出已经实时刷在黑窗里了，这里不再重复，只报指标
        self.var_status.set(
            f"{result.program}：{verdict} · {result.message} · "
            f"{len(result.output)} 字节输出 · 访问 {result.accesses} 次 · "
            f"仿真 {human_time(result.sim_seconds)} · "
            f"等效 {result.effective_bps:.4f} bit/s · 故障 {result.faults} 次"
        )

    def shutdown(self) -> None:
        """体面收摊：停泵、中止沙箱。窗口销毁前调用。"""
        try:
            self.pump.stop()
        except Exception:  # noqa: BLE001
            pass
        if self.sandbox is not None:
            self.sandbox.stop()
        if self.console_win is not None:
            self.console_win.close()
            self.console_win = None

    def on_stop(self) -> None:
        if self.sandbox is not None:
            self.sandbox.stop()
        # 真实模式下沙箱可能正卡在等人输入，补一个空回答把它放出来
        self.bridge.submit("")
        self.var_status.set("已请求中止……")
        self.btn_stop.configure(state="disabled")

    def on_clear(self) -> None:
        self.embedded.renderer.clear()
        if self.console_win is not None:
            self.console_win.renderer.clear()

    # ------------------------------------------------------------ 指令流说明

    def on_show_protocol(self) -> None:
        win = tk.Toplevel(self.master)
        win.title("沙箱 ↔ 内存条 指令协议")
        win.geometry("720x520")
        txt = tk.Text(win, bg="#0c0c0c", fg="#e8e8e8", font=("Consolas", 10), wrap="word")
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", PROTOCOL_DOC)
        txt.configure(state="disabled")

    # ------------------------------------------------------------ 性能对比

    def on_summary(self) -> None:
        r = self.last_result
        if r is None:
            messagebox.showinfo(TITLE, "先跑一次程序，才有数据可以对比。")
            return
        ref = DRAM_REFERENCE
        ours = r.effective_bps if r.effective_bps > 0 else float("nan")
        ref_bps = ref["bandwidth_bits_per_s"]
        ratio = ref_bps / ours if ours > 0 else float("inf")
        win = tk.Toplevel(self.master)
        win.title("性能对比 · 请准备好心理承受能力")
        win.geometry("660x480")
        txt = tk.Text(win, bg="#0c0c0c", fg="#e8e8e8", font=("Consolas", 11), wrap="word")
        txt.pack(fill="both", expand=True, padx=8, pady=8)

        lines = [
            "=" * 58,
            "  纸介质内存 vs 现实世界",
            "=" * 58,
            f"  本方案等效速率 : {ours:.4f} bit/s",
            f"  参考基准       : {ref['profile']}，{ref_bps / 1e9:.1f} Gbit/s",
            f"  差距           : {ratio:,.0f} 倍",
            "",
            f"  本次真实耗时   : {r.real_seconds:.2f} s",
            f"  仿真世界耗时   : {human_time(r.sim_seconds)}",
            f"  内存访问次数   : {r.accesses:,}",
            f"  故障/幻觉次数  : {r.faults:,}",
            f"  CPU 步数       : {r.steps:,}",
            "",
            "  如果把 1 GiB 数据交给本方案：",
        ]
        secs = (1024**3 * 8) / ours if ours > 0 else float("inf")
        lines.append(f"    写入需要 {human_time(secs)}")
        lines.append(f"    而 DDR5 大约需要 {1024**3 * 8 / ref_bps:.3f} 秒")
        lines += [
            "",
            "=" * 58,
            "  结论：建议继续使用 DDR5。",
            "  （本结论未经严格同行评议，但物理定律已代为签字）",
            "=" * 58,
        ]
        txt.insert("1.0", "\n".join(lines))
        txt.configure(state="disabled")


PROTOCOL_DOC = '''沙箱 ↔ 内存条 · 严格指令协议
================================================================

沙箱只发指令，内存条只回数据。双方都不许说废话。

沙箱 → 内存条                          内存条 → 沙箱
------------------------------------   -------------------------
READ 0x0010 4                          0000002a
WRITE 0x0010 4 0000002a                OK 0000002a
ERASE 0                                OK

格式约束
--------
· 地址一律 0x 前缀 + 4 位十六进制
· 长度是十进制字节数
· 数据是连写的十六进制：无空格、无 0x、无分隔符
· 回复必须只有一行，不许解释、不许道歉、不许加代码块

容错
----
解码时允许 AI 手滑写成 "00 00 00 2a" 或 "0x0000002a"，
解析器会容忍空格、逗号、下划线、连字符和 0x 前缀，但会拒绝：

· 长度不匹配（要 4 字节只给 2 字节）
· 长度是奇数
· 出现非十六进制字符
· 夹带解释文字导致解析不出干净 token

一旦拒绝，就记为一次故障，并让它按错误内容落纸。
因为内存条不该有"差不多对"这种状态。

为什么这么设计
--------------
1. 解析必须无歧义——内存读写不能靠猜。
2. 不守格式就是可检测的故障。这比"AI 记错了但假装没事"诚实得多。

在"能工智人 · 真实"模式下，这条协议就是给你看的：
黑窗底部出现输入栏，你按上面的格式手敲，沙箱才继续往下走。
'''


def main() -> None:
    root = tk.Tk()
    root.title(TITLE)
    root.geometry("1120x980")
    try:
        root.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    Launcher(root)
    root.mainloop()
