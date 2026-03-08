"""
Teams Transcriber — メインGUIモジュール。
"""
from __future__ import annotations

import os
import subprocess
import threading
import traceback
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from teams_transcriber.config import (
    APP_NAME,
    APP_VERSION,
    FFMPEG_TIMEOUT,
    DIARIZATION_BACKEND,
    SPEAKER_ENROLLMENT_SIMILARITY_THRESHOLD,
    WHISPER_MODELS,
    DEFAULT_MODEL,
)
from teams_transcriber.audio import extract_audio
from teams_transcriber.diarizer import diarize
from teams_transcriber.merger import merge_consecutive
from teams_transcriber.output_writer import write_output
from teams_transcriber.settings import AppSettings
from teams_transcriber.credentials import get_hf_token, set_hf_token
from teams_transcriber.speaker_enrollment import (
    add_enrollment,
    compute_embeddings_for_diarized_speakers,
    delete_enrollment,
    list_enrollments,
    load_enrollment_embeddings,
    match_speakers_to_enrollment,
)
from teams_transcriber.synchronizer import (
    assign_speakers_from_diarization,
    detect_offset,
    synchronize,
)
from teams_transcriber.transcript_parser import is_single_speaker, parse_transcript
from teams_transcriber.whisper_runner import transcribe
from teams_transcriber.gui.theme import THEME as T

# 話者分離モード
DIARIZATION_MODES = ["自動", "常に音声話者分離", "使用しない"]

# 話者分離エンジン（表示名 → バックエンド識別子）
DIARIZATION_BACKEND_CHOICES = ["ローカル（diarize）", "pyannote（HFトークン要）"]
DIARIZATION_BACKEND_MAP = {
    "ローカル（diarize）": "diarize",
    "pyannote（HFトークン要）": "pyannote",
}
DIARIZATION_BACKEND_TO_LABEL = {v: k for k, v in DIARIZATION_BACKEND_MAP.items()}


class App:
    # ── フォント定数 ──
    _FT = ("Segoe UI", 18, "bold")
    _FS = ("Segoe UI", 10)
    _FB = ("Segoe UI Semibold", 11)
    _FL = ("Consolas", 9)
    _LOG_MIN_HEIGHT = 140

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.configure(bg=T["bg"])
        self.root.resizable(True, True)
        self._setup_ttk_styles()

        self.running: bool = False
        self._cancel_flag = threading.Event()
        self._heartbeat_after_id: str | None = None  # 話者分離中の「まだ処理中」表示用

        self.settings = AppSettings()
        self._tmp_wav: str | None = None

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.bind("<F5>", lambda e: self._start() if not self.running else None)

    # ── ライフサイクル ──

    def _quit(self) -> None:
        self._save_settings()
        self.root.destroy()

    def _save_settings(self) -> None:
        self.settings.set("last_mp4_dir", os.path.dirname(self.mp4_var.get()))
        self.settings.set("last_transcript_dir", os.path.dirname(self.transcript_var.get()))
        self.settings.set("whisper_model", self.model_var.get())
        self.settings.set("proxy_url", self.proxy_var.get().strip())
        set_hf_token(self.settings, self.hf_token_var.get().strip())
        self.settings.set("diarization_mode", self.diarization_var.get())
        backend_label = self.diarization_backend_var.get()
        self.settings.set(
            "diarization_backend",
            DIARIZATION_BACKEND_MAP.get(backend_label, "diarize"),
        )
        self.settings.set("use_speaker_enrollment", self.use_speaker_enrollment_var.get())
        self.settings.save()

    # ── UI構築 ──

    def _setup_ttk_styles(self) -> None:
        """ttk ウィジェットをダークテーマに合わせる。"""
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "TCombobox",
            fieldbackground=T["input"],
            background=T["card"],
            foreground=T["fg"],
            arrowcolor=T["fg2"],
        )
        style.map("TCombobox", fieldbackground=[("readonly", T["input"])])
        style.configure(
            "TT.Horizontal.TProgressbar",
            troughcolor=T["card"], background=T["accent"],
            thickness=8,
        )

    def _build(self) -> None:
        root = self.root
        c = tk.Frame(root, bg=T["bg"])
        c.pack(fill="both", expand=True, padx=28, pady=20)

        # 上段: フォーム（拡張しないのでログが隠れない）
        upper = tk.Frame(c, bg=T["bg"])
        upper.pack(fill="x")

        self._build_header(upper)
        tk.Frame(upper, bg=T["accent"], height=2).pack(fill="x", pady=(12, 14))
        self._build_inputs(upper)
        tk.Frame(upper, bg=T["border"], height=1).pack(fill="x", pady=(8, 8))
        self._build_model_row(upper)
        self._build_proxy_row(upper)
        self._build_hf_token_row(upper)
        self._build_diarization_backend_row(upper)
        self._build_diarization_row(upper)
        self._build_speaker_enrollment_section(upper)
        self._build_output_row(upper)
        self._build_buttons(upper)
        self._build_progress(upper)

        # ログ欄: 残りスペースを占有（最低高さを確保）
        self._build_log_panel(c)

    def _build_header(self, parent: tk.Frame) -> None:
        h = tk.Frame(parent, bg=T["bg"])
        h.pack(fill="x")
        tk.Label(h, text="🎙", font=("Segoe UI Emoji", 24),
                 bg=T["bg"], fg=T["accent"]).pack(side="left", padx=(0, 10))
        tk.Label(h, text=APP_NAME, font=self._FT,
                 bg=T["bg"], fg=T["fg"]).pack(side="left")
        tk.Label(h, text=f"  v{APP_VERSION}", font=self._FS,
                 bg=T["bg"], fg=T["dim"]).pack(side="left", pady=(8, 0))

    def _labeled_entry(
        self,
        parent: tk.Frame,
        label: str,
        var: tk.StringVar,
        browse_cmd: Any,
        hint: str = "",
    ) -> None:
        row = tk.Frame(parent, bg=T["bg"])
        row.pack(fill="x", pady=(0, 6))

        tk.Label(row, text=label, font=self._FS, bg=T["bg"], fg=T["fg2"],
                 width=12, anchor="e").pack(side="left", padx=(0, 6))
        tk.Entry(
            row, textvariable=var, font=self._FS,
            bg=T["input"], fg=T["input_fg"], insertbackground=T["input_fg"], relief="flat",
            highlightthickness=1, highlightbackground=T["border"],
            highlightcolor=T["accent"],
        ).pack(side="left", fill="x", expand=True, ipady=5, ipadx=6)

        btn = tk.Label(row, text=" 参照 ", font=self._FS, bg=T["card2"], fg=T["fg2"],
                       padx=10, pady=4, cursor="hand2")
        btn.pack(side="left", padx=(4, 0))
        btn.bind("<Button-1>", lambda e: browse_cmd())

        if hint:
            tk.Label(row, text=hint, font=self._FS, bg=T["bg"], fg=T["dim"],
                     padx=6).pack(side="left", padx=(6, 0))

    def _build_inputs(self, parent: tk.Frame) -> None:
        self.mp4_var = tk.StringVar()
        self.transcript_var = tk.StringVar()

        self._labeled_entry(
            parent, "MP4ファイル:", self.mp4_var, self._browse_mp4,
        )
        self._labeled_entry(
            parent, "文字起こし:", self.transcript_var, self._browse_transcript,
            hint=".vtt / .docx",
        )

        # MP4が変わったら出力パスを自動更新
        self.mp4_var.trace_add("write", self._on_mp4_change)

    def _build_model_row(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=T["bg"])
        row.pack(fill="x", pady=(0, 6))

        tk.Label(row, text="Whisperモデル:", font=self._FS, bg=T["bg"], fg=T["fg2"],
                 width=12, anchor="e").pack(side="left", padx=(0, 6))

        self.model_var = tk.StringVar(value=self.settings.get("whisper_model", DEFAULT_MODEL))
        model_cb = ttk.Combobox(
            row, textvariable=self.model_var,
            values=WHISPER_MODELS, state="readonly", width=12,
            font=self._FS,
        )
        model_cb.pack(side="left")

        tk.Label(row, text="  ※ large-v3-turbo が速度・精度のバランス最良",
                 font=self._FS, bg=T["bg"], fg=T["dim"]).pack(side="left", padx=(8, 0))

    def _build_proxy_row(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=T["bg"])
        row.pack(fill="x", pady=(0, 6))

        tk.Label(row, text="プロキシ URL:", font=self._FS, bg=T["bg"], fg=T["fg2"],
                 width=12, anchor="e").pack(side="left", padx=(0, 6))
        self.proxy_var = tk.StringVar(value=self.settings.get("proxy_url", ""))
        tk.Entry(
            row, textvariable=self.proxy_var, font=self._FS,
            bg=T["input"], fg=T["input_fg"], insertbackground=T["input_fg"], relief="flat",
            highlightthickness=1, highlightbackground=T["border"],
            highlightcolor=T["accent"],
        ).pack(side="left", fill="x", expand=True, ipady=5, ipadx=6)
        tk.Label(row, text="  省略可 例: http://proxy.example.com:8080",
                 font=self._FS, bg=T["bg"], fg=T["dim"]).pack(side="left", padx=(6, 0))

    def _build_hf_token_row(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=T["bg"])
        row.pack(fill="x", pady=(0, 6))

        tk.Label(row, text="HFトークン:", font=self._FS, bg=T["bg"], fg=T["fg2"],
                 width=12, anchor="e").pack(side="left", padx=(0, 6))
        self.hf_token_var = tk.StringVar(value=get_hf_token(self.settings))
        tk.Entry(
            row, textvariable=self.hf_token_var, font=self._FS, show="*",
            bg=T["input"], fg=T["input_fg"], insertbackground=T["input_fg"], relief="flat",
            highlightthickness=1, highlightbackground=T["border"],
            highlightcolor=T["accent"],
        ).pack(side="left", fill="x", expand=True, ipady=5, ipadx=6)
        tk.Label(
            row,
            text="  ※「pyannote」を選んだ時のみ必要",
            font=self._FS, bg=T["bg"], fg=T["dim"],
        ).pack(side="left", padx=(6, 0))

    def _build_diarization_backend_row(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=T["bg"])
        row.pack(fill="x", pady=(0, 6))

        tk.Label(
            row, text="話者分離エンジン:",
            font=self._FS, bg=T["bg"], fg=T["fg2"],
            width=12, anchor="e",
        ).pack(side="left", padx=(0, 6))

        saved = self.settings.get("diarization_backend", DIARIZATION_BACKEND)
        self.diarization_backend_var = tk.StringVar(
            value=DIARIZATION_BACKEND_TO_LABEL.get(saved, DIARIZATION_BACKEND_CHOICES[0])
        )
        backend_cb = ttk.Combobox(
            row,
            textvariable=self.diarization_backend_var,
            values=DIARIZATION_BACKEND_CHOICES,
            state="readonly",
            width=22,
            font=self._FS,
        )
        backend_cb.pack(side="left")
        tk.Label(
            row,
            text="  ※短時間で完了させるなら「ローカル」推奨（HF不要）",
            font=self._FS, bg=T["bg"], fg=T["dim"],
        ).pack(side="left", padx=(8, 0))
        # 話者登録の有効/無効はバックエンド変更で更新する
        backend_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_speaker_enrollment_state())

    def _build_diarization_row(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=T["bg"])
        row.pack(fill="x", pady=(0, 6))

        tk.Label(row, text="話者分離:", font=self._FS, bg=T["bg"], fg=T["fg2"],
                 width=12, anchor="e").pack(side="left", padx=(0, 6))

        self.diarization_var = tk.StringVar(
            value=self.settings.get("diarization_mode", "自動")
        )
        dia_cb = ttk.Combobox(
            row, textvariable=self.diarization_var,
            values=DIARIZATION_MODES, state="readonly", width=18,
            font=self._FS,
        )
        dia_cb.pack(side="left")

        tk.Label(row, text="  ※「自動」= 単一話者検出時のみ音声話者分離を実行",
                 font=self._FS, bg=T["bg"], fg=T["dim"]).pack(side="left", padx=(8, 0))

    def _build_speaker_enrollment_section(self, parent: tk.Frame) -> None:
        """話者登録ブロック: 一覧・追加フォーム・削除・「話者登録を使って名前を付ける」チェック"""
        from tkinter import messagebox as mb

        lf = tk.LabelFrame(
            parent, text="  話者登録（pyannote / diarize とも登録名で表示可能。diarize 時は HFトークン要）  ",
            font=self._FS, bg=T["card"], fg=T["fg2"],
            highlightbackground=T["border"], highlightthickness=1,
        )
        lf.pack(fill="x", pady=(0, 10))

        # 一覧（表示名, 音声, 区間, 品質）
        list_f = tk.Frame(lf, bg=T["card"])
        list_f.pack(fill="x", pady=(6, 6))
        self.enrollment_listbox = tk.Listbox(
            list_f, height=3, font=self._FS,
            bg=T["input"], fg=T["input_fg"], selectbackground=T["accent"],
            highlightthickness=0,
        )
        self.enrollment_listbox.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(list_f, command=self.enrollment_listbox.yview)
        sb.pack(side="right", fill="y")
        self.enrollment_listbox.configure(yscrollcommand=sb.set)

        # 追加フォーム（2行に分けてボタンが隠れないようにする）
        form_f = tk.Frame(lf, bg=T["card"])
        form_f.pack(fill="x", pady=(4, 6))

        # 1行目: 表示名 + 音声ファイル
        row1 = tk.Frame(form_f, bg=T["card"])
        row1.pack(fill="x", pady=(0, 6))
        tk.Label(row1, text="表示名:", font=self._FS, bg=T["card"], fg=T["fg2"], width=6, anchor="e").pack(side="left", padx=(0, 4))
        self.enrollment_name_var = tk.StringVar()
        tk.Entry(row1, textvariable=self.enrollment_name_var, font=self._FS, width=14,
                 bg=T["input"], fg=T["input_fg"], insertbackground=T["input_fg"], highlightthickness=0).pack(side="left", padx=(0, 12), ipady=4, ipadx=4)
        tk.Label(row1, text="音声:", font=self._FS, bg=T["card"], fg=T["fg2"], width=4, anchor="e").pack(side="left", padx=(0, 4))
        self.enrollment_audio_var = tk.StringVar()
        tk.Entry(row1, textvariable=self.enrollment_audio_var, font=self._FS,
                 bg=T["input"], fg=T["input_fg"], insertbackground=T["input_fg"], highlightthickness=0).pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=4, ipadx=4)
        btn_browse_audio = tk.Label(row1, text=" 参照 ", font=self._FS, bg=T["card2"], fg=T["fg2"], padx=10, pady=4, cursor="hand2")
        btn_browse_audio.pack(side="left")

        def _browse_enrollment_audio() -> None:
            path = filedialog.askopenfilename(
                title="参照音声・動画を選択（WAV/MP3/MP4等）",
                filetypes=[
                    ("音声・動画", "*.wav *.mp3 *.wave *.mp4 *.mkv *.webm"),
                    ("音声", "*.wav *.mp3 *.wave"),
                    ("動画", "*.mp4 *.mkv *.webm"),
                    ("すべてのファイル", "*.*"),
                ],
            )
            if path:
                self.enrollment_audio_var.set(path)

        btn_browse_audio.bind("<Button-1>", lambda e: _browse_enrollment_audio())

        # 2行目: 開始秒・終了秒 + 登録・削除ボタン
        row2 = tk.Frame(form_f, bg=T["card"])
        row2.pack(fill="x")
        tk.Label(row2, text="開始秒:", font=self._FS, bg=T["card"], fg=T["fg2"], width=6, anchor="e").pack(side="left", padx=(0, 4))
        self.enrollment_start_var = tk.StringVar()
        tk.Entry(row2, textvariable=self.enrollment_start_var, font=self._FS, width=6,
                 bg=T["input"], fg=T["input_fg"], insertbackground=T["input_fg"], highlightthickness=0).pack(side="left", padx=(0, 8), ipady=4, ipadx=4)
        tk.Label(row2, text="終了秒:", font=self._FS, bg=T["card"], fg=T["fg2"], width=6, anchor="e").pack(side="left", padx=(0, 4))
        self.enrollment_end_var = tk.StringVar()
        tk.Entry(row2, textvariable=self.enrollment_end_var, font=self._FS, width=6,
                 bg=T["input"], fg=T["input_fg"], insertbackground=T["input_fg"], highlightthickness=0).pack(side="left", padx=(0, 16), ipady=4, ipadx=4)

        def _do_add_enrollment() -> None:
            name = self.enrollment_name_var.get().strip()
            audio_path = self.enrollment_audio_var.get().strip()
            if not name:
                mb.showerror("入力エラー", "表示名を入力してください。")
                return
            if not audio_path or not os.path.isfile(audio_path):
                mb.showerror("入力エラー", "音声ファイルを選択し、存在するパスを指定してください。")
                return
            start_s = self.enrollment_start_var.get().strip()
            end_s = self.enrollment_end_var.get().strip()
            start_sec: float | None = float(start_s) if start_s else None
            end_sec: float | None = float(end_s) if end_s else None
            if start_s and end_s and start_sec is not None and end_sec is not None and start_sec >= end_sec:
                mb.showerror("入力エラー", "開始秒は終了秒より小さくしてください。")
                return
            hf_token = self.hf_token_var.get().strip()
            if not hf_token:
                mb.showerror("話者登録", "pyannote 用の HFトークン を入力してください。")
                return
            try:
                uid, warnings = add_enrollment(name, audio_path, hf_token, start_sec, end_sec)
                self._refresh_enrollment_list()
                self._refresh_speaker_enrollment_state()
                self.enrollment_name_var.set("")
                self.enrollment_audio_var.set("")
                self.enrollment_start_var.set("")
                self.enrollment_end_var.set("")
                self.log(f"話者登録を追加しました: {name} (ID: {uid})", "ok")
                for w in warnings:
                    self.log(f"  ⚠ {w}", "warn")
            except Exception as ex:
                self.log(f"話者登録の追加に失敗しました: {ex}", "err")
                mb.showerror("話者登録", str(ex))

        btn_add = tk.Label(row2, text="  登録  ", font=self._FB, bg=T["accent"], fg="white", padx=14, pady=6, cursor="hand2")
        btn_add.pack(side="left", padx=(0, 8))
        btn_add.bind("<Button-1>", lambda e: _do_add_enrollment())
        btn_add.bind("<Enter>", lambda e: btn_add.configure(bg=T["accent2"]))
        btn_add.bind("<Leave>", lambda e: btn_add.configure(bg=T["accent"]))

        def _do_delete_enrollment() -> None:
            sel = self.enrollment_listbox.curselection()
            if not sel:
                mb.showinfo("話者登録", "削除する話者を一覧で選択してください。")
                return
            idx = int(sel[0])
            enrollments = list_enrollments()
            if idx >= len(enrollments):
                return
            sp = enrollments[idx]
            if mb.askyesno("話者登録", f"「{sp.name}」を削除しますか？"):
                delete_enrollment(sp.id)
                self._refresh_enrollment_list()
                self._refresh_speaker_enrollment_state()
                self.log(f"話者登録を削除しました: {sp.name}", "dim")

        btn_del = tk.Label(row2, text="  削除  ", font=self._FS, bg=T["card2"], fg=T["fg2"], padx=12, pady=6, cursor="hand2")
        btn_del.pack(side="left")
        btn_del.bind("<Button-1>", lambda e: _do_delete_enrollment())
        btn_del.bind("<Enter>", lambda e: btn_del.configure(fg=T["fg"]))
        btn_del.bind("<Leave>", lambda e: btn_del.configure(fg=T["fg2"]))

        self._refresh_enrollment_list = self._build_refresh_enrollment_list()
        self._refresh_speaker_enrollment_state = self._build_refresh_speaker_enrollment_state()

        # 「話者登録を使って名前を付ける」チェック（pyannote かつ 登録≥1 のときのみ有効）
        row_cb = tk.Frame(lf, bg=T["card"])
        row_cb.pack(fill="x", pady=(6, 4))
        self.use_speaker_enrollment_var = tk.BooleanVar(value=self.settings.get("use_speaker_enrollment", False))
        self.use_speaker_enrollment_cb = tk.Checkbutton(
            row_cb, text="話者登録を使って名前を付ける（分離結果の SPEAKER_00 等を登録名に変換）",
            variable=self.use_speaker_enrollment_var, font=self._FS, bg=T["card"], fg=T["fg2"],
            activebackground=T["card"], activeforeground=T["fg2"], selectcolor=T["input"],
        )
        self.use_speaker_enrollment_cb.pack(side="left")
        self._refresh_speaker_enrollment_state()
        # 起動直後に登録一覧を表示
        self._refresh_enrollment_list()

    def _build_refresh_enrollment_list(self) -> Any:
        def refresh() -> None:
            self.enrollment_listbox.delete(0, "end")
            for e in list_enrollments():
                rng = ""
                if e.start_sec is not None or e.end_sec is not None:
                    rng = f" [{e.start_sec or 0}-{e.end_sec or '?'}s]"
                status = {"ok": "", "multiple_speakers": "⚠複数話者?", "low_speech_ratio": "⚠発話少"}.get(e.quality_status, "")
                line = f"{e.name} | {os.path.basename(e.audio_path)}{rng} {status}".strip()
                self.enrollment_listbox.insert("end", line)
        return refresh

    def _build_refresh_speaker_enrollment_state(self) -> Any:
        def refresh() -> None:
            enrollments = list_enrollments()
            enabled = len(enrollments) >= 1
            self.use_speaker_enrollment_cb.configure(state="normal" if enabled else "disabled")
            if not enabled:
                self.use_speaker_enrollment_var.set(False)
        return refresh

    def _build_output_row(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=T["bg"])
        row.pack(fill="x", pady=(0, 12))

        tk.Label(row, text="出力ファイル:", font=self._FS, bg=T["bg"], fg=T["fg2"],
                 width=12, anchor="e").pack(side="left", padx=(0, 6))
        self.output_var = tk.StringVar()
        self.output_lbl = tk.Label(
            row, textvariable=self.output_var,
            font=self._FS, bg=T["bg"], fg=T["dim"],
            anchor="w",
        )
        self.output_lbl.pack(side="left", fill="x", expand=True)

    def _build_buttons(self, parent: tk.Frame) -> None:
        bf = tk.Frame(parent, bg=T["bg"])
        bf.pack(fill="x", pady=(0, 10))

        self.start_btn = tk.Label(
            bf, text="  ▶  文字起こし開始  ", font=self._FB,
            bg=T["accent"], fg="white", pady=12, cursor="hand2",
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.start_btn.bind("<Button-1>", lambda e: self._start())
        self.start_btn.bind("<Enter>", lambda e: (not self.running) and
                            self.start_btn.configure(bg=T["accent2"]))
        self.start_btn.bind("<Leave>", lambda e: (not self.running) and
                            self.start_btn.configure(bg=T["accent"]))

        self.cancel_btn = tk.Label(
            bf, text="  ■  中止  ", font=self._FB,
            bg=T["card2"], fg=T["fg2"], pady=12, padx=24, cursor="hand2",
        )
        self.cancel_btn.pack(side="right")
        self.cancel_btn.bind("<Button-1>", lambda e: self._do_cancel())
        self.cancel_btn.bind("<Enter>", lambda e: self.cancel_btn.configure(bg=T["err"], fg="white"))
        self.cancel_btn.bind("<Leave>", lambda e: self.cancel_btn.configure(bg=T["card2"], fg=T["fg2"]))

    def _build_progress(self, parent: tk.Frame) -> None:
        pf = tk.Frame(parent, bg=T["bg"])
        pf.pack(fill="x", pady=(0, 8))

        self.step_label = tk.Label(pf, text="待機中", font=self._FS,
                                   bg=T["bg"], fg=T["fg2"])
        self.step_label.pack(side="left")

        self.progress_var = tk.DoubleVar(value=0.0)
        pb = ttk.Progressbar(
            pf, variable=self.progress_var,
            style="TT.Horizontal.TProgressbar",
            mode="determinate", maximum=100,
        )
        pb.pack(side="right", fill="x", expand=True, padx=(12, 0))

    def _build_log_panel(self, parent: tk.Frame) -> None:
        logf = tk.Frame(parent, bg=T["card"],
                        highlightbackground=T["border"], highlightthickness=1)
        logf.pack(fill="both", expand=True)
        logf.configure(height=max(200, self._LOG_MIN_HEIGHT))

        logh = tk.Frame(logf, bg=T["card"])
        logh.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(logh, text="ログ", font=self._FS, bg=T["card"], fg=T["dim"]).pack(side="left")

        self.log_text = tk.Text(
            logf, font=self._FL, bg=T["trbg"], fg=T["fg2"],
            relief="flat", insertbackground=T["fg"],
            selectbackground=T["accent"],
            wrap="word", padx=12, pady=8,
            highlightthickness=0, borderwidth=0,
        )
        self.log_text.pack(fill="both", expand=True)

        sb = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        sb.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=sb.set)

        for tag, color in [
            ("ok", T["ok"]), ("err", T["err"]), ("warn", T["warn"]),
            ("acc", T["accent"]), ("dim", T["dim"]),
        ]:
            self.log_text.tag_configure(tag, foreground=color)

        self.log_text.configure(state="disabled")

    # ── UI更新ヘルパー ──

    def _ui(self, fn: Any) -> None:
        self.root.after(0, fn)

    _MAX_LOG_LINES = 500

    def log(self, msg: str, tag: str | None = None) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n", tag or "")
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > self._MAX_LOG_LINES:
            self.log_text.delete("1.0", f"{lines - self._MAX_LOG_LINES}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _log(self, msg: str, tag: str | None = None) -> None:
        self._ui(lambda: self.log(msg, tag))

    def _step(self, text: str, progress: float | None = None) -> None:
        def _do() -> None:
            self.step_label.configure(text=text)
            if progress is not None:
                self.progress_var.set(progress)
        self._ui(_do)

    def _start_diarization_heartbeat(self) -> None:
        """話者分離中のログ用タイマーを開始（数秒ごとに「まだ処理中」を表示）"""
        self._stop_diarization_heartbeat()
        self._heartbeat_after_id = self.root.after(
            8000,
            self._diarization_heartbeat_tick,
        )

    def _stop_diarization_heartbeat(self) -> None:
        """話者分離中のログ用タイマーを停止"""
        if self._heartbeat_after_id is not None:
            self.root.after_cancel(self._heartbeat_after_id)
            self._heartbeat_after_id = None

    def _diarization_heartbeat_tick(self) -> None:
        """定期的にログへ「まだ話者分離を実行中」を出す"""
        self._heartbeat_after_id = None
        if not self.running:
            return
        self.log("  ... 話者分離を実行中（しばらくお待ちください）", "dim")
        self._heartbeat_after_id = self.root.after(
            8000,
            self._diarization_heartbeat_tick,
        )

    # ── ファイル参照 ──

    def _browse_mp4(self) -> None:
        init = self.settings.get("last_mp4_dir", os.path.expanduser("~"))
        path = filedialog.askopenfilename(
            initialdir=init,
            title="MP4ファイルを選択",
            filetypes=[("動画ファイル", "*.mp4 *.mkv *.webm"), ("すべてのファイル", "*.*")],
        )
        if path:
            self.mp4_var.set(path)

    def _browse_transcript(self) -> None:
        init = self.settings.get("last_transcript_dir", os.path.expanduser("~"))
        path = filedialog.askopenfilename(
            initialdir=init,
            title="Teamsトランスクリプトを選択",
            filetypes=[
                ("トランスクリプト", "*.vtt *.docx"),
                ("WebVTT", "*.vtt"),
                ("Word文書", "*.docx"),
                ("すべてのファイル", "*.*"),
            ],
        )
        if path:
            self.transcript_var.set(path)

    def _on_mp4_change(self, *_: Any) -> None:
        mp4 = self.mp4_var.get()
        if mp4:
            stem = os.path.splitext(mp4)[0]
            self.output_var.set(stem + "_transcript.txt")

    # ── 開始・中止 ──

    def _start(self) -> None:
        if self.running:
            return

        mp4 = self.mp4_var.get().strip()
        transcript = self.transcript_var.get().strip()

        if not mp4:
            self.log("✗ MP4ファイルを指定してください", "err")
            return
        if not os.path.isfile(mp4):
            self.log(f"✗ MP4ファイルが見つかりません: {mp4}", "err")
            return
        if not transcript:
            self.log("✗ 文字起こしファイルを指定してください", "err")
            return
        if not os.path.isfile(transcript):
            self.log(f"✗ 文字起こしファイルが見つかりません: {transcript}", "err")
            return

        out_path = self.output_var.get().strip()
        if not out_path:
            self.log("✗ 出力ファイルパスが設定されていません", "err")
            return

        self._save_settings()
        self.running = True
        self._cancel_flag.clear()
        self.start_btn.configure(bg=T["dim"], cursor="arrow")

        # ログクリア
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        diarization_backend = DIARIZATION_BACKEND_MAP.get(
            self.diarization_backend_var.get(), "diarize"
        )
        use_speaker_enrollment = (
            self.use_speaker_enrollment_var.get()
            and len(list_enrollments()) >= 1
        )
        threading.Thread(
            target=self._run,
            args=(
                mp4, transcript, out_path, self.model_var.get(),
                self.proxy_var.get().strip(),
                self.hf_token_var.get().strip(),
                self.diarization_var.get(),
                diarization_backend,
                use_speaker_enrollment,
            ),
            daemon=True,
        ).start()

    def _do_cancel(self) -> None:
        if self.running:
            self._cancel_flag.set()
            self._log("中止リクエスト送信...", "warn")

    def _check_cancel(self) -> None:
        if self._cancel_flag.is_set():
            raise InterruptedError()

    # ── メインパイプライン ──

    def _run(
        self, mp4: str, transcript: str, out_path: str, model: str,
        proxy: str = "", hf_token: str = "", diarization_mode: str = "自動",
        diarization_backend: str = "diarize",
        use_speaker_enrollment: bool = False,
    ) -> None:
        wav_path: str | None = None
        ok = False

        try:
            # Step 1: Teamsトランスクリプト解析
            self._step("文字起こし解析", 5.0)
            self._log("Teamsトランスクリプトを解析中...", "acc")
            teams_segs = parse_transcript(transcript)
            self._log(f"  {len(teams_segs)} セグメント検出")
            if not teams_segs:
                self._log("⚠ 有効なセグメントが見つかりませんでした", "warn")
            self._check_cancel()

            # 話者分離を使うかどうか判定
            use_diarization = False
            if diarization_mode == "常に音声話者分離":
                use_diarization = True
                self._log("話者分離モード: 常に音声話者分離", "acc")
            elif diarization_mode == "自動":
                if is_single_speaker(teams_segs):
                    use_diarization = True
                    self._log("単一話者を検出 → 音声話者分離を使用します", "acc")
                else:
                    self._log("複数話者を検出 → Teamsの話者情報を使用します", "dim")
            else:
                self._log("話者分離モード: 使用しない", "dim")

            if use_diarization and diarization_backend == "pyannote" and not hf_token:
                self._log("⚠ HFトークンが未設定のため話者分離をスキップします（pyannote使用時）", "warn")
                use_diarization = False

            # Step 2: 音声抽出
            self._step("音声抽出", 10.0)
            self._log("ffmpegで音声を抽出中...", "acc")
            stem = os.path.splitext(mp4)[0]
            wav_path = stem + "_tmp_audio.wav"
            try:
                extract_audio(mp4, wav_path, progress_callback=self._log)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"音声抽出がタイムアウトしました（制限: {FFMPEG_TIMEOUT}秒）。"
                    "非常に長い録画の場合は config.py の FFMPEG_TIMEOUT を大きくしてください。"
                )
            self._log("✓ 音声抽出完了", "ok")
            self._check_cancel()

            # Step 3: Whisper文字起こし
            self._step("Whisper 実行中", 20.0)
            self._log(f"Whisperで文字起こし中... （モデル: {model} / 言語: 日本語）", "acc")

            if proxy:
                self._log(f"  プロキシ使用: {proxy}", "dim")
            whisper_segs = transcribe(
                wav_path,
                model,
                progress_callback=self._whisper_progress,
                download_callback=self._log,
                proxy_url=proxy or None,
                cancel_flag=self._cancel_flag,
            )
            self._log(f"✓ Whisper完了: {len(whisper_segs)} セグメント", "ok")
            self._step("Whisper 完了", 70.0)
            self._check_cancel()

            if use_diarization:
                # Step 4: 話者分離（バックエンドは config で diarize / pyannote）
                # 進捗は 75%〜82% の範囲で表示（pyannote 時は hook で細かく更新、diarize 時は 75% のまま）
                self._step("話者分離", 75.0)
                self._ui(self._start_diarization_heartbeat)
                try:
                    diarize_result = diarize(
                        wav_path,
                        hf_token or "",
                        progress_callback=self._log,
                        progress_pct_callback=lambda frac: self._ui(
                            lambda f=frac: self.progress_var.set(75.0 + 7.0 * f)
                        ),
                        backend=diarization_backend,
                        return_embeddings=use_speaker_enrollment,
                    )
                finally:
                    self._ui(self._stop_diarization_heartbeat)
                self._check_cancel()

                speaker_map = None
                if isinstance(diarize_result, tuple):
                    diarized_segs, diar_embeddings = diarize_result
                else:
                    diarized_segs = diarize_result

                if use_speaker_enrollment:
                    if diarization_backend == "pyannote" and isinstance(diarize_result, tuple):
                        enrolled = load_enrollment_embeddings()
                        speaker_map = match_speakers_to_enrollment(
                            diar_embeddings, enrolled, SPEAKER_ENROLLMENT_SIMILARITY_THRESHOLD
                        )
                        if speaker_map:
                            self._log(f"  話者登録と照合: {len(speaker_map)} 名を割り当てました", "acc")
                        else:
                            speaker_map = None
                    elif diarization_backend == "diarize":
                        if hf_token:
                            self._log("  diarize の話者を登録名と照合中（embedding 取得）...", "acc")
                            diar_embeddings = compute_embeddings_for_diarized_speakers(
                                wav_path,
                                diarized_segs,
                                hf_token,
                                progress_callback=lambda msg: self._log(msg, "dim"),
                            )
                            if diar_embeddings:
                                enrolled = load_enrollment_embeddings()
                                speaker_map = match_speakers_to_enrollment(
                                    diar_embeddings, enrolled, SPEAKER_ENROLLMENT_SIMILARITY_THRESHOLD
                                )
                                if speaker_map:
                                    self._log(f"  話者登録と照合: {len(speaker_map)} 名を割り当てました", "acc")
                                else:
                                    speaker_map = None
                            else:
                                speaker_map = None
                        else:
                            self._log("  ⚠ diarize で話者登録を使うには HFトークン を入力してください", "warn")
                            speaker_map = None

                # Step 5: Whisper + Diarization → 話者割り当て
                self._step("話者割り当て", 82.0)
                self._log("Whisperセグメントに話者分離結果を割り当て中...", "acc")
                synced = assign_speakers_from_diarization(
                    whisper_segs, diarized_segs, speaker_map=speaker_map
                )
            else:
                # Step 4: オフセット自動検出（従来フロー）
                self._step("オフセット検出", 75.0)
                self._log("タイムスタンプのオフセットを自動検出中...", "acc")
                offset = detect_offset(whisper_segs, teams_segs)
                sign = "+" if offset >= 0 else ""
                self._log(f"  自動検出オフセット: {sign}{offset:.1f}秒", "ok")
                self._log(
                    f"  （Teamsの会議開始が録音開始より {abs(offset):.1f}秒 "
                    + ("先行" if offset > 0 else "後続") + "）",
                    "dim",
                )
                self._check_cancel()

                # Step 5: 話者同期（従来フロー）
                self._step("話者割り当て", 80.0)
                self._log("Whisperセグメントに話者情報を付与中...", "acc")
                synced = synchronize(whisper_segs, teams_segs, offset)

            unknown_count = sum(1 for s in synced if s.speaker == "不明")
            if unknown_count:
                self._log(f"  ⚠ {unknown_count} セグメントに話者を割当できませんでした", "warn")
            self._check_cancel()

            # Step 6: 連続同一話者をマージ
            self._step("セグメント結合", 88.0)
            merged = merge_consecutive(synced)
            self._log(f"  {len(synced)} → {len(merged)} 行にまとめました", "dim")
            self._check_cancel()

            # Step 7: 出力書き込み
            self._step("ファイル出力", 95.0)
            self._log(f"書き込み中... → {os.path.basename(out_path)}", "acc")
            write_output(merged, out_path)

            # 完了
            ok = True
            self._step("完了", 100.0)
            self._log("─" * 40, "dim")
            self._log("✓ 完了！", "ok")
            self._log(f"  {out_path}", "ok")
            self._ui(lambda: self._show_open(out_path))

        except InterruptedError:
            self._step("中止", 0.0)
            self._log("中止されました", "warn")
        except (FileNotFoundError, ValueError) as e:
            self._step("エラー", 0.0)
            self._log(f"✗ {e}", "err")
        except Exception as e:
            self._step("エラー", 0.0)
            tb_str = traceback.format_exc()
            self._log(f"✗ 予期しないエラー: {e}", "err")
            for line in tb_str.splitlines():
                self._log(f"  {line}", "dim")
            crash_log = os.path.join(
                os.environ.get("APPDATA", os.path.expanduser("~")),
                "TeamsTranscriber",
                "last_error.txt",
            )
            try:
                os.makedirs(os.path.dirname(crash_log), exist_ok=True)
                with open(crash_log, "w", encoding="utf-8") as f:
                    f.write(tb_str)
            except Exception:
                pass
            self._ui(
                lambda p=crash_log: messagebox.showerror(
                    "予期しないエラー",
                    "詳細は以下のファイルを参照してください。\n\n" + p,
                )
            )
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
            self.running = False
            self._ui(lambda: self.start_btn.configure(bg=T["accent"], cursor="hand2"))
            if ok:
                self._ui(lambda: self.root.lift())

    def _whisper_progress(self, count: int, preview: str) -> None:
        self._log(f"  [{count}] {preview}...", "dim")
        pct = min(68.0, 20.0 + count * 0.05)
        self._ui(lambda: self.progress_var.set(pct))

    def _show_open(self, filepath: str) -> None:
        folder = os.path.dirname(filepath)
        btn = tk.Label(
            self.log_text, text="  📂 フォルダを開く  ",
            font=("Segoe UI", 9), bg=T["accent"], fg="white",
            cursor="hand2", pady=3,
        )
        self.log_text.configure(state="normal")
        self.log_text.insert("end", "\n")
        self.log_text.window_create("end", window=btn)
        self.log_text.insert("end", "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        btn.bind("<Button-1>", lambda e: os.startfile(folder))
