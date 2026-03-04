"""
Teams Transcriber — メインGUIモジュール。
"""
from __future__ import annotations

import os
import threading
from typing import Any

import tkinter as tk
from tkinter import filedialog, ttk

from config import APP_NAME, APP_VERSION, WHISPER_MODELS, DEFAULT_MODEL
from teams_transcriber.audio import extract_audio
from teams_transcriber.diarizer import diarize
from teams_transcriber.merger import merge_consecutive
from teams_transcriber.output_writer import write_output
from teams_transcriber.settings import AppSettings
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


class App:
    # ── フォント定数 ──
    _FT = ("Segoe UI", 16, "bold")
    _FS = ("Segoe UI", 9)
    _FB = ("Segoe UI Semibold", 11)
    _FL = ("Consolas", 9)

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.configure(bg=T["bg"])
        self.root.resizable(True, True)

        self.running: bool = False
        self._cancel_flag = threading.Event()

        self.settings = AppSettings()
        self._tmp_wav: str | None = None

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    # ── ライフサイクル ──

    def _quit(self) -> None:
        self._save_settings()
        self.root.destroy()

    def _save_settings(self) -> None:
        self.settings.set("last_mp4_dir", os.path.dirname(self.mp4_var.get()))
        self.settings.set("last_transcript_dir", os.path.dirname(self.transcript_var.get()))
        self.settings.set("whisper_model", self.model_var.get())
        self.settings.set("proxy_url", self.proxy_var.get().strip())
        self.settings.set("hf_token", self.hf_token_var.get().strip())
        self.settings.set("diarization_mode", self.diarization_var.get())
        self.settings.save()

    # ── UI構築 ──

    def _build(self) -> None:
        root = self.root
        c = tk.Frame(root, bg=T["bg"])
        c.pack(fill="both", expand=True, padx=24, pady=16)

        self._build_header(c)
        tk.Frame(c, bg=T["border"], height=1).pack(fill="x", pady=(10, 10))
        self._build_inputs(c)
        tk.Frame(c, bg=T["border"], height=1).pack(fill="x", pady=(8, 8))
        self._build_model_row(c)
        self._build_proxy_row(c)
        self._build_hf_token_row(c)
        self._build_diarization_row(c)
        self._build_output_row(c)
        self._build_buttons(c)
        self._build_progress(c)
        self._build_log_panel(c)

    def _build_header(self, parent: tk.Frame) -> None:
        h = tk.Frame(parent, bg=T["bg"])
        h.pack(fill="x")
        tk.Label(h, text="🎙", font=("Segoe UI Emoji", 20),
                 bg=T["bg"], fg=T["accent"]).pack(side="left", padx=(0, 8))
        tk.Label(h, text=APP_NAME, font=self._FT,
                 bg=T["bg"], fg=T["fg"]).pack(side="left")
        tk.Label(h, text=f"v{APP_VERSION}", font=self._FS,
                 bg=T["bg"], fg=T["dim"]).pack(side="left", padx=(8, 0), pady=(6, 0))

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
            bg=T["input"], fg=T["fg"], insertbackground=T["fg"], relief="flat",
            highlightthickness=1, highlightbackground=T["border"],
            highlightcolor=T["accent"],
        ).pack(side="left", fill="x", expand=True, ipady=4)

        btn = tk.Label(row, text="...", font=self._FS, bg=T["card"], fg=T["fg2"],
                       padx=10, pady=2, cursor="hand2")
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
            bg=T["input"], fg=T["fg"], insertbackground=T["fg"], relief="flat",
            highlightthickness=1, highlightbackground=T["border"],
            highlightcolor=T["accent"],
        ).pack(side="left", fill="x", expand=True, ipady=4)
        tk.Label(row, text="  省略可 例: http://proxy.example.com:8080",
                 font=self._FS, bg=T["bg"], fg=T["dim"]).pack(side="left", padx=(6, 0))

    def _build_hf_token_row(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=T["bg"])
        row.pack(fill="x", pady=(0, 6))

        tk.Label(row, text="HFトークン:", font=self._FS, bg=T["bg"], fg=T["fg2"],
                 width=12, anchor="e").pack(side="left", padx=(0, 6))
        self.hf_token_var = tk.StringVar(value=self.settings.get("hf_token", ""))
        tk.Entry(
            row, textvariable=self.hf_token_var, font=self._FS, show="*",
            bg=T["input"], fg=T["fg"], insertbackground=T["fg"], relief="flat",
            highlightthickness=1, highlightbackground=T["border"],
            highlightcolor=T["accent"],
        ).pack(side="left", fill="x", expand=True, ipady=4)
        tk.Label(row, text="  話者分離に必要（huggingface.co/settings/tokens）",
                 font=self._FS, bg=T["bg"], fg=T["dim"]).pack(side="left", padx=(6, 0))

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

    def _build_output_row(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=T["bg"])
        row.pack(fill="x", pady=(0, 10))

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
        bf.pack(fill="x", pady=(0, 8))

        self.start_btn = tk.Label(
            bf, text="▶  文字起こし開始", font=self._FB,
            bg=T["accent"], fg="white", pady=10, cursor="hand2",
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.start_btn.bind("<Button-1>", lambda e: self._start())
        self.start_btn.bind("<Enter>", lambda e: (not self.running) and
                            self.start_btn.configure(bg=T["accent2"]))
        self.start_btn.bind("<Leave>", lambda e: (not self.running) and
                            self.start_btn.configure(bg=T["accent"]))

        self.cancel_btn = tk.Label(
            bf, text="■  中止", font=self._FB,
            bg=T["card"], fg=T["fg2"], pady=10, padx=20, cursor="hand2",
        )
        self.cancel_btn.pack(side="right")
        self.cancel_btn.bind("<Button-1>", lambda e: self._do_cancel())
        self.cancel_btn.bind("<Enter>", lambda e: self.cancel_btn.configure(bg=T["err"], fg="white"))
        self.cancel_btn.bind("<Leave>", lambda e: self.cancel_btn.configure(bg=T["card"], fg=T["fg2"]))

    def _build_progress(self, parent: tk.Frame) -> None:
        pf = tk.Frame(parent, bg=T["bg"])
        pf.pack(fill="x", pady=(0, 6))

        self.step_label = tk.Label(pf, text="待機中", font=self._FS,
                                   bg=T["bg"], fg=T["fg2"])
        self.step_label.pack(side="left")

        self.progress_var = tk.DoubleVar(value=0.0)
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "TT.Horizontal.TProgressbar",
            troughcolor=T["card"], background=T["accent"],
            thickness=6,
        )
        pb = ttk.Progressbar(
            pf, variable=self.progress_var,
            style="TT.Horizontal.TProgressbar",
            mode="determinate", maximum=100,
        )
        pb.pack(side="right", fill="x", expand=True, padx=(10, 0))

    def _build_log_panel(self, parent: tk.Frame) -> None:
        logf = tk.Frame(parent, bg=T["card"],
                        highlightbackground=T["border"], highlightthickness=1)
        logf.pack(fill="both", expand=True)

        logh = tk.Frame(logf, bg=T["card"])
        logh.pack(fill="x", padx=10, pady=(8, 0))
        tk.Label(logh, text="📋 ログ", font=self._FS, bg=T["card"], fg=T["dim"]).pack(side="left")

        self.log_text = tk.Text(
            logf, font=self._FL, bg=T["card"], fg=T["fg2"],
            relief="flat", insertbackground=T["fg"],
            selectbackground=T["accent"],
            wrap="word", padx=10, pady=6,
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

        threading.Thread(
            target=self._run,
            args=(
                mp4, transcript, out_path, self.model_var.get(),
                self.proxy_var.get().strip(),
                self.hf_token_var.get().strip(),
                self.diarization_var.get(),
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

            if use_diarization and not hf_token:
                self._log("⚠ HFトークンが未設定のため話者分離をスキップします", "warn")
                use_diarization = False

            # Step 2: 音声抽出
            self._step("音声抽出", 10.0)
            self._log("ffmpegで音声を抽出中...", "acc")
            stem = os.path.splitext(mp4)[0]
            wav_path = stem + "_tmp_audio.wav"
            extract_audio(mp4, wav_path, progress_callback=self._log)
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
                # Step 4: pyannote話者分離
                self._step("話者分離", 75.0)
                self._log("pyannote で話者分離を実行中...", "acc")
                diarized_segs = diarize(
                    wav_path, hf_token, progress_callback=self._log,
                )
                self._check_cancel()

                # Step 5: Whisper + Diarization → 話者割り当て
                self._step("話者割り当て", 82.0)
                self._log("Whisperセグメントに話者分離結果を割り当て中...", "acc")
                synced = assign_speakers_from_diarization(whisper_segs, diarized_segs)
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
        except FileNotFoundError as e:
            self._step("エラー", 0.0)
            self._log(f"✗ {e}", "err")
        except ValueError as e:
            self._step("エラー", 0.0)
            self._log(f"✗ {e}", "err")
        except Exception as e:
            self._step("エラー", 0.0)
            self._log(f"✗ 予期しないエラー: {e}", "err")
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
