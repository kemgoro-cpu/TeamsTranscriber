"""
Teams Transcriber — エントリポイント

Teams会議録画MP4から高精度テキスト + 話者情報付きの文字起こしを生成します。

必要なパッケージ:
    pip install faster-whisper python-docx

外部依存:
    ffmpeg（PATH上にインストール） — https://ffmpeg.org/

使い方:
    python main.py
"""
from __future__ import annotations

import ctypes
import subprocess
import sys


def ensure_packages() -> None:
    """必要なパッケージが未インストールの場合は自動インストールする。"""
    required = {
        "faster_whisper": "faster-whisper",
        "docx":           "python-docx",
        "pyannote":       "pyannote.audio",
        "torch":          "torch",
    }
    for module, pip_name in required.items():
        try:
            __import__(module)
        except ImportError:
            print(f"[インストール] {pip_name} ...", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


def main() -> None:
    ensure_packages()

    import tkinter as tk
    import tkinter.messagebox as mb

    from teams_transcriber.audio import check_ffmpeg
    from teams_transcriber.gui.app import App

    # Windows DPIスケーリング対応
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

    root = tk.Tk()
    root.update_idletasks()

    # ffmpegの存在確認
    if not check_ffmpeg():
        mb.showerror(
            "ffmpegが見つかりません",
            "ffmpegがPATHにインストールされていません。\n\n"
            "https://ffmpeg.org/ からインストールし、\n"
            "システムのPATHに追加してください。",
        )
        root.destroy()
        return

    ww, wh = 780, 720
    x = (root.winfo_screenwidth()  - ww) // 2
    y = (root.winfo_screenheight() - wh) // 2
    root.geometry(f"{ww}x{wh}+{x}+{y}")
    root.minsize(600, 500)

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("teams.transcriber")
    except Exception:
        pass

    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
