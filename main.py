"""
Teams Transcriber — エントリポイント

Teams会議録画MP4から高精度テキスト + 話者情報付きの文字起こしを生成します。

必要なパッケージ:
    pip install faster-whisper python-docx
話者分離（完全ローカル）: pip install diarize
話者分離（高精度・pyannote）: pip install pyannote.audio torchcodec ＋ PYANNOTE_SETUP.md の手順

外部依存:
    ffmpeg（PATH上にインストール） — https://ffmpeg.org/

使い方:
    python main.py
"""
from __future__ import annotations

import os
import traceback
import warnings

# 依存ライブラリ由来の警告を抑止（pyannote / lightning_fabric）
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom.*")

import ctypes
import subprocess
import sys


def _app_data_dir() -> str:
    return os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~")),
        "TeamsTranscriber",
    )


def _last_error_path() -> str:
    return os.path.join(_app_data_dir(), "last_error.txt")


def _rotate_last_error() -> None:
    """起動時に前回の last_error.txt を .old に退避する。"""
    path = _last_error_path()
    if not os.path.exists(path):
        return
    old_path = path + ".old"
    try:
        if os.path.exists(old_path):
            os.remove(old_path)
        os.replace(path, old_path)
    except Exception:
        pass


def ensure_packages() -> None:
    """必要な基本パッケージが未インストールの場合は自動インストールする。

    torch / pyannote.audio は巨大なため自動インストールしない。
    話者分離機能を使う場合は手動でインストールすること:
        pip install torch pyannote.audio
    """
    required = {
        "faster_whisper": "faster-whisper",
        "docx":           "python-docx",
    }
    for module, pip_name in required.items():
        try:
            __import__(module)
        except ImportError:
            print(f"[インストール] {pip_name} ...", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


def main() -> None:
    ensure_packages()
    _rotate_last_error()

    try:
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

        ww, wh = 800, 860
        x = (root.winfo_screenwidth()  - ww) // 2
        y = (root.winfo_screenheight() - wh) // 2
        root.geometry(f"{ww}x{wh}+{x}+{y}")
        root.minsize(640, 560)

        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("teams.transcriber")
        except Exception:
            pass

        App(root)
        root.mainloop()

    except Exception:
        tb_str = traceback.format_exc()
        path = _last_error_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(tb_str)
        except Exception:
            pass
        try:
            import tkinter.messagebox as mb
            mb.showerror(
                "起動に失敗しました",
                "詳細は以下のファイルを参照してください。\n\n" + path,
            )
        except Exception:
            print(path, file=sys.stderr)
            print(tb_str, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
