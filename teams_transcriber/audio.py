"""
MP4ファイルから音声（WAV）を抽出するモジュール。
"""
from __future__ import annotations

import re
import shutil
import subprocess
from typing import Callable

from .config import FFMPEG_CMD, FFMPEG_TIMEOUT


def check_ffmpeg() -> bool:
    """ffmpegがPATH上に存在するか確認する。"""
    return shutil.which(FFMPEG_CMD) is not None


def extract_audio(
    mp4_path: str,
    wav_path: str,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """
    MP4ファイルから16kHz・モノラルのWAVファイルを抽出する。

    Whisperは16kHzモノラルのPCMを期待するため、ffmpegで直接変換する。

    Args:
        mp4_path: 入力MP4ファイルパス
        wav_path: 出力WAVファイルパス
        progress_callback: ログメッセージを受け取るコールバック

    Raises:
        FileNotFoundError: ffmpegがPATHに存在しない場合
        RuntimeError: ffmpegが非0終了した場合
        subprocess.TimeoutExpired: タイムアウトした場合
    """
    if not check_ffmpeg():
        raise FileNotFoundError(
            "ffmpegがPATHに見つかりません。https://ffmpeg.org/ からインストールしてください。"
        )

    cmd = [
        FFMPEG_CMD,
        "-y",             # 上書き確認なし
        "-i", mp4_path,
        "-vn",            # 映像トラックを除外
        "-ac", "1",       # モノラル
        "-ar", "16000",   # 16kHz（Whisperのネイティブレート）
        "-sample_fmt", "s16",  # 16bit PCM
        wav_path,
    ]

    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # ffmpegのstderrから進捗を解析してコールバックに流す
    duration_sec: float | None = None
    _re_duration = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)")
    _re_time     = re.compile(r"time=\s*(\d+):(\d+):(\d+\.\d+)")

    assert proc.stderr is not None
    for line in proc.stderr:
        line = line.rstrip()
        if not line:
            continue

        if duration_sec is None:
            m = _re_duration.search(line)
            if m:
                h, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
                duration_sec = h * 3600 + mi * 60 + s

        m = _re_time.search(line)
        if m and progress_callback:
            h, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
            elapsed = h * 3600 + mi * 60 + s
            if duration_sec and duration_sec > 0:
                pct = min(100.0, elapsed / duration_sec * 100)
                progress_callback(f"  音声抽出中... {pct:.0f}%")
            else:
                progress_callback(f"  音声抽出中... {int(elapsed)}秒")

    proc.wait(timeout=FFMPEG_TIMEOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpegが終了コード {proc.returncode} で失敗しました")
