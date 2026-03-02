"""
faster-whisper を使って音声ファイルを文字起こしするモジュール。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from config import (
    WHISPER_LANGUAGE,
    WHISPER_BEAM_SIZE,
    WHISPER_VAD_FILTER,
)


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str


def transcribe(
    wav_path: str,
    model_name: str,
    language: str = WHISPER_LANGUAGE,
    progress_callback: Callable[[int, str], None] | None = None,
    cancel_flag: threading.Event | None = None,
) -> list[WhisperSegment]:
    """
    faster-whisper で音声ファイルを文字起こしする。

    Args:
        wav_path: 入力WAVファイルパス
        model_name: Whisperモデル名（"tiny"/"base"/"small"/"medium"/"large-v3"）
        language: 言語コード（デフォルト "ja"）
        progress_callback: (セグメント番号, テキストプレビュー) を受け取るコールバック
        cancel_flag: セットされたらキャンセルする threading.Event

    Returns:
        WhisperSegment のリスト（startの昇順）

    Raises:
        ImportError: faster-whisper がインストールされていない場合
        InterruptedError: キャンセルされた場合
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "faster-whisper がインストールされていません: pip install faster-whisper"
        ) from e

    model = WhisperModel(model_name, device="auto", compute_type="int8")

    segments_iter, _info = model.transcribe(
        wav_path,
        language=language,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=WHISPER_VAD_FILTER,
    )

    results: list[WhisperSegment] = []
    for seg in segments_iter:
        if cancel_flag and cancel_flag.is_set():
            raise InterruptedError()

        results.append(WhisperSegment(start=seg.start, end=seg.end, text=seg.text.strip()))

        if progress_callback:
            preview = seg.text.strip()[:30]
            progress_callback(len(results), preview)

    return results
