"""
pyannote-audio を使った話者分離（Speaker Diarization）モジュール。

HuggingFaceの pyannote/speaker-diarization-3.1 パイプラインを使用し、
音声ファイルから話者ごとのセグメントを生成する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from config import DIARIZATION_MODEL


@dataclass
class DiarizedSegment:
    speaker: str   # "SPEAKER_00", "SPEAKER_01", ...
    start: float   # 秒
    end: float


def diarize(
    wav_path: str,
    hf_token: str,
    num_speakers: int | None = None,
    progress_callback: Callable[[str, str | None], None] | None = None,
) -> list[DiarizedSegment]:
    """
    pyannote-audio で音声ファイルの話者分離を行う。

    Args:
        wav_path: 入力WAVファイルパス
        hf_token: HuggingFace APIトークン（pyannoteモデルへのアクセスに必要）
        num_speakers: 話者数のヒント（Noneなら自動推定）
        progress_callback: (メッセージ, タグ) を受け取るログコールバック

    Returns:
        DiarizedSegment のリスト（時間順）
    """
    def _log(msg: str, tag: str | None = None) -> None:
        if progress_callback:
            progress_callback(msg, tag)

    try:
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "pyannote.audio がインストールされていません: pip install pyannote.audio"
        ) from e

    _log("話者分離モデルを読み込み中...", "acc")
    pipeline = Pipeline.from_pretrained(
        DIARIZATION_MODEL,
        use_auth_token=hf_token,
    )

    # GPU が利用可能なら使用
    try:
        import torch
        if torch.cuda.is_available():
            import torch
            pipeline.to(torch.device("cuda"))
            _log("  GPU (CUDA) を使用", "dim")
        else:
            _log("  CPU を使用", "dim")
    except Exception:
        _log("  CPU を使用", "dim")

    _log("話者分離を実行中（数分かかる場合があります）...", "acc")

    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    diarization = pipeline(wav_path, **kwargs)

    results: list[DiarizedSegment] = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        results.append(DiarizedSegment(
            speaker=speaker,
            start=turn.start,
            end=turn.end,
        ))

    _log(f"✓ 話者分離完了: {len(results)} セグメント", "ok")

    # ユニーク話者数をログ
    speakers = sorted(set(seg.speaker for seg in results))
    _log(f"  検出話者数: {len(speakers)} ({', '.join(speakers)})", "dim")

    return results
