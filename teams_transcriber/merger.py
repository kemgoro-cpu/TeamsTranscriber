"""
連続する同一話者のセグメントをマージするモジュール。
"""
from __future__ import annotations

from dataclasses import dataclass

from config import GAP_THRESHOLD_SEC
from teams_transcriber.synchronizer import SyncedSegment


@dataclass
class MergedSegment:
    speaker: str
    text: str


def merge_consecutive(
    segments: list[SyncedSegment],
    gap_threshold: float = GAP_THRESHOLD_SEC,
) -> list[MergedSegment]:
    """
    連続する同一話者のセグメントを1行にまとめる。

    ただし、gap_threshold 秒以上の空白がある場合は、
    同一話者でも別行として扱う（自然な発話の区切りを維持する）。

    Args:
        segments: 話者情報付きセグメントのリスト
        gap_threshold: この秒数以上の空白は強制的に改行

    Returns:
        MergedSegment のリスト
    """
    if not segments:
        return []

    results: list[MergedSegment] = []
    current_speaker = segments[0].speaker
    current_texts: list[str] = [segments[0].text]
    prev_end = segments[0].end

    for seg in segments[1:]:
        gap = seg.start - prev_end
        same_speaker = seg.speaker == current_speaker
        within_gap = gap < gap_threshold

        if same_speaker and within_gap:
            current_texts.append(seg.text)
        else:
            text = " ".join(current_texts).strip()
            if text:
                results.append(MergedSegment(speaker=current_speaker, text=text))
            current_speaker = seg.speaker
            current_texts = [seg.text]

        prev_end = seg.end

    # 最後のグループを追加
    text = " ".join(current_texts).strip()
    if text:
        results.append(MergedSegment(speaker=current_speaker, text=text))

    return results
