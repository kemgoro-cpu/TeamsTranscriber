"""
WhisperセグメントとTeamsセグメントをタイムスタンプで同期し、
話者情報を付与するモジュール。
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median

from config import (
    DETECT_OFFSET_SAMPLE_N,
    DETECT_OFFSET_MIN_RATIO,
    UNKNOWN_SPEAKER,
)
from teams_transcriber.transcript_parser import TeamsSegment
from teams_transcriber.whisper_runner import WhisperSegment


@dataclass
class SyncedSegment:
    speaker: str
    start: float   # WAV上の時刻（秒）
    end: float
    text: str      # Whisperの高精度テキスト


def _text_similarity(a: str, b: str) -> float:
    """2つの文字列のテキスト類似度を返す（0.0〜1.0）。"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def detect_offset(
    whisper_segs: list[WhisperSegment],
    teams_segs: list[TeamsSegment],
    sample_n: int = DETECT_OFFSET_SAMPLE_N,
) -> float:
    """
    WhisperとTeamsのテキスト類似度を比較し、時間オフセットを自動検出する。

    WhisperとTeamsは同じ音声の転写なので、対応するセグメント同士は
    文字レベルで類似しているはず。この類似度を使って、どのWhisperセグメントが
    どのTeamsセグメントと対応するかを推定し、タイムスタンプの差（オフセット）を算出する。

    Returns:
        offset_sec: このオフセットを引いた値がWAV上の時刻になる。
                    wav_time = teams_time - offset_sec
                    例: Teamsが会議開始120秒でTeamsに対応する音声がWAV上の0秒なら、offset = 120.0
    """
    if not whisper_segs or not teams_segs:
        return 0.0

    anchors = whisper_segs[:sample_n]
    candidate_offsets: list[float] = []

    for w_seg in anchors:
        if not w_seg.text:
            continue

        best_ratio = 0.0
        best_offset = 0.0

        for t_seg in teams_segs:
            if not t_seg.text:
                continue
            ratio = _text_similarity(w_seg.text, t_seg.text)
            if ratio > best_ratio:
                best_ratio = ratio
                # offset = teams_time - wav_time
                best_offset = t_seg.start - w_seg.start

        if best_ratio >= DETECT_OFFSET_MIN_RATIO:
            candidate_offsets.append(best_offset)

    if not candidate_offsets:
        return 0.0

    # 中央値で外れ値を排除
    return median(candidate_offsets)


def synchronize(
    whisper_segs: list[WhisperSegment],
    teams_segs: list[TeamsSegment],
    offset_sec: float,
) -> list[SyncedSegment]:
    """
    各Whisperセグメントに、時間重複が最大のTeamsセグメントの話者を割り当てる。

    Args:
        whisper_segs: Whisperの文字起こし結果
        teams_segs: Teamsのトランスクリプト（話者情報あり）
        offset_sec: Teamsタイムスタンプ → WAV時刻への変換オフセット
                    wav_time = teams_time - offset_sec

    Returns:
        話者情報が付与されたSyncedSegmentのリスト
    """
    if not teams_segs:
        # Teamsセグメントがない場合は話者「不明」で返す
        return [
            SyncedSegment(speaker=UNKNOWN_SPEAKER, start=w.start, end=w.end, text=w.text)
            for w in whisper_segs
        ]

    # Teamsセグメントの時刻をWAV軸に変換（コピーして変更しない）
    adjusted: list[tuple[float, float, str]] = [
        (t.start - offset_sec, t.end - offset_sec, t.speaker)
        for t in teams_segs
    ]

    results: list[SyncedSegment] = []
    j = 0  # Teamsセグメントのポインタ（O(n+m)最適化）

    for w in whisper_segs:
        w_dur = w.end - w.start
        if w_dur <= 0:
            results.append(SyncedSegment(
                speaker=UNKNOWN_SPEAKER, start=w.start, end=w.end, text=w.text
            ))
            continue

        # Whisperセグメントが始まる前に終わっているTeamsセグメントをスキップ
        while j < len(adjusted) - 1 and adjusted[j][1] <= w.start:
            j += 1

        # j から前後のTeamsセグメントを走査して最大重複を探す
        best_overlap = 0.0
        best_speaker = UNKNOWN_SPEAKER
        k = j
        while k < len(adjusted):
            t_start, t_end, t_speaker = adjusted[k]
            if t_start >= w.end:
                break  # これ以降は重複しない
            overlap = min(w.end, t_end) - max(w.start, t_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = t_speaker
            k += 1

        results.append(SyncedSegment(
            speaker=best_speaker,
            start=w.start,
            end=w.end,
            text=w.text,
        ))

    return results
