"""synchronizer の単体テスト。"""
from __future__ import annotations

from teams_transcriber.transcript_parser import TeamsSegment
from teams_transcriber.synchronizer import SyncedSegment, detect_offset, synchronize
from teams_transcriber.whisper_runner import WhisperSegment


def test_detect_offset_empty() -> None:
    assert detect_offset([], []) == 0.0
    assert detect_offset([], [TeamsSegment("A", 0.0, 10.0, "x")]) == 0.0
    assert detect_offset([WhisperSegment(0.0, 5.0, "x")], []) == 0.0


def test_detect_offset_same_times() -> None:
    """Whisper と Teams が同じタイムスタンプなら offset=0。"""
    w = [WhisperSegment(10.0, 15.0, "同じテキスト")]
    t = [TeamsSegment("話者", 10.0, 15.0, "同じテキスト")]
    assert detect_offset(w, t) == 0.0


def test_detect_offset_teams_120_sec_later() -> None:
    """Teams が 120 秒先行している場合、offset ≈ 120。"""
    w = [WhisperSegment(0.0, 5.0, "こんにちは")]
    t = [TeamsSegment("話者", 120.0, 125.0, "こんにちは")]
    assert abs(detect_offset(w, t) - 120.0) < 0.01


def test_synchronize_empty_teams() -> None:
    w = [WhisperSegment(0.0, 5.0, "テキスト")]
    synced = synchronize(w, [], 0.0)
    assert len(synced) == 1
    assert synced[0].speaker == "不明"
    assert synced[0].text == "テキスト"


def test_synchronize_with_offset() -> None:
    """offset=10: Teams の 10〜15 が WAV の 0〜5 に対応。"""
    w = [WhisperSegment(0.0, 5.0, "発話")]
    t = [TeamsSegment("田中", 10.0, 15.0, "発話")]
    synced = synchronize(w, t, 10.0)
    assert len(synced) == 1
    assert synced[0].speaker == "田中"
    assert synced[0].start == 0.0
    assert synced[0].end == 5.0
