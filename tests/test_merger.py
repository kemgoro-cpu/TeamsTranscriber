"""merger の単体テスト。"""
from __future__ import annotations

from teams_transcriber.synchronizer import SyncedSegment
from teams_transcriber.merger import MergedSegment, merge_consecutive


def test_merge_consecutive_empty() -> None:
    assert merge_consecutive([]) == []


def test_merge_consecutive_single() -> None:
    segs = [SyncedSegment("A", 0.0, 5.0, "テキスト")]
    merged = merge_consecutive(segs)
    assert len(merged) == 1
    assert merged[0].speaker == "A"
    assert merged[0].text == "テキスト"


def test_merge_consecutive_same_speaker_continuous() -> None:
    segs = [
        SyncedSegment("A", 0.0, 2.0, "こんにちは"),
        SyncedSegment("A", 2.0, 4.0, "よろしく"),
    ]
    merged = merge_consecutive(segs, gap_threshold=0.5)
    assert len(merged) == 1
    assert merged[0].speaker == "A"
    assert "こんにちは" in merged[0].text and "よろしく" in merged[0].text


def test_merge_consecutive_two_speakers() -> None:
    segs = [
        SyncedSegment("A", 0.0, 3.0, "発話A"),
        SyncedSegment("B", 3.0, 6.0, "発話B"),
    ]
    merged = merge_consecutive(segs)
    assert len(merged) == 2
    assert merged[0].speaker == "A"
    assert merged[0].text == "発話A"
    assert merged[1].speaker == "B"
    assert merged[1].text == "発話B"


def test_merge_consecutive_gap_breaks_merge() -> None:
    """同一話者でも gap_threshold を超える空白では別行。"""
    segs = [
        SyncedSegment("A", 0.0, 2.0, "前半"),
        SyncedSegment("A", 5.0, 7.0, "後半"),  # 3秒の空白
    ]
    merged = merge_consecutive(segs, gap_threshold=0.5)
    assert len(merged) == 2
    assert merged[0].text == "前半"
    assert merged[1].text == "後半"
