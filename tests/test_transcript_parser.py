"""transcript_parser の単体テスト。"""
from __future__ import annotations

import tempfile
import os

import pytest

from teams_transcriber.transcript_parser import (
    TeamsSegment,
    is_single_speaker,
    parse_transcript,
    parse_vtt,
)


def test_is_single_speaker_empty() -> None:
    assert is_single_speaker([]) is True


def test_is_single_speaker_one_speaker() -> None:
    segs = [
        TeamsSegment(speaker="田中", start=0.0, end=10.0, text="こんにちは"),
        TeamsSegment(speaker="田中", start=10.0, end=20.0, text="よろしく"),
    ]
    assert is_single_speaker(segs) is True


def test_is_single_speaker_two_speakers() -> None:
    segs = [
        TeamsSegment(speaker="田中", start=0.0, end=10.0, text="こんにちは"),
        TeamsSegment(speaker="佐藤", start=10.0, end=20.0, text="よろしく"),
    ]
    assert is_single_speaker(segs) is False


def test_is_single_speaker_empty_speaker_counted_as_one() -> None:
    segs = [
        TeamsSegment(speaker="", start=0.0, end=5.0, text="テキスト"),
    ]
    assert is_single_speaker(segs) is True


def test_parse_vtt_pattern_a() -> None:
    """<v 話者名>テキスト</v> 形式。"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".vtt", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:05.000\n"
            "<v 田中>こんにちは</v>\n\n"
            "00:00:05.500 --> 00:00:10.000\n"
            "<v 佐藤>よろしくお願いします</v>\n"
        )
        path = f.name
    try:
        segs = parse_vtt(path)
        assert len(segs) == 2
        assert segs[0].speaker == "田中"
        assert segs[0].start == 1.0
        assert segs[0].end == 5.0
        assert "こんにちは" in segs[0].text
        assert segs[1].speaker == "佐藤"
        assert segs[1].start == 5.5
        assert segs[1].end == 10.0
    finally:
        os.unlink(path)


def test_parse_transcript_vtt() -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".vtt", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:03.000\n"
            "<v 話者1>テスト発話</v>\n"
        )
        path = f.name
    try:
        segs = parse_transcript(path)
        assert len(segs) == 1
        assert segs[0].speaker == "話者1"
        assert segs[0].text.strip() == "テスト発話"
    finally:
        os.unlink(path)


def test_parse_transcript_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="サポートされていないファイル形式"):
        parse_transcript("dummy.txt")
