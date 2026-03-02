"""
Teamsのトランスクリプトファイル（.vtt / .docx）を解析するモジュール。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass
class TeamsSegment:
    speaker: str
    start: float   # 会議開始からの秒数
    end: float
    text: str


# ── 内部ヘルパー ──────────────────────────────────────────────────────────────

def _parse_vtt_time(ts: str) -> float:
    """'HH:MM:SS.mmm' または 'MM:SS.mmm' → 秒（float）。"""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    raise ValueError(f"タイムスタンプを解析できません: {ts!r}")


def _strip_html(text: str) -> str:
    """HTMLタグを除去し、エンティティをデコードする。"""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
    return text.strip()


def _parse_hms(ts: str) -> float:
    """'H:MM:SS' または 'MM:SS' → 秒（float）。"""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    raise ValueError(f"タイムスタンプを解析できません: {ts!r}")


# ── VTT解析 ──────────────────────────────────────────────────────────────────

_RE_TIMING = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{3})"
)
_RE_V_TAG = re.compile(r"<v\s+([^>]+)>")
_RE_TIMESTAMP_TAG = re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>")


def parse_vtt(vtt_path: str) -> list[TeamsSegment]:
    """
    WebVTT形式のTeamsトランスクリプトを解析する。

    対応フォーマット:
        Pattern A: '<v 話者名>テキスト</v>'
        Pattern B: 'TIMING\\n話者名\\nテキスト'
    """
    with open(vtt_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    # ダブル改行でcueブロックに分割
    blocks = re.split(r"\n\n+", content.strip())
    segments: list[TeamsSegment] = []

    for block in blocks:
        lines = [l.rstrip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        # タイミング行を探す
        timing_line = None
        timing_idx = -1
        for i, line in enumerate(lines):
            m = _RE_TIMING.search(line)
            if m:
                timing_line = m
                timing_idx = i
                break

        if timing_line is None:
            continue

        start = _parse_vtt_time(timing_line.group(1).replace(",", "."))
        end   = _parse_vtt_time(timing_line.group(2).replace(",", "."))

        # タイミング行以降のテキスト行を結合
        text_lines = lines[timing_idx + 1:]
        raw_text = " ".join(text_lines)

        # Pattern A: <v 話者名>テキスト
        vm = _RE_V_TAG.search(raw_text)
        if vm:
            speaker = vm.group(1).strip()
            text = _RE_TIMESTAMP_TAG.sub("", raw_text)
            text = _strip_html(text)
        else:
            # Pattern B: 最初の行が話者名、残りがテキスト
            if len(text_lines) >= 2:
                speaker = _strip_html(text_lines[0])
                text = _strip_html(" ".join(text_lines[1:]))
            elif len(text_lines) == 1:
                # 話者情報なし
                speaker = "不明"
                text = _strip_html(text_lines[0])
            else:
                continue

        text = text.strip()
        if not text:
            continue

        segments.append(TeamsSegment(speaker=speaker, start=start, end=end, text=text))

    return segments


# ── DOCX解析 ─────────────────────────────────────────────────────────────────

_RE_TIMESTAMP_HMS = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
_RE_SPEAKER_TIME_INLINE = re.compile(r"^(.+?)\s{2,}(\d{1,2}:\d{2}:\d{2})$")


def parse_docx(docx_path: str) -> list[TeamsSegment]:
    """
    Word形式のTeamsトランスクリプトを解析する。

    Teamsが出力するDOCXには主に以下のパターンがある:
        Pattern 1:
            話者名
            H:MM:SS
            テキスト...
        Pattern 2:
            話者名  H:MM:SS  （同一段落、スペース区切り）
            テキスト...
    """
    try:
        from docx import Document  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("python-docx がインストールされていません: pip install python-docx") from e

    doc = Document(docx_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs]

    segments: list[TeamsSegment] = []
    i = 0
    n = len(paragraphs)

    while i < n:
        para = paragraphs[i]
        if not para:
            i += 1
            continue

        speaker: str | None = None
        start: float | None = None

        # Pattern 2: "話者名  0:05:30" が同一行
        m = _RE_SPEAKER_TIME_INLINE.match(para)
        if m:
            speaker = m.group(1).strip()
            try:
                start = _parse_hms(m.group(2))
            except ValueError:
                pass

        # Pattern 1: 次の行がタイムスタンプか確認
        if speaker is None and i + 1 < n and _RE_TIMESTAMP_HMS.match(paragraphs[i + 1]):
            speaker = para
            try:
                start = _parse_hms(paragraphs[i + 1])
                i += 1  # タイムスタンプ行をスキップ
            except ValueError:
                speaker = None

        if speaker is None or start is None:
            i += 1
            continue

        # 次のセグメント開始まで続くテキストを収集
        i += 1
        text_parts: list[str] = []
        while i < n:
            p = paragraphs[i]
            if not p:
                i += 1
                continue
            # 次の話者行かどうか判定
            is_next_speaker = False
            if _RE_TIMESTAMP_HMS.match(p):
                # スタンドアロンのタイムスタンプ → 次セグメントの時刻
                is_next_speaker = True
            elif i + 1 < n and _RE_TIMESTAMP_HMS.match(paragraphs[i + 1]):
                # 次の行がタイムスタンプ → この行が次の話者名
                is_next_speaker = True
            elif _RE_SPEAKER_TIME_INLINE.match(p):
                is_next_speaker = True

            if is_next_speaker:
                break
            text_parts.append(p)
            i += 1

        text = " ".join(text_parts).strip()
        if not text:
            continue

        segments.append(TeamsSegment(speaker=speaker, start=start, end=0.0, text=text))

    # end を次セグメントの start から補完
    for j in range(len(segments) - 1):
        segments[j].end = segments[j + 1].start
    if segments:
        last = segments[-1]
        # 最後のセグメントはテキスト文字数から推定（約3文字/秒）
        last.end = last.start + max(5.0, len(last.text) / 3.0)

    return segments


# ── 公開インターフェース ──────────────────────────────────────────────────────

def parse_transcript(path: str) -> list[TeamsSegment]:
    """拡張子から形式を自動判定し、解析結果を返す。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vtt":
        return parse_vtt(path)
    elif ext in (".docx", ".doc"):
        return parse_docx(path)
    else:
        raise ValueError(f"サポートされていないファイル形式です: {ext!r}（.vtt または .docx を指定してください）")
