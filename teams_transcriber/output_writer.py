"""
文字起こし結果をテキストファイルに書き出すモジュール。
"""
from __future__ import annotations

from .config import OUTPUT_ENCODING
from teams_transcriber.merger import MergedSegment


def write_output(segments: list[MergedSegment], output_path: str) -> None:
    """
    マージ済みセグメントを '話者名: テキスト' 形式で1行ずつ書き出す。

    エンコードは utf-8-sig（BOM付きUTF-8）。
    Windows のメモ帳・Word でそのまま開ける。

    Args:
        segments: マージ済みセグメントのリスト
        output_path: 出力ファイルパス
    """
    with open(output_path, "w", encoding=OUTPUT_ENCODING) as f:
        for seg in segments:
            f.write(f"{seg.speaker}: {seg.text}\n")
