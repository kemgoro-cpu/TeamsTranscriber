"""
TeamsTranscriber — 設定定数モジュール。
"""
from __future__ import annotations

APP_NAME    = "Teams Transcriber"
APP_VERSION = "1.0.0"

# ffmpeg
FFMPEG_CMD     = "ffmpeg"
FFMPEG_TIMEOUT = 7200  # 秒（2時間録画を想定）

# Whisper
WHISPER_MODELS   = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
DEFAULT_MODEL    = "small"
WHISPER_LANGUAGE = "ja"
WHISPER_BEAM_SIZE   = 5
WHISPER_VAD_FILTER  = True   # 無音区間スキップ（日本語の間合いに有効）

# 同期・マージ
GAP_THRESHOLD_SEC       = 3.0   # 同一話者でもこの秒数以上の空白は別行扱い
DETECT_OFFSET_SAMPLE_N  = 30    # オフセット自動検出に使うWhisperセグメント数
DETECT_OFFSET_MIN_RATIO = 0.3   # 採用する最低テキスト類似度

# 出力
OUTPUT_ENCODING = "utf-8-sig"  # BOM付きUTF-8（Windowsメモ帳・Word対応）
UNKNOWN_SPEAKER = "不明"
