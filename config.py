"""
TeamsTranscriber — 設定定数モジュール。
"""
from __future__ import annotations

import os

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
WHISPER_VAD_FILTER  = False   # 無音区間スキップ（日本語の間合いに有効）

# 同期・マージ
GAP_THRESHOLD_SEC       = 0.5   # 同一話者でもこの秒数以上の空白は別行扱い
DETECT_OFFSET_SAMPLE_N  = 30    # オフセット自動検出に使うWhisperセグメント数
DETECT_OFFSET_MIN_RATIO = 0.3   # 採用する最低テキスト類似度

# 出力
OUTPUT_ENCODING = "utf-8-sig"  # BOM付きUTF-8（Windowsメモ帳・Word対応）
UNKNOWN_SPEAKER = "不明"

# 話者分離
# バックエンド: "diarize" = ローカル専用（HF不要・短時間）, "pyannote" = pyannote.audio（HF要・高精度だが CPU では遅い）
# ※ 処理を短時間で完了させたい場合は "diarize" のままにし、GUI でも「ローカル（diarize）」を選ぶ
DIARIZATION_BACKEND        = "diarize"
DIARIZATION_MODEL          = "pyannote/speaker-diarization-3.1"  # pyannote 時のみ使用
DIARIZATION_SPEAKER_PREFIX = "話者"  # SPEAKER_00 → 話者1 に変換
# pyannote のバッチサイズ（None=自動: GPU時は16、CPU時は1）。高速化には GPU 利用を推奨
PYANNOTE_SEGMENTATION_BATCH_SIZE: int | None = None
PYANNOTE_EMBEDDING_BATCH_SIZE: int | None = None
# True にすると pyannote を常に CPU で実行。False なら GPU を試し、失敗時は自動で CPU に切り替え
PYANNOTE_FORCE_CPU = True

# 話者登録（pyannote 使用時のみ有効）
SPEAKER_ENROLLMENT_DIR: str = os.path.join(
    os.environ.get("APPDATA", os.path.expanduser("~")),
    "TeamsTranscriber",
    "speaker_enrollment",
)
# 登録話者と分離結果の照合で使う類似度の最低値（これ未満は「不明」のまま）
SPEAKER_ENROLLMENT_SIMILARITY_THRESHOLD = 0.5
# 品質チェック: 発話区間の割合がこの値未満なら「発話少なめ」警告
SPEAKER_ENROLLMENT_MIN_SPEECH_RATIO = 0.3
# diarize で照合するとき、話者ごとの音声をこの秒数までに制限して embedding 取得（長いと遅いため）
SPEAKER_ENROLLMENT_MAX_EMBEDDING_DURATION = 120
