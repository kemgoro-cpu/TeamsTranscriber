"""
faster-whisper を使って音声ファイルを文字起こしするモジュール。
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config import (
    WHISPER_LANGUAGE,
    WHISPER_BEAM_SIZE,
    WHISPER_VAD_FILTER,
)

# モデルごとの概算ダウンロードサイズ（MB）
_APPROX_SIZE_MB: dict[str, int] = {
    "tiny":             74,
    "base":            141,
    "small":           466,
    "medium":         1500,
    "large-v3":       3100,
    "large-v3-turbo": 1600,
}


@dataclass
class WhisperSegment:
    start: float
    end: float
    text: str


def _hf_hub_cache() -> Path:
    """HuggingFaceのhubキャッシュディレクトリを返す。"""
    if "HF_HUB_CACHE" in os.environ:
        return Path(os.environ["HF_HUB_CACHE"])
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"]) / "hub"
    if "HUGGINGFACE_HUB_CACHE" in os.environ:
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    return Path.home() / ".cache" / "huggingface" / "hub"


def _is_model_cached(model_name: str) -> bool:
    """モデルがローカルキャッシュ済みかどうかを確認する。"""
    repo_id = f"Systran/faster-whisper-{model_name}"
    repo_dir = _hf_hub_cache() / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return False
    for snap in snapshots.iterdir():
        if snap.is_dir() and any(snap.glob("model.*")):
            return True
    return False


def _ensure_model_downloaded(
    model_name: str,
    log: Callable[[str], None] | None = None,
    proxy_url: str | None = None,
) -> None:
    """
    モデルが未ダウンロードの場合、進捗を表示しながらダウンロードする。
    キャッシュ済みの場合は何もしない。

    キャッシュディレクトリのサイズを3秒ごとに監視し、
    取得済みMBとパーセントをログに流す。

    Args:
        proxy_url: プロキシURL（例: "http://proxy.example.com:8080"）。
                   省略時は環境変数 HTTPS_PROXY / HTTP_PROXY を使用。
    """
    if _is_model_cached(model_name):
        if log:
            log(f"  キャッシュ済みモデル使用: {model_name}")
        return

    approx_mb = _APPROX_SIZE_MB.get(model_name, 0)
    if log:
        size_hint = f"（約 {approx_mb} MB）" if approx_mb else ""
        log(f"  モデル「{model_name}」を初回ダウンロード中... {size_hint}")
        log("  ※ ネットワーク速度によっては数分〜十数分かかります")

    repo_id = f"Systran/faster-whisper-{model_name}"
    repo_dir = _hf_hub_cache() / f"models--{repo_id.replace('/', '--')}"
    approx_bytes = approx_mb * 1024 * 1024

    # プロキシ設定（環境変数経由が huggingface_hub / requests に最も確実に効く）
    _orig_env: dict[str, str] = {}
    if proxy_url:
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            _orig_env[key] = os.environ.get(key, "")
            os.environ[key] = proxy_url

    stop_event = threading.Event()

    def _monitor() -> None:
        last_size = -1
        while not stop_event.is_set():
            try:
                size = sum(
                    f.stat().st_size for f in repo_dir.rglob("*") if f.is_file()
                )
                if size != last_size and log:
                    mb = size / 1024 / 1024
                    if approx_bytes > 0:
                        pct = min(99, int(size / approx_bytes * 100))
                        log(f"  ダウンロード中... {mb:.0f}/{approx_mb} MB ({pct}%)")
                    else:
                        log(f"  ダウンロード中... {mb:.0f} MB 取得済み")
                    last_size = size
            except Exception:
                pass
            stop_event.wait(3)

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()

    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
        snapshot_download(repo_id=repo_id)
        if log:
            total_mb = sum(
                f.stat().st_size for f in repo_dir.rglob("*") if f.is_file()
            ) / 1024 / 1024
            log(f"  ✓ ダウンロード完了: {total_mb:.0f} MB")
    except Exception as e:
        if log:
            log(f"  ⚠ ダウンロードエラー: {e}")
        raise
    finally:
        stop_event.set()
        monitor.join(timeout=5)
        # 環境変数を元に戻す
        if proxy_url:
            for key, val in _orig_env.items():
                if val:
                    os.environ[key] = val
                else:
                    os.environ.pop(key, None)


def transcribe(
    wav_path: str,
    model_name: str,
    language: str = WHISPER_LANGUAGE,
    progress_callback: Callable[[int, str], None] | None = None,
    download_callback: Callable[[str], None] | None = None,
    proxy_url: str | None = None,
    cancel_flag: threading.Event | None = None,
) -> list[WhisperSegment]:
    """
    faster-whisper で音声ファイルを文字起こしする。

    Args:
        wav_path: 入力WAVファイルパス
        model_name: Whisperモデル名
        language: 言語コード（デフォルト "ja"）
        progress_callback: (セグメント番号, テキストプレビュー) を受け取るコールバック
        download_callback: ダウンロード進捗メッセージ（文字列）を受け取るコールバック
        proxy_url: プロキシURL（初回ダウンロード時のみ使用）
        cancel_flag: セットされたらキャンセルする threading.Event

    Raises:
        ImportError: faster-whisper がインストールされていない場合
        InterruptedError: キャンセルされた場合
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "faster-whisper がインストールされていません: pip install faster-whisper"
        ) from e

    # 初回のみダウンロード（キャッシュ済みなら即座に返る）
    _ensure_model_downloaded(model_name, log=download_callback, proxy_url=proxy_url)

    if cancel_flag and cancel_flag.is_set():
        raise InterruptedError()

    model = WhisperModel(model_name, device="auto", compute_type="int8")

    segments_iter, _info = model.transcribe(
        wav_path,
        language=language,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=WHISPER_VAD_FILTER,
    )

    results: list[WhisperSegment] = []
    for seg in segments_iter:
        if cancel_flag and cancel_flag.is_set():
            raise InterruptedError()

        results.append(WhisperSegment(start=seg.start, end=seg.end, text=seg.text.strip()))

        if progress_callback:
            preview = seg.text.strip()[:30]
            progress_callback(len(results), preview)

    return results
