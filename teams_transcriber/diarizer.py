"""
話者分離（Speaker Diarization）モジュール。

バックエンド:
- "diarize": ローカル専用パッケージ（HF不要・完全ローカル）
- "pyannote": pyannote.audio（HFトークンとgatedモデルへのアクセスが必要）
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import (
    DIARIZATION_BACKEND,
    DIARIZATION_MODEL,
    PYANNOTE_EMBEDDING_BATCH_SIZE,
    PYANNOTE_FORCE_CPU,
    PYANNOTE_SEGMENTATION_BATCH_SIZE,
)

# pyannote の hook で使う進捗の重み（合計 1.0）
_DIARIZE_STEP_WEIGHT = {"segmentation": 0.20, "embeddings": 0.70, "other": 0.10}


@dataclass
class DiarizedSegment:
    speaker: str   # "SPEAKER_00", "SPEAKER_01", ...
    start: float   # 秒
    end: float


def _diarize_with_diarize_package(
    wav_path: str,
    progress_callback: Callable[[str, str | None], None] | None = None,
) -> list[DiarizedSegment]:
    """
    diarize パッケージ（FoxNoseTech）で話者分離。HF不要・完全ローカル。
    """
    def _log(msg: str, tag: str | None = None) -> None:
        if progress_callback:
            progress_callback(msg, tag)

    try:
        from diarize import diarize as diarize_fn  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "話者分離に必要なパッケージがありません: pip install diarize\n"
            "（HFトークン不要・完全ローカルで動作します）"
        ) from e

    _log("話者分離を実行中（ローカルエンジン）...", "acc")
    result = diarize_fn(wav_path)

    segments: list[DiarizedSegment] = []
    for seg in result.segments:
        speaker = getattr(seg, "speaker", None)
        if speaker is None:
            continue
        # 数値の場合は SPEAKER_00 形式に統一
        if isinstance(speaker, int):
            speaker = f"SPEAKER_{speaker:02d}"
        else:
            speaker = str(speaker)
        segments.append(DiarizedSegment(
            speaker=speaker,
            start=float(seg.start),
            end=float(seg.end),
        ))

    _log(f"✓ 話者分離完了: {len(segments)} セグメント", "ok")
    speakers = sorted(set(s.speaker for s in segments))
    _log(f"  検出話者数: {len(speakers)} ({', '.join(speakers)})", "dim")
    return segments


def _diarize_with_pyannote(
    wav_path: str,
    hf_token: str,
    num_speakers: int | None,
    progress_callback: Callable[[str, str | None], None] | None = None,
    progress_pct_callback: Callable[[float], None] | None = None,
    return_embeddings: bool = False,
) -> list[DiarizedSegment] | tuple[list[DiarizedSegment], dict[str, np.ndarray]]:
    """pyannote.audio で話者分離（HFトークン・gatedモデルへのアクセス必要）。"""
    _progress_frac = [0.0]  # 0.0〜1.0、mutable で hook から更新

    def _log(msg: str, tag: str | None = None) -> None:
        if progress_callback:
            progress_callback(msg, tag)

    def _set_progress(frac: float) -> None:
        _progress_frac[0] = min(1.0, max(0.0, frac))
        if progress_pct_callback:
            progress_pct_callback(_progress_frac[0])

    def _hook(step_name: str, step_artefact: object, file: object = None, **kwargs: object) -> None:
        completed = kwargs.get("completed")
        total = kwargs.get("total")
        if total is not None and total > 0 and completed is not None:
            w = _DIARIZE_STEP_WEIGHT.get(step_name, _DIARIZE_STEP_WEIGHT["other"])
            base = _progress_frac[0] if step_name != "embeddings" else _DIARIZE_STEP_WEIGHT["segmentation"]
            if step_name == "embeddings":
                frac = _DIARIZE_STEP_WEIGHT["segmentation"] + w * (completed / total)
            else:
                frac = base + w * (completed / total)
            _set_progress(frac)
            _log(f"  話者分離: {step_name} {completed}/{total}", "dim")
        elif step_artefact is not None:
            if step_name == "segmentation":
                _set_progress(_DIARIZE_STEP_WEIGHT["segmentation"])
                _log("  話者分離: セグメント化完了", "dim")
            elif step_name == "embeddings":
                _set_progress(_DIARIZE_STEP_WEIGHT["segmentation"] + _DIARIZE_STEP_WEIGHT["embeddings"])
                _log("  話者分離: 埋め込み完了", "dim")
            elif step_name == "discrete_diarization":
                _set_progress(1.0)
                _log("  話者分離: クラスタリング完了", "dim")

    try:
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "pyannote.audio がインストールされていません: pip install pyannote.audio\n"
            "（完全ローカルのみなら config.DIARIZATION_BACKEND = 'diarize' と pip install diarize を推奨）"
        ) from e

    _log("話者分離モデルを読み込み中（pyannote）...", "acc")
    # 古い pyannote は hf_hub_download(use_auth_token=...) を呼ぶが、新版 huggingface_hub は
    # token= のみ受け付ける。pipeline と model の両方が hf_hub_download を参照しているため、両方パッチする。
    import huggingface_hub
    _real_hf = huggingface_hub.hf_hub_download

    def _hf_hub_download_compat(*args: object, **kwargs: object) -> object:
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _real_hf(*args, **kwargs)

    import pyannote.audio.core.pipeline as _pyannote_pipeline_module
    import pyannote.audio.core.model as _pyannote_model_module
    _pyannote_pipeline_module.hf_hub_download = _hf_hub_download_compat
    _pyannote_model_module.hf_hub_download = _hf_hub_download_compat
    try:
        pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL,
            use_auth_token=hf_token,
        )
    finally:
        _pyannote_pipeline_module.hf_hub_download = _real_hf
        _pyannote_model_module.hf_hub_download = _real_hf

    try:
        import torch
        use_cuda = not PYANNOTE_FORCE_CPU and torch.cuda.is_available()
        if use_cuda:
            try:
                pipeline.to(torch.device("cuda"))
                _log("  GPU (CUDA) を使用", "dim")
            except Exception:
                use_cuda = False
                pipeline.to(torch.device("cpu"))
                _log("  GPU の利用をスキップし CPU を使用", "dim")
        if not use_cuda:
            pipeline.to(torch.device("cpu"))
            if PYANNOTE_FORCE_CPU and torch.cuda.is_available():
                _log("  CPU を使用（config.PYANNOTE_FORCE_CPU=True）", "dim")
            else:
                _log("  CPU を使用", "dim")

        # バッチサイズを拡大すると GPU 時に大幅短縮可能（CPU 時は 1 推奨）
        seg_batch = PYANNOTE_SEGMENTATION_BATCH_SIZE
        emb_batch = PYANNOTE_EMBEDDING_BATCH_SIZE
        if seg_batch is None:
            seg_batch = 16 if use_cuda else 1
        if emb_batch is None:
            emb_batch = 16 if use_cuda else 1
        pipeline.segmentation_batch_size = seg_batch
        pipeline.embedding_batch_size = emb_batch
        if seg_batch > 1 or emb_batch > 1:
            _log(f"  バッチサイズ: segmentation={seg_batch}, embedding={emb_batch}", "dim")
    except Exception:
        _log("  CPU を使用", "dim")
        use_cuda = False

    _log("話者分離を実行中（数分かかる場合があります）...", "acc")
    kwargs: dict = {"hook": _hook}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if return_embeddings:
        kwargs["return_embeddings"] = True

    # torchcodec が未導入 or  import 失敗時でも動くよう、WAV を自前で読み
    # waveform の dict を渡すと pyannote は AudioDecoder を使わない
    try:
        import torchaudio  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "pyannote 用に torchaudio が必要です: pip install torchaudio"
        ) from None
    waveform, sample_rate = torchaudio.load(wav_path)
    # (channel, time) のまま渡す（pyannote の仕様）
    audio_input = {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "uri": "audio",
    }

    try:
        pipeline_out = pipeline(audio_input, **kwargs)
    except Exception as e:
        err_msg = str(e).lower()
        if "403" in err_msg or "401" in err_msg or "gated" in err_msg or "authorized" in err_msg:
            raise RuntimeError(
                "Hugging Face の gated モデルへのアクセスが許可されていません。\n"
                "以下3つのページでそれぞれ「Agree」等を完了してください（同じアカウントでログイン）:\n"
                "  • https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  • https://huggingface.co/pyannote/segmentation-3.0\n"
                "  • https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                "詳しくはプロジェクト内 PYANNOTE_SETUP.md を参照してください。"
            ) from e
        # cuDNN 等の GPU 実行時エラー時は CPU で再実行（落ちずに完了させる）
        try:
            import torch
            _log("  GPU 実行でエラーがあったため CPU で再実行します...", "warn")
            pipeline.to(torch.device("cpu"))
            pipeline.segmentation_batch_size = 1
            pipeline.embedding_batch_size = 1
            pipeline_out = pipeline(audio_input, **kwargs)
        except Exception:
            raise

    # return_embeddings 時は (diarization, centroids) のタプルが返る場合がある
    diarization = pipeline_out
    centroids = None
    if return_embeddings and isinstance(pipeline_out, (list, tuple)) and len(pipeline_out) >= 2:
        diarization = pipeline_out[0]
        centroids = pipeline_out[1]

    # pyannote のバージョンによっては Annotation そのものではなく
    # DiarizeOutput 等のラッパーが返るため、itertracks を持つオブジェクトを取り出す。
    ann = diarization
    if not hasattr(ann, "itertracks"):
        candidate = None
        # よくある属性名を順に試す
        for key in ("speaker_diarization", "diarization", "annotation"):
            val = getattr(diarization, key, None) if not isinstance(diarization, dict) else diarization.get(key)
            if val is not None and callable(getattr(val, "itertracks", None)):
                candidate = val
                break
        # 上で見つからなければ、任意の属性のうち itertracks を持つものを探す
        if candidate is None and not isinstance(diarization, dict):
            for name in dir(diarization):
                if name.startswith("_"):
                    continue
                try:
                    val = getattr(diarization, name)
                    if val is not None and callable(getattr(val, "itertracks", None)):
                        candidate = val
                        break
                except Exception:
                    continue
        if candidate is None and isinstance(diarization, dict):
            for name, val in diarization.items():
                if val is not None and callable(getattr(val, "itertracks", None)):
                    candidate = val
                    break
        if candidate is None:
            typ = type(diarization).__name__
            attrs = list(diarization.keys()) if isinstance(diarization, dict) else [a for a in dir(diarization) if not a.startswith("_")]
            raise RuntimeError(
                "pyannote の出力形式が想定外です（itertracks を持つオブジェクトが見つかりませんでした）。"
                f" 型: {typ}, 属性・キー: {attrs[:20]}"
            )
        ann = candidate

    results: list[DiarizedSegment] = []
    for turn, _track, speaker in ann.itertracks(yield_label=True):
        results.append(DiarizedSegment(
            speaker=speaker,
            start=turn.start,
            end=turn.end,
        ))
    _log(f"✓ 話者分離完了: {len(results)} セグメント", "ok")
    speakers = sorted(set(seg.speaker for seg in results))
    _log(f"  検出話者数: {len(speakers)} ({', '.join(speakers)})", "dim")

    if return_embeddings and centroids is not None:
        # ann.labels() の順序と centroids の行が対応すると仮定
        labels_list = list(ann.labels()) if hasattr(ann, "labels") else list(speakers)
        embeddings_dict: dict[str, np.ndarray] = {}
        cent = np.asarray(centroids)
        if len(cent.shape) == 1:
            cent = cent.reshape(1, -1)
        for i, label in enumerate(labels_list):
            if i < len(cent):
                embeddings_dict[label] = np.asarray(cent[i], dtype=np.float32)
        return (results, embeddings_dict)
    return results


def diarize(
    wav_path: str,
    hf_token: str,
    num_speakers: int | None = None,
    progress_callback: Callable[[str, str | None], None] | None = None,
    progress_pct_callback: Callable[[float], None] | None = None,
    backend: str | None = None,
    return_embeddings: bool = False,
) -> list[DiarizedSegment] | tuple[list[DiarizedSegment], dict[str, np.ndarray]]:
    """
    音声ファイルの話者分離を行う。
    backend が "pyannote" のときは pyannote.audio（hf_token 必須）、
    それ以外はローカルパッケージ "diarize"（HF不要）を使用する。
    backend が None のときは config.DIARIZATION_BACKEND に従う。
    progress_pct_callback は 0.0〜1.0 の進捗を受け取り、pyannote 使用時のみ呼ばれる。
    return_embeddings が True かつ backend が pyannote のとき、
    (セグメントリスト, 話者ラベル→embedding の dict) を返す。それ以外はリストのみ。
    """
    use_backend = (backend or DIARIZATION_BACKEND).strip().lower()
    if use_backend == "pyannote":
        return _diarize_with_pyannote(
            wav_path, hf_token, num_speakers, progress_callback, progress_pct_callback,
            return_embeddings=return_embeddings,
        )
    return _diarize_with_diarize_package(wav_path, progress_callback)
