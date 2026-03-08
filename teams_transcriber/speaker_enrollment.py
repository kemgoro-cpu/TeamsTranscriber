"""
話者登録モジュール。

参照音声（WAV/MP3、任意で区間指定）を登録し、pyannote の embedding を保存する。
話者分離結果との照合で「SPEAKER_00」等を登録名に変換する際に利用する。
pyannote でも diarize（ローカル）でも、登録名での表示が可能（diarize の場合は HFトークン で embedding のみ取得）。
"""
from __future__ import annotations

import json
import os
from typing import Callable
import shutil
import subprocess
import uuid
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from config import (
    DIARIZATION_MODEL,
    FFMPEG_CMD,
    SPEAKER_ENROLLMENT_DIR,
    SPEAKER_ENROLLMENT_MAX_EMBEDDING_DURATION,
    SPEAKER_ENROLLMENT_MIN_SPEECH_RATIO,
    SPEAKER_ENROLLMENT_SIMILARITY_THRESHOLD,
)


@dataclass
class EnrolledSpeaker:
    id: str
    name: str
    audio_path: str
    embedding_path: str
    start_sec: float | None = None
    end_sec: float | None = None
    quality_status: str = "ok"  # "ok" | "multiple_speakers" | "low_speech_ratio"


_INDEX_FILENAME = "index.json"


def _ensure_dir() -> str:
    os.makedirs(SPEAKER_ENROLLMENT_DIR, exist_ok=True)
    return SPEAKER_ENROLLMENT_DIR


def _index_path() -> str:
    return os.path.join(_ensure_dir(), _INDEX_FILENAME)


def _load_index() -> list[dict]:
    path = _index_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_index(entries: list[dict]) -> None:
    with open(_index_path(), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _load_audio_segment(
    audio_path: str,
    start_sec: float | None = None,
    end_sec: float | None = None,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """音声ファイルを読み、指定区間を切り出し (samples, sr) を返す。モノラル・target_sr に正規化。"""
    try:
        import torchaudio  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("話者登録には torchaudio が必要です: pip install torchaudio") from None

    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception:
        # MP3 等で torchaudio が読めない場合は ffmpeg で一時 WAV に変換
        if not shutil.which(FFMPEG_CMD):
            raise
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.close()
        try:
            subprocess.run(
                [FFMPEG_CMD, "-y", "-i", audio_path, "-ac", "1", "-ar", str(target_sr), tmp.name],
                check=True, capture_output=True, timeout=60
            )
            waveform, sr = torchaudio.load(tmp.name)
        finally:
            if os.path.isfile(tmp.name):
                os.unlink(tmp.name)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    arr = waveform.numpy()[0]
    n = len(arr)
    if start_sec is not None or end_sec is not None:
        s = int((start_sec or 0.0) * sr)
        e = int((end_sec or (n / sr)) * sr) if end_sec is not None else n
        s = max(0, min(s, n))
        e = max(s, min(e, n))
        arr = arr[s:e]
    return arr, sr


def _speech_ratio_from_diarization(diarization, duration_sec: float) -> float:
    """発話区間の合計 / 全体の長さ。"""
    if duration_sec <= 0:
        return 0.0
    total = 0.0
    # itertracks() は (segment, track) の2要素。yield_label=True なら (segment, track, label) の3要素
    for turn, *_ in diarization.itertracks():
        total += turn.duration
    return total / duration_sec


def _load_pyannote_pipeline(hf_token: str):
    """pyannote の Pipeline を 1 回だけロードして返す。再利用用。"""
    import torch
    import pyannote.audio.core.pipeline as _pyannote_pipeline_module
    import pyannote.audio.core.model as _pyannote_model_module
    from huggingface_hub import hf_hub_download as _real_hf

    def _hook_compat(*args: object, **kwargs: object) -> object:
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _real_hf(*args, **kwargs)

    _pyannote_pipeline_module.hf_hub_download = _hook_compat
    _pyannote_model_module.hf_hub_download = _hook_compat
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
    finally:
        _pyannote_pipeline_module.hf_hub_download = _real_hf
        _pyannote_model_module.hf_hub_download = _real_hf

    from config import PYANNOTE_FORCE_CPU
    if PYANNOTE_FORCE_CPU or not torch.cuda.is_available():
        pipeline.to(torch.device("cpu"))
    else:
        pipeline.to(torch.device("cuda"))
    return pipeline


def _embedding_only_on_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    pipeline: object,
) -> np.ndarray | None:
    """
    パイプライン内蔵の「声の特徴を出す」部分だけを呼び、1人分の特徴ベクトルを返す。
    失敗時（音声が短い・内部APIが使えない等）は None を返し、呼び出し側でフルパイプラインにフォールバックする。
    """
    emb = getattr(pipeline, "_embedding", None)
    if emb is None or not callable(emb):
        return None
    min_samples = getattr(emb, "min_num_samples", None)
    emb_sr = getattr(emb, "sample_rate", None)
    if min_samples is None or emb_sr is None:
        return None
    try:
        import torch
        import torchaudio  # type: ignore[import-untyped]
        w = torch.from_numpy(waveform).float()
        sr = sample_rate
        if sr != emb_sr:
            w = torchaudio.functional.resample(w.unsqueeze(0), sr, emb_sr).squeeze(0)
        num_samples = w.shape[0]
        if num_samples < min_samples:
            return None
        w = w.unsqueeze(0).unsqueeze(0)
        out = emb(w, masks=None)
        if out is None or not hasattr(out, "shape"):
            return None
        out = np.asarray(out, dtype=np.float32)
        if out.size == 0 or (out.shape[0] == 0):
            return None
        vec = out[0]
        if np.any(np.isnan(vec)):
            return None
        return vec
    except Exception:
        return None


def _run_pipeline_on_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    pipeline: object,
    return_embeddings: bool = True,
) -> tuple:
    """既にロード済みの pipeline で waveform を処理。return (ann, centroids or None)。"""
    import torch
    w = torch.from_numpy(waveform).float().unsqueeze(0)
    audio_input = {"waveform": w, "sample_rate": sample_rate, "uri": "ref"}
    out = pipeline(audio_input, return_embeddings=return_embeddings)
    if return_embeddings and isinstance(out, (list, tuple)) and len(out) == 2:
        diarization, centroids = out[0], out[1]
    else:
        diarization = out
        centroids = None
    ann = diarization
    if not hasattr(ann, "itertracks"):
        for key in ("speaker_diarization", "diarization", "annotation"):
            val = getattr(diarization, key, None)
            if val is not None and callable(getattr(val, "itertracks", None)):
                ann = val
                break
    return ann, centroids


def _run_pyannote_on_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    hf_token: str,
    return_embeddings: bool = True,
):
    """waveform (1d) と sample_rate を渡し、pyannote の pipeline を実行。return (diarization, embeddings or None)。"""
    pipeline = _load_pyannote_pipeline(hf_token)
    return _run_pipeline_on_waveform(waveform, sample_rate, pipeline, return_embeddings)


def add_enrollment(
    name: str,
    audio_path: str,
    hf_token: str,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[str, list[str]]:
    """
    話者を登録する。音声をコピーし、embedding を計算して保存する。

    Returns:
        (登録ID, 品質警告メッセージのリスト)
    """
    _ensure_dir()
    if start_sec is not None and end_sec is not None and start_sec >= end_sec:
        raise ValueError("開始秒は終了秒より小さくしてください")
    if not name or not name.strip():
        raise ValueError("表示名を入力してください")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"音声・動画ファイルが見つかりません: {audio_path}")

    ext = os.path.splitext(audio_path)[1].lower() or ".wav"
    if ext not in (".wav", ".wave", ".mp3", ".mp4", ".mkv", ".webm"):
        ext = ".wav"
    uid = str(uuid.uuid4())[:8]
    dest_audio = os.path.join(SPEAKER_ENROLLMENT_DIR, f"{uid}{ext}")
    shutil.copy2(audio_path, dest_audio)

    try:
        arr, sr = _load_audio_segment(dest_audio, start_sec, end_sec)
    except Exception:
        os.remove(dest_audio)
        raise
    duration_sec = len(arr) / sr
    if duration_sec < 3.0:
        os.remove(dest_audio)
        raise ValueError("登録に使う音声は 3 秒以上必要です")

    ann, centroids = _run_pyannote_on_waveform(arr, sr, hf_token, return_embeddings=True)
    quality_status = "ok"
    warnings: list[str] = []

    labels = list(ann.labels()) if hasattr(ann, "labels") else []
    num_speakers = len(labels)
    if num_speakers >= 2:
        quality_status = "multiple_speakers"
        warnings.append("登録音声に複数話者が含まれている可能性があります。識別精度が落ちることがあります。")

    speech_ratio = _speech_ratio_from_diarization(ann, duration_sec)
    if speech_ratio < SPEAKER_ENROLLMENT_MIN_SPEECH_RATIO:
        if quality_status == "ok":
            quality_status = "low_speech_ratio"
        warnings.append("発話が少ない区間です。登録の精度が落ちる可能性があります。")

    if centroids is None or len(centroids) == 0:
        embedding = np.zeros(512, dtype=np.float32)
    else:
        c0 = centroids[0] if isinstance(centroids, np.ndarray) else np.array(centroids[0], dtype=np.float32)
        embedding = np.asarray(c0, dtype=np.float32)

    emb_path = os.path.join(SPEAKER_ENROLLMENT_DIR, f"{uid}.npy")
    np.save(emb_path, embedding)

    entries = _load_index()
    rel_audio = os.path.basename(dest_audio)
    rel_emb = os.path.basename(emb_path)
    entries.append({
        "id": uid,
        "name": name.strip(),
        "audio_path": rel_audio,
        "embedding_path": rel_emb,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "quality_status": quality_status,
    })
    _save_index(entries)
    return uid, warnings


def list_enrollments() -> list[EnrolledSpeaker]:
    """登録一覧を返す。"""
    entries = _load_index()
    base = SPEAKER_ENROLLMENT_DIR
    result = []
    for e in entries:
        result.append(EnrolledSpeaker(
            id=e["id"],
            name=e["name"],
            audio_path=os.path.join(base, e["audio_path"]),
            embedding_path=os.path.join(base, e["embedding_path"]),
            start_sec=e.get("start_sec"),
            end_sec=e.get("end_sec"),
            quality_status=e.get("quality_status", "ok"),
        ))
    return result


def delete_enrollment(speaker_id: str) -> None:
    """指定 ID の登録を削除する（音声ファイルと embedding も削除）。"""
    entries = _load_index()
    base = SPEAKER_ENROLLMENT_DIR
    new_entries = []
    for e in entries:
        if e["id"] == speaker_id:
            for key in ("audio_path", "embedding_path"):
                p = os.path.join(base, e[key])
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            continue
        new_entries.append(e)
    _save_index(new_entries)


def compute_embeddings_for_diarized_speakers(
    wav_path: str,
    diarized_segs: list,
    hf_token: str,
    min_speaker_duration_sec: float = 1.0,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray]:
    """
    diarize バックエンドで得た話者分離結果について、話者ごとに音声を切り出して
    pyannote で embedding を取得する。返り値は { "SPEAKER_00": vec, ... }。
    パイプラインは 1 回だけロードし全話者で再利用。話者ごとの音声は
    SPEAKER_ENROLLMENT_MAX_EMBEDDING_DURATION 秒までに制限して処理する。
    progress_callback(msg) で進捗メッセージを渡せる。
    """
    from teams_transcriber.diarizer import DiarizedSegment

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    if not diarized_segs or not hf_token:
        return {}
    try:
        arr, sr = _load_audio_segment(wav_path, None, None)
    except Exception:
        return {}

    by_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for seg in diarized_segs:
        if isinstance(seg, DiarizedSegment):
            by_speaker[seg.speaker].append((seg.start, seg.end))

    # 処理対象になる話者だけ先にリスト化（duration 不足でスキップする分を除く概算）
    speakers_to_process: list[tuple[str, list[tuple[float, float]]]] = []
    max_samples = int(SPEAKER_ENROLLMENT_MAX_EMBEDDING_DURATION * sr)
    for speaker, segments in by_speaker.items():
        chunks = []
        for start_sec, end_sec in segments:
            s = int(start_sec * sr)
            e = int(end_sec * sr)
            s = max(0, min(s, len(arr)))
            e = max(s, min(e, len(arr)))
            if e > s:
                chunks.append(arr[s:e])
        if not chunks:
            continue
        concat = np.concatenate(chunks)
        if len(concat) > max_samples:
            concat = concat[:max_samples]
        duration_sec = len(concat) / sr
        if duration_sec >= min_speaker_duration_sec:
            speakers_to_process.append((speaker, segments))
    total = len(speakers_to_process)
    if total == 0:
        return {}

    _log("  パイプラインを読み込み中...")
    pipeline = _load_pyannote_pipeline(hf_token)
    result: dict[str, np.ndarray] = {}
    for idx, (speaker, segments) in enumerate(speakers_to_process, start=1):
        _log(f"  登録名と照合中: {speaker} ({idx}/{total})")
        chunks = []
        for start_sec, end_sec in segments:
            s = int(start_sec * sr)
            e = int(end_sec * sr)
            s = max(0, min(s, len(arr)))
            e = max(s, min(e, len(arr)))
            if e > s:
                chunks.append(arr[s:e])
        concat = np.concatenate(chunks)
        if len(concat) > max_samples:
            concat = concat[:max_samples]
        vec = _embedding_only_on_waveform(concat, sr, pipeline)
        if vec is not None:
            result[speaker] = vec
            continue
        try:
            ann, centroids = _run_pipeline_on_waveform(
                concat, sr, pipeline, return_embeddings=True
            )
            if centroids is not None and len(centroids) > 0:
                c0 = centroids[0] if isinstance(centroids, np.ndarray) else np.array(centroids[0], dtype=np.float32)
                result[speaker] = np.asarray(c0, dtype=np.float32)
        except Exception:
            continue
    return result


def load_enrollment_embeddings() -> list[tuple[str, np.ndarray]]:
    """登録されている (表示名, embedding) のリストを返す。照合用。"""
    enrolled = list_enrollments()
    out = []
    for e in enrolled:
        if not os.path.isfile(e.embedding_path):
            continue
        emb = np.load(e.embedding_path)
        out.append((e.name, np.asarray(emb, dtype=np.float32)))
    return out


def match_speakers_to_enrollment(
    diarized_embeddings: dict[str, np.ndarray],
    enrolled: list[tuple[str, np.ndarray]],
    threshold: float = SPEAKER_ENROLLMENT_SIMILARITY_THRESHOLD,
) -> dict[str, str]:
    """
    話者ラベル → embedding の dict と登録 (名前, embedding) を照合し、
    SPEAKER_XX → 表示名 のマップを返す。類似度が threshold 未満は従来ラベルにしない（呼び出し側でフォールバック）。
    """
    if not enrolled or not diarized_embeddings:
        return {}
    names = [n for n, _ in enrolled]
    embs = np.array([e for _, e in enrolled], dtype=np.float32)
    # 正規化
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs = embs / norms
    used = set()
    speaker_map: dict[str, str] = {}
    # 類似度でソートし、貪欲に 1 対 1 で割り当て
    pairs: list[tuple[float, str, int]] = []
    for spk, vec in diarized_embeddings.items():
        v = np.asarray(vec, dtype=np.float32).flatten()
        v = v / (np.linalg.norm(v) or 1)
        sim = embs @ v
        for i, s in enumerate(sim):
            pairs.append((float(s), spk, i))
    pairs.sort(key=lambda x: -x[0])
    for sim, spk, idx in pairs:
        if spk in speaker_map or idx in used:
            continue
        if sim < threshold:
            continue
        speaker_map[spk] = names[idx]
        used.add(idx)
    return speaker_map
