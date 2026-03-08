"""
Hugging Face トークンの取得・保存。

優先順位: 環境変数 HF_TOKEN → OS の credential store（keyring）→ 設定ファイル（後方互換）。
設定ファイルには平文で保存せず、keyring があれば keyring にのみ保存する。
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from teams_transcriber.settings import AppSettings

_SERVICE = "TeamsTranscriber"
_USERNAME_HF = "hf_token"


def get_hf_token(settings: "AppSettings") -> str:
    """
    環境変数 → keyring → settings の順で HF トークンを取得する。
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        import keyring
        token = keyring.get_password(_SERVICE, _USERNAME_HF) or ""
        if token:
            return token
    except Exception:
        pass
    return (settings.get("hf_token") or "").strip()


def set_hf_token(settings: "AppSettings", token: str) -> None:
    """
    HF トークンを keyring に保存する（利用可能な場合）。
    settings.json には保存しないため、保存前に settings から hf_token キーを削除する。
    """
    token = token.strip()
    try:
        import keyring
        if token:
            keyring.set_password(_SERVICE, _USERNAME_HF, token)
        else:
            try:
                keyring.delete_password(_SERVICE, _USERNAME_HF)
            except Exception:
                pass
        settings.remove_key("hf_token")
        return
    except Exception:
        pass
    settings.remove_key("hf_token")
