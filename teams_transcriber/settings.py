"""
アプリケーション設定の永続化モジュール。

設定は %APPDATA%\TeamsTranscriber\settings.json に保存されます。
"""
from __future__ import annotations

import json
import os
from typing import Any


class AppSettings:
    def __init__(self) -> None:
        d = os.path.join(
            os.environ.get("APPDATA", os.path.expanduser("~")),
            "TeamsTranscriber",
        )
        self._path = os.path.join(d, "settings.json")
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                self._d: dict[str, Any] = json.load(f)
        except json.JSONDecodeError:
            # 破損した設定ファイルを .bak にリネームして新規作成
            bak = self._path + ".bak"
            if os.path.exists(self._path):
                os.replace(self._path, bak)
            self._d = {}
        except Exception:
            self._d = {}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._d, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._d[key] = value
