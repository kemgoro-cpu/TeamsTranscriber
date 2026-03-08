"""
UIテーマカラー定数モジュール。

ダークテーマ。入力欄は背景を明るめにして視認性を確保する。
"""
from __future__ import annotations

# ダークテーマカラーパレット（ティール系アクセント）
THEME: dict[str, str] = {
    "bg":       "#0d0d10",
    "bg2":      "#12121a",
    "card":     "#18182a",
    "card2":    "#1e1e32",
    "input":    "#2d2d42",
    "input_fg": "#e8e8f0",
    "accent":   "#00a896",
    "accent2":  "#00c4b0",
    "ok":       "#00b894",
    "warn":     "#f0c14b",
    "err":      "#e17055",
    "fg":       "#f2f2f8",
    "fg2":      "#a0a0b8",
    "dim":      "#606078",
    "border":   "#2a2a40",
    "trbg":     "#252538",
}
