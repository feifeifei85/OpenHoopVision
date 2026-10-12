from collections.abc import Mapping
from typing import Any

import cv2


def text_dimensions(text: str, font_scale: float, thickness: int) -> tuple[tuple[int, int], int]:
    """Measure a label using the shared Hershey simplex font."""
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)


def player_overlay_label(player: Mapping[str, Any]) -> str:
    """Prefer a known jersey number over the current identity label."""
    number = player.get("jersey_number")
    if number is not None:
        return f"#{number}"
    return player.get("identity_label") or ""

__all__ = ["player_overlay_label", "text_dimensions"]
