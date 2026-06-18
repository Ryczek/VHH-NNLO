"""Load 7-point scale-refitted A_i coefficients from scale_coefficients.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

ScaleKey = Tuple[float, float]
ScaleDict = Dict[ScaleKey, np.ndarray]

SEVEN_POINT_SCALES: Tuple[ScaleKey, ...] = (
    (0.5, 0.5),
    (1.0, 0.5),
    (0.5, 1.0),
    (1.0, 1.0),
    (2.0, 1.0),
    (1.0, 2.0),
    (2.0, 2.0),
)

SCALE_COEFFICIENTS_FILENAME = "scale_coefficients.json"


def scale_coefficients_path(data_dir: Path) -> Path:
    return data_dir / SCALE_COEFFICIENTS_FILENAME


def _scale_key_from_str(text: str) -> ScaleKey:
    fs, rs = text.split(",", 1)
    return (float(fs), float(rs))


def load_scale_coefficients(path: Path) -> Tuple[Optional[ScaleDict], Optional[ScaleDict]]:
    if not path.is_file():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    lo_raw = payload.get("A_LO_by_scale", {})
    nn_raw = payload.get("A_NNLO_by_scale", {})
    if not lo_raw or not nn_raw:
        return None, None
    lo = {_scale_key_from_str(k): np.asarray(v, dtype=float) for k, v in lo_raw.items()}
    nn = {_scale_key_from_str(k): np.asarray(v, dtype=float) for k, v in nn_raw.items()}
    return lo, nn
