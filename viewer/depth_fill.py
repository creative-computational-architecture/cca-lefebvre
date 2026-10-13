
from __future__ import annotations
import numpy as np
from viewer.colour_gap import COLOUR_GAP_DISPLAY_RGB
DEPTH_FILL_FIELD = 'fill_share'
DEPTH_FILL_PRIOR_M = 16.384
DEPTH_FILL_CONF = 2
DEPTH_FILL_SCALAR = 'cca_depth_fill'
DEPTH_FILL_UI_LABEL = 'SENSOR DOLGU'
DEPTH_FILL_LEGEND_SHARE = 0.5
DEPTH_FILL_DISPLAY_RGB = COLOUR_GAP_DISPLAY_RGB

def depth_fill_column(schema: object) -> str | None:
    names = {prop.name.lower(): prop.name for prop in schema.vertex.properties}
    return names.get(DEPTH_FILL_FIELD)

def validated_fill_share(values: object) -> np.ndarray:
    shares = np.asarray(values, dtype=np.float64)
    if shares.ndim != 1:
        raise ValueError(f'{DEPTH_FILL_FIELD} must be one-dimensional')
    shares = np.where(np.isnan(shares), 0.0, shares)
    return np.clip(shares, 0.0, 1.0).astype(np.float32)

def apply_depth_fill_display(colors: np.ndarray, fill_share: object) -> np.ndarray:
    source = np.asarray(colors, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('colours must be an Nx3 array')
    shares = validated_fill_share(fill_share).astype(np.float64)
    if shares.shape != (len(source),):
        raise ValueError(f'{DEPTH_FILL_FIELD} must carry one value per colour')
    muted = np.asarray(DEPTH_FILL_DISPLAY_RGB, dtype=np.float64)
    blended = source * (1.0 - shares)[:, None] + muted[None, :] * shares[:, None]
    return np.clip(np.rint(blended), 0.0, 255.0).astype(np.uint8)

def labelled_count(fill_share: object) -> int:
    return int(np.count_nonzero(validated_fill_share(fill_share) >= DEPTH_FILL_LEGEND_SHARE))
__all__ = ['DEPTH_FILL_CONF', 'DEPTH_FILL_DISPLAY_RGB', 'DEPTH_FILL_FIELD', 'DEPTH_FILL_LEGEND_SHARE', 'DEPTH_FILL_PRIOR_M', 'DEPTH_FILL_SCALAR', 'DEPTH_FILL_UI_LABEL', 'apply_depth_fill_display', 'depth_fill_column', 'labelled_count', 'validated_fill_share']
