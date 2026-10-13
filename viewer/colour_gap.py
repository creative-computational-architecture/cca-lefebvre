
from __future__ import annotations
import numpy as np
from viewer.brand import SKIN_C_DARK
COLOUR_GAP_FIELD = 'rgb_kf_dist'
COLOUR_GAP_UNASSIGNED = -1.0
COLOUR_GAP_PLACEHOLDER_RGB = (255, 0, 255)
COLOUR_GAP_SCALAR = 'cca_colour_gap'
COLOUR_GAP_DISPLAY_HEX = SKIN_C_DARK['muted']
COLOUR_GAP_DISPLAY_RGB = tuple((int(COLOUR_GAP_DISPLAY_HEX.lstrip('#')[index:index + 2], 16) for index in (0, 2, 4)))
COLOUR_GAP_UI_LABEL = 'RENK YOK'

def colour_gap_column(schema: object) -> str | None:
    names = {prop.name.lower(): prop.name for prop in schema.vertex.properties}
    return names.get(COLOUR_GAP_FIELD)

def placeholder_mask(colors: np.ndarray) -> np.ndarray:
    return np.all(np.asarray(colors) == np.asarray(COLOUR_GAP_PLACEHOLDER_RGB, dtype=np.uint8), axis=1)

def apply_colour_gap_display(colors: np.ndarray, gap: np.ndarray) -> np.ndarray:
    output = np.array(colors, dtype=np.uint8, copy=True)
    if output.size and gap.any():
        output[gap & placeholder_mask(output)] = COLOUR_GAP_DISPLAY_RGB
    return output
__all__ = ['COLOUR_GAP_DISPLAY_HEX', 'COLOUR_GAP_DISPLAY_RGB', 'COLOUR_GAP_FIELD', 'COLOUR_GAP_PLACEHOLDER_RGB', 'COLOUR_GAP_SCALAR', 'COLOUR_GAP_UI_LABEL', 'COLOUR_GAP_UNASSIGNED', 'apply_colour_gap_display', 'colour_gap_column', 'placeholder_mask']
