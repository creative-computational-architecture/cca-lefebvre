
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from viewer.depth_fill import DEPTH_FILL_FIELD
from viewer.evidence_grade import DISPLAY_ALPHA_FLOOR
from viewer.floating_votes import floating_hidden
from viewer.sparse_votes import sparse_hidden
SIZE_FACTOR_FLOOR = 1.0 / 3.0
SIZE_BINS: tuple[tuple[float, float], ...] = ((SIZE_FACTOR_FLOOR, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 0.999), (0.999, 1.0))
SURFACE_SIZE_MAX = 1.8
SURFACE_SIZE_EDGES = (SIZE_FACTOR_FLOOR, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4, 1.6, SURFACE_SIZE_MAX, float('inf'))
SURFACE_SUPPORT_SIZE_NOTICE = 'Nokta boyutu yerel yuzey destegini ve varsa gorunum alphasini yansitir. Yuzey destegi onayli konum dogrulugu veya olcum belirsizligi degildir.'
SKY_FILL_HIDE_SHARE = 0.5
SKY_FILL_UI_LABEL = 'SENSOR DOLGU'
SKY_FILL_TOGGLE_TEXT = 'SENSOR DOLGU: goster'
SKY_FILL_TOGGLE_TOOLTIP = 'Experimental sensor-fill display label: conf=2 and raw depth>=16.384 m. Labelled points may be hidden from the view. Source records and point counts remain unchanged. This is not a calibrated physical classification.'
SKY_FILL_UNAVAILABLE_REASON = f'bu katmanda {DEPTH_FILL_FIELD} sutunu veya kanit derecesi yok; gokyuzu dolgusu ayirt edilemez, hicbir nokta gizlenmez'
FLOATING_UI_LABEL = 'HAVADA'
FLOATING_TOGGLE_TEXT = 'HAVADA: goster'
FLOATING_TOGGLE_TOOLTIP = 'Experimental free-space votes may hide points in the view when at least two supporting views and the selected ratio threshold are available. Source records are retained. This is a display heuristic, not certified measurement evidence.'
FLOATING_SLIDER_TOOLTIP = 'Esik t: dusuk uc daha cok nokta gizler, yuksek uc daha az. Nokta sayisini degistirmez, yalnizca cizilip cizilmedigini.'
FLOATING_UNAVAILABLE_REASON = 'Free-space vote sidecar (.floating.npz) is unavailable. No points are hidden by this heuristic.'
SPARSE_UI_LABEL = 'SEYREK'
SPARSE_TOGGLE_TEXT = 'SEYREK: goster'
SPARSE_TOGGLE_TOOLTIP = 'The mean distance to eight neighbouring points is compared to the selected threshold. Sparse points may be hidden from the view. Thin real surfaces may also be sparse. Source records remain unchanged; this is not certified measurement evidence.'
SPARSE_SLIDER_TOOLTIP = 'Esik s (metre): dusuk uc daha cok nokta gizler, yuksek uc daha az. Nokta sayisini degistirmez, yalnizca cizilip cizilmedigini.'
SPARSE_UNAVAILABLE_REASON = 'Neighbour-distance sidecar (.sparse.npz) is unavailable. No points are hidden by this heuristic.'

def size_factor(alpha: object, *, floor: float=DISPLAY_ALPHA_FLOOR, surface_support: object | None=None) -> np.ndarray:
    support = None
    if surface_support is not None:
        support = np.asarray(surface_support, dtype=np.float64)
        if support.ndim != 1 or not np.isfinite(support).all():
            raise ValueError('surface_support must be a finite one-dimensional array')
        if np.any((support < 0.0) | (support > 1.0)):
            raise ValueError('surface_support must be within [0, 1]')
    values = np.ones(support.shape, dtype=np.float64) if alpha is None and support is not None else np.asarray(alpha, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError('alpha must be one-dimensional')
    if not 0.0 <= float(floor) < 1.0:
        raise ValueError(f'floor must be within [0, 1), got {floor}')
    if support is not None:
        if support.shape != values.shape:
            raise ValueError('surface_support must carry one value per alpha')
        values = np.nan_to_num(values, nan=float(floor), posinf=1.0, neginf=float(floor))
    ramp = np.clip((values - float(floor)) / (1.0 - float(floor)), 0.0, 1.0)
    factors = SIZE_FACTOR_FLOOR + (1.0 - SIZE_FACTOR_FLOOR) * ramp
    if support is None:
        return factors
    return np.clip(factors * (0.65 + 1.15 * support), SIZE_FACTOR_FLOOR, SURFACE_SIZE_MAX)

@dataclass(frozen=True, slots=True)
class SizeBucket:
    mask: np.ndarray
    factor: float

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.mask))

def size_buckets(alpha: object, *, floor: float=DISPLAY_ALPHA_FLOOR, surface_support: object | None=None) -> tuple[SizeBucket, ...]:
    factors = size_factor(alpha, floor=floor, surface_support=surface_support)
    if surface_support is not None:
        buckets = []
        for low, high in zip(SURFACE_SIZE_EDGES[:-1], SURFACE_SIZE_EDGES[1:]):
            mask = (factors >= low) & (factors < high)
            if mask.any():
                buckets.append(SizeBucket(mask, float(factors[mask].mean())))
        return tuple(buckets)
    buckets: list[SizeBucket] = []
    for low, high in SIZE_BINS:
        mask = (factors >= low) & (factors < high)
        if not mask.any():
            continue
        buckets.append(SizeBucket(mask, float(factors[mask].mean())))
    opaque = factors >= 1.0
    if opaque.any():
        buckets.append(SizeBucket(opaque, 1.0))
    return tuple(buckets)

def sky_fill_hidden(fill_labelled: object, grade: object) -> np.ndarray:
    labelled = np.asarray(fill_labelled)
    if labelled.ndim != 1:
        raise ValueError('fill_labelled must be one-dimensional')
    values = np.asarray(grade, dtype=np.float64)
    if values.shape != labelled.shape:
        raise ValueError('grade must carry one value per point')
    return np.asarray(labelled, dtype=bool) & (values <= 0.0)
__all__ = ['FLOATING_SLIDER_TOOLTIP', 'FLOATING_TOGGLE_TEXT', 'FLOATING_TOGGLE_TOOLTIP', 'FLOATING_UI_LABEL', 'FLOATING_UNAVAILABLE_REASON', 'SIZE_BINS', 'SIZE_FACTOR_FLOOR', 'SKY_FILL_HIDE_SHARE', 'SKY_FILL_TOGGLE_TEXT', 'SKY_FILL_TOGGLE_TOOLTIP', 'SKY_FILL_UI_LABEL', 'SKY_FILL_UNAVAILABLE_REASON', 'SPARSE_SLIDER_TOOLTIP', 'SPARSE_TOGGLE_TEXT', 'SPARSE_TOGGLE_TOOLTIP', 'SPARSE_UI_LABEL', 'SPARSE_UNAVAILABLE_REASON', 'SURFACE_SIZE_MAX', 'SURFACE_SUPPORT_SIZE_NOTICE', 'SizeBucket', 'floating_hidden', 'size_buckets', 'size_factor', 'sky_fill_hidden', 'sparse_hidden']
