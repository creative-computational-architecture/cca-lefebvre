
from __future__ import annotations
import math
from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np
from viewer.depth_fill import DEPTH_FILL_FIELD, validated_fill_share
CONF_MEAN_FIELD = 'conf_mean'
N_VIEWS_FIELD = 'n_views'
CONF_MAX = 2.0
N_VIEWS_SATURATION = 3.0
DEFAULT_PIVOT = 0.15
PIVOT_MIN = 0.01
PIVOT_MAX = 1.0
DISPLAY_ALPHA_FLOOR = 0.12
GRADE_UI_LABEL = 'KANIT DERECESI'
GRADE_UNAVAILABLE_LABEL = 'derece yok'

class EvidenceGradeError(ValueError):
    pass

def _floor(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise EvidenceGradeError('floor must be numeric')
    floor = float(value)
    if not math.isfinite(floor):
        raise EvidenceGradeError('floor must be finite')
    if not 0.0 <= floor < 1.0:
        raise EvidenceGradeError(f'floor must be within [0, 1), got {floor}')
    return floor

def _pivot(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise EvidenceGradeError('pivot must be numeric')
    pivot = float(value)
    if not math.isfinite(pivot):
        raise EvidenceGradeError('pivot must be finite')
    if pivot <= 0.0:
        raise EvidenceGradeError('pivot must be positive')
    if not PIVOT_MIN <= pivot <= PIVOT_MAX:
        raise EvidenceGradeError(f'pivot must be between {PIVOT_MIN} and {PIVOT_MAX}, got {pivot}')
    return pivot

def _column(values: object, label: str) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise EvidenceGradeError(f'{label} must be numeric') from exc
    if array.dtype.kind not in 'biuf':
        raise EvidenceGradeError(f'{label} must be numeric')
    if array.ndim != 1:
        raise EvidenceGradeError(f'{label} must be one-dimensional')
    return array.astype(np.float64, copy=False)

@dataclass(frozen=True, slots=True)
class EvidenceGradeState:
    enabled: bool = False
    pivot: float = DEFAULT_PIVOT

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise EvidenceGradeError('enabled must be boolean')
        object.__setattr__(self, 'pivot', _pivot(self.pivot))

@dataclass(frozen=True, slots=True)
class EvidenceGradeResult:
    available: bool
    unavailable_reason: str | None
    grade: np.ndarray | None
    point_count: int

    def __post_init__(self) -> None:
        if self.available and self.unavailable_reason is not None:
            raise EvidenceGradeError('an available grade cannot carry an unavailable reason')
        if self.available and self.grade is None:
            raise EvidenceGradeError('an available grade needs a grade array')
        if not self.available and (not self.unavailable_reason):
            raise EvidenceGradeError('an unavailable grade needs a reason')
        if not self.available and self.grade is not None:
            raise EvidenceGradeError('an unavailable grade cannot carry values')
        if self.grade is not None and len(self.grade) != self.point_count:
            raise EvidenceGradeError('grade length must equal the point count')

    @property
    def status_label(self) -> str:
        return GRADE_UI_LABEL if self.available else GRADE_UNAVAILABLE_LABEL

@dataclass(frozen=True, slots=True)
class EvidenceGradeSummary:
    point_count: int
    pivot: float
    faded_count: int
    zero_count: int
    floor: float = DISPLAY_ALPHA_FLOOR

    @property
    def faded_fraction(self) -> float:
        return self.faded_count / self.point_count if self.point_count else 0.0

    @property
    def zero_fraction(self) -> float:
        return self.zero_count / self.point_count if self.point_count else 0.0

def compute_grade(conf_mean: object, n_views: object, *, fill_share: object | None=None) -> np.ndarray:
    conf = _column(conf_mean, CONF_MEAN_FIELD)
    views = _column(n_views, N_VIEWS_FIELD)
    if conf.shape != views.shape:
        raise EvidenceGradeError(f'{CONF_MEAN_FIELD} and {N_VIEWS_FIELD} must have the same length')
    conf = np.where(np.isnan(conf), 0.0, conf)
    views = np.where(np.isnan(views), 0.0, views)
    conf_term = np.clip(conf / CONF_MAX, 0.0, 1.0)
    view_term = np.clip(views, 0.0, N_VIEWS_SATURATION) / N_VIEWS_SATURATION
    grade = np.clip(conf_term * view_term, 0.0, 1.0)
    if fill_share is not None:
        try:
            shares = validated_fill_share(fill_share)
        except (TypeError, ValueError) as exc:
            raise EvidenceGradeError(f'{DEPTH_FILL_FIELD} must be numeric') from exc
        if shares.shape != grade.shape:
            raise EvidenceGradeError(f'{DEPTH_FILL_FIELD} must have the same length as {CONF_MEAN_FIELD}')
        grade = np.clip(grade * (1.0 - shares.astype(np.float64)), 0.0, 1.0)
    return grade

def display_alpha(grade: object, pivot: float=DEFAULT_PIVOT, *, floor: float=DISPLAY_ALPHA_FLOOR) -> np.ndarray:
    value = _pivot(pivot)
    base = _floor(floor)
    values = _column(grade, 'grade')
    if values.size and (not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0):
        raise EvidenceGradeError('grade values must be finite and within [0, 1]')
    ramp = np.clip(values / value, 0.0, 1.0)
    return np.clip(base + (1.0 - base) * ramp, base, 1.0)

def _lookup(scalars: object) -> Mapping[str, object]:
    if isinstance(scalars, np.ndarray) and scalars.dtype.names:
        return {name: scalars[name] for name in scalars.dtype.names}
    if isinstance(scalars, Mapping):
        return scalars
    keys = getattr(scalars, 'keys', None)
    if callable(keys):
        try:
            return {str(name): scalars[name] for name in keys()}
        except (TypeError, KeyError):
            pass
    raise EvidenceGradeError('scalars must be a mapping of per-point arrays or a structured array')

def resolve_grade(scalars: object) -> EvidenceGradeResult:
    table = _lookup(scalars)
    names = {str(name).lower(): name for name in table}
    missing = [field for field in (CONF_MEAN_FIELD, N_VIEWS_FIELD) if field not in names]
    point_count = 0
    for name in table:
        try:
            point_count = int(len(table[name]))
        except TypeError:
            continue
        break
    if missing:
        return EvidenceGradeResult(available=False, unavailable_reason='source has no ' + ', '.join(missing) + ' evidence column' + ('s' if len(missing) > 1 else ''), grade=None, point_count=point_count)
    fill_name = names.get(DEPTH_FILL_FIELD)
    grade = compute_grade(table[names[CONF_MEAN_FIELD]], table[names[N_VIEWS_FIELD]], fill_share=None if fill_name is None else table[fill_name])
    return EvidenceGradeResult(available=True, unavailable_reason=None, grade=grade, point_count=int(len(grade)))

def summarize_grade(grade: object, pivot: float=DEFAULT_PIVOT) -> EvidenceGradeSummary:
    value = _pivot(pivot)
    values = _column(grade, 'grade')
    return EvidenceGradeSummary(point_count=int(values.size), pivot=value, faded_count=int(np.count_nonzero(values < value)), zero_count=int(np.count_nonzero(values <= 0.0)), floor=DISPLAY_ALPHA_FLOOR)
__all__ = ['CONF_MEAN_FIELD', 'DEFAULT_PIVOT', 'DISPLAY_ALPHA_FLOOR', 'GRADE_UI_LABEL', 'GRADE_UNAVAILABLE_LABEL', 'N_VIEWS_FIELD', 'PIVOT_MAX', 'PIVOT_MIN', 'EvidenceGradeError', 'EvidenceGradeResult', 'EvidenceGradeState', 'EvidenceGradeSummary', 'compute_grade', 'display_alpha', 'resolve_grade', 'summarize_grade']
