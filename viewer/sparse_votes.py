
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
LOGGER = logging.getLogger(__name__)
SPARSE_SIDECAR_SUFFIX = '.sparse.npz'
SPARSE_SIDECAR_META_SUFFIX = '.sparse.json'
SPARSE_SIDECAR_SCHEMA = 'cca.sparse-kdist.v1'
SPARSE_KDIST_COLUMN = 'kdist_m'
SPARSE_KNN = 8
DEFAULT_SPARSE_THRESHOLD_M = 0.1
SPARSE_THRESHOLD_MIN_M = 0.06
SPARSE_THRESHOLD_MAX_M = 0.3
SPARSE_DEFAULT_MIN_M = 0.1
SPARSE_DEFAULT_MAX_M = 0.25

@dataclass(frozen=True, slots=True)
class SparseStats:
    n_points: int
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None

    @property
    def default_threshold_m(self) -> float:
        return default_sparse_threshold(self.p95)

def default_sparse_threshold(p95: object) -> float:
    if isinstance(p95, bool) or not isinstance(p95, (int, float, np.floating)):
        return DEFAULT_SPARSE_THRESHOLD_M
    value = float(p95)
    if not np.isfinite(value):
        return DEFAULT_SPARSE_THRESHOLD_M
    return float(min(max(value, SPARSE_DEFAULT_MIN_M), SPARSE_DEFAULT_MAX_M))

def sparse_hidden(kdist_m: object, *, s: float=DEFAULT_SPARSE_THRESHOLD_M) -> np.ndarray:
    values = np.asarray(kdist_m)
    if values.ndim != 1:
        raise ValueError('kdist_m must be one-dimensional')
    values = values.astype(np.float64, copy=False)
    return np.isfinite(values) & (values > float(s))

def sparse_sidecar_paths(ply_path: Path) -> tuple[Path, Path]:
    source = Path(ply_path)
    name = source.name
    stem = name[:-4] if name.lower().endswith('.ply') else source.stem
    return (source.with_name(stem + SPARSE_SIDECAR_SUFFIX), source.with_name(stem + SPARSE_SIDECAR_META_SUFFIX))

def _read_sidecar_meta(meta_path: Path) -> dict | None:
    if not meta_path.is_file():
        LOGGER.debug('sparse kdist: no %s', meta_path.name)
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        LOGGER.info('sparse kdist: cannot read %s (%s)', meta_path.name, error)
        return None
    if not isinstance(meta, dict) or meta.get('schema') != SPARSE_SIDECAR_SCHEMA:
        LOGGER.info('sparse kdist: %s is not %s', meta_path.name, SPARSE_SIDECAR_SCHEMA)
        return None
    return meta

def _percentile(meta: dict, key: str) -> float | None:
    raw = meta.get(key)
    if raw is None or isinstance(raw, bool) or (not isinstance(raw, (int, float))):
        return None
    value = float(raw)
    return value if np.isfinite(value) else None

def load_sparse_stats(ply_path: Path, n_points: int | None=None) -> SparseStats | None:
    _values_path, meta_path = sparse_sidecar_paths(ply_path)
    meta = _read_sidecar_meta(meta_path)
    if meta is None:
        return None
    declared = int(meta.get('n_points', -1))
    if n_points is not None and declared != int(n_points):
        LOGGER.info('sparse kdist: %s declares %s points, geometry has %s', meta_path.name, meta.get('n_points'), n_points)
        return None
    return SparseStats(n_points=declared, p50=_percentile(meta, 'p50'), p90=_percentile(meta, 'p90'), p95=_percentile(meta, 'p95'))

def load_sparse_sidecar(ply_path: Path, n_points: int) -> np.ndarray | None:
    values_path, meta_path = sparse_sidecar_paths(ply_path)
    if not values_path.is_file():
        LOGGER.debug('sparse kdist: no sidecar next to %s', ply_path)
        return None
    meta = _read_sidecar_meta(meta_path)
    if meta is None:
        return None
    if int(meta.get('n_points', -1)) != int(n_points):
        LOGGER.info('sparse kdist: %s declares %s points, geometry has %s', meta_path.name, meta.get('n_points'), n_points)
        return None
    try:
        with np.load(values_path, allow_pickle=False) as data:
            if SPARSE_KDIST_COLUMN not in data:
                LOGGER.info('sparse kdist: %s lacks %s', values_path.name, SPARSE_KDIST_COLUMN)
                return None
            values = np.asarray(data[SPARSE_KDIST_COLUMN])
    except (OSError, ValueError) as error:
        LOGGER.info('sparse kdist: cannot read %s (%s)', values_path.name, error)
        return None
    if values.ndim != 1 or len(values) != int(n_points):
        LOGGER.info('sparse kdist: %s/%s has %s values, geometry has %s', values_path.name, SPARSE_KDIST_COLUMN, values.shape, n_points)
        return None
    return values.astype(np.float32, copy=False)
__all__ = ['DEFAULT_SPARSE_THRESHOLD_M', 'SPARSE_DEFAULT_MAX_M', 'SPARSE_DEFAULT_MIN_M', 'SPARSE_KDIST_COLUMN', 'SPARSE_KNN', 'SPARSE_SIDECAR_META_SUFFIX', 'SPARSE_SIDECAR_SCHEMA', 'SPARSE_SIDECAR_SUFFIX', 'SPARSE_THRESHOLD_MAX_M', 'SPARSE_THRESHOLD_MIN_M', 'SparseStats', 'default_sparse_threshold', 'load_sparse_sidecar', 'load_sparse_stats', 'sparse_hidden', 'sparse_sidecar_paths']
