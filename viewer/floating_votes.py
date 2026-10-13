
from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
LOGGER = logging.getLogger(__name__)
FLOATING_SIDECAR_SUFFIX = '.floating.npz'
FLOATING_SIDECAR_META_SUFFIX = '.floating.json'
FLOATING_SIDECAR_SCHEMA = 'cca.floating-votes.v1'
FLOATING_VOTE_COLUMNS = ('n_confirm', 'n_through', 'n_through_strong', 'n_confirm_tight')
DEFAULT_FLOATING_THRESHOLD = 0.5
FLOATING_MIN_THROUGH = 2
FLOATING_THRESHOLD_MIN = 0.3
FLOATING_THRESHOLD_MAX = 0.9

@dataclass(frozen=True, slots=True)
class FloatingVotes:
    n_confirm: np.ndarray
    n_through: np.ndarray
    n_through_strong: np.ndarray
    n_confirm_tight: np.ndarray
    source: Path

    def __len__(self) -> int:
        return len(self.n_through)

def floating_hidden(n_through: object, n_confirm: object, *, t: float=DEFAULT_FLOATING_THRESHOLD, min_through: int=FLOATING_MIN_THROUGH) -> np.ndarray:
    through = np.asarray(n_through)
    confirm = np.asarray(n_confirm)
    if through.ndim != 1:
        raise ValueError('n_through must be one-dimensional')
    if confirm.shape != through.shape:
        raise ValueError('n_confirm must carry one value per point')
    through = through.astype(np.float64, copy=False)
    confirm = confirm.astype(np.float64, copy=False)
    total = through + confirm
    ratio = np.zeros(through.shape, dtype=np.float64)
    np.divide(through, total, out=ratio, where=total > 0)
    return (through >= float(min_through)) & (ratio > float(t))

def sidecar_paths(ply_path: Path) -> tuple[Path, Path]:
    source = Path(ply_path)
    name = source.name
    stem = name[:-4] if name.lower().endswith('.ply') else source.stem
    return (source.with_name(stem + FLOATING_SIDECAR_SUFFIX), source.with_name(stem + FLOATING_SIDECAR_META_SUFFIX))

def load_floating_votes(ply_path: Path, n_points: int) -> FloatingVotes | None:
    votes_path, meta_path = sidecar_paths(ply_path)
    if not votes_path.is_file():
        LOGGER.debug('floating votes: no sidecar next to %s', ply_path)
        return None
    if not meta_path.is_file():
        LOGGER.info('floating votes: %s has no %s', votes_path.name, meta_path.name)
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        LOGGER.info('floating votes: cannot read %s (%s)', meta_path.name, error)
        return None
    if not isinstance(meta, dict) or meta.get('schema') != FLOATING_SIDECAR_SCHEMA:
        LOGGER.info('floating votes: %s is not %s', meta_path.name, FLOATING_SIDECAR_SCHEMA)
        return None
    if int(meta.get('n_points', -1)) != int(n_points):
        LOGGER.info('floating votes: %s declares %s points, geometry has %s', meta_path.name, meta.get('n_points'), n_points)
        return None
    try:
        with np.load(votes_path, allow_pickle=False) as data:
            missing = [name for name in FLOATING_VOTE_COLUMNS if name not in data]
            if missing:
                LOGGER.info('floating votes: %s lacks %s', votes_path.name, ', '.join(missing))
                return None
            columns = {name: np.asarray(data[name]) for name in FLOATING_VOTE_COLUMNS}
    except (OSError, ValueError) as error:
        LOGGER.info('floating votes: cannot read %s (%s)', votes_path.name, error)
        return None
    for name, values in columns.items():
        if values.ndim != 1 or len(values) != int(n_points):
            LOGGER.info('floating votes: %s/%s has %s values, geometry has %s', votes_path.name, name, values.shape, n_points)
            return None
    return FloatingVotes(source=votes_path, **columns)
__all__ = ['DEFAULT_FLOATING_THRESHOLD', 'FLOATING_MIN_THROUGH', 'FLOATING_SIDECAR_META_SUFFIX', 'FLOATING_SIDECAR_SCHEMA', 'FLOATING_SIDECAR_SUFFIX', 'FLOATING_THRESHOLD_MAX', 'FLOATING_THRESHOLD_MIN', 'FLOATING_VOTE_COLUMNS', 'FloatingVotes', 'floating_hidden', 'load_floating_votes', 'sidecar_paths']
