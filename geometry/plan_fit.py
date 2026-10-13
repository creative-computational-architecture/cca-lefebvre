
from __future__ import annotations
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from geometry.placement import placement_matrix
from geometry.ply_preserve import iter_ply_vertex_chunks, read_ply_schema

class PlanFitError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class PlanFitConfig:
    grid_m: float = 0.25
    angle_step_deg: float = 5.0
    candidate_count: int = 8
    peaks_per_angle: int = 2
    match_tolerance_m: float = 0.5
    max_scan_cells: int = 60000
    min_control_cells: int = 12
    min_specificity_gain: float = 0.1
    max_holdout_gap: float = 0.15
    competing_score_gap: float = 0.03
    candidate_yaws_deg: tuple[float, ...] | None = None
    structural_layer_tokens: tuple[str, ...] = ('duvar', 'kolon', 'kapi', 'pencere', 'halihazr', 'halihazir')

    def __post_init__(self) -> None:
        numeric = (self.grid_m, self.angle_step_deg, self.match_tolerance_m)
        if not all((math.isfinite(float(value)) and float(value) > 0.0 for value in numeric)):
            raise PlanFitError('plan-fit numeric settings must be positive and finite')
        if self.angle_step_deg > 90.0:
            raise PlanFitError('plan-fit angle step cannot exceed 90 degrees')
        if self.candidate_count < 1 or self.peaks_per_angle < 1:
            raise PlanFitError('plan-fit candidate counts must be positive')
        if self.max_scan_cells < 100:
            raise PlanFitError('plan-fit max_scan_cells is too small')
        if self.min_control_cells < 3:
            raise PlanFitError('plan-fit controls need at least three cells')
        for value in (self.min_specificity_gain, self.max_holdout_gap, self.competing_score_gap):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise PlanFitError('plan-fit control shares must be finite in [0, 1]')
        if self.candidate_yaws_deg is not None:
            values = tuple(self.candidate_yaws_deg)
            if not values or not all((math.isfinite(float(value)) for value in values)):
                raise PlanFitError('candidate yaw list must contain finite values')
            object.__setattr__(self, 'candidate_yaws_deg', tuple((float(value) % 360.0 for value in values)))

@dataclass(frozen=True, slots=True)
class PlanFitEvidence:
    fit_cells: int
    validation_cells: int
    fit_overlap: float
    validation_overlap: float | None
    validation_median_m: float | None
    validation_p90_m: float | None
    null_overlap_q90: float | None
    specificity_gain: float | None
    null_count: int
    competing_hypotheses: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def review_supported(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), 'review_supported': self.review_supported, 'protocol': 'spatial-checkerboard-fit-holdout+16-rigid-shift-controls.v1', 'measurement_certified': False}

@dataclass(frozen=True, slots=True)
class PlanFitHypothesis:
    hypothesis_id: str
    scan_id: str
    transform_plan_from_scan_zup: tuple[tuple[float, ...], ...]
    yaw_deg: float
    rank_score: float
    overlap_ratio: float
    median_distance_m: float
    p90_distance_m: float
    matched_cells: int
    support_cells: int
    evidence: PlanFitEvidence | None = None

    @property
    def review_supported(self) -> bool:
        return self.evidence is not None and self.evidence.review_supported

    def to_dict(self) -> dict[str, Any]:
        return {'hypothesis_id': self.hypothesis_id, 'scan_id': self.scan_id, 'transform_plan_from_scan_zup': [list(row) for row in self.transform_plan_from_scan_zup], 'yaw_deg': self.yaw_deg, 'rank_score': self.rank_score, 'overlap_ratio': self.overlap_ratio, 'median_distance_m': self.median_distance_m, 'p90_distance_m': self.p90_distance_m, 'matched_cells': self.matched_cells, 'support_cells': self.support_cells, 'state': 'REVIEW_CANDIDATE', 'evidence': self.evidence.to_dict() if self.evidence else None, 'review_supported': self.review_supported}

@dataclass(frozen=True, slots=True)
class PlanFitResult:
    scan_id: str
    scan_path: str
    plan_path: str
    source_point_count: int
    support_cell_count: int
    effective_grid_m: float
    scan_sha256: str
    plan_sha256: str
    raw_points_preserved: bool
    hypotheses: tuple[PlanFitHypothesis, ...]
    plan_transform: tuple[tuple[float, ...], ...] = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    search_pose_audit: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {'schema': 'cca.plan-fit-candidates.v1', 'scan_id': self.scan_id, 'scan_path': self.scan_path, 'plan_path': self.plan_path, 'source_point_count': self.source_point_count, 'support_cell_count': self.support_cell_count, 'effective_grid_m': self.effective_grid_m, 'scan_sha256': self.scan_sha256, 'plan_sha256': self.plan_sha256, 'raw_points_preserved': self.raw_points_preserved, 'fixed_scale': 1.0, 'axis': 'CAD_Z_up', 'distance_cutoff_m': None, 'search_protocol': 'spatial-fit-only; validation never ranks or refines poses', 'plan_transform': [list(row) for row in self.plan_transform], 'search_pose_audit': list(self.search_pose_audit), 'hypotheses': [item.to_dict() for item in self.hypotheses]}

@dataclass(frozen=True, slots=True)
class _ScoredPlacement:
    yaw_deg: float
    tx_m: float
    ty_m: float
    rank_score: float
    overlap_ratio: float
    median_distance_m: float
    p90_distance_m: float
    matched_cells: int

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def validate_plan_fit_for_apply(result: PlanFitResult, hypothesis: PlanFitHypothesis, *, plan_transform: Sequence[Sequence[float]]) -> None:
    if hypothesis not in result.hypotheses or hypothesis.scan_id != result.scan_id:
        raise PlanFitError('PLAN FIT candidate is not part of this result')
    if not hypothesis.review_supported:
        reasons = ', '.join(hypothesis.evidence.reasons) if hypothesis.evidence else 'NO_CONTROLS'
        raise PlanFitError(f'PLAN FIT cannot apply: {reasons}; manual placement remains available')
    if not np.allclose(_validated_zup_matrix(plan_transform), result.plan_transform, atol=1e-09, rtol=0):
        raise PlanFitError('PLAN FIT underlay moved since search; run it again')
    if _sha256(Path(result.scan_path)) != result.scan_sha256 or _sha256(Path(result.plan_path)) != result.plan_sha256:
        raise PlanFitError('PLAN FIT source evidence changed since search; run it again')

def _validated_zup_matrix(value: Sequence[Sequence[float]] | None) -> np.ndarray:
    if value is None:
        return np.eye(4, dtype=np.float64)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise PlanFitError('plan transform must be a finite 4x4 matrix')
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-09):
        raise PlanFitError('plan transform has an invalid homogeneous row')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-06):
        raise PlanFitError('plan transform must preserve scale 1.0')
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-06):
        raise PlanFitError('plan transform must be proper rigid')
    if not np.allclose(rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-06):
        raise PlanFitError('plan transform must preserve CAD Z-up')
    return matrix

def _load_plan_samples(source: Path, transform: np.ndarray, config: PlanFitConfig) -> np.ndarray:
    try:
        payload = json.loads(source.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PlanFitError(f'cannot read metric plan geometry: {source}') from error
    if not isinstance(payload, Mapping):
        raise PlanFitError('plan geometry must be a JSON object')
    if payload.get('schema') != 'cca.plan-geometry.v1':
        raise PlanFitError('unsupported plan geometry schema')
    if payload.get('output_unit') != 'meter':
        raise PlanFitError('plan geometry must already be in metres')
    raw_segments = payload.get('segments')
    if not isinstance(raw_segments, list):
        raise PlanFitError('plan geometry has no segment list')
    tokens = tuple((token.casefold() for token in config.structural_layer_tokens))
    structural = [segment for segment in raw_segments if isinstance(segment, Mapping) and any((token in str(segment.get('layer', '')).casefold() for token in tokens))]
    selected = structural or [segment for segment in raw_segments if isinstance(segment, Mapping)]
    spacing = config.grid_m * 0.5
    sampled: list[np.ndarray] = []
    for segment in selected:
        points = np.asarray(segment.get('points_m'), dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            continue
        if not np.isfinite(points[:, :2]).all():
            raise PlanFitError('plan geometry contains non-finite coordinates')
        points3 = np.column_stack((points[:, :2], points[:, 2] if points.shape[1] > 2 else np.zeros(len(points))))
        world = (transform @ np.column_stack((points3, np.ones(len(points3)))).T).T
        for start, end in zip(world[:-1, :2], world[1:, :2], strict=True):
            length = float(np.linalg.norm(end - start))
            if length <= 1e-09:
                continue
            steps = max(1, math.ceil(length / spacing))
            factors = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
            sampled.append(start[None, :] + factors[:, None] * (end - start)[None, :])
    if not sampled:
        raise PlanFitError('plan geometry has no usable metric line support')
    return np.unique(np.round(np.concatenate(sampled, axis=0), 6), axis=0)

def _load_scan_xy(source: Path, config: PlanFitConfig) -> tuple[np.ndarray, int, float]:
    schema = read_ply_schema(source)
    names = {item.name for item in schema.vertex.properties}
    if not {'x', 'y', 'z'}.issubset(names):
        raise PlanFitError('scan PLY must contain x, y, and z properties')
    chunks: list[np.ndarray] = []
    point_count = 0
    for chunk in iter_ply_vertex_chunks(source):
        xyz = np.column_stack((chunk['x'], chunk['y'], chunk['z'])).astype(np.float64, copy=False)
        if not np.isfinite(xyz).all():
            raise PlanFitError('scan PLY contains non-finite coordinates')
        chunks.append(xyz[:, :2])
        point_count += len(xyz)
    if not chunks:
        raise PlanFitError('scan PLY contains no vertices')
    xy = np.concatenate(chunks, axis=0)
    effective_grid = float(config.grid_m)
    cells = np.unique(np.rint(xy / effective_grid).astype(np.int64), axis=0)
    if len(cells) > config.max_scan_cells:
        factor = math.ceil(math.sqrt(len(cells) / config.max_scan_cells))
        effective_grid *= factor
        cells = np.unique(np.rint(xy / effective_grid).astype(np.int64), axis=0)
    return (cells.astype(np.float64) * effective_grid, point_count, effective_grid)

def _boundary_cells(cells_xy: np.ndarray, grid_m: float) -> np.ndarray:
    quantized = np.rint(cells_xy / grid_m).astype(np.int64)
    occupied = {tuple(row) for row in quantized.tolist()}
    neighbours = ((1, 0), (-1, 0), (0, 1), (0, -1))
    boundary = [row for row in quantized if any(((int(row[0] + dx), int(row[1] + dy)) not in occupied for dx, dy in neighbours))]
    if not boundary:
        boundary = quantized.tolist()
    return np.asarray(boundary, dtype=np.float64) * grid_m

def _rotation(yaw_deg: float) -> np.ndarray:
    radians = math.radians(float(yaw_deg))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)

def _candidate_peaks(plan_raster: np.ndarray, local_scan_xy: np.ndarray, grid_m: float, peaks_per_angle: int) -> list[tuple[int, int, float]]:
    offsets = np.unique(np.rint(local_scan_xy / grid_m).astype(np.int64), axis=0)
    radius_x = int(np.max(np.abs(offsets[:, 0])))
    radius_y = int(np.max(np.abs(offsets[:, 1])))
    kernel = np.zeros((radius_y * 2 + 1, radius_x * 2 + 1), dtype=np.float32)
    kernel[offsets[:, 1] + radius_y, offsets[:, 0] + radius_x] = 1.0
    correlation = fftconvolve(plan_raster.astype(np.float32), kernel[::-1, ::-1], mode='same')
    flat = correlation.ravel()
    count = min(max(peaks_per_angle * 8, peaks_per_angle), len(flat))
    indexes = np.argpartition(flat, -count)[-count:]
    ordered = indexes[np.argsort(flat[indexes])[::-1]]
    peaks: list[tuple[int, int, float]] = []
    suppression_cells = max(1, round(1.5 / grid_m))
    for index in ordered:
        score = float(flat[index])
        if score <= 0.0:
            continue
        iy, ix = np.unravel_index(int(index), correlation.shape)
        if any((abs(ix - prior_x) <= suppression_cells and abs(iy - prior_y) <= suppression_cells for prior_y, prior_x, _prior_score in peaks)):
            continue
        peaks.append((int(iy), int(ix), score))
        if len(peaks) >= peaks_per_angle:
            break
    return peaks

def _rank_candidate(boundary_xy: np.ndarray, plan_tree: cKDTree, yaw_deg: float, tx_m: float, ty_m: float, tolerance_m: float) -> _ScoredPlacement:
    rotation = _rotation(yaw_deg)
    placed = boundary_xy @ rotation.T + np.asarray([tx_m, ty_m])
    distances = np.asarray(plan_tree.query(placed, k=1)[0], dtype=np.float64)
    overlap = float(np.mean(distances <= tolerance_m))
    median = float(np.median(distances))
    p90 = float(np.quantile(distances, 0.9))
    matched = int(np.count_nonzero(distances <= tolerance_m))
    rank_score = overlap - 0.15 * min(median / tolerance_m, 4.0) - 0.05 * min(p90 / (2.0 * tolerance_m), 4.0)
    return _ScoredPlacement(yaw_deg=float(yaw_deg), tx_m=float(tx_m), ty_m=float(ty_m), rank_score=float(rank_score), overlap_ratio=overlap, median_distance_m=median, p90_distance_m=p90, matched_cells=matched)

def _spatial_split(boundary: np.ndarray, grid_m: float) -> tuple[np.ndarray, np.ndarray]:
    cells = np.rint(boundary / grid_m).astype(np.int64)
    fit_mask = (cells[:, 0] + cells[:, 1]) % 2 == 0
    if min(np.count_nonzero(fit_mask), np.count_nonzero(~fit_mask)) < 3:
        order = np.lexsort((cells[:, 1], cells[:, 0]))
        fit_mask = np.zeros(len(cells), dtype=bool)
        fit_mask[order[::2]] = True
    return (boundary[fit_mask], boundary[~fit_mask])

def _control_candidate(fit: np.ndarray, validation: np.ndarray, plan_tree: cKDTree, candidate: _ScoredPlacement, settings: PlanFitConfig, competing: tuple[str, ...]) -> PlanFitEvidence:
    reasons = []
    if min(len(fit), len(validation)) < settings.min_control_cells:
        reasons.append('INSUFFICIENT_SPATIAL_CONTROL')
    if not len(validation):
        return PlanFitEvidence(len(fit), 0, candidate.overlap_ratio, None, None, None, None, None, 0, competing, tuple(reasons))
    checked = _rank_candidate(validation, plan_tree, candidate.yaw_deg, candidate.tx_m, candidate.ty_m, settings.match_tolerance_m)
    radius = max(4.0 * settings.match_tolerance_m, 0.25 * float(np.linalg.norm(np.ptp(fit, axis=0))))
    null_overlaps = []
    for scale in (1.0, 2.0):
        for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
            null_overlaps.append(_rank_candidate(validation, plan_tree, candidate.yaw_deg, candidate.tx_m + scale * radius * math.cos(angle), candidate.ty_m + scale * radius * math.sin(angle), settings.match_tolerance_m).overlap_ratio)
    null_q90 = float(np.quantile(null_overlaps, 0.9))
    gain = checked.overlap_ratio - null_q90
    if checked.overlap_ratio < 0.5 or checked.median_distance_m > settings.match_tolerance_m:
        reasons.append('WEAK_VALIDATION_SUPPORT')
    if candidate.overlap_ratio - checked.overlap_ratio > settings.max_holdout_gap:
        reasons.append('FIT_HOLDOUT_DISAGREEMENT')
    if gain < settings.min_specificity_gain:
        reasons.append('DENSE_OR_REPEATED_SUPPORT')
    if competing:
        reasons.append('COMPETING_PLACEMENTS')
    return PlanFitEvidence(len(fit), len(validation), candidate.overlap_ratio, checked.overlap_ratio, checked.median_distance_m, checked.p90_distance_m, null_q90, gain, len(null_overlaps), competing, tuple(reasons))

def fit_scan_to_plan(scan_ply: str | Path, plan_geometry: str | Path, *, scan_id: str, plan_transform: Sequence[Sequence[float]] | None=None, scan_z_m: float=0.0, config: PlanFitConfig | None=None) -> PlanFitResult:
    settings = config or PlanFitConfig()
    scan_source = Path(scan_ply).resolve()
    plan_source = Path(plan_geometry).resolve()
    if not scan_source.is_file() or scan_source.suffix.lower() != '.ply':
        raise PlanFitError('plan fit requires an existing PLY scan')
    if not plan_source.is_file() or plan_source.suffix.lower() != '.json':
        raise PlanFitError('plan fit requires metric plan geometry JSON')
    if not scan_id.strip() or not scan_id.isascii():
        raise PlanFitError('plan-fit scan id must be non-empty ASCII')
    if not math.isfinite(float(scan_z_m)):
        raise PlanFitError('plan-fit scan Z must be finite')
    scan_hash_before = _sha256(scan_source)
    plan_hash_before = _sha256(plan_source)
    plan_samples = _load_plan_samples(plan_source, _validated_zup_matrix(plan_transform), settings)
    scan_cells, source_point_count, effective_grid = _load_scan_xy(scan_source, settings)
    boundary = _boundary_cells(scan_cells, effective_grid)
    fit_boundary, validation_boundary = _spatial_split(boundary, effective_grid)
    if len(fit_boundary) < 3:
        raise PlanFitError('plan fit needs at least three spatial fit cells')
    scan_center = np.median(fit_boundary, axis=0)
    local_boundary = fit_boundary - scan_center
    plan_min = np.floor(plan_samples.min(axis=0) / effective_grid) * effective_grid
    plan_max = np.ceil(plan_samples.max(axis=0) / effective_grid) * effective_grid
    shape_xy = np.rint((plan_max - plan_min) / effective_grid).astype(int) + 1
    if int(shape_xy[0]) * int(shape_xy[1]) > 8000000:
        raise PlanFitError('plan raster is too large for interactive plan fit')
    plan_raster = np.zeros((int(shape_xy[1]), int(shape_xy[0])), dtype=bool)
    plan_indexes = np.rint((plan_samples - plan_min) / effective_grid).astype(int)
    plan_raster[plan_indexes[:, 1], plan_indexes[:, 0]] = True
    dilation_steps = max(1, math.ceil(settings.match_tolerance_m / effective_grid))
    plan_raster = binary_dilation(plan_raster, iterations=dilation_steps)
    plan_tree = cKDTree(plan_samples)
    scored: list[_ScoredPlacement] = []
    yaw_values = np.asarray(settings.candidate_yaws_deg, dtype=np.float64) if settings.candidate_yaws_deg is not None else np.arange(0.0, 360.0, settings.angle_step_deg, dtype=np.float64)
    for yaw_deg in yaw_values:
        rotation = _rotation(float(yaw_deg))
        local_rotated = local_boundary @ rotation.T
        for iy, ix, _correlation in _candidate_peaks(plan_raster, local_rotated, effective_grid, settings.peaks_per_angle):
            target_center = plan_min + np.asarray([ix, iy], dtype=np.float64) * effective_grid
            translation = target_center - rotation @ scan_center
            scored.append(_rank_candidate(fit_boundary, plan_tree, float(yaw_deg), float(translation[0]), float(translation[1]), settings.match_tolerance_m))
    if not scored:
        raise PlanFitError('plan fit produced no placement candidate')
    scored.sort(key=lambda item: item.rank_score, reverse=True)
    selected: list[_ScoredPlacement] = []
    for candidate in scored:
        duplicate = any((math.hypot(candidate.tx_m - prior.tx_m, candidate.ty_m - prior.ty_m) < 1.0 and abs((candidate.yaw_deg - prior.yaw_deg + 180.0) % 360.0 - 180.0) < settings.angle_step_deg for prior in selected))
        if duplicate:
            continue
        selected.append(candidate)
    hypotheses: list[PlanFitHypothesis] = []
    for index, candidate in enumerate(selected[:settings.candidate_count], start=1):
        competing = tuple((f'PF{other_index}' for other_index, other in enumerate(selected, start=1) if other_index != index and abs(other.rank_score - candidate.rank_score) <= settings.competing_score_gap))
        evidence = _control_candidate(fit_boundary, validation_boundary, plan_tree, candidate, settings, competing)
        all_cells = _rank_candidate(boundary, plan_tree, candidate.yaw_deg, candidate.tx_m, candidate.ty_m, settings.match_tolerance_m)
        matrix = placement_matrix(tx_m=candidate.tx_m, ty_m=candidate.ty_m, tz_m=float(scan_z_m), yaw_deg=candidate.yaw_deg)
        hypotheses.append(PlanFitHypothesis(hypothesis_id=f'PF{index}', scan_id=scan_id, transform_plan_from_scan_zup=tuple((tuple((float(value) for value in row)) for row in matrix)), yaw_deg=(candidate.yaw_deg + 180.0) % 360.0 - 180.0, rank_score=candidate.rank_score, overlap_ratio=all_cells.overlap_ratio, median_distance_m=all_cells.median_distance_m, p90_distance_m=all_cells.p90_distance_m, matched_cells=all_cells.matched_cells, support_cells=len(boundary), evidence=evidence))
    scan_hash_after = _sha256(scan_source)
    plan_hash_after = _sha256(plan_source)
    if scan_hash_after != scan_hash_before or plan_hash_after != plan_hash_before:
        raise PlanFitError('plan fit source evidence changed during processing')
    return PlanFitResult(scan_id=scan_id, scan_path=str(scan_source), plan_path=str(plan_source), source_point_count=source_point_count, support_cell_count=len(boundary), effective_grid_m=effective_grid, scan_sha256=scan_hash_before, plan_sha256=plan_hash_before, raw_points_preserved=True, hypotheses=tuple(hypotheses), plan_transform=tuple((tuple((float(v) for v in row)) for row in _validated_zup_matrix(plan_transform))), search_pose_audit=tuple(({'hypothesis_id': f'PF{i}', **asdict(p)} for i, p in enumerate(selected, start=1))))
__all__ = ['PlanFitConfig', 'PlanFitError', 'PlanFitEvidence', 'PlanFitHypothesis', 'PlanFitResult', 'fit_scan_to_plan', 'validate_plan_fit_for_apply']
