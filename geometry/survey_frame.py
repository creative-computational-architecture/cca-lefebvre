
from __future__ import annotations
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal
import numpy as np
from geometry.alignment_hypotheses import Matrix4

class SurveyFrameError(ValueError):
    pass
SourceAxis = Literal['auto', 'CAD_Z_up', 'ARKit_Y_up', 'X_up', 'X_down', 'Y_up', 'Y_down', 'Z_up', 'Z_down']

@dataclass(frozen=True, slots=True)
class SurveyFrameConfig:
    ground_percentile: float = 15.0
    ground_inlier_m: float = 0.04
    ransac_iterations: int = 400
    max_tilt_deg: float = 60.0
    wall_band_m: tuple[float, float] = (0.9, 1.4)
    yaw_step_deg: float = 0.25
    yaw_fine_step_deg: float = 0.02
    projection_bin_m: float = 0.02
    sample_points: int = 400000
    ground_test_z_m: float = 0.1
    front_side: Literal['auto', 'keep', 'flip'] = 'auto'
    seed: int = 0

    def __post_init__(self) -> None:
        positive = (self.ground_inlier_m, self.max_tilt_deg, self.yaw_step_deg, self.yaw_fine_step_deg, self.projection_bin_m, self.ground_test_z_m)
        if not all((math.isfinite(float(v)) and float(v) > 0.0 for v in positive)):
            raise SurveyFrameError('survey frame numeric settings must be positive and finite')
        if not 1.0 <= self.ground_percentile <= 60.0:
            raise SurveyFrameError('ground_percentile must lie between 1 and 60')
        if self.max_tilt_deg >= 89.0:
            raise SurveyFrameError('max_tilt_deg must be less than 89 degrees')
        if self.ransac_iterations < 10 or self.sample_points < 1000:
            raise SurveyFrameError('ransac_iterations and sample_points are too small')
        low, high = self.wall_band_m
        if not (math.isfinite(low) and math.isfinite(high) and (low < high)):
            raise SurveyFrameError('wall_band_m must be an ordered finite band')
        if self.front_side not in {'auto', 'keep', 'flip'}:
            raise SurveyFrameError('front_side must be auto, keep, or flip')

@dataclass(frozen=True, slots=True)
class SurveyFrame:
    transform_survey_from_zup: Matrix4
    ground_centroid_zup: tuple[float, float, float]
    ground_tilt_deg: float
    ground_sigma_m: float
    yaw_deg: float
    long_axis_rotated_90: bool
    facade_y_m: float
    scanner_side_counts: dict[str, int]
    flipped_180: bool
    extents_m: dict[str, tuple[float, float]]
    point_count: int

    def rotation(self) -> np.ndarray:
        return np.asarray(self.transform_survey_from_zup, dtype=np.float64)[:3, :3]

    def translation(self) -> np.ndarray:
        return np.asarray(self.transform_survey_from_zup, dtype=np.float64)[:3, 3]

@dataclass(frozen=True, slots=True)
class SourceOrientation:
    requested_axis: str
    detected_up_axis: str
    transform_zup_from_source: Matrix4
    score: float
    ground_support_ratio: float
    points_above_ground_ratio: float
    probe_tilt_deg: float
    probe_sigma_m: float
    probe_point_count: int
_AXIS_ROTATIONS: dict[str, np.ndarray] = {'Z_up': np.eye(3, dtype=np.float64), 'Z_down': np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]), 'Y_up': np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]), 'Y_down': np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]), 'X_up': np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), 'X_down': np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])}
_AXIS_ALIASES = {'CAD_Z_up': 'Z_up', 'ARKit_Y_up': 'Y_up', **{name: name for name in _AXIS_ROTATIONS}}
_AUTO_AXIS_ORDER = ('Z_up', 'Y_up', 'X_up', 'Z_down', 'Y_down', 'X_down')

def _as_points(points_xyz: Any) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise SurveyFrameError('points must be an (N, 3) array')
    points = points[:, :3]
    if points.shape[0] < 1000:
        raise SurveyFrameError('at least 1000 points are needed to estimate a survey frame')
    if not np.isfinite(points).all():
        raise SurveyFrameError('points must be finite')
    return points

def _rotation_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = np.cross(a, b)
    c = float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / s ** 2)

def _estimate_ground(points: np.ndarray, config: SurveyFrameConfig) -> tuple[np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(config.seed)
    z = points[:, 2]
    low = points[z < np.percentile(z, config.ground_percentile)]
    if low.shape[0] < 3:
        raise SurveyFrameError('not enough low points for ground estimation')
    best: tuple[int, np.ndarray, np.ndarray] | None = None
    minimum_vertical = math.cos(math.radians(config.max_tilt_deg))
    for _ in range(config.ransac_iterations):
        a, b, c = low[rng.choice(low.shape[0], 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-09:
            continue
        normal = normal / norm
        if abs(normal[2]) < minimum_vertical:
            continue
        inliers = int((np.abs((low - a) @ normal) < config.ground_inlier_m).sum())
        if best is None or inliers > best[0]:
            best = (inliers, normal, a)
    if best is None:
        raise SurveyFrameError('no horizontal ground plane candidate was found')
    _, normal, anchor = best
    ground = low[np.abs((low - anchor) @ normal) < config.ground_inlier_m]
    centroid = ground.mean(axis=0)
    _, _, vt = np.linalg.svd(ground - centroid, full_matrices=False)
    normal = vt[2] if vt[2][2] > 0 else -vt[2]
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    sigma = float(np.std((ground - centroid) @ normal))
    return (normal, centroid, tilt, sigma)

def _axis_probe(sample: np.ndarray, axis: str, config: SurveyFrameConfig) -> tuple[float, float, float, float, float] | None:
    rotation = _AXIS_ROTATIONS[axis]
    zup = sample @ rotation.T
    probe_config = replace(config, ransac_iterations=min(config.ransac_iterations, 120))
    try:
        normal, centroid, tilt, sigma = _estimate_ground(zup, probe_config)
    except SurveyFrameError:
        return None
    low_limit = np.percentile(zup[:, 2], config.ground_percentile)
    low = zup[zup[:, 2] < low_limit]
    support = float((np.abs((low - centroid) @ normal) < config.ground_inlier_m).mean())
    signed = (zup - centroid) @ normal
    above = float((signed >= -config.ground_inlier_m).mean())
    level_rotation = _rotation_a_to_b(normal, np.array([0.0, 0.0, 1.0]))
    levelled = (zup - centroid) @ level_rotation.T
    robust_range = np.percentile(levelled, 99.0, axis=0) - np.percentile(levelled, 1.0, axis=0)
    horizontal_span = max(float(robust_range[0]), float(robust_range[1]), 1e-06)
    vertical_span = max(float(robust_range[2]), 1e-06)
    architectural_shape = min(horizontal_span / vertical_span, 4.0) / 4.0
    noise_weight = 1.0 / (1.0 + sigma / config.ground_inlier_m)
    score = support * above * noise_weight * (0.85 + 0.15 * architectural_shape)
    return (score, support, above, tilt, sigma)

def normalize_source_points(points_xyz: Any, source_axis: SourceAxis='auto', config: SurveyFrameConfig | None=None) -> tuple[np.ndarray, SourceOrientation]:
    config = config or SurveyFrameConfig()
    points = _as_points(points_xyz)
    if not isinstance(source_axis, str):
        raise SurveyFrameError('source_axis must be text')
    if source_axis != 'auto' and source_axis not in _AXIS_ALIASES:
        choices = ', '.join(('auto', *_AXIS_ALIASES))
        raise SurveyFrameError(f'unsupported source_axis {source_axis!r}; expected one of {choices}')
    rng = np.random.default_rng(config.seed)
    probe_count = min(points.shape[0], min(config.sample_points, 200000))
    sample = points if probe_count == points.shape[0] else points[rng.choice(points.shape[0], probe_count, replace=False)]
    axes = _AUTO_AXIS_ORDER if source_axis == 'auto' else (_AXIS_ALIASES[source_axis],)
    candidates: list[tuple[float, str, float, float, float, float]] = []
    for axis in axes:
        probe = _axis_probe(sample, axis, config)
        if probe is not None:
            score, support, above, tilt, sigma = probe
            candidates.append((score, axis, support, above, tilt, sigma))
    if not candidates:
        raise SurveyFrameError('no supported ground plane was found while normalizing the source axis')
    score, axis, support, above, tilt, sigma = max(candidates, key=lambda candidate: candidate[0])
    if source_axis == 'auto' and (support < 0.015 or above < 0.65):
        raise SurveyFrameError(f'automatic up-axis detection found no reliable ground boundary (support={support:.3f}, above={above:.3f})')
    rotation = _AXIS_ROTATIONS[axis]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    transform: Matrix4 = tuple((tuple((float(value) for value in row)) for row in matrix))
    orientation = SourceOrientation(requested_axis=source_axis, detected_up_axis=axis, transform_zup_from_source=transform, score=float(score), ground_support_ratio=float(support), points_above_ground_ratio=float(above), probe_tilt_deg=float(tilt), probe_sigma_m=float(sigma), probe_point_count=int(sample.shape[0]))
    return (points @ rotation.T, orientation)

def _estimate_yaw(sample_xy: np.ndarray, config: SurveyFrameConfig) -> float:
    bin_m = config.projection_bin_m

    def sharpness(theta_deg: float) -> float:
        t = math.radians(theta_deg)
        c, s = (math.cos(t), math.sin(t))
        u = sample_xy[:, 0] * c - sample_xy[:, 1] * s
        v = sample_xy[:, 0] * s + sample_xy[:, 1] * c
        hu, _ = np.histogram(u, bins=np.arange(u.min(), u.max() + bin_m, bin_m))
        hv, _ = np.histogram(v, bins=np.arange(v.min(), v.max() + bin_m, bin_m))
        return float((hu.astype(np.float64) ** 2).sum() + (hv.astype(np.float64) ** 2).sum())
    coarse = np.arange(0.0, 90.0, config.yaw_step_deg)
    scores = np.array([sharpness(a) for a in coarse])
    best = float(coarse[int(scores.argmax())])
    fine = np.arange(best - 1.0, best + 1.0, config.yaw_fine_step_deg)
    fine_scores = np.array([sharpness(a) for a in fine])
    return float(fine[int(fine_scores.argmax())])

def estimate_survey_frame(points_xyz: Any, config: SurveyFrameConfig | None=None) -> SurveyFrame:
    config = config or SurveyFrameConfig()
    points = _as_points(points_xyz)
    normal, centroid, tilt, sigma = _estimate_ground(points, config)
    if tilt > config.max_tilt_deg:
        raise SurveyFrameError(f'ground tilt {tilt:.2f} deg exceeds max_tilt_deg={config.max_tilt_deg:.2f}; the scan is not gravity aligned')
    r_level = _rotation_a_to_b(normal, np.array([0.0, 0.0, 1.0]))
    levelled = (points - centroid) @ r_level.T
    rng = np.random.default_rng(config.seed)
    band = (levelled[:, 2] > config.wall_band_m[0]) & (levelled[:, 2] < config.wall_band_m[1])
    sample = levelled[band][:, :2]
    if sample.shape[0] < 100:
        raise SurveyFrameError('too few points in the wall band to estimate yaw')
    if sample.shape[0] > config.sample_points:
        sample = sample[rng.choice(sample.shape[0], config.sample_points, replace=False)]
    yaw = _estimate_yaw(sample, config)
    theta = math.radians(yaw)
    r_yaw = np.array([[math.cos(theta), -math.sin(theta), 0.0], [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    aligned = levelled @ r_yaw.T
    rotated_90 = False
    thin = aligned[::20, :2]
    _, _, vt = np.linalg.svd(thin - thin.mean(axis=0), full_matrices=False)
    if abs(vt[0][1]) > abs(vt[0][0]):
        r_90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        aligned = aligned @ r_90.T
        r_yaw = r_90 @ r_yaw
        rotated_90 = True
    band = (aligned[:, 2] > config.wall_band_m[0]) & (aligned[:, 2] < config.wall_band_m[1])
    y_all = aligned[:, 1]
    hist, edges = np.histogram(aligned[band, 1], bins=np.arange(y_all.min(), y_all.max() + 0.02, 0.02))
    k = int(hist.argmax())
    facade_y = float((edges[k] + edges[k + 1]) / 2.0)
    ground_mask = np.abs(aligned[:, 2]) < config.ground_test_z_m
    n_small = int((ground_mask & (y_all < facade_y - 0.3)).sum())
    n_large = int((ground_mask & (y_all > facade_y + 0.3)).sum())
    if config.front_side == 'auto':
        flipped = n_large > n_small
    else:
        flipped = config.front_side == 'flip'
    r_flip = np.diag([-1.0, -1.0, 1.0]) if flipped else np.eye(3)
    final = aligned @ r_flip.T
    rotation = r_flip @ r_yaw @ r_level
    offset = final[:, :2].min(axis=0)
    translation = -rotation @ centroid + np.array([-offset[0], -offset[1], 0.0])
    final[:, 0] -= offset[0]
    final[:, 1] -= offset[1]
    facade_after = (-facade_y if flipped else facade_y) - offset[1]
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-08):
        raise SurveyFrameError('survey frame rotation is not a proper rotation')
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    transform: Matrix4 = tuple((tuple((float(v) for v in row)) for row in matrix))
    return SurveyFrame(transform_survey_from_zup=transform, ground_centroid_zup=tuple((float(v) for v in centroid)), ground_tilt_deg=tilt, ground_sigma_m=sigma, yaw_deg=yaw, long_axis_rotated_90=rotated_90, facade_y_m=float(facade_after), scanner_side_counts={'ground_points_small_y': n_small, 'ground_points_large_y': n_large}, flipped_180=bool(flipped), extents_m={axis: (float(final[:, i].min()), float(final[:, i].max())) for i, axis in enumerate('xyz')}, point_count=int(final.shape[0]))

def apply_survey_frame(points_xyz: Any, frame: SurveyFrame) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise SurveyFrameError('points must be an (N, 3) array')
    return points[:, :3] @ frame.rotation().T + frame.translation()

def frame_to_dict(frame: SurveyFrame) -> dict[str, Any]:
    payload = asdict(frame)
    payload['transform_survey_from_zup'] = [list(row) for row in frame.transform_survey_from_zup]
    payload['ground_centroid_zup'] = list(frame.ground_centroid_zup)
    payload['extents_m'] = {k: list(v) for k, v in frame.extents_m.items()}
    payload['schema'] = 'cca.survey-frame.v1'
    payload['scale'] = 1.0
    payload['determinant'] = 1.0
    return payload

def orientation_to_dict(orientation: SourceOrientation) -> dict[str, Any]:
    payload = asdict(orientation)
    payload['transform_zup_from_source'] = [list(row) for row in orientation.transform_zup_from_source]
    payload['schema'] = 'cca.survey-source-orientation.v1'
    payload['scale'] = 1.0
    payload['determinant'] = 1.0
    return payload
__all__ = ['SourceAxis', 'SourceOrientation', 'SurveyFrame', 'SurveyFrameConfig', 'SurveyFrameError', 'apply_survey_frame', 'estimate_survey_frame', 'frame_to_dict', 'normalize_source_points', 'orientation_to_dict']
