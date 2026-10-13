
from __future__ import annotations
import copy
import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal
import numpy as np
import open3d as o3d
from geometry.register import CloudWorkset, EdgeResult, ICPScale, project_yaw_translation
UNIT_METERS = 'meter'
AXIS_Z_UP = 'Z_UP'
EDGE_SCHEMA = 'cca.fixed_scale.zup.pair.v1'
_BASIS_YUP_TO_ZUP = np.eye(4, dtype=np.float64)
_BASIS_YUP_TO_ZUP[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
_BASIS_ZUP_TO_YUP = _BASIS_YUP_TO_ZUP.T

@dataclass(frozen=True)
class RegistrationContract:
    units: str = UNIT_METERS
    axis: str = AXIS_Z_UP
    fixed_scale: float = 1.0

@dataclass(frozen=True)
class PairRegistrationConfig:
    scales: tuple[ICPScale, ...] = (ICPScale(0.3, 0.75, 24), ICPScale(0.15, 0.35, 20), ICPScale(0.075, 0.18, 16))
    coarse_method: Literal['ransac', 'fgr'] = 'ransac'
    feature_radius_factor: float = 5.0
    normal_radius_factor: float = 3.0
    ransac_max_iterations: int = 50000
    ransac_confidence: float = 0.999
    ransac_edge_length: float = 0.9
    robust_kernel: Literal['huber', 'tukey'] = 'huber'
    robust_kernel_factor: float = 1.5
    colored_icp: bool = True
    colored_lambda_geometric: float = 0.9
    min_colored_points: int = 100
    min_color_variance: float = 0.0001
    min_overlap: float = 0.2
    min_fitness: float = 0.2
    max_inlier_rmse_m: float = 0.18
    min_correspondences: int = 50
    max_translation_m: float | None = None
    min_observability_ratio: float = 0.001
    max_abs_yaw_deg: float = 180.0
    max_pitch_roll_deg: float = 0.01
    scale_tolerance: float = 1e-08
    duplicate_tolerance_m: float = 1e-06
    reject_isotropic_ratio: float = 0.94
    random_seed: int = 731951

@dataclass(frozen=True)
class PairQuality:
    overlap: float = 0.0
    fitness: float = 0.0
    reverse_fitness: float = 0.0
    inlier_rmse_m: float | None = None
    reverse_inlier_rmse_m: float | None = None
    correspondences: int = 0
    reverse_correspondences: int = 0
    observable_dof: int = 0
    reverse_observable_dof: int = 0
    observability_ratio: float = 0.0
    reverse_observability_ratio: float = 0.0
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation_norm_m: float = 0.0
    yaw_deg: float = 0.0
    pitch_roll_deg: float = 0.0
    scale: float = 1.0
    scale_deviation: float = 0.0
    determinant: float = 1.0

@dataclass
class PairRegistrationResult:
    source_id: str
    target_id: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    transform_target_from_source_zup: np.ndarray
    initial_transform_target_from_source_zup: np.ndarray
    quality: PairQuality
    coarse_method: str
    coarse_fitness: float = 0.0
    coarse_inlier_rmse_m: float | None = None
    iterations_run: tuple[int, ...] = ()
    colored_attempted: bool = False
    colored_used: bool = False
    source_point_count: int = 0
    target_point_count: int = 0
    source_isotropy_ratio: float = 0.0
    target_isotropy_ratio: float = 0.0
    source_memory_hash_before: str = ''
    source_memory_hash_after: str = ''
    target_memory_hash_before: str = ''
    target_memory_hash_after: str = ''
    source_file_hash_before: str | None = None
    source_file_hash_after: str | None = None
    target_file_hash_before: str | None = None
    target_file_hash_after: str | None = None
    information_yup: np.ndarray = field(default_factory=lambda: np.zeros((6, 6), dtype=np.float64), repr=False)
    units: str = UNIT_METERS
    axis: str = AXIS_Z_UP
    fixed_scale: float = 1.0

    @property
    def gate(self) -> str:
        return 'green' if self.accepted else 'red'

    @property
    def raw_points_preserved(self) -> bool:
        memory_ok = self.source_memory_hash_before == self.source_memory_hash_after and self.target_memory_hash_before == self.target_memory_hash_after
        file_ok = all((before is None or before == after for before, after in ((self.source_file_hash_before, self.source_file_hash_after), (self.target_file_hash_before, self.target_file_hash_after))))
        return bool(memory_ok and file_ok)

    def to_dict(self) -> dict[str, Any]:
        quality = asdict(self.quality)
        return {'schema': EDGE_SCHEMA, 'source': self.source_id, 'target': self.target_id, 'accepted': self.accepted, 'gate': self.gate, 'rejection_reasons': list(self.rejection_reasons), 'T_target_from_source_zup': self.transform_target_from_source_zup.tolist(), 'T_initial_target_from_source_zup': self.initial_transform_target_from_source_zup.tolist(), 'quality': quality, 'coarse': {'method': self.coarse_method, 'fitness': self.coarse_fitness, 'inlier_rmse_m': self.coarse_inlier_rmse_m}, 'iterations_run': list(self.iterations_run), 'colored_attempted': self.colored_attempted, 'colored_used': self.colored_used, 'source_point_count': self.source_point_count, 'target_point_count': self.target_point_count, 'source_isotropy_ratio': self.source_isotropy_ratio, 'target_isotropy_ratio': self.target_isotropy_ratio, 'raw_points_preserved': self.raw_points_preserved, 'units': self.units, 'axis': self.axis, 'fixed_scale': self.fixed_scale}

    def accepted_edge_payload(self) -> dict[str, Any] | None:
        if not self.accepted or not self.raw_points_preserved:
            return None
        return {'schema': EDGE_SCHEMA, 'source': self.source_id, 'target': self.target_id, 'transform_source_to_target': self.transform_target_from_source_zup.tolist(), 'fitness': self.quality.fitness, 'inlier_rmse_m': self.quality.inlier_rmse_m, 'correspondence_count': self.quality.correspondences, 'information_yup': self.information_yup.tolist(), 'accepted': True, 'certain': False, 'units': self.units, 'axis': self.axis, 'scale': self.fixed_scale}

    def as_register_edge(self, gps_distance_m: float=0.0) -> EdgeResult | None:
        if not self.accepted or not self.raw_points_preserved:
            return None
        return EdgeResult(source=self.source_id, target=self.target_id, gps_distance_m=float(gps_distance_m), initial_transform=_zup_transform_to_yup(self.initial_transform_target_from_source_zup), transform_source_to_target=_zup_transform_to_yup(self.transform_target_from_source_zup), fitness=self.quality.fitness, inlier_rmse_m=float(self.quality.inlier_rmse_m or 0.0), correspondence_count=self.quality.correspondences, accepted=True, rejection_reasons=(), information=self.information_yup.copy(), iterations_run=self.iterations_run, certain=False)
CloudInput = o3d.geometry.PointCloud | np.ndarray

def _zup_transform_to_yup(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    return _BASIS_ZUP_TO_YUP @ matrix @ _BASIS_YUP_TO_ZUP

def _yup_transform_to_zup(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    return _BASIS_YUP_TO_ZUP @ matrix @ _BASIS_ZUP_TO_YUP

def project_zup_yaw_translation(transform: np.ndarray) -> np.ndarray:
    return _yup_transform_to_zup(project_yaw_translation(_zup_transform_to_yup(transform)))

def _copy_cloud(value: CloudInput, label: str) -> o3d.geometry.PointCloud:
    if isinstance(value, o3d.geometry.PointCloud):
        cloud = copy.deepcopy(value)
    else:
        points = np.asarray(value, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError(f'{label} points must have shape (N, 3)')
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.copy()))
    points = np.asarray(cloud.points)
    if len(points) < 3:
        raise ValueError(f'{label} has fewer than three points')
    if not np.isfinite(points).all():
        raise ValueError(f'{label} contains non-finite points')
    if cloud.has_colors() and (not np.isfinite(np.asarray(cloud.colors)).all()):
        raise ValueError(f'{label} contains non-finite colors')
    return cloud

def _cloud_memory_hash(cloud: o3d.geometry.PointCloud) -> str:
    digest = hashlib.sha256()
    for name, values in ((b'points', np.asarray(cloud.points)), (b'colors', np.asarray(cloud.colors)), (b'normals', np.asarray(cloud.normals))):
        array = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(name)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()

def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _canonical_geometry_hash(cloud: o3d.geometry.PointCloud, tolerance_m: float) -> str:
    points = np.asarray(cloud.points)
    quantized = np.rint(points / tolerance_m).astype(np.int64)
    order = np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))
    canonical = np.ascontiguousarray(quantized[order])
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype=np.int64).tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()

def _isotropy_ratio(cloud: o3d.geometry.PointCloud) -> float:
    points = np.asarray(cloud.points)
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(max(eigenvalues[-1], 0.0))
    return float(max(eigenvalues[0], 0.0) / largest) if largest > 1e-15 else 1.0

def _validate_config(config: PairRegistrationConfig) -> None:
    if not config.scales:
        raise ValueError('at least one ICP scale is required')
    previous = math.inf
    for scale in config.scales:
        if scale.voxel_m <= 0.0 or scale.max_correspondence_m <= 0.0:
            raise ValueError('voxel and correspondence distances must be positive')
        if scale.iterations < 1 or scale.voxel_m >= previous:
            raise ValueError('ICP scales must be positive and coarse to fine')
        previous = scale.voxel_m
    if config.coarse_method not in {'ransac', 'fgr'}:
        raise ValueError('coarse_method must be ransac or fgr')
    if config.duplicate_tolerance_m <= 0.0:
        raise ValueError('duplicate_tolerance_m must be positive')
    if not 0.0 <= config.min_overlap <= 1.0:
        raise ValueError('min_overlap must be between zero and one')
    if not 0.0 <= config.min_fitness <= 1.0:
        raise ValueError('min_fitness must be between zero and one')
    if config.max_translation_m is not None and (not math.isfinite(config.max_translation_m) or config.max_translation_m <= 0.0):
        raise ValueError('max_translation_m must be positive or None')
    if not 0.0 < config.min_observability_ratio < 1.0:
        raise ValueError('min_observability_ratio must be between zero and one')

def _contract_reasons(contract: RegistrationContract) -> list[str]:
    reasons: list[str] = []
    if contract.units != UNIT_METERS:
        reasons.append('unit_contract_not_meter')
    if contract.axis != AXIS_Z_UP:
        reasons.append('axis_contract_not_z_up')
    if not math.isclose(contract.fixed_scale, 1.0, abs_tol=1e-12):
        reasons.append('fixed_scale_contract_violation')
    return reasons

def _to_yup_cloud(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    result = copy.deepcopy(cloud)
    result.transform(_BASIS_ZUP_TO_YUP)
    return result

def _prepare_workset(cloud_yup: o3d.geometry.PointCloud, scales: Sequence[ICPScale], normal_radius_factor: float) -> CloudWorkset:
    pyramid: list[o3d.geometry.PointCloud] = []
    for scale in scales:
        sampled = cloud_yup.voxel_down_sample(scale.voxel_m)
        if len(sampled.points) < 6:
            raise ValueError(f'too few points after voxel preprocessing at {scale.voxel_m:g} m')
        sampled.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=scale.voxel_m * normal_radius_factor, max_nn=50))
        pyramid.append(sampled)
    return CloudWorkset(point_count=len(cloud_yup.points), min_bound=cloud_yup.get_min_bound(), max_bound=cloud_yup.get_max_bound(), pyramid=tuple(pyramid))

def _fpfh(cloud: o3d.geometry.PointCloud, voxel_m: float, radius_factor: float) -> o3d.pipelines.registration.Feature:
    return o3d.pipelines.registration.compute_fpfh_feature(cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_m * radius_factor, max_nn=100))

def _coarse_registration(source: CloudWorkset, target: CloudWorkset, config: PairRegistrationConfig) -> o3d.pipelines.registration.RegistrationResult:
    source_coarse = source.pyramid[0]
    target_coarse = target.pyramid[0]
    scale = config.scales[0]
    source_feature = _fpfh(source_coarse, scale.voxel_m, config.feature_radius_factor)
    target_feature = _fpfh(target_coarse, scale.voxel_m, config.feature_radius_factor)
    if config.coarse_method == 'fgr':
        return o3d.pipelines.registration.registration_fgr_based_on_feature_matching(source_coarse, target_coarse, source_feature, target_feature, o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=scale.max_correspondence_m, iteration_number=64, maximum_tuple_count=2000))
    o3d.utility.random.seed(config.random_seed)
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source_coarse, target_coarse, source_feature, target_feature, True, scale.max_correspondence_m, o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(config.ransac_edge_length), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(scale.max_correspondence_m)], o3d.pipelines.registration.RANSACConvergenceCriteria(config.ransac_max_iterations, config.ransac_confidence))

def _evaluation(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, max_correspondence_m: float, transform: np.ndarray) -> o3d.pipelines.registration.RegistrationResult:
    return o3d.pipelines.registration.evaluate_registration(source, target, max_correspondence_m, transform)

def _matched_observability(moving: o3d.geometry.PointCloud, fixed: o3d.geometry.PointCloud, transform_yup: np.ndarray, registration: o3d.pipelines.registration.RegistrationResult, eigenvalue_ratio: float) -> tuple[int, float]:
    pairs = np.asarray(registration.correspondence_set, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[0] < 4 or pairs.shape[1] != 2:
        return (0, 0.0)
    if not fixed.has_normals():
        return (0, 0.0)
    moving_points = np.asarray(moving.points)[pairs[:, 0]]
    rotation = transform_yup[:3, :3]
    translation = transform_yup[:3, 3]
    transformed = moving_points @ rotation.T + translation
    centered = transformed - transformed.mean(axis=0)
    normals = np.asarray(fixed.normals)[pairs[:, 1]]
    horizontal_radius = float(np.sqrt(np.mean(centered[:, 0] ** 2 + centered[:, 2] ** 2)))
    if horizontal_radius <= 1e-12:
        return (0, 0.0)
    yaw_column = (normals[:, 0] * centered[:, 2] - normals[:, 2] * centered[:, 0]) / horizontal_radius
    jacobian = np.column_stack((normals, yaw_column))
    normal_matrix = jacobian.T @ jacobian / float(len(jacobian))
    eigenvalues = np.maximum(np.linalg.eigvalsh(normal_matrix), 0.0)
    largest = float(eigenvalues[-1])
    if largest <= 1e-15:
        return (0, 0.0)
    normalized = eigenvalues / largest
    observable_dof = int(np.count_nonzero(normalized >= eigenvalue_ratio))
    return (observable_dof, float(normalized[0]))

def _candidate_score(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, max_correspondence_m: float, transform: np.ndarray) -> tuple[float, float]:
    result = _evaluation(source, target, max_correspondence_m, transform)
    rmse = float(result.inlier_rmse) if len(result.correspondence_set) else math.inf
    return (float(result.fitness), -rmse)

def _robust_estimation(scale: ICPScale, config: PairRegistrationConfig) -> o3d.pipelines.registration.TransformationEstimationPointToPlane:
    kernel_width = max(scale.voxel_m * config.robust_kernel_factor, 1e-06)
    if config.robust_kernel == 'tukey':
        loss = o3d.pipelines.registration.TukeyLoss(k=kernel_width)
    else:
        loss = o3d.pipelines.registration.HuberLoss(k=kernel_width)
    return o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

def _robust_gravity_locked_icp(source: CloudWorkset, target: CloudWorkset, initial_transform_yup: np.ndarray, config: PairRegistrationConfig) -> tuple[np.ndarray, tuple[int, ...]]:
    current = project_yaw_translation(initial_transform_yup)
    iteration_counts: list[int] = []
    for source_cloud, target_cloud, scale in zip(source.pyramid, target.pyramid, config.scales):
        used = 0
        estimation = _robust_estimation(scale, config)
        for _ in range(scale.iterations):
            before = current.copy()
            registration = o3d.pipelines.registration.registration_icp(source_cloud, target_cloud, scale.max_correspondence_m, current, estimation, o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0, relative_rmse=0.0, max_iteration=1))
            current = project_yaw_translation(registration.transformation)
            used += 1
            translation_delta = float(np.linalg.norm(current[:3, 3] - before[:3, 3]))
            rotation_delta = current[:3, :3] @ before[:3, :3].T
            yaw_delta = abs(math.atan2(rotation_delta[0, 2], rotation_delta[0, 0]))
            if translation_delta < 1e-07 and yaw_delta < 1e-07:
                break
        iteration_counts.append(used)
    return (current, tuple(iteration_counts))

def _has_usable_color(source: CloudWorkset, target: CloudWorkset, config: PairRegistrationConfig) -> bool:
    source_cloud = source.pyramid[-1]
    target_cloud = target.pyramid[-1]
    if not source_cloud.has_colors() or not target_cloud.has_colors():
        return False
    if min(len(source_cloud.colors), len(target_cloud.colors)) < config.min_colored_points:
        return False
    source_variance = float(np.var(np.asarray(source_cloud.colors), axis=0).mean())
    target_variance = float(np.var(np.asarray(target_cloud.colors), axis=0).mean())
    return min(source_variance, target_variance) >= config.min_color_variance

def _colored_gravity_locked_icp(source: CloudWorkset, target: CloudWorkset, initial_transform_yup: np.ndarray, config: PairRegistrationConfig) -> np.ndarray:
    current = project_yaw_translation(initial_transform_yup)
    estimation = o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=config.colored_lambda_geometric)
    for source_cloud, target_cloud, scale in zip(source.pyramid, target.pyramid, config.scales):
        for _ in range(scale.iterations):
            before = current.copy()
            registration = o3d.pipelines.registration.registration_colored_icp(source_cloud, target_cloud, scale.max_correspondence_m, current, estimation, o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0, relative_rmse=0.0, max_iteration=1))
            current = project_yaw_translation(registration.transformation)
            if np.linalg.norm(current - before) < 1e-08:
                break
    return current

def _quality(source: CloudWorkset, target: CloudWorkset, transform_yup: np.ndarray, config: PairRegistrationConfig) -> PairQuality:
    finest = config.scales[-1]
    forward = _evaluation(source.pyramid[-1], target.pyramid[-1], finest.max_correspondence_m, transform_yup)
    inverse = np.linalg.inv(transform_yup)
    reverse = _evaluation(target.pyramid[-1], source.pyramid[-1], finest.max_correspondence_m, inverse)
    transform_zup = _yup_transform_to_zup(transform_yup)
    rotation = transform_zup[:3, :3]
    singular_values = np.linalg.svd(rotation, compute_uv=False)
    scale = float(np.mean(singular_values))
    scale_deviation = float(np.max(np.abs(singular_values - 1.0)))
    transformed_up = rotation @ np.array([0.0, 0.0, 1.0])
    pitch_roll_deg = math.degrees(math.acos(float(np.clip(transformed_up[2], -1.0, 1.0))))
    translation = tuple((float(value) for value in transform_zup[:3, 3]))
    forward_count = len(forward.correspondence_set)
    reverse_count = len(reverse.correspondence_set)
    observable_dof, observability_ratio = _matched_observability(source.pyramid[-1], target.pyramid[-1], transform_yup, forward, config.min_observability_ratio)
    reverse_observable_dof, reverse_observability_ratio = _matched_observability(target.pyramid[-1], source.pyramid[-1], inverse, reverse, config.min_observability_ratio)
    return PairQuality(overlap=min(float(forward.fitness), float(reverse.fitness)), fitness=float(forward.fitness), reverse_fitness=float(reverse.fitness), inlier_rmse_m=float(forward.inlier_rmse) if forward_count else None, reverse_inlier_rmse_m=float(reverse.inlier_rmse) if reverse_count else None, correspondences=forward_count, reverse_correspondences=reverse_count, observable_dof=observable_dof, reverse_observable_dof=reverse_observable_dof, observability_ratio=observability_ratio, reverse_observability_ratio=reverse_observability_ratio, translation_m=translation, translation_norm_m=float(np.linalg.norm(transform_zup[:3, 3])), yaw_deg=math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])), pitch_roll_deg=pitch_roll_deg, scale=scale, scale_deviation=scale_deviation, determinant=float(np.linalg.det(rotation)))

def _quality_reasons(quality: PairQuality, config: PairRegistrationConfig) -> list[str]:
    reasons: list[str] = []
    if quality.overlap < config.min_overlap:
        reasons.append('overlap_below_min')
    if quality.fitness < config.min_fitness:
        reasons.append('fitness_below_min')
    rmses = [value for value in (quality.inlier_rmse_m, quality.reverse_inlier_rmse_m) if value is not None]
    if len(rmses) != 2 or max(rmses) > config.max_inlier_rmse_m:
        reasons.append('rmse_above_max')
    if min(quality.correspondences, quality.reverse_correspondences) < config.min_correspondences:
        reasons.append('correspondence_count_below_min')
    if min(quality.observable_dof, quality.reverse_observable_dof) < 4:
        reasons.append('geometry_underconstrained')
    if config.max_translation_m is not None and quality.translation_norm_m > config.max_translation_m:
        reasons.append('translation_above_max')
    if abs(quality.yaw_deg) > config.max_abs_yaw_deg + 1e-09:
        reasons.append('yaw_above_max')
    if quality.pitch_roll_deg > config.max_pitch_roll_deg:
        reasons.append('pitch_roll_above_max')
    if quality.scale_deviation > config.scale_tolerance:
        reasons.append('scale_not_locked')
    if not math.isclose(quality.determinant, 1.0, abs_tol=config.scale_tolerance):
        reasons.append('rotation_not_proper')
    return reasons

def _rejected_result(source_id: str, target_id: str, reasons: Sequence[str], source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, source_hash: str, target_hash: str, config: PairRegistrationConfig, contract: RegistrationContract, source_isotropy: float=0.0, target_isotropy: float=0.0) -> PairRegistrationResult:
    return PairRegistrationResult(source_id=source_id, target_id=target_id, accepted=False, rejection_reasons=tuple(sorted(set(reasons))), transform_target_from_source_zup=np.eye(4, dtype=np.float64), initial_transform_target_from_source_zup=np.eye(4, dtype=np.float64), quality=PairQuality(), coarse_method=config.coarse_method, source_point_count=len(source.points), target_point_count=len(target.points), source_isotropy_ratio=source_isotropy, target_isotropy_ratio=target_isotropy, source_memory_hash_before=source_hash, source_memory_hash_after=_cloud_memory_hash(source), target_memory_hash_before=target_hash, target_memory_hash_after=_cloud_memory_hash(target), units=UNIT_METERS, axis=AXIS_Z_UP, fixed_scale=1.0)

def register_pair(source: CloudInput, target: CloudInput, *, source_id: str='source', target_id: str='target', config: PairRegistrationConfig | None=None, contract: RegistrationContract | None=None, initial_transform_zup: np.ndarray | None=None) -> PairRegistrationResult:
    config = config or PairRegistrationConfig()
    contract = contract or RegistrationContract()
    _validate_config(config)
    source_original = _copy_cloud(source, 'source')
    target_original = _copy_cloud(target, 'target')
    source_hash = _cloud_memory_hash(source_original)
    target_hash = _cloud_memory_hash(target_original)
    contract_reasons = _contract_reasons(contract)
    if contract_reasons:
        return _rejected_result(source_id, target_id, contract_reasons, source_original, target_original, source_hash, target_hash, config, contract)
    if _canonical_geometry_hash(source_original, config.duplicate_tolerance_m) == _canonical_geometry_hash(target_original, config.duplicate_tolerance_m):
        return _rejected_result(source_id, target_id, ['duplicate_geometry'], source_original, target_original, source_hash, target_hash, config, contract)
    source_isotropy = _isotropy_ratio(source_original)
    target_isotropy = _isotropy_ratio(target_original)
    if min(source_isotropy, target_isotropy) >= config.reject_isotropic_ratio:
        return _rejected_result(source_id, target_id, ['symmetric_geometry_ambiguous'], source_original, target_original, source_hash, target_hash, config, contract, source_isotropy, target_isotropy)
    try:
        source_workset = _prepare_workset(_to_yup_cloud(source_original), config.scales, config.normal_radius_factor)
        target_workset = _prepare_workset(_to_yup_cloud(target_original), config.scales, config.normal_radius_factor)
        coarse = _coarse_registration(source_workset, target_workset, config)
    except (RuntimeError, ValueError) as exc:
        return _rejected_result(source_id, target_id, [f'coarse_registration_failed:{type(exc).__name__}'], source_original, target_original, source_hash, target_hash, config, contract, source_isotropy, target_isotropy)
    coarse_yup = project_yaw_translation(coarse.transformation)
    candidates = [coarse_yup]
    if initial_transform_zup is not None:
        initial = np.asarray(initial_transform_zup, dtype=np.float64)
        if initial.shape != (4, 4) or not np.isfinite(initial).all():
            return _rejected_result(source_id, target_id, ['initial_transform_invalid'], source_original, target_original, source_hash, target_hash, config, contract, source_isotropy, target_isotropy)
        candidates.append(project_yaw_translation(_zup_transform_to_yup(initial)))
    first_scale = config.scales[0]
    initial_yup = max(candidates, key=lambda candidate: _candidate_score(source_workset.pyramid[0], target_workset.pyramid[0], first_scale.max_correspondence_m, candidate))
    refined_yup, iterations_run = _robust_gravity_locked_icp(source_workset, target_workset, initial_yup, config)
    colored_attempted = False
    colored_used = False
    if config.colored_icp and _has_usable_color(source_workset, target_workset, config):
        colored_attempted = True
        try:
            colored_yup = _colored_gravity_locked_icp(source_workset, target_workset, refined_yup, config)
            finest = config.scales[-1]
            geometric_score = _candidate_score(source_workset.pyramid[-1], target_workset.pyramid[-1], finest.max_correspondence_m, refined_yup)
            colored_score = _candidate_score(source_workset.pyramid[-1], target_workset.pyramid[-1], finest.max_correspondence_m, colored_yup)
            if colored_score >= geometric_score:
                refined_yup = colored_yup
                colored_used = True
        except RuntimeError:
            colored_used = False
    quality = _quality(source_workset, target_workset, refined_yup, config)
    reasons = _quality_reasons(quality, config)
    accepted = not reasons
    information = np.zeros((6, 6), dtype=np.float64)
    if accepted:
        finest = config.scales[-1]
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source_workset.pyramid[-1], target_workset.pyramid[-1], finest.max_correspondence_m, refined_yup)
    source_hash_after = _cloud_memory_hash(source_original)
    target_hash_after = _cloud_memory_hash(target_original)
    if source_hash_after != source_hash or target_hash_after != target_hash:
        reasons.append('source_mutation_detected')
        accepted = False
    coarse_eval = _evaluation(source_workset.pyramid[0], target_workset.pyramid[0], first_scale.max_correspondence_m, coarse_yup)
    coarse_count = len(coarse_eval.correspondence_set)
    return PairRegistrationResult(source_id=source_id, target_id=target_id, accepted=accepted, rejection_reasons=tuple(sorted(set(reasons))), transform_target_from_source_zup=project_zup_yaw_translation(_yup_transform_to_zup(refined_yup)), initial_transform_target_from_source_zup=project_zup_yaw_translation(_yup_transform_to_zup(initial_yup)), quality=quality, coarse_method=config.coarse_method, coarse_fitness=float(coarse_eval.fitness), coarse_inlier_rmse_m=float(coarse_eval.inlier_rmse) if coarse_count else None, iterations_run=iterations_run, colored_attempted=colored_attempted, colored_used=colored_used, source_point_count=len(source_original.points), target_point_count=len(target_original.points), source_isotropy_ratio=source_isotropy, target_isotropy_ratio=target_isotropy, source_memory_hash_before=source_hash, source_memory_hash_after=source_hash_after, target_memory_hash_before=target_hash, target_memory_hash_after=target_hash_after, information_yup=information)

def register_pair_files(source_path: str | Path, target_path: str | Path, **kwargs: Any) -> PairRegistrationResult:
    source_file = Path(source_path).resolve()
    target_file = Path(target_path).resolve()
    source_before = _file_hash(source_file)
    target_before = _file_hash(target_file)
    source = o3d.io.read_point_cloud(str(source_file), remove_nan_points=False, remove_infinite_points=False)
    target = o3d.io.read_point_cloud(str(target_file), remove_nan_points=False, remove_infinite_points=False)
    result = register_pair(source, target, **kwargs)
    source_after = _file_hash(source_file)
    target_after = _file_hash(target_file)
    reasons = list(result.rejection_reasons)
    if source_after != source_before:
        reasons.append('source_file_mutation_detected')
    if target_after != target_before:
        reasons.append('target_file_mutation_detected')
    file_mutation_detected = bool(source_after != source_before or target_after != target_before)
    return replace(result, accepted=result.accepted and (not file_mutation_detected), rejection_reasons=tuple(sorted(set(reasons))), source_file_hash_before=source_before, source_file_hash_after=source_after, target_file_hash_before=target_before, target_file_hash_after=target_after)
__all__ = ['AXIS_Z_UP', 'EDGE_SCHEMA', 'UNIT_METERS', 'PairQuality', 'PairRegistrationConfig', 'PairRegistrationResult', 'RegistrationContract', 'project_zup_yaw_translation', 'register_pair', 'register_pair_files']
