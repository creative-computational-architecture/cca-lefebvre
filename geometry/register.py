
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
try:
    from .ply_preserve import compatible_vertex_schema, merged_vertex_header, non_vertex_elements, read_ply_schema, transform_ply_preserving_properties
except ImportError:
    from ply_preserve import compatible_vertex_schema, merged_vertex_header, non_vertex_elements, read_ply_schema, transform_ply_preserving_properties
RUN_ID_RE = re.compile('^[A-Za-z0-9._-]+$')
SCHEMA_VERSION = 1
DEFAULT_SEED = 0

class ManifestError(ValueError):
    pass

@dataclass(frozen=True)
class RunSpec:
    run_id: str
    ply_text: str
    ply_path: Path
    gps_enu_m: np.ndarray
    seed_transform: np.ndarray | None

@dataclass(frozen=True)
class Manifest:
    path: Path
    reference_run_id: str
    runs: tuple[RunSpec, ...]
    pairs: tuple[tuple[str, str], ...] | None

@dataclass(frozen=True)
class ICPScale:
    voxel_m: float
    max_correspondence_m: float
    iterations: int

@dataclass(frozen=True)
class RegistrationConfig:
    scales: tuple[ICPScale, ...] = (ICPScale(0.8, 1.6, 35), ICPScale(0.4, 0.8, 25), ICPScale(0.2, 0.4, 20))
    min_fitness: float = 0.2
    max_inlier_rmse_m: float = 0.35
    min_correspondences: int = 300
    gps_neighbors: int = 3
    seed: int = DEFAULT_SEED

@dataclass
class CloudWorkset:
    point_count: int
    min_bound: np.ndarray
    max_bound: np.ndarray
    pyramid: tuple[o3d.geometry.PointCloud, ...]

@dataclass
class EdgeResult:
    source: str
    target: str
    gps_distance_m: float
    initial_transform: np.ndarray
    transform_source_to_target: np.ndarray
    fitness: float
    inlier_rmse_m: float
    correspondence_count: int
    accepted: bool
    rejection_reasons: tuple[str, ...]
    information: np.ndarray
    iterations_run: tuple[int, ...]
    certain: bool = False

def _as_matrix(value: Any, label: str, base_dir: Path) -> np.ndarray:
    if isinstance(value, str):
        matrix_path = Path(value)
        if not matrix_path.is_absolute():
            matrix_path = base_dir / matrix_path
        if not matrix_path.is_file():
            raise ManifestError(f'{label}: transform file does not exist: {value}')
        value = json.loads(matrix_path.read_text(encoding='utf-8'))
    if isinstance(value, dict):
        for key in ('matrix', 'transform', 'T_global_from_run'):
            if key in value:
                value = value[key]
                break
        else:
            raise ManifestError(f'{label}: transform object has no matrix field')
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ManifestError(f'{label}: transform must be 4x4')
    if not np.isfinite(matrix).all():
        raise ManifestError(f'{label}: transform contains non-finite values')
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-06):
        raise ManifestError(f'{label}: transform has an invalid homogeneous row')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.002):
        raise ManifestError(f'{label}: transform rotation is not rigid')
    if np.linalg.det(rotation) < 0.0:
        raise ManifestError(f'{label}: transform rotation contains a reflection')
    return matrix

def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise ManifestError(f'manifest does not exist: {path}')
    try:
        raw = json.loads(manifest_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ManifestError(f'invalid manifest JSON: {exc}') from exc
    if raw.get('version', SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ManifestError(f'manifest version must be {SCHEMA_VERSION}')
    raw_runs = raw.get('runs')
    if not isinstance(raw_runs, list) or len(raw_runs) < 2:
        raise ManifestError('manifest needs at least two runs')
    runs: list[RunSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_runs):
        if not isinstance(item, dict):
            raise ManifestError(f'runs[{index}] must be an object')
        run_id = item.get('run_id')
        if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
            raise ManifestError(f'runs[{index}].run_id must use ASCII letters, numbers, dot, dash, or underscore')
        if run_id in seen:
            raise ManifestError(f'duplicate run_id: {run_id}')
        seen.add(run_id)
        ply_text = item.get('ply')
        if not isinstance(ply_text, str) or not ply_text:
            raise ManifestError(f'{run_id}: ply is required')
        ply_path = Path(ply_text)
        if not ply_path.is_absolute():
            ply_path = manifest_path.parent / ply_path
        ply_path = ply_path.resolve()
        if not ply_path.is_file():
            raise ManifestError(f'{run_id}: PLY does not exist: {ply_text}')
        gps = item.get('gps_enu_m', item.get('gps_enu'))
        gps_array = np.asarray(gps, dtype=np.float64) if gps is not None else np.array([])
        if gps_array.shape != (3,) or not np.isfinite(gps_array).all():
            raise ManifestError(f'{run_id}: gps_enu_m must be [east, north, up]')
        seed_keys = [key for key in ('seed_transform', 'seed', 'transform') if key in item]
        if len(seed_keys) > 1:
            raise ManifestError(f'{run_id}: use only one of seed_transform, seed, transform')
        seed_transform = None
        if seed_keys:
            seed_transform = _as_matrix(item[seed_keys[0]], f'{run_id}.{seed_keys[0]}', manifest_path.parent)
        runs.append(RunSpec(run_id, ply_text, ply_path, gps_array, seed_transform))
    reference = raw.get('reference_run_id', runs[0].run_id)
    if reference not in seen:
        raise ManifestError(f'reference_run_id is unknown: {reference}')
    pair_data = raw.get('pairs')
    pairs: tuple[tuple[str, str], ...] | None = None
    if pair_data is not None:
        if not isinstance(pair_data, list) or not pair_data:
            raise ManifestError('pairs must be a non-empty list when provided')
        parsed: set[tuple[str, str]] = set()
        for index, pair in enumerate(pair_data):
            if isinstance(pair, dict):
                source, target = (pair.get('source'), pair.get('target'))
            elif isinstance(pair, list) and len(pair) == 2:
                source, target = pair
            else:
                raise ManifestError(f'pairs[{index}] must be [source, target] or an object')
            if source not in seen or target not in seen or source == target:
                raise ManifestError(f'pairs[{index}] contains an unknown or repeated run')
            parsed.add(tuple(sorted((source, target))))
        pairs = tuple(sorted(parsed))
    runs.sort(key=lambda run: run.run_id)
    return Manifest(manifest_path, reference, tuple(runs), pairs)

def yaw_matrix(yaw_rad: float) -> np.ndarray:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    return result

def yaw_from_transform(transform: np.ndarray) -> float:
    matrix = np.asarray(transform, dtype=np.float64)
    rotation = matrix[:3, :3]
    return float(math.atan2(rotation[0, 2] - rotation[2, 0], rotation[0, 0] + rotation[2, 2]))

def project_yaw_translation(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('transform must be a finite 4x4 matrix')
    projected = yaw_matrix(yaw_from_transform(matrix))
    projected[:3, 3] = matrix[:3, 3]
    return projected

def is_gravity_locked(transform: np.ndarray, atol: float=1e-08) -> bool:
    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    return bool(np.allclose(rotation @ np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0], atol=atol) and np.allclose(rotation.T @ rotation, np.eye(3), atol=atol) and np.isclose(np.linalg.det(rotation), 1.0, atol=atol))

def _gps_xyz(gps_enu_m: np.ndarray) -> np.ndarray:
    east, north, up = np.asarray(gps_enu_m, dtype=np.float64)
    return np.array([east, up, north], dtype=np.float64)

def initial_global_transforms(manifest: Manifest) -> dict[str, np.ndarray]:
    explicit: dict[str, np.ndarray] = {}
    for run in manifest.runs:
        if run.seed_transform is not None:
            explicit[run.run_id] = project_yaw_translation(run.seed_transform)
        else:
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = _gps_xyz(run.gps_enu_m)
            explicit[run.run_id] = transform
    anchor_inverse = np.linalg.inv(explicit[manifest.reference_run_id])
    return {run_id: project_yaw_translation(anchor_inverse @ transform) for run_id, transform in explicit.items()}

def _gps_distance(a: RunSpec, b: RunSpec) -> float:
    return float(np.linalg.norm(a.gps_enu_m - b.gps_enu_m))

def _mst_pairs(runs: Sequence[RunSpec]) -> set[tuple[str, str]]:
    parent = {run.run_id: run.run_id for run in runs}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> bool:
        ra, rb = (find(a), find(b))
        if ra == rb:
            return False
        parent[max(ra, rb)] = min(ra, rb)
        return True
    weighted: list[tuple[float, str, str]] = []
    for i, source in enumerate(runs):
        for target in runs[i + 1:]:
            weighted.append((_gps_distance(source, target), source.run_id, target.run_id))
    result: set[tuple[str, str]] = set()
    for _, source, target in sorted(weighted):
        if union(source, target):
            result.add(tuple(sorted((source, target))))
    return result

def candidate_pairs(manifest: Manifest, gps_neighbors: int=3) -> tuple[tuple[str, str], ...]:
    if manifest.pairs is not None:
        return manifest.pairs
    runs = manifest.runs
    result = _mst_pairs(runs)
    k = max(1, min(int(gps_neighbors), len(runs) - 1))
    for source in runs:
        nearest = sorted(((_gps_distance(source, target), target.run_id) for target in runs if target.run_id != source.run_id))[:k]
        for _, target_id in nearest:
            result.add(tuple(sorted((source.run_id, target_id))))
    return tuple(sorted(result))

def _validate_config(config: RegistrationConfig) -> None:
    if not config.scales:
        raise ValueError('at least one ICP scale is required')
    previous = float('inf')
    for scale in config.scales:
        if scale.voxel_m <= 0.0 or scale.max_correspondence_m <= 0.0:
            raise ValueError('ICP voxel and correspondence distances must be positive')
        if scale.iterations < 1:
            raise ValueError('ICP scale iterations must be positive')
        if scale.voxel_m > previous:
            raise ValueError('ICP scales must be ordered coarse to fine')
        previous = scale.voxel_m
    if not 0.0 <= config.min_fitness <= 1.0:
        raise ValueError('min_fitness must be between zero and one')
    if config.max_inlier_rmse_m <= 0.0 or config.min_correspondences < 1:
        raise ValueError('evidence thresholds must be positive')

def _prepare_cloud(path: Path, scales: Sequence[ICPScale]) -> CloudWorkset:
    cloud = o3d.io.read_point_cloud(str(path), remove_nan_points=False, remove_infinite_points=False)
    count = len(cloud.points)
    if count < 3:
        raise ValueError(f'PLY has fewer than three points: {path}')
    points = np.asarray(cloud.points)
    if not np.isfinite(points).all():
        raise ValueError(f'PLY contains non-finite points: {path}')
    pyramid: list[o3d.geometry.PointCloud] = []
    for scale in scales:
        sampled = cloud.voxel_down_sample(scale.voxel_m)
        if len(sampled.points) < 3:
            raise ValueError(f'PLY has too little geometry at voxel {scale.voxel_m}: {path}')
        sampled.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=scale.voxel_m * 3.0, max_nn=40))
        pyramid.append(sampled)
    return CloudWorkset(count, cloud.get_min_bound(), cloud.get_max_bound(), tuple(pyramid))

def _wrapped_angle_delta(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

def gravity_locked_multiscale_icp(source: CloudWorkset, target: CloudWorkset, initial_transform: np.ndarray, scales: Sequence[ICPScale]) -> tuple[np.ndarray, tuple[int, ...]]:
    current = project_yaw_translation(initial_transform)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    iteration_counts: list[int] = []
    for source_cloud, target_cloud, scale in zip(source.pyramid, target.pyramid, scales):
        used = 0
        for _ in range(scale.iterations):
            before = current.copy()
            registration = o3d.pipelines.registration.registration_icp(source_cloud, target_cloud, scale.max_correspondence_m, current, estimation, o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0, relative_rmse=0.0, max_iteration=1))
            current = project_yaw_translation(registration.transformation)
            used += 1
            translation_delta = float(np.linalg.norm(current[:3, 3] - before[:3, 3]))
            yaw_delta = _wrapped_angle_delta(yaw_from_transform(current), yaw_from_transform(before))
            if translation_delta < 1e-07 and yaw_delta < 1e-07:
                break
        iteration_counts.append(used)
    return (current, tuple(iteration_counts))

def register_edge(source_spec: RunSpec, target_spec: RunSpec, source_cloud: CloudWorkset, target_cloud: CloudWorkset, source_global_seed: np.ndarray, target_global_seed: np.ndarray, config: RegistrationConfig) -> EdgeResult:
    initial = project_yaw_translation(np.linalg.inv(target_global_seed) @ source_global_seed)
    refined, iterations_run = gravity_locked_multiscale_icp(source_cloud, target_cloud, initial, config.scales)
    finest = config.scales[-1]
    evaluation = o3d.pipelines.registration.evaluate_registration(source_cloud.pyramid[-1], target_cloud.pyramid[-1], finest.max_correspondence_m, refined)
    fitness = float(evaluation.fitness)
    rmse = float(evaluation.inlier_rmse)
    correspondence_count = len(evaluation.correspondence_set)
    reasons: list[str] = []
    if fitness < config.min_fitness:
        reasons.append('fitness_below_min')
    if not math.isfinite(rmse) or rmse > config.max_inlier_rmse_m:
        reasons.append('rmse_above_max')
    if correspondence_count < config.min_correspondences:
        reasons.append('correspondence_count_below_min')
    accepted = not reasons
    information = np.zeros((6, 6), dtype=np.float64)
    if accepted:
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source_cloud.pyramid[-1], target_cloud.pyramid[-1], finest.max_correspondence_m, refined)
    return EdgeResult(source=source_spec.run_id, target=target_spec.run_id, gps_distance_m=_gps_distance(source_spec, target_spec), initial_transform=initial, transform_source_to_target=refined, fitness=fitness, inlier_rmse_m=rmse, correspondence_count=correspondence_count, accepted=accepted, rejection_reasons=tuple(reasons), information=information, iterations_run=iterations_run)

def _accepted_components(run_ids: Sequence[str], edges: Sequence[EdgeResult]) -> list[list[str]]:
    adjacency = {run_id: set() for run_id in run_ids}
    for edge in edges:
        if edge.accepted:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
    unseen = set(run_ids)
    components: list[list[str]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        component: list[str] = []
        unseen.remove(start)
        while stack:
            item = stack.pop()
            component.append(item)
            for neighbor in sorted(adjacency[item], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda items: items[0])

def _mark_certain_spanning_edges(run_ids: Sequence[str], edges: Sequence[EdgeResult]) -> None:
    for edge in edges:
        edge.certain = False
    parent = {run_id: run_id for run_id in run_ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> bool:
        ra, rb = (find(a), find(b))
        if ra == rb:
            return False
        parent[max(ra, rb)] = min(ra, rb)
        return True
    ranked = sorted((edge for edge in edges if edge.accepted), key=lambda edge: (-edge.fitness, edge.inlier_rmse_m, -edge.correspondence_count, edge.source, edge.target))
    for edge in ranked:
        if union(edge.source, edge.target):
            edge.certain = True

def _pose_from_parameters(values: Sequence[float]) -> np.ndarray:
    pose = yaw_matrix(float(values[0]))
    pose[:3, 3] = np.asarray(values[1:4], dtype=np.float64)
    return pose

def _edge_error_vector(source_global: np.ndarray, target_global: np.ndarray, measured_source_to_target: np.ndarray) -> np.ndarray:
    predicted = project_yaw_translation(np.linalg.inv(target_global) @ source_global)
    measured = project_yaw_translation(measured_source_to_target)
    translation_error = predicted[:3, 3] - measured[:3, 3]
    yaw_error = math.atan2(math.sin(yaw_from_transform(predicted) - yaw_from_transform(measured)), math.cos(yaw_from_transform(predicted) - yaw_from_transform(measured)))
    return np.array([*translation_error, yaw_error], dtype=np.float64)

def pose_graph_residual_summary(transforms: dict[str, np.ndarray], edges: Sequence[EdgeResult]) -> dict[str, float]:
    residuals = [_edge_error_vector(transforms[edge.source], transforms[edge.target], edge.transform_source_to_target) for edge in edges if edge.accepted]
    if not residuals:
        return {'accepted_edge_count': 0, 'translation_rmse_m': 0.0, 'yaw_rmse_deg': 0.0, 'combined_rmse': 0.0}
    array = np.vstack(residuals)
    translation_rmse = float(np.sqrt(np.mean(np.sum(array[:, :3] ** 2, axis=1))))
    yaw_rmse = float(np.sqrt(np.mean(array[:, 3] ** 2)))
    combined = float(np.sqrt(np.mean(np.sum(array[:, :3] ** 2, axis=1) + array[:, 3] ** 2)))
    return {'accepted_edge_count': len(residuals), 'translation_rmse_m': translation_rmse, 'yaw_rmse_deg': math.degrees(yaw_rmse), 'combined_rmse': combined}

def _tree_initialized_poses(run_ids: Sequence[str], reference_run_id: str, edges: Sequence[EdgeResult], fallback: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    poses = {reference_run_id: np.eye(4, dtype=np.float64)}
    tree_edges = sorted((edge for edge in edges if edge.accepted and edge.certain), key=lambda edge: (edge.source, edge.target))
    while len(poses) < len(run_ids):
        changed = False
        for edge in tree_edges:
            measured = project_yaw_translation(edge.transform_source_to_target)
            if edge.source in poses and edge.target not in poses:
                poses[edge.target] = project_yaw_translation(poses[edge.source] @ np.linalg.inv(measured))
                changed = True
            elif edge.target in poses and edge.source not in poses:
                poses[edge.source] = project_yaw_translation(poses[edge.target] @ measured)
                changed = True
        if not changed:
            break
    for run_id in run_ids:
        poses.setdefault(run_id, project_yaw_translation(fallback[run_id]))
    return poses

def optimize_pose_graph(manifest: Manifest, edges: Sequence[EdgeResult], initial_global: dict[str, np.ndarray], config: RegistrationConfig) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    run_ids = [run.run_id for run in manifest.runs]
    _mark_certain_spanning_edges(run_ids, edges)
    accepted = [edge for edge in edges if edge.accepted]
    initial = _tree_initialized_poses(run_ids, manifest.reference_run_id, accepted, initial_global)
    free_ids = [run_id for run_id in run_ids if run_id != manifest.reference_run_id]
    x0 = np.concatenate([np.array([yaw_from_transform(initial[run_id]), *initial[run_id][:3, 3]], dtype=np.float64) for run_id in free_ids])
    raw_weights = np.array([edge.fitness * math.sqrt(max(edge.correspondence_count, 1)) / max(edge.inlier_rmse_m, 0.0001) for edge in accepted], dtype=np.float64)
    median_weight = float(np.median(raw_weights)) if len(raw_weights) else 1.0
    weights = np.clip(raw_weights / max(median_weight, 1e-12), 0.2, 5.0)
    translation_scale = max(config.scales[-1].voxel_m, 0.01)
    yaw_scale = math.radians(10.0)

    def unpack(parameters: np.ndarray) -> dict[str, np.ndarray]:
        poses = {manifest.reference_run_id: np.eye(4, dtype=np.float64)}
        for index, run_id in enumerate(free_ids):
            poses[run_id] = _pose_from_parameters(parameters[index * 4:index * 4 + 4])
        return poses

    def residual_function(parameters: np.ndarray) -> np.ndarray:
        poses = unpack(parameters)
        output: list[np.ndarray] = []
        for edge, weight in zip(accepted, weights):
            error = _edge_error_vector(poses[edge.source], poses[edge.target], edge.transform_source_to_target)
            normalized = np.array([error[0] / translation_scale, error[1] / translation_scale, error[2] / translation_scale, error[3] / yaw_scale])
            output.append(math.sqrt(float(weight)) * normalized)
        return np.concatenate(output)
    before = pose_graph_residual_summary(initial, accepted)
    solution = least_squares(residual_function, x0, method='trf', loss='soft_l1', f_scale=1.0, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500)
    optimized = {run_id: project_yaw_translation(pose) for run_id, pose in unpack(solution.x).items()}
    after = pose_graph_residual_summary(optimized, accepted)
    diagnostics: dict[str, Any] = {'optimizer': 'scipy_least_squares_4dof_soft_l1', 'reference_fixed': True, 'success': bool(solution.success), 'message': _ascii_text(solution.message), 'nfev': int(solution.nfev), 'cost': float(solution.cost), 'translation_scale_m': translation_scale, 'yaw_scale_deg': 10.0, 'before': before, 'after': after}
    if not solution.success:
        raise RuntimeError(f'4-DOF pose-graph optimization failed: {solution.message}')
    if after['combined_rmse'] > before['combined_rmse'] + 1e-10:
        raise RuntimeError('4-DOF pose-graph optimization increased edge residual')
    return (optimized, diagnostics)

def _matrix_json(matrix: np.ndarray) -> list[list[float]]:
    clean = np.asarray(matrix, dtype=np.float64).copy()
    clean[np.abs(clean) < 5e-13] = 0.0
    return [[round(float(value), 12) for value in row] for row in clean]

def _ascii_text(value: Any) -> str:
    return str(value).encode('ascii', 'backslashreplace').decode('ascii')

def _edge_json(edge: EdgeResult) -> dict[str, Any]:
    return {'accepted': edge.accepted, 'certain_pose_graph_edge': edge.certain, 'correspondence_count': edge.correspondence_count, 'fitness': round(edge.fitness, 9), 'gps_distance_m': round(edge.gps_distance_m, 6), 'initial_transform_source_to_target': _matrix_json(edge.initial_transform), 'inlier_rmse_m': round(edge.inlier_rmse_m, 9), 'iterations_run': list(edge.iterations_run), 'rejection_reasons': list(edge.rejection_reasons), 'source': edge.source, 'target': edge.target, 'transform_source_to_target': _matrix_json(edge.transform_source_to_target)}

def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def _write_json_ascii(path: Path, data: dict[str, Any]) -> None:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + '\n'
    _write_bytes_atomic(path, text.encode('ascii'))

def _build_report(data: dict[str, Any]) -> str:
    lines = ['# LiDAR Merge Report', '', f"Status: `{data['status']}`", f"Reference run: `{data['reference_run_id']}`", 'Gravity constraint: ARKit Y-up; independent yaw plus XYZ per run.', 'GPS role: coarse translation seed and candidate proposal only.', 'Point policy: source clouds untouched; no point or range rejection.', '', '## Runs', '', '| Run | Points | PLY | Aligned preview |', '|---|---:|---|---|']
    for run_id, run in sorted(data['runs'].items()):
        lines.append(f"| `{run_id}` | {run['point_count']} | `{_ascii_text(run['ply'])}` | `{_ascii_text(run.get('aligned_preview', '-'))}` |")
    lines.extend(['', '## Pairwise geometric evidence', '', '| Source | Target | Accepted | Fitness | RMSE m | Correspondences | Reasons |', '|---|---|---|---:|---:|---:|---|'])
    for edge in data['edges']:
        reasons = ','.join(edge['rejection_reasons']) or '-'
        lines.append(f"| `{edge['source']}` | `{edge['target']}` | {str(edge['accepted']).lower()} | {edge['fitness']:.6f} | {edge['inlier_rmse_m']:.6f} | {edge['correspondence_count']} | {reasons} |")
    if data.get('components'):
        lines.extend(['', '## Accepted-edge components', ''])
        for component in data['components']:
            lines.append('- ' + ', '.join((f'`{run_id}`' for run_id in component)))
    transforms = data.get('transforms', {})
    if transforms:
        pose_graph = data['pose_graph']
        lines.extend(['', '## 4-DOF pose graph', '', f"Optimizer: `{pose_graph['optimizer']}`", f"Reference fixed: {str(pose_graph['reference_fixed']).lower()}", f"Accepted-edge combined residual before: {pose_graph['before']['combined_rmse']:.9f}", f"Accepted-edge combined residual after: {pose_graph['after']['combined_rmse']:.9f}"])
        lines.extend(['', '## Final transforms', '', 'Convention: `p_global = T_global_from_run @ p_run`.', '', '| Run | Yaw deg | X m | Y m | Z m |', '|---|---:|---:|---:|---:|'])
        for run_id, item in sorted(transforms.items()):
            tx, ty, tz = item['translation_xyz_m']
            lines.append(f"| `{run_id}` | {item['yaw_deg']:.6f} | {tx:.6f} | {ty:.6f} | {tz:.6f} |")
    lines.extend(['', '## Notes', ''])
    if data['status'] == 'ok':
        lines.append('- All accepted runs are connected to the reference by geometric evidence.')
        if data.get('preview_status') == 'full_merge_written':
            lines.append('- Full-resolution previews preserve every input vertex property.')
        elif data.get('preview_status') == 'heterogeneous_schema_no_full_merge':
            lines.append('- Registration and per-run aligned previews completed.')
            lines.append('- Full merged preview was not written because PLY schemas differ.')
            lines.append('- No property was silently dropped or filled with an invented value.')
        elif data.get('preview_status') == 'non_vertex_elements_no_full_merge':
            lines.append('- Registration and per-run aligned previews completed.')
            lines.append('- Full merged preview was not written because a source has non-vertex elements.')
            lines.append('- Faces or other elements were not silently dropped.')
    else:
        lines.append('- Registration stopped before pose-graph output because evidence is disconnected.')
        lines.append('- Rejected edges were not inserted into the pose graph.')
    lines.append('')
    report = '\n'.join(lines)
    report.encode('ascii')
    return report

def _write_report(path: Path, data: dict[str, Any]) -> None:
    _write_bytes_atomic(path, _build_report(data).encode('ascii'))

def _write_full_previews(manifest: Manifest, transforms: dict[str, np.ndarray], output_dir: Path) -> tuple[dict[str, str], str | None, dict[str, int], str]:
    schemas = {run.run_id: read_ply_schema(run.ply_path) for run in manifest.runs}
    first_schema = schemas[manifest.runs[0].run_id]
    homogeneous = all((compatible_vertex_schema(first_schema, schemas[run.run_id]) for run in manifest.runs[1:]))
    vertex_only = all((not non_vertex_elements(schema) for schema in schemas.values()))
    mergeable = homogeneous and vertex_only
    if not homogeneous:
        preview_status = 'heterogeneous_schema_no_full_merge'
    elif not vertex_only:
        preview_status = 'non_vertex_elements_no_full_merge'
    else:
        preview_status = 'full_merge_written'
    expected = sum((schema.vertex.count for schema in schemas.values()))
    aligned_paths: dict[str, str] = {}
    output_counts: dict[str, int] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.registration-preview-', dir=output_dir) as raw:
        stage = Path(raw)
        stage_aligned = stage / 'aligned-preview'
        stage_aligned.mkdir()
        stage_merged = stage / 'merged-preview.ply' if mergeable else None
        if stage_merged is not None:
            merged_context = stage_merged.open('wb')
            merged_context.write(merged_vertex_header(first_schema, expected))
        else:
            merged_context = None
        try:
            for run in manifest.runs:
                destination = stage_aligned / f'{run.run_id}.ply'
                written_schema = transform_ply_preserving_properties(run.ply_path, destination, transforms[run.run_id], merged_vertex_stream=merged_context)
                if written_schema.vertex.count != schemas[run.run_id].vertex.count:
                    raise RuntimeError(f'aligned preview lost points for {run.run_id}')
                aligned_paths[run.run_id] = f'aligned-preview/{run.run_id}.ply'
                output_counts[run.run_id] = written_schema.vertex.count
        finally:
            if merged_context is not None:
                merged_context.close()
        if stage_merged is not None:
            merged_schema = read_ply_schema(stage_merged)
            if merged_schema.vertex.count != expected:
                raise RuntimeError('merged preview point count differs from input total')
        final_aligned = output_dir / 'aligned-preview'
        previous_aligned = stage / 'previous-aligned-preview'
        if final_aligned.exists():
            os.replace(final_aligned, previous_aligned)
        try:
            os.replace(stage_aligned, final_aligned)
        except BaseException:
            if previous_aligned.exists() and (not final_aligned.exists()):
                os.replace(previous_aligned, final_aligned)
            raise
        final_merged = output_dir / 'merged-preview.ply'
        previous_merged = stage / 'previous-merged-preview.ply'
        if final_merged.exists():
            os.replace(final_merged, previous_merged)
        try:
            if stage_merged is not None:
                os.replace(stage_merged, final_merged)
        except BaseException:
            if previous_merged.exists() and (not final_merged.exists()):
                os.replace(previous_merged, final_merged)
            raise
    merged_text = 'merged-preview.ply' if mergeable else None
    return (aligned_paths, merged_text, output_counts, preview_status)

def _clear_stale_previews(output_dir: Path) -> None:
    merged = output_dir / 'merged-preview.ply'
    merged.unlink(missing_ok=True)
    aligned = output_dir / 'aligned-preview'
    if aligned.is_dir():
        shutil.rmtree(aligned)
    elif aligned.exists():
        aligned.unlink()

def _validate_artifact_paths(manifest: Manifest, output_dir: Path, report: Path) -> None:
    sources = {run.ply_path.resolve() for run in manifest.runs}
    artifacts = {'transform_json': (output_dir / 'registration-transforms.json').resolve(), 'merged_preview': (output_dir / 'merged-preview.ply').resolve(), 'report': report.resolve()}
    artifacts.update({f'aligned_preview_{run.run_id}': (output_dir / 'aligned-preview' / f'{run.run_id}.ply').resolve() for run in manifest.runs})
    for label, path in artifacts.items():
        if path in sources:
            raise ValueError(f'{label} would overwrite an input PLY: {path}')
    reverse: dict[Path, str] = {}
    for label, path in artifacts.items():
        if path in reverse:
            raise ValueError(f'output paths overlap: {reverse[path]} and {label}: {path}')
        reverse[path] = label

def dry_run_plan(manifest: Manifest, config: RegistrationConfig) -> dict[str, Any]:
    _validate_config(config)
    runs = {run.run_id: run for run in manifest.runs}
    pairs = candidate_pairs(manifest, config.gps_neighbors)
    return {'dry_run': True, 'reference_run_id': manifest.reference_run_id, 'run_count': len(manifest.runs), 'runs': [{'run_id': run.run_id, 'ply': run.ply_text, 'gps_enu_m': [float(value) for value in run.gps_enu_m], 'has_explicit_seed': run.seed_transform is not None} for run in manifest.runs], 'candidate_pairs': [{'source': source, 'target': target, 'gps_distance_m': round(_gps_distance(runs[source], runs[target]), 6)} for source, target in pairs], 'scales': [{'voxel_m': scale.voxel_m, 'max_correspondence_m': scale.max_correspondence_m, 'iterations': scale.iterations} for scale in config.scales], 'evidence_gate': {'min_fitness': config.min_fitness, 'max_inlier_rmse_m': config.max_inlier_rmse_m, 'min_correspondences': config.min_correspondences}, 'seed': config.seed}

def _run_registration_impl(manifest: Manifest, output_dir: str | Path, report_path: str | Path, config: RegistrationConfig | None=None) -> dict[str, Any]:
    if config is None:
        config = RegistrationConfig()
    _validate_config(config)
    o3d.utility.random.seed(config.seed)
    output = Path(output_dir)
    report = Path(report_path)
    _validate_artifact_paths(manifest, output, report)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'registration-transforms.json'
    _write_json_ascii(status_path, {'schema_version': SCHEMA_VERSION, 'status': 'in_progress', 'reference_run_id': manifest.reference_run_id})
    _write_bytes_atomic(report, b'# LiDAR Merge Report\n\nStatus: `in_progress`\n')
    worksets = {run.run_id: _prepare_cloud(run.ply_path, config.scales) for run in manifest.runs}
    runs = {run.run_id: run for run in manifest.runs}
    initial = initial_global_transforms(manifest)
    edges = [register_edge(runs[source], runs[target], worksets[source], worksets[target], initial[source], initial[target], config) for source, target in candidate_pairs(manifest, config.gps_neighbors)]
    components = _accepted_components([run.run_id for run in manifest.runs], edges)
    connected = len(components) == 1
    data: dict[str, Any] = {'schema_version': SCHEMA_VERSION, 'status': 'ok' if connected else 'insufficient_geometric_evidence', 'reference_run_id': manifest.reference_run_id, 'transform_convention': 'p_global = T_global_from_run @ p_run', 'gravity_constraint': 'ARKit Y-up; independent yaw plus XYZ per run', 'gps_role': 'coarse translation seed and candidate proposal only', 'point_policy': 'no input point or depth-range rejection; rigid transforms only', 'config': dry_run_plan(manifest, config), 'runs': {run.run_id: {'ply': run.ply_text, 'gps_enu_m': [round(float(value), 6) for value in run.gps_enu_m], 'point_count': worksets[run.run_id].point_count, 'bbox_min_xyz_m': [round(float(value), 6) for value in worksets[run.run_id].min_bound], 'bbox_max_xyz_m': [round(float(value), 6) for value in worksets[run.run_id].max_bound]} for run in manifest.runs}, 'edges': [_edge_json(edge) for edge in edges], 'components': components, 'preview_status': 'not_written_insufficient_geometric_evidence', 'transforms': {}}
    if connected:
        transforms, pose_graph_diagnostics = optimize_pose_graph(manifest, edges, initial, config)
        aligned, merged, output_counts, preview_status = _write_full_previews(manifest, transforms, output)
        for run_id, path in aligned.items():
            data['runs'][run_id]['aligned_preview'] = path
            if output_counts[run_id] != data['runs'][run_id]['point_count']:
                raise RuntimeError(f'aligned preview lost points for {run_id}')
        data['preview_status'] = preview_status
        data['aligned_point_count_total'] = sum(output_counts.values())
        if merged is not None:
            data['merged_preview'] = merged
            data['merged_point_count'] = sum(output_counts.values())
        data['pose_graph'] = pose_graph_diagnostics
        data['transforms'] = {run_id: {'T_global_from_run': _matrix_json(transform), 'yaw_deg': round(math.degrees(yaw_from_transform(transform)), 9), 'translation_xyz_m': [round(float(value), 9) for value in transform[:3, 3]], 'gravity_locked': is_gravity_locked(transform)} for run_id, transform in sorted(transforms.items())}
    else:
        _clear_stale_previews(output)
    data['edges'] = [_edge_json(edge) for edge in edges]
    _write_json_ascii(status_path, data)
    _write_report(report, data)
    return data

def run_registration(manifest: Manifest, output_dir: str | Path, report_path: str | Path, config: RegistrationConfig | None=None) -> dict[str, Any]:
    output = Path(output_dir)
    report = Path(report_path)
    status_path = output / 'registration-transforms.json'
    try:
        return _run_registration_impl(manifest, output, report, config)
    except (ManifestError, ValueError, RuntimeError, OSError) as exc:
        marker_is_current = False
        try:
            marker = json.loads(status_path.read_text(encoding='ascii'))
            marker_is_current = marker.get('status') == 'in_progress'
        except (OSError, ValueError):
            pass
        if marker_is_current:
            failure = {'schema_version': SCHEMA_VERSION, 'status': 'error', 'reference_run_id': manifest.reference_run_id, 'error_type': type(exc).__name__, 'error': _ascii_text(exc)}
            try:
                _write_json_ascii(status_path, failure)
                _write_bytes_atomic(report, f'# LiDAR Merge Report\n\nStatus: `error`\n\nError: `{_ascii_text(exc)}`\n'.encode('ascii'))
            except OSError:
                pass
        raise

def _csv_numbers(text: str, cast: type=float) -> list[Any]:
    try:
        return [cast(part.strip()) for part in text.split(',') if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc

def _config_from_args(args: argparse.Namespace) -> RegistrationConfig:
    voxels = _csv_numbers(args.voxel_sizes, float)
    distances = _csv_numbers(args.max_correspondence, float)
    iterations = _csv_numbers(args.iterations, int)
    if not len(voxels) == len(distances) == len(iterations):
        raise ManifestError('voxel-sizes, max-correspondence, and iterations need the same item count')
    return RegistrationConfig(scales=tuple((ICPScale(voxel, distance, iteration) for voxel, distance, iteration in zip(voxels, distances, iterations))), min_fitness=args.min_fitness, max_inlier_rmse_m=args.max_inlier_rmse, min_correspondences=args.min_correspondences, gps_neighbors=args.gps_neighbors, seed=args.seed)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Register independent ARKit Y-up PLY runs with GPS-seeded, gravity-locked Open3D ICP and a robust 4-DOF pose graph.')
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--out', type=Path, default=Path('data/registration'))
    parser.add_argument('--report', type=Path, default=Path('field/merge-report.md'))
    parser.add_argument('--dry-run', action='store_true', help='validate and print plan only')
    parser.add_argument('--voxel-sizes', default='0.8,0.4,0.2')
    parser.add_argument('--max-correspondence', default='1.6,0.8,0.4', help='ICP nearest-neighbor radii; these are not LiDAR range cutoffs')
    parser.add_argument('--iterations', default='35,25,20')
    parser.add_argument('--min-fitness', type=float, default=0.2)
    parser.add_argument('--max-inlier-rmse', type=float, default=0.35)
    parser.add_argument('--min-correspondences', type=int, default=300)
    parser.add_argument('--gps-neighbors', type=int, default=3)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    return parser

def main(argv: Sequence[str] | None=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        config = _config_from_args(args)
        if args.dry_run:
            plan = dry_run_plan(manifest, config)
            print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=True))
            return 0
        result = run_registration(manifest, args.out, args.report, config)
    except (ManifestError, ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f'registration error: {_ascii_text(exc)}\n')
    print(json.dumps({'status': result['status'], 'preview_status': result.get('preview_status'), 'transform_json': str(Path(args.out) / 'registration-transforms.json'), 'report': str(args.report)}, sort_keys=True, ensure_ascii=True))
    return 0 if result['status'] == 'ok' else 2
if __name__ == '__main__':
    raise SystemExit(main())
