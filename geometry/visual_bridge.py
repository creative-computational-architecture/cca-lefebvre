
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
try:
    from field.m0util import backproject_world, list_frames, load_pose_intrinsics_rgb_wh, read_depth, read_sensor_image, scale_K
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from field.m0util import backproject_world, list_frames, load_pose_intrinsics_rgb_wh, read_depth, read_sensor_image, scale_K
try:
    from .register import ICPScale, is_gravity_locked, project_yaw_translation, yaw_from_transform
except ImportError:
    from register import ICPScale, is_gravity_locked, project_yaw_translation, yaw_from_transform
SCHEMA = 'visual_depth_bridge_evidence_v2'
DEFAULT_SEED = 0
HASH_BYTE_PAIRS = ((0, 1), (4, 5), (8, 9), (12, 13), (16, 17), (20, 21), (24, 25), (28, 29))

class BridgeInputError(ValueError):
    pass

@dataclass(frozen=True)
class BridgeConfig:
    max_image_width: int = 960
    orb_features: int = 1400
    sift_features: int = 1600
    max_sift_fallbacks_per_pair: int = 24
    retrieval_top_k: int = 10
    max_candidates: int = 260
    max_temporal_followups: int = 96
    temporal_followup_radius: int = 5
    trajectory_neighbors: int = 3
    max_seed_candidate_distance_m: float = 12.0
    ratio_test: float = 0.76
    max_hamming: float = 72.0
    sift_ratio_test: float = 0.74
    homography_threshold_px: float = 8.0
    min_visual_matches: int = 18
    min_homography_inliers: int = 14
    min_homography_ratio: float = 0.25
    min_grid_cells: int = 4
    min_image_span_fraction: float = 0.2
    depth_neighborhood_px: int = 2
    ransac_iterations: int = 2200
    ransac_threshold_m: float = 0.28
    min_pair_3d_inliers: int = 8
    min_pair_3d_ratio: float = 0.22
    cluster_yaw_deg: float = 5.0
    cluster_translation_m: float = 1.5
    min_independent_pairs: int = 3
    min_camera_span_m: float = 0.75
    min_aggregate_3d_inliers: int = 60
    min_aggregate_3d_ratio: float = 0.25
    max_aggregate_median_m: float = 0.13
    max_aggregate_p90_m: float = 0.3
    min_inlier_span_m: float = 1.5
    min_loo_passes: int = 2
    max_loo_median_m: float = 0.2
    crop_margin_m: float = 3.0
    local_icp_trim_fraction: float = 0.8
    local_icp_min_correspondences: int = 30
    min_icp_refined_to_initial_fitness_ratio: float = 0.9
    max_icp_yaw_correction_deg: float = 3.0
    max_icp_translation_correction_m: float = 0.8
    min_local_symmetric_fitness_025m: float = 0.12
    max_rgb_ratio: float = 0.92
    max_seed_yaw_delta_deg: float = 18.0
    max_seed_translation_delta_m: float = 5.0
    random_seed: int = DEFAULT_SEED

@dataclass(frozen=True)
class RunInSide:
    run_id: str
    T_side_from_run: np.ndarray
    workset_path: Path | None

@dataclass(frozen=True)
class SideSpec:
    side_id: str
    anchor_run: str
    runs: tuple[RunInSide, ...]

@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    source_endpoint: str
    target_endpoint: str
    source: SideSpec
    target: SideSpec
    seed_target_from_source: np.ndarray
    seed_method: str

@dataclass(frozen=True)
class SearchUnit:
    side: SideSpec
    T_global_from_side: np.ndarray
    context_method: str

@dataclass(frozen=True)
class FrameSpec:
    run_id: str
    frame_index: int
    rgb_path: Path
    depth_path: Path
    json_path: Path
    conf_path: Path | None
    T_side_from_run: np.ndarray

@dataclass
class ImageFeature:
    frame: FrameSpec
    xy_full: np.ndarray
    descriptors: np.ndarray
    image_wh: tuple[int, int]
    camera_center: np.ndarray | None = None
    recovered_depths: np.ndarray | None = None
    recovery_support: np.ndarray | None = None

@dataclass(frozen=True)
class RetrievalIndex:
    postings: tuple[dict[int, list[int]], ...]
    target_word_counts: np.ndarray
    max_document_frequency: int

@dataclass
class PairEvidence:
    source_feature: ImageFeature
    target_feature: ImageFeature
    retrieval_score: float
    retrieval_shared_words: int
    descriptor_kind: str = 'ORB'
    visual_matches: int = 0
    homography_inliers: int = 0
    homography_ratio: float = 0.0
    source_grid_cells: int = 0
    target_grid_cells: int = 0
    source_image_span: tuple[float, float] = (0.0, 0.0)
    target_image_span: tuple[float, float] = (0.0, 0.0)
    rgb_ratio: float | None = None
    depth_correspondences: int = 0
    depth_conf_histogram: dict[str, int] = field(default_factory=dict)
    depth_range_min_m: float | None = None
    depth_range_max_m: float | None = None
    transform_target_from_source: np.ndarray | None = None
    ransac_inliers: int = 0
    ransac_ratio: float = 0.0
    residual_median_m: float | None = None
    residual_p90_m: float | None = None
    inlier_source_points: np.ndarray | None = None
    inlier_target_points: np.ndarray | None = None
    source_camera_center: np.ndarray | None = None
    target_camera_center: np.ndarray | None = None
    matched_source_xy: np.ndarray | None = None
    matched_target_xy: np.ndarray | None = None
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def has_3d_model(self) -> bool:
        return self.transform_target_from_source is not None and (not self.rejection_reasons)

def _matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise BridgeInputError(f'{label} must be a finite 4x4 matrix')
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-06):
        raise BridgeInputError(f'{label} has an invalid homogeneous row')
    projected = project_yaw_translation(matrix)
    if not np.allclose(projected, matrix, atol=0.002):
        raise BridgeInputError(f'{label} is not Y-up yaw plus XYZ translation')
    if not is_gravity_locked(projected, atol=2e-06):
        raise BridgeInputError(f'{label} is not a proper gravity-locked transform')
    return projected

def _resolve_repo_path(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()

def _component_for_run(evidence: dict[str, Any], run_id: str) -> dict[str, Any]:
    for component in evidence.get('components', []):
        if any((item.get('run') == run_id for item in component.get('runs', []))):
            return component
    raise BridgeInputError(f'run is absent from components: {run_id}')

def _run_side(component: dict[str, Any], item: dict[str, Any], repo_root: Path) -> SearchUnit:
    run_id = item.get('run')
    if not isinstance(run_id, str) or not run_id:
        raise BridgeInputError('component run has no run id')
    workset = _resolve_repo_path(repo_root, item.get('registration_workset'))
    if workset is not None and (not workset.is_file()):
        raise BridgeInputError(f'registration workset is missing: {workset}')
    global_from_run = _matrix(item.get('T_global_from_run'), f'{run_id}.T_global_from_run')
    component_id = str(component.get('id', run_id))
    status = component.get('status')
    unit_id = component_id if status == 'ISOLATED' else f'RUN_{run_id}'
    side = SideSpec(unit_id, run_id, (RunInSide(run_id, np.eye(4, dtype=np.float64), workset),))
    return SearchUnit(side, global_from_run, 'T_global_from_run')

def build_search_units(evidence: dict[str, Any], repo_root: str | Path) -> tuple[SearchUnit, ...]:
    root = Path(repo_root).resolve()
    units: list[SearchUnit] = []
    for component in evidence.get('components', []):
        if not isinstance(component, dict):
            raise BridgeInputError('component entry must be an object')
        status = component.get('status')
        component_id = component.get('id')
        runs_data = component.get('runs', [])
        if not isinstance(component_id, str) or not component_id:
            raise BridgeInputError('component has no id')
        if not isinstance(runs_data, list) or not runs_data:
            raise BridgeInputError(f'{component_id}: component has no runs')
        if status == 'ACCEPTED':
            side_runs: list[RunInSide] = []
            for item in runs_data:
                if not isinstance(item, dict):
                    raise BridgeInputError(f'{component_id}: run entry must be an object')
                run_id = item.get('run')
                if not isinstance(run_id, str) or not run_id:
                    raise BridgeInputError(f'{component_id}: component run has no run id')
                transform = _matrix(item.get('C_component_from_run'), f'{run_id}.C_component_from_run')
                workset = _resolve_repo_path(root, item.get('registration_workset'))
                if workset is not None and (not workset.is_file()):
                    raise BridgeInputError(f'registration workset is missing: {workset}')
                side_runs.append(RunInSide(run_id, transform, workset))
            anchor = component.get('anchor_run')
            if not isinstance(anchor, str) or anchor not in {item.run_id for item in side_runs}:
                raise BridgeInputError(f'{component_id}: invalid anchor_run')
            global_from_side = _matrix(component.get('H_global_seed_from_component'), f'{component_id}.H_global_seed_from_component')
            units.append(SearchUnit(SideSpec(component_id, anchor, tuple(side_runs)), global_from_side, 'H_global_seed_from_component'))
        elif status in {'PROVISIONAL', 'ISOLATED'}:
            for item in runs_data:
                if not isinstance(item, dict):
                    raise BridgeInputError(f'{component_id}: run entry must be an object')
                units.append(_run_side(component, item, root))
        else:
            raise BridgeInputError(f'{component_id}: unsafe component status for bridge search: {status}')
    units.sort(key=lambda item: min((run.run_id for run in item.side.runs)))
    ids = [item.side.side_id for item in units]
    if len(ids) != len(set(ids)):
        raise BridgeInputError('measurement unit ids are not unique')
    return tuple(units)

def build_all_cross_component_specs(evidence: dict[str, Any], repo_root: str | Path) -> tuple[tuple[SearchUnit, ...], tuple[PairSpec, ...]]:
    units = build_search_units(evidence, repo_root)
    specs: list[PairSpec] = []
    for index, (source_unit, target_unit) in enumerate(combinations(units, 2), 1):
        source = source_unit.side
        target = target_unit.side
        seed_target_from_source = project_yaw_translation(np.linalg.inv(target_unit.T_global_from_side) @ source_unit.T_global_from_side)
        specs.append(PairSpec(f'X{index:02d}_{source.side_id}__{target.side_id}', source.anchor_run, target.anchor_run, source, target, seed_target_from_source, f'global_context:{source_unit.context_method}->{target_unit.context_method}'))
    return (units, tuple(specs))

def _side_for_endpoint(evidence: dict[str, Any], endpoint: str, repo_root: Path) -> SideSpec:
    component = _component_for_run(evidence, endpoint)
    component_runs = component.get('runs', [])
    expand = component.get('status') == 'ACCEPTED'
    selected = component_runs if expand else [item for item in component_runs if item.get('run') == endpoint]
    runs: list[RunInSide] = []
    for item in selected:
        run_id = item.get('run')
        if not isinstance(run_id, str):
            raise BridgeInputError('component run has no run id')
        transform = _matrix(item.get('C_component_from_run'), f'{run_id}.C_component_from_run') if expand else np.eye(4, dtype=np.float64)
        workset = _resolve_repo_path(repo_root, item.get('registration_workset'))
        if workset is not None and (not workset.is_file()):
            raise BridgeInputError(f'registration workset is missing: {workset}')
        runs.append(RunInSide(run_id, transform, workset))
    if not runs:
        raise BridgeInputError(f'component has no usable runs for {endpoint}')
    anchor = str(component.get('anchor_run', endpoint)) if expand else endpoint
    side_id = str(component.get('id', endpoint)) if expand else endpoint
    return SideSpec(side_id, anchor, tuple(runs))

def load_pair_specs(evidence_path: str | Path, pair_ids: Sequence[str], repo_root: str | Path | None=None) -> tuple[dict[str, Any], tuple[PairSpec, ...], Path]:
    path = Path(evidence_path).resolve()
    if not path.is_file():
        raise BridgeInputError(f'evidence file does not exist: {evidence_path}')
    evidence = json.loads(path.read_text(encoding='utf-8'))
    if evidence.get('COLMAP_used') is not False:
        raise BridgeInputError('authoritative evidence must explicitly state COLMAP_used=false')
    root = Path(repo_root).resolve() if repo_root else path.parents[2]
    edge_map = {edge.get('id'): edge for edge in evidence.get('provisional_edges', []) if isinstance(edge, dict)}
    specs: list[PairSpec] = []
    for pair_id in pair_ids:
        edge = edge_map.get(pair_id)
        if edge is None:
            raise BridgeInputError(f'unknown provisional edge id: {pair_id}')
        source_endpoint, target_endpoint = (edge.get('source'), edge.get('target'))
        if not isinstance(source_endpoint, str) or not isinstance(target_endpoint, str):
            raise BridgeInputError(f'{pair_id}: source and target runs are required')
        source = _side_for_endpoint(evidence, source_endpoint, root)
        target = _side_for_endpoint(evidence, target_endpoint, root)
        endpoint_seed = _matrix(edge.get('T_target_from_source'), f'{pair_id}.seed')
        source_endpoint_transform = next((item.T_side_from_run for item in source.runs if item.run_id == source_endpoint))
        target_endpoint_transform = next((item.T_side_from_run for item in target.runs if item.run_id == target_endpoint))
        side_seed = project_yaw_translation(target_endpoint_transform @ endpoint_seed @ np.linalg.inv(source_endpoint_transform))
        specs.append(PairSpec(pair_id, source_endpoint, target_endpoint, source, target, side_seed, str(edge.get('method', 'provisional_context'))))
    return (evidence, tuple(specs), root)

def load_all_pair_specs(evidence_path: str | Path, repo_root: str | Path | None=None) -> tuple[dict[str, Any], tuple[SearchUnit, ...], tuple[PairSpec, ...], Path]:
    path = Path(evidence_path).resolve()
    if not path.is_file():
        raise BridgeInputError(f'evidence file does not exist: {evidence_path}')
    evidence = json.loads(path.read_text(encoding='utf-8'))
    if evidence.get('COLMAP_used') is not False:
        raise BridgeInputError('authoritative evidence must explicitly state COLMAP_used=false')
    root = Path(repo_root).resolve() if repo_root else path.parents[2]
    units, specs = build_all_cross_component_specs(evidence, root)
    return (evidence, units, specs, root)

def enumerate_side_frames(side: SideSpec, raw_root: str | Path) -> tuple[FrameSpec, ...]:
    root = Path(raw_root).resolve()
    output: list[FrameSpec] = []
    for run in side.runs:
        run_dir = root / run.run_id
        if not run_dir.is_dir():
            raise BridgeInputError(f'raw run directory is missing: {run_dir}')
        for raw in list_frames(run_dir):
            if raw.rgb_path is None:
                continue
            if raw.depth_path is None:
                raise BridgeInputError(f'RGB frame has no depth: {raw.rgb_path}')
            output.append(FrameSpec(run.run_id, raw.idx, raw.rgb_path, raw.depth_path, raw.json_path, raw.conf_path, run.T_side_from_run))
    return tuple(output)

def _imread(path: Path, flags: int) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if not len(encoded):
        return None
    return cv2.imdecode(encoded, flags)

def _imwrite(path: Path, image: np.ndarray, parameters: Sequence[int]=()) -> bool:
    success, encoded = cv2.imencode(path.suffix, image, list(parameters))
    if not success:
        return False
    try:
        encoded.tofile(path)
    except OSError:
        return False
    return True

def _read_depth_unicode(path: Path) -> np.ndarray:
    if path.suffix.lower() != '.png':
        return read_depth(path)
    array = _imread(path, cv2.IMREAD_UNCHANGED)
    if array is None:
        raise OSError(f'depth PNG cannot be read: {path}')
    if array.ndim != 2 or array.dtype != np.uint16:
        raise ValueError(f'{path.name}: uint16 single-channel depth was expected')
    output = array.astype(np.float32) * 0.001
    output[array == 0] = np.nan
    return output

def _extract_feature(frame: FrameSpec, config: BridgeConfig) -> ImageFeature | None:
    gray = read_sensor_image(frame.rgb_path, cv2.IMREAD_GRAYSCALE)
    height, width = gray.shape
    sensor_wh = (width, height)
    pose, _, rgb_wh = load_pose_intrinsics_rgb_wh(frame.json_path, legacy_rgb_wh=sensor_wh)
    if rgb_wh != sensor_wh:
        raise ValueError(f'{frame.rgb_path.name}: stored RGB {sensor_wh} != intrinsics basis {rgb_wh}')
    scale = min(1.0, config.max_image_width / float(width))
    work = gray
    if scale < 1.0:
        work = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(nfeatures=config.orb_features, scaleFactor=1.2, nlevels=8, edgeThreshold=19, fastThreshold=10)
    keypoints, descriptors = orb.detectAndCompute(work, None)
    if descriptors is None or len(descriptors) < 8:
        return None
    xy = np.asarray([item.pt for item in keypoints], dtype=np.float32) / scale
    camera_center = _apply(frame.T_side_from_run, pose[:3, 3][None, :])[0]
    return ImageFeature(frame, xy, np.asarray(descriptors, dtype=np.uint8), (width, height), camera_center)

def _extract_sift_feature(feature: ImageFeature, config: BridgeConfig) -> ImageFeature | None:
    gray = read_sensor_image(feature.frame.rgb_path, cv2.IMREAD_GRAYSCALE)
    sensor_wh = (int(gray.shape[1]), int(gray.shape[0]))
    _, _, rgb_wh = load_pose_intrinsics_rgb_wh(feature.frame.json_path, legacy_rgb_wh=sensor_wh)
    if rgb_wh != sensor_wh:
        raise ValueError(f'{feature.frame.rgb_path.name}: stored RGB {sensor_wh} != intrinsics basis {rgb_wh}')
    height, width = gray.shape
    scale = min(1.0, config.max_image_width / float(width))
    work = gray
    if scale < 1.0:
        work = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    detector = cv2.SIFT_create(nfeatures=config.sift_features, contrastThreshold=0.025)
    keypoints, descriptors = detector.detectAndCompute(work, None)
    if descriptors is None or len(descriptors) < 8:
        return None
    xy = np.asarray([item.pt for item in keypoints], dtype=np.float32) / scale
    return ImageFeature(feature.frame, xy, np.asarray(descriptors, dtype=np.float32), (width, height), feature.camera_center)

def extract_features(frames: Sequence[FrameSpec], config: BridgeConfig, progress_label: str='frames') -> tuple[ImageFeature, ...]:
    features: list[ImageFeature] = []
    for index, frame in enumerate(frames, 1):
        item = _extract_feature(frame, config)
        if item is not None:
            features.append(item)
        if index % 250 == 0 or index == len(frames):
            print(f'visual_bridge: {progress_label} features {index}/{len(frames)}', file=sys.stderr, flush=True)
    return tuple(features)

def _descriptor_words(descriptors: np.ndarray, byte_pair: tuple[int, int]) -> np.ndarray:
    first, second = byte_pair
    return np.unique(descriptors[:, first].astype(np.uint16) | descriptors[:, second].astype(np.uint16) << np.uint16(8))

def build_retrieval_index(target: Sequence[ImageFeature]) -> RetrievalIndex:
    postings: list[dict[int, list[int]]] = []
    target_word_counts = np.zeros(len(target), dtype=np.int64)
    for byte_pair in HASH_BYTE_PAIRS:
        table: dict[int, list[int]] = defaultdict(list)
        for target_index, item in enumerate(target):
            words = _descriptor_words(item.descriptors, byte_pair)
            target_word_counts[target_index] += len(words)
            for word in words:
                table[int(word)].append(target_index)
        postings.append(dict(table))
    return RetrievalIndex(tuple(postings), target_word_counts, max(3, math.ceil(len(target) * 0.15)))

def retrieve_candidates(source: Sequence[ImageFeature], target: Sequence[ImageFeature], config: BridgeConfig, retrieval_index: RetrievalIndex | None=None) -> list[tuple[int, int, float, int]]:
    if not source or not target:
        return []
    index = retrieval_index or build_retrieval_index(target)
    if len(index.target_word_counts) != len(target):
        raise ValueError('retrieval index does not match target feature count')
    source_top: list[tuple[int, int, float, int]] = []
    for source_index, item in enumerate(source):
        scores: dict[int, float] = defaultdict(float)
        shared: dict[int, int] = defaultdict(int)
        source_word_count = 0
        for table, byte_pair in zip(index.postings, HASH_BYTE_PAIRS):
            words = _descriptor_words(item.descriptors, byte_pair)
            source_word_count += len(words)
            for word in words:
                documents = table.get(int(word), ())
                frequency = len(documents)
                if frequency == 0 or frequency > index.max_document_frequency:
                    continue
                weight = math.log((len(target) + 1.0) / (frequency + 1.0))
                for target_index in documents:
                    scores[target_index] += weight
                    shared[target_index] += 1
        ranked: list[tuple[float, int, int]] = []
        for target_index, raw_score in scores.items():
            normalizer = math.sqrt(max(1.0, float(source_word_count * index.target_word_counts[target_index])))
            ranked.append((raw_score / normalizer, shared[target_index], target_index))
        ranked.sort(reverse=True)
        kept_targets: list[tuple[str, int]] = []
        kept = 0
        for score, word_count, target_index in ranked:
            target_frame = target[target_index].frame
            if any((run_id == target_frame.run_id and abs(frame_index - target_frame.frame_index) < 18 for run_id, frame_index in kept_targets)):
                continue
            source_top.append((source_index, target_index, score, word_count))
            kept_targets.append((target_frame.run_id, target_frame.frame_index))
            kept += 1
            if kept >= config.retrieval_top_k:
                break
    by_target: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for source_index, target_index, score, _ in source_top:
        by_target[target_index].append((score, source_index))
    reciprocal: set[tuple[int, int]] = set()
    for target_index, items in by_target.items():
        items.sort(reverse=True)
        for _, source_index in items[:config.retrieval_top_k]:
            reciprocal.add((source_index, target_index))
    candidates = [item for item in source_top if (item[0], item[1]) in reciprocal]
    candidates.sort(key=lambda item: (-item[2], -item[3], item[0], item[1]))
    diverse: list[tuple[int, int, float, int]] = []
    for candidate in candidates:
        source_index, target_index, _, _ = candidate
        sf, tf = (source[source_index].frame, target[target_index].frame)
        redundant = False
        for old_source, old_target, _, _ in diverse:
            osf, otf = (source[old_source].frame, target[old_target].frame)
            if sf.run_id == osf.run_id and tf.run_id == otf.run_id and (abs(sf.frame_index - osf.frame_index) < 18) and (abs(tf.frame_index - otf.frame_index) < 18):
                redundant = True
                break
        if not redundant:
            diverse.append(candidate)
        if len(diverse) >= config.max_candidates:
            break
    return diverse

def trajectory_seed_candidates(source: Sequence[ImageFeature], target: Sequence[ImageFeature], seed_target_from_source: np.ndarray, config: BridgeConfig) -> list[tuple[int, int, float, int]]:
    if not source or not target:
        return []
    target_centers = np.asarray([item.camera_center for item in target], dtype=np.float64)
    if target_centers.shape != (len(target), 3) or not np.isfinite(target_centers).all():
        return []
    source_centers = np.asarray([item.camera_center for item in source], dtype=np.float64)
    if source_centers.shape != (len(source), 3) or not np.isfinite(source_centers).all():
        return []
    predicted = _apply(seed_target_from_source, source_centers)
    neighbors = min(config.trajectory_neighbors, len(target))
    distances, indices = cKDTree(target_centers).query(predicted, k=neighbors)
    distances = np.asarray(distances)
    indices = np.asarray(indices)
    if neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    ranked: list[tuple[float, int, int]] = []
    for source_index in range(len(source)):
        for distance, target_index in zip(distances[source_index], indices[source_index]):
            if distance <= config.max_seed_candidate_distance_m:
                ranked.append((float(distance), source_index, int(target_index)))
    ranked.sort()
    diverse: list[tuple[int, int, float, int]] = []
    for distance, source_index, target_index in ranked:
        sf, tf = (source[source_index].frame, target[target_index].frame)
        if any((sf.run_id == source[old_source].frame.run_id and tf.run_id == target[old_target].frame.run_id and (abs(sf.frame_index - source[old_source].frame.frame_index) < 18) and (abs(tf.frame_index - target[old_target].frame.frame_index) < 18) for old_source, old_target, _, _ in diverse)):
            continue
        diverse.append((source_index, target_index, -distance, -1))
        if len(diverse) >= config.max_candidates:
            break
    return diverse

def _merge_candidate_proposals(visual: Sequence[tuple[int, int, float, int]], trajectory: Sequence[tuple[int, int, float, int]], limit: int) -> list[tuple[int, int, float, int]]:
    visual_budget = (limit + 1) // 2
    trajectory_budget = limit - visual_budget
    output: list[tuple[int, int, float, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(items: Iterable[tuple[int, int, float, int]], budget: int) -> None:
        added = 0
        for item in items:
            key = (item[0], item[1])
            if key in seen:
                continue
            output.append(item)
            seen.add(key)
            added += 1
            if added >= budget or len(output) >= limit:
                break
    add(visual, visual_budget)
    add(trajectory, trajectory_budget)
    if len(output) < limit:
        add(list(visual)[visual_budget:] + list(trajectory)[trajectory_budget:], limit - len(output))
    return output

def _mutual_ratio_matches(source: ImageFeature, target: ImageFeature, config: BridgeConfig) -> list[tuple[int, int, float]]:
    is_binary = source.descriptors.dtype == np.uint8 and target.descriptors.dtype == np.uint8
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING if is_binary else cv2.NORM_L2)
    ratio = config.ratio_test if is_binary else config.sift_ratio_test
    max_distance = config.max_hamming if is_binary else float('inf')
    forward = matcher.knnMatch(source.descriptors, target.descriptors, k=2)
    reverse = matcher.knnMatch(target.descriptors, source.descriptors, k=2)
    reverse_good: set[tuple[int, int]] = set()
    for pair in reverse:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance and first.distance <= max_distance:
            reverse_good.add((first.trainIdx, first.queryIdx))
    output: list[tuple[int, int, float]] = []
    for pair in forward:
        if len(pair) != 2:
            continue
        first, second = pair
        key = (first.queryIdx, first.trainIdx)
        if first.distance < ratio * second.distance and first.distance <= max_distance and (key in reverse_good):
            output.append((first.queryIdx, first.trainIdx, float(first.distance)))
    return output

def _grid_cells(points: np.ndarray, image_wh: tuple[int, int]) -> int:
    if len(points) == 0:
        return 0
    width, height = image_wh
    columns = np.clip((points[:, 0] / width * 4).astype(int), 0, 3)
    rows = np.clip((points[:, 1] / height * 3).astype(int), 0, 2)
    return len(set(zip(columns.tolist(), rows.tolist())))

def _image_span(points: np.ndarray, image_wh: tuple[int, int]) -> tuple[float, float]:
    if len(points) == 0:
        return (0.0, 0.0)
    width, height = image_wh
    span = np.ptp(points, axis=0)
    return (float(span[0] / width), float(span[1] / height))

def _sample_depth(depth: np.ndarray, xy_rgb: np.ndarray, rgb_wh: tuple[int, int], radius: int) -> np.ndarray:
    height, width = depth.shape
    rgb_width, rgb_height = rgb_wh
    us = np.rint(xy_rgb[:, 0] * width / rgb_width).astype(int)
    vs = np.rint(xy_rgb[:, 1] * height / rgb_height).astype(int)
    values = np.full(len(xy_rgb), np.nan, dtype=np.float64)
    for index, (u, v) in enumerate(zip(us, vs)):
        u0, u1 = (max(0, u - radius), min(width, u + radius + 1))
        v0, v1 = (max(0, v - radius), min(height, v + radius + 1))
        patch = depth[v0:v1, u0:u1]
        valid = patch[np.isfinite(patch) & (patch > 0.05)]
        if len(valid):
            values[index] = float(np.median(valid))
    return values

def _sample_confidence(path: Path | None, xy_rgb: np.ndarray, rgb_wh: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.full(len(xy_rgb), -1, dtype=np.int16)
    confidence = _imread(path, cv2.IMREAD_UNCHANGED)
    if confidence is None or confidence.ndim != 2:
        return np.full(len(xy_rgb), -1, dtype=np.int16)
    height, width = confidence.shape
    rgb_width, rgb_height = rgb_wh
    us = np.clip(np.rint(xy_rgb[:, 0] * width / rgb_width).astype(int), 0, width - 1)
    vs = np.clip(np.rint(xy_rgb[:, 1] * height / rgb_height).astype(int), 0, height - 1)
    return confidence[vs, us].astype(np.int16)

def _rgb_control(source_path: Path, target_path: Path, source_xy: np.ndarray, target_xy: np.ndarray, seed: int) -> float | None:
    if len(source_xy) < 4:
        return None
    try:
        source = read_sensor_image(source_path, cv2.IMREAD_COLOR)
        target = read_sensor_image(target_path, cv2.IMREAD_COLOR)
    except OSError:
        return None
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    def pixels(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
        x = np.clip(np.rint(xy[:, 0]).astype(int), 0, image.shape[1] - 1)
        y = np.clip(np.rint(xy[:, 1]).astype(int), 0, image.shape[0] - 1)
        return image[y, x]
    first = pixels(source_lab, source_xy)
    second = pixels(target_lab, target_xy)
    matched = np.linalg.norm(first - second, axis=1)
    rng = np.random.default_rng(seed)
    shuffled = np.linalg.norm(first - second[rng.permutation(len(second))], axis=1)
    denominator = float(np.median(shuffled))
    if denominator <= 1e-06:
        return None
    return float(np.median(matched) / denominator)

def estimate_gravity_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or (len(source) < 2):
        raise ValueError('source and target must be matching Nx3 arrays with N>=2')
    source_xz = source[:, [0, 2]]
    target_xz = target[:, [0, 2]]
    source_center = source_xz.mean(axis=0)
    target_center = target_xz.mean(axis=0)
    covariance = (source_xz - source_center).T @ (target_xz - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation_2d = vt.T @ u.T
    if np.linalg.det(rotation_2d) < 0.0:
        vt[-1] *= -1.0
        rotation_2d = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [[rotation_2d[0, 0], 0.0, rotation_2d[0, 1]], [0.0, 1.0, 0.0], [rotation_2d[1, 0], 0.0, rotation_2d[1, 1]]]
    transform[:3, 3] = np.median(target - source @ transform[:3, :3].T, axis=0)
    return project_yaw_translation(transform)

def _apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]

def gravity_ransac(source_points: np.ndarray, target_points: np.ndarray, threshold_m: float, iterations: int, seed: int) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    count = len(source)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or (count < 2):
        return (None, np.zeros(count, dtype=bool), np.full(count, np.inf))
    rng = np.random.default_rng(seed)
    best_transform: np.ndarray | None = None
    best_mask = np.zeros(count, dtype=bool)
    best_key = (-1, -float('inf'))
    for _ in range(iterations):
        indices = rng.choice(count, size=2, replace=False)
        if np.linalg.norm(source[indices[0], [0, 2]] - source[indices[1], [0, 2]]) < 0.25:
            continue
        if np.linalg.norm(target[indices[0], [0, 2]] - target[indices[1], [0, 2]]) < 0.25:
            continue
        candidate = estimate_gravity_transform(source[indices], target[indices])
        residuals = np.linalg.norm(_apply(candidate, source) - target, axis=1)
        mask = residuals <= threshold_m
        inliers = int(mask.sum())
        key = (inliers, -float(np.median(residuals[mask])) if inliers else -float('inf'))
        if key > best_key:
            best_key = key
            best_mask = mask
            best_transform = candidate
    if best_transform is None or best_mask.sum() < 2:
        return (None, best_mask, np.full(count, np.inf))
    for _ in range(4):
        best_transform = estimate_gravity_transform(source[best_mask], target[best_mask])
        residuals = np.linalg.norm(_apply(best_transform, source) - target, axis=1)
        updated = residuals <= threshold_m
        if np.array_equal(updated, best_mask):
            break
        if updated.sum() < 2:
            break
        best_mask = updated
    residuals = np.linalg.norm(_apply(best_transform, source) - target, axis=1)
    return (best_transform, best_mask, residuals)

def _backproject_matches(feature: ImageFeature, xy: np.ndarray, config: BridgeConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = _read_depth_unicode(feature.frame.depth_path)
    pose, intrinsics, rgb_wh = load_pose_intrinsics_rgb_wh(feature.frame.json_path, legacy_rgb_wh=feature.image_wh)
    if feature.image_wh != rgb_wh:
        raise ValueError(f'{feature.frame.rgb_path.name}: feature pixels are not in the declared sensor raster')
    values = _sample_depth(depth, xy, rgb_wh, config.depth_neighborhood_px)
    if feature.recovered_depths is not None and len(xy):
        distances, indices = cKDTree(feature.xy_full).query(xy, k=1)
        recovered = feature.recovered_depths[indices]
        use = (distances < 0.01) & np.isfinite(recovered)
        values[use] = recovered[use]
    valid = np.isfinite(values)
    depth_wh = (depth.shape[1], depth.shape[0])
    K_depth = scale_K(intrinsics, rgb_wh, depth_wh)
    u_depth = xy[:, 0] * depth_wh[0] / rgb_wh[0]
    v_depth = xy[:, 1] * depth_wh[1] / rgb_wh[1]
    world = np.full((len(xy), 3), np.nan, dtype=np.float64)
    if valid.any():
        run_points = backproject_world(u_depth[valid], v_depth[valid], values[valid], K_depth, pose, 'arkit-gl')
        world[valid] = _apply(feature.frame.T_side_from_run, run_points)
    confidence = _sample_confidence(feature.frame.conf_path, xy, rgb_wh)
    camera_center = _apply(feature.frame.T_side_from_run, pose[:3, 3][None, :])[0]
    return (world, valid, values, np.append(camera_center, 1.0)[:3], confidence)

def verify_frame_pair(source: ImageFeature, target: ImageFeature, retrieval_score: float, shared_words: int, config: BridgeConfig, descriptor_kind: str='ORB') -> PairEvidence:
    evidence = PairEvidence(source, target, retrieval_score, shared_words, descriptor_kind)
    matches = _mutual_ratio_matches(source, target, config)
    evidence.visual_matches = len(matches)
    if len(matches) < config.min_visual_matches:
        evidence.rejection_reasons.append('visual_matches_below_min')
        return evidence
    source_xy = np.asarray([source.xy_full[i] for i, _, _ in matches], dtype=np.float32)
    target_xy = np.asarray([target.xy_full[j] for _, j, _ in matches], dtype=np.float32)
    method = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)
    _, mask = cv2.findHomography(source_xy, target_xy, method, config.homography_threshold_px, maxIters=10000, confidence=0.999)
    if mask is None:
        evidence.rejection_reasons.append('homography_failed')
        return evidence
    inlier_mask = mask.reshape(-1).astype(bool)
    source_xy = source_xy[inlier_mask]
    target_xy = target_xy[inlier_mask]
    evidence.homography_inliers = len(source_xy)
    evidence.homography_ratio = len(source_xy) / len(matches)
    evidence.source_grid_cells = _grid_cells(source_xy, source.image_wh)
    evidence.target_grid_cells = _grid_cells(target_xy, target.image_wh)
    evidence.source_image_span = _image_span(source_xy, source.image_wh)
    evidence.target_image_span = _image_span(target_xy, target.image_wh)
    if evidence.homography_inliers < config.min_homography_inliers:
        evidence.rejection_reasons.append('homography_inliers_below_min')
    if evidence.homography_ratio < config.min_homography_ratio:
        evidence.rejection_reasons.append('homography_ratio_below_min')
    if min(evidence.source_grid_cells, evidence.target_grid_cells) < config.min_grid_cells:
        evidence.rejection_reasons.append('image_inliers_not_distributed')
    if min(max(evidence.source_image_span), max(evidence.target_image_span)) < config.min_image_span_fraction:
        evidence.rejection_reasons.append('image_inlier_span_below_min')
    if evidence.rejection_reasons:
        return evidence
    evidence.matched_source_xy = source_xy
    evidence.matched_target_xy = target_xy
    evidence.rgb_ratio = _rgb_control(source.frame.rgb_path, target.frame.rgb_path, source_xy, target_xy, config.random_seed + source.frame.frame_index + target.frame.frame_index)
    source_world, source_valid, source_depth, source_camera, source_conf = _backproject_matches(source, source_xy, config)
    target_world, target_valid, target_depth, target_camera, target_conf = _backproject_matches(target, target_xy, config)
    valid = source_valid & target_valid
    evidence.depth_correspondences = int(valid.sum())
    evidence.source_camera_center = source_camera
    evidence.target_camera_center = target_camera
    if valid.any():
        used_depth = np.concatenate([source_depth[valid], target_depth[valid]])
        evidence.depth_range_min_m = float(np.min(used_depth))
        evidence.depth_range_max_m = float(np.max(used_depth))
        conf_values = np.concatenate([source_conf[valid], target_conf[valid]])
        evidence.depth_conf_histogram = {str(value): int(np.sum(conf_values == value)) for value in (-1, 0, 1, 2)}
    if evidence.depth_correspondences < config.min_pair_3d_inliers:
        evidence.rejection_reasons.append('depth_correspondences_below_min')
        return evidence
    transform, ransac_mask, residuals = gravity_ransac(source_world[valid], target_world[valid], config.ransac_threshold_m, config.ransac_iterations, config.random_seed + 31 * source.frame.frame_index + target.frame.frame_index)
    if transform is None:
        evidence.rejection_reasons.append('gravity_ransac_failed')
        return evidence
    evidence.transform_target_from_source = transform
    evidence.ransac_inliers = int(ransac_mask.sum())
    evidence.ransac_ratio = evidence.ransac_inliers / evidence.depth_correspondences
    if evidence.ransac_inliers:
        evidence.residual_median_m = float(np.median(residuals[ransac_mask]))
        evidence.residual_p90_m = float(np.percentile(residuals[ransac_mask], 90))
        evidence.inlier_source_points = source_world[valid][ransac_mask]
        evidence.inlier_target_points = target_world[valid][ransac_mask]
    if evidence.ransac_inliers < config.min_pair_3d_inliers:
        evidence.rejection_reasons.append('pair_3d_inliers_below_min')
    if evidence.ransac_ratio < config.min_pair_3d_ratio:
        evidence.rejection_reasons.append('pair_3d_ratio_below_min')
    return evidence

def _angle_delta_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = yaw_from_transform(first) - yaw_from_transform(second)
    return abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta))))

def _transform_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    return (_angle_delta_deg(first, second), float(np.linalg.norm(first[:3, 3] - second[:3, 3])))

def _is_valid_gravity_transform(transform: np.ndarray) -> bool:
    return bool(transform.shape == (4, 4) and np.isfinite(transform).all() and np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-08) and is_gravity_locked(transform))

def seed_conflict_diagnostics(visual_transform: np.ndarray, context_seed: np.ndarray, config: BridgeConfig) -> dict[str, Any]:
    yaw_delta, translation_delta = _transform_distance(visual_transform, context_seed)
    conflicts: list[str] = []
    if yaw_delta > config.max_seed_yaw_delta_deg:
        conflicts.append('context_seed_yaw_conflict')
    if translation_delta > config.max_seed_translation_delta_m:
        conflicts.append('context_seed_translation_conflict')
    return {'yaw_delta_to_visual_deg': yaw_delta, 'translation_delta_to_visual_m': translation_delta, 'diagnostic_conflicts': conflicts, 'affects_acceptance': False}

def cluster_pair_models(pairs: Sequence[PairEvidence], config: BridgeConfig) -> list[PairEvidence]:
    candidates = [item for item in pairs if item.has_3d_model]
    best: list[PairEvidence] = []
    best_score = -1.0
    for center in candidates:
        assert center.transform_target_from_source is not None
        cluster = []
        for item in candidates:
            assert item.transform_target_from_source is not None
            yaw_delta, translation_delta = _transform_distance(center.transform_target_from_source, item.transform_target_from_source)
            if yaw_delta <= config.cluster_yaw_deg and translation_delta <= config.cluster_translation_m:
                cluster.append(item)
        cluster.sort(key=lambda item: (-item.ransac_inliers, -item.homography_inliers, -item.retrieval_score, item.source_feature.frame.run_id, item.source_feature.frame.frame_index, item.target_feature.frame.run_id, item.target_feature.frame.frame_index))
        independent: list[PairEvidence] = []
        used_source: set[tuple[str, int]] = set()
        used_target: set[tuple[str, int]] = set()
        for item in cluster:
            source_key = (item.source_feature.frame.run_id, item.source_feature.frame.frame_index)
            target_key = (item.target_feature.frame.run_id, item.target_feature.frame.frame_index)
            if source_key in used_source or target_key in used_target:
                continue
            independent.append(item)
            used_source.add(source_key)
            used_target.add(target_key)
        score = float(sum((item.ransac_inliers for item in independent)))
        if score > best_score:
            best, best_score = (independent, score)
    return best

def _frame_pair_id(item: PairEvidence) -> tuple[str, int, str, int]:
    return (item.source_feature.frame.run_id, item.source_feature.frame.frame_index, item.target_feature.frame.run_id, item.target_feature.frame.frame_index)

def _frame_pair_index_key(item: PairEvidence) -> tuple[int, int, str, str]:
    return (item.source_feature.frame.frame_index, item.target_feature.frame.frame_index, item.source_feature.frame.run_id, item.target_feature.frame.run_id)

def _pair_model_sort_key(item: PairEvidence) -> tuple[Any, ...]:
    transform = item.transform_target_from_source
    transform_key = tuple((float(value) for value in np.asarray(transform).ravel())) if transform is not None else ()
    source = item.source_feature.frame
    target = item.target_feature.frame
    return (-item.ransac_inliers, -item.homography_inliers, -item.retrieval_score, source.frame_index, target.frame_index, source.run_id, target.run_id, transform_key)

def _normalized_model_distance(first: PairEvidence, second: PairEvidence, config: BridgeConfig) -> float:
    assert first.transform_target_from_source is not None
    assert second.transform_target_from_source is not None
    yaw_delta, translation_delta = _transform_distance(first.transform_target_from_source, second.transform_target_from_source)
    yaw_scale = config.cluster_yaw_deg
    translation_scale = config.cluster_translation_m
    yaw_score = yaw_delta / yaw_scale if yaw_scale > 0.0 else 0.0 if yaw_delta == 0.0 else math.inf
    translation_score = translation_delta / translation_scale if translation_scale > 0.0 else 0.0 if translation_delta == 0.0 else math.inf
    return yaw_score + translation_score

def _cluster_representative(cluster: Sequence[PairEvidence], config: BridgeConfig) -> PairEvidence:
    return min(cluster, key=lambda candidate: (sum((_normalized_model_distance(candidate, other, config) for other in cluster)), _frame_pair_index_key(candidate)))

def cluster_pair_model_hypotheses(pairs: Sequence[PairEvidence], config: BridgeConfig, *, limit: int | None=None) -> tuple[tuple[PairEvidence, ...], ...]:
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError('hypothesis limit must be a non-negative integer or None')
    candidates = sorted((item for item in pairs if item.has_3d_model), key=_pair_model_sort_key)
    raw_clusters: list[list[PairEvidence]] = []
    for item in candidates:
        assert item.transform_target_from_source is not None
        for cluster in raw_clusters:
            compatible = True
            for member in cluster:
                assert member.transform_target_from_source is not None
                yaw_delta, translation_delta = _transform_distance(item.transform_target_from_source, member.transform_target_from_source)
                if yaw_delta > config.cluster_yaw_deg or translation_delta > config.cluster_translation_m:
                    compatible = False
                    break
            if compatible:
                cluster.append(item)
                break
        else:
            raw_clusters.append([item])
    retained_clusters: list[tuple[PairEvidence, ...]] = []
    retained_keys: list[tuple[frozenset[tuple[str, int, str, int]], np.ndarray]] = []
    for raw_cluster in raw_clusters:
        retained: list[PairEvidence] = []
        used_source: set[tuple[str, int]] = set()
        used_target: set[tuple[str, int]] = set()
        for item in raw_cluster:
            source_key = (item.source_feature.frame.run_id, item.source_feature.frame.frame_index)
            target_key = (item.target_feature.frame.run_id, item.target_feature.frame.frame_index)
            if source_key in used_source or target_key in used_target:
                continue
            retained.append(item)
            used_source.add(source_key)
            used_target.add(target_key)
        if not retained:
            continue
        representative = _cluster_representative(retained, config)
        frame_ids = frozenset((_frame_pair_id(item) for item in retained))
        assert representative.transform_target_from_source is not None
        if any((frame_ids == existing_ids and np.allclose(representative.transform_target_from_source, existing_transform, rtol=0.0, atol=1e-06) for existing_ids, existing_transform in retained_keys)):
            continue
        retained_clusters.append(tuple(retained))
        retained_keys.append((frame_ids, representative.transform_target_from_source))
    output = tuple(retained_clusters)
    return output if limit is None else output[:limit]

def global_correspondence_consensus(pairs: Sequence[PairEvidence], config: BridgeConfig) -> list[PairEvidence]:
    candidates = [item for item in pairs if item.has_3d_model]
    if len(candidates) < 2:
        return candidates
    source = np.vstack([item.inlier_source_points for item in candidates if item.inlier_source_points is not None])
    target = np.vstack([item.inlier_target_points for item in candidates if item.inlier_target_points is not None])
    transform, _, _ = gravity_ransac(source, target, config.ransac_threshold_m, config.ransac_iterations * 3, config.random_seed + 7000)
    if transform is None:
        return []
    scored: list[tuple[int, float, PairEvidence]] = []
    for item in candidates:
        assert item.inlier_source_points is not None and item.inlier_target_points is not None
        residuals = np.linalg.norm(_apply(transform, item.inlier_source_points) - item.inlier_target_points, axis=1)
        count = int(np.sum(residuals <= config.ransac_threshold_m))
        ratio = count / len(residuals) if len(residuals) else 0.0
        if count >= config.min_pair_3d_inliers and ratio >= config.min_pair_3d_ratio:
            scored.append((count, ratio, item))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2].source_feature.frame.run_id, row[2].source_feature.frame.frame_index, row[2].target_feature.frame.run_id, row[2].target_feature.frame.frame_index))
    independent: list[PairEvidence] = []
    used_source: set[tuple[str, int]] = set()
    used_target: set[tuple[str, int]] = set()
    for _, _, item in scored:
        source_key = (item.source_feature.frame.run_id, item.source_feature.frame.frame_index)
        target_key = (item.target_feature.frame.run_id, item.target_feature.frame.frame_index)
        if source_key in used_source or target_key in used_target:
            continue
        independent.append(item)
        used_source.add(source_key)
        used_target.add(target_key)
    return independent

def _point_span(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.ptp(points, axis=0)))

def _camera_span(pairs: Sequence[PairEvidence], source: bool) -> float:
    centers = [item.source_camera_center if source else item.target_camera_center for item in pairs]
    valid = np.asarray([item for item in centers if item is not None], dtype=np.float64)
    return _point_span(valid) if len(valid) else 0.0

def _aggregate_model(pairs: Sequence[PairEvidence], config: BridgeConfig, seed_offset: int=0) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not pairs:
        empty = np.empty((0, 3), dtype=np.float64)
        return (None, np.empty(0, dtype=bool), np.empty(0), empty, empty)
    source = np.vstack([item.inlier_source_points for item in pairs if item.inlier_source_points is not None])
    target = np.vstack([item.inlier_target_points for item in pairs if item.inlier_target_points is not None])
    transform, mask, residuals = gravity_ransac(source, target, config.ransac_threshold_m, config.ransac_iterations * 2, config.random_seed + seed_offset)
    return (transform, mask, residuals, source, target)

def _leave_one_out(pairs: Sequence[PairEvidence], config: BridgeConfig) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if len(pairs) < 3:
        return output
    for held_index, held in enumerate(pairs):
        train = [item for index, item in enumerate(pairs) if index != held_index]
        transform, _, _, _, _ = _aggregate_model(train, config, 1000 + held_index)
        if transform is None or held.inlier_source_points is None or held.inlier_target_points is None:
            continue
        residuals = np.linalg.norm(_apply(transform, held.inlier_source_points) - held.inlier_target_points, axis=1)
        median = float(np.median(residuals)) if len(residuals) else float('inf')
        output.append({'held_source': _frame_label(held.source_feature.frame), 'held_target': _frame_label(held.target_feature.frame), 'correspondences': len(residuals), 'median_m': median, 'p90_m': float(np.percentile(residuals, 90)) if len(residuals) else None, 'passed': bool(median <= config.max_loo_median_m)})
    return output

def _load_side_cloud(side: SideSpec) -> o3d.geometry.PointCloud:
    combined = o3d.geometry.PointCloud()
    for run in side.runs:
        if run.workset_path is None:
            continue
        cloud = o3d.io.read_point_cloud(str(run.workset_path), remove_nan_points=True, remove_infinite_points=True)
        if len(cloud.points) == 0:
            continue
        cloud.transform(run.T_side_from_run)
        combined += cloud
    return combined

def _crop_cloud(cloud: o3d.geometry.PointCloud, evidence_points: np.ndarray, margin_m: float) -> o3d.geometry.PointCloud:
    if len(cloud.points) == 0 or len(evidence_points) == 0:
        return o3d.geometry.PointCloud()
    lower = np.min(evidence_points, axis=0) - margin_m
    upper = np.max(evidence_points, axis=0) + margin_m
    return cloud.crop(o3d.geometry.AxisAlignedBoundingBox(lower, upper))

def gravity_locked_local_icp(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, initial_transform: np.ndarray, scales: Sequence[ICPScale], trim_fraction: float, min_correspondences: int) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    if not 0.0 < trim_fraction <= 1.0:
        raise ValueError('local ICP trim fraction must be in (0, 1]')
    if min_correspondences < 3:
        raise ValueError('local ICP needs at least three correspondences')
    initial = np.asarray(initial_transform, dtype=np.float64)
    if not _is_valid_gravity_transform(initial):
        raise ValueError('local ICP initial transform must already be gravity locked')
    current = initial.copy()
    iteration_counts: list[int] = []
    final_correspondences: list[int] = []
    for scale in scales:
        source_points = np.asarray(source.voxel_down_sample(scale.voxel_m).points)
        target_points = np.asarray(target.voxel_down_sample(scale.voxel_m).points)
        if len(source_points) < min_correspondences or len(target_points) < min_correspondences:
            iteration_counts.append(0)
            final_correspondences.append(0)
            continue
        target_tree = cKDTree(target_points)
        used = 0
        accepted_count = 0
        for _ in range(scale.iterations):
            transformed = _apply(current, source_points)
            distances, target_indices = target_tree.query(transformed, k=1)
            source_indices = np.flatnonzero(distances <= scale.max_correspondence_m)
            if len(source_indices) < min_correspondences:
                break
            matched_targets = target_points[target_indices[source_indices]]
            reverse_indices = cKDTree(transformed).query(matched_targets, k=1)[1]
            source_indices = source_indices[reverse_indices == source_indices]
            if len(source_indices) < min_correspondences:
                break
            kept_distances = distances[source_indices]
            if trim_fraction < 1.0:
                trim_limit = float(np.quantile(kept_distances, trim_fraction))
                source_indices = source_indices[kept_distances <= trim_limit]
            if len(source_indices) < min_correspondences:
                break
            matched_indices = target_indices[source_indices]
            delta = estimate_gravity_transform(transformed[source_indices], target_points[matched_indices])
            candidate = project_yaw_translation(delta @ current)
            translation_delta = float(np.linalg.norm(candidate[:3, 3] - current[:3, 3]))
            yaw_delta = math.radians(_angle_delta_deg(candidate, current))
            current = candidate
            used += 1
            accepted_count = len(source_indices)
            if translation_delta < 1e-05 and yaw_delta < 1e-05:
                break
        iteration_counts.append(used)
        final_correspondences.append(accepted_count)
    return (current, tuple(iteration_counts), tuple(final_correspondences))

def _evaluate_direction(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transform: np.ndarray, threshold_m: float) -> dict[str, Any]:
    result = o3d.pipelines.registration.evaluate_registration(source, target, threshold_m, transform)
    return {'fitness': float(result.fitness), 'rmse_m': float(result.inlier_rmse), 'correspondences': len(result.correspondence_set)}

def _cloud_rgb_ratio(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, transform: np.ndarray, max_distance_m: float, seed: int) -> float | None:
    if not source.has_colors() or not target.has_colors():
        return None
    source_points = _apply(transform, np.asarray(source.points))
    target_points = np.asarray(target.points)
    if not len(source_points) or not len(target_points):
        return None
    distances, indices = cKDTree(target_points).query(source_points, k=1)
    valid = distances <= max_distance_m
    if valid.sum() < 10:
        return None
    source_color = np.asarray(source.colors)[valid]
    target_color = np.asarray(target.colors)[indices[valid]]
    matched = np.linalg.norm(source_color - target_color, axis=1)
    rng = np.random.default_rng(seed)
    shuffled = np.linalg.norm(source_color - target_color[rng.permutation(len(target_color))], axis=1)
    denominator = float(np.median(shuffled))
    return float(np.median(matched) / denominator) if denominator > 1e-09 else None

def local_icp_evidence(spec: PairSpec, visual_transform: np.ndarray, source_evidence_points: np.ndarray, target_evidence_points: np.ndarray, config: BridgeConfig) -> dict[str, Any]:
    source_full = _load_side_cloud(spec.source)
    target_full = _load_side_cloud(spec.target)
    source = _crop_cloud(source_full, source_evidence_points, config.crop_margin_m)
    target = _crop_cloud(target_full, target_evidence_points, config.crop_margin_m)
    if len(source.points) < 50 or len(target.points) < 50:
        return {'available': False, 'source_points': len(source.points), 'target_points': len(target.points), 'reason': 'local_workset_points_below_min'}
    scales = (ICPScale(0.2, 0.4, 12), ICPScale(0.1, 0.25, 12), ICPScale(0.05, 0.15, 10))
    refined, iterations, correspondence_counts = gravity_locked_local_icp(source, target, visual_transform, scales, config.local_icp_trim_fraction, config.local_icp_min_correspondences)
    yaw_correction, translation_correction = _transform_distance(refined, visual_transform)
    finest_source = source.voxel_down_sample(0.1)
    finest_target = target.voxel_down_sample(0.1)

    def evaluate(transform: np.ndarray) -> dict[str, Any]:
        thresholds: dict[str, Any] = {}
        for distance in (0.1, 0.25):
            forward = _evaluate_direction(finest_source, finest_target, transform, distance)
            reverse = _evaluate_direction(finest_target, finest_source, np.linalg.inv(transform), distance)
            thresholds[f'{distance:.2f}m'] = {'source_to_target': forward, 'target_to_source': reverse, 'symmetric_fitness': min(forward['fitness'], reverse['fitness'])}
        return thresholds
    initial_thresholds = evaluate(visual_transform)
    refined_thresholds = evaluate(refined)
    initial_fitness = initial_thresholds['0.25m']['symmetric_fitness']
    refined_fitness = refined_thresholds['0.25m']['symmetric_fitness']
    fitness_ratio = refined_fitness / initial_fitness if initial_fitness > 1e-12 else None
    refinement_non_regressing = bool(initial_fitness <= 1e-12 or refined_fitness >= initial_fitness * config.min_icp_refined_to_initial_fitness_ratio)
    initial_rgb_ratio = _cloud_rgb_ratio(finest_source, finest_target, visual_transform, 0.25, config.random_seed)
    refined_rgb_ratio = _cloud_rgb_ratio(finest_source, finest_target, refined, 0.25, config.random_seed)
    return {'available': True, 'source_points': len(source.points), 'target_points': len(target.points), 'iterations': list(iterations), 'reciprocal_trimmed_correspondences': list(correspondence_counts), 'solver': 'gravity_locked_reciprocal_trimmed_point_to_point', 'T_initial_target_from_source': visual_transform.tolist(), 'T_target_from_source': refined.tolist(), 'gravity_locked': is_gravity_locked(refined), 'yaw_correction_deg': yaw_correction, 'translation_correction_m': translation_correction, 'initial_thresholds': initial_thresholds, 'thresholds': refined_thresholds, 'refined_to_initial_symmetric_fitness_ratio_025m': fitness_ratio, 'refinement_non_regressing': refinement_non_regressing, 'initial_rgb_matched_to_shuffled_ratio': initial_rgb_ratio, 'rgb_matched_to_shuffled_ratio': refined_rgb_ratio}

def _frame_label(frame: FrameSpec) -> str:
    return f'{frame.run_id}/frame_{frame.frame_index:05d}'

def _pair_json(item: PairEvidence) -> dict[str, Any]:
    proposal = 'visual_ORB_words'
    if item.retrieval_shared_words == -1:
        proposal = 'trajectory_seed_neighbor'
    elif item.retrieval_shared_words == -2:
        proposal = 'temporal_neighborhood_followup'
    output: dict[str, Any] = {'source': _frame_label(item.source_feature.frame), 'target': _frame_label(item.target_feature.frame), 'retrieval_score': item.retrieval_score, 'retrieval_shared_words': item.retrieval_shared_words, 'proposal': proposal, 'descriptor_kind': item.descriptor_kind, 'visual_matches': item.visual_matches, 'homography_inliers': item.homography_inliers, 'homography_ratio': item.homography_ratio, 'source_grid_cells': item.source_grid_cells, 'target_grid_cells': item.target_grid_cells, 'source_image_span': list(item.source_image_span), 'target_image_span': list(item.target_image_span), 'rgb_matched_to_shuffled_ratio': item.rgb_ratio, 'depth_correspondences': item.depth_correspondences, 'depth_conf_histogram': item.depth_conf_histogram, 'depth_range_min_m': item.depth_range_min_m, 'depth_range_max_m': item.depth_range_max_m, 'ransac_inliers': item.ransac_inliers, 'ransac_ratio': item.ransac_ratio, 'residual_median_m': item.residual_median_m, 'residual_p90_m': item.residual_p90_m, 'rejection_reasons': item.rejection_reasons}
    if item.transform_target_from_source is not None:
        output['T_target_from_source'] = item.transform_target_from_source.tolist()
        output['yaw_deg'] = math.degrees(yaw_from_transform(item.transform_target_from_source)) % 360.0
    return output

def _draw_debug_pair(item: PairEvidence, path: Path) -> None:
    if item.matched_source_xy is None or item.matched_target_xy is None:
        return
    try:
        source = read_sensor_image(item.source_feature.frame.rgb_path, cv2.IMREAD_COLOR)
        target = read_sensor_image(item.target_feature.frame.rgb_path, cv2.IMREAD_COLOR)
    except OSError:
        return
    display_height = 540

    def resize(image: np.ndarray) -> tuple[np.ndarray, float]:
        scale = display_height / image.shape[0]
        return (cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale)
    left, left_scale = resize(source)
    right, right_scale = resize(target)
    canvas = np.hstack([left, right])
    limit = min(80, len(item.matched_source_xy))
    for index in np.linspace(0, len(item.matched_source_xy) - 1, limit, dtype=int):
        first = tuple(np.rint(item.matched_source_xy[index] * left_scale).astype(int))
        second_raw = np.rint(item.matched_target_xy[index] * right_scale).astype(int)
        second = (int(second_raw[0] + left.shape[1]), int(second_raw[1]))
        color = tuple((int(value) for value in (37 * index % 255, (97 * index + 80) % 255, (53 * index + 160) % 255)))
        cv2.circle(canvas, first, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, second, 3, color, -1, cv2.LINE_AA)
        cv2.line(canvas, first, second, color, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _imwrite(path, canvas):
        raise OSError(f'debug image cannot be written: {path}')

@dataclass(frozen=True)
class _BridgeSearchEvidence:
    visual_candidates: tuple[tuple[int, int, float, int], ...]
    trajectory_candidates: tuple[tuple[int, int, float, int], ...]
    candidates: tuple[tuple[int, int, float, int], ...]
    followup_candidates: tuple[tuple[int, int, float, int], ...]
    verified: tuple[PairEvidence, ...]
    sift_fallbacks: int

def _collect_bridge_evidence(spec: PairSpec, source_features: Sequence[ImageFeature], target_features: Sequence[ImageFeature], config: BridgeConfig, shared_sift_cache: dict[Path, ImageFeature | None] | None, target_retrieval_index: RetrievalIndex | None) -> _BridgeSearchEvidence:
    visual_candidates = retrieve_candidates(source_features, target_features, config, target_retrieval_index)
    trajectory_candidates = trajectory_seed_candidates(source_features, target_features, spec.seed_target_from_source, config)
    candidates = _merge_candidate_proposals(visual_candidates, trajectory_candidates, config.max_candidates)
    verified: list[PairEvidence] = []
    sift_cache = shared_sift_cache if shared_sift_cache is not None else {}
    sift_fallbacks = 0

    def sift(item: ImageFeature) -> ImageFeature | None:
        path = item.frame.rgb_path
        if path not in sift_cache:
            sift_cache[path] = _extract_sift_feature(item, config)
        return sift_cache[path]

    def verify_candidate(source_index: int, target_index: int, score: float, words: int) -> PairEvidence:
        nonlocal sift_fallbacks
        item = verify_frame_pair(source_features[source_index], target_features[target_index], score, words, config)
        if 'visual_matches_below_min' in item.rejection_reasons and sift_fallbacks < config.max_sift_fallbacks_per_pair:
            sift_fallbacks += 1
            sift_source = sift(source_features[source_index])
            sift_target = sift(target_features[target_index])
            if sift_source is not None and sift_target is not None:
                sift_item = verify_frame_pair(sift_source, sift_target, score, words, config, 'SIFT')
                if (sift_item.ransac_inliers, sift_item.homography_inliers, sift_item.visual_matches) > (item.ransac_inliers, item.homography_inliers, item.visual_matches):
                    item = sift_item
        return item
    for index, (source_index, target_index, score, words) in enumerate(candidates, 1):
        item = verify_candidate(source_index, target_index, score, words)
        verified.append(item)
        if index % 40 == 0 or index == len(candidates):
            print(f'visual_bridge: {spec.pair_id} verified {index}/{len(candidates)}', file=sys.stderr, flush=True)
    initial_keys = {(item[0], item[1]) for item in candidates}
    followup_ranked: list[tuple[int, int, int]] = []
    radius = config.temporal_followup_radius
    for candidate, item in zip(candidates, verified):
        if not item.has_3d_model:
            continue
        source_center, target_center = (candidate[0], candidate[1])
        source_run = source_features[source_center].frame.run_id
        target_run = target_features[target_center].frame.run_id
        for source_delta in range(-radius, radius + 1):
            for target_delta in range(-radius, radius + 1):
                source_index = source_center + source_delta
                target_index = target_center + target_delta
                if not (0 <= source_index < len(source_features) and 0 <= target_index < len(target_features)):
                    continue
                if source_features[source_index].frame.run_id != source_run:
                    continue
                if target_features[target_index].frame.run_id != target_run:
                    continue
                if (source_index, target_index) in initial_keys:
                    continue
                followup_ranked.append((abs(source_delta) + abs(target_delta), source_index, target_index))
    followup_ranked.sort()
    followup_candidates: list[tuple[int, int, float, int]] = []
    followup_seen: set[tuple[int, int]] = set()
    for _, source_index, target_index in followup_ranked:
        key = (source_index, target_index)
        if key in followup_seen:
            continue
        followup_seen.add(key)
        followup_candidates.append((source_index, target_index, 0.0, -2))
        if len(followup_candidates) >= config.max_temporal_followups:
            break
    for index, (source_index, target_index, score, words) in enumerate(followup_candidates, 1):
        verified.append(verify_candidate(source_index, target_index, score, words))
        if index % 40 == 0 or index == len(followup_candidates):
            print(f'visual_bridge: {spec.pair_id} temporal followup {index}/{len(followup_candidates)}', file=sys.stderr, flush=True)
    return _BridgeSearchEvidence(visual_candidates=tuple(visual_candidates), trajectory_candidates=tuple(trajectory_candidates), candidates=tuple(candidates), followup_candidates=tuple(followup_candidates), verified=tuple(verified), sift_fallbacks=sift_fallbacks)

def _select_legacy_cluster(verified: Sequence[PairEvidence], config: BridgeConfig) -> tuple[tuple[PairEvidence, ...], str]:
    transform_cluster = cluster_pair_models(verified, config)
    correspondence_cluster = global_correspondence_consensus(verified, config)

    def cluster_quality(items: Sequence[PairEvidence]) -> tuple[int, int]:
        return (len(items), sum((item.ransac_inliers for item in items)))
    if cluster_quality(correspondence_cluster) > cluster_quality(transform_cluster):
        cluster = correspondence_cluster
        cluster_method = 'global_3d_correspondence_consensus'
    else:
        cluster = transform_cluster
        cluster_method = 'per_frame_transform_cluster'
    return (tuple(cluster), cluster_method)

def _evaluate_bridge_cluster(spec: PairSpec, source_features: Sequence[ImageFeature], target_features: Sequence[ImageFeature], config: BridgeConfig, search: _BridgeSearchEvidence, cluster: Sequence[PairEvidence], cluster_method: str, *, reject_invalid_icp_transform: bool) -> dict[str, Any]:
    verified = search.verified
    transform, aggregate_mask, aggregate_residuals, aggregate_source, aggregate_target = _aggregate_model(cluster, config)
    loo = _leave_one_out(cluster, config)
    reasons: list[str] = []
    if len(cluster) < config.min_independent_pairs:
        reasons.append('independent_frame_pairs_below_min')
    source_camera_span = _camera_span(cluster, True)
    target_camera_span = _camera_span(cluster, False)
    if source_camera_span < config.min_camera_span_m:
        reasons.append('source_camera_span_below_min')
    if target_camera_span < config.min_camera_span_m:
        reasons.append('target_camera_span_below_min')
    aggregate_inliers = int(aggregate_mask.sum())
    aggregate_ratio = aggregate_inliers / len(aggregate_mask) if len(aggregate_mask) else 0.0
    if aggregate_inliers < config.min_aggregate_3d_inliers:
        reasons.append('aggregate_3d_inliers_below_min')
    if aggregate_ratio < config.min_aggregate_3d_ratio:
        reasons.append('aggregate_3d_ratio_below_min')
    median = float(np.median(aggregate_residuals[aggregate_mask])) if aggregate_inliers else None
    p90 = float(np.percentile(aggregate_residuals[aggregate_mask], 90)) if aggregate_inliers else None
    if median is None or median > config.max_aggregate_median_m:
        reasons.append('aggregate_median_residual_above_max')
    if p90 is None or p90 > config.max_aggregate_p90_m:
        reasons.append('aggregate_p90_residual_above_max')
    inlier_span = _point_span(aggregate_target[aggregate_mask]) if aggregate_inliers else 0.0
    if inlier_span < config.min_inlier_span_m:
        reasons.append('aggregate_3d_span_below_min')
    loo_passes = sum((bool(item['passed']) for item in loo))
    if loo_passes < config.min_loo_passes:
        reasons.append('leave_one_frame_out_passes_below_min')
    seed_diagnostics: dict[str, Any] = {'yaw_delta_to_visual_deg': None, 'translation_delta_to_visual_m': None, 'diagnostic_conflicts': [], 'affects_acceptance': False}
    icp: dict[str, Any] = {'available': False, 'reason': 'no_visual_depth_model'}
    final_transform = transform
    if transform is not None:
        seed_diagnostics = seed_conflict_diagnostics(transform, spec.seed_target_from_source, config)
        if aggregate_inliers:
            icp = local_icp_evidence(spec, transform, aggregate_source[aggregate_mask], aggregate_target[aggregate_mask], config)
            if not icp.get('available'):
                reasons.append('local_bidirectional_icp_unavailable')
            else:
                if icp.get('refinement_non_regressing') is not True:
                    reasons.append('icp_refinement_regressed')
                if icp['yaw_correction_deg'] > config.max_icp_yaw_correction_deg:
                    reasons.append('icp_yaw_correction_above_max')
                if icp['translation_correction_m'] > config.max_icp_translation_correction_m:
                    reasons.append('icp_translation_correction_above_max')
                threshold_025 = icp['thresholds']['0.25m']
                if threshold_025['symmetric_fitness'] < config.min_local_symmetric_fitness_025m:
                    reasons.append('local_symmetric_fitness_below_min')
                correspondence_counts = [threshold_025.get(direction, {}).get('correspondences', 0) for direction in ('source_to_target', 'target_to_source')]
                if min(correspondence_counts) < config.local_icp_min_correspondences:
                    reasons.append('local_icp_correspondences_below_min')
                rgb_ratio = icp.get('rgb_matched_to_shuffled_ratio')
                if rgb_ratio is None:
                    reasons.append('local_rgb_control_unavailable')
                elif rgb_ratio > config.max_rgb_ratio:
                    reasons.append('local_rgb_control_failed')
                if reject_invalid_icp_transform:
                    try:
                        candidate_transform = np.asarray(icp['T_target_from_source'], dtype=np.float64)
                    except (KeyError, TypeError, ValueError):
                        candidate_transform = np.empty(0, dtype=np.float64)
                    if not _is_valid_gravity_transform(candidate_transform):
                        reasons.append('icp_result_not_gravity_locked')
                        final_transform = None
                    else:
                        final_transform = candidate_transform
                else:
                    final_transform = _matrix(icp['T_target_from_source'], 'ICP result')
    accepted = transform is not None and (not reasons)
    status = 'ACCEPTED_VISUAL_DEPTH_BRIDGE' if accepted else 'REJECTED_INSUFFICIENT_EVIDENCE'
    selected_ids = {id(item) for item in cluster}
    ranked_evidence = sorted(verified, key=lambda item: (id(item) not in selected_ids, -item.ransac_inliers, -item.homography_inliers, -item.retrieval_score))
    result: dict[str, Any] = {'id': spec.pair_id, 'source_endpoint': spec.source_endpoint, 'target_endpoint': spec.target_endpoint, 'source_side': spec.source.side_id, 'target_side': spec.target.side_id, 'source_runs': [item.run_id for item in spec.source.runs], 'target_runs': [item.run_id for item in spec.target.runs], 'source_rgb_frames': len(source_features), 'target_rgb_frames': len(target_features), 'retrieval_candidates': len(search.candidates), 'temporal_followup_candidates': len(search.followup_candidates), 'evaluated_frame_pairs': len(verified), 'visual_proposals_available': len(search.visual_candidates), 'trajectory_proposals_available': len(search.trajectory_candidates), 'sift_fallbacks_evaluated': search.sift_fallbacks, 'frame_pairs_with_3d_model': sum((item.has_3d_model for item in verified)), 'independent_cluster_pairs': len(cluster), 'cluster_method': cluster_method, 'source_camera_span_m': source_camera_span, 'target_camera_span_m': target_camera_span, 'aggregate_depth_correspondences': len(aggregate_mask), 'aggregate_3d_inliers': aggregate_inliers, 'aggregate_3d_ratio': aggregate_ratio, 'aggregate_residual_median_m': median, 'aggregate_residual_p90_m': p90, 'aggregate_inlier_span_m': inlier_span, 'leave_one_frame_out': loo, 'leave_one_frame_out_passes': loo_passes, 'seed': {'context_only': True, 'method': spec.seed_method, 'T_target_side_from_source_side': spec.seed_target_from_source.tolist(), **seed_diagnostics}, 'local_icp': icp, 'accepted': accepted, 'status': status, 'rejection_reasons': sorted(set(reasons)), 'selected_frame_pairs': [_pair_json(item) for item in cluster], 'top_checked_frame_pairs': [_pair_json(item) for item in ranked_evidence[:40]]}
    if final_transform is not None:
        result['T_target_side_from_source_side'] = final_transform.tolist()
        result['yaw_deg'] = math.degrees(yaw_from_transform(final_transform)) % 360.0
        result['gravity_locked'] = is_gravity_locked(final_transform)
    return result

def _evaluate_bridge_from_search(spec: PairSpec, source_features: Sequence[ImageFeature], target_features: Sequence[ImageFeature], config: BridgeConfig, search: _BridgeSearchEvidence, debug_dir: Path | None) -> dict[str, Any]:
    cluster, cluster_method = _select_legacy_cluster(search.verified, config)
    result = _evaluate_bridge_cluster(spec, source_features, target_features, config, search, cluster, cluster_method, reject_invalid_icp_transform=False)
    if debug_dir is not None:
        for index, item in enumerate(cluster[:8], 1):
            _draw_debug_pair(item, debug_dir / spec.pair_id / f'match-{index:02d}.jpg')
    return result

def evaluate_bridge(spec: PairSpec, source_features: Sequence[ImageFeature], target_features: Sequence[ImageFeature], config: BridgeConfig, debug_dir: Path | None=None, shared_sift_cache: dict[Path, ImageFeature | None] | None=None, target_retrieval_index: RetrievalIndex | None=None) -> dict[str, Any]:
    search = _collect_bridge_evidence(spec, source_features, target_features, config, shared_sift_cache, target_retrieval_index)
    return _evaluate_bridge_from_search(spec, source_features, target_features, config, search, debug_dir)
_DEFAULT_BRIDGE_EVALUATOR = evaluate_bridge

def _mode_result_sort_key(result: dict[str, Any], cluster: Sequence[PairEvidence]) -> tuple[Any, ...]:
    median = result['aggregate_residual_median_m']
    return (not bool(result['accepted']), -int(result['independent_cluster_pairs']), -int(result['aggregate_3d_inliers']), float(median) if median is not None else math.inf, tuple(sorted((_frame_pair_id(item) for item in cluster))))

def _evaluate_bridge_hypotheses_from_search(spec: PairSpec, source_features: Sequence[ImageFeature], target_features: Sequence[ImageFeature], config: BridgeConfig, search: _BridgeSearchEvidence, debug_dir: Path | None, *, max_hypotheses: int | None=None) -> tuple[dict[str, Any], ...]:
    if max_hypotheses is not None and (isinstance(max_hypotheses, bool) or not isinstance(max_hypotheses, int) or max_hypotheses < 0):
        raise ValueError('max_hypotheses must be a non-negative integer or None')
    legacy_cluster, legacy_method = _select_legacy_cluster(search.verified, config)
    clusters = cluster_pair_model_hypotheses(search.verified, config)
    legacy_ids = frozenset((_frame_pair_id(item) for item in legacy_cluster))
    cluster_inputs = [(cluster, 'per_frame_transform_cluster') for cluster in clusters]
    legacy_support_injected = not any((frozenset((_frame_pair_id(item) for item in cluster)) == legacy_ids for cluster in clusters))
    if legacy_support_injected:
        cluster_inputs.append((legacy_cluster, legacy_method))
    evaluated: list[tuple[dict[str, Any], tuple[PairEvidence, ...]]] = []
    for cluster, cluster_method in cluster_inputs:
        result = _evaluate_bridge_cluster(spec, source_features, target_features, config, search, cluster, cluster_method, reject_invalid_icp_transform=True)
        evaluated.append((result, cluster))
    evaluated.sort(key=lambda item: _mode_result_sort_key(item[0], item[1]))
    mode_ids = tuple((f'M{index + 1}' for index in range(len(evaluated))))
    selected_index = next((index for index, (_result, cluster) in enumerate(evaluated) if frozenset((_frame_pair_id(item) for item in cluster)) == legacy_ids), None)
    legacy_selected_mode_id = mode_ids[selected_index] if selected_index is not None else None
    legacy_selection_state = 'injected_exact_legacy_support' if legacy_support_injected else 'matched_exact_support'
    output: list[dict[str, Any]] = []
    for index, (base_result, cluster) in enumerate(evaluated):
        result = dict(base_result)
        result['mode_id'] = mode_ids[index]
        result['legacy_selected_mode_id'] = legacy_selected_mode_id
        result['legacy_selection_state'] = legacy_selection_state
        output.append(result)
        if debug_dir is not None:
            mode_dir = debug_dir / spec.pair_id / mode_ids[index]
            for pair_index, item in enumerate(cluster[:8], 1):
                _draw_debug_pair(item, mode_dir / f'match-{pair_index:02d}.jpg')
    return tuple(output if max_hypotheses is None else output[:max_hypotheses])

def evaluate_bridge_hypotheses(spec: PairSpec, source_features: Sequence[ImageFeature], target_features: Sequence[ImageFeature], config: BridgeConfig, debug_dir: Path | None=None, shared_sift_cache: dict[Path, ImageFeature | None] | None=None, target_retrieval_index: RetrievalIndex | None=None, *, max_hypotheses: int | None=None) -> tuple[dict[str, Any], ...]:
    search = _collect_bridge_evidence(spec, source_features, target_features, config, shared_sift_cache, target_retrieval_index)
    return _evaluate_bridge_hypotheses_from_search(spec, source_features, target_features, config, search, debug_dir, max_hypotheses=max_hypotheses)

def _config_json(config: BridgeConfig) -> dict[str, Any]:
    return {key: value for key, value in config.__dict__.items() if isinstance(value, (str, int, float, bool, type(None)))}

def _report_text(result: dict[str, Any]) -> str:
    lines = ['# Visual-Depth Bridge Evidence', '', f"Status: `{result['status']}`", f"Search mode: `{result['search_mode']}`", f"Measurement units: `{result.get('measurement_unit_count') or 'selected'}`", f"Cross-unit pairs evaluated: `{result['cross_unit_pair_count']}`", '', 'COLMAP was not used. Raw captures were read-only. No LiDAR range cutoff was applied.', 'Provisional/GPS transforms were context-only seeds and did not create accepted edges.', '', '## Pair decisions', '', '| Pair | Sides | RGB frames | 3D pairs | Aggregate inliers | Local sym fit 25cm | Decision |', '|---|---|---:|---:|---:|---:|---|']
    for item in result['pairs']:
        icp = item.get('local_icp', {})
        fit = None
        if icp.get('available'):
            fit = icp['thresholds']['0.25m']['symmetric_fitness']
        fit_text = f'{fit:.4f}' if fit is not None else 'n/a'
        lines.append(f"| {item['id']} | {item['source_side']} -> {item['target_side']} | {item['source_rgb_frames']} + {item['target_rgb_frames']} | {item['independent_cluster_pairs']} | {item['aggregate_3d_inliers']} | {fit_text} | {item['status']} |")
    lines.extend(['', '## Gate failures', ''])
    for item in result['pairs']:
        failures = ', '.join(item['rejection_reasons']) or 'none'
        lines.append(f"- {item['id']}: {failures}")
    lines.extend(['', '## Context-seed diagnostics', ''])
    conflict_rows = 0
    for item in result['pairs']:
        conflicts = item.get('seed', {}).get('diagnostic_conflicts', [])
        if conflicts:
            conflict_rows += 1
            lines.append(f"- {item['id']}: {', '.join(conflicts)}; diagnostic only")
    if not conflict_rows:
        lines.append('- No visual model was in diagnostic conflict with its context seed.')
    lines.extend(['', '## Interpretation', '', 'Only ACCEPTED_VISUAL_DEPTH_BRIDGE edges may enter the measurement graph.', 'Rejected transforms remain diagnostic candidates and must not merge components.', ''])
    return '\n'.join(lines)

def run_search(evidence_path: str | Path, raw_root: str | Path, pair_ids: Sequence[str] | None, output_dir: str | Path, config: BridgeConfig | None=None, repo_root: str | Path | None=None, all_cross_components: bool=False, *, bridge_evaluator: Callable[..., Mapping[str, Any]] | None=None) -> dict[str, Any]:
    chosen = config or BridgeConfig()
    if all_cross_components:
        if pair_ids:
            raise BridgeInputError('all-cross-components mode is mutually exclusive with explicit pair ids')
        _evidence, search_units, specs, root = load_all_pair_specs(evidence_path, repo_root)
        search_mode = 'all_cross_components'
    else:
        if not pair_ids:
            raise BridgeInputError('explicit pair mode requires at least one pair id')
        _evidence, specs, root = load_pair_specs(evidence_path, pair_ids, repo_root)
        search_units = ()
        search_mode = 'provisional_edge_selection'
    raw = Path(raw_root).resolve()
    if not raw.is_dir():
        raise BridgeInputError(f'raw root does not exist: {raw_root}')
    output = Path(output_dir).resolve()
    input_paths = {Path(evidence_path).resolve(), raw}
    if output in input_paths or raw in output.parents:
        raise BridgeInputError('output directory cannot be inside a raw capture directory')
    output.mkdir(parents=True, exist_ok=True)
    feature_cache: dict[tuple[str, tuple[float, ...]], tuple[ImageFeature, ...]] = {}
    retrieval_index_cache: dict[tuple[str, tuple[float, ...]], RetrievalIndex] = {}
    shared_sift_cache: dict[Path, ImageFeature | None] = {}

    def side_key(side: SideSpec) -> tuple[str, tuple[float, ...]]:
        return (side.side_id, tuple((float(value) for run in side.runs for value in run.T_side_from_run.ravel())))

    def side_features(side: SideSpec) -> tuple[ImageFeature, ...]:
        key = side_key(side)
        if key not in feature_cache:
            frames = enumerate_side_frames(side, raw)
            feature_cache[key] = extract_features(frames, chosen, side.side_id)
        return feature_cache[key]

    def side_retrieval_index(side: SideSpec) -> RetrievalIndex:
        key = side_key(side)
        if key not in retrieval_index_cache:
            retrieval_index_cache[key] = build_retrieval_index(side_features(side))
        return retrieval_index_cache[key]
    evaluator = bridge_evaluator if bridge_evaluator is not None else evaluate_bridge
    pair_results: list[dict[str, Any]] = []
    pair_hypotheses: list[dict[str, Any]] = []
    for spec in specs:
        source_features = side_features(spec.source)
        target_features = side_features(spec.target)
        target_index = side_retrieval_index(spec.target)
        evaluator_args = (spec, source_features, target_features, chosen, output / 'debug-matches', shared_sift_cache, target_index)
        if evaluator is _DEFAULT_BRIDGE_EVALUATOR:
            search = _collect_bridge_evidence(spec, source_features, target_features, chosen, shared_sift_cache, target_index)
            legacy_result = _evaluate_bridge_from_search(spec, source_features, target_features, chosen, search, output / 'debug-matches')
            modes = _evaluate_bridge_hypotheses_from_search(spec, source_features, target_features, chosen, search, output / 'debug-matches')
        else:
            legacy_result = dict(evaluator(*evaluator_args))
            injected_mode = dict(legacy_result)
            injected_mode['mode_id'] = 'M1'
            injected_mode['legacy_selected_mode_id'] = 'M1'
            injected_mode['legacy_selection_state'] = 'injected_single_dict_legacy'
            modes = (injected_mode,)
        pair_results.append(legacy_result)
        pair_hypotheses.append({'id': spec.pair_id, 'legacy_selected_mode_id': modes[0]['legacy_selected_mode_id'], 'legacy_selection_state': modes[0]['legacy_selection_state'], 'modes': list(modes)})
    dataset_rgb_total = 0
    dataset_runs = []
    for run_dir in sorted((path for path in raw.iterdir() if path.is_dir() and path.name != '_mesh_only')):
        rgb_count = len(list(run_dir.glob('frame_*.jpg')))
        if rgb_count:
            dataset_runs.append({'run': run_dir.name, 'rgb_frames': rgb_count})
            dataset_rgb_total += rgb_count
    accepted = sum((bool(item['accepted']) for item in pair_results))
    result = {'schema': SCHEMA, 'status': 'accepted_edges_found' if accepted else 'insufficient_cross_component_evidence', 'search_mode': search_mode, 'units': 'meter', 'axis': 'ARKit_Y_up', 'matrix_convention': 'column_vector; p_target = T_target_from_source @ p_source', 'COLMAP_used': False, 'raw_capture_policy': 'read_only', 'range_cutoff_used': False, 'gps_and_provisional_seed_policy': 'context_only_not_acceptance_evidence', 'source_registration_evidence': os.path.relpath(Path(evidence_path).resolve(), root), 'dataset_inventory': {'runs': dataset_runs, 'rgb_frames_total': dataset_rgb_total}, 'searched_unique_rgb_frames': len({str(feature.frame.rgb_path) for features in feature_cache.values() for feature in features}), 'searched_unique_sift_frames': len(shared_sift_cache), 'measurement_units': [{'id': unit.side.side_id, 'runs': [run.run_id for run in unit.side.runs], 'global_context_method': unit.context_method} for unit in search_units], 'measurement_unit_count': len(search_units) if search_units else None, 'cross_unit_pair_count': len(specs), 'config': _config_json(chosen), 'accepted_edge_count': accepted, 'pairs': pair_results, 'pair_hypotheses': pair_hypotheses}
    json_path = output / 'visual-depth-bridge-evidence.json'
    report_path = output / 'visual-depth-bridge-report.md'
    json_tmp = output / '.visual-depth-bridge-evidence.json.tmp'
    report_tmp = output / '.visual-depth-bridge-report.md.tmp'
    json_tmp.write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n', encoding='ascii')
    report_tmp.write_text(_report_text(result), encoding='ascii')
    os.replace(json_tmp, json_path)
    os.replace(report_tmp, report_path)
    return result

def _parse_args(argv: Sequence[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Search conservative RGB plus depth cross-component bridges without COLMAP.')
    parser.add_argument('evidence', type=Path, help='Authoritative registration evidence JSON')
    parser.add_argument('--raw-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pairs', help='Comma-separated provisional edge ids')
    mode.add_argument('--all-cross-components', action='store_true', help='Search every unordered pair of evidence-safe measurement units')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path)
    parser.add_argument('--max-candidates', type=int, default=BridgeConfig.max_candidates)
    parser.add_argument('--retrieval-top-k', type=int, default=BridgeConfig.retrieval_top_k)
    parser.add_argument('--orb-features', type=int, default=BridgeConfig.orb_features)
    parser.add_argument('--max-sift-fallbacks', type=int, default=BridgeConfig.max_sift_fallbacks_per_pair)
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None=None) -> int:
    args = _parse_args(argv)
    pair_ids = [item.strip() for item in args.pairs.split(',') if item.strip()] if args.pairs else None
    if args.pairs is not None and (not pair_ids):
        raise SystemExit('--pairs must contain at least one edge id')
    if args.max_candidates < 1 or args.retrieval_top_k < 1:
        raise SystemExit('candidate and retrieval limits must be positive')
    if args.max_sift_fallbacks < 0:
        raise SystemExit('--max-sift-fallbacks cannot be negative')
    config = BridgeConfig(max_candidates=args.max_candidates, retrieval_top_k=args.retrieval_top_k, orb_features=args.orb_features, max_sift_fallbacks_per_pair=args.max_sift_fallbacks)
    result = run_search(args.evidence, args.raw_root, pair_ids, args.out, config, args.repo_root, args.all_cross_components)
    summary = {'status': result['status'], 'accepted_edge_count': result['accepted_edge_count'], 'evidence': str((args.out / 'visual-depth-bridge-evidence.json').resolve()), 'report': str((args.out / 'visual-depth-bridge-report.md').resolve())}
    print(json.dumps(summary, ensure_ascii=True))
    return 0 if result['accepted_edge_count'] else 2
if __name__ == '__main__':
    raise SystemExit(main())
