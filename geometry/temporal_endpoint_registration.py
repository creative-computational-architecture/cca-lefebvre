
from __future__ import annotations
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import numpy as np
from geometry.alignment_hypotheses import AlignmentHypothesis, HypothesisFeatures, HypothesisState, PairHypothesisSet, make_null_hypothesis, validate_zup_rigid_transform, yup_transform_to_zup
from geometry.register import EdgeResult, is_gravity_locked, project_yaw_translation
from geometry.temporal_sequence import TemporalPair
from geometry.visual_bridge import BridgeConfig, FrameSpec, PairSpec, RunInSide, SideSpec, build_retrieval_index, enumerate_side_frames, evaluate_bridge, evaluate_bridge_hypotheses, extract_features
SCHEMA = 'cca.temporal-endpoint-registration.v1'

class TemporalEndpointError(ValueError):
    pass

def _is_gravity_locked_rigid_transform(transform: np.ndarray, *, atol: float=1e-06) -> bool:
    matrix = np.asarray(transform, dtype=np.float64)
    return bool(matrix.shape == (4, 4) and np.isfinite(matrix).all() and np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol) and is_gravity_locked(matrix, atol=atol))

@dataclass(frozen=True, slots=True)
class EndpointWindowConfig:
    max_frames_per_endpoint: int = 180
    min_frames_per_endpoint: int = 6
    max_seconds_per_endpoint: float = 30.0

    def __post_init__(self) -> None:
        if self.max_frames_per_endpoint < 1:
            raise TemporalEndpointError('endpoint frame maximum must be positive')
        if not 1 <= self.min_frames_per_endpoint <= self.max_frames_per_endpoint:
            raise TemporalEndpointError('endpoint frame minimum is invalid')
        if isinstance(self.max_seconds_per_endpoint, bool) or not math.isfinite(self.max_seconds_per_endpoint) or self.max_seconds_per_endpoint < 0:
            raise TemporalEndpointError('endpoint seconds must be finite and nonnegative')

@dataclass(frozen=True, slots=True)
class TemporalEndpointResult:
    source_id: str
    target_id: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    transform_target_from_source_yup: np.ndarray
    fitness: float
    rmse_m: float
    overlap: float
    correspondences: int
    information_yup: np.ndarray
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_points_preserved(self) -> bool:
        return bool(self.evidence.get('raw_points_preserved', True))

    def as_register_edge(self, gps_distance_m: float=0.0) -> EdgeResult | None:
        if not self.accepted:
            return None
        return EdgeResult(source=self.source_id, target=self.target_id, gps_distance_m=float(gps_distance_m), initial_transform=np.eye(4, dtype=np.float64), transform_source_to_target=self.transform_target_from_source_yup.copy(), fitness=self.fitness, inlier_rmse_m=self.rmse_m, correspondence_count=self.correspondences, accepted=True, rejection_reasons=(), information=self.information_yup.copy(), iterations_run=(), certain=False)

    def to_dict(self) -> dict[str, Any]:
        return {'schema': SCHEMA, 'source_id': self.source_id, 'target_id': self.target_id, 'accepted': self.accepted, 'rejection_reasons': list(self.rejection_reasons), 'transform_target_from_source_yup': self.transform_target_from_source_yup.tolist(), 'fitness': self.fitness, 'rmse_m': self.rmse_m, 'overlap': self.overlap, 'correspondences': self.correspondences, 'raw_points_preserved': self.raw_points_preserved, 'fixed_scale': 1.0, 'axis': 'ARKit_Y_up', 'evidence': self.evidence}

def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def select_endpoint_frames(frames: Sequence[FrameSpec], endpoint: str, config: EndpointWindowConfig | None=None) -> tuple[FrameSpec, ...]:
    chosen = config or EndpointWindowConfig()
    ordered = tuple(sorted(frames, key=lambda item: item.frame_index))
    if not ordered:
        return ()
    if endpoint.upper() not in {'HEAD', 'TAIL'}:
        raise TemporalEndpointError('endpoint must be HEAD or TAIL')
    if chosen.max_seconds_per_endpoint > 0:
        clocks = _endpoint_clocks(ordered)
        if clocks is None:
            raise TemporalEndpointError('endpoint frame clock unavailable; timed search cannot be verified')
        if np.any(np.diff(clocks) < 0):
            raise TemporalEndpointError('endpoint frame clock regressed within one run')
        try:
            raw_frames = list(ordered[0].json_path.parent.glob('frame_*.json'))
            raw_frames.sort(key=lambda path: int(path.stem.rsplit('_', 1)[-1]))
            anchor_path = raw_frames[0] if endpoint.upper() == 'HEAD' else raw_frames[-1]
            anchor = json.loads(anchor_path.read_text(encoding='utf-8')).get('time')
        except (OSError, ValueError, IndexError, AttributeError) as error:
            raise TemporalEndpointError('raw capture endpoint clock unavailable') from error
        if isinstance(anchor, bool) or not isinstance(anchor, (int, float)) or (not math.isfinite(anchor)):
            raise TemporalEndpointError('raw capture endpoint clock unavailable')
        ordered = tuple((frame for frame, stamp in zip(ordered, clocks) if abs(stamp - anchor) <= chosen.max_seconds_per_endpoint))
    count = min(len(ordered), chosen.max_frames_per_endpoint)
    if endpoint.upper() == 'HEAD':
        return ordered[:count]
    if endpoint.upper() == 'TAIL':
        return ordered[-count:]
    raise TemporalEndpointError('endpoint must be HEAD or TAIL')

def _endpoint_clocks(frames: Sequence[FrameSpec]) -> np.ndarray | None:
    times = []
    for frame in frames:
        try:
            value = json.loads(frame.json_path.read_text(encoding='utf-8')).get('time')
        except (OSError, UnicodeError, ValueError, AttributeError):
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)):
            return None
        times.append(float(value))
    return np.asarray(times, dtype=float)

def endpoint_window_evidence(frames: Sequence[FrameSpec], config: EndpointWindowConfig) -> dict:
    clocks = _endpoint_clocks(frames)
    return {'max_seconds': config.max_seconds_per_endpoint, 'clock_source': 'FRAME_CLOCK' if clocks is not None and len(clocks) else 'FRAME_INDEX_FALLBACK', 'first_sensor_time_s': float(clocks[0]) if clocks is not None and len(clocks) else None, 'last_sensor_time_s': float(clocks[-1]) if clocks is not None and len(clocks) else None, 'point_policy': 'SEARCH_WINDOW_ONLY_NO_SOURCE_POINTS_REMOVED'}

def _workset_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise TemporalEndpointError(f'registration workset is missing: {resolved}')
    return resolved

def _side(run_id: str, workset: Path | None) -> SideSpec:
    return SideSpec(side_id=f'RUN_{run_id}', anchor_run=run_id, runs=(RunInSide(run_id, np.eye(4, dtype=np.float64), workset),))

def _quality_from_bridge(result: dict[str, Any]) -> tuple[float, float, int]:
    local = result.get('local_icp', {})
    if not isinstance(local, dict) or not local.get('available'):
        return (0.0, 0.0, 0)
    thresholds = local.get('thresholds', {})
    threshold = thresholds.get('0.25m', {}) if isinstance(thresholds, dict) else {}
    if not isinstance(threshold, dict):
        return (0.0, 0.0, 0)
    forward = threshold.get('source_to_target', {})
    reverse = threshold.get('target_to_source', {})
    if not isinstance(forward, dict) or not isinstance(reverse, dict):
        return (0.0, 0.0, 0)
    fitness = float(threshold.get('symmetric_fitness', 0.0))
    rmses = [float(item.get('rmse_m', 0.0)) for item in (forward, reverse) if int(item.get('correspondences', 0)) > 0]
    count = min(int(forward.get('correspondences', 0)), int(reverse.get('correspondences', 0)))
    return (fitness, max(rmses, default=0.0), count)

def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()

def _endpoint_paths(frames: Sequence[FrameSpec], raw_root: Path) -> tuple[tuple[str, Path], ...]:
    entries: list[tuple[str, Path]] = []
    for frame in frames:
        paths = (frame.json_path, frame.rgb_path, frame.depth_path, frame.conf_path)
        for path in paths:
            if path is None:
                continue
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(raw_root).as_posix()
            except ValueError as exc:
                raise TemporalEndpointError(f'endpoint evidence is outside raw root: {resolved}') from exc
            entries.append((relative, resolved))
    entries.sort(key=lambda item: item[0])
    return tuple(entries)

def _raw_endpoint_leaf_sha256(relative: str, path: Path) -> str:
    if path.is_file():
        digest = _sha256(path)
        if digest is not None:
            return digest
    return _canonical_sha256({'path': relative, 'state': 'MISSING'})

def _endpoint_merkle(entries: Sequence[tuple[str, Path]]) -> str:
    leaves = [{'path': relative, 'sha256': _raw_endpoint_leaf_sha256(relative, path)} for relative, path in entries]
    return _canonical_sha256(leaves)

def _frame_inventory_sha256(entries: Sequence[tuple[str, Path]]) -> str:
    return _canonical_sha256([relative for relative, _path in entries])

def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)

def _selected_frame_pairs(result: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    value = result.get('selected_frame_pairs', ())
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        return tuple((item for item in value if isinstance(item, Mapping)))
    return ()

def _minimum_image_grid_cells(pairs: Sequence[Mapping[str, Any]]) -> int | None:
    values = [int(item[key]) for item in pairs for key in ('source_grid_cells', 'target_grid_cells') if item.get(key) is not None]
    return min(values) if values else None

def _minimum_image_span(pairs: Sequence[Mapping[str, Any]]) -> float | None:
    values: list[float] = []
    for item in pairs:
        for key in ('source_image_span', 'target_image_span'):
            raw = item.get(key)
            if isinstance(raw, Sequence) and (not isinstance(raw, (str, bytes))) and (len(raw) == 2):
                values.append(max(float(raw[0]), float(raw[1])))
    return min(values) if values else None

def _features_from_bridge(result: Mapping[str, Any]) -> HypothesisFeatures:
    selected = _selected_frame_pairs(result)
    visual_matches = sum((int(item.get('visual_matches', 0)) for item in selected))
    image_inliers = sum((int(item.get('homography_inliers', 0)) for item in selected))
    image_ratio = image_inliers / visual_matches if visual_matches else None
    local = result.get('local_icp', {})
    local = local if isinstance(local, Mapping) else {}
    thresholds = local.get('thresholds', {})
    thresholds = thresholds if isinstance(thresholds, Mapping) else {}
    threshold = thresholds.get('0.25m', {})
    threshold = threshold if isinstance(threshold, Mapping) else {}
    forward = threshold.get('source_to_target', {})
    reverse = threshold.get('target_to_source', {})
    forward = forward if isinstance(forward, Mapping) else {}
    reverse = reverse if isinstance(reverse, Mapping) else {}
    rmses = [float(item['rmse_m']) for item in (forward, reverse) if int(item.get('correspondences', 0)) > 0 and item.get('rmse_m') is not None]
    correspondences = min(int(forward.get('correspondences', 0)), int(reverse.get('correspondences', 0)))
    return HypothesisFeatures(independent_frame_pairs=int(result.get('independent_cluster_pairs', 0)), image_inliers=image_inliers, image_inlier_ratio=image_ratio, image_grid_cells=_minimum_image_grid_cells(selected), image_span_fraction=_minimum_image_span(selected), depth_inliers=int(result.get('aggregate_3d_inliers', 0)), depth_inlier_ratio=_optional_float(result.get('aggregate_3d_ratio')), residual_median_m=_optional_float(result.get('aggregate_residual_median_m')), residual_p90_m=_optional_float(result.get('aggregate_residual_p90_m')), inlier_span_m=_optional_float(result.get('aggregate_inlier_span_m')), evaluation_voxel_m=0.25 if threshold else None, bidirectional_overlap=_optional_float(threshold.get('symmetric_fitness')), symmetric_fitness_025m=_optional_float(threshold.get('symmetric_fitness')), rmse_m=max(rmses, default=None), correspondences=correspondences, normal_agreement=_optional_float(result.get('normal_agreement')), icp_translation_correction_m=_optional_float(local.get('translation_correction_m')), icp_yaw_correction_deg=_optional_float(local.get('yaw_correction_deg')), rgb_matched_to_shuffled_ratio=_optional_float(local.get('rgb_matched_to_shuffled_ratio')), confidence_support=_optional_float(result.get('confidence_support')), inverse_translation_error_m=_optional_float(result.get('inverse_translation_error_m')), inverse_yaw_error_deg=_optional_float(result.get('inverse_yaw_error_deg')), rerun_agreement=_optional_float(result.get('rerun_agreement')), bootstrap_yaw_spread_deg=_optional_float(result.get('bootstrap_yaw_spread_deg')), bootstrap_translation_spread_m=_optional_float(result.get('bootstrap_translation_spread_m')))

def _batch_seed(initial: np.ndarray | None) -> np.ndarray:
    if initial is None:
        return np.eye(4, dtype=np.float64)
    raw_seed = np.asarray(initial, dtype=np.float64)
    if not _is_gravity_locked_rigid_transform(raw_seed):
        raise TemporalEndpointError('endpoint seed must be gravity locked Y-up')
    return project_yaw_translation(raw_seed)

def _registration_frames(side, raw, endpoint, config, mode, focus=()):
    frames = enumerate_side_frames(side, raw)
    if mode == 'ENDPOINT':
        return select_endpoint_frames(frames, endpoint, config)
    if mode != 'OVERLAP':
        raise TemporalEndpointError('search mode must be ENDPOINT or OVERLAP')
    count = (config or EndpointWindowConfig()).max_frames_per_endpoint
    if len(frames) <= count:
        return frames
    if focus:
        near = [f for f in frames if min((abs(f.frame_index - i) for i in focus)) <= 120]
        exact = [f for f in near if f.frame_index in focus]
        others = [f for f in near if f.frame_index not in focus]
        near_budget = min(count * 2 // 3, len(near))
        selected = {f.frame_index: f for f in exact[:near_budget]}
        remaining = near_budget - len(selected)
        if remaining > 0:
            selected.update(((others[i].frame_index, others[i]) for i in np.linspace(0, len(others) - 1, remaining, dtype=int)))
        rest = [f for f in frames if f.frame_index not in selected]
        remaining = min(count - len(selected), len(rest))
        if remaining:
            selected.update(((rest[i].frame_index, rest[i]) for i in np.linspace(0, len(rest) - 1, remaining, dtype=int)))
        return tuple((selected[k] for k in sorted(selected)))
    return tuple((frames[i] for i in np.linspace(0, len(frames) - 1, count, dtype=int)))

def _overlap_refinement_focus(modes):
    allowed = {'independent_frame_pairs_below_min', 'leave_one_frame_out_passes_below_min'}
    promising = [m for m in modes if not m.get('accepted') and m.get('independent_cluster_pairs', 0) >= 2 and m.get('rejection_reasons') and (set(m['rejection_reasons']) <= allowed)]
    if not promising:
        return None
    best = max(promising, key=lambda m: m.get('aggregate_3d_inliers', 0))
    focus = {'source': [], 'target': []}
    for pair in best.get('selected_frame_pairs', ()):
        for side in focus:
            label = pair.get(side, '')
            if '/frame_' in label and label.rsplit('_', 1)[1].isdigit():
                focus[side].append(int(label.rsplit('_', 1)[1]))
    return focus if all(focus.values()) else None

def _hypothesis_support_rank(result):
    return max(((h.state is HypothesisState.ELIGIBLE_FOR_REVIEW, h.features.independent_frame_pairs or 0, h.features.depth_inliers or 0) for h in result.hypotheses if h.features is not None), default=(False, 0, 0))

def endpoint_camera_seed(source_frames: Sequence[FrameSpec], target_frames: Sequence[FrameSpec], initial: np.ndarray | None=None) -> tuple[np.ndarray, str]:
    if initial is not None:
        return (_batch_seed(initial), 'current_project_pose_context_only')
    if not source_frames or not target_frames:
        return (np.eye(4), 'identity_no_endpoint_camera')
    try:
        poses = []
        for frame in (source_frames[-1], target_frames[0]):
            doc = json.loads(frame.json_path.read_text(encoding='utf-8'))
            values = doc.get('cameraPoseARFrame', doc.get('pose'))
            pose = np.asarray(values, dtype=float).reshape(4, 4)
            if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-06) or (not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=0.0001)) or (not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=0.0001)):
                raise ValueError('endpoint camera pose is not rigid')
            poses.append(pose)
        source, target = poses
        seed = project_yaw_translation(target @ np.linalg.inv(source))
        seed[:3, 3] = target[:3, 3] - seed[:3, :3] @ source[:3, 3]
        return (seed, 'endpoint_camera_colocation_context_only')
    except (OSError, ValueError, TypeError, AttributeError, np.linalg.LinAlgError):
        return (np.eye(4), 'identity_invalid_endpoint_camera')

def register_temporal_endpoint_hypotheses(*, temporal_pair: TemporalPair, order_evidence_sha256: str, raw_root: str | Path, source_workset_yup: str | Path | None, target_workset_yup: str | Path | None, initial_target_from_source_yup: np.ndarray | None=None, window_config: EndpointWindowConfig | None=None, bridge_config: BridgeConfig | None=None, bridge_evaluator: Callable[..., Sequence[Mapping[str, Any]]]=evaluate_bridge_hypotheses, max_hypotheses: int | None=None, search_mode: str='ENDPOINT', ray_recoveries: dict | None=None, focus_frame_indices: dict | None=None) -> PairHypothesisSet:
    if max_hypotheses is not None and (isinstance(max_hypotheses, bool) or not isinstance(max_hypotheses, int) or max_hypotheses < 0):
        raise TemporalEndpointError('max_hypotheses must be a non-negative integer or None')
    raw = Path(raw_root).resolve()
    if not raw.is_dir():
        raise TemporalEndpointError(f'raw capture root is missing: {raw}')
    source_workset = _workset_path(source_workset_yup)
    target_workset = _workset_path(target_workset_yup)
    source_side = _side(temporal_pair.source_scan_id, source_workset)
    target_side = _side(temporal_pair.target_scan_id, target_workset)
    focus = focus_frame_indices or {}
    source_frames = _registration_frames(source_side, raw, 'TAIL', window_config, search_mode, focus.get('source', ()))
    target_frames = _registration_frames(target_side, raw, 'HEAD', window_config, search_mode, focus.get('target', ()))
    source_entries = _endpoint_paths(source_frames, raw)
    target_entries = _endpoint_paths(target_frames, raw)
    source_endpoint_merkle_before = _endpoint_merkle(source_entries)
    target_endpoint_merkle_before = _endpoint_merkle(target_entries)
    source_frame_inventory_sha256 = _frame_inventory_sha256(source_entries)
    target_frame_inventory_sha256 = _frame_inventory_sha256(target_entries)
    source_workset_sha256_before = _sha256(source_workset)
    target_workset_sha256_before = _sha256(target_workset)
    chosen_window = window_config or EndpointWindowConfig()
    if search_mode == 'OVERLAP':
        chosen_window = replace(chosen_window, max_seconds_per_endpoint=0.0)
    early_reasons: list[str] = []
    if len(source_frames) < chosen_window.min_frames_per_endpoint:
        early_reasons.append('source_tail_frames_below_min')
    if len(target_frames) < chosen_window.min_frames_per_endpoint:
        early_reasons.append('target_head_frames_below_min')
    seed, seed_method = endpoint_camera_seed(source_frames, target_frames, initial_target_from_source_yup) if search_mode == 'ENDPOINT' else (_batch_seed(initial_target_from_source_yup), 'overlap_context_only')
    spec = PairSpec(pair_id=temporal_pair.pair_id, source_endpoint=temporal_pair.source_scan_id, target_endpoint=temporal_pair.target_scan_id, source=source_side, target=target_side, seed_target_from_source=seed, seed_method=seed_method)
    chosen_bridge = bridge_config or BridgeConfig()
    raw_modes: Sequence[Mapping[str, Any]] = ()
    recovered_counts = {'source': 0, 'target': 0}
    if not early_reasons:
        source_features = extract_features(source_frames, chosen_bridge, f'{temporal_pair.source_scan_id}:TAIL')
        target_features = extract_features(target_frames, chosen_bridge, f'{temporal_pair.target_scan_id}:HEAD')
        if ray_recoveries:
            from geometry.ray_recovery import attach_recovery_to_features
            recovered_counts['source'] = attach_recovery_to_features(source_features, ray_recoveries.get('source'))
            recovered_counts['target'] = attach_recovery_to_features(target_features, ray_recoveries.get('target'))
        evaluated = bridge_evaluator(spec, source_features, target_features, chosen_bridge, target_retrieval_index=build_retrieval_index(target_features), max_hypotheses=max_hypotheses)
        if isinstance(evaluated, Mapping) or not isinstance(evaluated, Sequence):
            raise TemporalEndpointError('batch bridge evaluator must return a sequence of mappings')
        if any((not isinstance(item, Mapping) for item in evaluated)):
            raise TemporalEndpointError('batch bridge evaluator must return a sequence of mappings')
        raw_modes = evaluated if max_hypotheses is None else evaluated[:max_hypotheses]
    source_endpoint_merkle_after = _endpoint_merkle(source_entries)
    target_endpoint_merkle_after = _endpoint_merkle(target_entries)
    source_workset_sha256_after = _sha256(source_workset)
    target_workset_sha256_after = _sha256(target_workset)
    mutation_reasons: list[str] = []
    if source_endpoint_merkle_before != source_endpoint_merkle_after:
        mutation_reasons.append('source_raw_endpoint_mutated')
    if target_endpoint_merkle_before != target_endpoint_merkle_after:
        mutation_reasons.append('target_raw_endpoint_mutated')
    if source_workset_sha256_before != source_workset_sha256_after:
        mutation_reasons.append('source_workset_mutated')
    if target_workset_sha256_before != target_workset_sha256_after:
        mutation_reasons.append('target_workset_mutated')
    hypotheses: list[AlignmentHypothesis] = [make_null_hypothesis(temporal_pair, early_reasons)]
    mode_to_hypothesis: dict[str, str] = {}
    temporal_audit = {'gap_s': temporal_pair.gap_s, 'gap_source': temporal_pair.gap_source, 'long_gap': temporal_pair.long_gap, 'long_gap_cutoff_s': temporal_pair.long_gap_cutoff_s, 'temporal_prior': temporal_pair.temporal_prior, 'ranking_only': True}
    endpoint_windows = {'search_mode': search_mode, 'focus_frame_indices': focus, 'recovered_feature_counts': recovered_counts, 'ray_recoveries': {side: model['provenance'] for side, model in (ray_recoveries or {}).items()}, 'seed_method': seed_method, 'seed_affects_acceptance': False, 'source_time_window': endpoint_window_evidence(source_frames, chosen_window), 'target_time_window': endpoint_window_evidence(target_frames, chosen_window), 'source': 'TAIL' if search_mode == 'ENDPOINT' else 'WHOLE_RUN_SPREAD', 'target': 'HEAD' if search_mode == 'ENDPOINT' else 'WHOLE_RUN_SPREAD', 'source_frame_count': len(source_frames), 'target_frame_count': len(target_frames), 'source_first_index': source_frames[0].frame_index if source_frames else None, 'source_last_index': source_frames[-1].frame_index if source_frames else None, 'target_first_index': target_frames[0].frame_index if target_frames else None, 'target_last_index': target_frames[-1].frame_index if target_frames else None}
    for index, raw_mode in enumerate(raw_modes, 1):
        hypothesis_id = f'H{index}'
        mode_id = raw_mode.get('mode_id')
        if isinstance(mode_id, str) and mode_id:
            mode_to_hypothesis[mode_id] = hypothesis_id
        reasons = [str(item) for item in raw_mode.get('rejection_reasons', ())]
        reasons.extend(mutation_reasons)
        raw_transform = raw_mode.get('T_target_side_from_source_side')
        transform = None
        features = None
        if raw_transform is None:
            if bool(raw_mode.get('accepted')):
                reasons.append('accepted_bridge_has_no_transform')
        else:
            try:
                converted = yup_transform_to_zup(raw_transform)
                transform = validate_zup_rigid_transform(converted, label=f'{hypothesis_id} converted transform')
            except (TypeError, ValueError):
                reasons.append('bridge_transform_not_fixed_scale_rigid_yup')
            else:
                features = _features_from_bridge(raw_mode)
        if not bool(raw_mode.get('accepted')) and (not reasons):
            reasons.append('bridge_candidate_not_accepted')
        reasons = sorted(set(reasons))
        state = HypothesisState.ELIGIBLE_FOR_REVIEW if transform is not None and bool(raw_mode.get('accepted')) and (not reasons) else HypothesisState.REJECTED
        hypotheses.append(AlignmentHypothesis(hypothesis_id=hypothesis_id, pair_id=temporal_pair.pair_id, source_scan_id=temporal_pair.source_scan_id, target_scan_id=temporal_pair.target_scan_id, state=state, transform_target_from_source_zup=transform, features=features, rejection_reasons=tuple(reasons), evidence={'raw_bridge_result': dict(raw_mode), 'endpoint_windows': endpoint_windows, 'temporal_audit': temporal_audit}))
    legacy_mode_ids = {item.get('legacy_selected_mode_id') for item in raw_modes if item.get('legacy_selected_mode_id') is not None}
    legacy_selected_hypothesis_id = None
    if len(legacy_mode_ids) == 1:
        legacy_mode_id = next(iter(legacy_mode_ids))
        if isinstance(legacy_mode_id, str):
            legacy_selected_hypothesis_id = mode_to_hypothesis.get(legacy_mode_id)
    source_hashes = {'raw_endpoint_merkle_before': source_endpoint_merkle_before, 'raw_endpoint_merkle_after': source_endpoint_merkle_after, 'frame_inventory': source_frame_inventory_sha256, 'workset_yup_before': source_workset_sha256_before, 'workset_yup_after': source_workset_sha256_after}
    target_hashes = {'raw_endpoint_merkle_before': target_endpoint_merkle_before, 'raw_endpoint_merkle_after': target_endpoint_merkle_after, 'frame_inventory': target_frame_inventory_sha256, 'workset_yup_before': target_workset_sha256_before, 'workset_yup_after': target_workset_sha256_after}
    parent_hashes = (order_evidence_sha256, source_endpoint_merkle_before, target_endpoint_merkle_before, source_workset_sha256_before or 'MISSING', target_workset_sha256_before or 'MISSING')
    result = PairHypothesisSet(pair=temporal_pair, hypotheses=tuple(hypotheses), source_hashes=source_hashes, target_hashes=target_hashes, parent_hashes=parent_hashes, legacy_selected_hypothesis_id=legacy_selected_hypothesis_id, raw_points_preserved=not mutation_reasons)
    extra_focus = _overlap_refinement_focus(raw_modes) if search_mode == 'OVERLAP' and focus_frame_indices is None and (not mutation_reasons) else None
    if extra_focus:
        print(f'{temporal_pair.pair_id}: shared geometry found; refining nearby RGB views', flush=True)
        refined = register_temporal_endpoint_hypotheses(temporal_pair=temporal_pair, order_evidence_sha256=order_evidence_sha256, raw_root=raw, source_workset_yup=source_workset, target_workset_yup=target_workset, initial_target_from_source_yup=initial_target_from_source_yup, window_config=chosen_window, bridge_config=chosen_bridge, bridge_evaluator=bridge_evaluator, max_hypotheses=max_hypotheses, search_mode=search_mode, ray_recoveries=ray_recoveries, focus_frame_indices=extra_focus)
        chosen = refined if _hypothesis_support_rank(refined) >= _hypothesis_support_rank(result) else result
        audit = {'focus_frames': extra_focus, 'coarse_artifact_sha256': result.artifact_sha256, 'refined_artifact_sha256': refined.artifact_sha256, 'coarse_support': _hypothesis_support_rank(result), 'refined_support': _hypothesis_support_rank(refined), 'acceptance_gates_changed': False}
        null = replace(chosen.hypotheses[0], evidence={**chosen.hypotheses[0].evidence, 'additional_view_search': audit})
        return replace(chosen, hypotheses=(null, *chosen.hypotheses[1:]), artifact_sha256='')
    return result

def register_temporal_endpoints(*, temporal_pair: TemporalPair, raw_root: str | Path, source_workset_yup: str | Path | None, target_workset_yup: str | Path | None, initial_target_from_source_yup: np.ndarray | None=None, window_config: EndpointWindowConfig | None=None, bridge_config: BridgeConfig | None=None, bridge_evaluator: Callable[..., dict[str, Any]]=evaluate_bridge) -> TemporalEndpointResult:
    raw = Path(raw_root).resolve()
    if not raw.is_dir():
        raise TemporalEndpointError(f'raw capture root is missing: {raw}')
    source_workset = _workset_path(source_workset_yup)
    target_workset = _workset_path(target_workset_yup)
    source_side = _side(temporal_pair.source_scan_id, source_workset)
    target_side = _side(temporal_pair.target_scan_id, target_workset)
    source_frames = select_endpoint_frames(enumerate_side_frames(source_side, raw), 'TAIL', window_config)
    target_frames = select_endpoint_frames(enumerate_side_frames(target_side, raw), 'HEAD', window_config)
    source_entries = _endpoint_paths(source_frames, raw)
    target_entries = _endpoint_paths(target_frames, raw)
    source_endpoint_merkle_before = _endpoint_merkle(source_entries)
    target_endpoint_merkle_before = _endpoint_merkle(target_entries)
    chosen_window = window_config or EndpointWindowConfig()
    early_reasons: list[str] = []
    if len(source_frames) < chosen_window.min_frames_per_endpoint:
        early_reasons.append('source_tail_frames_below_min')
    if len(target_frames) < chosen_window.min_frames_per_endpoint:
        early_reasons.append('target_head_frames_below_min')
    seed, seed_method = endpoint_camera_seed(source_frames, target_frames, initial_target_from_source_yup)
    spec = PairSpec(pair_id=temporal_pair.pair_id, source_endpoint=temporal_pair.source_scan_id, target_endpoint=temporal_pair.target_scan_id, source=source_side, target=target_side, seed_target_from_source=seed, seed_method=seed_method)
    chosen_bridge = bridge_config or BridgeConfig()
    source_hash_before = _sha256(source_workset)
    target_hash_before = _sha256(target_workset)
    if early_reasons:
        bridge_result: dict[str, Any] = {'accepted': False, 'rejection_reasons': early_reasons, 'local_icp': {'available': False, 'reason': 'endpoint_frames_below_min'}}
    else:
        source_features = extract_features(source_frames, chosen_bridge, f'{temporal_pair.source_scan_id}:TAIL')
        target_features = extract_features(target_frames, chosen_bridge, f'{temporal_pair.target_scan_id}:HEAD')
        bridge_result = bridge_evaluator(spec, source_features, target_features, chosen_bridge, target_retrieval_index=build_retrieval_index(target_features))
    source_endpoint_merkle_after = _endpoint_merkle(source_entries)
    target_endpoint_merkle_after = _endpoint_merkle(target_entries)
    source_hash_after = _sha256(source_workset)
    target_hash_after = _sha256(target_workset)
    mutation_reasons: list[str] = []
    if source_endpoint_merkle_before != source_endpoint_merkle_after:
        mutation_reasons.append('source_raw_endpoint_mutated')
    if target_endpoint_merkle_before != target_endpoint_merkle_after:
        mutation_reasons.append('target_raw_endpoint_mutated')
    if source_hash_before != source_hash_after:
        mutation_reasons.append('source_workset_mutated')
    if target_hash_before != target_hash_after:
        mutation_reasons.append('target_workset_mutated')
    raw_points_preserved = not mutation_reasons
    accepted = bool(bridge_result.get('accepted')) and (not mutation_reasons)
    reasons = sorted({*(str(item) for item in bridge_result.get('rejection_reasons', ())), *mutation_reasons})
    raw_transform = bridge_result.get('T_target_side_from_source_side')
    transform = seed
    if raw_transform is not None:
        try:
            raw_candidate = np.asarray(raw_transform, dtype=np.float64)
        except (TypeError, ValueError):
            raw_candidate = np.empty((0, 0), dtype=np.float64)
        if not _is_gravity_locked_rigid_transform(raw_candidate):
            accepted = False
            reasons.append('bridge_transform_not_fixed_scale_rigid_yup')
        else:
            transform = project_yaw_translation(raw_candidate)
    elif accepted:
        accepted = False
        reasons.append('accepted_bridge_has_no_transform')
    fitness, rmse, correspondences = _quality_from_bridge(bridge_result)
    information_scale = max(float(correspondences), 1.0) * max(fitness, 0.01)
    information = np.eye(6, dtype=np.float64) * information_scale
    evidence = {'temporal_pair': temporal_pair.to_dict(), 'endpoint_windows': {'seed_method': seed_method, 'seed_affects_acceptance': False, 'source_time_window': endpoint_window_evidence(source_frames, chosen_window), 'target_time_window': endpoint_window_evidence(target_frames, chosen_window), 'source': 'TAIL', 'target': 'HEAD', 'source_frame_count': len(source_frames), 'target_frame_count': len(target_frames), 'source_first_index': source_frames[0].frame_index if source_frames else None, 'source_last_index': source_frames[-1].frame_index if source_frames else None, 'target_first_index': target_frames[0].frame_index if target_frames else None, 'target_last_index': target_frames[-1].frame_index if target_frames else None}, 'bridge': bridge_result, 'source_raw_endpoint_merkle_before': source_endpoint_merkle_before, 'source_raw_endpoint_merkle_after': source_endpoint_merkle_after, 'target_raw_endpoint_merkle_before': target_endpoint_merkle_before, 'target_raw_endpoint_merkle_after': target_endpoint_merkle_after, 'source_workset_sha256_before': source_hash_before, 'source_workset_sha256_after': source_hash_after, 'target_workset_sha256_before': target_hash_before, 'target_workset_sha256_after': target_hash_after, 'raw_points_preserved': raw_points_preserved, 'time_used_as_context_only': True}
    return TemporalEndpointResult(source_id=temporal_pair.source_scan_id, target_id=temporal_pair.target_scan_id, accepted=accepted, rejection_reasons=tuple(sorted(set(reasons))), transform_target_from_source_yup=transform, fitness=fitness, rmse_m=rmse, overlap=fitness, correspondences=correspondences, information_yup=information, evidence=evidence)
__all__ = ['SCHEMA', 'EndpointWindowConfig', 'TemporalEndpointError', 'TemporalEndpointResult', 'register_temporal_endpoint_hypotheses', 'register_temporal_endpoints', 'select_endpoint_frames']
