
from __future__ import annotations
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from geometry.alignment_hypotheses import AlignmentHypothesis, HypothesisState, PairHypothesisSet
from geometry.auto_registration import AXIS_Z_UP, UNIT_METERS, PairRegistrationConfig, RegistrationContract, register_pair_files
from geometry.register import Manifest, RegistrationConfig, RunSpec, optimize_pose_graph
from geometry.temporal_endpoint_registration import TemporalEndpointResult, register_temporal_endpoint_hypotheses
from geometry.temporal_sequence import TemporalPair, TemporalSequence, build_temporal_sequence
from viewer.alignment_workflow import FIXED_SCALE, POSE_GRAPH_DOF, AlignmentQuality, PairRegistrationJob, PairRegistrationResult, PoseGraphJobContract, PoseGraphResult
from viewer.alignment_scope import alignment_exclusions, resolve_alignment_exclusion
from viewer.project_model import ProjectModel, ScanEntry
_BASIS_YUP_TO_ZUP = np.eye(4, dtype=np.float64)
_BASIS_YUP_TO_ZUP[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
_BASIS_ZUP_TO_YUP = _BASIS_YUP_TO_ZUP.T
_PAIR_AUDIT_REASON = 'pair_raw_points_not_preserved'

class RegistrationServiceError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class ResolvedScanGeometry:
    scan_id: str
    path: Path
    sha256: str

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    except OSError as error:
        raise RegistrationServiceError(f'cannot hash scan geometry: {path.name}') from error
    return digest.hexdigest()

def _transform_sha256(transform: object) -> str:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RegistrationServiceError('scan transform must be a finite 4x4 matrix')
    encoded = json.dumps(matrix.tolist(), ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()

def _zup_to_yup(transform: object) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RegistrationServiceError('project transform must be a finite 4x4 matrix')
    return _BASIS_ZUP_TO_YUP @ matrix @ _BASIS_YUP_TO_ZUP

def _yup_to_zup(transform: object) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RegistrationServiceError('pose graph transform must be a finite 4x4 matrix')
    return _BASIS_YUP_TO_ZUP @ matrix @ _BASIS_ZUP_TO_YUP

def _has_explicit_placement(scan: ScanEntry) -> bool:
    return bool(scan.placement_locked or isinstance(scan.metadata.get('manual_placement'), Mapping) or isinstance(scan.metadata.get('plan_fit'), Mapping))

def _trusted_relative_seed(source_scan: ScanEntry, target_scan: ScanEntry) -> np.ndarray | None:
    source_pose = np.asarray(source_scan.transform, dtype=np.float64)
    target_pose = np.asarray(target_scan.transform, dtype=np.float64)
    identity = np.eye(4, dtype=np.float64)
    for scan, pose in ((source_scan, source_pose), (target_scan, target_pose)):
        if np.allclose(pose, identity, rtol=0.0, atol=1e-12) and (not _has_explicit_placement(scan)):
            return None
    return np.linalg.inv(target_pose) @ source_pose

def _result_field(result: object, name: str) -> object:
    if not hasattr(result, name):
        raise RegistrationServiceError(f'pair registrar omitted {name}')
    return getattr(result, name)

def _quality_field(quality: object, name: str, default: object=None) -> object:
    return getattr(quality, name, default)

def _hash_side_preserved(hashes: Mapping[str, str | None], label: str) -> bool:
    raw_keys = ('raw_endpoint_merkle_before', 'raw_endpoint_merkle_after')
    missing_raw = [key for key in raw_keys if key not in hashes]
    if missing_raw:
        raise RegistrationServiceError(f'{label} hash evidence is missing: ' + ', '.join(missing_raw))
    raw_before, raw_after = (hashes[key] for key in raw_keys)
    if not isinstance(raw_before, str) or not isinstance(raw_after, str):
        raise RegistrationServiceError(f'{label} raw endpoint hash evidence must contain digests')
    workset_keys = ('workset_yup_before', 'workset_yup_after')
    workset_presence = tuple((key in hashes for key in workset_keys))
    if workset_presence[0] is not workset_presence[1]:
        raise RegistrationServiceError(f'{label} workset hash evidence requires before and after together')
    workset_preserved = True
    if all(workset_presence):
        workset_before, workset_after = (hashes[key] for key in workset_keys)
        workset_preserved = workset_before == workset_after
    return raw_before == raw_after and workset_preserved

def _derived_raw_preservation(pair_set: PairHypothesisSet) -> bool:
    return _hash_side_preserved(pair_set.source_hashes, 'source') and _hash_side_preserved(pair_set.target_hashes, 'target')

def _audit_only_pair_set(pair_set: PairHypothesisSet) -> PairHypothesisSet:
    hypotheses = tuple((replace(hypothesis, state=HypothesisState.REJECTED, rejection_reasons=tuple(dict.fromkeys((*hypothesis.rejection_reasons, _PAIR_AUDIT_REASON)))) if hypothesis.state is HypothesisState.ELIGIBLE_FOR_REVIEW else hypothesis for hypothesis in pair_set.hypotheses))
    if hypotheses == pair_set.hypotheses:
        return pair_set
    return PairHypothesisSet(pair=pair_set.pair, hypotheses=hypotheses, source_hashes=pair_set.source_hashes, target_hashes=pair_set.target_hashes, parent_hashes=pair_set.parent_hashes, legacy_selected_hypothesis_id=pair_set.legacy_selected_hypothesis_id, raw_points_preserved=False, fixed_scale=pair_set.fixed_scale, axis=pair_set.axis)

class ProjectRegistrationService:

    def __init__(self, project: ProjectModel, *, project_root: str | Path | None=None, pair_config: PairRegistrationConfig | None=None, pose_graph_config: RegistrationConfig | None=None, pair_registrar: Callable[..., object]=register_pair_files, pose_graph_optimizer: Callable[..., object]=optimize_pose_graph, temporal_registrar: Callable[..., object]=register_temporal_endpoint_hypotheses, search_mode: str='ENDPOINT') -> None:
        project.validate()
        self.project = project
        if project_root is not None:
            self.project_root = Path(project_root).resolve()
        elif project.source_manifest:
            self.project_root = Path(project.source_manifest).resolve().parent
        else:
            self.project_root = None
        self.pair_config = pair_config
        self.pose_graph_config = pose_graph_config or RegistrationConfig()
        self._pair_registrar = pair_registrar
        self._pose_graph_optimizer = pose_graph_optimizer
        self._temporal_registrar = temporal_registrar
        if search_mode not in {'ENDPOINT', 'OVERLAP'}:
            raise RegistrationServiceError('unknown RGB search mode')
        self.search_mode = search_mode
        self._geometry_results: dict[str, object] = {}
        self._edge_hashes: dict[str, Mapping[str, str]] = {}
        self._edge_pose_hashes: dict[str, Mapping[str, str]] = {}
        self._edge_auxiliary_hashes: dict[str, tuple[tuple[Path, str], ...]] = {}
        self._temporal_sequence_cache: TemporalSequence | None = None

    def _scan(self, scan_id: str) -> ScanEntry:
        scan = self.project.scan(scan_id)
        if scan is None:
            raise RegistrationServiceError(f'unknown project scan: {scan_id}')
        return scan

    def resolve_rgb_core(self, scan_id: str) -> ResolvedScanGeometry:
        scan = self._scan(scan_id)
        layer = scan.layer('rgb/core')
        if layer is None:
            raise RegistrationServiceError(f'{scan_id} has no rgb/core layer')
        candidate = Path(layer.path)
        if not candidate.is_absolute():
            if self.project_root is None:
                raise RegistrationServiceError(f'{scan_id} rgb/core is relative but project root is unknown')
            candidate = self.project_root / candidate
        path = candidate.resolve()
        if path.suffix.lower() != '.ply':
            raise RegistrationServiceError(f'{scan_id} rgb/core must be a PLY file')
        if not path.is_file():
            raise RegistrationServiceError(f'{scan_id} rgb/core file does not exist')
        return ResolvedScanGeometry(scan_id=scan_id, path=path, sha256=_sha256(path))

    def _resolve_layer_path(self, scan_id: str, key: str) -> Path | None:
        scan = self._scan(scan_id)
        layer = scan.layer(key)
        if layer is None:
            return None
        candidate = Path(layer.path)
        if not candidate.is_absolute():
            if self.project_root is None:
                return None
            candidate = self.project_root / candidate
        resolved = candidate.resolve()
        return resolved if resolved.is_file() else None

    def resolve_rgb_full_yup(self, scan_id: str) -> Path | None:
        scan = self._scan(scan_id)
        layer = scan.layer('rgb/full')
        if layer is None:
            return None
        source_axis = str(layer.metadata.get('source_axis', ''))
        if source_axis not in {'ARKit_Y_up', 'Y_UP', 'Y-up'}:
            return None
        return self._resolve_layer_path(scan_id, 'rgb/full')

    def temporal_sequence(self) -> TemporalSequence | None:
        if self._temporal_sequence_cache is not None:
            return self._temporal_sequence_cache
        raw_value = self.project.metadata.get('raw_data_root')
        if not isinstance(raw_value, str) or not raw_value.strip():
            return None
        raw_root = Path(raw_value).resolve()
        if not raw_root.is_dir():
            return None
        try:
            sequence = build_temporal_sequence(raw_root, (scan.scan_id for scan in self.project.scan_rail))
        except ValueError as error:
            raise RegistrationServiceError(f'cannot build chronological scan sequence: {error}') from error
        self._temporal_sequence_cache = sequence
        return sequence

    def _temporal_pair(self, job: PairRegistrationJob) -> TemporalPair | None:
        sequence = self.temporal_sequence()
        if sequence is None:
            return None
        pair = sequence.pair(job.source_scan_id, job.target_scan_id)
        if self.search_mode == 'OVERLAP':
            ids = sequence.ordered_scan_ids
            if job.source_scan_id not in ids or job.target_scan_id not in ids or job.source_scan_id == job.target_scan_id:
                raise RegistrationServiceError('overlap needs two distinct known scans')
            runs = {r.scan_id: r for r in sequence.runs}
            gap = (runs[job.target_scan_id].capture_start - runs[job.source_scan_id].capture_end).total_seconds()
            pair = TemporalPair(pair_index=ids.index(job.source_scan_id) * len(ids) + ids.index(job.target_scan_id) + 1, source_scan_id=job.source_scan_id, target_scan_id=job.target_scan_id, gap_s=gap, gap_source='CAPTURE_CLOCK_CONTEXT', gap_median_s=0.0, gap_mad_s=0.0, long_gap_cutoff_s=0.0, long_gap=False, temporal_prior=1.0, endpoint_policy='RGB_WHOLE_RUN_OVERLAP')
        if pair is None:
            if sequence.adjacent_pair(job.source_scan_id, job.target_scan_id) is not None:
                raise RegistrationServiceError('automatic registration direction must be chronological tail-to-head')
            raise RegistrationServiceError('automatic registration requires a chronological neighbouring pair')
        excluded = {scan_id: resolve_alignment_exclusion(self.project, scan_id) for scan_id in (pair.source_scan_id, pair.target_scan_id)}
        blocked = [f'{scan_id}:{reason}' for scan_id, reason in excluded.items() if reason]
        if blocked:
            raise RegistrationServiceError('automatic registration endpoint is excluded: ' + ', '.join(blocked))
        return pair

    @staticmethod
    def _validate_job_contract(job: PairRegistrationJob | PoseGraphJobContract) -> None:
        if not math.isclose(float(job.fixed_scale), FIXED_SCALE, abs_tol=1e-12):
            raise RegistrationServiceError('registration scale must remain 1.0')
        if job.degrees_of_freedom != POSE_GRAPH_DOF:
            raise RegistrationServiceError('registration must remain CAD Z-up 4DOF')

    @staticmethod
    def _verify_hashes(geometries: Mapping[str, ResolvedScanGeometry]) -> dict[str, str]:
        after = {scan_id: _sha256(item.path) for scan_id, item in geometries.items()}
        changed = [scan_id for scan_id, item in geometries.items() if after[scan_id] != item.sha256]
        if changed:
            raise RegistrationServiceError('registration changed source PLY files: ' + ', '.join(sorted(changed)))
        return after

    def _pair_geometry(self, job: PairRegistrationJob) -> tuple[ResolvedScanGeometry, ResolvedScanGeometry]:
        source = self.resolve_rgb_core(job.source_scan_id)
        target = self.resolve_rgb_core(job.target_scan_id)
        if source.path == target.path:
            raise RegistrationServiceError('pair scans resolve to the same PLY file')
        return (source, target)

    def register_pair_hypotheses(self, job: PairRegistrationJob) -> PairHypothesisSet:
        self._validate_job_contract(job)
        temporal_pair = self._temporal_pair(job)
        if temporal_pair is None:
            raise RegistrationServiceError('endpoint hypotheses require a chronological adjacent pair')
        return self._register_pair_hypotheses(job, temporal_pair)

    def prepare_selected_hypothesis(self, job: PairRegistrationJob, pair_set: PairHypothesisSet, hypothesis_id: str) -> PairRegistrationResult:
        self._validate_job_contract(job)
        temporal_pair = self._temporal_pair(job)
        if temporal_pair is None or pair_set.pair != temporal_pair:
            raise RegistrationServiceError('selected hypothesis does not match the current chronological pair')
        sequence = self.temporal_sequence()
        if sequence is None or not pair_set.parent_hashes or pair_set.parent_hashes[0] != sequence.evidence_sha256:
            raise RegistrationServiceError('selected hypothesis has stale order evidence')
        if not pair_set.raw_points_preserved or not _derived_raw_preservation(pair_set):
            raise RegistrationServiceError('selected hypothesis cannot use changed raw evidence')
        selected = next((item for item in pair_set.hypotheses if item.hypothesis_id == hypothesis_id), None)
        if selected is None:
            raise RegistrationServiceError('selected hypothesis is not stored')
        if selected.state is not HypothesisState.ELIGIBLE_FOR_REVIEW:
            raise RegistrationServiceError('only an explicitly eligible Hn can enter Review/Apply')
        prepared = replace(pair_set, legacy_selected_hypothesis_id=hypothesis_id, artifact_sha256='')
        return self._register_temporal_pair(job, prepared)

    def _register_pair_hypotheses(self, job: PairRegistrationJob, temporal_pair: TemporalPair) -> PairHypothesisSet:
        sequence = self.temporal_sequence()
        if sequence is None:
            raise RegistrationServiceError('chronological sequence evidence is missing')
        raw_root = Path(str(self.project.metadata['raw_data_root'])).resolve()
        physical_roots = {run.raw_path.resolve().parent for run in sequence.runs if run.raw_path is not None}
        if len(physical_roots) == 1:
            raw_root = next(iter(physical_roots))
        source_scan = self._scan(job.source_scan_id)
        target_scan = self._scan(job.target_scan_id)
        initial_zup = _trusted_relative_seed(source_scan, target_scan)
        source_core, target_core = self._pair_geometry(job)
        core_geometries = {source_core.scan_id: source_core, target_core.scan_id: target_core}
        source_workset = self.resolve_rgb_full_yup(job.source_scan_id)
        target_workset = self.resolve_rgb_full_yup(job.target_scan_id)
        recovery_models = {}
        for side, scan in (('source', source_scan), ('target', target_scan)):
            record = scan.metadata.get('raw_correction', {})
            if record.get('ray_recovery'):
                from viewer.ray_recovery_service import load_recovery
                path = Path(record['manifest']).parent / 'ray-recovery.json'
                recovered = load_recovery(path, raw_root / scan.scan_id)
                if recovered['provenance']['sha256'] != record['ray_recovery']['sha256']:
                    raise RegistrationServiceError('ray recovery evidence hash changed')
                recovery_models[side] = recovered
        try:
            result = self._temporal_registrar(temporal_pair=temporal_pair, order_evidence_sha256=sequence.evidence_sha256, raw_root=raw_root, source_workset_yup=source_workset, target_workset_yup=target_workset, initial_target_from_source_yup=_zup_to_yup(initial_zup) if initial_zup is not None else None, **{'search_mode': 'OVERLAP'} if self.search_mode == 'OVERLAP' else {}, **{'ray_recoveries': recovery_models} if recovery_models else {})
        except Exception as error:
            self._verify_hashes(core_geometries)
            raise RegistrationServiceError(f'temporal endpoint registration failed: {error}') from error
        self._verify_hashes(core_geometries)
        if not isinstance(result, PairHypothesisSet):
            raise RegistrationServiceError('endpoint registrar must return a PairHypothesisSet')
        if result.pair != temporal_pair:
            raise RegistrationServiceError('endpoint registrar returned the wrong chronological pair')
        if not result.parent_hashes or result.parent_hashes[0] != sequence.evidence_sha256:
            raise RegistrationServiceError('endpoint registrar did not preserve order evidence')
        derived_preserved = _derived_raw_preservation(result)
        if result.raw_points_preserved is not derived_preserved:
            raise RegistrationServiceError('endpoint registrar raw_points_preserved does not match derived hash evidence')
        if not math.isclose(float(result.fixed_scale), FIXED_SCALE, abs_tol=1e-12):
            raise RegistrationServiceError('endpoint registrar changed scan scale')
        return result if derived_preserved else _audit_only_pair_set(result)

    def _clear_edge_cache(self, edge_id: str) -> None:
        self._geometry_results.pop(edge_id, None)
        self._edge_hashes.pop(edge_id, None)
        self._edge_pose_hashes.pop(edge_id, None)
        self._edge_auxiliary_hashes.pop(edge_id, None)

    def register_pair(self, job: PairRegistrationJob) -> PairRegistrationResult:
        self._validate_job_contract(job)
        temporal_pair = self._temporal_pair(job)
        if temporal_pair is not None:
            self._clear_edge_cache(job.edge_id)
            hypotheses = self._register_pair_hypotheses(job, temporal_pair)
            return self._register_temporal_pair(job, hypotheses)
        source, target = self._pair_geometry(job)
        geometries = {source.scan_id: source, target.scan_id: target}
        source_pose = np.asarray(self._scan(source.scan_id).transform, dtype=np.float64)
        target_pose = np.asarray(self._scan(target.scan_id).transform, dtype=np.float64)
        initial_target_from_source = _trusted_relative_seed(self._scan(source.scan_id), self._scan(target.scan_id))
        kwargs: dict[str, object] = {'source_id': source.scan_id, 'target_id': target.scan_id, 'contract': RegistrationContract(units=UNIT_METERS, axis=AXIS_Z_UP, fixed_scale=FIXED_SCALE), 'initial_transform_zup': initial_target_from_source}
        if self.pair_config is not None:
            kwargs['config'] = self.pair_config
        try:
            geometry_result = self._pair_registrar(source.path, target.path, **kwargs)
        except Exception as error:
            self._verify_hashes(geometries)
            raise RegistrationServiceError(f'pair registration failed: {error}') from error
        hashes_after = self._verify_hashes(geometries)
        if _result_field(geometry_result, 'source_id') != source.scan_id:
            raise RegistrationServiceError('pair registrar returned the wrong source id')
        if _result_field(geometry_result, 'target_id') != target.scan_id:
            raise RegistrationServiceError('pair registrar returned the wrong target id')
        if _result_field(geometry_result, 'units') != UNIT_METERS:
            raise RegistrationServiceError('pair registrar returned non-metric evidence')
        if _result_field(geometry_result, 'axis') != AXIS_Z_UP:
            raise RegistrationServiceError('pair registrar returned a non-Z-up transform')
        if not math.isclose(float(_result_field(geometry_result, 'fixed_scale')), FIXED_SCALE, abs_tol=1e-12):
            raise RegistrationServiceError('pair registrar changed scan scale')
        if _result_field(geometry_result, 'raw_points_preserved') is not True:
            raise RegistrationServiceError('pair registrar did not preserve raw evidence')
        quality = _result_field(geometry_result, 'quality')
        rmse = _quality_field(quality, 'inlier_rmse_m', 0.0)
        rmse_m = 0.0 if rmse is None else float(rmse)
        coarse_method = str(_result_field(geometry_result, 'coarse_method')).upper()
        colored_used = bool(_result_field(geometry_result, 'colored_used'))
        method = f'FPFH_{coarse_method}_ICP' + ('_COLOR' if colored_used else '')
        workflow_quality = AlignmentQuality(fitness=float(_quality_field(quality, 'fitness', 0.0)), rmse_m=rmse_m, overlap_ratio=float(_quality_field(quality, 'overlap', 0.0)), correspondence_count=int(_quality_field(quality, 'correspondences', 0)), color_consistency=None, method=method)
        accepted = _result_field(geometry_result, 'accepted') is True
        reasons = tuple((str(reason) for reason in _result_field(geometry_result, 'rejection_reasons')))
        transforms: dict[str, object] = {}
        if accepted:
            target_from_source = np.asarray(_result_field(geometry_result, 'transform_target_from_source_zup'), dtype=np.float64)
            source_global = target_pose @ target_from_source
            transforms = {source.scan_id: source_global.tolist(), target.scan_id: target_pose.tolist()}
        to_dict = getattr(geometry_result, 'to_dict', None)
        geometry_evidence = to_dict() if callable(to_dict) else {'accepted': accepted, 'rejection_reasons': list(reasons)}
        evidence = {'geometry': geometry_evidence, 'source_path': str(source.path), 'target_path': str(target.path), 'source_sha256_before': source.sha256, 'source_sha256_after': hashes_after[source.scan_id], 'target_sha256_before': target.sha256, 'target_sha256_after': hashes_after[target.scan_id], 'raw_points_preserved': True, 'target_scan_held_fixed_in_preview': True}
        self._geometry_results[job.edge_id] = geometry_result
        self._edge_hashes[job.edge_id] = {source.scan_id: source.sha256, target.scan_id: target.sha256}
        self._edge_pose_hashes[job.edge_id] = {source.scan_id: _transform_sha256(source_pose), target.scan_id: _transform_sha256(target_pose)}
        self._edge_auxiliary_hashes[job.edge_id] = ()
        return PairRegistrationResult(edge_id=job.edge_id, transforms=transforms, quality=workflow_quality, eligible_for_apply=accepted, rejection_reasons=reasons, evidence=evidence)

    def _register_temporal_pair(self, job: PairRegistrationJob, pair_set: PairHypothesisSet) -> PairRegistrationResult:
        topology = 'RGB_WHOLE_RUN_OVERLAP' if self.search_mode == 'OVERLAP' else 'CHRONOLOGICAL_NEIGHBOURS_PRIMARY'
        endpoint_policy = 'WHOLE_RUN_SHARED_AREA' if self.search_mode == 'OVERLAP' else 'SOURCE_TAIL_TO_TARGET_HEAD'
        method_prefix = 'OVERLAP' if self.search_mode == 'OVERLAP' else 'TEMPORAL'
        selected = self._legacy_selected_hypothesis(pair_set)
        if selected is None or selected.state is not HypothesisState.ELIGIBLE_FOR_REVIEW:
            reasons = self._legacy_ineligible_reasons(pair_set, selected)
            return PairRegistrationResult(edge_id=job.edge_id, transforms={}, quality=AlignmentQuality(fitness=0.0, rmse_m=0.0, overlap_ratio=0.0, correspondence_count=0, color_consistency=None, method=f'{method_prefix}_RGB_DEPTH_HYPOTHESES_INELIGIBLE'), eligible_for_apply=False, rejection_reasons=reasons, evidence={'hypothesis_set': pair_set.to_dict(), 'legacy_selected_hypothesis_id': selected.hypothesis_id if selected is not None else None, 'raw_points_preserved': pair_set.raw_points_preserved, 'candidate_topology': topology, 'endpoint_policy': endpoint_policy})
        features = selected.features
        transform_value = selected.transform_target_from_source_zup
        if features is None or transform_value is None:
            raise RegistrationServiceError('eligible endpoint hypothesis omitted transform evidence')
        source_scan = self._scan(job.source_scan_id)
        target_scan = self._scan(job.target_scan_id)
        source_pose = np.asarray(source_scan.transform, dtype=np.float64)
        target_pose = np.asarray(target_scan.transform, dtype=np.float64)
        target_from_source_zup = np.asarray(transform_value, dtype=np.float64)
        source_core = self.resolve_rgb_core(job.source_scan_id)
        target_core = self.resolve_rgb_core(job.target_scan_id)
        core_geometries = {source_core.scan_id: source_core, target_core.scan_id: target_core}
        hashes_after = self._verify_hashes(core_geometries)
        source_workset = self.resolve_rgb_full_yup(job.source_scan_id)
        target_workset = self.resolve_rgb_full_yup(job.target_scan_id)
        fitness = float(features.symmetric_fitness_025m if features.symmetric_fitness_025m is not None else features.bidirectional_overlap or 0.0)
        overlap = float(features.bidirectional_overlap if features.bidirectional_overlap is not None else fitness)
        rmse_m = float(features.rmse_m or 0.0)
        correspondences = int(features.correspondences)
        information_scale = max(float(correspondences), 1.0) * max(fitness, 0.01)
        endpoint_result = TemporalEndpointResult(source_id=job.source_scan_id, target_id=job.target_scan_id, accepted=True, rejection_reasons=(), transform_target_from_source_yup=_zup_to_yup(target_from_source_zup), fitness=fitness, rmse_m=rmse_m, overlap=overlap, correspondences=correspondences, information_yup=np.eye(6, dtype=np.float64) * information_scale, evidence={'hypothesis_set_artifact_sha256': pair_set.artifact_sha256, 'legacy_selected_hypothesis_id': selected.hypothesis_id, 'selected_hypothesis': selected.to_dict()})
        evidence = {'temporal_endpoint': endpoint_result.to_dict(), 'hypothesis_set': pair_set.to_dict(), 'legacy_selected_hypothesis_id': selected.hypothesis_id, 'source_path': str(source_core.path), 'target_path': str(target_core.path), 'source_sha256_before': source_core.sha256, 'source_sha256_after': hashes_after[source_core.scan_id], 'target_sha256_before': target_core.sha256, 'target_sha256_after': hashes_after[target_core.scan_id], 'raw_points_preserved': True, 'target_scan_held_fixed_in_preview': True, 'candidate_topology': topology, 'endpoint_policy': endpoint_policy}
        self._geometry_results[job.edge_id] = endpoint_result
        self._edge_hashes[job.edge_id] = {source_core.scan_id: source_core.sha256, target_core.scan_id: target_core.sha256}
        self._edge_pose_hashes[job.edge_id] = {source_scan.scan_id: _transform_sha256(source_pose), target_scan.scan_id: _transform_sha256(target_pose)}
        self._edge_auxiliary_hashes[job.edge_id] = tuple(((path, _sha256(path)) for path in (source_workset, target_workset) if path is not None))
        return PairRegistrationResult(edge_id=job.edge_id, transforms={job.source_scan_id: (target_pose @ target_from_source_zup).tolist(), job.target_scan_id: target_pose.tolist()}, quality=AlignmentQuality(fitness=fitness, rmse_m=rmse_m, overlap_ratio=overlap, correspondence_count=correspondences, color_consistency=features.rgb_matched_to_shuffled_ratio, method=f'{method_prefix}_RGB_DEPTH_HYPOTHESIS_{selected.hypothesis_id}'), eligible_for_apply=True, rejection_reasons=(), evidence=evidence)

    @staticmethod
    def _legacy_selected_hypothesis(pair_set: PairHypothesisSet) -> AlignmentHypothesis | None:
        selected_id = pair_set.legacy_selected_hypothesis_id
        if selected_id is None:
            return None
        return next((item for item in pair_set.hypotheses if item.hypothesis_id == selected_id), None)

    @staticmethod
    def _legacy_ineligible_reasons(pair_set: PairHypothesisSet, selected: AlignmentHypothesis | None) -> tuple[str, ...]:
        if selected is None:
            return ('legacy_selected_hypothesis_missing',)
        if selected.hypothesis_id == 'H0':
            return tuple(dict.fromkeys((*selected.rejection_reasons, 'legacy_selected_hypothesis_is_null')))
        return tuple(dict.fromkeys((*selected.rejection_reasons, 'legacy_selected_hypothesis_rejected')))

    def _pose_graph_edges(self, job: PoseGraphJobContract) -> list[object]:
        edges: list[object] = []
        for edge_id in job.edge_ids:
            geometry_result = self._geometry_results.get(edge_id)
            if geometry_result is None:
                raise RegistrationServiceError(f'pose graph edge has no pair evidence: {edge_id}')
            if _result_field(geometry_result, 'accepted') is not True:
                raise RegistrationServiceError(f'pose graph edge was rejected by geometry: {edge_id}')
            adapter = getattr(geometry_result, 'as_register_edge', None)
            if not callable(adapter):
                raise RegistrationServiceError('pair result has no pose graph adapter')
            edge = adapter()
            if edge is None:
                raise RegistrationServiceError('rejected pair cannot enter pose graph')
            blocked = [scan_id for scan_id in (edge.source, edge.target) if resolve_alignment_exclusion(self.project, scan_id)]
            if blocked:
                raise RegistrationServiceError('pose graph endpoint is excluded: ' + ', '.join(blocked))
            edges.append(edge)
        return edges

    def _verify_pair_evidence_is_current(self, job: PoseGraphJobContract) -> None:
        for edge_id in job.edge_ids:
            expected = self._edge_hashes.get(edge_id)
            expected_poses = self._edge_pose_hashes.get(edge_id)
            expected_auxiliary = self._edge_auxiliary_hashes.get(edge_id)
            if expected is None:
                raise RegistrationServiceError(f'pose graph edge has no source hash evidence: {edge_id}')
            for scan_id, expected_hash in expected.items():
                current = self.resolve_rgb_core(scan_id)
                if current.sha256 != expected_hash:
                    raise RegistrationServiceError(f'pose graph pair evidence is stale for scan: {scan_id}')
            if expected_poses is None:
                raise RegistrationServiceError(f'pose graph edge has no placement hash evidence: {edge_id}')
            for scan_id, expected_hash in expected_poses.items():
                scan = self._scan(scan_id)
                if _transform_sha256(scan.transform) != expected_hash:
                    raise RegistrationServiceError(f'pose graph placement evidence is stale for scan: {scan_id}')
            if expected_auxiliary is None:
                raise RegistrationServiceError(f'pose graph edge has no workset hash evidence: {edge_id}')
            for path, expected_hash in expected_auxiliary:
                if _sha256(path) != expected_hash:
                    raise RegistrationServiceError(f'pose graph endpoint workset evidence is stale: {path.name}')

    def _manifest(self, anchor_scan_id: str, scan_ids: set[str] | None=None) -> Manifest:
        runs: list[RunSpec] = []
        for scan in self.project.scan_rail:
            if scan_ids is not None and scan.scan_id not in scan_ids:
                continue
            geometry = self.resolve_rgb_core(scan.scan_id)
            runs.append(RunSpec(run_id=scan.scan_id, ply_text=str(geometry.path), ply_path=geometry.path, gps_enu_m=np.zeros(3, dtype=np.float64), seed_transform=_zup_to_yup(scan.transform)))
        manifest_path = Path(self.project.source_manifest).resolve() if self.project.source_manifest else (self.project_root or Path.cwd()) / 'cca-points-project.json'
        return Manifest(path=manifest_path, reference_run_id=anchor_scan_id, runs=tuple(runs), pairs=None)

    def optimize_pose_graph(self, job: PoseGraphJobContract) -> PoseGraphResult:
        self._validate_job_contract(job)
        if job.anchor_scan_id not in {scan.scan_id for scan in self.project.scan_rail}:
            raise RegistrationServiceError('pose graph anchor is not in the project')
        exclusions = alignment_exclusions(self.project)
        if job.anchor_scan_id in exclusions:
            raise RegistrationServiceError('pose graph anchor is excluded')
        self._verify_pair_evidence_is_current(job)
        edges = self._pose_graph_edges(job)
        active_ids = {scan.scan_id for scan in self.project.scan_rail}
        if exclusions:
            active_ids = {job.anchor_scan_id}
            while True:
                connected = active_ids | {node for edge in edges if edge.source in active_ids or edge.target in active_ids for node in (edge.source, edge.target)}
                if connected == active_ids:
                    break
                active_ids = connected
            edges = [edge for edge in edges if edge.source in active_ids and edge.target in active_ids]
            if not edges:
                raise RegistrationServiceError('pose graph anchor has no in-scope geometric connection')
        manifest = self._manifest(job.anchor_scan_id, active_ids)
        geometries = {run.run_id: ResolvedScanGeometry(scan_id=run.run_id, path=run.ply_path, sha256=_sha256(run.ply_path)) for run in manifest.runs}
        initial_global = {scan.scan_id: _zup_to_yup(scan.transform) for scan in self.project.scan_rail if scan.scan_id in active_ids}
        try:
            output = self._pose_graph_optimizer(manifest, edges, initial_global, self.pose_graph_config)
        except Exception as error:
            self._verify_hashes(geometries)
            raise RegistrationServiceError(f'pose graph optimization failed: {error}') from error
        hashes_after = self._verify_hashes(geometries)
        if not isinstance(output, tuple) or len(output) != 2 or (not isinstance(output[0], Mapping)) or (not isinstance(output[1], Mapping)):
            raise RegistrationServiceError('pose graph optimizer returned an invalid result')
        optimized_yup, diagnostics = output
        scan_ids = active_ids
        if set(optimized_yup) != scan_ids:
            raise RegistrationServiceError('pose graph optimizer returned incomplete scan poses')
        if diagnostics.get('success') is not True:
            raise RegistrationServiceError('pose graph optimizer did not report success')
        before = diagnostics.get('before')
        after = diagnostics.get('after')
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise RegistrationServiceError('pose graph diagnostics omitted residuals')
        anchor_global_yup = initial_global[job.anchor_scan_id]
        transforms = {scan.scan_id: np.asarray(scan.transform).tolist() for scan in self.project.scan_rail}
        transforms.update({scan_id: _yup_to_zup(anchor_global_yup @ np.asarray(matrix)).tolist() for scan_id, matrix in optimized_yup.items()})
        loop_closures = max(0, len(edges) - (len(scan_ids) - 1))
        return PoseGraphResult(job_id=job.job_id, transforms=transforms, before_rmse_m=float(before.get('translation_rmse_m', 0.0)), after_rmse_m=float(after.get('translation_rmse_m', 0.0)), loop_closure_count=loop_closures, evidence={'optimizer': diagnostics.get('optimizer'), 'diagnostics': dict(diagnostics), 'anchor_scan_id': job.anchor_scan_id, 'anchor_global_pose_preserved': True, 'alignment_exclusions': dict(exclusions), 'unchanged_outside_component': sorted(set(transforms) - active_ids), 'source_sha256_before': {scan_id: item.sha256 for scan_id, item in geometries.items()}, 'source_sha256_after': hashes_after, 'raw_points_preserved': True, 'fixed_scale': FIXED_SCALE, 'degrees_of_freedom': POSE_GRAPH_DOF})
__all__ = ['ProjectRegistrationService', 'RegistrationServiceError', 'ResolvedScanGeometry']
