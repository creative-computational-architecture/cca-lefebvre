
from __future__ import annotations
import copy
import hashlib
import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType, ModuleType
from typing import Any, Protocol, TypeAlias
FIXED_SCALE = 1.0
POSE_GRAPH_DOF = 'Z_UP_4DOF'
Matrix4: TypeAlias = tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]

class AlignmentWorkflowError(ValueError):
    pass

class EdgeStatus(str, Enum):
    CANDIDATE = 'CANDIDATE'
    QUEUED = 'QUEUED'
    RUNNING = 'RUNNING'
    ACCEPTED = 'ACCEPTED'
    REVIEW = 'REVIEW'
    REJECTED = 'REJECTED'
EDGE_COLORS = MappingProxyType({EdgeStatus.CANDIDATE: '#6C6C68', EdgeStatus.QUEUED: '#4C78A8', EdgeStatus.RUNNING: '#E08430', EdgeStatus.REVIEW: '#D6A84B', EdgeStatus.ACCEPTED: '#3F8F63', EdgeStatus.REJECTED: '#C8102E'})

def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlignmentWorkflowError(f'{label} must be numeric')
    output = float(value)
    if not math.isfinite(output):
        raise AlignmentWorkflowError(f'{label} must be finite')
    return output

def _unit_interval(value: object, label: str) -> float:
    output = _finite_number(value, label)
    if not 0.0 <= output <= 1.0:
        raise AlignmentWorkflowError(f'{label} must be between zero and one')
    return output

def _matrix4(value: object, label: str) -> Matrix4:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AlignmentWorkflowError(f'{label} must be a 4x4 matrix')
    if len(value) != 4:
        raise AlignmentWorkflowError(f'{label} must be a 4x4 matrix')
    rows: list[tuple[float, float, float, float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise AlignmentWorkflowError(f'{label} must be a 4x4 matrix')
        if len(row) != 4:
            raise AlignmentWorkflowError(f'{label} must be a 4x4 matrix')
        rows.append(tuple((_finite_number(item, f'{label}[{row_index}][{column_index}]') for column_index, item in enumerate(row))))
    matrix = tuple(rows)
    for actual, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0), strict=True):
        if not math.isclose(actual, expected, abs_tol=1e-08):
            raise AlignmentWorkflowError(f'{label} must be affine')
    rotation = tuple((row[:3] for row in matrix[:3]))
    for first in range(3):
        for second in range(3):
            dot = sum((rotation[row][first] * rotation[row][second] for row in range(3)))
            expected = 1.0 if first == second else 0.0
            if not math.isclose(dot, expected, abs_tol=1e-05):
                raise AlignmentWorkflowError(f'{label} scale must remain 1.0')
    determinant = rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1]) - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0]) + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    if not math.isclose(determinant, 1.0, abs_tol=1e-05):
        raise AlignmentWorkflowError(f'{label} must be a proper rigid transform')
    if any((not math.isclose(actual, expected, abs_tol=1e-05) for actual, expected in zip((rotation[0][2], rotation[1][2], rotation[2][0], rotation[2][1], rotation[2][2]), (0.0, 0.0, 0.0, 0.0, 1.0), strict=True))):
        raise AlignmentWorkflowError(f'{label} must preserve CAD Z-up')
    return matrix

def _matrix_json(matrix: Matrix4) -> list[list[float]]:
    return [list(row) for row in matrix]

def _stable_json_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False, default=str).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()

@dataclass(frozen=True, slots=True)
class AlignmentQuality:
    fitness: float
    rmse_m: float
    overlap_ratio: float
    correspondence_count: int
    color_consistency: float | None = None
    method: str = 'GEOMETRY'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'fitness', _unit_interval(self.fitness, 'fitness'))
        object.__setattr__(self, 'rmse_m', _finite_number(self.rmse_m, 'rmse_m'))
        if self.rmse_m < 0.0:
            raise AlignmentWorkflowError('rmse_m cannot be negative')
        object.__setattr__(self, 'overlap_ratio', _unit_interval(self.overlap_ratio, 'overlap_ratio'))
        if isinstance(self.correspondence_count, bool) or not isinstance(self.correspondence_count, int) or self.correspondence_count < 0:
            raise AlignmentWorkflowError('correspondence_count must be non-negative')
        if self.color_consistency is not None:
            object.__setattr__(self, 'color_consistency', _unit_interval(self.color_consistency, 'color_consistency'))
        if not self.method or not self.method.isascii():
            raise AlignmentWorkflowError('quality method must be non-empty ASCII')

    def summary(self) -> str:
        color = 'n/a' if self.color_consistency is None else f'{self.color_consistency:.2f}'
        return f'fit {self.fitness:.2f} / rmse {self.rmse_m:.3f} m / overlap {self.overlap_ratio:.2f} / color {color}'

    def to_dict(self) -> dict[str, object]:
        return {'fitness': self.fitness, 'rmse_m': self.rmse_m, 'overlap_ratio': self.overlap_ratio, 'correspondence_count': self.correspondence_count, 'color_consistency': self.color_consistency, 'method': self.method}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> AlignmentQuality:
        return cls(fitness=payload.get('fitness', 0.0), rmse_m=payload.get('rmse_m', payload.get('inlier_rmse_m', 0.0)), overlap_ratio=payload.get('overlap_ratio', payload.get('overlap', 0.0)), correspondence_count=payload.get('correspondence_count', 0), color_consistency=payload.get('color_consistency'), method=str(payload.get('method', 'GEOMETRY')))

@dataclass(frozen=True, slots=True)
class PairRegistrationJob:
    job_id: str
    edge_id: str
    source_scan_id: str
    target_scan_id: str
    fixed_scale: float = FIXED_SCALE
    degrees_of_freedom: str = POSE_GRAPH_DOF

    def to_dict(self) -> dict[str, object]:
        return {'job_id': self.job_id, 'edge_id': self.edge_id, 'source_scan_id': self.source_scan_id, 'target_scan_id': self.target_scan_id, 'fixed_scale': self.fixed_scale, 'degrees_of_freedom': self.degrees_of_freedom}

@dataclass(frozen=True, slots=True)
class PairRegistrationResult:
    edge_id: str
    transforms: Mapping[str, Matrix4]
    quality: AlignmentQuality
    eligible_for_apply: bool = True
    rejection_reasons: tuple[str, ...] = ()
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.edge_id:
            raise AlignmentWorkflowError('pair result edge_id is required')
        normalized = {str(scan_id): _matrix4(matrix, f'pair result {scan_id}') for scan_id, matrix in self.transforms.items()}
        if not normalized and self.eligible_for_apply:
            raise AlignmentWorkflowError('pair result must include preview transforms')
        if type(self.eligible_for_apply) is not bool:
            raise AlignmentWorkflowError('eligible_for_apply must be a boolean')
        reasons = tuple((str(reason) for reason in self.rejection_reasons))
        if any((not reason or not reason.isascii() for reason in reasons)):
            raise AlignmentWorkflowError('rejection reasons must be non-empty ASCII')
        if self.eligible_for_apply and reasons:
            raise AlignmentWorkflowError('applicable pair result cannot have rejection reasons')
        if not self.eligible_for_apply and (not reasons):
            raise AlignmentWorkflowError('rejected pair result must explain its rejection')
        object.__setattr__(self, 'transforms', MappingProxyType(normalized))
        object.__setattr__(self, 'rejection_reasons', reasons)
        object.__setattr__(self, 'evidence', MappingProxyType(dict(self.evidence)))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PairRegistrationResult:
        quality = payload.get('quality')
        if not isinstance(quality, AlignmentQuality):
            if not isinstance(quality, Mapping):
                raise AlignmentWorkflowError('pair result quality is required')
            quality = AlignmentQuality.from_mapping(quality)
        transforms = payload.get('transforms', payload.get('preview_transforms', {}))
        if not isinstance(transforms, Mapping):
            raise AlignmentWorkflowError('pair result transforms are required')
        evidence = payload.get('evidence', {})
        if not isinstance(evidence, Mapping):
            raise AlignmentWorkflowError('pair result evidence must be an object')
        return cls(edge_id=str(payload.get('edge_id', '')), transforms=transforms, quality=quality, eligible_for_apply=payload.get('eligible_for_apply', payload.get('accepted', True)), rejection_reasons=tuple(payload.get('rejection_reasons', ())), evidence=evidence)

@dataclass(frozen=True, slots=True)
class PoseGraphJobContract:
    job_id: str
    anchor_scan_id: str
    edge_ids: tuple[str, ...]
    edges: tuple[Mapping[str, object], ...]
    status: EdgeStatus = EdgeStatus.QUEUED
    fixed_scale: float = FIXED_SCALE
    degrees_of_freedom: str = POSE_GRAPH_DOF

    def to_dict(self) -> dict[str, object]:
        return {'job_id': self.job_id, 'anchor_scan_id': self.anchor_scan_id, 'edge_ids': list(self.edge_ids), 'edges': [dict(edge) for edge in self.edges], 'status': self.status.value, 'fixed_scale': self.fixed_scale, 'degrees_of_freedom': self.degrees_of_freedom}

@dataclass(frozen=True, slots=True)
class PoseGraphResult:
    job_id: str
    transforms: Mapping[str, Matrix4]
    before_rmse_m: float
    after_rmse_m: float
    loop_closure_count: int = 0
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_id:
            raise AlignmentWorkflowError('pose graph job_id is required')
        normalized = {str(scan_id): _matrix4(matrix, f'pose graph result {scan_id}') for scan_id, matrix in self.transforms.items()}
        if not normalized:
            raise AlignmentWorkflowError('pose graph result must include transforms')
        object.__setattr__(self, 'transforms', MappingProxyType(normalized))
        before = _finite_number(self.before_rmse_m, 'before_rmse_m')
        after = _finite_number(self.after_rmse_m, 'after_rmse_m')
        if before < 0.0 or after < 0.0:
            raise AlignmentWorkflowError('pose graph RMSE cannot be negative')
        object.__setattr__(self, 'before_rmse_m', before)
        object.__setattr__(self, 'after_rmse_m', after)
        if isinstance(self.loop_closure_count, bool) or not isinstance(self.loop_closure_count, int) or self.loop_closure_count < 0:
            raise AlignmentWorkflowError('loop_closure_count must be non-negative')
        object.__setattr__(self, 'evidence', MappingProxyType(dict(self.evidence)))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PoseGraphResult:
        transforms = payload.get('transforms', payload.get('preview_transforms'))
        if not isinstance(transforms, Mapping):
            raise AlignmentWorkflowError('pose graph transforms are required')
        evidence = payload.get('evidence', {})
        if not isinstance(evidence, Mapping):
            raise AlignmentWorkflowError('pose graph evidence must be an object')
        return cls(job_id=str(payload.get('job_id', '')), transforms=transforms, before_rmse_m=payload.get('before_rmse_m', 0.0), after_rmse_m=payload.get('after_rmse_m', 0.0), loop_closure_count=payload.get('loop_closure_count', 0), evidence=evidence)

class AutoRegistrationService(Protocol):

    def register_pair(self, job: PairRegistrationJob) -> PairRegistrationResult | Mapping[str, object]:
        ...

    def optimize_pose_graph(self, job: PoseGraphJobContract) -> PoseGraphResult | Mapping[str, object]:
        ...

class LazyAutoRegistrationService:

    def __init__(self, import_module: Callable[[str], ModuleType]=importlib.import_module) -> None:
        self._import_module = import_module
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            try:
                self._module = self._import_module('geometry.auto_registration')
            except (ImportError, ModuleNotFoundError) as error:
                raise AlignmentWorkflowError('geometry.auto_registration is not available') from error
        return self._module

    def register_pair(self, job: PairRegistrationJob) -> PairRegistrationResult:
        function = getattr(self._load(), 'register_pair', None)
        if not callable(function):
            raise AlignmentWorkflowError('auto-registration service has no register_pair')
        result = function(job.to_dict())
        if isinstance(result, PairRegistrationResult):
            return result
        if not isinstance(result, Mapping):
            raise AlignmentWorkflowError('register_pair returned an invalid result')
        return PairRegistrationResult.from_mapping(result)

    def optimize_pose_graph(self, job: PoseGraphJobContract) -> PoseGraphResult:
        function = getattr(self._load(), 'optimize_pose_graph', None)
        if not callable(function):
            raise AlignmentWorkflowError('auto-registration service has no optimize_pose_graph')
        result = function(job.to_dict())
        if isinstance(result, PoseGraphResult):
            return result
        if not isinstance(result, Mapping):
            raise AlignmentWorkflowError('optimize_pose_graph returned an invalid result')
        return PoseGraphResult.from_mapping(result)

@dataclass(slots=True)
class AlignmentEdge:
    edge_id: str
    source_scan_id: str
    target_scan_id: str
    rail_index: int
    status: EdgeStatus = EdgeStatus.CANDIDATE
    result: PairRegistrationResult | None = None
    message: str = ''
    applied: bool = False
    preview_project_hash: str | None = None

@dataclass(frozen=True, slots=True)
class RailEdgeView:
    edge_id: str
    source_scan_id: str
    target_scan_id: str
    label: str
    status: str
    color: str
    quality: str
    selected: bool
    applied: bool
    rail_index: int

@dataclass(frozen=True, slots=True)
class AlignmentAudit:
    audit_id: str
    action: str
    project_hash_before: str
    project_hash_after: str
    edge_ids: tuple[str, ...]
    transforms: Mapping[str, Matrix4]
    pose_graph_job_id: str | None
    fixed_scale: float = FIXED_SCALE
    source_state_mutated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {'audit_id': self.audit_id, 'action': self.action, 'project_hash_before': self.project_hash_before, 'project_hash_after': self.project_hash_after, 'edge_ids': list(self.edge_ids), 'transforms': {scan_id: _matrix_json(matrix) for scan_id, matrix in self.transforms.items()}, 'pose_graph_job_id': self.pose_graph_job_id, 'fixed_scale': self.fixed_scale, 'source_state_mutated': self.source_state_mutated}

@dataclass(frozen=True, slots=True)
class AlignmentApplyResult:
    project_state: Mapping[str, object]
    audit: AlignmentAudit

@dataclass(slots=True)
class _UndoFrame:
    project_state: dict[str, Any]
    edge_applied: dict[str, bool]
    pose_graph_applied: bool

def _coerce_pair_result(value: PairRegistrationResult | Mapping[str, object]) -> PairRegistrationResult:
    return value if isinstance(value, PairRegistrationResult) else PairRegistrationResult.from_mapping(value)

def _coerce_pose_result(value: PoseGraphResult | Mapping[str, object]) -> PoseGraphResult:
    return value if isinstance(value, PoseGraphResult) else PoseGraphResult.from_mapping(value)

def _state_scan_rows(state: Mapping[str, object]) -> list[dict[str, Any]]:
    raw = state.get('scan_rail', state.get('scans'))
    if not isinstance(raw, list):
        raise AlignmentWorkflowError('project state must contain a mutable scan rail')
    if not all((isinstance(item, dict) for item in raw)):
        raise AlignmentWorkflowError('project scan rail rows must be objects')
    return raw

def _state_scan_id(row: Mapping[str, object]) -> str:
    value = row.get('scan_id', row.get('id'))
    return value if isinstance(value, str) else ''

def _validate_state_scale(state: Mapping[str, object]) -> None:
    contract = state.get('contracts', {})
    if isinstance(contract, Mapping):
        fixed_scale = contract.get('fixed_scale', FIXED_SCALE)
        scale_locked = contract.get('scale_locked', True)
        if _finite_number(fixed_scale, 'project fixed_scale') != FIXED_SCALE:
            raise AlignmentWorkflowError('project scale must remain 1.0')
        if scale_locked is not True:
            raise AlignmentWorkflowError('project scale must remain locked')
    for row in _state_scan_rows(state):
        row_contract = row.get('contracts', {})
        if isinstance(row_contract, Mapping):
            fixed_scale = row_contract.get('fixed_scale', FIXED_SCALE)
            if _finite_number(fixed_scale, 'scan fixed_scale') != FIXED_SCALE:
                raise AlignmentWorkflowError('scan scale must remain 1.0')

def _apply_transforms_copy(state: Mapping[str, object], transforms: Mapping[str, Matrix4]) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(state))
    _validate_state_scale(candidate)
    rows = _state_scan_rows(candidate)
    by_id = {_state_scan_id(row): row for row in rows}
    if '' in by_id or len(by_id) != len(rows):
        raise AlignmentWorkflowError('project scan ids must be unique and non-empty')
    missing = sorted(set(transforms) - set(by_id))
    if missing:
        raise AlignmentWorkflowError(f"alignment references unknown scans: {', '.join(missing)}")
    for scan_id, matrix in transforms.items():
        by_id[scan_id]['transform'] = _matrix_json(_matrix4(matrix, f'apply {scan_id}'))
        contract = by_id[scan_id].setdefault('contracts', {})
        if not isinstance(contract, dict):
            raise AlignmentWorkflowError('scan contracts must be an object')
        contract['scale_locked'] = True
        contract['fixed_scale'] = FIXED_SCALE
    _validate_state_scale(candidate)
    return candidate

class AlignmentWorkflow:

    def __init__(self, scan_ids: Sequence[str], project_state: Mapping[str, object], service: AutoRegistrationService | None=None) -> None:
        normalized = tuple(scan_ids)
        if len(normalized) < 2:
            raise AlignmentWorkflowError('alignment needs at least two scans')
        if any((not isinstance(scan_id, str) or not scan_id for scan_id in normalized)):
            raise AlignmentWorkflowError('scan ids must be non-empty text')
        if len(set(normalized)) != len(normalized):
            raise AlignmentWorkflowError('scan ids must be unique')
        original_rows = _state_scan_rows(project_state)
        if {_state_scan_id(row) for row in original_rows} != set(normalized):
            raise AlignmentWorkflowError('scan ids must match the project scan rail')
        _validate_state_scale(project_state)
        self.scan_ids = normalized
        self._source_project_state = project_state
        self._project_state = copy.deepcopy(dict(project_state))
        if service is None:
            raise AlignmentWorkflowError('alignment requires an explicit project registration service')
        self.service = service
        self._source_state_hash = _stable_json_hash(project_state)
        self._service_project_hash = _stable_json_hash(self._project_state)
        self.edges: dict[str, AlignmentEdge] = {}
        self.selected_edge_id: str | None = None
        self.pose_graph_job: PoseGraphJobContract | None = None
        self.pose_graph_result: PoseGraphResult | None = None
        self.pose_graph_applied = False
        self._pose_graph_preview_hash: str | None = None
        self._undo_stack: list[_UndoFrame] = []
        self._audit_sequence = 0
        self.audits: list[AlignmentAudit] = []

    @property
    def project_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._project_state)

    def selected_edge(self) -> AlignmentEdge:
        return self._edge()

    def _current_project_hash(self) -> str:
        return _stable_json_hash(self._project_state)

    def _guard_source_state(self) -> None:
        if _stable_json_hash(self._source_project_state) != self._source_state_hash:
            raise AlignmentWorkflowError('source project state changed outside the alignment workflow')

    def _guard_service_state(self) -> None:
        if self._current_project_hash() != self._service_project_hash:
            raise AlignmentWorkflowError('alignment project changed; create a new workflow before another pair')

    def _edge(self, edge_id: str | None=None) -> AlignmentEdge:
        requested = edge_id or self.selected_edge_id
        if requested is None or requested not in self.edges:
            raise AlignmentWorkflowError('select an alignment edge first')
        return self.edges[requested]

    def _invalidate_pose_graph(self) -> None:
        self.pose_graph_job = None
        self.pose_graph_result = None
        self.pose_graph_applied = False
        self._pose_graph_preview_hash = None

    def select_pair(self, source_scan_id: str, target_scan_id: str) -> AlignmentEdge:
        self._guard_source_state()
        self._guard_service_state()
        if source_scan_id == target_scan_id:
            raise AlignmentWorkflowError('alignment pair must contain two scans')
        if source_scan_id not in self.scan_ids or target_scan_id not in self.scan_ids:
            raise AlignmentWorkflowError('alignment pair contains an unknown scan')
        pair = frozenset((source_scan_id, target_scan_id))
        existing = next((edge for edge in self.edges.values() if frozenset((edge.source_scan_id, edge.target_scan_id)) == pair), None)
        if existing is None:
            index = len(self.edges)
            edge_id = f'edge-{index + 1:04d}'
            existing = AlignmentEdge(edge_id=edge_id, source_scan_id=source_scan_id, target_scan_id=target_scan_id, rail_index=index)
            self.edges[edge_id] = existing
        self.selected_edge_id = existing.edge_id
        return existing

    def queue(self, edge_id: str | None=None) -> AlignmentEdge:
        edge = self._edge(edge_id)
        if edge.status is not EdgeStatus.CANDIDATE:
            raise AlignmentWorkflowError('only candidate edges can be queued')
        edge.status = EdgeStatus.QUEUED
        edge.message = 'Queued for pair registration'
        return edge

    def run_next(self) -> AlignmentEdge | None:
        self._guard_source_state()
        self._guard_service_state()
        edge = next((item for item in self.edges.values() if item.status is EdgeStatus.QUEUED), None)
        if edge is None:
            return None
        edge.status = EdgeStatus.RUNNING
        edge.message = 'Pair registration is running'
        job = PairRegistrationJob(job_id=f'pair-job-{edge.rail_index + 1:04d}', edge_id=edge.edge_id, source_scan_id=edge.source_scan_id, target_scan_id=edge.target_scan_id)
        preview_hash = self._current_project_hash()
        try:
            result = _coerce_pair_result(self.service.register_pair(job))
            if result.edge_id != edge.edge_id:
                raise AlignmentWorkflowError('pair result edge_id does not match its job')
            if not set(result.transforms).issubset({edge.source_scan_id, edge.target_scan_id}):
                raise AlignmentWorkflowError('pair result contains an unrelated scan')
        except Exception as error:
            edge.status = EdgeStatus.REVIEW
            edge.result = None
            edge.message = f'Registration failed: {error}'
            edge.preview_project_hash = None
            return edge
        edge.result = result
        edge.preview_project_hash = preview_hash
        edge.status = EdgeStatus.REVIEW
        edge.message = 'Preview ready for operator review' if result.eligible_for_apply else 'Geometry rejected: ' + ', '.join(result.rejection_reasons)
        return edge

    def attach_pair_result(self, edge_id: str, result: PairRegistrationResult | Mapping[str, object]) -> AlignmentEdge:
        self._guard_source_state()
        self._guard_service_state()
        edge = self._edge(edge_id)
        if edge.status in {EdgeStatus.QUEUED, EdgeStatus.RUNNING, EdgeStatus.ACCEPTED}:
            raise AlignmentWorkflowError('selected variation cannot replace the current edge state')
        prepared = _coerce_pair_result(result)
        if prepared.edge_id != edge.edge_id:
            raise AlignmentWorkflowError('pair result edge_id does not match selection')
        if not set(prepared.transforms).issubset({edge.source_scan_id, edge.target_scan_id}):
            raise AlignmentWorkflowError('pair result contains an unrelated scan')
        edge.result = prepared
        edge.preview_project_hash = self._current_project_hash()
        edge.status = EdgeStatus.REVIEW
        edge.applied = False
        edge.message = 'Selected variation ready for operator review' if prepared.eligible_for_apply else 'Selected variation is not eligible for review'
        self._invalidate_pose_graph()
        return edge

    def accept(self, edge_id: str | None=None) -> AlignmentEdge:
        self._guard_source_state()
        edge = self._edge(edge_id)
        if edge.status is not EdgeStatus.REVIEW or edge.result is None:
            raise AlignmentWorkflowError('only a completed review preview can be accepted')
        if not edge.result.eligible_for_apply:
            raise AlignmentWorkflowError('rejected registration cannot be accepted')
        if edge.preview_project_hash != self._current_project_hash():
            raise AlignmentWorkflowError('stale alignment preview cannot be accepted')
        edge.status = EdgeStatus.ACCEPTED
        edge.message = 'Accepted for pose graph'
        self._invalidate_pose_graph()
        return edge

    def reject(self, edge_id: str | None=None, reason: str='Operator rejected') -> AlignmentEdge:
        edge = self._edge(edge_id)
        if edge.status is EdgeStatus.RUNNING:
            raise AlignmentWorkflowError('a running registration cannot be rejected synchronously')
        edge.status = EdgeStatus.REJECTED
        edge.applied = False
        edge.message = reason if reason.isascii() else 'Operator rejected'
        self._invalidate_pose_graph()
        return edge

    def cancel(self, edge_id: str | None=None) -> AlignmentEdge:
        edge = self._edge(edge_id)
        if edge.status is EdgeStatus.RUNNING:
            raise AlignmentWorkflowError('a running registration needs service cancellation')
        edge.status = EdgeStatus.CANDIDATE
        edge.result = None
        edge.preview_project_hash = None
        edge.message = 'Preview canceled'
        edge.applied = False
        self._invalidate_pose_graph()
        return edge

    def rail_records(self) -> tuple[RailEdgeView, ...]:
        return tuple((RailEdgeView(edge_id=edge.edge_id, source_scan_id=edge.source_scan_id, target_scan_id=edge.target_scan_id, label=f'{edge.source_scan_id} -> {edge.target_scan_id}', status=edge.status.value, color=EDGE_COLORS[edge.status], quality='REJECTED / ' + ', '.join(edge.result.rejection_reasons) + ' / ' + edge.result.quality.summary() if edge.result is not None and (not edge.result.eligible_for_apply) else edge.result.quality.summary() if edge.result is not None else edge.message, selected=edge.edge_id == self.selected_edge_id, applied=edge.applied, rail_index=edge.rail_index) for edge in sorted(self.edges.values(), key=lambda item: item.rail_index)))

    def create_pose_graph_job(self, anchor_scan_id: str) -> PoseGraphJobContract:
        self._guard_source_state()
        self._guard_service_state()
        if anchor_scan_id not in self.scan_ids:
            raise AlignmentWorkflowError('pose graph anchor is not in the project')
        from types import SimpleNamespace
        from viewer.alignment_scope import alignment_exclusions
        exclusions = alignment_exclusions(SimpleNamespace(scans=[SimpleNamespace(scan_id=scan_id) for scan_id in self.scan_ids], metadata=self._project_state.get('metadata', {})))
        if anchor_scan_id in exclusions:
            raise AlignmentWorkflowError('pose graph anchor is excluded')
        accepted = [edge for edge in self.edges.values() if edge.status is EdgeStatus.ACCEPTED and edge.result is not None]
        if not accepted:
            raise AlignmentWorkflowError('pose graph needs at least one accepted edge')
        if any((edge.source_scan_id in exclusions or edge.target_scan_id in exclusions for edge in accepted)):
            raise AlignmentWorkflowError('pose graph contains an excluded endpoint')
        current_hash = self._current_project_hash()
        if any((edge.preview_project_hash != current_hash for edge in accepted)):
            raise AlignmentWorkflowError('pose graph contains a stale pair preview')
        connected = {anchor_scan_id}
        changed = True
        while changed:
            changed = False
            for edge in accepted:
                pair = {edge.source_scan_id, edge.target_scan_id}
                if connected & pair and (not pair <= connected):
                    connected.update(pair)
                    changed = True
        missing = sorted(set(self.scan_ids) - connected)
        if missing and (not exclusions):
            raise AlignmentWorkflowError('pose graph is disconnected; missing scans: ' + ', '.join(missing))
        if exclusions:
            accepted = [edge for edge in accepted if edge.source_scan_id in connected and edge.target_scan_id in connected]
            if not accepted:
                raise AlignmentWorkflowError('pose graph anchor has no in-scope geometric connection')
        edges = tuple((MappingProxyType({'edge_id': edge.edge_id, 'source_scan_id': edge.source_scan_id, 'target_scan_id': edge.target_scan_id, 'quality': edge.result.quality.to_dict(), 'transforms': {scan_id: _matrix_json(matrix) for scan_id, matrix in edge.result.transforms.items()}}) for edge in accepted))
        self.pose_graph_job = PoseGraphJobContract(job_id=f'pose-graph-{len(self.audits) + 1:04d}', anchor_scan_id=anchor_scan_id, edge_ids=tuple((edge.edge_id for edge in accepted)), edges=edges)
        self.pose_graph_result = None
        self.pose_graph_applied = False
        self._pose_graph_preview_hash = None
        return self.pose_graph_job

    def run_pose_graph(self) -> PoseGraphJobContract:
        self._guard_source_state()
        self._guard_service_state()
        job = self.pose_graph_job
        if job is None or job.status is not EdgeStatus.QUEUED:
            raise AlignmentWorkflowError('queue a pose graph job first')
        running = replace(job, status=EdgeStatus.RUNNING)
        self.pose_graph_job = running
        preview_hash = self._current_project_hash()
        try:
            result = _coerce_pose_result(self.service.optimize_pose_graph(running))
            if result.job_id != running.job_id:
                raise AlignmentWorkflowError('pose graph result job_id does not match')
            if not set(result.transforms).issubset(set(self.scan_ids)):
                raise AlignmentWorkflowError('pose graph result contains an unknown scan')
        except Exception as error:
            self.pose_graph_result = None
            self.pose_graph_job = replace(running, status=EdgeStatus.REVIEW)
            raise AlignmentWorkflowError(f'pose graph failed: {error}') from error
        if preview_hash != self._current_project_hash():
            self.pose_graph_result = None
            self.pose_graph_job = replace(running, status=EdgeStatus.REVIEW)
            raise AlignmentWorkflowError('project changed while pose graph was running')
        self.pose_graph_result = result
        self.pose_graph_job = replace(running, status=EdgeStatus.REVIEW)
        self._pose_graph_preview_hash = preview_hash
        return self.pose_graph_job

    def accept_pose_graph(self) -> PoseGraphJobContract:
        self._guard_source_state()
        job = self.pose_graph_job
        if job is None or job.status is not EdgeStatus.REVIEW or self.pose_graph_result is None:
            raise AlignmentWorkflowError('pose graph preview is not ready for acceptance')
        if self._pose_graph_preview_hash != self._current_project_hash():
            raise AlignmentWorkflowError('stale pose graph preview cannot be accepted')
        self.pose_graph_job = replace(job, status=EdgeStatus.ACCEPTED)
        return self.pose_graph_job

    def cancel_pose_graph(self) -> None:
        if self.pose_graph_job is not None and self.pose_graph_job.status is EdgeStatus.RUNNING:
            raise AlignmentWorkflowError('a running pose graph needs service cancellation')
        self._invalidate_pose_graph()

    def _accepted_transforms(self, edge_ids: Sequence[str] | None) -> tuple[dict[str, Matrix4], tuple[str, ...], str | None]:
        current_hash = self._current_project_hash()
        if edge_ids is None and (self.pose_graph_job is not None and self.pose_graph_job.status is EdgeStatus.ACCEPTED and (self.pose_graph_result is not None)):
            if self._pose_graph_preview_hash != current_hash:
                raise AlignmentWorkflowError('stale pose graph preview cannot be applied')
            return (dict(self.pose_graph_result.transforms), self.pose_graph_job.edge_ids, self.pose_graph_job.job_id)
        requested = tuple(edge_ids) if edge_ids is not None else tuple((edge.edge_id for edge in self.edges.values() if edge.status is EdgeStatus.ACCEPTED and (not edge.applied)))
        if not requested:
            raise AlignmentWorkflowError('there is no accepted alignment to apply')
        transforms: dict[str, Matrix4] = {}
        for edge_id in requested:
            edge = self._edge(edge_id)
            if edge.status is not EdgeStatus.ACCEPTED or edge.result is None:
                raise AlignmentWorkflowError('all applied edges must be accepted')
            if edge.preview_project_hash != current_hash:
                raise AlignmentWorkflowError('stale alignment preview cannot be applied')
            for scan_id, matrix in edge.result.transforms.items():
                previous = transforms.get(scan_id)
                if previous is not None and previous != matrix:
                    raise AlignmentWorkflowError('accepted pair previews conflict; run the pose graph before apply')
                transforms[scan_id] = matrix
        return (transforms, requested, None)

    def apply(self, edge_ids: Sequence[str] | None=None) -> AlignmentApplyResult:
        self._guard_source_state()
        transforms, applied_edges, pose_job_id = self._accepted_transforms(edge_ids)
        before = copy.deepcopy(self._project_state)
        before_hash = _stable_json_hash(before)
        candidate = _apply_transforms_copy(before, transforms)
        after_hash = _stable_json_hash(candidate)
        self._audit_sequence += 1
        audit = AlignmentAudit(audit_id=f'alignment-apply-{self._audit_sequence:04d}', action='APPLY_ALIGNMENT', project_hash_before=before_hash, project_hash_after=after_hash, edge_ids=tuple(applied_edges), transforms=MappingProxyType(dict(transforms)), pose_graph_job_id=pose_job_id)
        frame = _UndoFrame(project_state=before, edge_applied={edge_id: edge.applied for edge_id, edge in self.edges.items()}, pose_graph_applied=self.pose_graph_applied)
        self._undo_stack.append(frame)
        self._project_state = candidate
        for edge_id in applied_edges:
            self.edges[edge_id].applied = True
        if pose_job_id is not None:
            self.pose_graph_applied = True
        self.audits.append(audit)
        return AlignmentApplyResult(copy.deepcopy(candidate), audit)

    def undo(self) -> AlignmentApplyResult:
        self._guard_source_state()
        if not self._undo_stack:
            raise AlignmentWorkflowError('there is no alignment apply to undo')
        current = copy.deepcopy(self._project_state)
        frame = self._undo_stack.pop()
        restored = copy.deepcopy(frame.project_state)
        self._audit_sequence += 1
        audit = AlignmentAudit(audit_id=f'alignment-undo-{self._audit_sequence:04d}', action='UNDO_ALIGNMENT', project_hash_before=_stable_json_hash(current), project_hash_after=_stable_json_hash(restored), edge_ids=tuple((edge_id for edge_id, edge in self.edges.items() if edge.applied and (not frame.edge_applied.get(edge_id, False)))), transforms=MappingProxyType({}), pose_graph_job_id=self.pose_graph_job.job_id if self.pose_graph_applied and self.pose_graph_job is not None else None)
        self._project_state = restored
        for edge_id, applied in frame.edge_applied.items():
            self.edges[edge_id].applied = applied
        self.pose_graph_applied = frame.pose_graph_applied
        self.audits.append(audit)
        return AlignmentApplyResult(copy.deepcopy(restored), audit)
__all__ = ['EDGE_COLORS', 'FIXED_SCALE', 'POSE_GRAPH_DOF', 'AlignmentApplyResult', 'AlignmentAudit', 'AlignmentEdge', 'AlignmentQuality', 'AlignmentWorkflow', 'AlignmentWorkflowError', 'AutoRegistrationService', 'EdgeStatus', 'LazyAutoRegistrationService', 'Matrix4', 'PairRegistrationJob', 'PairRegistrationResult', 'PoseGraphJobContract', 'PoseGraphResult', 'RailEdgeView']
