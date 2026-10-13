
from __future__ import annotations
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias
import numpy as np
from geometry.temporal_sequence import TemporalPair
Matrix4: TypeAlias = tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]
SCHEMA = 'cca.alignment-hypotheses.v1'
_TOLERANCE = 1e-05
_SHA256_PATTERN = re.compile('^[0-9a-f]{64}$')
_B_YUP_TO_ZUP = np.array(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
_B_ZUP_TO_YUP = _B_YUP_TO_ZUP.T

class HypothesisState(str, Enum):
    NULL = 'NULL'
    ELIGIBLE_FOR_REVIEW = 'ELIGIBLE_FOR_REVIEW'
    REJECTED = 'REJECTED'

@dataclass(frozen=True, slots=True)
class HypothesisFeatures:
    independent_frame_pairs: int
    image_inliers: int
    image_inlier_ratio: float | None
    image_grid_cells: int | None
    image_span_fraction: float | None
    depth_inliers: int
    depth_inlier_ratio: float | None
    residual_median_m: float | None
    residual_p90_m: float | None
    inlier_span_m: float | None
    evaluation_voxel_m: float | None
    bidirectional_overlap: float | None
    symmetric_fitness_025m: float | None
    rmse_m: float | None
    correspondences: int
    normal_agreement: float | None
    icp_translation_correction_m: float | None
    icp_yaw_correction_deg: float | None
    rgb_matched_to_shuffled_ratio: float | None
    confidence_support: float | None
    inverse_translation_error_m: float | None
    inverse_yaw_error_deg: float | None
    rerun_agreement: float | None
    bootstrap_yaw_spread_deg: float | None
    bootstrap_translation_spread_m: float | None

    def __post_init__(self) -> None:
        _validate_nonnegative_ints(self, 'independent_frame_pairs', 'image_inliers', 'image_grid_cells', 'depth_inliers', 'correspondences')
        _validate_fraction_fields(self, 'image_inlier_ratio', 'image_span_fraction', 'depth_inlier_ratio', 'bidirectional_overlap', 'symmetric_fitness_025m', 'normal_agreement', 'confidence_support', 'rerun_agreement')
        _validate_nonnegative_float_fields(self, 'residual_median_m', 'residual_p90_m', 'inlier_span_m', 'rmse_m', 'icp_translation_correction_m', 'icp_yaw_correction_deg', 'rgb_matched_to_shuffled_ratio', 'inverse_translation_error_m', 'inverse_yaw_error_deg', 'bootstrap_yaw_spread_deg', 'bootstrap_translation_spread_m')
        if self.evaluation_voxel_m is not None:
            _validate_finite_float(self.evaluation_voxel_m, 'evaluation_voxel_m')
            if self.evaluation_voxel_m <= 0.0:
                raise ValueError('evaluation_voxel_m must be greater than zero')

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

@dataclass(frozen=True, slots=True)
class AlignmentHypothesis:
    hypothesis_id: str
    pair_id: str
    source_scan_id: str
    target_scan_id: str
    state: HypothesisState
    transform_target_from_source_zup: Matrix4 | None
    features: HypothesisFeatures | None
    rejection_reasons: tuple[str, ...]
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        for label in ('hypothesis_id', 'pair_id', 'source_scan_id', 'target_scan_id'):
            value = getattr(self, label)
            if not isinstance(value, str) or not value:
                raise ValueError(f'{label} must be a non-empty string')
        if not isinstance(self.state, HypothesisState):
            raise ValueError('state must be a HypothesisState')
        reasons = _normalise_reasons(self.rejection_reasons)
        object.__setattr__(self, 'rejection_reasons', reasons)
        if not isinstance(self.evidence, Mapping):
            raise ValueError('evidence must be a mapping')
        object.__setattr__(self, 'evidence', _freeze_json_mapping(self.evidence, 'evidence'))
        if self.features is not None and (not isinstance(self.features, HypothesisFeatures)):
            raise ValueError('features must be HypothesisFeatures or None')

    def to_dict(self) -> dict[str, Any]:
        return {'hypothesis_id': self.hypothesis_id, 'pair_id': self.pair_id, 'source_scan_id': self.source_scan_id, 'target_scan_id': self.target_scan_id, 'state': self.state.value, 'transform_target_from_source_zup': [list(row) for row in self.transform_target_from_source_zup] if self.transform_target_from_source_zup is not None else None, 'features': self.features.to_dict() if self.features is not None else None, 'rejection_reasons': list(self.rejection_reasons), 'evidence': _thaw_json_mapping(self.evidence, 'evidence')}

@dataclass(frozen=True, slots=True)
class PairHypothesisSet:
    pair: TemporalPair
    hypotheses: tuple[AlignmentHypothesis, ...]
    source_hashes: Mapping[str, str | None]
    target_hashes: Mapping[str, str | None]
    parent_hashes: tuple[str, ...]
    legacy_selected_hypothesis_id: str | None = None
    artifact_sha256: str = ''
    raw_points_preserved: bool = True
    fixed_scale: float = 1.0
    axis: str = 'CAD_Z_up'

    def __post_init__(self) -> None:
        _validate_temporal_pair(self.pair)
        hypotheses = tuple(self.hypotheses)
        if not hypotheses:
            raise ValueError('hypothesis set requires H0 first')
        object.__setattr__(self, 'hypotheses', hypotheses)
        for index, hypothesis in enumerate(hypotheses):
            if not isinstance(hypothesis, AlignmentHypothesis):
                raise ValueError('hypotheses must contain AlignmentHypothesis values')
            expected_id = f'H{index}'
            if hypothesis.hypothesis_id != expected_id:
                if index == 0:
                    raise ValueError('hypothesis set requires H0 first')
                raise ValueError('hypothesis ids must be unique and sequential H1 through Hn')
            self._validate_hypothesis(hypothesis, is_null=index == 0)
        source_hashes = _normalise_hash_mapping(self.source_hashes, 'source_hashes')
        target_hashes = _normalise_hash_mapping(self.target_hashes, 'target_hashes')
        parent_hashes = _normalise_parent_hashes(self.parent_hashes)
        object.__setattr__(self, 'source_hashes', source_hashes)
        object.__setattr__(self, 'target_hashes', target_hashes)
        object.__setattr__(self, 'parent_hashes', parent_hashes)
        _validate_contract_scalars(self)
        selected = self.legacy_selected_hypothesis_id
        if selected is not None and (not isinstance(selected, str) or selected not in {item.hypothesis_id for item in hypotheses}):
            raise ValueError('legacy_selected_hypothesis_id must name a stored H0 or Hn')
        computed = self._computed_artifact_sha256()
        supplied = self.artifact_sha256
        if supplied == '':
            object.__setattr__(self, 'artifact_sha256', computed)
        elif not isinstance(supplied, str) or not _SHA256_PATTERN.fullmatch(supplied):
            raise ValueError('artifact_sha256 must be a lowercase SHA-256 digest')
        elif supplied != computed:
            raise ValueError('artifact_sha256 does not match canonical artifact payload')

    def _validate_hypothesis(self, hypothesis: AlignmentHypothesis, *, is_null: bool) -> None:
        if hypothesis.pair_id != self.pair.pair_id or hypothesis.source_scan_id != self.pair.source_scan_id or hypothesis.target_scan_id != self.pair.target_scan_id:
            raise ValueError('hypothesis pair direction must exactly match the TemporalPair')
        if is_null:
            if hypothesis.state is not HypothesisState.NULL:
                raise ValueError('H0 must have NULL state')
            if hypothesis.transform_target_from_source_zup is not None:
                raise ValueError('NULL hypothesis has no transform')
            if hypothesis.features is not None:
                raise ValueError('H0 has no evidence features')
            return
        if hypothesis.state is HypothesisState.NULL:
            raise ValueError('only H0 may have NULL state')
        if hypothesis.state is HypothesisState.ELIGIBLE_FOR_REVIEW:
            if hypothesis.transform_target_from_source_zup is None:
                raise ValueError('eligible hypothesis requires a rigid transform')
            if hypothesis.rejection_reasons:
                raise ValueError('eligible hypothesis cannot have rejection reasons')
        elif hypothesis.state is HypothesisState.REJECTED:
            if not hypothesis.rejection_reasons:
                raise ValueError('rejected hypothesis requires rejection reasons')
        else:
            raise ValueError('non-null hypothesis has an unsupported state')
        transform = hypothesis.transform_target_from_source_zup
        if transform is not None:
            object.__setattr__(hypothesis, 'transform_target_from_source_zup', validate_zup_rigid_transform(transform, label=f'{hypothesis.hypothesis_id} transform'))
            if hypothesis.features is None:
                raise ValueError('stored valid transform requires evidence features')
        elif hypothesis.features is not None:
            raise ValueError('hypothesis without a transform cannot retain features')

    def _artifact_payload(self) -> dict[str, Any]:
        return {'schema': SCHEMA, 'pair': self.pair.to_dict(), 'hypotheses': [item.to_dict() for item in self.hypotheses], 'source_hashes': dict(self.source_hashes), 'target_hashes': dict(self.target_hashes), 'parent_hashes': list(self.parent_hashes), 'legacy_selected_hypothesis_id': self.legacy_selected_hypothesis_id, 'raw_points_preserved': self.raw_points_preserved, 'fixed_scale': self.fixed_scale, 'axis': self.axis}

    def _computed_artifact_sha256(self) -> str:
        encoded = json.dumps(self._artifact_payload(), ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self._artifact_payload()
        payload['artifact_sha256'] = self.artifact_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PairHypothesisSet:
        if not isinstance(payload, Mapping):
            raise ValueError('pair hypothesis payload must be a mapping')
        if payload.get('schema') != SCHEMA:
            raise ValueError(f"unsupported alignment hypothesis schema: {payload.get('schema')!r}")
        if 'artifact_sha256' not in payload:
            raise ValueError('artifact_sha256 is required for deserialization')
        if 'raw_points_preserved' not in payload:
            raise ValueError('raw_points_preserved is required for deserialization')
        pair_payload = _required_mapping(payload, 'pair')
        pair = _temporal_pair_from_dict(pair_payload)
        serialized_hypotheses = _required_sequence(payload, 'hypotheses')
        hypotheses = tuple((_hypothesis_from_dict(item) for item in serialized_hypotheses))
        return cls(pair=pair, hypotheses=hypotheses, source_hashes=_required_mapping(payload, 'source_hashes'), target_hashes=_required_mapping(payload, 'target_hashes'), parent_hashes=tuple(_required_sequence(payload, 'parent_hashes')), legacy_selected_hypothesis_id=payload.get('legacy_selected_hypothesis_id'), artifact_sha256=payload['artifact_sha256'], raw_points_preserved=payload['raw_points_preserved'], fixed_scale=payload.get('fixed_scale', 1.0), axis=payload.get('axis', 'CAD_Z_up'))

def make_null_hypothesis(pair: TemporalPair, reasons: Sequence[str]=()) -> AlignmentHypothesis:
    return AlignmentHypothesis(hypothesis_id='H0', pair_id=pair.pair_id, source_scan_id=pair.source_scan_id, target_scan_id=pair.target_scan_id, state=HypothesisState.NULL, transform_target_from_source_zup=None, features=None, rejection_reasons=reasons, evidence={})

def validate_zup_rigid_transform(value: object, *, label: str) -> Matrix4:
    return _validate_rigid_transform(value, label=label, vertical_axis=2)

def yup_transform_to_zup(transform: object) -> Matrix4:
    source = _validate_rigid_transform(transform, label='Y-up transform', vertical_axis=1)
    converted = _B_YUP_TO_ZUP @ np.asarray(source) @ _B_ZUP_TO_YUP
    return validate_zup_rigid_transform(converted, label='converted Z-up transform')

def zup_transform_to_yup(transform: object) -> np.ndarray:
    source = validate_zup_rigid_transform(transform, label='Z-up transform')
    converted = _B_ZUP_TO_YUP @ np.asarray(source) @ _B_YUP_TO_ZUP
    _validate_rigid_transform(converted, label='converted Y-up transform', vertical_axis=1)
    return converted

def _validate_rigid_transform(value: object, *, label: str, vertical_axis: int) -> Matrix4:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be a numeric 4x4 matrix') from exc
    if matrix.shape != (4, 4):
        raise ValueError(f'{label} must be a 4x4 matrix')
    if not np.isfinite(matrix).all():
        raise ValueError(f'{label} must contain only finite values')
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=_TOLERANCE, rtol=0.0):
        raise ValueError(f'{label} must be affine with bottom row [0, 0, 0, 1]')
    rotation = matrix[:3, :3]
    scales = np.linalg.norm(rotation, axis=0)
    if not np.allclose(scales, 1.0, atol=_TOLERANCE, rtol=0.0):
        raise ValueError(f'{label} must have fixed scale 1.0')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_TOLERANCE, rtol=0.0):
        raise ValueError(f'{label} rotation must be orthonormal')
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=_TOLERANCE, rel_tol=0.0):
        raise ValueError(f'{label} rotation determinant must be +1')
    vertical = np.zeros(3)
    vertical[vertical_axis] = 1.0
    if not np.allclose(rotation @ vertical, vertical, atol=_TOLERANCE, rtol=0.0):
        raise ValueError(f'{label} must preserve the world vertical axis')
    return tuple((tuple((float(entry) for entry in row)) for row in matrix))

def _validate_nonnegative_ints(record: HypothesisFeatures, *names: str) -> None:
    for name in names:
        value = getattr(record, name)
        if value is None and name == 'image_grid_cells':
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a non-negative integer or None')

def _validate_fraction_fields(record: HypothesisFeatures, *names: str) -> None:
    for name in names:
        value = getattr(record, name)
        if value is None:
            continue
        _validate_finite_float(value, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'{name} must be within [0, 1]')

def _validate_nonnegative_float_fields(record: HypothesisFeatures, *names: str) -> None:
    for name in names:
        value = getattr(record, name)
        if value is None:
            continue
        _validate_finite_float(value, name)
        if value < 0.0:
            raise ValueError(f'{name} must be non-negative')

def _validate_finite_float(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)):
        raise ValueError(f'{label} must be finite')

def _normalise_reasons(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError('rejection_reasons must be a sequence of non-empty strings')
    reasons = tuple(value)
    if any((not isinstance(item, str) or not item for item in reasons)):
        raise ValueError('rejection_reasons must be a sequence of non-empty strings')
    return reasons

def _freeze_json_mapping(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    frozen = _freeze_json_value(value, label)
    if not isinstance(frozen, Mapping):
        raise ValueError(f'{label} must be a mapping')
    return frozen

def _freeze_json_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f'{label} cannot contain non-finite numbers')
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f'{label} mapping keys must be strings')
            output[key] = _freeze_json_value(item, f'{label}.{key}')
        return MappingProxyType(output)
    if isinstance(value, (list, tuple)):
        return tuple((_freeze_json_value(item, label) for item in value))
    raise ValueError(f'{label} must contain only JSON values')

def _thaw_json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    thawed = _thaw_json_value(value, label)
    if not isinstance(thawed, dict):
        raise ValueError(f'{label} must be a mapping')
    return thawed

def _thaw_json_value(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item, f'{label}.{key}') for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item, label) for item in value]
    return value

def _normalise_hash_mapping(value: object, label: str) -> Mapping[str, str | None]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be a mapping')
    output: dict[str, str | None] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f'{label} keys must be non-empty strings')
        if item is not None and (not isinstance(item, str) or not _SHA256_PATTERN.fullmatch(item)):
            raise ValueError(f'{label} values must be lowercase SHA-256 digests or None')
        output[key] = item
    return MappingProxyType(output)

def _normalise_parent_hashes(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError('parent_hashes must be an ordered sequence')
    hashes = tuple(value)
    for item in hashes:
        if item != 'MISSING' and (not isinstance(item, str) or not _SHA256_PATTERN.fullmatch(item)):
            raise ValueError('parent_hashes must contain lowercase SHA-256 digests or MISSING')
    return hashes

def _validate_temporal_pair(pair: object) -> None:
    if not isinstance(pair, TemporalPair):
        raise ValueError('pair must be a TemporalPair')
    if not isinstance(pair.temporal_prior, (int, float)) or isinstance(pair.temporal_prior, bool):
        raise ValueError('pair temporal_prior must be finite within [0, 1]')
    if not math.isfinite(pair.temporal_prior) or not 0.0 <= pair.temporal_prior <= 1.0:
        raise ValueError('pair temporal_prior must be finite within [0, 1]')

def _validate_contract_scalars(value: PairHypothesisSet) -> None:
    if type(value.raw_points_preserved) is not bool:
        raise ValueError('raw_points_preserved must be an exact built-in bool')
    if isinstance(value.fixed_scale, bool) or not isinstance(value.fixed_scale, (int, float)):
        raise ValueError('fixed_scale must be 1.0')
    if not math.isfinite(value.fixed_scale) or not math.isclose(value.fixed_scale, 1.0, abs_tol=_TOLERANCE):
        raise ValueError('fixed_scale must be 1.0')
    if value.axis != 'CAD_Z_up':
        raise ValueError('axis must be CAD_Z_up')

def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f'{key} must be a mapping')
    return value

def _required_sequence(payload: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = payload.get(key)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f'{key} must be a sequence')
    return value

def _temporal_pair_from_dict(payload: Mapping[str, Any]) -> TemporalPair:
    keys = ('pair_index', 'source_scan_id', 'target_scan_id', 'gap_s', 'gap_source', 'gap_median_s', 'gap_mad_s', 'long_gap_cutoff_s', 'long_gap', 'temporal_prior', 'endpoint_policy')
    try:
        pair = TemporalPair(**{key: payload[key] for key in keys})
    except (KeyError, TypeError) as exc:
        raise ValueError('pair does not match the TemporalPair contract') from exc
    if payload.get('pair_id') != pair.pair_id:
        raise ValueError('serialized pair_id does not match TemporalPair direction')
    return pair

def _hypothesis_from_dict(payload: object) -> AlignmentHypothesis:
    if not isinstance(payload, Mapping):
        raise ValueError('hypothesis must be a mapping')
    feature_payload = payload.get('features')
    if feature_payload is not None and (not isinstance(feature_payload, Mapping)):
        raise ValueError('hypothesis features must be a mapping or None')
    try:
        return AlignmentHypothesis(hypothesis_id=payload['hypothesis_id'], pair_id=payload['pair_id'], source_scan_id=payload['source_scan_id'], target_scan_id=payload['target_scan_id'], state=HypothesisState(payload['state']), transform_target_from_source_zup=payload.get('transform_target_from_source_zup'), features=HypothesisFeatures(**feature_payload) if feature_payload is not None else None, rejection_reasons=tuple(_required_sequence(payload, 'rejection_reasons')), evidence=_required_mapping(payload, 'evidence'))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('hypothesis does not match the alignment contract') from exc
__all__ = ['SCHEMA', 'AlignmentHypothesis', 'HypothesisFeatures', 'HypothesisState', 'Matrix4', 'PairHypothesisSet', 'make_null_hypothesis', 'validate_zup_rigid_transform', 'yup_transform_to_zup', 'zup_transform_to_yup']
