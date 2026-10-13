
from __future__ import annotations
import json
import math
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias
PROJECT_SCHEMA = 'cca.points-project.v1'
WORKING_UNITS = 'meter'
WORKING_AXIS = 'CAD_Z_up'
FIXED_SCALE = 1.0
UNGROUPED = 'UNGROUPED'

class ProjectModelError(ValueError):
    pass

class PipelineStage(str, Enum):
    IMPORT = 'IMPORT'
    VALIDATE = 'VALIDATE'
    REVIEW = 'REVIEW'
    ALIGN = 'ALIGN'
    QC = 'QC'
    DXF_FIT = 'DXF FIT'
    SURFACE = 'SURFACE'
    EXPORT = 'EXPORT'
PIPELINE_STAGES = tuple(PipelineStage)
Matrix4: TypeAlias = tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]

def identity_transform() -> Matrix4:
    return ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))

def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProjectModelError(f'{label} must be numeric')
    output = float(value)
    if not math.isfinite(output):
        raise ProjectModelError(f'{label} must be finite')
    return output

def _text(value: object, default: str='') -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value
    return default

def _json_safe(value: object) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (not math.isfinite(value)):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes, bytearray))):
        return [_json_safe(item) for item in value]
    return str(value)

def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}

def _path_value(value: object) -> str | None:
    if isinstance(value, (str, Path)) and str(value):
        return str(value)
    if isinstance(value, Mapping):
        for key in ('path', 'file', 'geometry'):
            candidate = value.get(key)
            if isinstance(candidate, (str, Path)) and str(candidate):
                return str(candidate)
    return None

def _optional_count(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProjectModelError(f'{label} must be a non-negative integer')
    return value

def _normalize_transform(value: object, label: str='transform') -> Matrix4:
    if value is None:
        return identity_transform()
    if isinstance(value, Mapping):
        placement_keys = ('x_m', 'y_m', 'z_m', 'yaw_deg')
        if any((key in value for key in placement_keys)):
            x = _number(value.get('x_m', 0.0), f'{label}.x_m')
            y = _number(value.get('y_m', 0.0), f'{label}.y_m')
            z = _number(value.get('z_m', 0.0), f'{label}.z_m')
            yaw = math.radians(_number(value.get('yaw_deg', 0.0), f'{label}.yaw_deg'))
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            value = ((cosine, -sine, 0.0, x), (sine, cosine, 0.0, y), (0.0, 0.0, 1.0, z), (0.0, 0.0, 0.0, 1.0))
        else:
            for key in ('matrix', 'T_global_from_run', 'initial_transform'):
                if key in value:
                    value = value[key]
                    break
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ProjectModelError(f'{label} must be a 4x4 matrix')
    if len(value) == 16 and (not any((isinstance(item, Sequence) for item in value))):
        value = [value[index:index + 4] for index in range(0, 16, 4)]
    if len(value) != 4:
        raise ProjectModelError(f'{label} must be a 4x4 matrix')
    rows: list[tuple[float, float, float, float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ProjectModelError(f'{label} must be a 4x4 matrix')
        if len(row) != 4:
            raise ProjectModelError(f'{label} must be a 4x4 matrix')
        rows.append(tuple((_number(item, f'{label}[{row_index}][{column_index}]') for column_index, item in enumerate(row))))
    matrix = tuple(rows)
    if any((not math.isclose(matrix[3][index], expected, abs_tol=1e-08) for index, expected in enumerate((0.0, 0.0, 0.0, 1.0)))):
        raise ProjectModelError(f'{label} must be affine')
    rotation = [row[:3] for row in matrix[:3]]
    for first in range(3):
        for second in range(3):
            dot = sum((rotation[row][first] * rotation[row][second] for row in range(3)))
            expected = 1.0 if first == second else 0.0
            if not math.isclose(dot, expected, abs_tol=1e-05):
                raise ProjectModelError(f'{label} must have scale fixed at one')
    determinant = rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1]) - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0]) + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    if not math.isclose(determinant, 1.0, abs_tol=1e-05):
        raise ProjectModelError(f'{label} must be a proper rigid transform')
    if any((not math.isclose(actual, expected, abs_tol=1e-05) for actual, expected in zip((rotation[0][2], rotation[1][2], rotation[2][0], rotation[2][1], rotation[2][2]), (0.0, 0.0, 0.0, 0.0, 1.0), strict=True))):
        raise ProjectModelError(f'{label} must preserve CAD Z-up')
    return matrix

def _matrix_json(matrix: Matrix4) -> list[list[float]]:
    return [list(row) for row in matrix]

def _validate_contract(payload: Mapping[str, Any], label: str) -> None:
    contract = _mapping(payload.get('contracts'))
    units = contract.get('units', payload.get('units', WORKING_UNITS))
    axis = contract.get('axis', payload.get('axis', WORKING_AXIS))
    scale_locked = contract.get('scale_locked', payload.get('scale_locked', True))
    fixed_scale = contract.get('fixed_scale', payload.get('fixed_scale', FIXED_SCALE))
    if units != WORKING_UNITS:
        raise ProjectModelError(f'{label} must use metre working units')
    if axis != WORKING_AXIS:
        raise ProjectModelError(f'{label} must use CAD Z-up')
    if scale_locked is not True or _number(fixed_scale, f'{label}.fixed_scale') != 1.0:
        raise ProjectModelError(f'{label} scale must remain locked at one')
    point_policy = _mapping(payload.get('point_policy'))
    raw_preserved = contract.get('raw_points_preserved', payload.get('raw_points_preserved', True))
    if raw_preserved is not True or point_policy.get('raw_points_deleted') is True:
        raise ProjectModelError(f'{label} must preserve every raw point')
    cutoff = point_policy.get('distance_cutoff_m', point_policy.get('lidar_range_cutoff_m'))
    if cutoff is not None:
        raise ProjectModelError(f'{label} cannot declare a distance cutoff')

@dataclass(slots=True)
class ScanLayer:
    key: str
    path: str
    role: str = 'DISPLAY_EVIDENCE'
    point_count: int | None = None
    visible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key or not self.path:
            raise ProjectModelError('scan layer key and path are required')
        self.point_count = _optional_count(self.point_count, f'{self.key}.point_count')
        if type(self.visible) is not bool:
            raise ProjectModelError(f'{self.key}.visible must be a JSON boolean')
        self.metadata = _json_safe(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {'key': self.key, 'path': self.path, 'role': self.role, 'point_count': self.point_count, 'visible': self.visible, 'metadata': _json_safe(self.metadata)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScanLayer:
        return cls(key=_text(payload.get('key')), path=_text(payload.get('path')), role=_text(payload.get('role'), 'DISPLAY_EVIDENCE'), point_count=payload.get('point_count'), visible=payload.get('visible', False), metadata=dict(_mapping(payload.get('metadata'))))

@dataclass(slots=True)
class ScanEntry:
    scan_id: str
    rail_order: int
    source_path: str | None = None
    group_id: str = UNGROUPED
    status: str = 'IMPORTED'
    transform: Matrix4 = field(default_factory=identity_transform)
    layers: list[ScanLayer] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    units: str = WORKING_UNITS
    axis: str = WORKING_AXIS
    fixed_scale: float = FIXED_SCALE
    raw_points_preserved: bool = True
    placement_locked: bool = False

    def __post_init__(self) -> None:
        if not self.scan_id:
            raise ProjectModelError('scan id is required')
        if isinstance(self.rail_order, bool) or not isinstance(self.rail_order, int):
            raise ProjectModelError('scan rail order must be an integer')
        if self.rail_order < 0:
            raise ProjectModelError('scan rail order cannot be negative')
        _validate_contract({'units': self.units, 'axis': self.axis, 'fixed_scale': self.fixed_scale, 'scale_locked': True, 'raw_points_preserved': self.raw_points_preserved}, self.scan_id)
        self.transform = _normalize_transform(self.transform, f'{self.scan_id}.transform')
        if type(self.visible) is not bool:
            raise ProjectModelError(f'{self.scan_id}.visible must be a JSON boolean')
        if type(self.placement_locked) is not bool:
            raise ProjectModelError(f'{self.scan_id}.placement_locked must be a JSON boolean')
        layer_keys: set[str] = set()
        normalized_layers: list[ScanLayer] = []
        for layer in self.layers:
            item = layer if isinstance(layer, ScanLayer) else ScanLayer.from_dict(layer)
            if item.key in layer_keys:
                raise ProjectModelError(f'duplicate scan layer: {item.key}')
            layer_keys.add(item.key)
            normalized_layers.append(item)
        self.layers = normalized_layers
        self.quality = _json_safe(self.quality)
        self.metadata = _json_safe(self.metadata)

    def layer(self, key: str) -> ScanLayer | None:
        return next((layer for layer in self.layers if layer.key == key), None)

    def to_dict(self) -> dict[str, Any]:
        return {'scan_id': self.scan_id, 'rail_order': self.rail_order, 'source_path': self.source_path, 'group_id': self.group_id, 'status': self.status, 'transform': _matrix_json(self.transform), 'layers': [layer.to_dict() for layer in self.layers], 'quality': _json_safe(self.quality), 'visible': self.visible, 'placement_locked': self.placement_locked, 'metadata': _json_safe(self.metadata), 'contracts': {'units': WORKING_UNITS, 'axis': WORKING_AXIS, 'scale_locked': True, 'fixed_scale': FIXED_SCALE, 'raw_points_preserved': True}}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], rail_order: int) -> ScanEntry:
        _validate_contract(payload, _text(payload.get('scan_id'), 'scan'))
        contract = _mapping(payload.get('contracts'))
        raw_layers = payload.get('layers', [])
        if not isinstance(raw_layers, Sequence) or isinstance(raw_layers, (str, bytes, bytearray)):
            raise ProjectModelError('scan layers must be a list')
        return cls(scan_id=_text(payload.get('scan_id') or payload.get('id')), rail_order=rail_order, source_path=_path_value(payload.get('source_path')), group_id=_text(payload.get('group_id'), UNGROUPED), status=_text(payload.get('status'), 'IMPORTED'), transform=_normalize_transform(payload.get('transform')), layers=[ScanLayer.from_dict(_mapping(item)) for item in raw_layers], quality=dict(_mapping(payload.get('quality'))), visible=payload.get('visible', True), placement_locked=payload.get('placement_locked', False), metadata=dict(_mapping(payload.get('metadata'))), units=_text(contract.get('units', payload.get('units')), WORKING_UNITS), axis=_text(contract.get('axis', payload.get('axis')), WORKING_AXIS), fixed_scale=contract.get('fixed_scale', payload.get('fixed_scale', FIXED_SCALE)), raw_points_preserved=contract.get('raw_points_preserved', payload.get('raw_points_preserved', True)))

    @classmethod
    def from_manifest_run(cls, run: Mapping[str, Any], rail_order: int, display_contract: Mapping[str, Any] | None=None) -> ScanEntry:
        _validate_contract(run, _text(run.get('id'), 'scan'))
        scan_id = _text(run.get('id') or run.get('run_id') or run.get('name'))
        if not scan_id:
            raise ProjectModelError('manifest scan id is required')
        group_id = _text(run.get('scan_group') or run.get('group_id') or run.get('component_context'), UNGROUPED)
        default_visible = run.get('default_visible_layers', ('core',))
        if not isinstance(default_visible, Sequence) or isinstance(default_visible, (str, bytes, bytearray)):
            default_visible = ('core',)
        visible_names = {str(item) for item in default_visible}
        layer_counts = _mapping(run.get('viewer_layer_counts'))
        roles = display_contract or {}
        layers: list[ScanLayer] = []
        seen: set[str] = set()

        def add_layer(key: str, value: object, role: object, count: object=None, visible: bool=False, metadata: Mapping[str, Any] | None=None) -> None:
            path = _path_value(value)
            if not path or key in seen:
                return
            seen.add(key)
            layers.append(ScanLayer(key=key, path=path, role=_text(role, 'DISPLAY_EVIDENCE'), point_count=_optional_count(count, f'{scan_id}.{key}.point_count'), visible=visible, metadata=dict(metadata or {})))
        nested = run.get('viewer_layer_paths')
        if not isinstance(nested, Mapping):
            nested = run.get('viewer_layers')
        if isinstance(nested, Mapping):
            for mode in ('rgb', 'confcolor'):
                mode_layers = _mapping(nested.get(mode))
                for layer_name in ('core', 'body', 'shell'):
                    add_layer(f'{mode}/{layer_name}', mode_layers.get(layer_name), roles.get(layer_name, f'VIEWER_{layer_name.upper()}'), layer_counts.get(layer_name), layer_name in visible_names, {'mode': mode, 'layer': layer_name})
        add_layer('rgb/core', run.get('viewer_rgb'), roles.get('core', 'VIEWER_CORE'), run.get('viewer_point_count'), 'core' in visible_names)
        add_layer('confcolor/core', run.get('viewer_confcolor'), roles.get('core', 'VIEWER_CORE'), run.get('viewer_point_count'), 'core' in visible_names)
        geometry_contract = _mapping(run.get('geometry_contract'))
        add_layer('rgb/full', run.get('full_rgb') or run.get('source_path') or run.get('path'), run.get('full_rgb_geometry_role') or geometry_contract.get('primary') or 'PRIMARY_SCAN_POINTS', run.get('primary_point_count'), False, {'source_axis': run.get('full_rgb_source_axis')})
        add_layer('confcolor/full', run.get('full_confcolor'), run.get('full_confcolor_geometry_role') or 'RAW_FRAME_EVIDENCE', run.get('full_point_count'), False, {'source_axis': run.get('full_confcolor_source_axis')})
        raw_confcolor = run.get('raw_confcolor_zup')
        if raw_confcolor is None and run.get('full_confcolor_source_axis') == WORKING_AXIS:
            raw_confcolor = run.get('full_confcolor')
        raw_rgb = run.get('raw_rgb_zup')
        if raw_rgb is None and isinstance(raw_confcolor, str):
            sibling = raw_confcolor.replace('_confcolor_', '_rgb_')
            raw_rgb = sibling if sibling != raw_confcolor else None
        add_layer('rgb/raw', raw_rgb, run.get('raw_rgb_geometry_role') or 'RAW_FRAME_RGB_EVIDENCE', run.get('raw_point_count') or run.get('full_point_count'), False, {'source_axis': run.get('raw_source_axis', WORKING_AXIS)})
        add_layer('confcolor/raw', raw_confcolor, run.get('raw_confcolor_geometry_role') or 'RAW_FRAME_CONFIDENCE_EVIDENCE', run.get('raw_point_count') or run.get('full_point_count'), False, {'source_axis': run.get('raw_source_axis', WORKING_AXIS)})
        add_layer('rgb/full_zup', run.get('full_rgb_zup'), run.get('full_rgb_geometry_role') or 'PRIMARY_APP_FUSED_SURFACE', run.get('primary_point_count'), False, {'source_axis': WORKING_AXIS})
        generic_layers = run.get('layers')
        if isinstance(generic_layers, Mapping):
            for key, value in generic_layers.items():
                add_layer(str(key), value, 'DISPLAY_EVIDENCE')
        elif isinstance(generic_layers, Sequence) and (not isinstance(generic_layers, (str, bytes, bytearray))):
            for item in generic_layers:
                layer_payload = _mapping(item)
                key = _text(layer_payload.get('key') or layer_payload.get('name'))
                add_layer(key, layer_payload, layer_payload.get('role'), layer_payload.get('point_count'), layer_payload.get('visible', False) is True, _mapping(layer_payload.get('metadata')))
        quality = dict(_mapping(run.get('quality')))
        for key in ('viewer_layer_counts', 'viewer_point_count', 'primary_point_count', 'full_point_count'):
            if key in run:
                quality[key] = _json_safe(run[key])
        quality_transfer = geometry_contract.get('quality_transfer')
        if isinstance(quality_transfer, Mapping):
            quality['quality_transfer'] = _json_safe(quality_transfer)
        transform = next((run[key] for key in ('T_global_from_run', 'transform', 'initial_transform', 'placement') if key in run), None)
        known = {'id', 'run_id', 'name', 'scan_group', 'group_id', 'component_context', 'status', 'T_global_from_run', 'transform', 'initial_transform', 'placement', 'viewer_layer_paths', 'viewer_layers', 'viewer_rgb', 'viewer_confcolor', 'full_rgb', 'full_confcolor', 'full_rgb_zup', 'raw_rgb_zup', 'raw_confcolor_zup', 'raw_source_axis', 'raw_point_count', 'raw_rgb_geometry_role', 'raw_confcolor_geometry_role', 'layers', 'quality', 'viewer_layer_counts', 'viewer_point_count', 'primary_point_count', 'full_point_count', 'geometry_contract', 'initially_visible', 'placement_locked', 'default_visible_layers'}
        metadata = {key: _json_safe(value) for key, value in run.items() if key not in known}
        return cls(scan_id=scan_id, rail_order=rail_order, source_path=_path_value(run.get('source_path') or run.get('full_rgb') or run.get('path')), group_id=group_id, status=_text(run.get('status'), 'IMPORTED'), transform=_normalize_transform(transform, f'{scan_id}.transform'), layers=layers, quality=quality, visible=run.get('initially_visible', run.get('visible', rail_order == 0)), placement_locked=run.get('placement_locked', False), metadata=metadata)

@dataclass(slots=True)
class ScanGroup:
    group_id: str
    label: str = ''
    status: str = 'UNREVIEWED'
    scan_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ProjectModelError('scan group id is required')
        self.label = self.label or self.group_id
        self.scan_ids = list(dict.fromkeys(self.scan_ids))
        self.metadata = _json_safe(self.metadata)

    def add_scan(self, scan_id: str) -> None:
        if scan_id not in self.scan_ids:
            self.scan_ids.append(scan_id)

    def to_dict(self) -> dict[str, Any]:
        return {'group_id': self.group_id, 'label': self.label, 'status': self.status, 'scan_ids': list(self.scan_ids), 'metadata': _json_safe(self.metadata)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScanGroup:
        raw_ids = payload.get('scan_ids', payload.get('runs', []))
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes, bytearray)):
            raw_ids = []
        scan_ids = [_text(item.get('id')) if isinstance(item, Mapping) else _text(item) for item in raw_ids]
        return cls(group_id=_text(payload.get('group_id') or payload.get('id')), label=_text(payload.get('label') or payload.get('name')), status=_text(payload.get('status'), 'UNREVIEWED'), scan_ids=[item for item in scan_ids if item], metadata=dict(_mapping(payload.get('metadata'))))
_SOURCE_UNIT_TO_METER = {'m': 1.0, 'meter': 1.0, 'metre': 1.0, 'cm': 0.01, 'mm': 0.001}

@dataclass(slots=True)
class DXFPlanState:
    loaded: bool = False
    source_path: str | None = None
    geometry_path: str | None = None
    source_name: str | None = None
    source_unit: str | None = None
    source_to_meter: float | None = None
    visible: bool = False
    fit_status: str = 'NOT_LOADED'
    transform: Matrix4 = field(default_factory=identity_transform)
    segment_count: int | None = None
    bounds_m: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    units: str = WORKING_UNITS
    axis: str = WORKING_AXIS
    fixed_scale: float = FIXED_SCALE

    def __post_init__(self) -> None:
        _validate_contract({'units': self.units, 'axis': self.axis, 'scale_locked': True, 'fixed_scale': self.fixed_scale}, 'DXF plan')
        if type(self.loaded) is not bool or type(self.visible) is not bool:
            raise ProjectModelError('DXF loaded and visible flags must be JSON booleans')
        self.transform = _normalize_transform(self.transform, 'DXF transform')
        self.segment_count = _optional_count(self.segment_count, 'DXF segment_count')
        if self.source_unit is not None:
            normalized_unit = self.source_unit.strip().lower()
            self.source_unit = normalized_unit or None
        if self.source_to_meter is not None:
            self.source_to_meter = _number(self.source_to_meter, 'DXF source_to_meter')
            if self.source_to_meter <= 0.0:
                raise ProjectModelError('DXF source_to_meter must be positive')
        if self.source_unit is not None:
            expected_factor = _SOURCE_UNIT_TO_METER.get(self.source_unit)
            if expected_factor is None:
                raise ProjectModelError('DXF source_unit must be m, meter, metre, cm, or mm')
            if self.source_to_meter is None:
                self.source_to_meter = expected_factor
            elif not math.isclose(self.source_to_meter, expected_factor, rel_tol=0.0, abs_tol=1e-12):
                raise ProjectModelError(f'DXF {self.source_unit} source_to_meter must be {expected_factor:g}')
        if self.loaded and self.fit_status == 'NOT_LOADED':
            self.fit_status = 'READY_FOR_FIT'
        self.bounds_m = _json_safe(self.bounds_m)
        self.quality = _json_safe(self.quality)

    def to_dict(self) -> dict[str, Any]:
        return {'loaded': self.loaded, 'source_path': self.source_path, 'geometry_path': self.geometry_path, 'source_name': self.source_name, 'source_unit': self.source_unit, 'source_to_meter': self.source_to_meter, 'visible': self.visible, 'fit_status': self.fit_status, 'transform': _matrix_json(self.transform), 'segment_count': self.segment_count, 'bounds_m': _json_safe(self.bounds_m), 'quality': _json_safe(self.quality), 'contracts': {'units': WORKING_UNITS, 'axis': WORKING_AXIS, 'scale_locked': True, 'fixed_scale': FIXED_SCALE}}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DXFPlanState:
        _validate_contract(payload, 'DXF plan')
        contract = _mapping(payload.get('contracts'))
        return cls(loaded=payload.get('loaded', False), source_path=_path_value(payload.get('source_path')), geometry_path=_path_value(payload.get('geometry_path')), source_name=_text(payload.get('source_name')) or None, source_unit=_text(payload.get('source_unit')) or None, source_to_meter=payload.get('source_to_meter'), visible=payload.get('visible', False), fit_status=_text(payload.get('fit_status'), 'NOT_LOADED'), transform=_normalize_transform(payload.get('transform')), segment_count=payload.get('segment_count'), bounds_m=dict(_mapping(payload.get('bounds_m'))), quality=dict(_mapping(payload.get('quality'))), units=_text(contract.get('units', payload.get('units')), WORKING_UNITS), axis=_text(contract.get('axis', payload.get('axis')), WORKING_AXIS), fixed_scale=contract.get('fixed_scale', payload.get('fixed_scale', FIXED_SCALE)))

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> DXFPlanState:
        if not payload:
            return cls()
        _validate_contract(payload, 'DXF plan')
        source = _mapping(payload.get('source'))
        source_unit = _text(source.get('unit') or payload.get('source_unit') or payload.get('unit')).lower()
        source_to_meter = payload.get('source_to_meter')
        if source_to_meter is None:
            source_to_meter = _SOURCE_UNIT_TO_METER.get(source_unit)
        geometry_path = _path_value(payload.get('geometry_path') or payload.get('geometry'))
        source_path = _path_value(source.get('path') or payload.get('source_path'))
        source_name = _text(source.get('name') or payload.get('source_name')) or None
        loaded = bool(geometry_path or source_path or source_name)
        quality = {key: _json_safe(value) for key, value in payload.items() if key not in {'source', 'geometry', 'geometry_path', 'source_path', 'source_name', 'source_unit', 'unit', 'source_to_meter', 'visible', 'fit_status', 'status', 'transform', 'T_plan_from_source', 'segment_count', 'bounds_m'}}
        return cls(loaded=loaded, source_path=source_path, geometry_path=geometry_path, source_name=source_name, source_unit=source_unit or None, source_to_meter=source_to_meter, visible=payload.get('visible', False), fit_status=_text(payload.get('fit_status') or payload.get('status'), 'READY_FOR_FIT' if loaded else 'NOT_LOADED'), transform=_normalize_transform(payload.get('transform') or payload.get('T_plan_from_source')), segment_count=payload.get('segment_count'), bounds_m=dict(_mapping(payload.get('bounds_m'))), quality=quality)

@dataclass(slots=True)
class PipelineState:
    current: PipelineStage = PipelineStage.IMPORT
    completed: list[PipelineStage] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.current, PipelineStage):
            try:
                self.current = PipelineStage(str(self.current))
            except ValueError as exc:
                raise ProjectModelError('unknown pipeline stage') from exc
        normalized: list[PipelineStage] = []
        for item in self.completed:
            try:
                stage = item if isinstance(item, PipelineStage) else PipelineStage(str(item))
            except ValueError as exc:
                raise ProjectModelError('unknown completed pipeline stage') from exc
            if stage not in normalized:
                normalized.append(stage)
        current_index = PIPELINE_STAGES.index(self.current)
        if any((PIPELINE_STAGES.index(stage) >= current_index for stage in normalized)):
            raise ProjectModelError('completed stages must precede the current stage')
        expected = list(PIPELINE_STAGES[:len(normalized)])
        if normalized != expected:
            raise ProjectModelError('completed pipeline stages must be contiguous')
        self.completed = normalized

    def advance(self, target: PipelineStage | str | None=None) -> PipelineStage:
        current_index = PIPELINE_STAGES.index(self.current)
        if current_index == len(PIPELINE_STAGES) - 1:
            return self.current
        expected = PIPELINE_STAGES[current_index + 1]
        if target is not None:
            try:
                requested = target if isinstance(target, PipelineStage) else PipelineStage(target)
            except ValueError as exc:
                raise ProjectModelError('unknown pipeline stage') from exc
            if requested is not expected:
                raise ProjectModelError(f'next pipeline stage must be {expected.value}')
        self.completed = list(PIPELINE_STAGES[:current_index + 1])
        self.current = expected
        return self.current

    def to_dict(self) -> dict[str, Any]:
        return {'stages': [stage.value for stage in PIPELINE_STAGES], 'current': self.current.value, 'completed': [stage.value for stage in self.completed]}

    @classmethod
    def at(cls, current: PipelineStage | str) -> PipelineState:
        stage = current if isinstance(current, PipelineStage) else PipelineStage(current)
        index = PIPELINE_STAGES.index(stage)
        return cls(current=stage, completed=list(PIPELINE_STAGES[:index]))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PipelineState:
        return cls(current=payload.get('current', PipelineStage.IMPORT.value), completed=list(payload.get('completed', [])))

@dataclass(slots=True)
class ProjectModel:
    project_id: str = ''
    name: str = ''
    scans: list[ScanEntry] = field(default_factory=list)
    groups: list[ScanGroup] = field(default_factory=list)
    dxf_plan: DXFPlanState = field(default_factory=DXFPlanState)
    pipeline: PipelineState = field(default_factory=PipelineState)
    source_manifest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    units: str = WORKING_UNITS
    axis: str = WORKING_AXIS
    fixed_scale: float = FIXED_SCALE
    raw_points_preserved: bool = True

    def __post_init__(self) -> None:
        _validate_contract({'units': self.units, 'axis': self.axis, 'scale_locked': True, 'fixed_scale': self.fixed_scale, 'raw_points_preserved': self.raw_points_preserved}, 'project')
        normalized_scans: list[ScanEntry] = []
        seen: set[str] = set()
        for index, scan in enumerate(self.scans):
            item = scan if isinstance(scan, ScanEntry) else ScanEntry.from_dict(scan, index)
            if item.scan_id in seen:
                raise ProjectModelError(f'duplicate scan id: {item.scan_id}')
            seen.add(item.scan_id)
            item.rail_order = index
            normalized_scans.append(item)
        self.scans = normalized_scans
        self.groups = [group if isinstance(group, ScanGroup) else ScanGroup.from_dict(group) for group in self.groups]
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ProjectModelError('duplicate scan group id')
        self.metadata = _json_safe(self.metadata)
        self._sync_groups()

    @classmethod
    def empty(cls) -> ProjectModel:
        return cls()

    @classmethod
    def new(cls, name: str, project_id: str | None=None) -> ProjectModel:
        if not name.strip():
            raise ProjectModelError('project name is required')
        return cls(project_id=project_id or uuid.uuid4().hex, name=name.strip())

    @classmethod
    def open(cls, path: str | Path) -> ProjectModel:
        source = Path(path).resolve()
        try:
            payload = json.loads(source.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise ProjectModelError('cannot open project or manifest') from exc
        if not isinstance(payload, Mapping):
            raise ProjectModelError('project root must be an object')
        if payload.get('schema') == PROJECT_SCHEMA:
            project = cls.from_dict(payload)
            if project.source_manifest:
                source_manifest = Path(project.source_manifest)
                if not source_manifest.is_absolute():
                    project.source_manifest = str((source.parent / source_manifest).resolve())
            return project
        if isinstance(payload.get('runs') or payload.get('scans'), Sequence):
            return cls.from_manifest(payload, source)
        raise ProjectModelError('unsupported project or manifest schema')

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any], source_path: str | Path | None=None) -> ProjectModel:
        _validate_contract(payload, 'manifest')
        raw_runs = payload.get('runs', payload.get('scans', []))
        if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes, bytearray)):
            raise ProjectModelError('manifest scans must be a list')
        display_contract = _mapping(payload.get('display_contract'))
        scans = [ScanEntry.from_manifest_run(_mapping(run), index, display_contract) for index, run in enumerate(raw_runs)]
        raw_groups = payload.get('groups', [])
        groups: list[ScanGroup] = []
        if isinstance(raw_groups, Sequence) and (not isinstance(raw_groups, (str, bytes, bytearray))):
            groups = [ScanGroup.from_dict(_mapping(group)) for group in raw_groups]
        schema = _text(payload.get('schema'))
        declared_stage = _text(payload.get('pipeline_stage') or _mapping(payload.get('pipeline')).get('current'))
        if declared_stage:
            pipeline = PipelineState.at(declared_stage)
        elif schema == 'cca.native-manual-review.v1':
            pipeline = PipelineState.at(PipelineStage.REVIEW)
        else:
            pipeline = PipelineState.at(PipelineStage.VALIDATE if scans else PipelineStage.IMPORT)
        source = str(Path(source_path).resolve()) if source_path is not None else None
        name = _text(payload.get('name'))
        if not name and source_path is not None:
            name = Path(source_path).stem
        if not name:
            name = 'Imported Project'
        metadata = {'source_schema': schema or None, 'source_status': payload.get('status')}
        return cls(project_id=_text(payload.get('project_id')) or uuid.uuid4().hex, name=name, scans=scans, groups=groups, dxf_plan=DXFPlanState.from_manifest(_mapping(payload.get('plan'))), pipeline=pipeline, source_manifest=source, metadata=metadata)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProjectModel:
        if payload.get('schema') != PROJECT_SCHEMA:
            raise ProjectModelError('unsupported CCA Points project schema')
        _validate_contract(payload, 'project')
        raw_scans = payload.get('scan_rail', payload.get('scans', []))
        if not isinstance(raw_scans, Sequence) or isinstance(raw_scans, (str, bytes, bytearray)):
            raise ProjectModelError('project scan rail must be a list')
        raw_groups = payload.get('scan_groups', payload.get('groups', []))
        if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes, bytearray)):
            raise ProjectModelError('project scan groups must be a list')
        contract = _mapping(payload.get('contracts'))
        return cls(project_id=_text(payload.get('project_id')), name=_text(payload.get('name')), scans=[ScanEntry.from_dict(_mapping(scan), index) for index, scan in enumerate(raw_scans)], groups=[ScanGroup.from_dict(_mapping(group)) for group in raw_groups], dxf_plan=DXFPlanState.from_dict(_mapping(payload.get('dxf_plan', payload.get('plan', {})))), pipeline=PipelineState.from_dict(_mapping(payload.get('pipeline'))), source_manifest=_path_value(payload.get('source_manifest')), metadata=dict(_mapping(payload.get('metadata'))), units=_text(contract.get('units', payload.get('units')), WORKING_UNITS), axis=_text(contract.get('axis', payload.get('axis')), WORKING_AXIS), fixed_scale=contract.get('fixed_scale', payload.get('fixed_scale', FIXED_SCALE)), raw_points_preserved=contract.get('raw_points_preserved', payload.get('raw_points_preserved', True)))

    @property
    def scan_rail(self) -> tuple[ScanEntry, ...]:
        return tuple(self.scans)

    def scan(self, scan_id: str) -> ScanEntry | None:
        return next((scan for scan in self.scans if scan.scan_id == scan_id), None)

    def import_scan(self, scan: ScanEntry | Mapping[str, Any]) -> ScanEntry:
        index = len(self.scans)
        item = scan if isinstance(scan, ScanEntry) else ScanEntry.from_manifest_run(_mapping(scan), index)
        if self.scan(item.scan_id) is not None:
            raise ProjectModelError(f'duplicate scan id: {item.scan_id}')
        item.rail_order = index
        self.scans.append(item)
        self._sync_groups()
        if self.pipeline.current is PipelineStage.IMPORT:
            self.pipeline.advance(PipelineStage.VALIDATE)
        return item

    def move_scan(self, scan_id: str, new_index: int) -> None:
        if isinstance(new_index, bool) or not isinstance(new_index, int):
            raise ProjectModelError('scan rail index must be an integer')
        if not 0 <= new_index < len(self.scans):
            raise ProjectModelError('scan rail index is outside the project')
        old_index = next((index for index, scan in enumerate(self.scans) if scan.scan_id == scan_id), None)
        if old_index is None:
            raise ProjectModelError(f'unknown scan id: {scan_id}')
        item = self.scans.pop(old_index)
        self.scans.insert(new_index, item)
        for index, entry in enumerate(self.scans):
            entry.rail_order = index
        self._sync_groups()

    def _sync_groups(self) -> None:
        by_id = {group.group_id: group for group in self.groups}
        for group in self.groups:
            group.scan_ids = []
        for scan in self.scans:
            group = by_id.get(scan.group_id)
            if group is None:
                group = ScanGroup(group_id=scan.group_id)
                self.groups.append(group)
                by_id[scan.group_id] = group
            group.add_scan(scan.scan_id)
        rail_index = {scan.scan_id: scan.rail_order for scan in self.scans}
        for group in self.groups:
            group.scan_ids.sort(key=rail_index.__getitem__)

    def assign_scans_to_group(self, scan_ids: Sequence[str], group_id: str, label: str='') -> ScanGroup:
        normalized_id = _text(group_id)
        if not normalized_id:
            raise ProjectModelError('scan group id is required')
        requested = list(dict.fromkeys((_text(scan_id) for scan_id in scan_ids)))
        if not requested or any((not scan_id for scan_id in requested)):
            raise ProjectModelError('select at least one scan for grouping')
        missing = [scan_id for scan_id in requested if self.scan(scan_id) is None]
        if missing:
            raise ProjectModelError(f'unknown scan id: {missing[0]}')
        group = next((item for item in self.groups if item.group_id == normalized_id), None)
        if group is None:
            group = ScanGroup(group_id=normalized_id, label=_text(label) or normalized_id)
            self.groups.append(group)
        elif label:
            group.label = _text(label)
        for scan_id in requested:
            scan = self.scan(scan_id)
            if scan is not None:
                scan.group_id = normalized_id
        self._sync_groups()
        return group

    def set_scan_placement_locked(self, scan_id: str, locked: bool) -> ScanEntry:
        if type(locked) is not bool:
            raise ProjectModelError('placement lock must be a JSON boolean')
        scan = self.scan(scan_id)
        if scan is None:
            raise ProjectModelError(f'unknown scan id: {scan_id}')
        scan.placement_locked = locked
        scan.metadata = dict(scan.metadata)
        scan.metadata['placement_lock'] = {'locked': locked, 'mode': 'FIXED_SCALE_Z_UP', 'source_geometry': 'UNCHANGED'}
        return scan

    def validate(self) -> None:
        self.__post_init__()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {'schema': PROJECT_SCHEMA, 'project_id': self.project_id, 'name': self.name, 'contracts': {'units': WORKING_UNITS, 'axis': WORKING_AXIS, 'scale_locked': True, 'fixed_scale': FIXED_SCALE, 'raw_points_preserved': True, 'distance_cutoff_m': None}, 'pipeline': self.pipeline.to_dict(), 'source_manifest': self.source_manifest, 'scan_rail': [scan.to_dict() for scan in self.scans], 'scan_groups': [group.to_dict() for group in self.groups], 'dxf_plan': self.dxf_plan.to_dict(), 'metadata': _json_safe(self.metadata)}

    def to_json(self, indent: int=2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, indent=indent, sort_keys=True) + '\n'

    def save(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = self.to_json().encode('ascii')
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=target.parent, prefix=f'.{target.name}.', delete=False) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
                temporary = Path(stream.name)
            os.replace(temporary, target)
        except OSError as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise ProjectModelError('cannot save project') from exc
        return target
__all__ = ['FIXED_SCALE', 'PIPELINE_STAGES', 'PROJECT_SCHEMA', 'WORKING_AXIS', 'WORKING_UNITS', 'DXFPlanState', 'PipelineStage', 'PipelineState', 'ProjectModel', 'ProjectModelError', 'ScanEntry', 'ScanGroup', 'ScanLayer', 'identity_transform']
