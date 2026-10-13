
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import tempfile
from itertools import pairwise
from pathlib import Path
from typing import Any
import ezdxf
from ezdxf.disassemble import recursive_decompose
from ezdxf.path import make_path
_UNIT_SCALE_TO_METER = {'mm': 0.001, 'cm': 0.01, 'm': 1.0}

class PlanDxfError(ValueError):
    pass

def import_plan_dxf(source: str | Path, *, source_unit: str, curve_tolerance_m: float=0.02) -> dict[str, Any]:
    path = Path(source)
    if source_unit not in _UNIT_SCALE_TO_METER:
        allowed = ', '.join(sorted(_UNIT_SCALE_TO_METER))
        raise PlanDxfError(f'source_unit must be one of: {allowed}')
    if not math.isfinite(curve_tolerance_m) or curve_tolerance_m <= 0.0:
        raise PlanDxfError('curve_tolerance_m must be finite and positive')
    if not path.is_file():
        raise PlanDxfError(f'DXF plan does not exist: {path}')
    try:
        document = ezdxf.readfile(path)
    except (OSError, ezdxf.DXFError) as exc:
        raise PlanDxfError(f'cannot read DXF plan: {path.name}') from exc
    scale = _UNIT_SCALE_TO_METER[source_unit]
    tolerance_source = curve_tolerance_m / scale
    segments: list[dict[str, Any]] = []
    unsupported: dict[str, int] = {}
    source_entity_count = 0
    for entity in recursive_decompose(document.modelspace()):
        source_entity_count += 1
        entity_type = entity.dxftype()
        try:
            geometry = make_path(entity)
        except (TypeError, ValueError, AttributeError):
            unsupported[entity_type] = unsupported.get(entity_type, 0) + 1
            continue
        vertices = list(geometry.flattening(distance=tolerance_source, segments=8))
        if len(vertices) < 2:
            unsupported[entity_type] = unsupported.get(entity_type, 0) + 1
            continue
        if geometry.is_closed and (not vertices[0].isclose(vertices[-1])):
            vertices.append(vertices[0])
        layer = str(entity.dxf.get('layer', '0'))
        for first, second in pairwise(vertices):
            points = [[float(first.x) * scale, float(first.y) * scale, float(first.z) * scale], [float(second.x) * scale, float(second.y) * scale, float(second.z) * scale]]
            if not all((math.isfinite(value) for point in points for value in point)):
                raise PlanDxfError(f'DXF entity {entity_type} has non-finite coordinates')
            if points[0] == points[1]:
                continue
            segments.append({'layer': layer, 'source_type': entity_type, 'points_m': points})
    if not segments:
        raise PlanDxfError('DXF plan contains no supported line geometry')
    coordinate_rows = [point for segment in segments for point in segment['points_m']]
    bounds_min = [min((row[axis] for row in coordinate_rows)) for axis in range(3)]
    bounds_max = [max((row[axis] for row in coordinate_rows)) for axis in range(3)]
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'schema': 'cca.plan-geometry.v1', 'source': {'name': path.name, 'sha256': source_hash, 'unit': source_unit, 'header_insunits': int(document.header.get('$INSUNITS', 0)), 'entity_count': source_entity_count, 'unsupported_by_type': dict(sorted(unsupported.items()))}, 'output_unit': 'meter', 'scale_to_meter': scale, 'curve_tolerance_m': curve_tolerance_m, 'bounds_m': {'min': bounds_min, 'max': bounds_max}, 'segments': segments}

def _validated_plan_segments(plan_geometry: dict[str, Any]) -> list[dict[str, Any]]:
    if plan_geometry.get('schema') != 'cca.plan-geometry.v1':
        raise PlanDxfError('unsupported plan geometry schema')
    if plan_geometry.get('output_unit') != 'meter':
        raise PlanDxfError('plan geometry must be in metres')
    segments = plan_geometry.get('segments')
    if not isinstance(segments, list) or not segments:
        raise PlanDxfError('plan geometry must contain segments')
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise PlanDxfError(f'plan segment {index} must be an object')
        points = segment.get('points_m')
        if not isinstance(points, list) or len(points) != 2:
            raise PlanDxfError(f'plan segment {index} must contain two points')
        for point in points:
            if not isinstance(point, list) or len(point) != 3:
                raise PlanDxfError(f'plan segment {index} points must be XYZ')
            if not all((isinstance(value, (int, float)) and (not isinstance(value, bool)) and math.isfinite(float(value)) for value in point)):
                raise PlanDxfError(f'plan segment {index} coordinates must be finite')
    return segments

def _segment_dedupe_key(segment: dict[str, Any]) -> tuple[Any, ...]:
    points = segment['points_m']
    first = tuple((round(float(value), 9) for value in points[0]))
    second = tuple((round(float(value), 9) for value in points[1]))
    endpoints = (first, second) if first <= second else (second, first)
    return (str(segment.get('layer', '0')), endpoints)

def rebase_plan_geometry_for_viewer(plan_geometry: dict[str, Any], *, display_source_name: str, display_source_sha256: str, deduplicate: bool=True) -> dict[str, Any]:
    segments = _validated_plan_segments(plan_geometry)
    original_source = plan_geometry.get('source')
    if not isinstance(original_source, dict) or original_source.get('unit') != 'm':
        raise PlanDxfError('viewer rebase requires an explicit metre source import')
    if plan_geometry.get('scale_to_meter') != 1.0:
        raise PlanDxfError('viewer rebase source scale_to_meter must be 1.0')
    if not isinstance(display_source_name, str) or not display_source_name.isascii() or Path(display_source_name).name != display_source_name or (not display_source_name.lower().endswith('.dxf')):
        raise PlanDxfError('display_source_name must be an ASCII DXF filename')
    if not isinstance(display_source_sha256, str) or len(display_source_sha256) != 64 or any((character not in '0123456789abcdef' for character in display_source_sha256)):
        raise PlanDxfError('display_source_sha256 must be a lowercase SHA256')
    coordinates = [point for segment in segments for point in segment['points_m']]
    minimum = [min((float(point[axis]) for point in coordinates)) for axis in range(3)]
    maximum = [max((float(point[axis]) for point in coordinates)) for axis in range(3)]
    center_x = (minimum[0] + maximum[0]) / 2.0
    center_y = (minimum[1] + maximum[1]) / 2.0
    output_segments: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    duplicate_count = 0
    for segment in segments:
        candidate = {key: copy.deepcopy(value) for key, value in segment.items() if key != 'points_m'}
        candidate['points_m'] = [[float(point[0]) - center_x, float(point[1]) - center_y, float(point[2])] for point in segment['points_m']]
        key = _segment_dedupe_key(candidate)
        if deduplicate and key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        output_segments.append(candidate)
    rebased_coordinates = [point for segment in output_segments for point in segment['points_m']]
    rebased_minimum = [min((float(point[axis]) for point in rebased_coordinates)) for axis in range(3)]
    rebased_maximum = [max((float(point[axis]) for point in rebased_coordinates)) for axis in range(3)]
    return {'schema': 'cca.plan-geometry.v1', 'purpose': 'viewer-rebased', 'source': {'name': display_source_name, 'unit': 'cm', 'sha256': display_source_sha256}, 'provenance': {'original_unscaled_source': {'name': original_source.get('name'), 'sha256': original_source.get('sha256'), 'header_insunits': original_source.get('header_insunits'), 'coordinate_interpretation': 'meter'}, 'display_source_clean_cm': {'name': display_source_name, 'sha256': display_source_sha256, 'unit': 'centimeter'}}, 'output_unit': 'meter', 'segments_internal_unit': 'meter', 'source_plan_bounds_m': {'min': minimum, 'max': maximum}, 'bounds_m': {'min': rebased_minimum, 'max': rebased_maximum}, 'rebase': {'cad_bbox_center_xy_m': [center_x, center_y], 'viewer_origin_in_cad_zup_m': [center_x, center_y, 0.0], 'T_viewer_zup_from_cad_zup': [[1.0, 0.0, 0.0, -center_x], [0.0, 1.0, 0.0, -center_y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'T_cad_zup_from_viewer_zup': [[1.0, 0.0, 0.0, center_x], [0.0, 1.0, 0.0, center_y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 'inverse_formula': 'cad_xy_m = viewer_xy_m + cad_bbox_center_xy_m'}, 'simplification': {'curve_tolerance_m': float(plan_geometry.get('curve_tolerance_m', 0.0)), 'deduplicate_same_layer_geometry': bool(deduplicate), 'input_segment_count': len(segments), 'deduplicated_segment_count': duplicate_count, 'output_segment_count': len(output_segments)}, 'segments': output_segments}

def rebase_imported_plan_geometry_for_viewer(plan_geometry: dict[str, Any], *, deduplicate: bool=True) -> dict[str, Any]:
    _validated_plan_segments(plan_geometry)
    source = plan_geometry.get('source')
    if not isinstance(source, dict):
        raise PlanDxfError('imported plan source metadata is missing')
    source_unit = source.get('unit')
    expected_scale = _UNIT_SCALE_TO_METER.get(source_unit)
    scale = plan_geometry.get('scale_to_meter')
    if expected_scale is None or isinstance(scale, bool) or (not isinstance(scale, (int, float))) or (not math.isclose(float(scale), expected_scale, abs_tol=1e-12)):
        raise PlanDxfError('imported plan has an inconsistent explicit unit scale')
    source_hash = source.get('sha256')
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any((character not in '0123456789abcdef' for character in source_hash)):
        raise PlanDxfError('imported plan has an invalid source SHA256')
    source_name = source.get('name')
    display_name = source_name if isinstance(source_name, str) and source_name.isascii() and (Path(source_name).name == source_name) and source_name.lower().endswith('.dxf') else 'uploaded-plan.dxf'
    normalized = copy.deepcopy(plan_geometry)
    normalized['source']['unit'] = 'm'
    normalized['scale_to_meter'] = 1.0
    viewer = rebase_plan_geometry_for_viewer(normalized, display_source_name=display_name, display_source_sha256=source_hash, deduplicate=deduplicate)
    viewer['source'] = copy.deepcopy(source)
    viewer['provenance'] = {'uploaded_source': copy.deepcopy(source)}
    viewer['scale_to_meter'] = float(scale)
    return viewer

def write_viewer_plan_geometry(plan_geometry: dict[str, Any], destination: str | Path) -> None:
    if plan_geometry.get('schema') != 'cca.plan-geometry.v1' or plan_geometry.get('purpose') != 'viewer-rebased':
        raise PlanDxfError('unsupported viewer plan geometry schema')
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plan_geometry, ensure_ascii=True, sort_keys=True, separators=(',', ':')) + '\n'
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='ascii', newline='\n', dir=output.parent, prefix=f'.{output.name}.', suffix='.tmp', delete=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
