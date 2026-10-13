
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from geometry.rolove import placed_matrix, surfaces_in_world, transform_points

def read_scan(scan, source_layer_key=None):
    from export.survey_pack import _read_points
    from viewer.survey_service import _resolve_survey_layer
    if source_layer_key is None:
        layer, source = _resolve_survey_layer(scan)
    else:
        layer = scan.layer(source_layer_key)
        if layer is None:
            raise ValueError('Secilen kaynak katmani bulunamadi')
        source = Path(layer.path).expanduser().resolve()
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    points, rgb, _ = _read_points(source)
    axis = layer.metadata.get('source_axis', 'CAD_Z_up')
    from geometry.survey_frame import _AXIS_ALIASES, _AXIS_ROTATIONS
    if axis not in _AXIS_ALIASES:
        raise ValueError(f'Kaynak ekseni taninmiyor: {axis}')
    basis = np.eye(4)
    basis[:3, :3] = _AXIS_ROTATIONS[_AXIS_ALIASES[axis]]
    world = placed_matrix(scan.transform, scan.metadata) @ basis
    return (transform_points(points, world), rgb, source, digest, world)

def auto_alignment(scan, source_layer_key=None):
    from geometry.survey_frame import estimate_survey_frame, frame_to_dict, normalize_source_points, orientation_to_dict
    points, _rgb, _source, _digest, _world = read_scan(scan, source_layer_key)
    normalized, orientation = normalize_source_points(points, 'auto')
    frame = estimate_survey_frame(normalized)
    return {'delta': (np.asarray(frame.transform_survey_from_zup) @ np.asarray(orientation.transform_zup_from_source)).tolist(), 'method': 'AUTO_GROUND', 'orientation': orientation_to_dict(orientation), 'ground_frame': frame_to_dict(frame)}

def automatic_surfaces(scan, source_layer_key=None, config=None):
    from geometry.rolove_analysis import propose_surfaces
    points, _rgb, source, digest, _world = read_scan(scan, source_layer_key)
    result = propose_surfaces(points, config)
    result['source_sha256'] = digest
    result['source'] = str(source)
    return result

def analyze_scan_plan(scan, surface, source_layer_key=None, config=None):
    from geometry.rolove_analysis import analyze_floor_plan
    points, _rgb, source, digest, world = read_scan(scan, source_layer_key)
    return {'analysis': analyze_floor_plan(points, surface, config), 'source': str(source), 'source_hash': digest, 'transform': world.tolist()}

def export_scan_projections(scan, output_dir, source_layer_key=None):
    from export.rolove_projection import export_projection_pack
    source_layer_key = source_layer_key or scan.metadata.get('rolove', {}).get('source_layer_key')
    surfaces = surfaces_in_world(scan.transform, scan.metadata)
    if not surfaces:
        raise ValueError('Once en az bir izdusum yuzeyi ekleyin')
    points, rgb, source, digest, world = read_scan(scan, source_layer_key)
    return export_projection_pack(points, rgb, surfaces, output_dir, source=source, source_hash=digest, transform=world, alignment_evidence=scan.metadata.get('rolove', {}).get('alignment_evidence', {}))
