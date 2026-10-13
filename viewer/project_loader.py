
from __future__ import annotations
import copy
import json
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from geometry.ply_preserve import PlyFormatError, read_ply_schema
from geometry.temporal_sequence import apply_temporal_order
from viewer.project_model import PipelineStage, PipelineState, ProjectModel, ScanEntry, ScanLayer
from viewer.raw_manifest import CURRENT_RAW_PROCESSING_SCHEMA, is_supported_raw_processing_schema
TIMESTAMP_RUN_PATTERN = re.compile('^\\d{4}_\\d{2}_\\d{2}_\\d{2}_\\d{2}_\\d{2}$')
PORTABLE_PROJECT_SCHEMA = 'cca.lefebvre-survey-project.v1'
_PROCESSING_SCHEMA = CURRENT_RAW_PROCESSING_SCHEMA
_PROJECT_NAME = 'Untitled Survey'
_APP_FUSED_ROLES = frozenset({'PRIMARY_APP_FUSED_SURFACE', 'APP_FUSED_SURFACE'})
_REVIEW_LAYER_KEYS = frozenset((f'{mode}/{band}' for mode in ('rgb', 'confcolor') for band in ('core', 'body', 'shell')))
_GENERATED_SCAN_STATUSES = {'CORRECTION_READY', 'APP_FUSED_READY', 'RAW_CAPTURE_READY'}

class SurveyProjectError(ValueError):
    pass

def is_timestamped_run_id(value: str) -> bool:
    return isinstance(value, str) and TIMESTAMP_RUN_PATTERN.fullmatch(value) is not None



def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SurveyProjectError(f'{label} must be an object')
    return value

def _exact_one(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SurveyProjectError(f'{label} must be numeric')
    result = float(value)
    if not math.isclose(result, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SurveyProjectError(f'{label} must be 1.0')
    return result

def _safe_output_path(manifest_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise SurveyProjectError(f'{label} must be one local file name')
    path = (manifest_path.parent / value).resolve(strict=False)
    if path.parent != manifest_path.parent.resolve(strict=False) or not path.is_file():
        raise SurveyProjectError(f'{label} output is missing')
    return path

def _validated_corrected_layers(manifest_path: Path, run_id: str) -> tuple[list[ScanLayer], dict[str, Any]]:
    try:
        payload = json.loads(manifest_path.read_text(encoding='ascii'))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as error:
        raise SurveyProjectError('cannot read raw processing manifest') from error
    root = _mapping(payload, 'raw processing manifest')
    processing = _mapping(root.get('processing'), 'raw processing contract')
    outputs = _mapping(root.get('outputs'), 'raw processing outputs')
    if not is_supported_raw_processing_schema(root.get('schema')) or root.get('run_id') != run_id:
        raise SurveyProjectError('raw processing manifest identity does not match')
    if root.get('axis') != 'CAD_Z_UP' or root.get('unit') != 'metre':
        raise SurveyProjectError('corrected evidence must use metre CAD Z-up')
    _exact_one(root.get('scale'), 'raw processing scale')
    if root.get('source_preserved') is not True or root.get('no_fixed_depth_cutoff') is not True or processing.get('source_preserved') is not True or (processing.get('no_fixed_depth_cutoff') is not True) or (processing.get('depth_cutoff_m') is not None):
        raise SurveyProjectError('raw processing contract is unsafe')
    rgb = _mapping(outputs.get('rgb'), 'corrected RGB output')
    confidence = _mapping(outputs.get('confidence'), 'corrected confidence output')
    try:
        rgb_count = int(rgb.get('vertex_count', 0))
        confidence_count = int(confidence.get('vertex_count', 0))
    except (TypeError, ValueError) as error:
        raise SurveyProjectError('corrected point count is invalid') from error
    if rgb_count < 1 or rgb_count != confidence_count:
        raise SurveyProjectError('corrected RGB/confidence point counts differ')
    rgb_path = _safe_output_path(manifest_path, rgb.get('file'), 'corrected RGB')
    confidence_path = _safe_output_path(manifest_path, confidence.get('file'), 'corrected confidence')
    try:
        rgb_schema = read_ply_schema(rgb_path)
        confidence_schema = read_ply_schema(confidence_path)
    except (OSError, PlyFormatError, ValueError) as error:
        raise SurveyProjectError('corrected PLY output is invalid') from error
    if rgb_schema.vertex.count != rgb_count or confidence_schema.vertex.count != confidence_count or rgb_schema.vertex_signature != confidence_schema.vertex_signature:
        raise SurveyProjectError('corrected PLY schema does not match its manifest')
    voxel_value = processing.get('voxel_m')
    if isinstance(voxel_value, bool) or not isinstance(voxel_value, (int, float)):
        raise SurveyProjectError('raw processing voxel size is invalid')
    voxel_m = float(voxel_value)
    if not math.isfinite(voxel_m) or voxel_m <= 0.0:
        raise SurveyProjectError('raw processing voxel size is invalid')
    depth_correction = copy.deepcopy(processing.get('depth_correction'))
    ray_recovery = copy.deepcopy(processing.get('ray_recovery'))
    if isinstance(depth_correction, Mapping) and isinstance(ray_recovery, Mapping):
        family = 'recovered'
    elif isinstance(depth_correction, Mapping):
        family = 'corrected'
    else:
        family = 'raw'
    layer_keys = {'raw': ('rgb/raw', 'confcolor/raw'), 'corrected': ('rgb/corrected', 'confcolor/corrected'), 'recovered': ('rgb/recovered', 'confcolor/recovered')}[family]
    layer_roles = {'raw': ('RAW_FRAME_RGB_EVIDENCE', 'RAW_FRAME_CONFIDENCE_EVIDENCE'), 'corrected': ('CORRECTED_RAW_RGB', 'CORRECTED_RAW_CONFIDENCE'), 'recovered': ('RECOVERED_CORRECTED_RGB', 'RECOVERED_CORRECTED_CONFIDENCE')}[family]
    metadata = {'source_axis': 'CAD_Z_up', 'source_unit': 'metre', 'source_preserved': True, 'distance_cutoff_m': None, 'processing_manifest': str(manifest_path), 'voxel_m': voxel_m, 'depth_correction': depth_correction, 'ray_recovery': ray_recovery, 'evidence_family': family}
    layers = [ScanLayer(key=layer_keys[0], path=str(rgb_path), role=layer_roles[0], point_count=rgb_count, visible=True, metadata=metadata), ScanLayer(key=layer_keys[1], path=str(confidence_path), role=layer_roles[1], point_count=confidence_count, visible=True, metadata=metadata)]
    correction = {'manifest': str(manifest_path), 'point_count': rgb_count, 'voxel_m': voxel_m, 'source_preserved': True, 'distance_cutoff_m': None, 'output_axis': 'CAD_Z_up', 'depth_correction': depth_correction, 'ray_recovery': ray_recovery, 'evidence_family': family, 'used_frame_count': root.get('frames', {}).get('used', 0)}
    return (layers, correction)

def _processed_manifest_candidates(run_id: str, processed_roots: Sequence[str | Path]) -> tuple[Path, ...]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root_value in processed_roots:
        root = Path(root_value).expanduser().resolve(strict=False)
        for manifest in (root / run_id / 'processing-manifest.json', *sorted(root.glob(f'depth-corrected/*/{run_id}/processing-manifest.json'))):
            resolved = manifest.resolve(strict=False)
            if resolved not in seen and resolved.is_file():
                seen.add(resolved)
                candidates.append(resolved)
    return tuple(candidates)

def _first_corrected_evidence(run_id: str, processed_roots: Sequence[str | Path]) -> tuple[list[ScanLayer], dict[str, Any] | None, list[dict[str, str]]]:
    rejected: list[dict[str, str]] = []
    by_family: dict[str, tuple[list[ScanLayer], dict[str, Any]]] = {}
    for manifest in _processed_manifest_candidates(run_id, processed_roots):
        try:
            layers, correction = _validated_corrected_layers(manifest, run_id)
        except SurveyProjectError as error:
            rejected.append({'manifest': str(manifest), 'reason': str(error)})
            continue
        family = str(correction['evidence_family'])
        if family in {'raw', 'corrected'} and family not in by_family:
            by_family[family] = (layers, correction)
    layers = [layer for family in ('raw', 'corrected') for layer in by_family.get(family, ([], {}))[0]]
    correction = by_family.get('corrected', by_family.get('raw', ([], None)))[1]
    return (layers, correction, rejected)

def _app_fused_layer(run_dir: Path) -> tuple[ScanLayer | None, str | None]:
    points_path = run_dir / 'points.ply'
    if not points_path.is_file():
        return (None, None)
    try:
        schema = read_ply_schema(points_path)
    except (OSError, PlyFormatError, ValueError) as error:
        return (None, str(error))
    if schema.vertex.count < 1:
        return (None, 'app points.ply has zero vertices')
    return (ScanLayer(key='rgb/full', path=str(points_path.resolve()), role='PRIMARY_APP_FUSED_SURFACE', point_count=schema.vertex.count, visible=False, metadata={'source_axis': 'ARKit_Y_up', 'source_unit': 'metre', 'source_preserved': True, 'distance_cutoff_m': None}), None)

def build_scanner_app_scan_entry(run_dir: str | Path, *, processed_roots: Sequence[str | Path]=(), rail_order: int=0) -> ScanEntry:
    if isinstance(rail_order, bool) or not isinstance(rail_order, int):
        raise SurveyProjectError('scan rail order must be an integer')
    if rail_order < 0:
        raise SurveyProjectError('scan rail order cannot be negative')
    source = Path(run_dir).expanduser().resolve()
    from viewer.raw_processing_service import validate_all_data_folder
    inventory = validate_all_data_folder(source)
    corrected_layers, correction, rejected = _first_corrected_evidence(inventory.run_id, processed_roots)
    app_layer, app_error = _app_fused_layer(source)
    layers = list(corrected_layers)
    if app_layer is not None:
        layers.append(app_layer)
    metadata: dict[str, Any] = {'raw_all_data': {'path': str(source), 'frame_count': inventory.frame_count, 'depth_frame_count': inventory.depth_frame_count, 'confidence_frame_count': inventory.confidence_frame_count, 'rgb_frame_count': inventory.rgb_frame_count, 'eligible_rgb_frame_count': inventory.eligible_rgb_frame_count, 'source_axis': 'ARKit_Y_up', 'source_preserved': True, 'distance_cutoff_m': None}}
    if correction is not None:
        metadata['raw_correction'] = correction
        status = 'CORRECTION_READY'
    elif app_layer is not None:
        status = 'APP_FUSED_READY'
    else:
        status = 'RAW_CAPTURE_READY'
    if rejected:
        metadata['rejected_processing_manifests'] = rejected
    if app_error:
        metadata['app_fused_error'] = app_error
    return ScanEntry(scan_id=inventory.run_id, rail_order=rail_order, source_path=str(source), status=status, layers=layers, visible=True, metadata=metadata)

def build_survey_project(raw_root: str | Path, *, processed_roots: Sequence[str | Path]=()) -> ProjectModel:
    source_root = Path(raw_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f'Survey raw root does not exist: {source_root}')
    run_dirs = tuple(sorted((path for path in source_root.iterdir() if path.is_dir() and is_timestamped_run_id(path.name)), key=lambda path: path.name))
    if not run_dirs:
        raise FileNotFoundError(f'no timestamped All Data runs found: {source_root}')
    scans: list[ScanEntry] = []
    corrected_count = 0
    fused_count = 0
    raw_only_count = 0
    rejected_manifests: list[dict[str, str]] = []
    for index, run_dir in enumerate(run_dirs):
        scan = build_scanner_app_scan_entry(run_dir, processed_roots=processed_roots, rail_order=index)
        rejected = scan.metadata.get('rejected_processing_manifests', [])
        if isinstance(rejected, list):
            rejected_manifests.extend(rejected)
        correction = scan.metadata.get('raw_correction')
        app_layer = scan.layer('rgb/full')
        if correction is not None:
            corrected_count += 1
        if app_layer is not None:
            fused_count += 1
        if not scan.layers:
            raw_only_count += 1
        scans.append(scan)
    project = ProjectModel(project_id=uuid.uuid4().hex, name=_PROJECT_NAME, scans=scans, pipeline=PipelineState.at(PipelineStage.REVIEW if corrected_count or fused_count else PipelineStage.VALIDATE), metadata={'project_kind': 'SURVEY', 'portable_project_schema': PORTABLE_PROJECT_SCHEMA, 'raw_data_root': str(source_root), 'initial_evidence_profile': 'CORRECTED RGB' if any((scan.layer('rgb/corrected') is not None for scan in scans)) else 'RAW RGB' if corrected_count else 'PROCESSED FULL' if fused_count else None, 'initial_scan_layout': 'DISPLAY_ONLY_INVENTORY', 'evidence_inventory': {'scan_count': len(scans), 'corrected_scan_count': corrected_count, 'app_fused_scan_count': fused_count, 'raw_only_scan_count': raw_only_count, 'rejected_processing_manifests': rejected_manifests}, 'point_policy': {'raw_points_deleted': False, 'distance_cutoff_m': None}})
    apply_temporal_order(project, source_root)
    project.validate()
    return project

def _preserved_app_fused_layers(previous: ScanEntry, refreshed: ScanEntry, base: Path) -> list[ScanLayer]:

    def resolved(layer: ScanLayer) -> Path | None:
        if not layer.path:
            return None
        path = Path(layer.path).expanduser()
        return (path if path.is_absolute() else base / path).resolve(strict=False)
    raw_paths = {path for scan in (previous, refreshed) for layer in scan.layers if layer.key in {'rgb/raw', 'confcolor/raw', 'rgb/corrected', 'confcolor/corrected', 'rgb/recovered', 'confcolor/recovered'} and (path := resolved(layer)) is not None}
    retained = []
    for layer in previous.layers:
        if layer.key not in {'rgb/full', 'rgb/full_zup'} or layer.role not in _APP_FUSED_ROLES:
            continue
        if layer.metadata.get('source_axis') not in {'CAD_Z_up', 'ARKit_Y_up'}:
            continue
        source = resolved(layer)
        if source is None or source in raw_paths:
            continue
        try:
            count = read_ply_schema(source).vertex.count
        except (OSError, PlyFormatError, ValueError):
            continue
        if count < 1 or layer.point_count not in {None, count}:
            continue
        retained.append(copy.deepcopy(layer))
    return retained

def refresh_survey_project(existing: ProjectModel, raw_root: str | Path, *, processed_roots: Sequence[str | Path]=()) -> ProjectModel:
    if existing.metadata.get('portable_project_schema') != PORTABLE_PROJECT_SCHEMA:
        raise SurveyProjectError('project is not a portable survey index')
    refreshed = build_survey_project(raw_root, processed_roots=processed_roots)
    existing_scans = {scan.scan_id: scan for scan in existing.scan_rail}
    base = Path(existing.source_manifest).parent if existing.source_manifest else Path.cwd()
    for scan in refreshed.scan_rail:
        previous = existing_scans.get(scan.scan_id)
        if previous is None:
            continue
        saved_correction = previous.metadata.get('raw_correction', {})
        if scan.layer('rgb/corrected') is None and isinstance(saved_correction, Mapping) and (saved_correction.get('depth_correction') or saved_correction.get('ray_recovery')):
            manifest_value = saved_correction.get('manifest')
            if not isinstance(manifest_value, str) or not manifest_value.strip():
                raise SurveyProjectError(f'{scan.scan_id}: saved depth model manifest is missing')
            manifest = Path(manifest_value).expanduser()
            if not manifest.is_absolute():
                manifest = base / manifest
            model_layers, model_record = _validated_corrected_layers(manifest.resolve(), scan.scan_id)
            if model_record.get('depth_correction') != saved_correction.get('depth_correction') or model_record.get('ray_recovery') != saved_correction.get('ray_recovery'):
                raise SurveyProjectError(f'{scan.scan_id}: saved depth model provenance changed')
            model_keys = {layer.key for layer in model_layers}
            scan.layers = [layer for layer in scan.layers if layer.key not in model_keys]
            scan.layers.extend(model_layers)
            scan.metadata['raw_correction'] = model_record
            scan.status = 'CORRECTION_READY'
        elif isinstance(saved_correction, Mapping) and saved_correction.get('ray_recovery'):
            scan.metadata['saved_recovery_result'] = copy.deepcopy(saved_correction)
        app_layers = _preserved_app_fused_layers(previous, scan, base)
        app_keys = {layer.key for layer in app_layers}
        scan.layers = [layer for layer in scan.layers if layer.key not in app_keys] + app_layers
        scan.layers.extend((copy.deepcopy(layer) for layer in previous.layers if layer.key in _REVIEW_LAYER_KEYS))
        if app_layers and scan.status == 'RAW_CAPTURE_READY':
            scan.status = 'APP_FUSED_READY'
        scan.transform = copy.deepcopy(previous.transform)
        scan.group_id = previous.group_id
        scan.visible = previous.visible
        scan.placement_locked = previous.placement_locked
        scan.quality = copy.deepcopy(previous.quality)
        metadata = copy.deepcopy(previous.metadata)
        metadata.update(copy.deepcopy(scan.metadata))
        scan.metadata = metadata
        if previous.status not in _GENERATED_SCAN_STATUSES:
            scan.status = previous.status
    discovered = {scan.scan_id: scan for scan in refreshed.scans}
    retained_order = [discovered.pop(scan.scan_id, copy.deepcopy(scan)) for scan in existing.scan_rail]
    refreshed.scans = retained_order + list(discovered.values())
    for index, scan in enumerate(refreshed.scans):
        scan.rail_order = index
    refreshed.project_id = existing.project_id
    refreshed.name = existing.name
    refreshed.groups = copy.deepcopy(existing.groups)
    refreshed._sync_groups()
    refreshed.dxf_plan = copy.deepcopy(existing.dxf_plan)
    refreshed.pipeline = copy.deepcopy(existing.pipeline)
    refreshed.source_manifest = existing.source_manifest
    fused_count = sum((any((scan.layer(key) is not None for key in ('rgb/full', 'rgb/full_zup'))) for scan in refreshed.scan_rail))
    inventory = refreshed.metadata['evidence_inventory']
    inventory['app_fused_scan_count'] = fused_count
    inventory['raw_only_scan_count'] = sum((not scan.layers for scan in refreshed.scan_rail))
    refreshed.metadata['initial_evidence_profile'] = 'CORRECTED RGB' if any((scan.layer('rgb/corrected') is not None for scan in refreshed.scan_rail)) else 'RAW RGB' if inventory['corrected_scan_count'] else 'PROCESSED FULL' if fused_count else None
    if 'initial_evidence_profile' in existing.metadata and existing.metadata['initial_evidence_profile'] != 'REVIEW CORE':
        refreshed.metadata['initial_evidence_profile'] = copy.deepcopy(existing.metadata['initial_evidence_profile'])
    dynamic_keys = {'accepted_alignment_placements', 'evidence_inventory', 'initial_evidence_profile', 'initial_visible_scan_count', 'point_policy', 'portable_project_schema', 'project_kind', 'raw_data_root', 'temporal_sequence'}
    metadata = copy.deepcopy(existing.metadata)
    for key in dynamic_keys:
        if key in refreshed.metadata:
            metadata[key] = copy.deepcopy(refreshed.metadata[key])
        else:
            metadata.pop(key, None)
    refreshed.metadata = metadata
    refreshed.validate()
    return refreshed
__all__ = ['PORTABLE_PROJECT_SCHEMA', 'TIMESTAMP_RUN_PATTERN', 'SurveyProjectError', 'build_survey_project', 'build_scanner_app_scan_entry', 'is_timestamped_run_id', 'refresh_survey_project']
