
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from field import m0util
from field.make_color_clouds import ColorCloudResult, process_color_clouds
from viewer.raw_manifest import CURRENT_RAW_PROCESSING_SCHEMA
from geometry.ply_preserve import PlySchema, read_ply_schema, transform_ply_preserving_properties
SCHEMA = CURRENT_RAW_PROCESSING_SCHEMA
ALGORITHM = '3D_SCANNER_ALL_DATA_FULLFRAME_CONF_KEYFRAME_RGB_DEPTHFILL_V3'
DEFAULT_VOXEL_M = 0.05
DEPTH_FILL_JUSTIFICATION = 'Experimental display label for conf=2 and depth>=16.384 m. No source point is removed; the rule is not calibrated measurement evidence.'
_RUN_ID = re.compile('^[A-Za-z0-9][A-Za-z0-9._-]*$')
_HASH_CHUNK_BYTES = 1024 * 1024

def default_processed_output_root(*, frozen: bool | None=None, platform_name: str | None=None, home: str | Path | None=None) -> Path:
    override = os.environ.get('CCA_LEFEBVRE_PROCESSED_ROOT', '').strip()
    if override:
        return Path(override).expanduser().resolve()
    current_platform = sys.platform if platform_name is None else platform_name
    home_path = Path.home() if home is None else Path(home).expanduser()
    if current_platform == 'darwin':
        return (home_path / 'Library' / 'Application Support' / 'CCA Lefebvre' / 'processed').resolve()
    if current_platform == 'win32':
        base = Path(os.environ.get('LOCALAPPDATA', str(home_path)))
    else:
        base = Path(os.environ.get('XDG_DATA_HOME', str(home_path / '.local' / 'share')))
    return (base / 'CCA Lefebvre' / 'processed').resolve()
DEFAULT_OUTPUT_ROOT = default_processed_output_root()

class RawProcessingError(RuntimeError):
    pass

class ColorProcessor(Protocol):

    def __call__(self, run_dir: str | Path, *, voxel_m: float, chunk_frames: int, output_dir: str | Path, output_stem: str | None, progress: Callable[[str], None] | None) -> ColorCloudResult:
        ...

class PlyTransformer(Protocol):

    def __call__(self, source: str | Path, destination: str | Path, transform: object) -> PlySchema:
        ...

@dataclass(frozen=True, slots=True)
class AllDataInventory:
    run_dir: Path
    run_id: str
    frame_count: int
    depth_frame_count: int
    confidence_frame_count: int
    rgb_frame_count: int
    eligible_rgb_frame_count: int
    eligible_depth_confidence_frame_count: int = 0
    keyframe_count: int = 0

@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    file_count: int
    total_bytes: int
    tree_sha256: str
    category_counts: dict[str, int]
    category_sha256: dict[str, str]

@dataclass(frozen=True, slots=True)
class RawProcessingResult:
    run_id: str
    output_dir: Path
    rgb_zup_path: Path
    confidence_zup_path: Path
    manifest_path: Path
    point_count: int
    used_frame_count: int
    source_preserved: bool
    keyframe_count: int = 0
    rgb_assigned_point_count: int = 0
    rgb_unassigned_point_count: int = 0
    depth_fill_labelled_pixel_count: int = 0
    depth_fill_valid_pixel_count: int = 0
    depth_fill_voxel_count_share_ge_0_5: int = 0
    depth_correction: dict | None = None
    ray_recovery: dict | None = None

def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()

def validate_all_data_folder(run_dir: str | Path) -> AllDataInventory:
    source = _resolved(run_dir)
    if not source.is_dir():
        raise RawProcessingError(f'All Data folder does not exist: {source}')
    run_id = source.name
    if not _RUN_ID.fullmatch(run_id):
        raise RawProcessingError('All Data folder name must be a safe ASCII run id: letters, digits, dot, dash')
    try:
        frames = m0util.list_frames(source)
    except FileNotFoundError as error:
        raise RawProcessingError(str(error)) from error
    depth_count = sum((frame.depth_path is not None for frame in frames))
    confidence_count = sum((frame.conf_path is not None for frame in frames))
    rgb_count = sum((frame.rgb_path is not None for frame in frames))
    eligible_count = sum((frame.depth_path is not None and frame.conf_path is not None and (frame.rgb_path is not None) for frame in frames))
    depth_confidence_count = sum((frame.depth_path is not None and frame.conf_path is not None for frame in frames))
    keyframe_count = sum((frame.rgb_path is not None and frame.depth_path is not None for frame in frames))
    if not depth_count:
        raise RawProcessingError('All Data folder has no depth_*.png or depth_*.exr')
    if not confidence_count:
        raise RawProcessingError('All Data folder has no conf_*.png')
    if not rgb_count:
        raise RawProcessingError('All Data folder has no frame_*.jpg')
    if not eligible_count:
        raise RawProcessingError('All Data folder has no matching frame/depth/confidence/RGB index')
    return AllDataInventory(run_dir=source, run_id=run_id, frame_count=len(frames), depth_frame_count=depth_count, confidence_frame_count=confidence_count, rgb_frame_count=rgb_count, eligible_rgb_frame_count=eligible_count, eligible_depth_confidence_frame_count=depth_confidence_count, keyframe_count=keyframe_count)

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (block := stream.read(_HASH_CHUNK_BYTES)):
            digest.update(block)
    return digest.hexdigest()

def _source_category(path: Path) -> str:
    name = path.name
    if name.startswith('frame_') and path.suffix.lower() == '.json':
        return 'frame_json'
    if name.startswith('frame_') and path.suffix.lower() in {'.jpg', '.jpeg'}:
        return 'rgb'
    if name.startswith('depth_') and path.suffix.lower() in {'.png', '.exr'}:
        return 'depth'
    if name.startswith('conf_') and path.suffix.lower() == '.png':
        return 'confidence'
    if name == 'meta.json':
        return 'metadata'
    return 'other'

def snapshot_source(run_dir: str | Path) -> SourceSnapshot:
    source = _resolved(run_dir)
    files = sorted((path for path in source.rglob('*') if path.is_file()), key=lambda path: path.relative_to(source).as_posix())
    tree_digest = hashlib.sha256(b'cca.raw-source-tree.v1\x00')
    category_digests: dict[str, object] = {}
    category_counts: dict[str, int] = {}
    total_bytes = 0
    for path in files:
        relative = path.relative_to(source).as_posix()
        size = path.stat().st_size
        file_sha256 = _sha256_file(path)
        category = _source_category(path)
        category_counts[category] = category_counts.get(category, 0) + 1
        digest = category_digests.setdefault(category, hashlib.sha256(f'cca.raw-source-{category}.v1\x00'.encode('ascii')))
        entry = f'{relative}\x00{size}\x00{file_sha256}\n'.encode()
        tree_digest.update(entry)
        digest.update(entry)
        total_bytes += size
    return SourceSnapshot(file_count=len(files), total_bytes=total_bytes, tree_sha256=tree_digest.hexdigest(), category_counts=dict(sorted(category_counts.items())), category_sha256={category: digest.hexdigest() for category, digest in sorted(category_digests.items())})

def _snapshot_payload(snapshot: SourceSnapshot) -> dict[str, object]:
    return {'file_count': snapshot.file_count, 'total_bytes': snapshot.total_bytes, 'tree_sha256': snapshot.tree_sha256, 'category_counts': snapshot.category_counts, 'category_sha256': snapshot.category_sha256}

def _output_payload(path: Path, schema: PlySchema) -> dict[str, object]:
    return {'file': path.name, 'sha256': _sha256_file(path), 'bytes': path.stat().st_size, 'vertex_count': schema.vertex.count, 'format': schema.format, 'properties': [property_.name for property_ in schema.vertex.properties]}

def _write_ascii_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + '\n').encode('ascii')
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise

def process_raw_all_data(run_dir: str | Path, *, output_root: str | Path | None=None, voxel_m: float=DEFAULT_VOXEL_M, chunk_frames: int=80, progress: Callable[[str], None] | None=None, depth_correction: str | Path | None=None, ray_recovery: str | Path | None=None, color_processor: ColorProcessor=process_color_clouds, ply_transformer: PlyTransformer=transform_ply_preserving_properties) -> RawProcessingResult:
    if not math.isclose(voxel_m, DEFAULT_VOXEL_M, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('raw processing voxel_m is fixed at 0.05 metre')
    if chunk_frames < 1:
        raise ValueError('chunk_frames must be at least one')
    inventory = validate_all_data_folder(run_dir)
    depth_model = None
    model_sha256 = None
    if depth_correction is not None:
        from viewer.correction_models import load_run_model
        depth_model, model_sha256 = load_run_model(depth_correction, inventory.run_id)
    recovery_model = None
    if ray_recovery is not None:
        from viewer.ray_recovery_service import load_recovery
        recovery_model = load_recovery(ray_recovery, inventory.run_dir)
    processed_root = default_processed_output_root() if output_root is None else _resolved(output_root)
    target = (processed_root / inventory.run_id).resolve()
    if model_sha256 is not None:
        target = (processed_root / 'depth-corrected' / model_sha256 / inventory.run_id).resolve()
    if recovery_model is not None:
        recovery_key = hashlib.sha256(((model_sha256 or 'RAW') + recovery_model['provenance']['sha256']).encode('ascii')).hexdigest()
        target = (processed_root / 'ray-recovered' / recovery_key / inventory.run_id).resolve()
    if target == inventory.run_dir or inventory.run_dir in target.parents:
        raise RawProcessingError('processed output must be outside the raw source folder')

    def emit(message: str) -> None:
        if progress is not None:
            progress(message)
    emit('Hashing raw source before processing')
    source_before = snapshot_source(inventory.run_dir)
    target.mkdir(parents=True, exist_ok=True)
    rgb_zup = target / f'{inventory.run_id}_rgb_5cm_zup.ply'
    confidence_zup = target / f'{inventory.run_id}_confcolor_5cm_zup.ply'
    manifest_path = target / 'processing-manifest.json'
    with tempfile.TemporaryDirectory(prefix='.raw-yup-', dir=target) as temporary_dir:
        emit('Building RGB and confidence point clouds at 5 cm')
        color_result = color_processor(inventory.run_dir, voxel_m=voxel_m, chunk_frames=chunk_frames, output_dir=temporary_dir, output_stem=inventory.run_id, progress=progress, **{'depth_correction': depth_model} if depth_model is not None else {}, **{'ray_recovery': recovery_model} if recovery_model is not None else {})
        if color_result.voxel_m != voxel_m:
            raise RawProcessingError('color processor changed the requested voxel size')
        if color_result.point_count < 1:
            raise RawProcessingError('color processor returned an empty point cloud')
        if color_result.depth_correction != m0util.depth_correction_provenance(depth_model):
            raise RawProcessingError('color processor did not apply the requested depth model')
        if recovery_model and (not color_result.ray_recovery or color_result.ray_recovery.get('sha256') != recovery_model['provenance']['sha256']):
            raise RawProcessingError('color processor did not apply the requested ray recovery')
        emit('Transforming ARKit Y-up outputs to CAD Z-up')
        rgb_yup_schema = read_ply_schema(color_result.rgb_path)
        confidence_yup_schema = read_ply_schema(color_result.confidence_path)
        if rgb_yup_schema.vertex.count != color_result.point_count:
            raise RawProcessingError('RGB output point count differs from processor result')
        if confidence_yup_schema.vertex.count != color_result.point_count:
            raise RawProcessingError('confidence output point count differs from processor result')
        if rgb_yup_schema.vertex_signature != confidence_yup_schema.vertex_signature:
            raise RawProcessingError('RGB and confidence outputs have different schemas')
        ply_transformer(color_result.rgb_path, rgb_zup, m0util.ARKIT_WORLD_TO_ZUP)
        ply_transformer(color_result.confidence_path, confidence_zup, m0util.ARKIT_WORLD_TO_ZUP)
    rgb_schema = read_ply_schema(rgb_zup)
    confidence_schema = read_ply_schema(confidence_zup)
    if rgb_schema.vertex_signature != confidence_schema.vertex_signature:
        raise RawProcessingError('Z-up RGB and confidence schemas differ')
    if rgb_schema.vertex.count != color_result.point_count:
        raise RawProcessingError('Z-up RGB point count changed during transform')
    if confidence_schema.vertex.count != color_result.point_count:
        raise RawProcessingError('Z-up confidence point count changed during transform')
    emit('Hashing raw source after processing')
    source_after = snapshot_source(inventory.run_dir)
    source_preserved = source_before == source_after
    keyframe_count = int(getattr(color_result, 'keyframe_count', inventory.keyframe_count))
    rgb_assigned = int(getattr(color_result, 'rgb_assigned_point_count', 0))
    rgb_unassigned = int(getattr(color_result, 'rgb_unassigned_point_count', 0))
    keyframe_limit = int(color_result.max_keyframe_candidates)
    visibility_tolerance_floor_m = float(color_result.visibility_tolerance_floor_m)
    visibility_tolerance_quadratic_m_per_m2 = float(color_result.visibility_tolerance_quadratic_m_per_m2)
    unassigned_rgb = [int(value) for value in color_result.unassigned_rgb]
    unassigned_keyframe_distance = float(color_result.unassigned_keyframe_distance)
    depth_fill_prior_m = float(color_result.depth_fill_prior_m)
    depth_fill_conf = int(color_result.depth_fill_conf)
    depth_fill_labelled_pixels = int(color_result.depth_fill_labelled_pixel_count)
    depth_fill_valid_pixels = int(color_result.depth_fill_valid_pixel_count)
    depth_fill_voxels_ge_half = int(color_result.depth_fill_voxel_count_share_ge_0_5)
    depth_fill_property = str(color_result.depth_fill_property)
    for schema in (rgb_schema, confidence_schema):
        if depth_fill_property not in {p.name for p in schema.vertex.properties}:
            raise RawProcessingError(f'processed output lacks the {depth_fill_property} label property')
    payload: dict[str, object] = {'schema': SCHEMA, 'run_id': inventory.run_id, 'unit': 'metre', 'axis': 'CAD_Z_UP', 'scale': 1.0, 'no_fixed_depth_cutoff': True, 'source_preserved': source_preserved, 'processing': {'algorithm': ALGORITHM, 'voxel_m': float(voxel_m), 'depth_cutoff_m': None, 'no_fixed_depth_cutoff': True, 'valid_depth_policy': 'finite_and_above_sensor_invalid_floor', 'sensor_invalid_floor_m': float(m0util.MIN_VALID_DEPTH_M), 'source_preserved': source_preserved, 'coordinate_transform': 'ARKIT_Y_UP_TO_CAD_Z_UP_X_NEGZ_Y', 'depth_correction': color_result.depth_correction, 'depth_correction_sha256': model_sha256, 'ray_recovery': color_result.ray_recovery, 'geometry_frames': 'all_depth_confidence_frames', 'rgb_source': {'policy': 'nearest_keyframe_passing_occlusion_test', 'max_keyframe_candidates': keyframe_limit, 'visibility_tolerance_m': 'max(floor, quadratic * keyframe_depth^2)', 'visibility_tolerance_floor_m': visibility_tolerance_floor_m, 'visibility_tolerance_quadratic_m_per_m2': visibility_tolerance_quadratic_m_per_m2, 'unassigned_rgb': unassigned_rgb, 'unassigned_marker': {'property': 'rgb_kf_dist', 'value': unassigned_keyframe_distance}, 'no_point_rejected': True}, 'depth_fill': {'rule': f'conf=={depth_fill_conf} and depth_m>=prior', 'prior_m': depth_fill_prior_m, 'conf': depth_fill_conf, 'labelled_pixels': depth_fill_labelled_pixels, 'total_valid_pixels': depth_fill_valid_pixels, 'labelled_voxels_share_ge_0_5': depth_fill_voxels_ge_half, 'justification': DEPTH_FILL_JUSTIFICATION}}, 'frames': {'indexed': inventory.frame_count, 'with_depth': inventory.depth_frame_count, 'with_confidence': inventory.confidence_frame_count, 'with_rgb': inventory.rgb_frame_count, 'eligible_rgb_depth_confidence': inventory.eligible_rgb_frame_count, 'eligible_depth_confidence': inventory.eligible_depth_confidence_frame_count, 'rgb_keyframes': keyframe_count, 'used': color_result.used_frame_count, 'skipped_eligible': color_result.skipped_frame_count}, 'points': {'total': int(color_result.point_count), 'rgb_assigned': rgb_assigned, 'rgb_unassigned': rgb_unassigned}, 'source': {'path': str(inventory.run_dir), 'before': _snapshot_payload(source_before), 'after': _snapshot_payload(source_after)}, 'outputs': {'rgb': _output_payload(rgb_zup, rgb_schema), 'confidence': _output_payload(confidence_zup, confidence_schema)}}
    _write_ascii_json(manifest_path, payload)
    if depth_model is not None:
        _write_ascii_json(target / 'depth-correction-model.json', {key: value for key, value in depth_model.items() if not key.startswith('_')})
    if ray_recovery is not None:
        import shutil
        snapshot = target / 'ray-recovery.json'
        if Path(ray_recovery).resolve() != snapshot.resolve():
            shutil.copyfile(ray_recovery, snapshot)
    if not source_preserved:
        raise RawProcessingError(f'raw source changed during processing; rejected manifest: {manifest_path}')
    emit(f'Processed {color_result.point_count:,} points -> {target}')
    return RawProcessingResult(run_id=inventory.run_id, output_dir=target, rgb_zup_path=rgb_zup, confidence_zup_path=confidence_zup, manifest_path=manifest_path, point_count=color_result.point_count, used_frame_count=color_result.used_frame_count, source_preserved=True, keyframe_count=keyframe_count, rgb_assigned_point_count=rgb_assigned, rgb_unassigned_point_count=rgb_unassigned, depth_fill_labelled_pixel_count=depth_fill_labelled_pixels, depth_fill_valid_pixel_count=depth_fill_valid_pixels, depth_fill_voxel_count_share_ge_0_5=depth_fill_voxels_ge_half, depth_correction=color_result.depth_correction, ray_recovery=color_result.ray_recovery)
__all__ = ['DEFAULT_OUTPUT_ROOT', 'DEFAULT_VOXEL_M', 'AllDataInventory', 'RawProcessingError', 'RawProcessingResult', 'default_processed_output_root', 'process_raw_all_data', 'snapshot_source', 'validate_all_data_folder']
