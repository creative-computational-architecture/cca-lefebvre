
from __future__ import annotations
import hashlib
from pathlib import Path
from ingest.plan_dxf import import_plan_dxf, rebase_imported_plan_geometry_for_viewer, write_viewer_plan_geometry
from viewer.project_model import DXFPlanState
_UNIT_TO_METER = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}

class NativeImportError(ValueError):
    pass

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def import_dxf_plan_state(source: str | Path, *, source_unit: str, output_dir: str | Path, curve_tolerance_m: float=0.02) -> DXFPlanState:
    source_path = Path(source).resolve()
    unit = source_unit.strip().lower() if isinstance(source_unit, str) else ''
    if unit not in _UNIT_TO_METER:
        raise NativeImportError('DXF source unit must be m, cm, or mm')
    if source_path.suffix.lower() != '.dxf':
        raise NativeImportError('DXF plan input must use the .dxf extension')
    if not source_path.is_file():
        raise NativeImportError(f'DXF plan does not exist: {source_path}')
    source_sha_before = _sha256(source_path)
    imported = import_plan_dxf(source_path, source_unit=unit, curve_tolerance_m=curve_tolerance_m)
    if imported['source']['sha256'] != source_sha_before:
        raise NativeImportError('DXF source changed while it was being imported')
    viewer_geometry = rebase_imported_plan_geometry_for_viewer(imported)
    destination_root = Path(output_dir).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    derivation_key = hashlib.sha256(f'{source_sha_before}:{unit}:{curve_tolerance_m:.12g}'.encode('ascii')).hexdigest()[:12]
    destination = destination_root / f'plan-{source_sha_before[:16]}-{derivation_key}.json'
    write_viewer_plan_geometry(viewer_geometry, destination)
    source_sha_after = _sha256(source_path)
    if source_sha_after != source_sha_before:
        raise NativeImportError('DXF source changed while derived geometry was written')
    source_metadata = imported['source']
    segments = viewer_geometry['segments']
    return DXFPlanState(loaded=True, source_path=str(source_path), geometry_path=str(destination), source_name=source_path.name, source_unit=unit, source_to_meter=_UNIT_TO_METER[unit], visible=True, fit_status='READY_FOR_FIT', segment_count=len(segments), bounds_m=dict(viewer_geometry['bounds_m']), quality={'source_sha256_before': source_sha_before, 'source_sha256_after': source_sha_after, 'source_preserved': True, 'header_insunits': source_metadata.get('header_insunits'), 'unsupported_by_type': source_metadata.get('unsupported_by_type', {}), 'curve_tolerance_m': curve_tolerance_m, 'rebase': viewer_geometry.get('rebase', {})})
__all__ = ['NativeImportError', 'import_dxf_plan_state']
