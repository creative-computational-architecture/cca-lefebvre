
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import numpy as np
from geometry.ply_preserve import iter_ply_vertex_chunks, read_ply_schema
SCHEMA = 'cca.revit-handoff.v1'
Y_TO_Z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
XYZ = ('cartesianX', 'cartesianY', 'cartesianZ')
RGB = ('colorRed', 'colorGreen', 'colorBlue')

class RevitPackError(ValueError):
    pass

def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + '\n', encoding='ascii')

def rigid_transform(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-09, rtol=0)) or (not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-07, rtol=0)) or (not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-07, rtol=0)):
        raise RevitPackError('Only finite rigid transforms at scale 1 are accepted')
    return matrix

def resolve_source(scan: Mapping, base: Path) -> dict:
    layers = {layer['key']: layer for layer in scan.get('layers', [])}
    layer = layers.get('rgb/raw') or layers.get('rgb/full_zup') or layers.get('rgb/full')
    if layer is None:
        raise RevitPackError(f"{scan['scan_id']}: full RGB source missing")
    metadata = copy.deepcopy(layer.get('metadata', {}))
    source = Path(layer['path'])
    if not source.is_absolute():
        source = base / source
    replacement = None
    manifest = None
    if metadata.get('processing_manifest'):
        manifest_path = Path(metadata['processing_manifest'])
        if not manifest_path.is_absolute():
            manifest_path = base / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('run_id') != scan['scan_id']:
            raise RevitPackError('Processing manifest belongs to another scan')
        if metadata.get('ray_recovery') or manifest.get('processing', {}).get('ray_recovery'):
            model_hash = manifest.get('processing', {}).get('depth_correction_sha256', '')
            roots = [parent for parent in manifest_path.parents if parent.name == 'processed']
            if not roots or not re.fullmatch('[a-fA-F0-9]{64}', model_hash):
                raise RevitPackError('Recovered-ray source needs its observed depth-correction baseline')
            baseline = roots[0] / 'depth-corrected' / model_hash / scan['scan_id'] / 'processing-manifest.json'
            observed = json.loads(baseline.read_text(encoding='utf-8'))
            if observed.get('run_id') != scan['scan_id'] or observed.get('processing', {}).get('ray_recovery') or observed.get('processing', {}).get('depth_correction_sha256') != model_hash or (observed.get('source', {}).get('before', {}).get('tree_sha256') != manifest.get('source', {}).get('before', {}).get('tree_sha256')):
                raise RevitPackError('Observed baseline lineage does not match recovered source')
            replacement = {'excluded_hypothesis': str(source), 'reason': 'RAY_RECOVERY_NOT_MEASUREMENT'}
            manifest, manifest_path = (observed, baseline)
        output = manifest['outputs']['rgb']
        output_name = output['file']
        if Path(output_name).name != output_name:
            raise RevitPackError('Processing output must be a local file name')
        declared_source = manifest_path.parent / output_name
        if replacement is None and source.resolve() != declared_source.resolve():
            raise RevitPackError('Layer path differs from processing manifest RGB source')
        source = declared_source
        metadata['processing_manifest'] = str(manifest_path)
        metadata['ray_recovery'] = None
        metadata['depth_correction'] = manifest.get('processing', {}).get('depth_correction')
    elif metadata.get('ray_recovery'):
        raise RevitPackError('Recovered-ray source has no auditable processing manifest')
    source = source.resolve()
    if not source.is_file():
        raise RevitPackError(f'Source missing: {source}')
    axis = metadata.get('source_axis')
    if axis not in {'CAD_Z_up', 'CAD_Z_UP', 'ARKit_Y_up', 'ARKIT_Y_UP'}:
        raise RevitPackError(f'Source axis is not explicit: {source}')
    units = metadata.get('source_unit')
    if units not in {'m', 'meter', 'metre'}:
        raise RevitPackError('Source must explicitly be in metres')
    matrix = rigid_transform(scan['transform'])
    if axis in {'ARKit_Y_up', 'ARKIT_Y_UP'}:
        matrix = matrix @ Y_TO_Z
    return {'path': source, 'matrix': matrix, 'metadata': metadata, 'replacement': replacement, 'manifest': manifest, 'role': layer.get('role')}

def _write_e57(path: Path, data: dict, name: str) -> dict:
    import pye57
    with pye57.E57(str(path), mode='w') as writer:
        writer.write_scan_raw(data, name=name)
    error = 0.0
    with pye57.E57(str(path)) as reader:
        if reader.scan_count != 1:
            raise RevitPackError('Unexpected E57 scan count')
        decoded = reader.read_scan_raw(0)
        for key, expected in data.items():
            actual = decoded[key]
            if len(actual) != len(expected):
                raise RevitPackError('E57 roundtrip changed point count')
            for start in range(0, len(expected), 250000):
                a, b = (actual[start:start + 250000], expected[start:start + 250000])
                if key in RGB:
                    if not np.array_equal(a, b):
                        raise RevitPackError('E57 roundtrip changed RGB/order')
                else:
                    error = max(error, float(np.max(np.abs(a - b))))
                    if not np.array_equal(a, b.astype(np.float32).astype(np.float64)):
                        raise RevitPackError('E57 XYZ differs from declared float32 encoding')
        header = reader.get_header(0)
        if not np.allclose(header.rotation_matrix, np.eye(3)) or not np.allclose(header.translation, 0):
            raise RevitPackError('E57 pose would transform coordinates twice')
    return {'file': path.name, 'point_count': len(data[XYZ[0]]), 'bytes': path.stat().st_size, 'sha256': sha256(path), 'all_points_rgb_order_verified': True, 'max_encoding_error_m': error, 'pose': 'identity', 'coordinate_precision': 'float32'}

def export_scan(scan: Mapping, destination: Path, base: Path) -> dict:
    source = resolve_source(scan, base)
    path, matrix = (source['path'], source['matrix'])
    before = sha256(path)
    manifest = source['manifest']
    if manifest and manifest['outputs']['rgb']['sha256'].lower() != before:
        raise RevitPackError('RGB source does not match its processing manifest SHA256')
    schema = read_ply_schema(path)
    count = schema.vertex.count
    if count < 1:
        raise RevitPackError('Empty source')
    destination.mkdir(parents=True, exist_ok=False)
    data = {key: np.empty(count, dtype=np.float64 if key in XYZ else np.uint8) for key in XYZ + RGB}
    fill = np.zeros(count, dtype=bool)
    records = None
    offset = 0
    color_present = False
    try:
        for chunk in iter_ply_vertex_chunks(path):
            if records is None:
                records = np.lib.format.open_memmap(destination / 'source-vertices.npy', mode='w+', dtype=chunk.dtype, shape=(count,))
            end = offset + len(chunk)
            records[offset:end] = chunk
            points = np.column_stack([chunk[key] for key in ('x', 'y', 'z')]).astype(float)
            if not np.isfinite(points).all():
                raise RevitPackError('Non-finite XYZ retained in source; repair source diagnosis before E57 export')
            points = points @ matrix[:3, :3].T + matrix[:3, 3]
            for axis, key in enumerate(XYZ):
                data[key][offset:end] = points[:, axis]
            color_present = all((key in chunk.dtype.names for key in ('red', 'green', 'blue')))
            for old, key in zip(('red', 'green', 'blue'), RGB, strict=True):
                values = chunk[old] if color_present else np.full(len(chunk), 180, dtype=np.uint8)
                if np.any(values < 0) or np.any(values > 255) or (not np.isfinite(values).all()) or np.any(values != np.floor(values)):
                    raise RevitPackError('RGB must be exact integers in 0..255')
                data[key][offset:end] = values
            if 'fill_share' in chunk.dtype.names:
                fill[offset:end] = chunk['fill_share'] >= 0.5
            offset = end
        if offset != count:
            raise RevitPackError('Source count changed while reading')
        records.flush()
    finally:
        if records is not None:
            records._mmap.close()
    saved = np.load(destination / 'source-vertices.npy', mmap_mode='r', allow_pickle=False)
    offset = 0
    try:
        for chunk in iter_ply_vertex_chunks(path):
            if chunk.tobytes() != saved[offset:offset + len(chunk)].tobytes():
                raise RevitPackError('Source attribute sidecar failed exact roundtrip')
            offset += len(chunk)
    finally:
        saved._mmap.close()
    outputs = []
    for label, mask in (('observed', ~fill), ('sensor-fill', fill)):
        if np.any(mask):
            subset = {key: values[mask] for key, values in data.items()}
            result = _write_e57(destination / f'{label}.e57', subset, f"{scan['scan_id']} / {label}")
            outputs.append({'kind': label, **result})
            del subset
    if sum((item['point_count'] for item in outputs)) != count or sha256(path) != before:
        raise RevitPackError('Point conservation or source hash failed')
    return {'scan_id': scan['scan_id'], 'status': scan.get('status'), 'source': str(path), 'source_sha256': before, 'sources_unchanged': True, 'source_point_count': count, 'observed_count': int((~fill).sum()), 'sensor_fill_count': int(fill.sum()), 'source_vertices': f'{destination.name}/source-vertices.npy', 'source_vertices_sha256': sha256(destination / 'source-vertices.npy'), 'source_fields': [name for _, name in schema.vertex_signature], 'source_attributes_exact_roundtrip': True, 'source_axis': source['metadata']['source_axis'], 'source_metadata': source['metadata'], 'T_project_from_source': matrix.tolist(), 'T_project_from_scan_zup': scan['transform'], 'replacement': source['replacement'], 'source_role': source['role'], 'registration': 'STORED_PROJECT_PLACEMENT_NOT_REVIT_REGISTRATION', 'bounds_project_m': {'min': [float(data[key].min()) for key in XYZ], 'max': [float(data[key].max()) for key in XYZ]}, 'rgb': 'source' if color_present else 'neutral_missing_rgb', 'outputs': outputs}

def export_revit_pack(project: Mapping, destination: str | Path, *, base: str | Path, scan_ids: Sequence[str] | None=None, progress: Callable[[int, int, str], None]=lambda *args: None) -> dict:
    project = copy.deepcopy(project)
    scans = project.get('scan_rail', [])
    if scan_ids is not None:
        requested = set(scan_ids)
        if not requested or requested - {scan['scan_id'] for scan in scans}:
            raise RevitPackError('Select existing scan IDs')
        scans = [scan for scan in scans if scan['scan_id'] in requested]
    if not scans:
        raise RevitPackError('Project contains no scans')
    ids = [scan['scan_id'] for scan in scans]
    if len(set(ids)) != len(ids) or any((not re.fullmatch('[A-Za-z0-9_-]+', name) for name in ids)):
        raise RevitPackError('Scan IDs must be unique safe file names')
    base, destination = (Path(base).resolve(), Path(destination).resolve())
    for scan in scans:
        resolve_source(scan, base)
        raw_root = scan.get('source_path')
        if raw_root:
            raw_root = Path(raw_root)
            raw_root = (raw_root if raw_root.is_absolute() else base / raw_root).resolve()
            if raw_root.is_dir() and destination.is_relative_to(raw_root):
                raise RevitPackError('Export destination must be outside raw scan folders')
    destination.mkdir(parents=True, exist_ok=False)
    _json(destination / 'project-snapshot.ccapoints.json', project)
    state = {'schema': SCHEMA, 'state': 'BUILDING', 'units': 'meter', 'axis': 'CAD_Z_up', 'merged': False, 'revit_registered': False, 'measurement_calibrated': False, 'excluded_scan_ids': [s['scan_id'] for s in project['scan_rail'] if s['scan_id'] not in ids], 'scans': [], 'rcp_rcs_created': False}
    _json(destination / 'manifest.json', state)
    try:
        for index, scan in enumerate(scans, 1):
            progress(index - 1, len(scans), f"REVIT PACK / {scan['scan_id']}")
            record = export_scan(scan, destination / scan['scan_id'], base)
            state['scans'].append(record)
            _json(destination / 'manifest.json', state)
            progress(index, len(scans), f"VERIFIED / {record['source_point_count']:,} points / {record['sensor_fill_count']:,} sensor fill")
        with (destination / 'scan-index.csv').open('w', encoding='ascii', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['scan_id', 'points', 'observed', 'sensor_fill', 'project_status', 'revit_registered'])
            for item in state['scans']:
                writer.writerow([item['scan_id'], item['source_point_count'], item['observed_count'], item['sensor_fill_count'], item['status'], False])
        state['point_count'] = sum((item['source_point_count'] for item in state['scans']))
        state['state'] = 'E57_VERIFIED_RCP_PENDING'
        shutil.copyfile(Path(__file__).with_name('index_recap.ps1'), destination / 'INDEX-RECAP.ps1')
        shutil.copyfile(Path(__file__).with_name('index_recap.cmd'), destination / 'INDEX-RECAP.cmd')
        (destination / 'BASLA.txt').write_text("CCA LEFEBVRE / REVIT KARSILASTIRMA PAKETI\n\n1. Her tarama ayri klasor: observed.e57 ana veri, sensor-fill.e57 sensor dolgusu.\n2. ReCap'te observed.e57 dosyasini metre birimiyle indexleyip RCP/RCS kaydet.\n   Etkin mevcut ReCap lisansi varsa INDEX-RECAP.cmd ile yerel indexleme yapilabilir.\n3. Revit > Insert > Point Cloud ile RCP/RCS bagla. E57 dogrudan Revit formati degildir.\n4. Taramalarin TAMAMI hizali degil. Ayri tut; ortak sabit kolon/duvarlarla yerlesim yap.\n   3 bagimsiz kontrol noktasi ve ayri kontrol bolgesindeki sapmayi kaydet.\n5. Revit koordinati henuz belirlenmedi. Origin-to-origin yalniz ayni referans dogrulaninca.\n6. Turuncu/eksik goruntu tek basina bina degisikligi degildir. Gorulmeme/ortulme de olabilir.\n7. Bu paket kalibre olcum veya otomatik degisiklik raporu DEGILDIR.\n\nHer kaynak nokta korunur: observed + sensor-fill = kaynak sayisi.\nsource-vertices.npy tum orijinal alanlari ve sirayi tutar. XYZ kaynak koordinatindadir;\nE57 XYZ ise manifestteki T_project_from_source ile donusturulmustur. E57 pose=identity.\nManifest encoding error olcum hatasi degil, yalniz dosya kuantalizasyonudur.\nYerlesim tarama-bazli; ekranin envanter dizilim ofsetleri export edilmez.\nHam kaynaklar / mevcut proje / Revit modeli degistirilmedi.\n\nAutodesk: https://help.autodesk.com/cloudhelp/ENU/Reality-Capture/files/Creating_Scan_Projects/decap.html\nRevit: https://help.autodesk.com/cloudhelp/2021/ENU/Revit-Model/files/GUID-B89AD692-C705-458F-A638-EE7DD83D694C.htm\n", encoding='ascii')
    except Exception as error:
        state['state'] = 'FAILED_INCOMPLETE'
        state['error'] = str(error)
        _json(destination / 'manifest.json', state)
        raise
    _json(destination / 'manifest.json', state)
    return {'output_dir': str(destination), 'manifest_path': str(destination / 'manifest.json'), **state}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--scan', action='append', dest='scan_ids')
    args = parser.parse_args()
    project = json.loads(args.project.read_text(encoding='utf-8'))
    result = export_revit_pack(project, args.output, base=args.project.parent, scan_ids=args.scan_ids, progress=lambda done, total, message: print(f'{done}/{total} {message}', flush=True))
    print(f"E57 VERIFIED / {result['point_count']:,} points / {result['manifest_path']}", flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
