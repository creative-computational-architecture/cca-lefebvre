
import time
from pathlib import Path
import numpy as np
from geometry.ply_preserve import iter_ply_vertex_chunks, read_ply_schema
from geometry.surface_structure import surface_support
STATUS = {0: 'UNRESOLVED_RETAINED', 1: 'LOCAL_SURFACE_SUPPORTED', 2: 'SENSOR_FILL_QUARANTINED', 3: 'NONFINITE_RETAINED'}

def audit_point_usage(ply_path, output, *, progress=None):
    from viewer.raw_processing_service import _sha256_file, _write_ascii_json
    source, target = (Path(ply_path).resolve(), Path(output).resolve())
    if target.parent == source.parent or source.parent in target.parents:
        raise ValueError('point audit must be outside the source folder')
    start = time.monotonic()
    source_hash = _sha256_file(source)
    schema = read_ply_schema(source)
    n = schema.vertex.count
    xyz = np.empty((n, 3), np.float64)
    fill = np.zeros(n, np.float32)
    fields = {p.name for p in schema.vertex.properties}
    mixed = offset = observations = 0
    for chunk in iter_ply_vertex_chunks(source):
        stop = offset + len(chunk)
        xyz[offset:stop] = np.column_stack([chunk[k] for k in ('x', 'y', 'z')])
        if 'fill_share' in fields:
            f = np.asarray(chunk['fill_share'], np.float32)
            if not np.isfinite(f).all() or np.any((f < 0) | (f > 1)):
                raise ValueError('invalid raw fill_share; cannot silently quarantine points')
            fill[offset:stop] = f
            mixed += int(((f > 0) & (f < 1)).sum())
        if 'n_views' in fields:
            observations += int(np.sum(chunk['n_views'], dtype=np.float64))
        offset = stop
    if offset != n:
        raise ValueError('PLY vertex count does not match stream')
    finite = np.isfinite(xyz).all(axis=1)
    pure_fill = fill == 1
    used = finite & ~pure_fill
    support = np.zeros(n, np.float32)
    if progress:
        progress(f'{n:,} noktanin tamami sayildi / {int(used.sum()):,} geometriye katiliyor')
    support[used] = surface_support(xyz[used])
    state = np.zeros(n, np.uint8)
    state[(support >= 0.45) & used] = 1
    state[pure_fill] = 2
    state[~finite & ~pure_fill] = 3
    counts = {label: int((state == code).sum()) for code, label in STATUS.items()}
    if sum(counts.values()) != n or _sha256_file(source) != source_hash:
        raise ValueError('point count or immutable source check failed')
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = target.with_suffix('.npz')
    temp = sidecar.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, surface_support=support, status=state)
    temp.replace(sidecar)
    report = {'schema': 'cca.point-usage.v1', 'source': str(source), 'source_sha256': source_hash, 'scope': 'ALL_DERIVED_PLY_VERTICES_NOT_ALL_RAW_PIXELS', 'unit': 'm', 'vertices': n, 'accounted_vertices': sum(counts.values()), 'omitted_vertices': 0, 'geometric_query_and_reference_vertices': int(used.sum()), 'counts': counts, 'mixed_fill_vertices_retained': mixed, 'raw_observation_weight_sum': observations if 'n_views' in fields else None, 'fill_label_available': 'fill_share' in fields, 'fill_rule': 'explicit RAW conf2 AND depth>=16.384m label, never range_m alone', 'support_rule': '24 neighbours within local 0.30m, PCA support>=0.45; not certified position', 'source_preserved': True, 'moved_vertices': 0, 'deleted_vertices': 0, 'sidecar': str(sidecar), 'sidecar_sha256': _sha256_file(sidecar), 'index_contract': 'status[i] and surface_support[i] refer to input PLY vertex i', 'status_codes': STATUS, 'seconds': round(time.monotonic() - start, 3)}
    _write_ascii_json(target, report)
    return report

def make_point_usage_dialog(parent, report):
    from PySide6 import QtWidgets
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName('PointUsageDialog')
    dialog.setWindowTitle('CCA Lefebvre / Her noktanin hesabi')
    dialog.resize(670, 400)
    layout = QtWidgets.QVBoxLayout(dialog)
    text = QtWidgets.QTextBrowser()
    text.setObjectName('PointUsageReport')
    counts = report['counts']
    text.setPlainText(f"TUM TURETILMIS NOKTALAR: {report['vertices']:,}\nHesabi verilen: {report['accounted_vertices']:,} / Atlanan: {report['omitted_vertices']}\nGeometri aramasina katilan: {report['geometric_query_and_reference_vertices']:,}\n\nYerel yuzey destegi: {counts['LOCAL_SURFACE_SUPPORTED']:,}\nYeri cozulmemis, korunan: {counts['UNRESOLVED_RETAINED']:,}\nSensor dolgusu (2^14 mm + ham conf2): {counts['SENSOR_FILL_QUARANTINED']:,}\nGecersiz koordinat, korunan: {counts['NONFINITE_RETAINED']:,}\n\nYuzey destegi kesin konum degildir. Bu rapor hicbir noktayi tasimaz veya silmez.\nHer noktanin durumu ve destek degeri kaynak sirasiyla yan dosyada saklanir.\n" + ('' if report['fill_label_available'] else 'UYARI: Kaynakta fill etiketi yok; menzile gore nokta elenmedi.\n') + report['sidecar'])
    layout.addWidget(text)
    close = QtWidgets.QPushButton('Kapat')
    close.setObjectName('PointUsageClose')
    close.clicked.connect(dialog.accept)
    layout.addWidget(close)
    return dialog
