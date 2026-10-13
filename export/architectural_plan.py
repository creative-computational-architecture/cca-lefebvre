
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import ezdxf
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
KINDS = {'DUVAR': (7, '#1a1a1a'), 'KOLON': (1, '#C8102E'), 'PENCERE': (5, '#246b95'), 'KAPI': (30, '#9b5d00'), 'MERDIVEN': (3, '#397151'), 'BELIRSIZ': (8, '#737373')}
STAMP = 'ROLOVE TASLAK / KALIBRE EDILMEMIS / ADAYLARI KONTROL EDIN'

def draw_plan(analysis, destination, *, x_range=None):
    underlay = np.asarray(analysis['underlay_m'])
    span = np.ptp(underlay, axis=0) if len(underlay) else np.array([10, 5])
    width = x_range[1] - x_range[0] if x_range else max(span[0], 1)
    fig = Figure(figsize=(16, min(12, max(5, 14 * span[1] / width))))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    if len(underlay):
        ax.scatter(*underlay.T, s=0.3, c='#c5c5c5', linewidths=0, rasterized=True)
    labelled = set()
    for element in analysis['elements']:
        if element['review_status'] == 'DISLA':
            continue
        kind = element['kind']
        low = element['classification_confidence'] == 'LOW' or element['support'] == 'LOW'
        for line in element['lines_m']:
            xy = np.asarray(line)
            ax.plot(*xy.T, color=KINDS[kind][1], linewidth=1.3 if kind == 'DUVAR' else 1.8, linestyle='--' if low else '-', label=kind if kind not in labelled else None)
            labelled.add(kind)
        if element['lines_m'] and (len(analysis['elements']) < 25 or kind != 'DUVAR' or x_range):
            center = np.asarray(element['lines_m'][0]).mean(axis=0)
            if x_range and (not x_range[0] <= center[0] <= x_range[1]):
                continue
            ax.annotate(element['id'], center, xytext=(2, 4), textcoords='offset points', fontsize=6, color=KINDS[kind][1], clip_on=True)
    ax.set_aspect('equal')
    if x_range:
        ax.set_xlim(x_range)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_xlabel('Plan u [m]')
    ax.set_ylabel('Plan v [m]')
    title = 'Kat plani' if x_range is None else f'Kat plani detayi / u {x_range[0]:.1f}..{x_range[1]:.1f} m'
    ax.set_title(f'nokta by cca / {title}\n{STAMP}', loc='left', fontsize=11)
    if labelled:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    config = analysis['config']
    fig.text(0.08, 0.025, f"Kesit: {config['cut_height_m']:g} m / Kat: {config['storey_height_m']:g} m / Kesik: ELLE OLC / Gri: olculen noktalar / Gosterilmeyen eleman yok demek degildir", fontsize=8)
    fig.subplots_adjust(bottom=0.13, right=0.86)
    fig.savefig(destination, dpi=140, facecolor='white')
    fig.clear()

def export_architectural_plan(analysis, output_dir, *, source, source_hash, transform):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    doc = ezdxf.new('R2013')
    doc.units = 5
    doc.appids.new('NOKTA')
    if 'DASHED' not in doc.linetypes:
        doc.linetypes.new('DASHED', dxfattribs={'description': 'Review candidate', 'pattern': [0.3, 0.2, -0.1]})
    for kind, (color, _rgb) in KINDS.items():
        doc.layers.new(f'NOKTA_{kind}', dxfattribs={'color': color})
        doc.layers.new(f'NOKTA_{kind}_ELLE_OLC', dxfattribs={'color': color, 'linetype': 'DASHED'})
    doc.layers.new('NOKTA_NOKTA_ALTLIGI', dxfattribs={'color': 8})
    doc.layers.new('NOKTA_NOTLAR', dxfattribs={'color': 1})
    model = doc.modelspace()
    for x, y in analysis['underlay_m']:
        model.add_point((x * 100, y * 100), dxfattribs={'layer': 'NOKTA_NOKTA_ALTLIGI'})
    expected = []
    for element in analysis['elements']:
        if element['review_status'] == 'DISLA':
            continue
        low = element['classification_confidence'] == 'LOW' or element['support'] == 'LOW'
        layer = f"NOKTA_{element['kind']}" + ('_ELLE_OLC' if low else '')
        for line in element['lines_m']:
            vertices = np.asarray(line) * 100
            entity = model.add_lwpolyline(vertices.tolist(), dxfattribs={'layer': layer})
            entity.set_xdata('NOKTA', [(1000, element['id']), (1000, element['review_status'])])
            expected.append((layer, vertices))
        if element['lines_m']:
            center = np.asarray(element['lines_m'][0]).mean(axis=0) * 100
            model.add_text(f"{element['id']} {element['kind']}" + (' / ELLE OLC' if low else ' / ADAY'), dxfattribs={'height': 9, 'insert': tuple(center + [0, 12]), 'layer': 'NOKTA_NOTLAR'})
    xy = np.asarray(analysis['underlay_m'])
    top = xy.max(axis=0) * 100 if len(xy) else np.zeros(2)
    model.add_text(STAMP, dxfattribs={'height': 15, 'insert': (0, float(top[1] + 80)), 'layer': 'NOKTA_NOTLAR'})
    target = output / 'kat-plani.dxf'
    doc.saveas(target)
    reread = ezdxf.readfile(target)
    errors = len(reread.audit().errors)
    actual = list(reread.modelspace().query('LWPOLYLINE'))
    if errors or len(actual) != len(expected):
        raise ValueError('Kat plani DXF yeniden okuma dogrulamasi basarisiz')
    for entity, (layer, vertices) in zip(actual, expected, strict=True):
        if entity.dxf.layer != layer or not np.allclose(np.asarray(entity.get_points('xy')), vertices, atol=1e-07):
            raise ValueError('DXF katmani veya cm koordinatlari degisti')
    draw_plan(analysis, output / 'kat-plani.png')
    detail_files = []
    if len(xy) and np.ptp(xy[:, 0]) > 20:
        for index, start in enumerate(np.arange(xy[:, 0].min(), xy[:, 0].max(), 12), 1):
            name = f'kat-plani-detay-{index:02d}.png'
            draw_plan(analysis, output / name, x_range=(start - 0.25, min(start + 12.25, xy[:, 0].max() + 0.25)))
            detail_files.append(name)
    (output / 'elemanlar.json').write_text(json.dumps(analysis, indent=2, ensure_ascii=True), encoding='ascii')
    with Path(source).open('rb') as stream:
        after = hashlib.file_digest(stream, 'sha256').hexdigest()
    if after != source_hash:
        raise ValueError('Kaynak analiz sirasinda degisti; sonuc onaylanamadi')
    manifest = {'schema': 'nokta.architectural-plan-pack.v1', 'stamp': STAMP, 'source': str(source), 'source_sha256': source_hash, 'source_sha256_after': after, 'transform_world_from_source': np.asarray(transform).tolist(), 'surface': analysis['surface'], 'config': analysis['config'], 'source_count': analysis['source_count'], 'floor_point_count': analysis['floor_point_count'], 'dxf_unit': 'cm', 'dxf_audit_errors': errors, 'verified_polylines': len(actual), 'element_counts': {k: sum((e['kind'] == k and e['review_status'] != 'DISLA' for e in analysis['elements'])) for k in KINDS}, 'files': ['kat-plani.dxf', 'kat-plani.png', 'elemanlar.json', *detail_files]}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding='ascii')
    return output
