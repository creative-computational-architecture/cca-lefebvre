
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import ezdxf
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from geometry.rolove import project_surface

def export_projection_pack(points, rgb, surfaces, output_dir, *, source, source_hash, transform, alignment_evidence=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'nokta.rolove-projections.v1', 'stamp': 'ROLOVE TASLAK / KALIBRE EDILMEMIS', 'source': str(source), 'source_sha256': source_hash, 'transform_world_from_source': np.asarray(transform).tolist(), 'alignment_evidence': alignment_evidence or {}, 'internal_unit': 'm', 'dxf_unit': 'cm', 'surfaces': []}
    for number, surface in enumerate(surfaces, 1):
        result = project_surface(points, rgb, surface)
        stem = f'{number:02d}-' + re.sub('[^A-Za-z0-9_-]+', '-', surface.name).strip('-')
        uv = result['uv']
        minimum, maximum = (uv.min(axis=0), uv.max(axis=0))
        width, height = maximum - minimum
        fig = Figure(figsize=(16, min(18, max(5, 14 * height / max(width, 1)))))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        if result['rgb'] is not None:
            colors = np.asarray(result['rgb'], dtype=float) / 255.0
        else:
            colors = '#1a1a1a'
        ax.scatter(uv[:, 0], uv[:, 1], c=colors, s=0.6, marker='s', linewidths=0, rasterized=True)
        ax.set_aspect('equal')
        ax.set_xlabel('u [m]')
        ax.set_ylabel('v [m]')
        ax.grid(True, linewidth=0.3, alpha=0.4)
        ax.set_title(f'nokta by cca  /  {surface.name}\nROLOVE TASLAK / KALIBRE EDILMEMIS', loc='left')
        fig.text(0.08, 0.015, f"Bulut zarfi: {width * 100:.0f} x {height * 100:.0f} cm  |  Nokta: {result['selected_count']:,} / {result['source_count']:,}  |  Piksel: {surface.resolution_m * 100:.1f} cm  |  sha256 {source_hash[:12]}", fontsize=8)
        fig.savefig(output / f'{stem}.png', dpi=150, facecolor='white')
        fig.clear()
        document = ezdxf.new('R2013')
        document.units = 5
        document.layers.new('NOKTA_PROJECTION', dxfattribs={'color': 7})
        document.layers.new('NOKTA_NOTES', dxfattribs={'color': 1})
        model = document.modelspace()
        for u, v in uv:
            model.add_point((float(u * 100), float(v * 100)), dxfattribs={'layer': 'NOKTA_PROJECTION'})
        model.add_text(f'nokta by cca / {surface.name} / ROLOVE TASLAK / cm', dxfattribs={'height': 15, 'insert': (float(minimum[0] * 100), float(maximum[1] * 100 + 40)), 'layer': 'NOKTA_NOTES'})
        dxf = output / f'{stem}.dxf'
        document.saveas(dxf)
        errors = len(ezdxf.readfile(dxf).audit().errors)
        if errors:
            raise ValueError(f'DXF audit: {errors} hata')
        np.savez_compressed(output / f'{stem}.npz', **{k: v for k, v in result.items() if v is not None})
        manifest['surfaces'].append({'surface': surface.to_dict(), 'png': f'{stem}.png', 'dxf': dxf.name, 'evidence': f'{stem}.npz', 'dxf_audit_errors': errors, 'source_count': result['source_count'], 'selected_count': result['selected_count'], 'occupied_cells': len(uv), 'bounds_m': [minimum.tolist(), maximum.tolist()]})
    with Path(source).open('rb') as stream:
        after = hashlib.file_digest(stream, 'sha256').hexdigest()
    if after != source_hash:
        raise ValueError('Kaynak tarama islem sirasinda degisti')
    manifest['source_sha256_after'] = after
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding='ascii')
    return output
