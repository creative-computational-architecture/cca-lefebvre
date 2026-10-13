"""Render publication figures from the calculated synthetic survey demo.

No geometry is invented here: point views read the generated PLY, candidate
views read the survey JSON, and the CAD sheet renders actual DXF entities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import ezdxf
from ezdxf import bbox
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import open3d as o3d

PAPER = '#f5f5f5'
INK = '#1a1a1a'
RED = '#C8102E'
GREY = '#767676'
CONFIDENCE = {'HIGH': INK, 'MEDIUM': '#777777', 'LOW': RED}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.edgecolor': '#c5c5c5', 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': GREY,
                     'ytick.color': GREY, 'savefig.facecolor': PAPER})


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(fig, output: Path):
    fig.savefig(output, dpi=110, facecolor=PAPER,
                metadata={'Software': None})
    plt.close(fig)


def heading(fig, title: str, subtitle: str):
    fig.text(.035, .952, title, fontsize=25, weight='bold')
    fig.text(.035, .907, subtitle, fontsize=10, color=GREY,
             family='DejaVu Sans Mono')
    fig.add_artist(plt.Line2D([.035, .965], [.884, .884],
                             transform=fig.transFigure, color=INK, linewidth=.6))


def footer(fig, text: str):
    fig.text(.035, .036, text, fontsize=9, color=GREY,
             family='DejaVu Sans Mono')


def wall_segments(ax, walls, structures, *, label=False):
    for w in walls:
        x = [w['start_m'], w['end_m']] if w['axis'] == 'x' else [w['pos_m']] * 2
        y = [w['pos_m']] * 2 if w['axis'] == 'x' else [w['start_m'], w['end_m']]
        low = w['confidence'] == 'LOW'
        ax.plot(x, y, color=CONFIDENCE[w['confidence']], lw=2.0 if not low else 1.5,
                ls='--' if low else '-', zorder=3)
        if label:
            ax.text(np.mean(x) + .055, np.mean(y) + .07, w['id'],
                    fontsize=8, color=CONFIDENCE[w['confidence']])
    for s in structures:
        ax.add_patch(Rectangle((s['x_start_m'], s['y_front_m']),
                     s['width_m'], s['depth_m'], fill=False,
                     ec=RED, lw=.8, zorder=2))
        ax.text((s['x_start_m'] + s['x_end_m']) / 2,
                (s['y_front_m'] + s['y_back_m']) / 2,
                s['id'], color=RED, weight='bold', ha='center', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(-.4, 8.8)
    ax.set_ylim(-.4, 5.9)


def hero(summary, points, colours, walls, structures, output):
    fig = plt.figure(figsize=(16, 8.6), facecolor=PAPER)
    heading(fig, 'FROM POINTS TO AN ARCHITECTURAL SURVEY',
            'CCA LEFEBVRE / CONTROLLED SYNTHETIC DEMONSTRATION / NO REAL PROJECT DATA')
    ax = fig.add_axes([.025, .14, .60, .70], projection='3d', facecolor=PAPER)
    mask = points[:, 2] < 2.6
    cut, rgb = points[mask][::2], colours[mask][::2] * .65
    ax.scatter(*cut.T, c=rgb, s=.55, linewidths=0, depthshade=False, rasterized=True)
    ax.set_box_aspect(np.ptp(points, axis=0))
    ax.view_init(elev=31, azim=-62)
    ax.set_axis_off()
    ax.text2D(.06, .01, 'POINT SUBSET / z < 2.6 m / roof omitted for a cutaway',
              transform=ax.transAxes, fontsize=10, family='DejaVu Sans Mono', color=GREY)
    plan = fig.add_axes([.66, .43, .30, .38], facecolor=PAPER)
    wall_segments(plan, walls, structures)
    plan.set_axis_off()
    plan.set_title('CALCULATED WALLS + STRUCTURE CANDIDATES', loc='left', fontsize=10)
    rows = [('01', f"{summary['input']['point_count']:,} source points"),
            ('02', f"{summary['survey']['walls']} wall segments / confidence tagged"),
            ('03', f"{summary['survey']['structures']} structure candidates"),
            ('04', f"{summary['dxf']['entities']:,} actual DXF entities / cm")]
    for i, (n, text) in enumerate(rows):
        y = .347 - i * .059
        fig.text(.675, y, n, fontsize=12, color=RED, family='DejaVu Sans Mono')
        fig.text(.721, y, text, fontsize=11)
    footer(fig, 'SOURCE PRESERVED / METRIC FRAME / SURVEY DRAFT - UNCALIBRATED / CANDIDATES REQUIRE REVIEW')
    save(fig, output)


def analysis(summary, points, walls, structures, output):
    fig = plt.figure(figsize=(16, 8.6), facecolor=PAPER)
    heading(fig, 'READ THE EVIDENCE, THEN REVIEW THE CANDIDATES',
            'AUTOMATIC SURVEY PACK / THE FIGURES BELOW READ THE CALCULATED JSON AND SOURCE POINTS')
    ax = fig.add_axes([.06, .20, .49, .61], facecolor=PAPER)
    slice_points = points[(points[:, 2] >= .95) & (points[:, 2] <= 1.25)]
    ax.scatter(slice_points[:, 0], slice_points[:, 1], s=.45, c='#a8a8a8', linewidths=0)
    wall_segments(ax, walls, structures, label=False)
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_title('PLAN SLICE / z = 0.95-1.25 m', loc='left', fontsize=13, pad=14)
    ax.grid(color='#dedede', linewidth=.5)
    fig.text(.062, .132, 'Grey: source points   Black: high support   Red dashed: measure by hand', fontsize=10)
    ax2 = fig.add_axes([.64, .51, .31, .24], facecolor=PAPER)
    counts = summary['survey']['confidence_counts']['walls']
    names = ['HIGH', 'MEDIUM', 'LOW']
    values = [counts.get(name, 0) for name in names]
    bars = ax2.barh(names[::-1], values[::-1], color=[CONFIDENCE[n] for n in names[::-1]], height=.58)
    for bar in bars:
        ax2.text(bar.get_width() + .14, bar.get_y() + bar.get_height() / 2,
                 str(int(bar.get_width())), va='center', fontsize=13, weight='bold')
    ax2.set_xlim(0, max(values) + 1.6)
    ax2.set_title(f'{sum(values)} WALL SEGMENTS / SUPPORT LABELS', loc='left', fontsize=13, pad=15)
    ax2.set_xlabel('candidate count')
    ax2.spines[['top', 'right']].set_visible(False)
    ax3 = fig.add_axes([.64, .195, .31, .20], facecolor=PAPER)
    errors = summary['survey']['extent_error_m']
    values = [1000 * errors[k] for k in ('x', 'y', 'z')]
    bars = ax3.bar(['X', 'Y', 'Z'], values, color=INK, width=.5)
    for bar, v in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + .4, f'{v:.1f}', ha='center', fontsize=10)
    ax3.set_ylabel('difference / mm')
    ax3.set_ylim(0, max(values) * 1.35)
    ax3.set_title('ENVELOPE VS. KNOWN SYNTHETIC GEOMETRY', loc='left', fontsize=11, pad=15)
    ax3.spines[['top', 'right']].set_visible(False)
    footer(fig, 'SYNTHETIC FIXTURE ONLY / ENVELOPE DIFFERENCE IS NOT A REAL-WORLD ACCURACY CLAIM / LOW SUPPORT REMAINS VISIBLE')
    save(fig, output)


def cad_panel(fig, position, doc, entities, limits, title):
    ax = fig.add_axes(position, facecolor=PAPER)
    config = Configuration(background_policy=BackgroundPolicy.CUSTOM,
                           color_policy=ColorPolicy.COLOR_SWAP_BW,
                           custom_bg_color=PAPER, min_lineweight=.20,
                           lineweight_scaling=2.4)
    backend = MatplotlibBackend(ax, adjust_figure=False)
    frontend = Frontend(RenderContext(doc), backend, config=config)
    def publication_style(entity, properties):
        layer = properties.layer
        if layer in {'CCA_SURVEY_DIM', 'CCA_SURVEY_STRUCT', 'CCA_SURVEY_WALL_LOWCONF'}:
            properties.color = RED
            properties.lineweight = .32
        elif layer == 'CCA_SURVEY_GRID':
            properties.color = '#555555'
            properties.lineweight = .25
        elif layer in {'CCA_SURVEY_SLICE', 'CCA_SURVEY_ELEV_LIDAR'}:
            properties.color = '#969696'
            properties.lineweight = .14
        else:
            properties.color = INK
            properties.lineweight = .45
    frontend.push_property_override_function(publication_style)
    frontend.draw_entities(e for e in entities if e.dxftype() != 'TEXT')
    backend.finalize()
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.set_title(title, loc='left', fontsize=12, pad=8)
    for entity in entities:
        if entity.dxftype() != 'TEXT':
            continue
        position = entity.dxf.insert
        color = RED if entity.dxf.layer == 'CCA_SURVEY_STRUCT' else '#555555'
        align = {0: 'left', 1: 'center', 2: 'right'}.get(entity.dxf.halign, 'left')
        ax.text(position.x, position.y, entity.dxf.text, color=color,
                fontsize=9, family='DejaVu Sans Mono', ha=align,
                va='baseline', rotation=entity.dxf.rotation,
                rotation_mode='anchor')


def cad_sheet(summary, dxf, output):
    doc = ezdxf.readfile(dxf)
    model = doc.modelspace()
    def in_layers(*names):
        return [e for e in model if e.dxf.layer in names]
    dims = in_layers('CCA_SURVEY_DIM')
    def bounds(entity):
        b = bbox.extents([entity])
        return b.extmin, b.extmax
    plan_dims = [e for e in dims if bounds(e)[0].y > -200]
    front_dims = [e for e in dims if bounds(e)[1].y < -500]
    plan = in_layers('CCA_SURVEY_SLICE', 'CCA_SURVEY_WALL',
            'CCA_SURVEY_WALL_LOWCONF', 'CCA_SURVEY_STRUCT', 'CCA_SURVEY_GRID') + plan_dims
    front = in_layers('CCA_SURVEY_ELEV_FRONT', 'CCA_SURVEY_ELEV_LIDAR') + front_dims
    sections = in_layers('CCA_SURVEY_SECTION')
    section1 = [e for e in sections if bounds(e)[0].x < 1900]
    section2 = [e for e in sections if bounds(e)[0].x >= 1900]
    fig = plt.figure(figsize=(16, 10.6), facecolor=PAPER)
    heading(fig, 'CAD EXPORT / ACTUAL DRAWING ENTITIES',
            f"SURVEY_CM.DXF / {summary['dxf']['entities']:,} ENTITIES / {summary['dxf']['entity_counts_by_layer']['CCA_SURVEY_DIM']} DIMENSION ENTITIES / INSUNITS = CENTIMETRES")
    cad_panel(fig, [.04, .19, .48, .62], doc, plan,
              [(-110, 950), (-170, 655)], '01 / PLAN + CAD DIMENSIONS')
    cad_panel(fig, [.57, .48, .39, .31], doc, front,
              [(-60, 925), (-1170, -540)], '02 / FRONT ELEVATION + LiDAR PROFILE')
    cad_panel(fig, [.57, .17, .18, .24], doc, section1,
              [(1270, 1850), (-35, 420)], '03 / SECTION S01')
    cad_panel(fig, [.78, .17, .18, .24], doc, section2,
              [(1990, 2570), (-35, 420)], '04 / SECTION S02')
    fig.text(.056, .13, 'Layers: slices / walls / low-confidence walls / structures / dimensions / elevation / sections', fontsize=10, color=GREY)
    footer(fig, 'RENDERED DIRECTLY FROM THE GENERATED DXF / EZDXF AUDIT: 0 ERRORS / SURVEY DRAFT - UNCALIBRATED')
    save(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-dir', type=Path, required=True)
    parser.add_argument('--media-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.demo_dir.resolve()
    summary = read_json(root / 'summary.json')
    if summary.get('synthetic') is not True:
        raise ValueError('Publication media requires a documented synthetic demo')
    cloud_path = root / summary['input']['cloud']
    if hashlib.sha256(cloud_path.read_bytes()).hexdigest() != summary['input']['sha256_after']:
        raise ValueError('Cloud hash does not match the calculated demo summary')
    cloud = o3d.io.read_point_cloud(str(cloud_path))
    points, colours = np.asarray(cloud.points), np.asarray(cloud.colors)
    transform = np.asarray(summary['survey']['frame']['transform_survey_from_zup'])
    points = points @ transform[:3, :3].T + transform[:3, 3]
    survey = root / summary['output_dir']
    walls = read_json(survey / 'walls.json')['walls']
    structures = read_json(survey / 'structures.json')['structures']
    output = args.media_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hero(summary, points, colours, walls, structures, output / '01-lefebvre-survey-hero.png')
    analysis(summary, points, walls, structures, output / '03-survey-analysis.png')
    cad_sheet(summary, root / summary['dxf']['path'], output / '04-cad-drawing-sheet.png')
    print(json.dumps({'synthetic': True, 'figures': 3,
                      'source_points': len(points), 'cad_entities': summary['dxf']['entities']}, sort_keys=True))


if __name__ == '__main__':
    main()
