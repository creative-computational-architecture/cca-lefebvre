
from __future__ import annotations
import itertools
import math
import os
import tempfile
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import font_manager, rc_context
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from geometry.survey_plan import CONFIDENCE_LOW, CONFIDENCE_MEDIUM, SurveyPlanConfig, SurveyPlanError, find_facade, front_view_min_y
SHEET_STAMP = 'SURVEY DRAFT / UNCALIBRATED'
HAND_MEASURE_LABEL = 'ELLE OLC'
_UNIT_SCALE: dict[str, float] = {'cm': 100.0, 'mm': 1000.0}
_MUTED = '#555555'

class SurveySheetError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class SheetStyle:
    label_unit: Literal['cm', 'mm'] = 'cm'
    dpi: int = 150
    paper: str = '#FFFFFF'
    ink: str = '#1A1A1A'
    accent: str = '#C8102E'
    grid: str = '#B8B8B4'
    low_confidence: str = '#E08430'
    profile: str = '#3F8F63'
    grid_major_m: float = 1.0
    grid_minor_m: float = 0.5
    raster_m: float = 0.02
    inches_per_m: float = 0.64
    min_width_in: float = 10.0
    max_width_in: float = 30.0

    def __post_init__(self) -> None:
        if self.label_unit not in _UNIT_SCALE:
            raise SurveySheetError("label_unit must be 'cm' or 'mm'")
        if not isinstance(self.dpi, int) or isinstance(self.dpi, bool) or (not 50 <= self.dpi <= 600):
            raise SurveySheetError('dpi must be an integer between 50 and 600')
        positive = (self.grid_major_m, self.grid_minor_m, self.raster_m, self.inches_per_m, self.min_width_in, self.max_width_in)
        if not all((math.isfinite(float(v)) and float(v) > 0.0 for v in positive)):
            raise SurveySheetError('sheet style numeric settings must be positive and finite')
        if self.grid_minor_m > self.grid_major_m:
            raise SurveySheetError('grid_minor_m must not exceed grid_major_m')
        if self.min_width_in > self.max_width_in:
            raise SurveySheetError('min_width_in must not exceed max_width_in')
        for name in ('paper', 'ink', 'accent', 'grid', 'low_confidence', 'profile'):
            if not mcolors.is_color_like(getattr(self, name)):
                raise SurveySheetError(f'{name} is not a valid colour')

    @property
    def unit_scale(self) -> float:
        return _UNIT_SCALE[self.label_unit]

    def format_length(self, value_m: float) -> str:
        return f'{abs(float(value_m)) * self.unit_scale:.0f}'

@cache
def _pick_font(candidates: tuple[str, ...], fallback: str) -> str:
    names = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in names:
            return candidate
    return fallback

def font_sans() -> str:
    return _pick_font(('Space Grotesk', 'Segoe UI', 'DejaVu Sans'), 'sans-serif')

def font_mono() -> str:
    return _pick_font(('JetBrains Mono', 'Consolas', 'DejaVu Sans Mono'), 'monospace')

def _rc(style: SheetStyle) -> dict[str, Any]:
    return {'font.family': font_sans(), 'font.size': 8, 'axes.edgecolor': style.ink, 'axes.labelcolor': style.ink, 'axes.titlecolor': style.ink, 'xtick.color': style.ink, 'ytick.color': style.ink, 'figure.facecolor': style.paper, 'axes.facecolor': style.paper, 'savefig.facecolor': style.paper}

def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SurveySheetError(f'{label} must be a mapping')
    return value

def _sequence(value: Any, label: str) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SurveySheetError(f'{label} must be a sequence of mappings')
    items: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        items.append(_mapping(item, f'{label}[{index}]'))
    return items

def _num(item: Mapping[str, Any], key: str, label: str) -> float:
    if key not in item:
        raise SurveySheetError(f"{label} is missing '{key}'")
    try:
        value = float(item[key])
    except (TypeError, ValueError) as exc:
        raise SurveySheetError(f"{label} field '{key}' must be numeric") from exc
    if not math.isfinite(value):
        raise SurveySheetError(f"{label} field '{key}' must be finite")
    return value

def _opt_num(item: Mapping[str, Any], key: str, label: str) -> float | None:
    if item.get(key) is None:
        return None
    return _num(item, key, label)

def _text(item: Mapping[str, Any], key: str, label: str, default: str | None=None) -> str:
    value = item.get(key, default)
    if value is None:
        raise SurveySheetError(f"{label} is missing '{key}'")
    return str(value)

def _is_low(confidence: str) -> bool:
    return confidence == CONFIDENCE_LOW

def _xyz(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3 or array.shape[0] == 0:
        raise SurveySheetError('points must be a non-empty (N, 3) array')
    return array[:, :3]

def _rgb(rgb: Any, count: int) -> np.ndarray | None:
    if rgb is None:
        return None
    array = np.asarray(rgb)
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] < 3:
        raise SurveySheetError('rgb must be an (N, 3) array aligned with points')
    return np.clip(array[:, :3], 0, 255).astype(np.uint8)

def _frame_info(frame_info: Any) -> Mapping[str, Any]:
    if frame_info is None:
        return {}
    return _mapping(frame_info, 'frame_info')

def _destination(destination: Any) -> Path:
    path = Path(destination)
    if path.suffix.lower() != '.png':
        raise SurveySheetError('sheet destination must use the .png extension')
    return path

def _raster_frame(bounds: tuple[float, float, float, float], res: float) -> tuple[int, int, tuple[float, float, float, float]]:
    u0, u1, v0, v1 = bounds
    if not (u1 > u0 and v1 > v0):
        raise SurveySheetError('raster bounds must be ordered')
    width = math.ceil((u1 - u0) / res) + 1
    height = math.ceil((v1 - v0) / res) + 1
    return (width, height, (u0, u0 + width * res, v0, v0 + height * res))

def _inside(u: np.ndarray, v: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    u0, u1, v0, v1 = bounds
    return np.isfinite(u) & np.isfinite(v) & (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)

def _occupancy(u: np.ndarray, v: np.ndarray, res: float, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    width, height, extent = _raster_frame(bounds, res)
    grid = np.zeros((height, width), dtype=np.float32)
    m = _inside(u, v, bounds)
    if m.any():
        iu = ((u[m] - bounds[0]) / res).astype(np.intp)
        iv = ((v[m] - bounds[2]) / res).astype(np.intp)
        np.add.at(grid, (iv, iu), 1.0)
    return (grid, extent)

def _nearest_rgb(u: np.ndarray, v: np.ndarray, depth: np.ndarray, rgb: np.ndarray, res: float, bounds: tuple[float, float, float, float], *, nearest: Literal['min', 'max']) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    width, height, extent = _raster_frame(bounds, res)
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    m = _inside(u, v, bounds) & np.isfinite(depth)
    if m.any():
        iu = ((u[m] - bounds[0]) / res).astype(np.intp)
        iv = ((v[m] - bounds[2]) / res).astype(np.intp)
        order = np.argsort(depth[m], kind='stable')
        if nearest == 'min':
            order = order[::-1]
        image[iv[order], iu[order]] = rgb[m][order]
    return (image, extent)

def _show_density(ax: Axes, grid: np.ndarray, extent: tuple[float, float, float, float]) -> None:
    values = np.log1p(np.asarray(grid, dtype=np.float32))
    filled = values[values > 0]
    vmax = float(np.percentile(filled, 99)) if filled.size else 1.0
    ax.imshow(values, origin='lower', extent=extent, cmap='gray_r', vmin=0.0, vmax=max(vmax, 1.0), interpolation='nearest')

def _ticks(limits: tuple[float, float], step: float) -> np.ndarray:
    values = np.arange(math.floor(limits[0]), math.ceil(limits[1]) + 1e-09, step)
    return values[(values >= limits[0] - 1e-09) & (values <= limits[1] + 1e-09)]

def _draw_grid(ax: Axes, xlim: tuple[float, float], ylim: tuple[float, float], style: SheetStyle) -> None:
    major, minor = (style.grid_major_m, style.grid_minor_m)
    ax.set_xticks(_ticks(xlim, major))
    ax.set_yticks(_ticks(ylim, major))
    ax.set_xticks(_ticks(xlim, minor), minor=True)
    ax.set_yticks(_ticks(ylim, minor), minor=True)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, which='major', color=style.grid, lw=0.5, alpha=0.9)
    ax.grid(True, which='minor', color=style.grid, lw=0.3, alpha=0.5)
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

def _wall_pen(confidence: str, style: SheetStyle) -> tuple[str, float, str]:
    if _is_low(confidence):
        return (style.low_confidence, 0.8, '--')
    if confidence == CONFIDENCE_MEDIUM:
        return (style.ink, 1.4, '-')
    return (style.ink, 2.2, '-')

def _dim_label(style: SheetStyle, value_m: float, low: bool, text: str | None) -> str:
    label = style.format_length(value_m) if text is None else text
    return f'{label}  {HAND_MEASURE_LABEL}' if low else label

def _dim_h(ax: Axes, x0: float, x1: float, yy: float, style: SheetStyle, *, low: bool=False, text: str | None=None, above: bool=True, fontsize: float=7.0, tick: float=0.1) -> None:
    color = style.low_confidence if low else style.accent
    ax.plot([x0, x1], [yy, yy], color=color, lw=0.7, ls='--' if low else '-')
    for xx in (x0, x1):
        ax.plot([xx, xx], [yy - tick, yy + tick], color=color, lw=0.7)
    ax.text((x0 + x1) / 2.0, yy + (0.06 if above else -0.06), _dim_label(style, x1 - x0, low, text), ha='center', va='bottom' if above else 'top', fontsize=fontsize, color=color, family=font_mono())

def _dim_v(ax: Axes, y0: float, y1: float, xx: float, style: SheetStyle, *, low: bool=False, text: str | None=None, fontsize: float=7.0, tick: float=0.1, right: bool=False) -> None:
    color = style.low_confidence if low else style.accent
    ax.plot([xx, xx], [y0, y1], color=color, lw=0.7, ls='--' if low else '-')
    for yy in (y0, y1):
        ax.plot([xx - tick, xx + tick], [yy, yy], color=color, lw=0.7)
    ax.text(xx + (0.06 if right else -0.06), (y0 + y1) / 2.0, _dim_label(style, y1 - y0, low, text), ha='left' if right else 'right', va='center', fontsize=fontsize, color=color, family=font_mono(), rotation=90)

def _legend_note(ax: Axes, style: SheetStyle, x: float, y: float) -> None:
    ax.text(x, y, f'{HAND_MEASURE_LABEL} = turuncu kesik (dusuk guven, tek kesitte gorulen)', fontsize=7, color=style.low_confidence, family=font_mono())
_HEADER_IN = 1.15
_FOOTER_LINE_IN = 0.13
_MONO_CHAR_IN = {7.0: 0.062, 8.0: 0.071}

def _wrap(text: str, width_in: float, char_in: float) -> list[str]:
    columns = max(40, int(width_in * 0.97 / char_in))
    return textwrap.wrap(text, width=columns, break_long_words=False, break_on_hyphens=False) or ['']

def _figure(width_m: float, height_m: float, style: SheetStyle, frame_info: Mapping[str, Any], *, rows: int=1, extra_in: float=0.0) -> Figure:
    width_m = max(float(width_m), 1.0)
    height_m = max(float(height_m), 1.0)
    width_in = min(max(width_m * style.inches_per_m, style.min_width_in), style.max_width_in)
    footer_lines = _wrap(_footer_text(style, frame_info), width_in, _MONO_CHAR_IN[7.0])
    footer_in = 0.55 + _FOOTER_LINE_IN * len(footer_lines)
    scale = width_in / width_m
    height_in = rows * (height_m * scale + 0.9) + _HEADER_IN + footer_in + extra_in
    height_in = min(max(height_in, 4.5), style.max_width_in * 1.5)
    figure = Figure(figsize=(width_in, height_in), dpi=style.dpi, facecolor=style.paper)
    FigureCanvasAgg(figure)
    figure.text(0.01, 0.06 / height_in, '\n'.join(footer_lines), fontsize=7, family=font_mono(), color=_MUTED, va='bottom', linespacing=1.4)
    figure.subplots_adjust(top=1.0 - _HEADER_IN / height_in, bottom=footer_in / height_in, left=0.06, right=0.985)
    return figure

def _title_block(figure: Figure, style: SheetStyle, frame_info: Mapping[str, Any], title: str, subtitle: str) -> None:
    project = str(frame_info.get('project') or 'UNTITLED SCAN')
    height_in = figure.get_figheight()
    width_in = figure.get_figwidth()
    figure.text(0.01, 1.0 - 0.1 / height_in, f'CCA  //  {project}  //  {SHEET_STAMP}', fontsize=9, family=font_mono(), color=style.ink, va='top')
    figure.text(0.01, 1.0 - 0.3 / height_in, title, fontsize=13, family=font_sans(), weight='bold', color=style.ink, va='top')
    if subtitle:
        lines = _wrap(subtitle, width_in, _MONO_CHAR_IN[8.0])
        figure.text(0.01, 1.0 - 0.58 / height_in, '\n'.join(lines), fontsize=8, family=font_mono(), color=_MUTED, va='top', linespacing=1.3)

def _footer_text(style: SheetStyle, frame_info: Mapping[str, Any]) -> str:
    parts = [SHEET_STAMP]
    source_name = frame_info.get('source_name')
    if source_name:
        parts.append(f'kaynak {source_name}')
    sha = frame_info.get('source_sha256')
    if sha:
        parts.append(f'sha256 {str(sha)[:12]}')
    tilt = frame_info.get('ground_tilt_deg')
    yaw = frame_info.get('yaw_deg')
    if isinstance(tilt, (int, float)) and isinstance(yaw, (int, float)):
        parts.append(f'zemin egimi {float(tilt):.2f} deg | yaw {float(yaw):.2f} deg')
    sigma = frame_info.get('ground_sigma_m')
    if isinstance(sigma, (int, float)):
        parts.append(f'zemin sigma {float(sigma) * 100.0:.1f} cm')
    flipped = frame_info.get('flipped_180')
    if flipped is not None:
        parts.append('on yuz: 180 deg cevrildi (tarayici tarafi +y idi)' if flipped else 'on yuz: korundu (tarayici tarafi -y)')
    facade_y = frame_info.get('facade_y_m')
    if isinstance(facade_y, (int, float)):
        parts.append(f'cephe y={float(facade_y):.2f} m')
    parts.append(f'olcu etiketi {style.label_unit} | kotlar m (zemin z=0) | grid {style.grid_major_m:g} m')
    return '  |  '.join(parts)

def _save(figure: Figure, destination: Path, style: SheetStyle) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f'.{destination.stem}.', suffix='.png', delete=False) as stream:
            temporary = Path(stream.name)
        figure.savefig(temporary, dpi=style.dpi, format='png', bbox_inches='tight', pad_inches=0.15, facecolor=style.paper)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination

@dataclass(frozen=True, slots=True)
class _Box:
    id: str
    x0: float
    x1: float
    y_front: float
    y_back: float
    eave: float | None
    ridge: float | None
    roof_type: str
    confidence: str

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def depth(self) -> float:
        return self.y_back - self.y_front

    @property
    def low(self) -> bool:
        return _is_low(self.confidence)

    @property
    def top(self) -> float | None:
        values = [v for v in (self.eave, self.ridge) if v is not None]
        return max(values) if values else None

def _boxes(structures: Any) -> list[_Box]:
    boxes: list[_Box] = []
    for index, item in enumerate(_sequence(structures, 'structures')):
        label = f'structures[{index}]'
        box = _Box(id=_text(item, 'id', label, f'S{index + 1:02d}'), x0=_num(item, 'x_start_m', label), x1=_num(item, 'x_end_m', label), y_front=_num(item, 'y_front_m', label), y_back=_num(item, 'y_back_m', label), eave=_opt_num(item, 'eave_z_m', label), ridge=_opt_num(item, 'ridge_z_m', label), roof_type=_text(item, 'roof_type', label, 'belirsiz'), confidence=_text(item, 'confidence', label, CONFIDENCE_LOW))
        if box.x1 <= box.x0:
            raise SurveySheetError(f'{label} must have x_end_m > x_start_m')
        boxes.append(box)
    return sorted(boxes, key=lambda b: b.x0)

def _draw_elevation_boxes(ax: Axes, boxes: Sequence[_Box], style: SheetStyle, *, z_low: float, labels: bool) -> None:
    for box in boxes:
        color = style.low_confidence if box.low else style.accent
        for xx in (box.x0, box.x1):
            ax.plot([xx, xx], [z_low + 0.3, (box.top or 2.0) + 0.6], color=color, lw=0.5, ls=':')
        if box.eave is not None:
            ax.plot([box.x0, box.x1], [box.eave, box.eave], color=color, lw=0.8, ls='--')
            if labels:
                ax.text((box.x0 + box.x1) / 2.0, box.eave + 0.08, f'{box.id}  sacak +{box.eave:.2f}', ha='center', fontsize=6.5, color=color, family=font_mono())
        elif labels:
            ax.text((box.x0 + box.x1) / 2.0, (box.ridge or 1.5) + 0.08, f'{box.id}  sacak: {HAND_MEASURE_LABEL}', ha='center', fontsize=6.5, color=style.low_confidence, family=font_mono())
        if labels and box.ridge is not None and (box.eave is None or box.ridge - box.eave > 0.15):
            ax.text((box.x0 + box.x1) / 2.0, box.ridge + 0.3, f'mahya +{box.ridge:.2f}', ha='center', fontsize=6.5, color=_MUTED, family=font_mono())

def _draw_width_chain(ax: Axes, boxes: Sequence[_Box], style: SheetStyle, yy: float) -> None:
    for box in boxes:
        _dim_h(ax, box.x0, box.x1, yy, style, low=box.low, above=False)

def make_plan_sheet(slices: Mapping[str, Any], walls: Sequence[Mapping[str, Any]], structures: Sequence[Mapping[str, Any]], partitions: Mapping[str, Any], frame_info: Mapping[str, Any], destination: str | Path, style: SheetStyle | None=None) -> Path:
    style = style or SheetStyle()
    slices = _mapping(slices, 'slices')
    partitions = _mapping(partitions, 'partitions')
    info = _frame_info(frame_info)
    output = _destination(destination)
    wall_items = _sequence(walls, 'walls')
    boxes = _boxes(structures)
    bands = _mapping(slices.get('bands'), "slices['bands']")
    if not bands:
        raise SurveySheetError('slices carry no z bands')
    key, band = min(((k, _mapping(v, f"slices['bands'][{k!r}]")) for k, v in bands.items()), key=lambda kv: abs(sum((float(z) for z in kv[1]['z_band_m'])) / 2.0 - 1.1))
    z_band = tuple((float(v) for v in band['z_band_m']))
    grid = np.asarray(band['occupancy'], dtype=np.float32)
    extent = tuple((float(v) for v in slices['extent_m']))
    if grid.ndim != 2 or len(extent) != 4:
        raise SurveySheetError('slice occupancy must be a 2-D raster with a four-value extent')
    facade_y = _num(partitions, 'facade_y_m', 'partitions')
    positions = [float(v) for v in partitions.get('x_positions_m', [])]
    spacings = [float(v) for v in partitions.get('spacings_m', [])]
    confidences = [str(v) for v in partitions.get('confidence', [])]
    if len(spacings) != max(len(positions) - 1, 0):
        spacings = [b - a for a, b in itertools.pairwise(positions)]
    if len(confidences) != len(positions):
        confidences = [CONFIDENCE_LOW] * len(positions)
    x_lo, x_hi = (extent[0] - 0.5, extent[1] + 0.5)
    y_lo = min(extent[2], facade_y - 2.6) - 0.3
    y_hi = extent[3] + 0.5
    with rc_context(_rc(style)):
        figure = _figure(x_hi - x_lo, y_hi - y_lo, style, info)
        ax = figure.subplots()
        _show_density(ax, grid, extent)
        _draw_grid(ax, (x_lo, x_hi), (y_lo, y_hi), style)
        for index, wall in enumerate(wall_items):
            label = f'walls[{index}]'
            axis = _text(wall, 'axis', label)
            pos = _num(wall, 'pos_m', label)
            start, end = (_num(wall, 'start_m', label), _num(wall, 'end_m', label))
            color, lw, ls = _wall_pen(_text(wall, 'confidence', label, CONFIDENCE_LOW), style)
            if axis == 'x':
                ax.plot([start, end], [pos, pos], color=color, lw=lw, ls=ls)
            elif axis == 'y':
                ax.plot([pos, pos], [start, end], color=color, lw=lw, ls=ls)
            else:
                raise SurveySheetError(f"{label} axis must be 'x' or 'y'")
        for box in boxes:
            color = style.low_confidence if box.low else style.accent
            ax.add_patch(Rectangle((box.x0, box.y_front), box.width, box.depth, fill=False, ec=color, lw=1.0, ls='--' if box.low else '-'))
            text = f'{box.id}  {style.format_length(box.width)} x {style.format_length(box.depth)}'
            if box.low:
                text += f'  {HAND_MEASURE_LABEL}'
            ax.text((box.x0 + box.x1) / 2.0, box.y_back + 0.15, text, ha='center', fontsize=7, color=color, family=font_mono(), weight='bold')
            _dim_v(ax, box.y_front, box.y_back, box.x1 - 0.3, style, low=box.low, fontsize=6)
        for (a, b), spacing, (ca, cb) in zip(itertools.pairwise(positions), spacings, itertools.pairwise(confidences)):
            if b - a > 0.15:
                _dim_h(ax, a, b, facade_y - 0.9, style, low=_is_low(ca) or _is_low(cb), text=style.format_length(spacing))
        for xp in positions:
            ax.plot([xp, xp], [facade_y - 2.2, facade_y], color=style.accent, lw=0.4, ls=':')
        if len(positions) > 1:
            total_low = any((_is_low(c) for c in confidences))
            total = float(partitions.get('total_m', positions[-1] - positions[0]))
            _dim_h(ax, positions[0], positions[-1], facade_y - 1.9, style, low=total_low, text=f'{style.format_length(total)}  (cephe bolme zinciri)')
        ax.text(x_lo + 0.4, facade_y + 0.15, f'ON CEPHE  y={facade_y:.2f} m', fontsize=8, color=style.accent, family=font_mono())
        _legend_note(ax, style, x_lo + 0.4, y_lo + 0.15)
        ax.set_xlabel('x [m]  (tarayici tarafindan bakista saga artar)')
        ax.set_ylabel('y [m]')
        _title_block(figure, style, info, f'PLAN  -  kesit z = {z_band[0]:.2f}-{z_band[1]:.2f} m  |  olculer {style.label_unit}  |  {SHEET_STAMP}', f'dilim {key} | duvarlar: kalin = HIGH, ince = MEDIUM, turuncu kesik = LOW | yapi ayak izleri genislik x derinlik {style.label_unit}')
        return _save(figure, output, style)

def _front_limit(walls: Sequence[Mapping[str, Any]], facade_profile: Mapping[str, Any], y: np.ndarray, min_y: float | None) -> float:
    if min_y is not None:
        if not math.isfinite(float(min_y)):
            raise SurveySheetError('min_y must be finite')
        return float(min_y)
    try:
        facade = find_facade(list(walls))
        return float(front_view_min_y(list(walls), facade))
    except (SurveyPlanError, KeyError, TypeError, ValueError):
        facade_y = facade_profile.get('facade_y_m')
        if isinstance(facade_y, (int, float)):
            return float(facade_y) - SurveyPlanConfig().front_margin_m
        return float(np.nanmin(y))

def make_elevation_sheet(points: Any, rgb: Any | None, walls: Sequence[Mapping[str, Any]], structures: Sequence[Mapping[str, Any]], facade_profile: Mapping[str, Any], frame_info: Mapping[str, Any], destination: str | Path, style: SheetStyle | None=None, *, min_y: float | None=None) -> Path:
    style = style or SheetStyle()
    xyz = _xyz(points)
    colours = _rgb(rgb, xyz.shape[0])
    profile = _mapping(facade_profile, 'facade_profile')
    info = _frame_info(frame_info)
    output = _destination(destination)
    wall_items = _sequence(walls, 'walls')
    boxes = _boxes(structures)
    x, y, z = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
    limit = _front_limit(wall_items, profile, y, min_y)
    facade_y = _opt_num(profile, 'facade_y_m', 'facade_profile')
    profile_x = np.asarray(profile.get('x_m', []), dtype=np.float64)
    profile_z = np.asarray(profile.get('z_top_m', []), dtype=np.float64)
    if profile_x.shape != profile_z.shape:
        raise SurveySheetError('facade_profile x_m and z_top_m must have the same length')
    keep = y >= limit
    xs, ys, zs = (x[keep], y[keep], z[keep])
    finite_z = z[np.isfinite(z)]
    z_top = float(np.max(finite_z)) if finite_z.size else 3.0
    for box in boxes:
        if box.top is not None:
            z_top = max(z_top, box.top)
    x_lo, x_hi = (math.floor(float(np.nanmin(x))), math.ceil(float(np.nanmax(x))))
    z_lo, z_hi = (min(-0.8, math.floor(float(np.nanmin(finite_z)) if finite_z.size else 0.0)), math.ceil(z_top) + 0.6)
    bounds = (float(x_lo), float(x_hi), float(z_lo), float(z_hi))
    res = style.raster_m
    density, extent = _occupancy(xs, zs, res, bounds)
    rows: list[tuple[str, np.ndarray, bool]] = []
    if colours is not None:
        front_rgb, _ = _nearest_rgb(xs, zs, ys, colours[keep], res, bounds, nearest='min')
        rows.append(('ON GORUNUS (RGB) - tarayici tarafindan, x saga artar, en yakin nokta (en kucuk y) kazanir', front_rgb, False))
    rows.append((f'ON GORUNUS (nokta yogunlugu) - yapi genislikleri {style.label_unit}, cephe ust profili', density, False))
    if colours is not None:
        back_img, _ = _nearest_rgb(xs, zs, ys, colours[keep], res, bounds, nearest='max')
    else:
        back_img = density
    rows.append(('ARKA GORUNUS (x aynali) - arka taraftan, en uzak nokta (en buyuk y) kazanir', back_img, True))
    with rc_context(_rc(style)):
        figure = _figure(x_hi - x_lo + 1.0, z_hi - z_lo, style, info, rows=len(rows))
        axes = figure.subplots(len(rows), 1, sharex=False)
        axes = list(np.atleast_1d(axes))
        for ax, (title, image, mirrored) in zip(axes, rows):
            if image.ndim == 3:
                ax.imshow(image, origin='lower', extent=extent, interpolation='nearest')
            else:
                _show_density(ax, image, extent)
            _draw_grid(ax, (x_lo - 0.5, x_hi + 0.5), (z_lo, z_hi), style)
            ax.axhline(0.0, color=style.ink, lw=0.8)
            _draw_elevation_boxes(ax, boxes, style, z_low=z_lo, labels=True)
            if mirrored:
                ax.invert_xaxis()
                ax.set_xlabel('x [m]  (aynali: arka taraftan bakis)')
            else:
                ax.set_xlabel('x [m]')
            ax.set_ylabel('z [m]')
            ax.set_title(title, fontsize=9, loc='left')
        density_ax = axes[1] if colours is not None else axes[0]
        _draw_width_chain(density_ax, boxes, style, -0.45)
        if profile_x.size:
            density_ax.plot(profile_x, profile_z, color=style.profile, lw=0.8, alpha=0.9, label='cephe ust profili z99')
            density_ax.legend(loc='upper right', fontsize=7)
        for box in boxes:
            if box.eave is not None:
                _dim_v(density_ax, 0.0, box.eave, box.x0 + 0.25, style, low=box.low, fontsize=6, right=True)
        subtitle = f'on gorunus alt siniri y >= {limit:.2f} m (karsi taraf duvarlari haric)'
        if facade_y is not None:
            subtitle += f' | cephe y={facade_y:.2f} m'
        subtitle += ' | sacak = cephe duzlemi z99, mahya = tum derinlik z99'
        _title_block(figure, style, info, f'ON / ARKA GORUNUS  |  kotlar m (zemin = 0.00), genislikler {style.label_unit}  |  {SHEET_STAMP}', subtitle)
        return _save(figure, output, style)

def make_box_sheet(points: Any, structures: Sequence[Mapping[str, Any]], destination: str | Path, style: SheetStyle | None=None, *, frame_info: Mapping[str, Any] | None=None, min_y: float | None=None, plan_band_m: tuple[float, float]=(0.95, 1.25)) -> Path:
    style = style or SheetStyle()
    xyz = _xyz(points)
    boxes = _boxes(structures)
    info = _frame_info(frame_info)
    output = _destination(destination)
    if len(plan_band_m) != 2 or not plan_band_m[0] < plan_band_m[1]:
        raise SurveySheetError('plan_band_m must be an ordered (low, high) pair')
    x, y, z = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
    facade_y = boxes[0].y_front if boxes else float(np.nanmedian(y))
    limit = float(min_y) if min_y is not None else facade_y - SurveyPlanConfig().front_margin_m
    if not math.isfinite(limit):
        raise SurveySheetError('min_y must be finite')
    keep = y >= limit
    finite_z = z[np.isfinite(z)]
    tops = [b.top for b in boxes if b.top is not None]
    z_top = max([float(np.max(finite_z)) if finite_z.size else 3.0, *tops])
    x_lo, x_hi = (math.floor(float(np.nanmin(x))), math.ceil(float(np.nanmax(x))))
    z_lo, z_hi = (-1.4, math.ceil(z_top) + 0.6)
    y_lo = min(facade_y - 2.6, float(np.nanmin(y[keep])) if keep.any() else facade_y - 2.6) - 0.3
    y_hi = max(math.ceil(float(np.nanmax(y))), max((b.y_back for b in boxes), default=facade_y)) + 0.8
    res = style.raster_m
    elev, elev_extent = _occupancy(x[keep], z[keep], res, (float(x_lo), float(x_hi), -0.8, float(z_hi)))
    band = (z > plan_band_m[0]) & (z < plan_band_m[1])
    plan, plan_extent = _occupancy(x[band], y[band], res, (float(x_lo), float(x_hi), float(math.floor(y_lo)), float(y_hi)))
    with rc_context(_rc(style)):
        figure = _figure(x_hi - x_lo + 1.0, z_hi - z_lo + (y_hi - y_lo), style, info, rows=1, extra_in=1.2)
        top_ax, plan_ax = figure.subplots(2, 1, gridspec_kw={'height_ratios': [z_hi - z_lo, y_hi - y_lo]})
        _show_density(top_ax, elev, elev_extent)
        _draw_grid(top_ax, (x_lo - 0.5, x_hi + 0.5), (z_lo, z_hi), style)
        top_ax.axhline(0.0, color=style.ink, lw=0.8)
        for box in boxes:
            color = style.low_confidence if box.low else style.accent
            ls = '--' if box.low else '-'
            if box.eave is not None:
                top_ax.add_patch(Rectangle((box.x0, 0.0), box.width, box.eave, fill=False, ec=color, lw=2.2, ls=ls))
                top_ax.text((box.x0 + box.x1) / 2.0, box.eave + 0.1, f'{box.id}  +{box.eave:.2f}', ha='center', fontsize=8, color=color, family=font_mono(), weight='bold')
                _dim_v(top_ax, 0.0, box.eave, box.x0 + 0.25, style, low=box.low, fontsize=7, right=True)
                if box.ridge is not None and box.ridge - box.eave > 0.15:
                    top_ax.add_patch(Rectangle((box.x0, box.eave), box.width, box.ridge - box.eave, fill=False, ec=color, lw=1.0, ls='--'))
                    top_ax.text((box.x0 + box.x1) / 2.0, box.ridge + 0.1, f'mahya +{box.ridge:.2f}', ha='center', fontsize=7, color=_MUTED, family=font_mono())
            else:
                height = box.ridge if box.ridge is not None else 2.0
                top_ax.add_patch(Rectangle((box.x0, 0.0), box.width, height, fill=False, ec=style.low_confidence, lw=1.2, ls='--'))
                top_ax.text((box.x0 + box.x1) / 2.0, height + 0.1, f'{box.id}  sacak: {HAND_MEASURE_LABEL}', ha='center', fontsize=8, color=style.low_confidence, family=font_mono(), weight='bold')
        _draw_width_chain(top_ax, boxes, style, -0.45)
        top_ax.set_title(f'ON GORUNUS - kutle kutulari (genislik {style.label_unit}, sacak kotu m)', fontsize=9, loc='left')
        top_ax.set_ylabel('z [m]')
        _legend_note(top_ax, style, x_lo - 0.3, z_lo + 0.15)
        _show_density(plan_ax, plan, plan_extent)
        _draw_grid(plan_ax, (x_lo - 0.5, x_hi + 0.5), (y_lo, y_hi), style)
        for box in boxes:
            color = style.low_confidence if box.low else style.accent
            plan_ax.add_patch(Rectangle((box.x0, box.y_front), box.width, box.depth, fill=False, ec=color, lw=2.0, ls='--' if box.low else '-'))
            text = f'{box.id}  {style.format_length(box.width)} x {style.format_length(box.depth)}'
            if box.low:
                text += f'  {HAND_MEASURE_LABEL}'
            plan_ax.text((box.x0 + box.x1) / 2.0, box.y_back + 0.15, text, ha='center', fontsize=8, color=color, family=font_mono(), weight='bold')
            _dim_v(plan_ax, box.y_front, box.y_back, box.x1 - 0.3, style, low=box.low, fontsize=7)
        if boxes:
            chain = [boxes[0].x0, *(b.x1 for b in boxes)]
            for box in boxes:
                _dim_h(plan_ax, box.x0, box.x1, facade_y - 0.9, style, low=box.low)
            total_low = any((b.low for b in boxes))
            _dim_h(plan_ax, chain[0], chain[-1], facade_y - 1.9, style, low=total_low, text=f'{style.format_length(chain[-1] - chain[0])}  (yapi kutulari dis siniri)')
        plan_ax.text(x_lo - 0.3, facade_y + 0.15, f'ON CEPHE y={facade_y:.2f} m', fontsize=8, color=style.accent, family=font_mono())
        plan_ax.set_title(f'PLAN (kesit z={plan_band_m[0]:.2f}-{plan_band_m[1]:.2f} m) - ayak izleri (genislik x derinlik {style.label_unit})', fontsize=9, loc='left')
        plan_ax.set_xlabel('x [m]  (tarayici tarafindan bakista saga artar)')
        plan_ax.set_ylabel('y [m]')
        _title_block(figure, style, info, f'KUTLE KUTULARI  |  yapi = dikdortgen  |  {len(boxes)} yapi  |  {SHEET_STAMP}', 'bolunme: bolme duvarlari + cati basamaklari | derinlik = arka nokta y98 / bolme duvari uzunlugu | ic bolmeler kutuyu bolmez')
        return _save(figure, output, style)

def make_section_sheet(points: Any, structures: Sequence[Mapping[str, Any]], destination: str | Path, style: SheetStyle | None=None, *, frame_info: Mapping[str, Any] | None=None, half_band_m: float=0.5, min_y: float | None=None) -> Path:
    style = style or SheetStyle()
    xyz = _xyz(points)
    boxes = _boxes(structures)
    info = _frame_info(frame_info)
    output = _destination(destination)
    if not math.isfinite(float(half_band_m)) or float(half_band_m) <= 0.0:
        raise SurveySheetError('half_band_m must be positive and finite')
    x, y, z = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
    if not boxes:
        mid = float((np.nanmin(x) + np.nanmax(x)) / 2.0)
        panels: list[tuple[str, float, _Box | None]] = [(f'KESIT (tum tarama)  x={mid:.2f} m', mid, None)]
    else:
        panels = [(f'KESIT {b.id}  x={(b.x0 + b.x1) / 2.0:.2f} m', (b.x0 + b.x1) / 2.0, b) for b in boxes]
    facade_y = boxes[0].y_front if boxes else float(np.nanmedian(y))
    limit = float(min_y) if min_y is not None else facade_y - SurveyPlanConfig().front_margin_m
    if not math.isfinite(limit):
        raise SurveySheetError('min_y must be finite')
    finite_z = z[np.isfinite(z)]
    tops = [b.top for b in boxes if b.top is not None]
    z_top = max([float(np.max(finite_z)) if finite_z.size else 3.0, *tops])
    y_lo = math.floor(max(limit, float(np.nanmin(y))))
    y_hi = math.ceil(max(float(np.nanmax(y)), max((b.y_back for b in boxes), default=facade_y)))
    z_lo, z_hi = (-1.0, math.ceil(z_top) + 0.6)
    bounds = (float(y_lo), float(y_hi), -0.8, float(z_hi))
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    with rc_context(_rc(style)):
        figure = _figure((y_hi - y_lo + 1.0) * columns, (z_hi - z_lo) * rows, style, info, rows=1, extra_in=0.6 * rows)
        axes = list(np.atleast_1d(figure.subplots(rows, columns)).ravel())
        for ax in axes[len(panels):]:
            ax.set_visible(False)
        for ax, (title, mid, box) in zip(axes, panels):
            slab = (np.abs(x - mid) < float(half_band_m)) & (y >= limit)
            grid, extent = _occupancy(y[slab], z[slab], style.raster_m, bounds)
            _show_density(ax, grid, extent)
            _draw_grid(ax, (y_lo - 0.3, y_hi + 0.3), (z_lo, z_hi), style)
            ax.axhline(0.0, color=style.ink, lw=0.8)
            if box is not None:
                color = style.low_confidence if box.low else style.accent
                ax.plot([box.y_front, box.y_front], [-0.3, z_hi - 0.3], color=color, lw=0.6, ls=':')
                ax.plot([box.y_back, box.y_back], [-0.3, z_hi - 0.3], color=color, lw=0.6, ls=':')
                _dim_h(ax, box.y_front, box.y_back, -0.45, style, low=box.low, above=False)
                if box.eave is not None:
                    ax.plot([box.y_front - 0.3, box.y_back + 0.3], [box.eave, box.eave], color=color, lw=0.8, ls='--')
                    _dim_v(ax, 0.0, box.eave, box.y_front - 0.4, style, low=box.low, fontsize=6.5)
                    ax.text(box.y_front + 0.1, box.eave + 0.08, f'sacak +{box.eave:.2f}', fontsize=6.5, color=color, family=font_mono())
                else:
                    ax.text(box.y_front + 0.1, 1.5, f'sacak: {HAND_MEASURE_LABEL}', fontsize=6.5, color=style.low_confidence, family=font_mono())
                if box.ridge is not None and (box.eave is None or box.ridge - box.eave > 0.15):
                    ax.plot([box.y_front - 0.3, box.y_back + 0.3], [box.ridge, box.ridge], color=_MUTED, lw=0.6, ls='--')
                    ax.text(box.y_front + 0.1, box.ridge + 0.08, f'mahya +{box.ridge:.2f}  ({box.roof_type})', fontsize=6.5, color=_MUTED, family=font_mono())
                ax.text(box.y_front + 0.1, z_lo + 0.12, f'genislik {style.format_length(box.width)}', fontsize=6.5, color=color, family=font_mono())
            ax.set_title(f'{title}  (dilim +-{float(half_band_m):.2f} m)', fontsize=9, loc='left')
            ax.set_xlabel('y [m]  (cepheden arkaya)')
            ax.set_ylabel('z [m]')
        _title_block(figure, style, info, f'KESITLER  |  y-z, yapi ortasindan  |  derinlikler {style.label_unit}, kotlar m  |  {SHEET_STAMP}', f'{len(panels)} kesit | dilim genisligi {2.0 * float(half_band_m):.2f} m | y >= {limit:.2f} m')
        return _save(figure, output, style)

def make_slice_sheets(slices: Mapping[str, Any], out_dir: str | Path, style: SheetStyle | None=None, *, frame_info: Mapping[str, Any] | None=None) -> list[Path]:
    style = style or SheetStyle()
    slices = _mapping(slices, 'slices')
    info = _frame_info(frame_info)
    bands = _mapping(slices.get('bands'), "slices['bands']")
    if not bands:
        raise SurveySheetError('slices carry no z bands')
    extent = tuple((float(v) for v in slices['extent_m']))
    if len(extent) != 4:
        raise SurveySheetError('slices extent_m must have four values')
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    x_lo, x_hi, y_lo, y_hi = (extent[0] - 0.5, extent[1] + 0.5, extent[2] - 0.5, extent[3] + 0.5)
    written: list[Path] = []
    with rc_context(_rc(style)):
        for key in sorted(bands):
            band = _mapping(bands[key], f"slices['bands'][{key!r}]")
            z_band = tuple((float(v) for v in band['z_band_m']))
            grid = np.asarray(band['occupancy'], dtype=np.float32)
            if grid.ndim != 2:
                raise SurveySheetError(f'slice {key} occupancy must be a 2-D raster')
            figure = _figure(x_hi - x_lo, y_hi - y_lo, style, info)
            ax = figure.subplots()
            _show_density(ax, grid, extent)
            _draw_grid(ax, (x_lo, x_hi), (y_lo, y_hi), style)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            _title_block(figure, style, info, f'PLAN DILIMI  z = {z_band[0]:.2f}-{z_band[1]:.2f} m  |  {SHEET_STAMP}', f"dilim {key} | nokta yogunlugu (log) | raster {float(slices.get('resolution_m', style.raster_m)) * 100:.0f} cm")
            written.append(_save(figure, directory / f'slice_{key}.png', style))
        roof = slices.get('roof_z')
        if roof is not None:
            roof = np.asarray(roof, dtype=np.float32)
            if roof.ndim != 2:
                raise SurveySheetError('slices roof_z must be a 2-D raster')
            roof_rgb = slices.get('roof_rgb')
            panels = 2 if roof_rgb is not None else 1
            figure = _figure(x_hi - x_lo, (y_hi - y_lo) * panels, style, info, rows=1, extra_in=0.6 * panels)
            axes = list(np.atleast_1d(figure.subplots(panels, 1)))
            masked = np.ma.masked_invalid(roof)
            image = axes[0].imshow(masked, origin='lower', extent=extent, cmap='viridis', interpolation='nearest')
            figure.colorbar(image, ax=axes[0], fraction=0.02, pad=0.01, label='z [m]')
            _draw_grid(axes[0], (x_lo, x_hi), (y_lo, y_hi), style)
            axes[0].set_title('CATI KOTU - hucre basina en yuksek z', fontsize=9, loc='left')
            axes[0].set_ylabel('y [m]')
            if roof_rgb is not None:
                colours = np.asarray(roof_rgb, dtype=np.uint8)
                if colours.ndim != 3 or colours.shape[2] < 3:
                    raise SurveySheetError('slices roof_rgb must be an (H, W, 3) raster')
                axes[1].imshow(colours[:, :, :3], origin='lower', extent=extent, interpolation='nearest')
                _draw_grid(axes[1], (x_lo, x_hi), (y_lo, y_hi), style)
                axes[1].set_title('USTTEN GORUNUS (RGB) - en yuksek nokta kazanir', fontsize=9, loc='left')
                axes[1].set_ylabel('y [m]')
            axes[-1].set_xlabel('x [m]')
            _title_block(figure, style, info, f'CATI / USTTEN GORUNUS  |  kotlar m (zemin = 0.00)  |  {SHEET_STAMP}', 'roof_z = hucre basina max z | RGB = en yuksek noktanin rengi')
            written.append(_save(figure, directory / 'roof.png', style))
    return written
__all__ = ['HAND_MEASURE_LABEL', 'SHEET_STAMP', 'SheetStyle', 'SurveySheetError', 'font_mono', 'font_sans', 'make_box_sheet', 'make_elevation_sheet', 'make_plan_sheet', 'make_section_sheet', 'make_slice_sheets']
