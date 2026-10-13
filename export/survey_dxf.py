
from __future__ import annotations
import itertools
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import ezdxf
import numpy as np
from ezdxf import bbox, units
from ezdxf.enums import TextEntityAlignment
from geometry.survey_plan import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM, SCHEMA_PARTITIONS, SCHEMA_PROFILE, SCHEMA_SLICES
LAYER_SLICE = 'CCA_SURVEY_SLICE'
LAYER_WALL = 'CCA_SURVEY_WALL'
LAYER_WALL_LOWCONF = 'CCA_SURVEY_WALL_LOWCONF'
LAYER_STRUCT = 'CCA_SURVEY_STRUCT'
LAYER_DIM = 'CCA_SURVEY_DIM'
LAYER_GRID = 'CCA_SURVEY_GRID'
LAYER_TEXT = 'CCA_SURVEY_TEXT'
LAYER_ELEV_FRONT = 'CCA_SURVEY_ELEV_FRONT'
LAYER_ELEV_LIDAR = 'CCA_SURVEY_ELEV_LIDAR'
LAYER_SECTION = 'CCA_SURVEY_SECTION'
SURVEY_LAYERS = (LAYER_SLICE, LAYER_WALL, LAYER_WALL_LOWCONF, LAYER_STRUCT, LAYER_DIM, LAYER_GRID, LAYER_TEXT, LAYER_ELEV_FRONT, LAYER_ELEV_LIDAR, LAYER_SECTION)
DIMSTYLE = 'CCA_SURVEY'
LOW_CONFIDENCE_TEXT = 'ELLE OLC'
DRAFT_LABEL = 'SURVEY DRAFT / UNCALIBRATED'
UNIT_INSUNITS = {'cm': units.CM, 'mm': units.MM}
_M_TO_CM = 100.0
_CM_TO_UNIT = {'cm': 1.0, 'mm': 10.0}
_CONFIDENCES = (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW)
_LAYER_STYLE: dict[str, tuple[int, str]] = {LAYER_SLICE: (8, 'Continuous'), LAYER_WALL: (7, 'Continuous'), LAYER_WALL_LOWCONF: (30, 'DASHED'), LAYER_STRUCT: (1, 'Continuous'), LAYER_DIM: (1, 'Continuous'), LAYER_GRID: (9, 'DOT'), LAYER_TEXT: (7, 'Continuous'), LAYER_ELEV_FRONT: (7, 'Continuous'), LAYER_ELEV_LIDAR: (8, 'Continuous'), LAYER_SECTION: (7, 'Continuous')}
_SLICE_THIN_M = 0.03
_PLAN_BAND_CENTRE_M = 1.1
_ELEVATION_GAP_M = 4.0
_SECTION_GAP_M = 4.0
_SECTION_PITCH_GAP_M = 2.0
_ELEVATION_BACK_MARGIN_M = 1.0
_TEXT_HEIGHT_M = 0.25
_TITLE_HEIGHT_M = 0.5
_DIM_TEXT_M = 0.25
_DIM_ARROW_M = 0.25
_DIM_EXT_M = 0.125
_DIM_GAP_M = 0.0625
_CHAIN_OFFSET_M = 1.0
_TOTAL_OFFSET_M = 2.0
_DEPTH_OFFSET_M = 0.5
_HEIGHT_OFFSET_M = 0.5
_WIDTH_OFFSET_M = 1.0
_LABEL_LIFT_M = 0.5
_RIDGE_STEP_M = 0.15
_GRID_LABEL_M = 0.5
_ELEVATION_FALLBACK_TOP_M = 5.0

class SurveyDxfError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class SurveyDxfResult:
    path: str
    unit: str
    entity_counts_by_layer: dict[str, int]
    audit_errors: int

@dataclass(frozen=True, slots=True)
class _Units:
    unit: str
    factor: float

    def length(self, value_m: float) -> float:
        return value_m * _M_TO_CM * self.factor

    def xy(self, x_m: float, y_m: float) -> tuple[float, float, float]:
        return (self.length(x_m), self.length(y_m), 0.0)

@dataclass(frozen=True, slots=True)
class _Wall:
    id: str
    axis: str
    pos_m: float
    start_m: float
    end_m: float
    confidence: str

@dataclass(frozen=True, slots=True)
class _Structure:
    id: str
    x_start_m: float
    x_end_m: float
    y_front_m: float
    y_back_m: float
    eave_z_m: float | None
    ridge_z_m: float | None
    confidence: str

    @property
    def depth_m(self) -> float:
        return self.y_back_m - self.y_front_m

    @property
    def top_m(self) -> float:
        return max(self.eave_z_m or 0.0, self.ridge_z_m or 0.0)

@dataclass(frozen=True, slots=True)
class _Plan:
    bounds_m: tuple[float, float, float, float]
    resolution_m: float
    band_key: str
    occupancy: np.ndarray
    roof_z: np.ndarray | None

    @property
    def height_m(self) -> float:
        return self.bounds_m[3] - self.bounds_m[2]

def _number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SurveyDxfError(f'{label} must be numeric') from exc
    if not math.isfinite(number):
        raise SurveyDxfError(f'{label} must be finite')
    return number

def _optional_number(value: Any, label: str) -> float | None:
    return None if value is None else _number(value, label)

def _ascii(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.isascii():
        raise SurveyDxfError(f'{label} must be an ASCII string')
    return value

def _confidence(value: Any, label: str) -> str:
    if value not in _CONFIDENCES:
        raise SurveyDxfError(f'{label} confidence must be HIGH, MEDIUM or LOW')
    return str(value)

def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SurveyDxfError(f'{label} must be an object')
    return value

def _validate_slices(slices: Any) -> _Plan:
    data = _mapping(slices, 'slices')
    if data.get('schema') != SCHEMA_SLICES:
        raise SurveyDxfError('unsupported survey slices schema')
    bounds = data.get('bounds_m')
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        raise SurveyDxfError('slices bounds_m must hold x0, x1, y0, y1')
    x0, x1, y0, y1 = (_number(v, f'slices bounds_m[{i}]') for i, v in enumerate(bounds))
    if not (x0 < x1 and y0 < y1):
        raise SurveyDxfError('slices bounds_m must be ordered')
    resolution = _number(data.get('resolution_m'), 'slices resolution_m')
    if resolution <= 0.0:
        raise SurveyDxfError('slices resolution_m must be positive')
    bands = _mapping(data.get('bands'), 'slices bands')
    if not bands:
        raise SurveyDxfError('slices carry no z bands')
    best_key: str | None = None
    best_distance = math.inf
    for key in sorted(bands):
        band = _mapping(bands[key], f'slice band {key}')
        z_band = band.get('z_band_m')
        if not isinstance(z_band, (list, tuple)) or len(z_band) != 2:
            raise SurveyDxfError(f'slice band {key} needs a z_band_m pair')
        low, high = (_number(v, f'slice band {key} z_band_m') for v in z_band)
        distance = abs((low + high) / 2.0 - _PLAN_BAND_CENTRE_M)
        if distance < best_distance:
            best_key, best_distance = (key, distance)
    assert best_key is not None
    try:
        occupancy = np.asarray(bands[best_key]['occupancy'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise SurveyDxfError(f'slice band {best_key} occupancy must be a numeric raster') from exc
    if occupancy.ndim != 2:
        raise SurveyDxfError(f'slice band {best_key} occupancy must be a 2-D raster')
    roof_z: np.ndarray | None = None
    raw_roof = data.get('roof_z')
    if raw_roof is not None:
        try:
            roof_z = np.asarray(raw_roof, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SurveyDxfError('slices roof_z must be a numeric raster') from exc
        if roof_z.ndim != 2:
            raise SurveyDxfError('slices roof_z must be a 2-D raster')
    return _Plan((x0, x1, y0, y1), resolution, best_key, occupancy, roof_z)

def _validate_walls(walls: Any) -> list[_Wall]:
    if not isinstance(walls, Sequence) or isinstance(walls, (str, bytes)):
        raise SurveyDxfError('walls must be a list')
    result: list[_Wall] = []
    for index, raw in enumerate(walls):
        wall = _mapping(raw, f'wall {index}')
        label = f"wall {wall.get('id', index)}"
        axis = wall.get('axis')
        if axis not in ('x', 'y'):
            raise SurveyDxfError(f'{label} axis must be x or y')
        start = _number(wall.get('start_m'), f'{label} start_m')
        end = _number(wall.get('end_m'), f'{label} end_m')
        if end < start:
            raise SurveyDxfError(f'{label} end_m must not precede start_m')
        result.append(_Wall(_ascii(wall.get('id', f'W{index + 1:03d}'), f'{label} id'), str(axis), _number(wall.get('pos_m'), f'{label} pos_m'), start, end, _confidence(wall.get('confidence'), label)))
    return result

def _validate_structures(structures: Any) -> list[_Structure]:
    if not isinstance(structures, Sequence) or isinstance(structures, (str, bytes)):
        raise SurveyDxfError('structures must be a list')
    result: list[_Structure] = []
    for index, raw in enumerate(structures):
        structure = _mapping(raw, f'structure {index}')
        label = f"structure {structure.get('id', index)}"
        x_start = _number(structure.get('x_start_m'), f'{label} x_start_m')
        x_end = _number(structure.get('x_end_m'), f'{label} x_end_m')
        y_front = _number(structure.get('y_front_m'), f'{label} y_front_m')
        y_back = _number(structure.get('y_back_m'), f'{label} y_back_m')
        if x_end <= x_start or y_back < y_front:
            raise SurveyDxfError(f'{label} extents must be ordered')
        result.append(_Structure(_ascii(structure.get('id', f'S{index + 1:02d}'), f'{label} id'), x_start, x_end, y_front, y_back, _optional_number(structure.get('eave_z_m'), f'{label} eave_z_m'), _optional_number(structure.get('ridge_z_m'), f'{label} ridge_z_m'), _confidence(structure.get('confidence'), label)))
    return result

def _validate_partitions(partitions: Any) -> tuple[float, list[float], list[str]]:
    data = _mapping(partitions, 'partitions')
    if data.get('schema') != SCHEMA_PARTITIONS:
        raise SurveyDxfError('unsupported survey partitions schema')
    facade_y = _number(data.get('facade_y_m'), 'partitions facade_y_m')
    raw_x = data.get('x_positions_m')
    if not isinstance(raw_x, (list, tuple)):
        raise SurveyDxfError('partitions x_positions_m must be a list')
    xs = [_number(v, f'partitions x_positions_m[{i}]') for i, v in enumerate(raw_x)]
    if any((b < a for a, b in itertools.pairwise(xs))):
        raise SurveyDxfError('partitions x_positions_m must be ascending')
    raw_conf = data.get('confidence')
    if raw_conf is None:
        confidences = [CONFIDENCE_HIGH] * len(xs)
    else:
        if not isinstance(raw_conf, (list, tuple)) or len(raw_conf) != len(xs):
            raise SurveyDxfError('partitions confidence must match x_positions_m')
        confidences = [_confidence(v, f'partition {i}') for i, v in enumerate(raw_conf)]
    return (facade_y, xs, confidences)

def _validate_profile(profile: Any) -> list[tuple[float, float]]:
    data = _mapping(profile, 'facade_profile')
    if data.get('schema') != SCHEMA_PROFILE:
        raise SurveyDxfError('unsupported survey facade profile schema')
    xs, zs = (data.get('x_m'), data.get('z_top_m'))
    if not isinstance(xs, (list, tuple)) or not isinstance(zs, (list, tuple)) or len(xs) != len(zs):
        raise SurveyDxfError('facade_profile x_m and z_top_m must be lists of equal length')
    return [(_number(x, f'facade_profile x_m[{i}]'), _number(z, f'facade_profile z_top_m[{i}]')) for i, (x, z) in enumerate(zip(xs, zs))]

def _thin_cells(rows: np.ndarray, cols: np.ndarray, plan: _Plan) -> tuple[np.ndarray, np.ndarray]:
    x = plan.bounds_m[0] + (cols + 0.5) * plan.resolution_m
    y = plan.bounds_m[2] + (rows + 0.5) * plan.resolution_m
    if x.size == 0:
        return (x, y)
    key = np.stack([np.floor(x / _SLICE_THIN_M), np.floor(y / _SLICE_THIN_M)], axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    keep = np.sort(first)
    return (x[keep], y[keep])

def _slice_points(plan: _Plan) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(plan.occupancy > 0)
    return _thin_cells(rows, cols, plan)

def _skyline_points(plan: _Plan, facade_y_m: float) -> tuple[np.ndarray, np.ndarray]:
    if plan.roof_z is None or plan.roof_z.size == 0:
        return (np.empty(0), np.empty(0))
    roof = np.where(np.isfinite(plan.roof_z), plan.roof_z, -np.inf)
    row_y = plan.bounds_m[2] + (np.arange(roof.shape[0]) + 0.5) * plan.resolution_m
    roof[row_y < facade_y_m - _ELEVATION_BACK_MARGIN_M, :] = -np.inf
    top = roof.max(axis=0)
    cols = np.flatnonzero(np.isfinite(top))
    x = plan.bounds_m[0] + (cols + 0.5) * plan.resolution_m
    if x.size == 0:
        return (x, np.empty(0))
    _, first = np.unique(np.floor(x / _SLICE_THIN_M), return_index=True)
    keep = np.sort(first)
    return (x[keep], top[cols][keep])

def _add_layers(document: ezdxf.document.Drawing) -> None:
    for name, (color, linetype) in _LAYER_STYLE.items():
        if name not in document.layers:
            if linetype not in document.linetypes:
                linetype = 'Continuous'
            document.layers.add(name=name, color=color, linetype=linetype)

def _add_dimstyle(document: ezdxf.document.Drawing, scale: _Units) -> None:
    style = document.dimstyles.duplicate_entry('EZDXF', DIMSTYLE)
    style.dxf.dimlfac = 1.0
    style.dxf.dimdec = 0
    style.dxf.dimtxt = scale.length(_DIM_TEXT_M)
    style.dxf.dimasz = scale.length(_DIM_ARROW_M)
    style.dxf.dimexe = scale.length(_DIM_EXT_M)
    style.dxf.dimexo = scale.length(_DIM_EXT_M)
    style.dxf.dimgap = scale.length(_DIM_GAP_M)
    style.dxf.dimtad = 1
    style.dxf.dimtih = 0
    style.dxf.dimtoh = 0
    style.set_arrows(blk=ezdxf.ARROWS.architectural_tick)

def _add_text(model: Any, scale: _Units, text: str, position_m: tuple[float, float], *, layer: str, height_m: float=_TEXT_HEIGHT_M, align: TextEntityAlignment=TextEntityAlignment.BOTTOM_LEFT, rotation: float=0.0) -> None:
    entity = model.add_text(text, dxfattribs={'layer': layer, 'height': scale.length(height_m), 'rotation': rotation})
    entity.set_placement(scale.xy(*position_m), align=align)

def _add_line(model: Any, scale: _Units, a_m: tuple[float, float], b_m: tuple[float, float], layer: str) -> None:
    model.add_line(scale.xy(*a_m), scale.xy(*b_m), dxfattribs={'layer': layer})

def _add_rectangle(model: Any, scale: _Units, corner_m: tuple[float, float], opposite_m: tuple[float, float], layer: str) -> None:
    (x0, y0), (x1, y1) = (corner_m, opposite_m)
    points = [scale.xy(x0, y0), scale.xy(x1, y0), scale.xy(x1, y1), scale.xy(x0, y1)]
    model.add_lwpolyline(points, close=True, dxfattribs={'layer': layer})

def _add_dimension(model: Any, scale: _Units, p1_m: tuple[float, float], p2_m: tuple[float, float], base_m: tuple[float, float], *, low: bool, angle: float=0.0) -> None:
    dimension = model.add_linear_dim(base=scale.xy(*base_m), p1=scale.xy(*p1_m), p2=scale.xy(*p2_m), angle=angle, text=LOW_CONFIDENCE_TEXT if low else '<>', dimstyle=DIMSTYLE, dxfattribs={'layer': LAYER_DIM})
    dimension.render()

def _add_slice(model: Any, scale: _Units, plan: _Plan) -> None:
    xs, ys = _slice_points(plan)
    for x, y in zip(xs.tolist(), ys.tolist()):
        model.add_point(scale.xy(x, y), dxfattribs={'layer': LAYER_SLICE})

def _add_walls(model: Any, scale: _Units, walls: Sequence[_Wall]) -> None:
    for wall in walls:
        layer = LAYER_WALL_LOWCONF if wall.confidence == CONFIDENCE_LOW else LAYER_WALL
        if wall.axis == 'x':
            _add_line(model, scale, (wall.start_m, wall.pos_m), (wall.end_m, wall.pos_m), layer)
        else:
            _add_line(model, scale, (wall.pos_m, wall.start_m), (wall.pos_m, wall.end_m), layer)

def _add_structures(model: Any, scale: _Units, structures: Sequence[_Structure]) -> None:
    for structure in structures:
        _add_rectangle(model, scale, (structure.x_start_m, structure.y_front_m), (structure.x_end_m, structure.y_back_m), LAYER_STRUCT)
        width = scale.length(structure.x_end_m - structure.x_start_m)
        depth = scale.length(structure.depth_m)
        label = f'{structure.id} {width:.0f}x{depth:.0f} {scale.unit}'
        if structure.confidence == CONFIDENCE_LOW:
            label += f' {LOW_CONFIDENCE_TEXT}'
        _add_text(model, scale, label, ((structure.x_start_m + structure.x_end_m) / 2.0, (structure.y_front_m + structure.y_back_m) / 2.0), layer=LAYER_STRUCT, align=TextEntityAlignment.MIDDLE_CENTER)

def _add_plan_dimensions(model: Any, scale: _Units, facade_y_m: float, xs: Sequence[float], confidences: Sequence[str], structures: Sequence[_Structure]) -> None:
    chain_y = facade_y_m - _CHAIN_OFFSET_M
    for (a, b), (ca, cb) in zip(itertools.pairwise(xs), itertools.pairwise(confidences)):
        _add_dimension(model, scale, (a, facade_y_m), (b, facade_y_m), (a, chain_y), low=CONFIDENCE_LOW in (ca, cb))
    if len(xs) >= 2:
        total_y = facade_y_m - _TOTAL_OFFSET_M
        low = CONFIDENCE_LOW in (confidences[0], confidences[-1])
        _add_dimension(model, scale, (xs[0], facade_y_m), (xs[-1], facade_y_m), (xs[0], total_y), low=low)
    for structure in structures:
        x = structure.x_end_m
        _add_dimension(model, scale, (x, structure.y_front_m), (x, structure.y_back_m), (x - _DEPTH_OFFSET_M, structure.y_front_m), low=structure.confidence == CONFIDENCE_LOW, angle=90.0)

def _add_grid(model: Any, scale: _Units, plan: _Plan) -> None:
    x0, x1, y0, y1 = plan.bounds_m
    gx0, gx1 = (math.floor(x0), math.ceil(x1))
    gy0, gy1 = (math.floor(y0), math.ceil(y1))
    for x in range(gx0, gx1 + 1):
        _add_line(model, scale, (x, gy0), (x, gy1), LAYER_GRID)
        _add_text(model, scale, f'{x}', (x, gy0 - _GRID_LABEL_M), layer=LAYER_GRID, align=TextEntityAlignment.TOP_CENTER)
    for y in range(gy0, gy1 + 1):
        _add_line(model, scale, (gx0, y), (gx1, y), LAYER_GRID)
        _add_text(model, scale, f'{y}', (gx0 - _GRID_LABEL_M, y), layer=LAYER_GRID, align=TextEntityAlignment.MIDDLE_RIGHT)

def _add_title(model: Any, scale: _Units, plan: _Plan, project_name: str, facade_y_m: float) -> None:
    x0, _, _, y1 = plan.bounds_m
    title = ' // '.join(('CCA', project_name or 'SURVEY PACK', DRAFT_LABEL))
    _add_text(model, scale, title, (x0, y1 + 3.0 * _TITLE_HEIGHT_M), layer=LAYER_TEXT, height_m=_TITLE_HEIGHT_M)
    note = f'PLAN kesit {plan.band_key} | olculer {scale.unit} | kotlar m (zemin z=0) | ON CEPHE y={scale.length(facade_y_m):.0f} {scale.unit} | {LOW_CONFIDENCE_TEXT} = dusuk guven, elle olc'
    _add_text(model, scale, note, (x0, y1 + 1.5 * _TEXT_HEIGHT_M), layer=LAYER_TEXT)

def _add_front_elevation(model: Any, scale: _Units, plan: _Plan, facade_y_m: float, profile: Sequence[tuple[float, float]], structures: Sequence[_Structure]) -> float:
    x0, x1, y0, _ = plan.bounds_m
    origin_y = y0 - (plan.height_m + _ELEVATION_GAP_M)

    def at(x_m: float, z_m: float) -> tuple[float, float]:
        return (x_m, origin_y + z_m)
    _add_line(model, scale, at(x0, 0.0), at(x1, 0.0), LAYER_ELEV_FRONT)
    if len(profile) >= 2:
        model.add_lwpolyline([scale.xy(*at(x, z)) for x, z in profile], dxfattribs={'layer': LAYER_ELEV_FRONT})
    elif profile:
        model.add_point(scale.xy(*at(*profile[0])), dxfattribs={'layer': LAYER_ELEV_FRONT})
    sky_x, sky_z = _skyline_points(plan, facade_y_m)
    for x, z in zip(sky_x.tolist(), sky_z.tolist()):
        model.add_point(scale.xy(*at(x, z)), dxfattribs={'layer': LAYER_ELEV_LIDAR})
    top = max([_ELEVATION_FALLBACK_TOP_M, *(z for _, z in profile), *(s.top_m for s in structures), *sky_z.tolist()])
    for structure in structures:
        a, b = (structure.x_start_m, structure.x_end_m)
        low = structure.confidence == CONFIDENCE_LOW
        centre = (a + b) / 2.0
        if structure.eave_z_m is not None:
            _add_rectangle(model, scale, at(a, 0.0), at(b, structure.eave_z_m), LAYER_ELEV_FRONT)
            _add_dimension(model, scale, at(a, 0.0), at(a, structure.eave_z_m), at(a + _HEIGHT_OFFSET_M, 0.0), low=low, angle=90.0)
            _add_text(model, scale, f'{structure.id} SACAK +{structure.eave_z_m:.2f}', at(centre, structure.eave_z_m + _TEXT_HEIGHT_M / 2.0), layer=LAYER_TEXT, align=TextEntityAlignment.BOTTOM_CENTER)
        else:
            _add_text(model, scale, f'{structure.id} SACAK {LOW_CONFIDENCE_TEXT}', at(centre, _TEXT_HEIGHT_M), layer=LAYER_TEXT, align=TextEntityAlignment.BOTTOM_CENTER)
        ridge = structure.ridge_z_m
        if ridge is not None and (structure.eave_z_m is None or ridge - structure.eave_z_m > _RIDGE_STEP_M):
            _add_line(model, scale, at(a, ridge), at(b, ridge), LAYER_ELEV_FRONT)
            _add_text(model, scale, f'MAHYA +{ridge:.2f}', at(centre, ridge + _TEXT_HEIGHT_M / 2.0), layer=LAYER_TEXT, align=TextEntityAlignment.BOTTOM_CENTER)
        _add_dimension(model, scale, at(a, 0.0), at(b, 0.0), at(a, -_WIDTH_OFFSET_M), low=low)
    _add_text(model, scale, 'ON GORUNUS - tarayici tarafindan, x saga artar, en yakin nokta kazanir', at(x0, top + 2.0 * _LABEL_LIFT_M), layer=LAYER_TEXT)
    return origin_y

def _add_sections(model: Any, scale: _Units, plan: _Plan, structures: Sequence[_Structure]) -> None:
    if not structures:
        return
    _, x1, y0, _ = plan.bounds_m
    origin_x = x1 + _SECTION_GAP_M
    pitch = max((s.depth_m for s in structures)) + _SECTION_PITCH_GAP_M
    _add_text(model, scale, 'KESITLER (semasal): derinlik x sacak', (origin_x, y0 - 2.0 * _LABEL_LIFT_M), layer=LAYER_TEXT, align=TextEntityAlignment.TOP_LEFT)
    for index, structure in enumerate(structures):
        sx = origin_x + index * pitch
        depth = structure.depth_m
        _add_line(model, scale, (sx, y0), (sx + depth, y0), LAYER_SECTION)
        if structure.eave_z_m is not None:
            _add_rectangle(model, scale, (sx, y0), (sx + depth, y0 + structure.eave_z_m), LAYER_SECTION)
        ridge = structure.ridge_z_m
        if ridge is not None and (structure.eave_z_m is None or ridge - structure.eave_z_m > _RIDGE_STEP_M):
            _add_line(model, scale, (sx, y0 + ridge), (sx + depth, y0 + ridge), LAYER_SECTION)
        label = f'KESIT {structure.id}'
        if structure.confidence == CONFIDENCE_LOW:
            label += f' {LOW_CONFIDENCE_TEXT}'
        _add_text(model, scale, label, (sx + depth / 2.0, y0 + structure.top_m + _LABEL_LIFT_M), layer=LAYER_TEXT, align=TextEntityAlignment.BOTTOM_CENTER)

def _set_extents(document: ezdxf.document.Drawing) -> None:
    box = bbox.extents(document.modelspace(), fast=False, cache=bbox.Cache())
    if not box.has_data:
        raise SurveyDxfError('survey DXF modelspace has no bounded geometry')
    document.header['$EXTMIN'] = tuple((float(v) for v in box.extmin))
    document.header['$EXTMAX'] = tuple((float(v) for v in box.extmax))
    document.header['$LIMMIN'] = (float(box.extmin.x), float(box.extmin.y))
    document.header['$LIMMAX'] = (float(box.extmax.x), float(box.extmax.y))

def _atomic_write(document: ezdxf.document.Drawing, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f'.{destination.stem}.', suffix='.dxf', delete=False) as stream:
            temporary = Path(stream.name)
        previous = ezdxf.options.write_fixed_meta_data_for_testing
        try:
            ezdxf.options.write_fixed_meta_data_for_testing = True
            document.saveas(temporary)
        finally:
            ezdxf.options.write_fixed_meta_data_for_testing = previous
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def _verify(destination: Path) -> tuple[dict[str, int], int]:
    try:
        document = ezdxf.readfile(destination)
    except (OSError, ezdxf.DXFError) as exc:
        raise SurveyDxfError(f'written survey DXF cannot be read back: {destination}') from exc
    auditor = document.audit()
    if auditor.has_errors:
        first = auditor.errors[0]
        raise SurveyDxfError(f'survey DXF audit reported {len(auditor.errors)} error(s); first is {first.code}: {first.message}')
    counts = Counter((str(entity.dxf.get('layer', '0')) for entity in document.modelspace()))
    for name in SURVEY_LAYERS:
        counts.setdefault(name, 0)
    return (dict(sorted(counts.items())), len(auditor.errors))

def export_survey_dxf(slices: Mapping[str, Any], walls: Sequence[Mapping[str, Any]], structures: Sequence[Mapping[str, Any]], partitions: Mapping[str, Any], facade_profile: Mapping[str, Any], destination: str | Path, *, unit: str='cm', project_name: str='') -> SurveyDxfResult:
    if unit not in UNIT_INSUNITS:
        raise SurveyDxfError('unit must be cm or mm')
    output = Path(destination)
    if output.suffix.lower() != '.dxf':
        raise SurveyDxfError('DXF destination must use the .dxf extension')
    _ascii(project_name, 'project_name')
    plan = _validate_slices(slices)
    wall_items = _validate_walls(walls)
    structure_items = _validate_structures(structures)
    facade_y, xs, confidences = _validate_partitions(partitions)
    profile = _validate_profile(facade_profile)
    scale = _Units(unit, _CM_TO_UNIT[unit])
    document = ezdxf.new('R2010', setup=True)
    document.units = UNIT_INSUNITS[unit]
    document.header['$MEASUREMENT'] = 1
    _add_layers(document)
    _add_dimstyle(document, scale)
    model = document.modelspace()
    _add_grid(model, scale, plan)
    _add_slice(model, scale, plan)
    _add_walls(model, scale, wall_items)
    _add_structures(model, scale, structure_items)
    _add_plan_dimensions(model, scale, facade_y, xs, confidences, structure_items)
    _add_title(model, scale, plan, project_name, facade_y)
    _add_front_elevation(model, scale, plan, facade_y, profile, structure_items)
    _add_sections(model, scale, plan, structure_items)
    _set_extents(document)
    _atomic_write(document, output)
    counts, audit_errors = _verify(output)
    return SurveyDxfResult(str(output.resolve()), unit, counts, audit_errors)
__all__ = ['DIMSTYLE', 'DRAFT_LABEL', 'LAYER_DIM', 'LAYER_ELEV_FRONT', 'LAYER_ELEV_LIDAR', 'LAYER_GRID', 'LAYER_SECTION', 'LAYER_SLICE', 'LAYER_STRUCT', 'LAYER_TEXT', 'LAYER_WALL', 'LAYER_WALL_LOWCONF', 'LOW_CONFIDENCE_TEXT', 'SURVEY_LAYERS', 'UNIT_INSUNITS', 'SurveyDxfError', 'SurveyDxfResult', 'export_survey_dxf']
