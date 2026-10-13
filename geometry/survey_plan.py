
from __future__ import annotations
import itertools
import math
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.signal import find_peaks
SCHEMA_SLICES = 'cca.survey-slices.v1'
SCHEMA_WALLS = 'cca.survey-walls.v1'
SCHEMA_PARTITIONS = 'cca.survey-partitions.v1'
SCHEMA_STRUCTURES = 'cca.survey-structures.v1'
SCHEMA_PROFILE = 'cca.survey-facade-profile.v1'
CONFIDENCE_HIGH = 'HIGH'
CONFIDENCE_MEDIUM = 'MEDIUM'
CONFIDENCE_LOW = 'LOW'

class SurveyPlanError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class SurveyPlanConfig:
    slice_bands_m: tuple[tuple[float, float], ...] = ((0.25, 0.45), (0.55, 0.75), (0.95, 1.25), (1.45, 1.65), (1.95, 2.15), (2.45, 2.65), (2.95, 3.15))
    raster_m: float = 0.02
    wall_bands_m: tuple[tuple[float, float], ...] = ((0.6, 0.8), (0.95, 1.25), (1.5, 1.7), (2.0, 2.2))
    wall_histogram_bin_m: float = 0.02
    wall_peak_share: float = 0.02
    wall_peak_min_count: int = 30
    wall_corridor_m: float = 0.04
    wall_run_bin_m: float = 0.05
    wall_run_min_count: int = 4
    wall_gap_m: float = 0.25
    wall_min_length_m: float = 0.3
    wall_merge_tolerance_m: float = 0.05
    cross_wall_min_points: int = 900
    cross_wall_extent_bin_m: float = 0.1
    cross_wall_extent_min_count: int = 10
    cross_wall_dedupe_m: float = 0.3
    structure_min_width_m: float = 0.6
    roof_min_z_m: float = 0.5
    front_margin_m: float = 5.0
    profile_bin_m: float = 0.1

    def __post_init__(self) -> None:
        positive = (self.raster_m, self.wall_histogram_bin_m, self.wall_corridor_m, self.wall_run_bin_m, self.wall_gap_m, self.wall_min_length_m, self.wall_merge_tolerance_m, self.structure_min_width_m, self.front_margin_m, self.profile_bin_m)
        if not all((math.isfinite(float(v)) and float(v) > 0.0 for v in positive)):
            raise SurveyPlanError('survey plan numeric settings must be positive and finite')
        for band in (*self.slice_bands_m, *self.wall_bands_m):
            if len(band) != 2 or not band[0] < band[1]:
                raise SurveyPlanError('z bands must be ordered (low, high) pairs')

def _points(points_xyz: Any) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise SurveyPlanError('points must be an (N, 3) array')
    return points[:, :3]

def slice_key(band: tuple[float, float]) -> str:
    return f'z{round(band[0] * 100):03d}_{round(band[1] * 100):03d}'

def _occupancy(u: np.ndarray, v: np.ndarray, res: float, bounds: tuple[float, float, float, float]) -> np.ndarray:
    u0, u1, v0, v1 = bounds
    m = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    width = math.ceil((u1 - u0) / res) + 1
    height = math.ceil((v1 - v0) / res) + 1
    grid = np.zeros((height, width), dtype=np.uint16)
    if m.any():
        iu = ((u[m] - u0) / res).astype(int)
        iv = ((v[m] - v0) / res).astype(int)
        np.add.at(grid, (iv, iu), 1)
    return grid

def build_slices(points_xyz: Any, rgb: Any | None, config: SurveyPlanConfig | None=None) -> dict[str, Any]:
    config = config or SurveyPlanConfig()
    points = _points(points_xyz)
    x, y, z = (points[:, 0], points[:, 1], points[:, 2])
    bounds = (float(math.floor(x.min())), float(math.ceil(x.max())), float(math.floor(y.min())), float(math.ceil(y.max())))
    res = config.raster_m
    bands: dict[str, dict[str, Any]] = {}
    for band in config.slice_bands_m:
        m = (z > band[0]) & (z < band[1])
        bands[slice_key(band)] = {'z_band_m': [band[0], band[1]], 'occupancy': _occupancy(x[m], y[m], res, bounds)}
    width = math.ceil((bounds[1] - bounds[0]) / res) + 1
    height = math.ceil((bounds[3] - bounds[2]) / res) + 1
    iu = ((x - bounds[0]) / res).astype(int)
    iv = ((y - bounds[2]) / res).astype(int)
    roof = np.full((height, width), np.nan, dtype=np.float32)
    order = np.argsort(z)
    roof[iv[order], iu[order]] = z[order].astype(np.float32)
    roof_rgb = None
    if rgb is not None:
        colours = np.asarray(rgb)
        if colours.shape[0] == points.shape[0]:
            roof_rgb = np.full((height, width, 3), 255, dtype=np.uint8)
            roof_rgb[iv[order], iu[order]] = colours[order][:, :3].astype(np.uint8)
    extent = [bounds[0], bounds[0] + width * res, bounds[2], bounds[2] + height * res]
    return {'schema': SCHEMA_SLICES, 'resolution_m': res, 'bounds_m': list(bounds), 'extent_m': extent, 'bands': bands, 'roof_z': roof, 'roof_rgb': roof_rgb, 'point_count': int(points.shape[0])}

def _runs(mask: np.ndarray, gap_bins: int, min_bins: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = last = None
    for i, flag in enumerate(mask):
        if flag:
            if start is None:
                start = i
            last = i
        elif start is not None and last is not None and (i - last > gap_bins):
            if last - start + 1 >= min_bins:
                runs.append((start, last))
            start = None
    if start is not None and last is not None and (last - start + 1 >= min_bins):
        runs.append((start, last))
    return runs

def vectorize_walls(points_xyz: Any, config: SurveyPlanConfig | None=None) -> list[dict[str, Any]]:
    config = config or SurveyPlanConfig()
    points = _points(points_xyz)
    x, y, z = (points[:, 0], points[:, 1], points[:, 2])
    bin_m = config.wall_histogram_bin_m
    raw: list[dict[str, Any]] = []
    gap_bins = int(config.wall_gap_m / config.wall_run_bin_m)
    min_bins = int(config.wall_min_length_m / config.wall_run_bin_m)
    for band in config.wall_bands_m:
        m = (z > band[0]) & (z < band[1])
        if m.sum() < 50:
            continue
        for axis, along, across in (('x', x[m], y[m]), ('y', y[m], x[m])):
            hist, edges = np.histogram(across, bins=np.arange(across.min(), across.max() + bin_m, bin_m))
            peaks, _ = find_peaks(hist, height=max(hist.max() * config.wall_peak_share, config.wall_peak_min_count), distance=3)
            for k in peaks:
                centre = (edges[k] + edges[k + 1]) / 2.0
                selected = np.abs(across - centre) < config.wall_corridor_m
                a = along[selected]
                if a.shape[0] < 20:
                    continue
                hb, eb = np.histogram(a, bins=np.arange(a.min(), a.max() + config.wall_run_bin_m, config.wall_run_bin_m))
                for i0, i1 in _runs(hb >= config.wall_run_min_count, gap_bins, min_bins):
                    s, t = (float(eb[i0]), float(eb[i1 + 1]))
                    in_run = selected.copy()
                    in_run[selected] = (a >= s) & (a <= t)
                    raw.append({'axis': axis, 'pos_m': float(across[in_run].mean()), 'start_m': s, 'end_m': t, 'n_points': int(in_run.sum()), 'band': (band[0], band[1])})
    raw.sort(key=lambda s: (s['axis'], s['pos_m'], s['start_m']))
    merged: list[dict[str, Any]] = []
    tol = config.wall_merge_tolerance_m
    for seg in raw:
        target = None
        for item in merged:
            if item['axis'] == seg['axis'] and abs(item['pos_m'] - seg['pos_m']) < tol and (seg['start_m'] < item['end_m'] + 0.3) and (seg['end_m'] > item['start_m'] - 0.3):
                target = item
                break
        if target is None:
            merged.append({**seg, 'pos_list': [seg['pos_m']], 'bands': {seg['band']}, 'n_list': [seg['n_points']]})
            continue
        target['pos_list'].append(seg['pos_m'])
        target['bands'].add(seg['band'])
        target['n_list'].append(seg['n_points'])
        target['start_m'] = min(target['start_m'], seg['start_m'])
        target['end_m'] = max(target['end_m'], seg['end_m'])
        target['pos_m'] = float(np.average(target['pos_list'], weights=target['n_list']))
    walls: list[dict[str, Any]] = []
    for index, item in enumerate(merged):
        band_count = len(item['bands'])
        std = float(np.std(item['pos_list'])) if len(item['pos_list']) > 1 else 0.0
        if band_count >= 3 and std < 0.02:
            confidence = CONFIDENCE_HIGH
        elif band_count >= 2:
            confidence = CONFIDENCE_MEDIUM
        else:
            confidence = CONFIDENCE_LOW
        if item['axis'] == 'x':
            corridor = (np.abs(y - item['pos_m']) < 0.05) & (x >= item['start_m']) & (x <= item['end_m'])
        else:
            corridor = (np.abs(x - item['pos_m']) < 0.05) & (y >= item['start_m']) & (y <= item['end_m'])
        zz = z[corridor & (z > 0.1)]
        walls.append({'id': f'W{index + 1:03d}', 'axis': item['axis'], 'pos_m': round(item['pos_m'], 3), 'start_m': round(item['start_m'], 3), 'end_m': round(item['end_m'], 3), 'length_m': round(item['end_m'] - item['start_m'], 3), 'thickness_m': None, 'pair_of': None, 'z_min_m': round(float(np.percentile(zz, 1)), 2) if zz.size else None, 'z_max_m': round(float(np.percentile(zz, 99)), 2) if zz.size else None, 'n_points': int(sum(item['n_list'])), 'slice_heights': sorted([list(b) for b in item['bands']]), 'slice_pos_std_m': round(std, 4), 'confidence': confidence})
    _pair_thickness(walls)
    return walls

def _pair_thickness(walls: list[dict[str, Any]], low: float=0.06, high: float=0.3) -> None:
    for i, a in enumerate(walls):
        if a['pair_of'] is not None:
            continue
        best = None
        for j, b in enumerate(walls):
            if j == i or b['pair_of'] is not None or b['axis'] != a['axis']:
                continue
            gap = abs(a['pos_m'] - b['pos_m'])
            overlap = min(a['end_m'], b['end_m']) - max(a['start_m'], b['start_m'])
            fits = low <= gap <= high and overlap > 0.5 * min(a['length_m'], b['length_m'])
            if fits and (best is None or gap < best[0]):
                best = (gap, j)
        if best is not None:
            gap, j = best
            a['pair_of'] = walls[j]['id']
            walls[j]['pair_of'] = a['id']
            a['thickness_m'] = walls[j]['thickness_m'] = round(gap, 3)

def find_facade(walls: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [w for w in walls if w['axis'] == 'x']
    if not candidates:
        raise SurveyPlanError('no wall parallel to X was found; cannot define a facade')
    longest = max(candidates, key=lambda w: w['length_m'])
    pieces = [w for w in candidates if abs(w['pos_m'] - longest['pos_m']) < 0.06]
    weights = np.array([w['n_points'] for w in pieces], dtype=np.float64)
    y_m = float(np.average([w['pos_m'] for w in pieces], weights=weights))
    return {'y_m': round(y_m, 3), 'x_start_m': round(min((w['start_m'] for w in pieces)), 3), 'x_end_m': round(max((w['end_m'] for w in pieces)), 3), 'n_points': int(weights.sum()), 'piece_ids': [w['id'] for w in pieces], 'confidence': longest['confidence']}

def front_view_min_y(walls: list[dict[str, Any]], facade: dict[str, Any], config: SurveyPlanConfig | None=None) -> float:
    config = config or SurveyPlanConfig()
    limit = facade['y_m'] - config.front_margin_m
    opposite = [w for w in walls if w['axis'] == 'x' and w['pos_m'] < facade['y_m'] - 1.0 and (w['confidence'] != CONFIDENCE_LOW)]
    if opposite:
        nearest = max(opposite, key=lambda w: w['pos_m'])
        limit = max(limit, (nearest['pos_m'] + facade['y_m']) / 2.0)
    return round(float(limit), 3)

def cross_walls(points_xyz: Any, walls: list[dict[str, Any]], facade: dict[str, Any], config: SurveyPlanConfig | None=None, *, min_y: float | None=None) -> list[dict[str, Any]]:
    config = config or SurveyPlanConfig()
    points = _points(points_xyz)
    x, y, z = (points[:, 0], points[:, 1], points[:, 2])
    y0 = facade['y_m']
    floor_y = front_view_min_y(walls, facade, config) if min_y is None else min_y

    def extent(xp: float) -> tuple[float, float] | None:
        m = (np.abs(x - xp) < 0.05) & (z > 0.3) & (z < 3.5) & (y > floor_y)
        if m.sum() < 50:
            return None
        bins = np.arange(floor_y, y.max() + config.cross_wall_extent_bin_m, config.cross_wall_extent_bin_m)
        hist, edges = np.histogram(y[m], bins=bins)
        dense = np.where(hist >= config.cross_wall_extent_min_count)[0]
        if dense.size == 0:
            return None
        return (float(edges[dense[0]]), float(edges[dense[-1] + 1]))
    result: list[dict[str, Any]] = []
    seen: list[float] = []
    ordered = sorted([w for w in walls if w['axis'] == 'y' and w['n_points'] >= config.cross_wall_min_points and (w['confidence'] != CONFIDENCE_LOW)], key=lambda w: -w['n_points'])
    for w in ordered:
        if any((abs(w['pos_m'] - s) < config.cross_wall_dedupe_m for s in seen)):
            continue
        ext = extent(w['pos_m'])
        if ext is None or ext[0] > y0 + 0.6 or ext[1] < y0 + 0.8:
            continue
        seen.append(w['pos_m'])
        result.append({'id': w['id'], 'x_m': w['pos_m'], 'y_start_m': round(ext[0], 3), 'y_end_m': round(ext[1], 3), 'depth_m': round(ext[1] - y0, 3), 'z_max_m': w['z_max_m'], 'n_points': w['n_points'], 'confidence': w['confidence']})
    result.sort(key=lambda w: w['x_m'])
    return result

def partitions_along_facade(cross: list[dict[str, Any]], facade: dict[str, Any]) -> dict[str, Any]:
    xs = [w['x_m'] for w in cross]
    spacings = [round(b - a, 3) for a, b in itertools.pairwise(xs)]
    return {'schema': SCHEMA_PARTITIONS, 'facade_y_m': facade['y_m'], 'x_positions_m': xs, 'spacings_m': spacings, 'total_m': round(xs[-1] - xs[0], 3) if len(xs) > 1 else 0.0, 'confidence': [w['confidence'] for w in cross]}

def segment_structures(points_xyz: Any, cross: list[dict[str, Any]], facade: dict[str, Any], config: SurveyPlanConfig | None=None) -> list[dict[str, Any]]:
    config = config or SurveyPlanConfig()
    points = _points(points_xyz)
    x, y, z = (points[:, 0], points[:, 1], points[:, 2])
    y0 = facade['y_m']
    structures: list[dict[str, Any]] = []
    for index, (a, b) in enumerate(itertools.pairwise(cross)):
        width = b['x_m'] - a['x_m']
        if width < config.structure_min_width_m:
            continue
        inside = (x > a['x_m'] + 0.1) & (x < b['x_m'] - 0.1)
        front = inside & (np.abs(y - y0) < 0.35) & (z > config.roof_min_z_m)
        behind = inside & (y > y0) & (z > config.roof_min_z_m)
        ground = inside & (y > y0 - 2.0) & (y < y0) & (np.abs(z) < 0.25)
        eave = float(np.percentile(z[front], 99)) if front.sum() > 200 else None
        ridge = float(np.percentile(z[behind], 99)) if behind.sum() > 200 else None
        back = float(np.percentile(y[behind], 98)) if behind.sum() > 200 else None
        ground_z = float(np.median(z[ground])) if ground.sum() > 50 else None
        depth = max(a['depth_m'], b['depth_m'])
        if back is not None:
            depth = max(depth, back - y0)
        if eave is None or ridge is None:
            roof_type = 'belirsiz'
        elif ridge - eave > 0.15:
            roof_type = 'egimli'
        else:
            roof_type = 'duz'
        confidences = [a['confidence'], b['confidence']]
        confidence = CONFIDENCE_LOW if CONFIDENCE_LOW in confidences else CONFIDENCE_MEDIUM if CONFIDENCE_MEDIUM in confidences else CONFIDENCE_HIGH
        structures.append({'id': f'S{index + 1:02d}', 'x_start_m': a['x_m'], 'x_end_m': b['x_m'], 'width_m': round(width, 3), 'y_front_m': y0, 'y_back_m': round(y0 + depth, 3), 'depth_m': round(depth, 3), 'ground_z_m': None if ground_z is None else round(ground_z, 3), 'eave_z_m': None if eave is None else round(eave, 2), 'ridge_z_m': None if ridge is None else round(ridge, 2), 'roof_type': roof_type, 'n_points': int(inside.sum()), 'confidence': confidence, 'bounding_walls': [a['id'], b['id']], 'notes': ''})
    return structures

def facade_profile(points_xyz: Any, facade: dict[str, Any], config: SurveyPlanConfig | None=None) -> dict[str, Any]:
    config = config or SurveyPlanConfig()
    points = _points(points_xyz)
    x, y, z = (points[:, 0], points[:, 1], points[:, 2])
    y0 = facade['y_m']
    m = (np.abs(y - y0) < 0.25) & (z > 0.3)
    bins = np.arange(facade['x_start_m'], facade['x_end_m'] + config.profile_bin_m, config.profile_bin_m)
    idx = np.digitize(x[m], bins) - 1
    xs: list[float] = []
    zs: list[float] = []
    zm = z[m]
    for i in range(len(bins) - 1):
        sel = idx == i
        if sel.sum() >= 20:
            xs.append(round(float((bins[i] + bins[i + 1]) / 2.0), 3))
            zs.append(round(float(np.percentile(zm[sel], 99)), 3))
    return {'schema': SCHEMA_PROFILE, 'facade_y_m': y0, 'bin_m': config.profile_bin_m, 'x_m': xs, 'z_top_m': zs}
__all__ = ['CONFIDENCE_HIGH', 'CONFIDENCE_LOW', 'CONFIDENCE_MEDIUM', 'SCHEMA_PARTITIONS', 'SCHEMA_PROFILE', 'SCHEMA_SLICES', 'SCHEMA_STRUCTURES', 'SCHEMA_WALLS', 'SurveyPlanConfig', 'SurveyPlanError', 'build_slices', 'cross_walls', 'facade_profile', 'find_facade', 'front_view_min_y', 'partitions_along_facade', 'segment_structures', 'slice_key', 'vectorize_walls']
