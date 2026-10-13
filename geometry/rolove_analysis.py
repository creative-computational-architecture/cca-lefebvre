
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from scipy import ndimage
from geometry.rolove import ProjectionSurface
from geometry.survey_plan import _runs

@dataclass(frozen=True)
class PlanAnalysisConfig:
    cut_height_m: float = 1.1
    storey_height_m: float = 3.5
    resolution_m: float = 0.08
    plane_tolerance_m: float = 0.045
    max_planes: int = 32
    sample_points: int = 100000

    def __post_init__(self):
        values = (self.cut_height_m, self.storey_height_m, self.resolution_m, self.plane_tolerance_m)
        if not np.isfinite(values).all():
            raise ValueError('Analiz ayarlari sonlu olmali')
        if not 0.2 <= self.cut_height_m < self.storey_height_m:
            raise ValueError('Kesit yuksekligi 0.2 m veya ustu ve kat yuksekliginden kucuk olmali')
        if not 0.02 <= self.resolution_m <= 0.25 or not 0.005 <= self.plane_tolerance_m <= 0.15:
            raise ValueError('Analiz hucre boyu 2..25 cm, yuzey toleransi 0.5..15 cm olmali')
        if self.sample_points < 1000 or not 1 <= self.max_planes <= 100:
            raise ValueError('Yuzey arama butcesi gecersiz')

def _points(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or (not np.isfinite(points).all()):
        raise ValueError('Analiz icin en az uc sonlu XYZ noktasi gerekli')
    return points

def _basis(normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    if abs(normal[2]) > 0.9:
        if normal[2] < 0:
            normal = -normal
        u = np.array([1.0, 0.0, 0.0]) - normal * normal[0]
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
    else:
        v = np.array([0.0, 0.0, 1.0]) - normal * normal[2]
        v /= np.linalg.norm(v)
        u = np.cross(v, normal)
    return (u, v)

def detect_planes(points, config=None):
    import open3d as o3d
    config = config or PlanAnalysisConfig()
    points = _points(points)
    rng = np.random.default_rng(7)
    indices = np.sort(rng.choice(len(points), config.sample_points, replace=False)) if len(points) > config.sample_points else np.arange(len(points))
    sample = points[indices]
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample))
    cloud = cloud.voxel_down_sample(config.resolution_m * 0.75)
    work = np.asarray(cloud.points).copy()
    initial = len(work)
    planes = []
    attempts = 0
    o3d.utility.random.seed(7)
    for _ in range(config.max_planes):
        if len(work) < 60:
            break
        attempts += 1
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(work))
        _model, inliers = cloud.segment_plane(distance_threshold=config.plane_tolerance_m, ransac_n=3, num_iterations=250, probability=0.995)
        if len(inliers) < 45:
            break
        patch = work[inliers]
        centroid = patch.mean(axis=0)
        _, _, vt = np.linalg.svd(patch - centroid, full_matrices=False)
        normal = vt[-1]
        major = int(np.argmax(np.abs(normal)))
        if normal[major] < 0:
            normal = -normal
        u, v = _basis(normal)
        normal = np.cross(u, v)
        uv = (patch - centroid) @ np.column_stack((u, v))
        bounds = np.percentile(uv, [1, 99], axis=0)
        if np.min(bounds[1] - bounds[0]) >= 0.12:
            planes.append({'origin': centroid.tolist(), 'normal': normal.tolist(), 'u': u.tolist(), 'v': v.tolist(), 'bounds_m': bounds.tolist(), 'sample_support': len(inliers), 'residual_rms_m': float(np.sqrt(np.mean(((patch - centroid) @ normal) ** 2)))})
        keep = np.ones(len(work), dtype=bool)
        keep[inliers] = False
        work = work[keep]
    return {'planes': planes, 'source_count': len(points), 'workset_count': initial, 'unassigned_workset_count': len(work), 'search_budget_reached': attempts == config.max_planes and len(work) >= 60, 'method': 'RANSAC_SVD_CANDIDATES', 'config': asdict(config)}

def propose_surfaces(points, config=None):
    config = config or PlanAnalysisConfig()
    points = _points(points)
    result = detect_planes(points, config)
    horizontal = [p for p in result['planes'] if p['normal'][2] > 0.95]
    floors = []
    for plane in horizontal:
        depth = (points - plane['origin']) @ np.asarray(plane['normal'])
        if np.mean(depth >= -config.plane_tolerance_m * 2) >= 0.92:
            floors.append(plane)
    surfaces = []
    if floors:
        floor = max(floors, key=lambda p: p['sample_support'])
        origin = np.asarray(floor['origin']) + np.asarray(floor['normal']) * config.cut_height_m
        surfaces.append(ProjectionSurface('Oto-Plan', tuple(origin), tuple(floor['u']), tuple(floor['v']), depth_min=-0.15, depth_max=0.15))
    for plane in result['planes']:
        if abs(plane['normal'][2]) > 0.25:
            continue
        span = np.subtract(*plane['bounds_m'][::-1])
        if span[0] < 1 or span[1] < 1:
            continue
        if any((abs(np.dot(surface.normal, plane['normal'])) > 0.998 and abs(np.dot(np.subtract(surface.origin, plane['origin']), surface.normal)) < 0.12 for surface in surfaces)):
            continue
        surfaces.append(ProjectionSurface(f'Oto-Gorunus-{len(surfaces):02d}', tuple(plane['origin']), tuple(plane['u']), tuple(plane['v'])))
    result['surfaces'] = [s.to_dict() for s in surfaces]
    result['ground_found'] = bool(floors)
    return result

def _components(cells):
    todo = {tuple(cell) for cell in cells}
    groups = []
    while todo:
        seed = min(todo)
        todo.remove(seed)
        stack, group = ([seed], [seed])
        while stack:
            x, y = stack.pop()
            for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                neighbour = (x + dx, y + dy)
                if neighbour in todo:
                    todo.remove(neighbour)
                    stack.append(neighbour)
                    group.append(neighbour)
        groups.append(np.asarray(group))
    return groups

def _element(kind, lines, support, evidence, *, confidence='LOW'):
    return {'kind': kind, 'lines_m': [np.asarray(line).tolist() for line in lines], 'support': support, 'classification_confidence': confidence, 'review_status': 'ADAY', 'evidence': evidence, 'thickness_m': None, 'measurement_status': 'UNCALIBRATED'}

def _wall_evidence(points, planes, config):
    elements = []
    res = config.resolution_m
    bands = ((0.3, 0.55), (0.7, 0.9), (config.cut_height_m - 0.12, config.cut_height_m + 0.12), (min(2.0, config.storey_height_m - 0.5), min(2.25, config.storey_height_m - 0.2)))
    cut = 2
    seen = []
    for plane in planes:
        normal = np.asarray(plane['normal'])
        if abs(normal[2]) > 0.15:
            continue
        normal[2] = 0
        normal /= np.linalg.norm(normal)
        tangent = np.cross([0, 0, 1], normal)
        origin = np.asarray(plane['origin'])
        origin[2] = 0
        if any((abs(n @ normal) > 0.998 and abs((o - origin) @ normal) < 0.09 for o, n in seen)):
            continue
        seen.append((origin, normal))
        selected = np.abs((points - origin) @ normal) <= config.plane_tolerance_m * 1.5
        patch = points[selected]
        if len(patch) < 60:
            continue
        along = (patch - origin) @ tangent
        low = np.floor(along.min() / res) * res
        bins = np.floor((along - low) / res).astype(int)
        width = int(bins.max() + 1)
        if width > 200000:
            continue
        support = np.stack([np.bincount(bins[(patch[:, 2] >= a) & (patch[:, 2] <= b)], minlength=width) for a, b in bands])
        occupied = support >= 2
        for start, end in _runs(occupied.any(axis=0), 1, max(3, int(0.5 / res))):
            counts = occupied[:, start:end + 1].mean(axis=1)
            last_end, band_count, offsets = (-np.inf, 0, [])
            for k in sorted(range(len(bands)), key=lambda k: bands[k][1]):
                a, b = bands[k]
                if counts[k] <= 0.4 or a < last_end:
                    continue
                last_end = b
                band_count += 1
                inside = (bins >= start) & (bins <= end) & (patch[:, 2] >= a) & (patch[:, 2] <= b)
                offsets.append(float(np.median((patch[inside] - origin) @ normal)))
            spread = float(np.std(offsets)) if offsets else 0.0
            quality = 'HIGH' if band_count >= 3 and spread < 0.02 else 'MEDIUM' if band_count >= 2 else 'LOW'
            lines = []
            for a, b in _runs(occupied[cut, start:end + 1], 1, 2):
                s, t = (low + (start + a) * res, low + (start + b + 1) * res)
                lines.append([(origin + tangent * s)[:2], (origin + tangent * t)[:2]])
            if not lines:
                lines = [[(origin + tangent * (low + start * res))[:2], (origin + tangent * (low + (end + 1) * res))[:2]]]
                quality = 'LOW'
            elements.append(_element('DUVAR', lines, quality, {'height_bands_m': bands, 'band_coverage': counts.tolist(), 'independent_band_count': band_count, 'band_position_std_m': spread, 'plane_residual_rms_m': plane['residual_rms_m'], 'cut_supported': bool(occupied[cut, start:end + 1].any()), 'source_point_count': int(((bins >= start) & (bins <= end)).sum()), 'note': 'Gorulen duvar yuzu; kalinlik ve arka yuz bilinmiyor'}, confidence='MEDIUM' if band_count >= 2 else 'LOW'))
        elements.extend(_openings(patch, along, origin, tangent, config))
    return elements

def _openings(patch, along, origin, tangent, config):
    res = config.resolution_m
    low = np.floor(along.min() / res) * res
    ix = np.floor((along - low) / res).astype(int)
    iz = np.floor(patch[:, 2] / res).astype(int)
    height = int(np.ceil(config.storey_height_m / res)) + 1
    width = int(ix.max()) + 1
    if height * width > 4000000:
        return []
    grid = np.zeros((height, width), dtype=np.int32)
    keep = (iz >= 0) & (iz < height)
    np.add.at(grid, (iz[keep], ix[keep]), 1)
    occupied = grid > 0
    joined = ndimage.binary_closing(occupied, structure=np.ones((3, 3)))
    holes = ndimage.binary_fill_holes(joined) & ~joined
    labels, _ = ndimage.label(holes)
    candidates = []
    for obj in ndimage.find_objects(labels):
        if obj is None:
            continue
        zz, xx = obj
        z0, z1, x0, x1 = (zz.start, zz.stop, xx.start, xx.stop)
        if (x1 - x0) * res < 0.45 or (z1 - z0) * res < 0.45 or z0 * res < 0.35:
            continue
        candidates.append(('PENCERE', x0, x1, z0, z1))
    low_band = occupied[max(1, int(0.2 / res)):max(2, int(1.7 / res))].mean(axis=0)
    for a, b in _runs(low_band < 0.08, 0, max(3, int(0.6 / res))):
        above = np.flatnonzero(occupied[int(1.8 / res):, a:b + 1].mean(axis=1) > 0.55)
        if len(above) and (b - a + 1) * res <= 4:
            candidates.append(('KAPI', a, b + 1, max(1, int(0.2 / res)), int(1.8 / res) + int(above[0])))
    elements = []
    for kind, x0, x1, z0, z1 in candidates:
        if x0 < 2 or x1 + 2 > width or z1 + 2 > height:
            continue
        left = occupied[z0:z1, x0 - 2:x0].any(axis=1).mean()
        right = occupied[z0:z1, x1:x1 + 2].any(axis=1).mean()
        header = occupied[z1:z1 + 2, x0:x1].any(axis=0).mean()
        sill = occupied[max(0, z0 - 2):z0, x0:x1].any(axis=0).mean()
        void = float((~occupied[z0:z1, x0:x1]).mean())
        if min(left, right, header) < 0.55 or void < 0.8 or (kind == 'PENCERE' and sill < 0.55):
            continue
        a = (origin + tangent * (low + x0 * res))[:2]
        b = (origin + tangent * (low + x1 * res))[:2]
        elements.append(_element(kind, [[a, b]], 'MEDIUM', {'width_m': float((x1 - x0) * res), 'sill_height_m': float(z0 * res), 'head_height_m': float(z1 * res), 'jamb_coverage': [float(left), float(right)], 'header_coverage': float(header), 'sill_coverage': float(sill), 'void_fraction': void, 'note': 'Aciklik adayi; cam/kanat ve tarama golgesi ayrimi elle kontrol'}))
    return elements

def _columns(points, config):
    res = config.resolution_m
    cuts = tuple(np.linspace(0.3, min(2.2, config.storey_height_m - 0.2), 3))
    if min(np.diff(cuts)) < 0.15:
        return []
    maps = []
    for level in cuts:
        xy = points[np.abs(points[:, 2] - level) <= 0.12, :2]
        cells, counts = np.unique(np.floor(xy / res).astype(int), axis=0, return_counts=True)
        maps.append({tuple(cell) for cell in cells[counts >= 2]})
    persistent = set.intersection(*maps)
    elements = []
    for group in _components(persistent):
        a, b = (group.min(axis=0) * res, (group.max(axis=0) + 1) * res)
        size = b - a
        if size.min() < 0.12 or size.max() > 1.5 or size.max() / size.min() > 4:
            continue
        sel = np.all((points[:, :2] >= a - res) & (points[:, :2] <= b + res), axis=1)
        patch = points[sel]
        if len(patch) < 30:
            continue
        lo, hi = np.percentile(patch[:, :2], [2, 98], axis=0)
        corners = np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]], lo])
        lines, sides, coverage = ([], [], [])
        for axis, pos in ((1, lo[1]), (0, hi[0]), (1, hi[1]), (0, lo[0])):
            near = patch[np.abs(patch[:, axis] - pos) <= res]
            sides.append(len(near))
            along = near[:, 1 - axis]
            bins = np.unique(np.floor((along - lo[1 - axis]) / res))
            coverage.append(min(1.0, len(bins) * res / max(hi[1 - axis] - lo[1 - axis], res)))
        for k, count in enumerate(sides):
            if count >= 12 and coverage[k] >= 0.6:
                lines.append(corners[k:k + 2])
        if len(lines) < 2:
            continue
        elements.append(_element('KOLON', lines, 'HIGH', {'observed_sides': len(lines), 'side_point_counts': sides, 'side_coverage': coverage, 'height_levels_m': list(cuts), 'source_point_count': len(patch), 'footprint_extent_m': (hi - lo).tolist(), 'note': 'Dusuk/orta/ust kotta kalici prizma; tasiyicilik dogrulanmadi'}, confidence='MEDIUM'))
    return elements

def _stairs(points, config):
    import open3d as o3d
    from scipy.signal import find_peaks
    res = config.resolution_m
    treads = []
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(res * 0.5)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.14, max_nn=24))
    horizontal = np.asarray(cloud.points)[np.abs(np.asarray(cloud.normals)[:, 2]) > 0.975]
    if len(horizontal) < 30:
        return []
    height_bin = 0.025
    edges = np.arange(0, config.storey_height_m + height_bin, height_bin)
    histogram, edges = np.histogram(horizontal[:, 2], bins=edges)
    peaks, _ = find_peaks(histogram, height=12, distance=3, prominence=8)
    for index in peaks:
        z = float((edges[index] + edges[index + 1]) / 2)
        if z < 0.08 or z > config.storey_height_m - 0.2:
            continue
        patch = horizontal[np.abs(horizontal[:, 2] - z) < height_bin * 1.5]
        cells = np.unique(np.floor(patch[:, :2] / res).astype(int), axis=0)
        for group in _components(cells):
            if len(group) < 8:
                continue
            lo, hi = (group.min(axis=0) * res, (group.max(axis=0) + 1) * res)
            extent = hi - lo
            if not 0.12 <= extent.min() <= 0.65 or not 0.5 <= extent.max() <= 3.5:
                continue
            axis = int(np.argmin(extent))
            local = patch[np.all((patch[:, :2] >= lo) & (patch[:, :2] <= hi), axis=1)]
            treads.append({'z': float(np.median(local[:, 2])), 'lo': lo, 'hi': hi, 'center': (lo + hi) / 2, 'axis': axis})
    treads.sort(key=lambda t: (t['z'], *t['center']))
    elements, used = ([], set())
    for i, first in enumerate(treads):
        if i in used:
            continue
        chain, indices = ([first], [i])
        direction = None
        for j in range(i + 1, len(treads)):
            a, b = (chain[-1], treads[j])
            axis = a['axis']
            delta = b['center'] - a['center']
            rise = b['z'] - a['z']
            if j in used or b['axis'] != axis or (not 0.08 <= rise <= 0.25) or (not 0.12 <= abs(delta[axis]) <= 0.5) or (abs(delta[1 - axis]) > 0.2):
                continue
            sign = np.sign(delta[axis])
            if direction is not None and sign != direction:
                continue
            direction = sign
            chain.append(b)
            indices.append(j)
        if len(chain) < 3:
            continue
        used.update(indices)
        lines = []
        for tread in chain:
            a, b = (tread['lo'].copy(), tread['hi'].copy())
            axis = tread['axis']
            a[axis] = b[axis] = tread['center'][axis]
            lines.append([a, b])
        elements.append(_element('MERDIVEN', lines, 'MEDIUM', {'tread_levels_m': [t['z'] for t in chain], 'observed_treads': len(chain), 'rise_m': np.diff([t['z'] for t in chain]).tolist(), 'up_line_m': [chain[0]['center'].tolist(), chain[-1]['center'].tolist()], 'note': 'Gorulen basamaklar; doner merdiven ve eksik kol tamamlanmaz'}))
    return elements

def analyze_floor_plan(points, surface, config=None):
    config = config or PlanAnalysisConfig()
    points = _points(points)
    if surface.normal[2] < 0.85:
        raise ValueError('Kat plani icin yukari bakan yatay plan yuzeyi secin; once XYZ hizalayin')
    basis = np.column_stack((surface.u, surface.v, surface.normal))
    local = (points - surface.origin) @ basis
    local[:, 2] += config.cut_height_m
    keep = (local[:, 2] >= 0) & (local[:, 2] <= config.storey_height_m)
    floor = local[keep]
    if len(floor) < 60:
        raise ValueError('Secilen kat hacminde yeterli yuzey destegi yok; kotlari kontrol edin')
    proposals = detect_planes(floor, config)
    elements = _wall_evidence(floor, proposals['planes'], config)
    elements.extend(_columns(floor, config))
    elements.extend(_stairs(floor, config))
    containers = [e for e in elements if e['kind'] in ('KOLON', 'MERDIVEN')]
    retained = []
    for element in elements:
        merged = False
        if element['kind'] == 'DUVAR':
            vertices = np.vstack(element['lines_m'])
            for parent in containers:
                if parent['kind'] == 'MERDIVEN' and element['evidence']['cut_supported']:
                    continue
                bounds = np.vstack(parent['lines_m'])
                lo, hi = (bounds.min(axis=0) - config.resolution_m, bounds.max(axis=0) + config.resolution_m)
                if np.all((vertices >= lo) & (vertices <= hi)):
                    parent['evidence'].setdefault('supporting_face_hypotheses', []).append(element)
                    merged = True
                    break
        if not merged:
            retained.append(element)
    elements = retained
    for index, element in enumerate(elements, 1):
        element['id'] = f'E{index:04d}'
    cut_indices = np.flatnonzero(np.abs(local[:, 2] - config.cut_height_m) <= 0.15)
    xy = local[cut_indices, :2]
    if len(xy):
        _, first = np.unique(np.floor(xy / config.resolution_m).astype(int), axis=0, return_index=True)
        underlay = xy[first]
        evidence_indices = cut_indices[first]
    else:
        underlay, evidence_indices = (np.empty((0, 2)), np.array([], dtype=int))
    return {'schema': 'nokta.architectural-plan.v1', 'elements': elements, 'surface': surface.to_dict(), 'config': asdict(config), 'internal_unit': 'm', 'source_count': len(points), 'floor_point_count': len(floor), 'plane_search': proposals, 'underlay_m': underlay.tolist(), 'underlay_source_indices': evidence_indices.tolist(), 'limitations': ['Geometrik aday siniflari; egitilmis semantik model ve kalibre olcu degil', 'Gorunmeyen odalar, kalinliklar, kapi kanatlari tamamlanmadi', 'Pencere/kapi boslugu tarama golgesi olabilir; ELLE OLC', 'Merdiven aramasi duz ve plan eksenlerine yakin kollari destekler']}
