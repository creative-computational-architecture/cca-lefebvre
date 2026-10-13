
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .surface_structure import patch_geometry

@dataclass(frozen=True)
class StructureConfig:
    pixel_convention: str = 'corner'
    frame_budget: int = 96
    pixel_step: int = 8
    query_stride: int = 2
    patch_radius_px: int = 2
    min_support: float = 0.45
    max_thickness_m: float = 0.03
    agreement_m: float = 0.08
    minimum_change_m: float = 0.08
    min_baseline_m: float = 0.12
    min_views: int = 3
    max_shape_error: float = 0.35
    anchor_tolerance_m: float = 0.03

@dataclass(frozen=True)
class PatchCatalog:
    points: np.ndarray
    origins: np.ndarray
    frames: np.ndarray
    pixels: np.ndarray
    depths: np.ndarray
    sensor_fill: np.ndarray | None = None
    patch_fill: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.points)
        if self.points.shape != (n, 25, 3) or self.origins.shape != (n, 3) or self.frames.shape != (n,) or (self.pixels.shape != (n, 2)) or (self.depths.shape != (n,)):
            raise ValueError('invalid patch catalog shape')
        if not np.isfinite(self.points).all() or not np.isfinite(self.origins).all():
            raise ValueError('patch catalog must be finite')
        if not np.isfinite(self.depths).all() or np.any(self.depths <= 0):
            raise ValueError('patch depths must be positive metres')
        for flags in (self.sensor_fill, self.patch_fill):
            if flags is not None and np.asarray(flags).shape != (n,):
                raise ValueError('fill flags must carry one value per patch')

def shape_relations(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, float)
    eig = patch_geometry(p).eigenvalues
    a, b = np.triu_indices(p.shape[1], 1)
    q = np.empty((len(p), 5))
    for start in range(0, len(p), 2048):
        block = p[start:start + 2048]
        distances = np.linalg.norm(block[:, a] - block[:, b], axis=2)
        q[start:start + len(block)] = np.quantile(distances, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T
    return np.c_[eig / np.maximum(eig.sum(axis=1, keepdims=True), 1e-15), q / np.maximum(q[:, -1:], 1e-15)]

def finite_patch_hits(origin, direction, centers, normals, corners):
    origin, direction = (np.asarray(origin, float), np.asarray(direction, float))
    denominator = normals @ direction
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.einsum('ij,ij->i', centers - origin, normals) / denominator
        hit = origin + depth[:, None] * direction
        edges = np.roll(corners, -1, axis=1) - corners
        signs = np.einsum('bki,bi->bk', np.cross(edges, hit[:, None] - corners), normals)
    inside = np.all(signs >= -1e-10, axis=1) | np.all(signs <= 1e-10, axis=1)
    valid = inside & (np.abs(denominator) > 1e-06) & np.isfinite(depth) & (depth > 0)
    return np.where(valid, depth, np.nan)

def independent_hits(indices, depths, catalog, config):
    chosen = []
    for i in sorted(indices, key=lambda j: (int(catalog.frames[j]), abs(depths[j]))):
        if any((catalog.frames[i] == catalog.frames[j] or np.linalg.norm(catalog.origins[i] - catalog.origins[j]) < config.min_baseline_m for j in chosen)):
            continue
        chosen.append(int(i))
    return chosen

def search_catalog(catalog: PatchCatalog, *, config=None, progress=None, query_indices=None):
    c = config or StructureConfig()
    if c.min_views < 3 or c.agreement_m <= 0 or c.min_baseline_m <= 0:
        raise ValueError('structure search requires independent multi-view checks')
    if not np.isfinite(c.anchor_tolerance_m) or c.anchor_tolerance_m <= 0:
        raise ValueError('alignment anchor tolerance must be positive finite metres')
    anchor_tolerance = min(c.anchor_tolerance_m, c.agreement_m)
    geometry = patch_geometry(catalog.points)
    relation = shape_relations(catalog.points)
    corners = catalog.points[:, [0, 4, 24, 20]]
    delta = corners - geometry.centroid[:, None]
    corners = corners - np.einsum('bki,bi->bk', delta, geometry.normal)[..., None] * geometry.normal[:, None]
    radius2 = np.max(np.sum((corners - geometry.centroid[:, None]) ** 2, axis=2), axis=1)
    supported = (geometry.support >= c.min_support) & (np.sqrt(geometry.eigenvalues[:, 0]) <= c.max_thickness_m)
    if catalog.patch_fill is not None:
        supported &= ~catalog.patch_fill
    ids = np.arange(len(catalog.points)) if query_indices is None else np.asarray(query_indices, int)
    rows, counts = ([], {'queries': len(ids), 'sensor_fill_quarantined': 0, 'no_local_structure': 0, 'no_ray_support': 0, 'already_supported': 0, 'insufficient_views': 0, 'ambiguous_layers': 0, 'foreground_warning': 0, 'reviewable': 0})
    anchors, ledger = ([], [])

    def retain(index, reason):
        ledger.append({'frame_index': int(catalog.frames[index]), 'u': int(catalog.pixels[index, 0]), 'v': int(catalog.pixels[index, 1]), 'raw_depth_m': float(catalog.depths[index]), 'role': reason, 'raw_preserved': True})
    indexed_frame, angular_trees = (None, [])
    for number, i in enumerate(ids):
        if progress is not None and number % 1000 == 0:
            progress(f'Yapi/isin aramasi {number}/{len(ids)} / {len(rows)} yer adayi')
        if catalog.sensor_fill is not None and catalog.sensor_fill[i]:
            counts['sensor_fill_quarantined'] += 1
            retain(i, 'SENSOR_FILL')
            continue
        if geometry.support[i] < 0.15:
            counts['no_local_structure'] += 1
            retain(i, 'NO_LOCAL_STRUCTURE')
            continue
        origin = catalog.origins[i]
        direction = (catalog.points[i, 12] - origin) / catalog.depths[i]
        if indexed_frame != catalog.frames[i]:
            rel = geometry.centroid - origin
            distance = np.linalg.norm(rel, axis=1)
            eligible = supported & (catalog.frames != catalog.frames[i]) & (distance > 1e-09) & (np.linalg.norm(catalog.origins - origin, axis=1) >= c.min_baseline_m)
            ref_ids = np.flatnonzero(eligible)
            ratio2 = np.minimum(radius2[ref_ids] / distance[ref_ids] ** 2, 1.0)
            chord = np.sqrt(2.0 * (1.0 - np.sqrt(1.0 - ratio2)))
            chord[ratio2 >= 1.0] = 2.0
            bins = np.ceil(np.log2(np.maximum(chord, 1e-07))).astype(int)
            angular_trees = []
            for b in np.unique(bins):
                mask = bins == b
                refs = ref_ids[mask]
                angular_trees.append((cKDTree(rel[refs] / distance[refs, None]), refs, float(np.max(chord[mask])) + 1e-08))
            indexed_frame = catalog.frames[i]
        unit = direction / np.linalg.norm(direction)
        candidates = np.concatenate([refs[tree.query_ball_point(unit, radius)] for tree, refs, radius in angular_trees]) if angular_trees else np.empty(0, int)
        if not len(candidates):
            counts['no_ray_support'] += 1
            retain(i, 'NO_RAY_SUPPORT')
            continue
        values = finite_patch_hits(origin, direction, geometry.centroid[candidates], geometry.normal[candidates], corners[candidates])
        depths = np.full(len(catalog.points), np.nan)
        depths[candidates] = values
        hits = candidates[np.isfinite(values)]
        if not len(hits):
            counts['no_ray_support'] += 1
            retain(i, 'NO_RAY_SUPPORT')
            continue
        error = np.full(len(catalog.points), np.inf)
        error[hits] = np.linalg.norm(relation[hits] - relation[i], axis=1)
        layers = []
        for j in sorted(hits, key=lambda j: depths[j]):
            if not layers or depths[j] - depths[layers[-1][0]] > c.agreement_m:
                layers.append([int(j)])
            else:
                layers[-1].append(int(j))
        accepted = []
        for layer in layers:
            matching = [j for j in layer if error[j] <= c.max_shape_error]
            votes = independent_hits(matching, depths, catalog, c)
            if len(votes) < c.min_views:
                continue
            estimate = float(np.median(depths[votes[:-1]]))
            held_error = abs(float(depths[votes[-1]]) - estimate)
            if held_error <= c.agreement_m:
                accepted.append((votes, estimate, held_error))
        if not accepted:
            counts['insufficient_views'] += 1
            retain(i, 'INSUFFICIENT_INDEPENDENT_VIEWS')
            continue
        if len(accepted) != 1:
            counts['ambiguous_layers'] += 1
            retain(i, 'AMBIGUOUS_LAYERS')
            continue
        votes, estimate, held_error = accepted[0]
        foreground = [int(j) for j in hits if depths[j] < estimate - c.agreement_m]
        status = 'FOREGROUND_UNRESOLVED' if foreground else 'REVIEWABLE_STRUCTURE_CANDIDATE'
        row = {'frame_index': int(catalog.frames[i]), 'u': int(catalog.pixels[i, 0]), 'v': int(catalog.pixels[i, 1]), 'raw_depth_m': float(catalog.depths[i]), 'candidate_depth_m': estimate, 'displacement_m': float(estimate - catalog.depths[i]), 'status': status, 'applied': False, 'origin': origin.tolist(), 'measured_point': catalog.points[i, 12].tolist(), 'candidate_point': (origin + estimate * direction).tolist(), 'measured_patch': catalog.points[i].tolist(), 'measured_patch_has_sensor_fill': bool(catalog.patch_fill[i]) if catalog.patch_fill is not None else None, 'support_patches': [{'frame': int(catalog.frames[j]), 'pixel': catalog.pixels[j].tolist(), 'origin': catalog.origins[j].tolist(), 'points': catalog.points[j].tolist()} for j in votes], 'support_frames': [int(catalog.frames[j]) for j in votes], 'fit_frames': [int(catalog.frames[j]) for j in votes[:-1]], 'held_out_frame': int(catalog.frames[votes[-1]]), 'held_out_error_m': held_error, 'shape_error': float(np.median(error[votes])), 'foreground_patch_count': len(foreground), 'source_surface_support': float(geometry.support[i]), 'source_surface_normal': geometry.normal[i].tolist(), 'position_certified': False}
        if abs(estimate - catalog.depths[i]) < c.minimum_change_m:
            counts['already_supported'] += 1
            residual = float(np.max(np.abs(depths[votes] - catalog.depths[i])))
            if supported[i] and (not foreground) and (max(residual, held_error) <= anchor_tolerance):
                row.update(status='ALIGNMENT_ANCHOR_CANDIDATE', candidate_depth_m=float(catalog.depths[i]), candidate_point=catalog.points[i, 12].tolist(), displacement_m=0.0, consensus_depth_m=estimate, anchor_max_residual_m=residual, anchor_tolerance_m=anchor_tolerance, alignment_approved=False, eligible_for_correspondence_search=True)
                anchors.append(row)
                retain(i, 'ALIGNMENT_ANCHOR_CANDIDATE')
            else:
                retain(i, 'STABLE_NOT_ANCHOR')
            continue
        counts['foreground_warning' if foreground else 'reviewable'] += 1
        rows.append(row)
        retain(i, status)
        if progress is not None and (number % 1000 == 0 or number + 1 == len(ids)):
            progress(f'Yapi/isin aramasi {number + 1}/{len(ids)} / {len(rows)} yer adayi')
    return {'counts': counts, 'candidates': rows, 'alignment_anchors': anchors, 'query_ledger': ledger, 'reference_patches': int(supported.sum()), 'policy': 'KEEP_RAW__STRUCTURE_IS_NOT_POSITION_CERTIFICATION'}
