
from dataclasses import dataclass
import numpy as np
from .ray_structure import finite_patch_hits
from .surface_structure import patch_geometry

@dataclass(frozen=True)
class PatchworkingConfig:
    agreement_m: float = 0.08
    minimum_change_m: float = 0.08
    baseline_m: float = 0.12
    min_surface_support: float = 0.45
    max_thickness_m: float = 0.03
    subdivisions: int = 3
    allow_uncertain_visibility: bool = False

def _points(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (25, 3) or not np.isfinite(result).all():
        raise ValueError(f'{name}: finite 5x5 point patch required')
    return result

def reconstruct_patch(row, *, config=None):
    c = config or PatchworkingConfig()
    if type(c.allow_uncertain_visibility) is not bool:
        raise ValueError('uncertain visibility preview option must be bool')
    if type(c.subdivisions) is not int or not 1 <= c.subdivisions <= 8:
        raise ValueError('subdivisions must be an integer from 1 to 8')
    limits = (c.agreement_m, c.minimum_change_m, c.baseline_m, c.max_thickness_m)
    if not all((np.isfinite(x) and x > 0 for x in limits)):
        raise ValueError('reconstruction distances must be positive metres')
    if not np.isfinite(c.min_surface_support) or not 0 <= c.min_surface_support <= 1:
        raise ValueError('surface support must be between zero and one')
    raw = _points(row.get('measured_patch'), 'measured')
    empty = np.empty((0, 3))
    base = {'raw': raw.copy(), 'placed': raw.copy(), 'moved': np.zeros(25, bool), 'generated': empty, 'generated_uv': np.empty((0, 2)), 'held_out_errors_m': np.empty(0), 'unit': 'm', 'axis': 'ARKIT_Y_UP', 'measurement_certified': False, 'source_preserved': True, 'mode': 'LEFEBVRE_PATCHWORKING', 'alignment_eligible': False, 'visibility_unverified_hypothesis': False}

    def retain(reason):
        return {**base, 'reason': reason}
    if row.get('status') not in {'REVIEWABLE_STRUCTURE_CANDIDATE', 'VISIBILITY_UNVERIFIED'}:
        return retain('CONFLICT_OR_UNRESOLVED_RETAINED')
    uncertain = row.get('status') == 'VISIBILITY_UNVERIFIED' or bool(row.get('unverified_visibility_frames'))
    if uncertain and (not c.allow_uncertain_visibility):
        return retain('UNVERIFIED_VISIBILITY_RETAINED_BY_DEFAULT')
    if uncertain and (not row.get('unverified_visibility_frames')):
        return retain('MISSING_VISIBILITY_PROVENANCE')
    if row.get('free_space_conflict_frames') or row.get('foreground_patch_count', 0):
        return retain('VISIBILITY_CONFLICT_RETAINED')
    if row.get('measured_patch_has_sensor_fill') is not False:
        return retain('SENSOR_FILL_OR_UNKNOWN_PROVENANCE_RETAINED')
    origin = np.asarray(row.get('origin'), float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('finite source camera origin required')
    refs = row.get('support_patches', [])
    if len(refs) < 3:
        return retain('INSUFFICIENT_INDEPENDENT_PATCHES')
    frames = [r['frame'] for r in refs]
    if len(set(frames)) != len(frames) or row['frame_index'] in frames:
        return retain('DUPLICATE_OR_SOURCE_VIEW')
    if row.get('held_out_frame') not in frames:
        return retain('MISSING_HELD_OUT_VIEW')
    origins = np.asarray([origin, *[r['origin'] for r in refs]], float)
    if origins.shape != (len(refs) + 1, 3) or not np.isfinite(origins).all():
        raise ValueError('finite reference camera origins required')
    a, b = np.triu_indices(len(origins), 1)
    if np.any(np.linalg.norm(origins[a] - origins[b], axis=1) < c.baseline_m):
        return retain('CAMERAS_NOT_SPATIALLY_INDEPENDENT')
    patches = np.stack([_points(r['points'], 'reference') for r in refs])
    geometry = patch_geometry(patches)
    if np.any(geometry.support < c.min_surface_support) or np.any(np.sqrt(np.maximum(geometry.eigenvalues[:, 0], 0)) > c.max_thickness_m):
        return retain('WEAK_OR_THICK_REFERENCE')
    corners = patches[:, [0, 4, 24, 20]]
    held = frames.index(row['held_out_frame'])
    fit = np.arange(len(refs)) != held
    delta = raw - origin
    length = np.linalg.norm(delta, axis=1)
    if np.any(length <= 0):
        raise ValueError('raw samples must lie away from the camera')
    rays = (delta / length[:, None]).reshape(5, 5, 3)

    def cast(direction):
        depths = finite_patch_hits(origin, direction, geometry.centroid, geometry.normal, corners)
        if not np.isfinite(depths).all():
            return (None, None)
        estimate = float(np.median(depths[fit]))
        if np.max(np.abs(depths - estimate)) > c.agreement_m:
            return (None, None)
        return (origin + estimate * direction, float(abs(depths[held] - estimate)))
    errors = []
    for idx, direction in enumerate(rays.reshape(25, 3)):
        point, error = cast(direction)
        if point is not None:
            errors.append(error)
            if np.linalg.norm(point - raw[idx]) >= c.minimum_change_m:
                base['placed'][idx] = point
                base['moved'][idx] = True
    generated, generated_uv = ([], [])
    for iv in range(4 * c.subdivisions + 1):
        for iu in range(4 * c.subdivisions + 1):
            if iv % c.subdivisions == 0 and iu % c.subdivisions == 0:
                continue
            v, u = (iv / c.subdivisions, iu / c.subdivisions)
            v0, u0 = (min(int(v), 3), min(int(u), 3))
            y, x = (v - v0, u - u0)
            direction = (1 - x) * (1 - y) * rays[v0, u0] + x * (1 - y) * rays[v0, u0 + 1] + (1 - x) * y * rays[v0 + 1, u0] + x * y * rays[v0 + 1, u0 + 1]
            direction /= np.linalg.norm(direction)
            point, error = cast(direction)
            if point is not None:
                generated.append(point)
                generated_uv.append((u, v))
                errors.append(error)
    return {**base, 'generated': np.asarray(generated).reshape(-1, 3), 'generated_uv': np.asarray(generated_uv).reshape(-1, 2), 'held_out_errors_m': np.asarray(errors), 'visibility_unverified_hypothesis': uncertain, 'unverified_visibility_frames': list(row.get('unverified_visibility_frames', [])), 'alignment_eligible': False, 'reason': 'LOW_CONFIDENCE_VISIBILITY_HYPOTHESIS' if uncertain else 'FINITE_SURFACE_PREVIEW_ONLY'}
