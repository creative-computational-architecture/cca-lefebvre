
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class LocalGeometry:
    support: np.ndarray
    normal: np.ndarray
    centroid: np.ndarray
    eigenvalues: np.ndarray
    count: np.ndarray

def patch_geometry(neighbours: np.ndarray, valid: np.ndarray | None=None) -> LocalGeometry:
    p = np.asarray(neighbours, dtype=np.float64)
    if p.ndim != 3 or p.shape[2] != 3:
        raise ValueError('neighbourhoods must be BxKx3')
    mask = np.isfinite(p).all(axis=2)
    if valid is not None:
        v = np.asarray(valid, bool)
        if v.shape != mask.shape:
            raise ValueError('neighbourhood mask shape mismatch')
        mask &= v
    count = mask.sum(axis=1)
    safe = np.where(mask[..., None], p, 0.0)
    center = safe.sum(axis=1) / np.maximum(count, 1)[:, None]
    delta = np.where(mask[..., None], p - center[:, None, :], 0.0)
    cov = np.einsum('bki,bkj->bij', delta, delta) / np.maximum(count, 1)[:, None, None]
    eig, vec = np.linalg.eigh(cov)
    eig = np.maximum(eig, 0.0)
    smallest, middle, largest = eig.T
    planarity = (middle - smallest) / np.maximum(largest, 1e-15)
    flatness = np.clip(1.0 - 12.0 * smallest / np.maximum(eig.sum(axis=1), 1e-15), 0.0, 1.0)
    coverage = np.clip((count - 5.0) / 11.0, 0.0, 1.0)
    support = np.clip(planarity / 0.5, 0.0, 1.0) * flatness * coverage
    support[(middle < 1e-10) | (count < 6)] = 0.0
    return LocalGeometry(support.astype(np.float32), vec[:, :, 0], center, eig, count)

def surface_support(points: np.ndarray, *, radius_m: float=0.3, neighbours: int=24, chunk_size: int=32768, reference_mask: np.ndarray | None=None) -> np.ndarray:
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('points must be Nx3')
    if not np.isfinite(radius_m) or radius_m <= 0 or neighbours < 6 or (chunk_size < 1):
        raise ValueError('invalid neighbourhood configuration')
    finite = np.isfinite(p).all(axis=1)
    if reference_mask is not None:
        mask = np.asarray(reference_mask, bool)
        if mask.shape != finite.shape:
            raise ValueError('reference mask must carry one value per point')
        finite &= mask
    ids = np.flatnonzero(finite)
    output = np.zeros(len(p), dtype=np.float32)
    if len(ids) < 6:
        return output
    xyz = np.asarray(p[ids], dtype=np.float64)
    tree = cKDTree(xyz)
    k = min(int(neighbours), len(xyz))
    for start in range(0, len(xyz), chunk_size):
        stop = min(start + chunk_size, len(xyz))
        distance, index = tree.query(xyz[start:stop], k=k, distance_upper_bound=radius_m, workers=2)
        valid = np.isfinite(distance)
        patch = xyz[np.minimum(index, len(xyz) - 1)]
        output[ids[start:stop]] = patch_geometry(patch, valid).support
    return output

def relation_signature(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or (not np.isfinite(p).all()):
        raise ValueError('a finite Nx3 patch with at least 3 points is required')
    shape = patch_geometry(p[None]).eigenvalues[0]
    a, b = np.triu_indices(len(p), 1)
    distances = np.linalg.norm(p[a] - p[b], axis=1)
    return np.r_[shape / max(float(shape.sum()), 1e-15), np.quantile(distances, [0.1, 0.25, 0.5, 0.75, 0.9])]
