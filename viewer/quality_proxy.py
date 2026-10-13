
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from viewer.adaptive_proxy import AdaptiveSurface, build_adaptive_surface
_ADAPTIVE_SAMPLE_LIMIT = 40000
_NEIGHBOR_BATCH = 16384

def _neighborhoods(points: np.ndarray, cell_size: float):
    tree = cKDTree(points)
    count = len(points)
    native_spacing = cell_size
    if count >= 64:
        probes = np.linspace(0, count - 1, min(count, 4096), dtype=np.int64)
        distances, _indices = tree.query(points[probes], k=5)
        spacing = distances[:, 4]
        spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
        if len(spacing):
            native_spacing = max(float(np.median(spacing)), cell_size)
    for start in range(0, count, _NEIGHBOR_BATCH):
        stop = min(start + _NEIGHBOR_BATCH, count)
        distances, indices = tree.query(points[start:stop], k=min(12, count))
        if count == 1:
            distances, indices = (distances[:, None], indices[:, None])
        normals = np.tile([0.0, 0.0, 1.0], (stop - start, 1))
        support = np.full(stop - start, cell_size)
        if count >= 3:
            neighbors = points[indices]
            offsets = neighbors - neighbors.mean(axis=1, keepdims=True)
            covariance = np.einsum('nki,nkj->nij', offsets, offsets)
            _eigenvalues, vectors = np.linalg.eigh(covariance)
            normals = vectors[:, :, 0]
            if count >= 64:
                support[:] = np.maximum(cell_size, np.minimum(distances[:, 4], native_spacing * 2))
        yield (start, stop, normals, support, indices, distances)

def build_quality_surface(centers: np.ndarray, cell_size: float, colors: np.ndarray | None, *, grades: np.ndarray | None=None, weights: np.ndarray | None=None) -> AdaptiveSurface:
    points = np.asarray(centers, dtype=np.float64)
    count = len(points)
    if grades is not None:
        grades = np.asarray(grades, dtype=np.float32)
        if grades.shape != (count,):
            raise ValueError('quality display grades must supply one value per sample')
    if count <= _ADAPTIVE_SAMPLE_LIMIT:
        surface = build_adaptive_surface(points, cell_size, colors, grades=grades, weights=weights)
        patch_counts = np.bincount(surface.sample_patch_ids, minlength=len(surface.patch_group_ids))
        single = patch_counts[surface.sample_patch_ids] == 1
        if not single.any():
            return surface
        for start, stop, _normals, support, _indices, _distances in _neighborhoods(points, cell_size):
            samples = np.arange(start, stop)[single[start:stop] & (support > cell_size)]
            if not len(samples):
                continue
            patches = surface.sample_patch_ids[samples]
            quads = surface.vertices.reshape(-1, 4, 3)
            original = quads[patches].astype(np.float64)
            relative = original - original.mean(axis=1, keepdims=True)
            ratio = support[samples - start] / cell_size
            quads[patches] = points[samples, None, :] + relative * ratio[:, None, None]
        return surface
    vertices = np.empty((count, 4, 3), dtype=np.float32)
    faces = np.empty((count, 2, 3), dtype=np.int64)
    neighbor_chunks = []
    pattern = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    for start, stop, normals, support, indices, distances in _neighborhoods(points, cell_size):
        reference = np.zeros_like(normals)
        reference[np.arange(stop - start), np.argmin(np.abs(normals), axis=1)] = 1
        tangent = np.cross(normals, reference)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        bitangent = np.cross(normals, tangent)
        radius = (support * 0.65)[:, None]
        center = points[start:stop]
        vertices[start:stop, 0] = center - radius * (tangent + bitangent)
        vertices[start:stop, 1] = center + radius * (tangent - bitangent)
        vertices[start:stop, 2] = center + radius * (tangent + bitangent)
        vertices[start:stop, 3] = center + radius * (-tangent + bitangent)
        sample_ids = np.arange(start, stop, dtype=np.int32)
        faces[start:stop] = (4 * sample_ids.astype(np.int64))[:, None, None] + pattern
        nearby = indices[:, 1:5]
        valid = (nearby > sample_ids[:, None]) & (distances[:, 1:5] <= support[:, None] * 1.8)
        row, column = np.nonzero(valid)
        if len(row):
            neighbors = np.column_stack((sample_ids[row], nearby[row, column])).astype(np.int32)
            neighbors = neighbors[np.lexsort((neighbors[:, 1], neighbors[:, 0]))]
            neighbor_chunks.append(neighbors)
    neighbors = np.concatenate(neighbor_chunks) if neighbor_chunks else np.empty((0, 2), dtype=np.int32)
    ids = np.arange(count, dtype=np.int32)
    return AdaptiveSurface(vertices.reshape(-1, 3), faces.reshape(-1, 3), None if colors is None else np.repeat(colors, 4, axis=0), np.zeros((4 * count, 2), dtype=np.float32), None, ids.copy(), neighbors, np.full(count, 15, dtype=np.uint8), ids, None if grades is None else np.clip(grades, 0.0, 1.0).astype(np.float32), None)
