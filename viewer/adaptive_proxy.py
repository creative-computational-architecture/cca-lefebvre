
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
_NEIGHBORS = 20
_ATLAS_LIMIT = 4096
_RECTANGLE_LIMIT = 1022
_COVERAGE_MARGIN = 0.15

@dataclass(frozen=True, slots=True)
class AdaptiveSurface:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray | None
    texture_coordinates: np.ndarray
    texture_rgb: np.ndarray | None
    patch_group_ids: np.ndarray
    patch_neighbors: np.ndarray
    patch_boundary_mask: np.ndarray
    sample_patch_ids: np.ndarray
    patch_grade: np.ndarray | None = None
    texture_grade: np.ndarray | None = None

def _normal_frames(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.zeros_like(normals)
    reference[np.arange(len(normals)), np.argmin(np.abs(normals), axis=1)] = 1.0
    tangent = np.cross(normals, reference)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return (tangent, np.cross(normals, tangent))

def _fallback_frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(points)
    normals = np.tile([0.0, 0.0, 1.0], (count, 1))
    if count >= 3:
        _distances, indices = cKDTree(points).query(points, k=min(12, count))
        neighbors = points[indices]
        offsets = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum('nki,nkj->nij', offsets, offsets)
        _values, vectors = np.linalg.eigh(covariance)
        normals = vectors[:, :, 0]
    return _normal_frames(normals)

def _local_geometry(points: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(points)
    distances, indices = cKDTree(points).query(points, k=min(_NEIGHBORS, count), distance_upper_bound=cell_size * 1.9)
    if distances.ndim == 1:
        distances, indices = (distances[:, None], indices[:, None])
    valid = np.isfinite(distances)
    safe_indices = np.minimum(indices, count - 1)
    neighbors = points[safe_indices]
    sizes = valid.sum(axis=1)
    means = (neighbors * valid[:, :, None]).sum(axis=1) / sizes[:, None]
    offsets = (neighbors - means[:, None, :]) * valid[:, :, None]
    covariance = np.einsum('nki,nkj->nij', offsets, offsets)
    values, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    dominant = np.argmax(np.abs(normals), axis=1)
    signs = np.where(normals[np.arange(count), dominant] < 0, -1.0, 1.0)
    normals *= signs[:, None]
    reliable = (sizes >= 4) & (values[:, 1] > cell_size * cell_size * 0.025) & (values[:, 0] <= np.maximum(values[:, 1], 1e-30) * 0.018)
    projected = np.einsum('nki,ni->nk', neighbors - points[:, None, :], normals)
    projected[~valid] = np.inf
    projected.sort(axis=1)
    for split in range(3, projected.shape[1] - 2):
        enough = sizes >= split + 3
        left = projected[:, split - 1]
        right = projected[:, split]
        gap = np.zeros(count)
        np.subtract(right, left, out=gap, where=enough)
        last = projected[np.arange(count), sizes - 1]
        tight = np.zeros(count, dtype=bool)
        tight[enough] = (left[enough] - projected[enough, 0] <= gap[enough] * 0.35) & (last[enough] - right[enough] <= gap[enough] * 0.35)
        reliable &= ~(enough & tight & (gap > cell_size * 1e-06))
    return (normals, reliable, indices, distances)

def _surface_groups(points: np.ndarray, normals: np.ndarray, reliable: np.ndarray, indices: np.ndarray, distances: np.ndarray, cell_size: float) -> list[np.ndarray]:
    count = len(points)
    assigned = np.zeros(count, dtype=bool)
    groups: list[np.ndarray] = []
    max_residual = cell_size * 0.1
    cosine = math.cos(math.radians(8))
    adjacency = np.isfinite(distances) & (distances <= cell_size * 1.8) & (indices != np.arange(count)[:, None])
    for seed in range(count):
        if assigned[seed]:
            continue
        assigned[seed] = True
        if not reliable[seed]:
            groups.append(np.asarray([seed], dtype=np.int32))
            continue
        members = [seed]
        anchor, normal = (points[seed], normals[seed])
        cursor = 0
        fitted_size = 0
        while cursor < len(members):
            if len(members) >= max(16, fitted_size + 32):
                group_points = points[members]
                new_anchor = group_points.mean(axis=0)
                offsets = group_points - new_anchor
                _values, vectors = np.linalg.eigh(offsets.T @ offsets)
                new_normal = vectors[:, 0]
                if abs(new_normal @ normals[seed]) >= cosine and np.max(np.abs(offsets @ new_normal)) <= max_residual:
                    anchor, normal = (new_anchor, new_normal)
                fitted_size = len(members)
            current = members[cursor]
            cursor += 1
            candidates = indices[current, adjacency[current]]
            candidates = candidates[~assigned[candidates] & reliable[candidates]]
            if not len(candidates):
                continue
            candidates = candidates[(np.abs(normals[candidates] @ normal) >= cosine) & (np.abs((points[candidates] - anchor) @ normal) <= max_residual) & (np.abs((points[candidates] - points[current]) @ normals[current]) <= max_residual)]
            assigned[candidates] = True
            members.extend(candidates.tolist())
        groups.append(np.asarray(members, dtype=np.int32))
    return groups

def _rectangles(coordinates: np.ndarray) -> tuple[list[tuple[int, int, int, int]], dict[tuple[int, int], int]]:
    cells = {(int(u), int(v)) for u, v in coordinates}
    remaining = set(cells)
    rectangles: list[tuple[int, int, int, int]] = []
    owners: dict[tuple[int, int], int] = {}
    for u, v in sorted(cells, key=lambda key: (key[1], key[0])):
        if (u, v) not in remaining:
            continue
        width = 1
        while width < _RECTANGLE_LIMIT and (u + width, v) in remaining:
            width += 1
        height = 1
        while height < _RECTANGLE_LIMIT and all(((x, v + height) in remaining for x in range(u, u + width))):
            height += 1
        patch = len(rectangles)
        rectangles.append((u, v, width, height))
        for y in range(v, v + height):
            for x in range(u, u + width):
                remaining.remove((x, y))
                owners[x, y] = patch
    return (rectangles, owners)

def _rectangle_boundary(rectangle: tuple[int, int, int, int], owners: dict[tuple[int, int], int]) -> int:
    u, v, width, height = rectangle
    bottom = any(((x, v - 1) not in owners for x in range(u, u + width)))
    right = any(((u + width, y) not in owners for y in range(v, v + height)))
    top = any(((x, v + height) not in owners for x in range(u, u + width)))
    left = any(((u - 1, y) not in owners for y in range(v, v + height)))
    return int(bottom) | int(right) << 1 | int(top) << 2 | int(left) << 3

def _pack_texture(tiles: list[np.ndarray], grade_tiles: list[np.ndarray] | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    unique_tiles: list[np.ndarray] = []
    unique_grades: list[np.ndarray] = []
    constant_tiles: dict[bytes, int] = {}
    tile_ids = np.empty(len(tiles), dtype=np.int32)
    for patch, tile in enumerate(tiles):
        grade_tile = None if grade_tiles is None else grade_tiles[patch]
        if tile.shape[:2] == (1, 1):
            key = tile.tobytes()
            if grade_tile is not None:
                key += grade_tile.tobytes()
            tile_id = constant_tiles.get(key)
            if tile_id is None:
                tile_id = len(unique_tiles)
                constant_tiles[key] = tile_id
                unique_tiles.append(tile)
                unique_grades.append(grade_tile)
        else:
            tile_id = len(unique_tiles)
            unique_tiles.append(tile)
            unique_grades.append(grade_tile)
        tile_ids[patch] = tile_id
    tiles = unique_tiles
    area = sum(((tile.shape[0] + 2) * (tile.shape[1] + 2) for tile in tiles))
    longest = max((tile.shape[1] + 2 for tile in tiles))
    width = min(_ATLAS_LIMIT, 2 ** math.ceil(math.log2(max(longest, math.sqrt(area), 4))))
    placements: list[tuple[int, int]] = []
    while True:
        placements.clear()
        x = y = row_height = 0
        ordered = sorted(range(len(tiles)), key=lambda i: (-tiles[i].shape[0], i))
        packed: dict[int, tuple[int, int]] = {}
        for patch in ordered:
            tile = tiles[patch]
            height, tile_width = (tile.shape[0] + 2, tile.shape[1] + 2)
            if x + tile_width > width:
                y += row_height
                x = row_height = 0
            packed[patch] = (x, y)
            x += tile_width
            row_height = max(row_height, height)
        height = y + row_height
        if height <= _ATLAS_LIMIT:
            placements = [packed[i] for i in range(len(tiles))]
            break
        if width == _ATLAS_LIMIT:
            raise ValueError('display atlas exceeds the bounded 4096 pixel capacity')
        width = min(width * 2, _ATLAS_LIMIT)
    atlas = np.zeros((height, width, 3), dtype=np.uint8)
    grade_atlas = None if grade_tiles is None else np.zeros((height, width), dtype=np.float32)
    uv = np.empty((len(tiles), 4, 2), dtype=np.float32)
    for patch, (tile, (x, y)) in enumerate(zip(tiles, placements, strict=True)):
        tile_height, tile_width = tile.shape[:2]
        atlas[y:y + tile_height + 2, x:x + tile_width + 2] = np.pad(tile, ((1, 1), (1, 1), (0, 0)), mode='edge')
        if grade_atlas is not None:
            grade_atlas[y:y + tile_height + 2, x:x + tile_width + 2] = np.pad(unique_grades[patch], ((1, 1), (1, 1)), mode='edge')
        u0 = (x + 1 - _COVERAGE_MARGIN) / width
        u1 = (x + 1 + tile_width + _COVERAGE_MARGIN) / width
        v0 = 1 - (y + 1 + tile_height + _COVERAGE_MARGIN) / height
        v1 = 1 - (y + 1 - _COVERAGE_MARGIN) / height
        uv[patch] = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
    return (uv[tile_ids].reshape(-1, 2), atlas, grade_atlas)

def _validated_sample_grades(grades: object, weights: object, count: int) -> tuple[np.ndarray | None, np.ndarray]:
    if grades is None:
        return (None, np.ones(count, dtype=np.float64))
    grades = np.asarray(grades, dtype=np.float64)
    if grades.shape != (count,) or not np.isfinite(grades).all() or (grades.size and (grades.min() < 0.0 or grades.max() > 1.0)):
        raise ValueError('adaptive display grades must be finite values in [0, 1], one per sample')
    if weights is None:
        return (grades, np.ones(count, dtype=np.float64))
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (count,) or not np.isfinite(weights).all() or (weights.size and weights.min() <= 0):
        raise ValueError('adaptive display weights must be finite positive values, one per sample')
    return (grades, weights)

def build_adaptive_surface(centers: np.ndarray, cell_size_m: float, colors: np.ndarray | None=None, *, grades: np.ndarray | None=None, weights: np.ndarray | None=None) -> AdaptiveSurface:
    points = np.asarray(centers, dtype=np.float64)
    cell_size = float(cell_size_m)
    if points.ndim != 2 or points.shape[1] != 3 or (not len(points)) or (not np.isfinite(points).all()) or (not math.isfinite(cell_size)) or (cell_size <= 0):
        raise ValueError('adaptive display samples must be finite Nx3 points and a positive cell size')
    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype != np.uint8 or colors.shape != points.shape:
            raise ValueError('adaptive display colors must be uint8 Nx3')
    grades, weights = _validated_sample_grades(grades, weights, len(points))
    normals, reliable, indices, distances = _local_geometry(points, cell_size)
    groups = _surface_groups(points, normals, reliable, indices, distances, cell_size)
    fallback_frames: tuple[np.ndarray, np.ndarray] | None = None
    quads: list[np.ndarray] = []
    patch_colors: list[np.ndarray] = []
    patch_grades: list[float] = []
    group_ids: list[int] = []
    boundary: list[int] = []
    tiles: list[np.ndarray] = []
    grade_tiles: list[np.ndarray] = []
    sample_patch_ids = np.full(len(points), -1, dtype=np.int32)

    def add_fallback(samples: np.ndarray, group_id: int) -> None:
        nonlocal fallback_frames
        if fallback_frames is None:
            fallback_frames = _fallback_frames(points)
        tangents, bitangents = fallback_frames
        for sample in samples:
            sample_patch_ids[sample] = len(quads)
            tangent = tangents[sample] * cell_size * (0.5 + _COVERAGE_MARGIN)
            bitangent = bitangents[sample] * cell_size * (0.5 + _COVERAGE_MARGIN)
            quads.append(points[sample] + np.stack((-tangent - bitangent, tangent - bitangent, tangent + bitangent, -tangent + bitangent)))
            group_ids.append(group_id)
            boundary.append(15)
            if colors is not None:
                patch_colors.append(colors[sample])
                tiles.append(colors[sample].reshape(1, 1, 3))
            if grades is not None:
                patch_grades.append(float(grades[sample]))
                if colors is not None:
                    grade_tiles.append(np.full((1, 1), grades[sample], dtype=np.float32))
    for group_id, samples in enumerate(groups):
        if len(samples) < 4:
            add_fallback(samples, group_id)
            continue
        group_points = points[samples]
        origin = group_points.mean(axis=0)
        offsets = group_points - origin
        _values, vectors = np.linalg.eigh(offsets.T @ offsets)
        normal = vectors[:, 0]
        if normal @ normals[samples[0]] < 0:
            normal = -normal
        if np.max(np.abs(offsets @ normal)) > cell_size * 0.12:
            add_fallback(samples, group_id)
            continue
        tangent, bitangent = _normal_frames(normal[None, :])
        tangent, bitangent = (tangent[0], bitangent[0])
        projected = np.column_stack((offsets @ tangent, offsets @ bitangent))
        anchor = projected.min(axis=0)
        floating = (projected - anchor) / cell_size + 0.5
        if np.max(np.abs(floating)) > np.iinfo(np.int32).max:
            add_fallback(samples, group_id)
            continue
        coordinates = np.floor(floating + 1e-09).astype(np.int64)
        rectangles, owners = _rectangles(coordinates)
        if len(rectangles) >= len(samples):
            add_fallback(samples, group_id)
            continue
        patch_base = len(quads)
        local_patch_ids = np.asarray([owners[tuple(key)] for key in coordinates], dtype=np.int32)
        sample_patch_ids[samples] = patch_base + local_patch_ids
        color_cells: dict[tuple[int, int], np.ndarray] = {}
        grade_cells: dict[tuple[int, int], float] = {}
        rectangle_colors = rectangle_grades = None
        if colors is not None or grades is not None:
            keys, inverse = np.unique(coordinates, axis=0, return_inverse=True)
            counts = np.bincount(inverse)
            rectangle_counts = np.bincount(local_patch_ids, minlength=len(rectangles))
        if colors is not None:
            sums = np.column_stack([np.bincount(inverse, weights=colors[samples, channel]) for channel in range(3)])
            averages = np.rint(sums / counts[:, None]).clip(0, 255).astype(np.uint8)
            color_cells = {tuple(key): color for key, color in zip(keys, averages, strict=True)}
            rectangle_sums = np.column_stack([np.bincount(local_patch_ids, weights=colors[samples, channel], minlength=len(rectangles)) for channel in range(3)])
            rectangle_colors = np.rint(rectangle_sums / rectangle_counts[:, None])
        if grades is not None:
            group_weights = weights[samples]
            cell_weight = np.bincount(inverse, weights=group_weights)
            cell_grade = np.bincount(inverse, weights=group_weights * grades[samples]) / cell_weight
            grade_cells = {tuple(key): float(value) for key, value in zip(keys, cell_grade, strict=True)}
            rectangle_weight = np.bincount(local_patch_ids, weights=group_weights, minlength=len(rectangles))
            rectangle_grades = np.bincount(local_patch_ids, weights=group_weights * grades[samples], minlength=len(rectangles)) / rectangle_weight
        for local_patch, rectangle in enumerate(rectangles):
            u, v, width, height = rectangle
            uv_corners = anchor + cell_size * np.asarray([[u - 0.5 - _COVERAGE_MARGIN, v - 0.5 - _COVERAGE_MARGIN], [u + width - 0.5 + _COVERAGE_MARGIN, v - 0.5 - _COVERAGE_MARGIN], [u + width - 0.5 + _COVERAGE_MARGIN, v + height - 0.5 + _COVERAGE_MARGIN], [u - 0.5 - _COVERAGE_MARGIN, v + height - 0.5 + _COVERAGE_MARGIN]])
            quads.append(origin + uv_corners[:, :1] * tangent + uv_corners[:, 1:] * bitangent)
            group_ids.append(group_id)
            boundary.append(_rectangle_boundary(rectangle, owners))
            if colors is not None:
                tile = np.empty((height, width, 3), dtype=np.uint8)
                for dy in range(height):
                    for dx in range(width):
                        tile[height - 1 - dy, dx] = color_cells[u + dx, v + dy]
                tiles.append(tile)
                patch_colors.append(rectangle_colors[local_patch])
            if grades is not None:
                patch_grades.append(float(rectangle_grades[local_patch]))
                if colors is not None:
                    grade_tile = np.empty((height, width), dtype=np.float32)
                    for dy in range(height):
                        for dx in range(width):
                            grade_tile[height - 1 - dy, dx] = grade_cells[u + dx, v + dy]
                    grade_tiles.append(grade_tile)
    patch_count = len(quads)
    vertices = np.asarray(quads, dtype=np.float32).reshape(-1, 3)
    base = (4 * np.arange(patch_count, dtype=np.int64))[:, None, None]
    faces = (base + np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)).reshape(-1, 3)
    valid = np.isfinite(distances) & (distances <= cell_size * 1.8)
    rows, columns = np.nonzero(valid)
    pairs = np.column_stack((sample_patch_ids[rows], sample_patch_ids[indices[rows, columns]]))
    pairs.sort(axis=1)
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0).astype(np.int32)
    vertex_colors = texture = texture_grade = patch_grade = None
    texture_coordinates = np.zeros((4 * patch_count, 2), dtype=np.float32)
    if colors is not None:
        vertex_colors = np.repeat(np.asarray(patch_colors, dtype=np.uint8), 4, axis=0)
        texture_coordinates, texture, texture_grade = _pack_texture(tiles, grade_tiles if grades is not None else None)
    if grades is not None:
        patch_grade = np.clip(np.asarray(patch_grades, dtype=np.float32), 0.0, 1.0)
        if texture_grade is not None:
            texture_grade = np.clip(texture_grade, 0.0, 1.0)
    return AdaptiveSurface(vertices, faces, vertex_colors, texture_coordinates, texture, np.asarray(group_ids, dtype=np.int32), pairs.reshape(-1, 2), np.asarray(boundary, dtype=np.uint8), sample_patch_ids, patch_grade, texture_grade)
