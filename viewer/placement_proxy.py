
from __future__ import annotations
import hashlib
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from geometry.ply_preserve import iter_ply_vertex_chunks, read_ply_schema
from viewer.adaptive_proxy import build_adaptive_surface
from viewer.colour_gap import COLOUR_GAP_DISPLAY_RGB, COLOUR_GAP_FIELD, COLOUR_GAP_UNASSIGNED, colour_gap_column, placeholder_mask
from viewer.depth_fill import apply_depth_fill_display, depth_fill_column, validated_fill_share
from viewer.evidence_grade import CONF_MEAN_FIELD, N_VIEWS_FIELD, compute_grade
from viewer.floating_votes import FLOATING_MIN_THROUGH, load_floating_votes, sidecar_paths
from viewer.sparse_votes import load_sparse_sidecar, sparse_sidecar_paths
_CACHE_VERSION = 6
_CCA_CACHE_VERSION = 2
_CHUNK_VERTICES = 125000
_TARGET_SURFELS = 30000
_MAX_SURFELS = 40000
_AXES = {'CAD_Z_up', 'ARKit_Y_up'}
_RGB_CANDIDATES = (('red', 'green', 'blue'), ('r', 'g', 'b'), ('diffuse_red', 'diffuse_green', 'diffuse_blue'))

class PlacementProxyCapacityError(ValueError):
    pass

def _validated_cca(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError('CCA metre basina sonlu ve pozitif bir sayi olmali')
    result = float(value)
    if not math.isfinite(result) or result <= 0 or (not math.isfinite(1.0 / result)):
        raise ValueError('CCA metre basina sonlu ve pozitif bir sayi olmali')
    return result

def _detail_memory_budget() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available * 0.55)

def _check_detail_capacity(count: int, cca: float, budget: int | None) -> None:
    needed = count * 768 + 128 * 1024 * 1024
    if budget is not None and needed > budget:
        raise PlacementProxyCapacityError(f'{cca:g} CCA icin en az {needed / 1024 ** 3:.2f} GB gecici bellek tahmin ediliyor; guvenli bos butce {budget / 1024 ** 3:.2f} GB. Diger uygulamalari kapatin veya daha dusuk CCA secin. Istenen ayrinti otomatik dusurulmedi; kaynak noktalar degismedi.')

@dataclass(frozen=True, slots=True)
class PlacementProxy:
    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray | None
    source_point_count: int
    cell_size_m: float
    cache_path: Path
    texture_coordinates: np.ndarray
    texture_rgb: np.ndarray | None
    patch_group_ids: np.ndarray
    patch_neighbors: np.ndarray
    patch_boundary_mask: np.ndarray
    sample_patch_ids: np.ndarray
    sample_centers: np.ndarray
    sample_grade: np.ndarray | None = None
    sample_gap_fraction: np.ndarray | None = None
    patch_grade: np.ndarray | None = None
    texture_grade: np.ndarray | None = None
    sample_fill_share: np.ndarray | None = None
    sample_floating_t: np.ndarray | None = None
    sample_sparse_m: np.ndarray | None = None

def _floating_signature(source: Path) -> dict[str, int] | None:
    votes_path, meta_path = sidecar_paths(source)
    try:
        votes_stat, meta_stat = (votes_path.stat(), meta_path.stat())
    except OSError:
        return None
    return {'size': votes_stat.st_size, 'mtime_ns': votes_stat.st_mtime_ns, 'meta_size': meta_stat.st_size, 'meta_mtime_ns': meta_stat.st_mtime_ns}

def _sparse_signature(source: Path) -> dict[str, int] | None:
    values_path, meta_path = sparse_sidecar_paths(source)
    try:
        values_stat, meta_stat = (values_path.stat(), meta_path.stat())
    except OSError:
        return None
    return {'size': values_stat.st_size, 'mtime_ns': values_stat.st_mtime_ns, 'meta_size': meta_stat.st_size, 'meta_mtime_ns': meta_stat.st_mtime_ns}

def _source_signature(source: Path, source_axis: str, version: int=_CACHE_VERSION, *, cca_per_m: float | None=None) -> str:
    stat = source.stat()
    payload = {'version': version, 'path': os.path.normcase(str(source)), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'axis': source_axis}
    if cca_per_m is not None:
        payload['cca_per_m'] = float(cca_per_m)
        payload['cca_version'] = _CCA_CACHE_VERSION
    if version >= 5:
        payload['floating'] = _floating_signature(source)
    if version >= 6:
        payload['sparse'] = _sparse_signature(source)
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))

def _unit_array_ok(values: np.ndarray | None, shape: tuple[int, ...]) -> bool:
    return values is None or (values.dtype == np.float32 and values.shape == shape and bool(np.isfinite(values).all()) and (not values.size or (values.min() >= 0.0 and values.max() <= 1.0)))

def _metre_array_ok(values: np.ndarray | None, shape: tuple[int, ...]) -> bool:
    return values is None or (values.dtype == np.float32 and values.shape == shape and bool(np.isfinite(values).all()) and (not values.size or values.min() >= 0.0))

def _validated_proxy(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None, source_point_count: int, cell_size_m: float, cache_path: Path, texture_coordinates: np.ndarray, texture_rgb: np.ndarray | None, patch_group_ids: np.ndarray, patch_neighbors: np.ndarray, patch_boundary_mask: np.ndarray, sample_patch_ids: np.ndarray, sample_centers: np.ndarray, *, sample_grade: np.ndarray | None=None, sample_gap_fraction: np.ndarray | None=None, patch_grade: np.ndarray | None=None, texture_grade: np.ndarray | None=None, sample_fill_share: np.ndarray | None=None, sample_floating_t: np.ndarray | None=None, sample_sparse_m: np.ndarray | None=None) -> PlacementProxy:
    if vertices.dtype != np.float32 or vertices.ndim != 2 or vertices.shape[1] != 3 or (not len(vertices)) or (not np.isfinite(vertices).all()) or (faces.dtype != np.int64) or (faces.ndim != 2) or (faces.shape[1] != 3) or (not len(faces)) or (faces.min() < 0) or (faces.max() >= len(vertices)) or (source_point_count < 1) or (not math.isfinite(cell_size_m)) or (cell_size_m <= 0):
        raise ValueError('invalid placement proxy geometry')
    if colors is not None and (colors.dtype != np.uint8 or colors.shape != vertices.shape):
        raise ValueError('invalid placement proxy colors')
    patches = len(vertices) // 4
    if len(vertices) % 4 or len(faces) != patches * 2 or texture_coordinates.dtype != np.float32 or (texture_coordinates.shape != (len(vertices), 2)) or (not np.isfinite(texture_coordinates).all()) or np.any((texture_coordinates < 0) | (texture_coordinates > 1)) or (patch_group_ids.dtype != np.int32) or (patch_group_ids.shape != (patches,)) or np.any(patch_group_ids < 0) or (patch_neighbors.dtype != np.int32) or (patch_neighbors.ndim != 2) or (patch_neighbors.shape[1] != 2) or np.any((patch_neighbors < 0) | (patch_neighbors >= patches)) or (patch_boundary_mask.dtype != np.uint8) or (patch_boundary_mask.shape != (patches,)) or np.any(patch_boundary_mask > 15) or (sample_patch_ids.dtype != np.int32) or (sample_patch_ids.ndim != 1) or (not len(sample_patch_ids)) or np.any((sample_patch_ids < 0) | (sample_patch_ids >= patches)) or (sample_centers.dtype != np.float32) or (sample_centers.shape != (len(sample_patch_ids), 3)) or (not np.isfinite(sample_centers).all()) or (len(np.unique(sample_patch_ids)) != patches):
        raise ValueError('invalid placement proxy topology')
    if texture_rgb is not None and (texture_rgb.dtype != np.uint8 or texture_rgb.ndim != 3 or texture_rgb.shape[2] != 3 or (not min(texture_rgb.shape[:2]))):
        raise ValueError('invalid placement proxy texture')
    samples = len(sample_patch_ids)
    if not _unit_array_ok(sample_grade, (samples,)) or not _unit_array_ok(patch_grade, (patches,)) or (sample_grade is None) != (patch_grade is None) or (texture_grade is not None and (sample_grade is None or texture_rgb is None or (not _unit_array_ok(texture_grade, texture_rgb.shape[:2])))) or (texture_grade is None and sample_grade is not None and (texture_rgb is not None)):
        raise ValueError('invalid placement proxy evidence grade')
    if not _unit_array_ok(sample_gap_fraction, (samples,)):
        raise ValueError('invalid placement proxy colour gap')
    if not _unit_array_ok(sample_fill_share, (samples,)):
        raise ValueError('invalid placement proxy depth-fill share')
    if not _unit_array_ok(sample_floating_t, (samples,)):
        raise ValueError('invalid placement proxy floating threshold')
    if not _metre_array_ok(sample_sparse_m, (samples,)):
        raise ValueError('invalid placement proxy sparse threshold')
    return PlacementProxy(vertices, faces, colors, source_point_count, cell_size_m, cache_path, texture_coordinates, texture_rgb, patch_group_ids, patch_neighbors, patch_boundary_mask, sample_patch_ids, sample_centers, sample_grade, sample_gap_fraction, patch_grade, texture_grade, sample_fill_share, sample_floating_t, sample_sparse_m)

def _read_cache(cache_path: Path, signature: str, *, proxy_path: Path | None=None, version: int=_CACHE_VERSION) -> PlacementProxy | None:
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if str(data['signature'].item()) != signature:
                return None
            colors = data['colors'] if bool(data['has_colors'].item()) else None
            evidence: dict[str, np.ndarray | None] = {}
            if version >= 3:
                has_grade = bool(data['has_grade'].item())
                evidence = {'sample_grade': data['sample_grade'] if has_grade else None, 'patch_grade': data['patch_grade'] if has_grade else None, 'texture_grade': data['texture_grade'] if bool(data['has_texture_grade'].item()) else None, 'sample_gap_fraction': data['sample_gap_fraction'] if bool(data['has_gap'].item()) else None}
            if version >= 4:
                evidence['sample_fill_share'] = data['sample_fill_share'] if bool(data['has_fill'].item()) else None
            if version >= 5:
                evidence['sample_floating_t'] = data['sample_floating_t'] if bool(data['has_floating'].item()) else None
            if version >= 6:
                evidence['sample_sparse_m'] = data['sample_sparse_m'] if bool(data['has_sparse'].item()) else None
            return _validated_proxy(data['vertices'], data['faces'], colors, int(data['source_point_count'].item()), float(data['cell_size_m'].item()), proxy_path or cache_path, data['texture_coordinates'], data['texture_rgb'] if bool(data['has_texture'].item()) else None, data['patch_group_ids'], data['patch_neighbors'], data['patch_boundary_mask'], data['sample_patch_ids'], data['sample_centers'], **evidence)
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, EOFError, zipfile.BadZipFile):
        return None

def _write_cache(proxy: PlacementProxy, signature: str) -> None:
    proxy.cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f'.{proxy.cache_path.stem}.', suffix='.tmp', dir=proxy.cache_path.parent)
    temporary = Path(name)
    empty = np.empty(0, dtype=np.float32)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, signature=np.asarray(signature), vertices=proxy.vertices, faces=proxy.faces, colors=proxy.colors if proxy.colors is not None else np.empty((0, 3), dtype=np.uint8), has_colors=np.asarray(proxy.colors is not None), source_point_count=np.asarray(proxy.source_point_count), cell_size_m=np.asarray(proxy.cell_size_m), texture_coordinates=proxy.texture_coordinates, texture_rgb=proxy.texture_rgb if proxy.texture_rgb is not None else np.empty((0, 0, 3), dtype=np.uint8), has_texture=np.asarray(proxy.texture_rgb is not None), patch_group_ids=proxy.patch_group_ids, patch_neighbors=proxy.patch_neighbors, patch_boundary_mask=proxy.patch_boundary_mask, sample_patch_ids=proxy.sample_patch_ids, sample_centers=proxy.sample_centers, has_grade=np.asarray(proxy.sample_grade is not None), sample_grade=proxy.sample_grade if proxy.sample_grade is not None else empty, patch_grade=proxy.patch_grade if proxy.patch_grade is not None else empty, has_texture_grade=np.asarray(proxy.texture_grade is not None), texture_grade=proxy.texture_grade if proxy.texture_grade is not None else np.empty((0, 0), dtype=np.float32), has_gap=np.asarray(proxy.sample_gap_fraction is not None), sample_gap_fraction=proxy.sample_gap_fraction if proxy.sample_gap_fraction is not None else empty, has_fill=np.asarray(proxy.sample_fill_share is not None), sample_fill_share=proxy.sample_fill_share if proxy.sample_fill_share is not None else empty, has_floating=np.asarray(proxy.sample_floating_t is not None), sample_floating_t=proxy.sample_floating_t if proxy.sample_floating_t is not None else empty, has_sparse=np.asarray(proxy.sample_sparse_m is not None), sample_sparse_m=proxy.sample_sparse_m if proxy.sample_sparse_m is not None else empty)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, proxy.cache_path)
    finally:
        temporary.unlink(missing_ok=True)

def _voxel_reduce(sums: np.ndarray, counts: np.ndarray, cell_size: float, origin: np.ndarray, *, return_inverse: bool=False) -> tuple[np.ndarray, ...]:
    centers = sums[:, :3] / counts[:, None]
    coordinates = np.floor((centers - origin) / cell_size)
    pack = 1 << 21
    offset = pack // 2
    if coordinates.size and np.isfinite(coordinates).all() and (coordinates.min() >= -offset) and (coordinates.max() < offset):
        keys = coordinates[:, 0].astype(np.int64)
        keys += offset
        for column in (1, 2):
            keys *= pack
            keys += coordinates[:, column].astype(np.int64) + offset
        _keys, inverse = np.unique(keys, return_inverse=True)
    else:
        _keys, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    reduced = np.column_stack([np.bincount(inverse, weights=sums[:, column]) for column in range(sums.shape[1])])
    merged = np.bincount(inverse, weights=counts)
    if return_inverse:
        return (reduced, merged, inverse)
    return (reduced, merged)

def _initial_cell_size(points: np.ndarray) -> float:
    span = np.ptp(points, axis=0)
    area = span[0] * span[1] + span[1] * span[2] + span[0] * span[2]
    return max(math.sqrt(area / _TARGET_SURFELS), float(span.max()) / _TARGET_SURFELS, 0.001)

def _surfel_mesh(centers: np.ndarray, cell_size: float, colors: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    count = len(centers)
    normals = np.tile([0.0, 0.0, 1.0], (count, 1))
    if count >= 3:
        _distances, indices = cKDTree(centers).query(centers, k=min(12, count))
        neighbors = centers[indices]
        offsets = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum('nki,nkj->nij', offsets, offsets)
        _values, vectors = np.linalg.eigh(covariance)
        normals = vectors[:, :, 0]
    reference = np.zeros_like(normals)
    reference[np.arange(count), np.argmin(np.abs(normals), axis=1)] = 1.0
    tangent = np.cross(normals, reference)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(normals, tangent)
    radius = cell_size * 0.65
    corners = np.stack((centers + radius * (-tangent - bitangent), centers + radius * (tangent - bitangent), centers + radius * (tangent + bitangent), centers + radius * (-tangent + bitangent)), axis=1)
    base = (4 * np.arange(count, dtype=np.int64))[:, None, None]
    faces = (base + np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)).reshape(-1, 3)
    vertex_colors = None if colors is None else np.repeat(colors, 4, axis=0)
    return (corners.reshape(-1, 3).astype(np.float32), faces, vertex_colors)

def _from_samples(centers: np.ndarray, cell_size: float, colors: np.ndarray | None, source_count: int, cache_path: Path, *, explicit_detail: bool=False, grades: np.ndarray | None=None, weights: np.ndarray | None=None, gap_fraction: np.ndarray | None=None, fill_share: np.ndarray | None=None, floating_t: np.ndarray | None=None, sparse_m: np.ndarray | None=None) -> PlacementProxy:
    if explicit_detail:
        from viewer.quality_proxy import build_quality_surface
        surface = build_quality_surface(centers, cell_size, colors, grades=grades, weights=weights)
    else:
        surface = build_adaptive_surface(centers, cell_size, colors, grades=grades, weights=weights)
    return _validated_proxy(surface.vertices, surface.faces, surface.colors, source_count, cell_size, cache_path, surface.texture_coordinates, surface.texture_rgb, surface.patch_group_ids, surface.patch_neighbors, surface.patch_boundary_mask, surface.sample_patch_ids, np.asarray(centers, dtype=np.float32), sample_grade=None if grades is None else np.asarray(grades, dtype=np.float32), sample_gap_fraction=None if gap_fraction is None else np.asarray(gap_fraction, dtype=np.float32), patch_grade=surface.patch_grade, texture_grade=surface.texture_grade, sample_fill_share=None if fill_share is None else np.asarray(fill_share, dtype=np.float32), sample_floating_t=None if floating_t is None else np.asarray(floating_t, dtype=np.float32), sample_sparse_m=None if sparse_m is None else np.asarray(sparse_m, dtype=np.float32))

@dataclass(frozen=True, slots=True)
class _DisplayColumns:
    rgb: tuple[str, str, str] | None
    grade: tuple[str, str] | None
    gap: str | None
    fill: str | None = None

    @property
    def has_evidence(self) -> bool:
        return self.grade is not None or self.gap is not None or self.fill is not None

def _display_columns(schema) -> _DisplayColumns:
    names = {prop.name.lower(): prop.name for prop in schema.vertex.properties}
    rgb = next((tuple((names[channel] for channel in candidate)) for candidate in _RGB_CANDIDATES if all((channel in names for channel in candidate))), None)
    grade = None
    if CONF_MEAN_FIELD in names and N_VIEWS_FIELD in names:
        grade = (names[CONF_MEAN_FIELD], names[N_VIEWS_FIELD])
    return _DisplayColumns(rgb, grade, colour_gap_column(schema), depth_fill_column(schema))

def _legacy_cache_path(source: Path, source_axis: str, cache_root: Path, version: int) -> tuple[Path, str]:
    signature = _source_signature(source, source_axis, version=version)
    digest = hashlib.sha256(signature.encode('utf-8')).hexdigest()
    return (cache_root / f'placement-v{version}-{digest}.npz', signature)

def _upgrade_v1(source: Path, source_axis: str, cache_root: Path, cache_path: Path) -> PlacementProxy | None:
    old_path, signature = _legacy_cache_path(source, source_axis, cache_root, 1)
    try:
        with np.load(old_path, allow_pickle=False) as data:
            if str(data['signature'].item()) != signature:
                return None
            vertices, faces = (data['vertices'], data['faces'])
            colors = data['colors'] if bool(data['has_colors'].item()) else None
            count, cell = (int(data['source_point_count'].item()), float(data['cell_size_m'].item()))
            patches = len(vertices) // 4
            expected = (4 * np.arange(patches)[:, None, None] + np.asarray([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
            if vertices.dtype != np.float32 or vertices.shape != (patches * 4, 3) or (not patches) or (not np.isfinite(vertices).all()) or (faces.dtype != np.int64) or (not np.array_equal(faces, expected)) or (count < patches) or (not math.isfinite(cell)) or (cell <= 0):
                return None
            if colors is not None and (colors.dtype != np.uint8 or colors.shape != vertices.shape or (not np.all(colors.reshape(-1, 4, 3) == colors[::4, None, :]))):
                return None
            centers = vertices.astype(np.float64).reshape(-1, 4, 3).mean(axis=1)
            sample_colors = colors[::4].copy() if colors is not None else None
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, EOFError, zipfile.BadZipFile):
        return None
    return _from_samples(centers, cell, sample_colors, count, cache_path)

def _upgrade_v2(source: Path, source_axis: str, cache_root: Path, cache_path: Path) -> PlacementProxy | None:
    old_path, signature = _legacy_cache_path(source, source_axis, cache_root, 2)
    return _read_cache(old_path, signature, proxy_path=cache_path, version=2)

def _upgrade_v3(source: Path, source_axis: str, cache_root: Path, cache_path: Path) -> PlacementProxy | None:
    old_path, signature = _legacy_cache_path(source, source_axis, cache_root, 3)
    return _read_cache(old_path, signature, proxy_path=cache_path, version=3)

def _upgrade_v4(source: Path, source_axis: str, cache_root: Path, cache_path: Path) -> PlacementProxy | None:
    old_path, signature = _legacy_cache_path(source, source_axis, cache_root, 4)
    return _read_cache(old_path, signature, proxy_path=cache_path, version=4)

def _upgrade_v5(source: Path, source_axis: str, cache_root: Path, cache_path: Path) -> PlacementProxy | None:
    old_path, signature = _legacy_cache_path(source, source_axis, cache_root, 5)
    return _read_cache(old_path, signature, proxy_path=cache_path, version=5)

def _upgrade_previous(source: Path, source_axis: str, cache_root: Path, cache_path: Path) -> PlacementProxy | None:
    candidates = [version for version in (5, 4, 3, 2, 1) if _legacy_cache_path(source, source_axis, cache_root, version)[0].is_file()]
    if not candidates or _sparse_signature(source) is not None:
        return None
    if _floating_signature(source) is not None:
        candidates = [version for version in candidates if version >= 5]
    if not candidates:
        return None
    columns = _display_columns(read_ply_schema(source))
    upgrades = {5: _upgrade_v5, 4: _upgrade_v4, 3: _upgrade_v3, 2: _upgrade_v2, 1: _upgrade_v1}
    for version in candidates:
        if version < 4 and columns.fill is not None:
            return None
        if version < 3 and columns.has_evidence:
            return None
        proxy = upgrades[version](source, source_axis, cache_root, cache_path)
        if proxy is not None:
            return proxy
    return None

def _floating_point_ratio(votes, indices: np.ndarray) -> np.ndarray:
    through = votes.n_through[indices].astype(np.float64)
    confirm = votes.n_confirm[indices].astype(np.float64)
    total = through + confirm
    ratio = np.zeros(len(indices), dtype=np.float64)
    np.divide(through, total, out=ratio, where=total > 0)
    ratio[through < FLOATING_MIN_THROUGH] = -1.0
    return ratio

def majority_threshold(values: np.ndarray, groups: np.ndarray, count: int, *, maximum: float | None=1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    if values.shape != groups.shape:
        raise ValueError('one value per group member is required')
    if not len(values):
        return np.zeros(count, dtype=np.float32)
    sizes = np.bincount(groups, minlength=count)
    order = np.lexsort((-values, groups))
    starts = np.zeros(count, dtype=np.int64)
    np.cumsum(sizes[:-1], out=starts[1:])
    wanted = np.maximum(np.ceil(sizes / 2.0).astype(np.int64), 1)
    picked = np.where(sizes > 0, values[order[np.minimum(starts + wanted - 1, max(len(order) - 1, 0))]], 0.0)
    return np.clip(picked, 0.0, maximum).astype(np.float32)

def _build_proxy(source: Path, source_axis: str, cache_path: Path, *, cca_per_m: float | None=None) -> PlacementProxy:
    schema = read_ply_schema(source)
    columns = _display_columns(schema)
    channels = columns.rgb
    layout = 3
    rgb_slice = assigned_column = gap_column = grade_column = fill_column = None
    if channels is not None:
        rgb_slice = slice(layout, layout + 3)
        layout += 3
        if columns.gap is not None:
            assigned_column = layout
            layout += 1
    if columns.gap is not None:
        gap_column = layout
        layout += 1
    if columns.grade is not None:
        grade_column = layout
        layout += 1
    if columns.fill is not None:
        fill_column = layout
        layout += 1
    sums = np.empty((0, layout), dtype=np.float64)
    counts = np.empty(0, dtype=np.float64)
    votes = load_floating_votes(source, schema.vertex.count)
    kdist = load_sparse_sidecar(source, schema.vertex.count)
    tracked = votes is not None or kdist is not None
    point_ratio = point_kdist = point_cell = None
    read_vertices = 0
    origin = None
    cell_size = 0.001 if cca_per_m is None else 1.0 / cca_per_m
    budget = None if cca_per_m is None else _detail_memory_budget()
    for chunk in iter_ply_vertex_chunks(source, chunk_vertices=_CHUNK_VERTICES):
        points = np.column_stack((chunk['x'], chunk['y'], chunk['z'])).astype(np.float64)
        valid = np.isfinite(points).all(axis=1)
        chunk_base, read_vertices = (read_vertices, read_vertices + len(valid))
        points = points[valid]
        if not len(points):
            continue
        if source_axis == 'ARKit_Y_up':
            points = np.column_stack((points[:, 0], -points[:, 2], points[:, 1]))
        if origin is None:
            origin = points.min(axis=0)
            if cca_per_m is None:
                cell_size = _initial_cell_size(points)
        parts = [points]
        gap = None
        if columns.gap is not None:
            gap = np.asarray(chunk[columns.gap][valid], dtype=np.float64) == COLOUR_GAP_UNASSIGNED
        if channels is not None:
            rgb = np.column_stack([chunk[channel][valid] for channel in channels])
            if rgb.dtype.kind == 'f' and np.nanmax(rgb, initial=0.0) <= 1.0:
                rgb = rgb * 255.0
            rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0, 255)
            if gap is not None:
                assigned = ~(gap & placeholder_mask(rgb))
                parts.extend((rgb * assigned[:, None], assigned.astype(np.float64)))
            else:
                parts.append(rgb)
        if gap is not None:
            parts.append(gap.astype(np.float64))
        fill = None
        if columns.fill is not None:
            fill = validated_fill_share(chunk[columns.fill][valid])
        if columns.grade is not None:
            conf, views = (np.asarray(chunk[name][valid], dtype=np.float64) for name in columns.grade)
            parts.append(compute_grade(conf, views, fill_share=fill))
        if fill is not None:
            parts.append(fill.astype(np.float64))
        values = np.column_stack(parts)
        if tracked:
            kept = chunk_base + np.flatnonzero(valid)
            if votes is not None:
                ratio = _floating_point_ratio(votes, kept)
                point_ratio = ratio if point_ratio is None else np.concatenate((point_ratio, ratio))
            if kdist is not None:
                distances = np.asarray(kdist[kept], dtype=np.float64)
                distances = np.where(np.isfinite(distances), distances, 0.0)
                point_kdist = distances if point_kdist is None else np.concatenate((point_kdist, distances))
            arrived = len(sums) + np.arange(len(values), dtype=np.int64)
            point_cell = arrived if point_cell is None else np.concatenate((point_cell, arrived))
        tracking = {'return_inverse': True} if tracked else {}
        reduced = _voxel_reduce(np.vstack((sums, values)), np.concatenate((counts, np.ones(len(values)))), cell_size, origin, **tracking)
        sums, counts = (reduced[0], reduced[1])
        if tracked:
            point_cell = reduced[2][point_cell]
        if cca_per_m is None:
            while len(counts) > _MAX_SURFELS:
                cell_size *= max(1.1, math.sqrt(len(counts) / _TARGET_SURFELS))
                reduced = _voxel_reduce(sums, counts, cell_size, origin, **tracking)
                sums, counts = (reduced[0], reduced[1])
                if tracked:
                    point_cell = reduced[2][point_cell]
        else:
            _check_detail_capacity(len(counts), cca_per_m, budget)
    if not len(counts):
        raise ValueError(f'source has no finite XYZ for a placement proxy: {source}')
    centers = sums[:, :3] / counts[:, None]
    colors = None
    if channels is not None:
        if assigned_column is None:
            colors = np.rint(sums[:, rgb_slice] / counts[:, None]).clip(0, 255).astype(np.uint8)
        else:
            assigned = sums[:, assigned_column]
            coloured = assigned > 0
            colors = np.tile(np.asarray(COLOUR_GAP_DISPLAY_RGB, dtype=np.uint8), (len(counts), 1))
            colors[coloured] = np.rint(sums[coloured, rgb_slice] / assigned[coloured, None]).clip(0, 255).astype(np.uint8)
            del assigned, coloured
    gap_fraction = None
    if gap_column is not None:
        gap_fraction = np.clip(sums[:, gap_column] / counts, 0.0, 1.0).astype(np.float32)
    fill_share = None
    if fill_column is not None:
        fill_share = np.clip(sums[:, fill_column] / counts, 0.0, 1.0).astype(np.float32)
        if colors is not None:
            colors = apply_depth_fill_display(colors, fill_share)
    floating_t = None
    if votes is not None:
        floating_t = majority_threshold(point_ratio, point_cell, len(counts))
    sparse_m = None
    if kdist is not None:
        sparse_m = majority_threshold(point_kdist, point_cell, len(counts), maximum=None)
    grades = weights = None
    if grade_column is not None:
        grades = np.clip(sums[:, grade_column] / counts, 0.0, 1.0).astype(np.float32)
        weights = counts.astype(np.float32)
    del sums, counts, chunk, points, valid, values, parts, gap, fill
    del votes, kdist, point_ratio, point_kdist, point_cell, reduced
    if channels is not None:
        del rgb
    return _from_samples(centers, cell_size, colors, schema.vertex.count, cache_path, explicit_detail=cca_per_m is not None, grades=grades, weights=weights, gap_fraction=gap_fraction, fill_share=fill_share, floating_t=floating_t, sparse_m=sparse_m)

def load_or_build_proxy(source: Path, source_axis: str, cache_root: Path, *, cca_per_m: float | None=None) -> PlacementProxy:
    if source_axis not in _AXES:
        raise ValueError(f'unsupported source axis: {source_axis}')
    cca = _validated_cca(cca_per_m)
    source = Path(source).resolve(strict=True)
    signature = _source_signature(source, source_axis, cca_per_m=cca)
    digest = hashlib.sha256(signature.encode('utf-8')).hexdigest()
    cache_path = Path(cache_root).resolve() / f'placement-v{_CACHE_VERSION}-{digest}.npz'
    cached = _read_cache(cache_path, signature)
    if cached is not None and (cca is None or math.isclose(cached.cell_size_m, 1.0 / cca, rel_tol=1e-12, abs_tol=0.0)):
        return cached
    proxy = None if cca is not None else _upgrade_previous(source, source_axis, cache_path.parent, cache_path)
    try:
        if proxy is None:
            proxy = _build_proxy(source, source_axis, cache_path, cca_per_m=cca)
    except MemoryError as error:
        if cca is None:
            raise
        raise PlacementProxyCapacityError(f'{cca:g} CCA icin bellek yetersiz. Diger uygulamalari kapatin veya daha dusuk CCA secin; istenen ayrinti otomatik dusurulmedi.') from error
    if _source_signature(source, source_axis, cca_per_m=cca) != signature:
        raise ValueError(f'source changed while building placement proxy: {source}')
    _write_cache(proxy, signature)
    return proxy
