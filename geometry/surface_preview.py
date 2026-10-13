
from __future__ import annotations
import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import open3d as o3d
ALGORITHM_VERSION = 'cca-bpa-preview-v2-supported'
PRIMARY_ROLE = 'PRIMARY_APP_FUSED_SURFACE'
SOURCE_LAYER = 'rgb/core'
_SHA256 = re.compile('[0-9a-f]{64}')

class SurfacePreviewError(ValueError):
    pass

class SurfaceUnavailableError(SurfacePreviewError):
    pass

def sha256_file(path: str | Path, chunk_size: int=1024 * 1024) -> str:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise SurfacePreviewError('SHA-256 chunk size must be a positive integer')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

@dataclass(frozen=True)
class SurfaceCacheKey:
    run_id: str
    source_sha256: str
    algorithm_version: str
    viewer_voxel_m: float
    normal_radius_m: float
    normal_neighbors: int
    bpa_radii_m: tuple[float, float, float]
    max_edge_m: float
    support_radius_m: float = 0.0
    support_min_points: int = 10
    outlier_neighbors: int = 16
    outlier_std_ratio: float = 1.25

def surface_cache_eviction_keys(cached_keys: set[SurfaceCacheKey], *, visible_run_ids: set[str], protected_keys: set[SurfaceCacheKey]) -> tuple[SurfaceCacheKey, ...]:
    candidates = (key for key in cached_keys if key.run_id not in visible_run_ids and key not in protected_keys)
    return tuple(sorted(candidates, key=lambda key: (key.run_id, repr(key))))

@dataclass(frozen=True)
class SurfacePreviewSpec:
    run_id: str
    source_path: Path
    source_sha256: str
    source_role: str
    source_layer: str
    source_point_count: int
    viewer_voxel_m: float
    normal_radius_multiplier: float = 3.0
    normal_neighbors: int = 40
    bpa_radius_multipliers: tuple[float, float, float] = (1.2, 1.8, 2.5)
    max_edge_multiplier: float = 2.5
    support_radius_multiplier: float = 3.0
    support_min_points: int = 10
    outlier_neighbors: int = 16
    outlier_std_ratio: float = 1.25

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id or (not self.run_id.isascii()):
            raise SurfacePreviewError('run id must be non-empty ASCII')
        if not isinstance(self.source_path, Path):
            raise SurfacePreviewError('source path must be a pathlib Path')
        if not isinstance(self.source_sha256, str) or _SHA256.fullmatch(self.source_sha256) is None:
            raise SurfacePreviewError('source SHA-256 must contain 64 lowercase hex characters')
        if not isinstance(self.source_role, str) or self.source_role != PRIMARY_ROLE:
            raise SurfacePreviewError('surface input role must be PRIMARY_APP_FUSED_SURFACE')
        if not isinstance(self.source_layer, str) or self.source_layer != SOURCE_LAYER:
            raise SurfacePreviewError('surface input layer must be rgb/core')
        if isinstance(self.source_point_count, bool) or not isinstance(self.source_point_count, int):
            raise SurfacePreviewError('source point count must be an integer')
        if self.source_point_count < 1:
            raise SurfaceUnavailableError('run has zero fused-core points')
        _require_positive_real(self.viewer_voxel_m, 'viewer voxel')
        _require_positive_real(self.normal_radius_multiplier, 'normal radius multiplier')
        if type(self.bpa_radius_multipliers) is not tuple or len(self.bpa_radius_multipliers) != 3:
            raise SurfacePreviewError('BPA radius multipliers must be a three-value tuple')
        for index, multiplier in enumerate(self.bpa_radius_multipliers):
            _require_positive_real(multiplier, f'BPA radius multiplier {index}')
        _require_positive_real(self.max_edge_multiplier, 'max edge multiplier')
        _require_positive_real(self.support_radius_multiplier, 'support radius multiplier')
        _require_positive_real(self.outlier_std_ratio, 'outlier std ratio')
        if isinstance(self.normal_neighbors, bool) or not isinstance(self.normal_neighbors, int):
            raise SurfacePreviewError('normal_neighbors must be an integer')
        if self.normal_neighbors < 3:
            raise SurfacePreviewError('normal_neighbors must be at least 3')
        for label, value, minimum in (('support_min_points', self.support_min_points, 3), ('outlier_neighbors', self.outlier_neighbors, 3)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise SurfacePreviewError(f'{label} must be an integer at least {minimum}')
        object.__setattr__(self, 'source_path', self.source_path.resolve())

    @property
    def normal_radius_m(self) -> float:
        return self.viewer_voxel_m * self.normal_radius_multiplier

    @property
    def bpa_radii_m(self) -> tuple[float, float, float]:
        return tuple((self.viewer_voxel_m * value for value in self.bpa_radius_multipliers))

    @property
    def max_edge_m(self) -> float:
        return self.viewer_voxel_m * self.max_edge_multiplier

    @property
    def support_radius_m(self) -> float:
        return self.viewer_voxel_m * self.support_radius_multiplier

    @property
    def cache_key(self) -> SurfaceCacheKey:
        return SurfaceCacheKey(self.run_id, self.source_sha256, ALGORITHM_VERSION, self.viewer_voxel_m, self.normal_radius_m, self.normal_neighbors, self.bpa_radii_m, self.max_edge_m, self.support_radius_m, self.support_min_points, self.outlier_neighbors, self.outlier_std_ratio)

def make_surface_preview_spec(run: Mapping[str, Any], viewer_voxel_m: float, source_sha256: str) -> SurfacePreviewSpec:
    if not isinstance(run, Mapping):
        raise SurfacePreviewError('run must be a manifest mapping')
    run_id = run.get('id')
    if not isinstance(run_id, str):
        raise SurfacePreviewError('run id is missing')
    counts = run.get('viewer_layer_counts')
    core_count = counts.get('core', 0) if isinstance(counts, Mapping) else 0
    if isinstance(core_count, bool) or not isinstance(core_count, int) or core_count < 1:
        raise SurfaceUnavailableError(f'{run_id} has zero fused-core points')
    paths = run.get('viewer_layer_paths')
    rgb = paths.get('rgb') if isinstance(paths, Mapping) else None
    core_path = rgb.get('core') if isinstance(rgb, Mapping) else None
    if not isinstance(core_path, Path):
        raise SurfacePreviewError(f'{run_id} rgb/core path is missing')
    return SurfacePreviewSpec(run_id=run_id, source_path=core_path, source_sha256=source_sha256, source_role=run.get('full_rgb_geometry_role', ''), source_layer=SOURCE_LAYER, source_point_count=core_count, viewer_voxel_m=viewer_voxel_m)

def _require_positive_real(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (value <= 0.0):
        raise SurfacePreviewError(f'{label} must be a finite positive real number')

def triangle_max_edge_lengths(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    vertex_array, triangle_array = _validated_mesh_arrays(vertices, triangles)
    if not len(triangle_array):
        return np.empty(0, dtype=np.float64)
    first, second, third = (vertex_array[triangle_array[:, index]] for index in range(3))
    lengths = np.maximum.reduce((_safe_vector_lengths(first - second), _safe_vector_lengths(second - third), _safe_vector_lengths(third - first)))
    if not np.isfinite(lengths).all():
        raise SurfacePreviewError('mesh triangle edge lengths must be finite')
    return lengths

def retained_triangle_mask(vertices: np.ndarray, triangles: np.ndarray, max_edge_m: float) -> np.ndarray:
    _require_positive_real(max_edge_m, 'max_edge_m')
    vertex_array, triangle_array = _validated_mesh_arrays(vertices, triangles)
    if not len(triangle_array):
        return np.empty(0, dtype=bool)
    first, second, third = (vertex_array[triangle_array[:, index]] for index in range(3))
    maximum_edges = np.maximum.reduce((_safe_vector_lengths(first - second), _safe_vector_lengths(second - third), _safe_vector_lengths(third - first)))
    doubled_areas = _safe_vector_lengths(np.cross(second - first, third - first))
    return np.isfinite(maximum_edges) & np.isfinite(doubled_areas) & (maximum_edges <= max_edge_m) & (doubled_areas > 0.0)

@dataclass
class SurfacePreviewResult:
    spec: SurfacePreviewSpec
    cache_key: SurfaceCacheKey
    mesh: o3d.geometry.TriangleMesh = field(repr=False)
    source_sha256_before: str
    source_sha256_after: str
    source_point_count_before: int
    source_point_count_after: int
    generated_triangle_count: int
    retained_triangle_count: int
    removed_triangle_count: int
    maximum_retained_edge_m: float
    representation_input_point_count: int = 0
    representation_filtered_point_count: int = 0

def build_surface_preview(spec: SurfacePreviewSpec) -> SurfacePreviewResult:
    if not isinstance(spec, SurfacePreviewSpec):
        raise SurfacePreviewError('surface preview spec is invalid')
    before_hash = sha256_file(spec.source_path)
    if before_hash != spec.source_sha256:
        raise SurfacePreviewError('fused-core source changed before preview generation')
    cloud = o3d.io.read_point_cloud(str(spec.source_path))
    points = np.asarray(cloud.points)
    point_count_before = len(points)
    if point_count_before != spec.source_point_count or not _valid_point_array(points):
        raise SurfacePreviewError('fused-core point count or XYZ violates manifest contract')
    if point_count_before < 3:
        raise SurfaceUnavailableError('surface preview needs at least three fused-core points')
    representation_cloud = _representation_supported_cloud(cloud, spec)
    representation_count = len(np.asarray(representation_cloud.points))
    if representation_count < 3:
        raise SurfaceUnavailableError('representation filter retained fewer than three points')
    source_colors = _source_colors(representation_cloud, representation_count)
    representation_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=spec.normal_radius_m, max_nn=min(spec.normal_neighbors, representation_count)))
    representation_cloud.orient_normals_consistent_tangent_plane(min(spec.normal_neighbors, representation_count - 1))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(representation_cloud, o3d.utility.DoubleVector(spec.bpa_radii_m))
    triangles = np.asarray(mesh.triangles)
    generated_triangle_count = len(triangles)
    retained = retained_triangle_mask(np.asarray(mesh.vertices), triangles, spec.max_edge_m)
    mesh.remove_triangles_by_mask((~retained).tolist())
    mesh.remove_unreferenced_vertices()
    retained_triangle_count = len(mesh.triangles)
    if retained_triangle_count < 1:
        raise SurfaceUnavailableError('surface preview produced no guarded triangles')
    _assign_source_colors(mesh, representation_cloud, source_colors)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    retained_edges = triangle_max_edge_lengths(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    point_count_after = len(cloud.points)
    after_hash = sha256_file(spec.source_path)
    if after_hash != before_hash or point_count_after != point_count_before:
        raise SurfacePreviewError('fused-core source changed during preview generation')
    return SurfacePreviewResult(spec=spec, cache_key=spec.cache_key, mesh=mesh, source_sha256_before=before_hash, source_sha256_after=after_hash, source_point_count_before=point_count_before, source_point_count_after=point_count_after, generated_triangle_count=generated_triangle_count, retained_triangle_count=retained_triangle_count, removed_triangle_count=generated_triangle_count - retained_triangle_count, maximum_retained_edge_m=float(retained_edges.max()), representation_input_point_count=point_count_before, representation_filtered_point_count=representation_count)

def _representation_supported_cloud(cloud: o3d.geometry.PointCloud, spec: SurfacePreviewSpec) -> o3d.geometry.PointCloud:
    count = len(np.asarray(cloud.points))
    if count < 3:
        raise SurfaceUnavailableError('surface preview needs at least three fused-core points')
    neighbors = min(spec.outlier_neighbors, count - 1)
    if neighbors < 3:
        raise SurfaceUnavailableError('surface preview needs local point support')
    filtered, kept = cloud.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=spec.outlier_std_ratio)
    if len(kept) < spec.support_min_points:
        raise SurfaceUnavailableError('representation filter found too few supported points')
    labels = np.asarray(filtered.cluster_dbscan(eps=spec.support_radius_m, min_points=spec.support_min_points, print_progress=False), dtype=np.int32)
    retained = np.flatnonzero(labels >= 0)
    if len(retained) < 3:
        raise SurfaceUnavailableError('representation filter found no connected support')
    return filtered.select_by_index(retained.tolist())

def _validated_mesh_arrays(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        raw_vertices = np.asarray(vertices)
        raw_triangles = np.asarray(triangles)
    except (TypeError, ValueError) as error:
        raise SurfacePreviewError('mesh arrays are invalid') from error
    if raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3 or (not np.issubdtype(raw_vertices.dtype, np.number)) or np.issubdtype(raw_vertices.dtype, np.bool_):
        raise SurfacePreviewError('mesh vertices must be finite Nx3 coordinates')
    vertex_array = raw_vertices.astype(np.float64, copy=False)
    if not np.isfinite(vertex_array).all():
        raise SurfacePreviewError('mesh vertices must be finite Nx3 coordinates')
    if raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3 or (not np.issubdtype(raw_triangles.dtype, np.integer)) or np.issubdtype(raw_triangles.dtype, np.bool_):
        raise SurfacePreviewError('mesh triangles must be Mx3 integer indices')
    triangle_array = raw_triangles.astype(np.int64, copy=False)
    if triangle_array.size and (triangle_array.min() < 0 or triangle_array.max() >= len(vertex_array)):
        raise SurfacePreviewError('mesh triangle index is out of range')
    return (vertex_array, triangle_array)

def _safe_vector_lengths(vectors: np.ndarray) -> np.ndarray:
    return np.hypot(np.hypot(vectors[:, 0], vectors[:, 1]), vectors[:, 2])

def _valid_point_array(points: np.ndarray) -> bool:
    return points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()

def _source_colors(cloud: o3d.geometry.PointCloud, point_count: int) -> np.ndarray | None:
    if not cloud.has_colors():
        return None
    colors = np.asarray(cloud.colors)
    if colors.shape != (point_count, 3) or not np.isfinite(colors).all():
        raise SurfacePreviewError('fused-core RGB violates manifest contract')
    return colors.copy()

def _assign_source_colors(mesh: o3d.geometry.TriangleMesh, cloud: o3d.geometry.PointCloud, source_colors: np.ndarray | None) -> None:
    if source_colors is None:
        return
    cloud_index = o3d.geometry.KDTreeFlann(cloud)
    vertex_colors = np.empty((len(mesh.vertices), 3), dtype=np.float64)
    for index, vertex in enumerate(np.asarray(mesh.vertices)):
        neighbors, point_indices, _ = cloud_index.search_knn_vector_3d(vertex, 1)
        if neighbors != 1:
            raise SurfacePreviewError('could not assign fused-core RGB to preview mesh')
        vertex_colors[index] = source_colors[point_indices[0]]
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
