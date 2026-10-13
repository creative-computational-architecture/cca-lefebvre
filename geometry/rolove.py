
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from typing import Any
import numpy as np

def rigid_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Rijit donusum sonlu 4x4 matris olmali')
    rotation = matrix[:3, :3]
    if not (np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-08) and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-06) and np.isclose(np.linalg.det(rotation), 1, atol=1e-06)):
        raise ValueError('Donusum olcegi 1 olmali; ayna ve esnetme kullanilamaz')
    return matrix

def alignment_matrix(metadata: dict) -> np.ndarray:
    return rigid_matrix(metadata.get('rolove', {}).get('alignment', np.eye(4)))

def placed_matrix(placement: Any, metadata: dict) -> np.ndarray:
    return rigid_matrix(placement) @ alignment_matrix(metadata)

def transform_points(points: Any, matrix: Any) -> np.ndarray:
    matrix = rigid_matrix(matrix)
    return np.asarray(points, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]

def frame_from_three_points(points: Any) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape != (3, 3) or not np.isfinite(points).all():
        raise ValueError('Yuzey icin uc sonlu nokta secilmeli')
    x = points[1] - points[0]
    side = points[2] - points[0]
    if np.linalg.norm(x) < 0.01 or np.linalg.norm(side) < 0.01:
        raise ValueError('Noktalar birbirinden en az 1 cm uzakta olmali')
    x /= np.linalg.norm(x)
    z = np.cross(x, side)
    if np.linalg.norm(z) / np.linalg.norm(side) < 0.02:
        raise ValueError('Uc nokta ayni dogru uzerinde olamaz')
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack((x, y, z))
    matrix[:3, 3] = -matrix[:3, :3] @ points[0]
    return rigid_matrix(matrix)

def rotation_about(angles_deg: Any, pivot: Any) -> np.ndarray:
    angles = np.asarray(angles_deg, dtype=float)
    pivot = np.asarray(pivot, dtype=float)
    if angles.shape != (3,) or pivot.shape != (3,) or (not np.isfinite([angles, pivot]).all()):
        raise ValueError('XYZ acilari ve donus merkezi sonlu olmali')
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pivot - rotation @ pivot
    return matrix

def snap_angle(angle_deg: float, step_deg: float) -> float:
    if not np.isfinite([angle_deg, step_deg]).all() or step_deg <= 0:
        raise ValueError('Aci adimi sonlu ve sifirdan buyuk olmali')
    return float(np.sign(angle_deg) * np.floor(abs(angle_deg) / step_deg + 0.5) * step_deg)

def rotate_surface(surface, angle_deg, axis='v', step_deg=None):
    from scipy.spatial.transform import Rotation
    if axis not in ('u', 'v', 'normal'):
        raise ValueError('Yuzey donus ekseni u, v veya normal olmali')
    angle = snap_angle(angle_deg, step_deg) if step_deg is not None else angle_deg
    if not np.isfinite(angle):
        raise ValueError('Donus acisi sonlu olmali')
    vector = surface.normal if axis == 'normal' else np.asarray(getattr(surface, axis))
    rotation = Rotation.from_rotvec(vector * np.deg2rad(angle)).as_matrix()
    return replace(surface, u=tuple(rotation @ surface.u), v=tuple(rotation @ surface.v))

@dataclass(frozen=True)
class ProjectionSurface:
    name: str
    origin: tuple[float, float, float]
    u: tuple[float, float, float]
    v: tuple[float, float, float]
    depth_min: float | None = None
    depth_max: float | None = None
    resolution_m: float = 0.02

    def __post_init__(self) -> None:
        if not self.name or not self.name.isascii():
            raise ValueError('Yuzey adi bos olmayan ASCII metin olmali')
        vectors = np.asarray((self.origin, self.u, self.v), dtype=float)
        if vectors.shape != (3, 3) or not np.isfinite(vectors).all():
            raise ValueError('Yuzey eksenleri sonlu uc boyutlu vektor olmali')
        u, v = vectors[1:]
        if not (np.isclose(u @ u, 1) and np.isclose(v @ v, 1) and (abs(u @ v) < 1e-06)):
            raise ValueError('Yuzey eksenleri dik ve birim uzunlukta olmali')
        for value in (self.depth_min, self.depth_max):
            if value is not None and (not np.isfinite(value)):
                raise ValueError('Derinlik sonlu olmali')
        if self.depth_min is not None and self.depth_max is not None and (self.depth_min >= self.depth_max):
            raise ValueError('Derinlik baslangici bitisten kucuk olmali')
        if not np.isfinite(self.resolution_m) or self.resolution_m < 0.001:
            raise ValueError('Izdusum cozunurlugu en az 1 mm olmali')

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.u, self.v)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> ProjectionSurface:
        return cls(**payload)

    @classmethod
    def from_points(cls, name: str, points: Any, **kwargs: Any) -> ProjectionSurface:
        frame = frame_from_three_points(points)
        return cls(name, tuple(points[0]), tuple(frame[0, :3]), tuple(frame[1, :3]), **kwargs)

    @classmethod
    def axis_plane(cls, name: str, axis: str, offset: float, **kwargs: Any) -> ProjectionSurface:
        if axis == 'XY':
            return cls(name, (0, 0, offset), (1, 0, 0), (0, 1, 0), **kwargs)
        if axis == 'XZ':
            return cls(name, (0, offset, 0), (1, 0, 0), (0, 0, 1), **kwargs)
        if axis == 'YZ':
            return cls(name, (offset, 0, 0), (0, 1, 0), (0, 0, 1), **kwargs)
        raise ValueError('Yuzey XY, XZ veya YZ olmali')

def project_surface(points: Any, rgb: Any, surface: ProjectionSurface) -> dict:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or (not np.isfinite(points).all()):
        raise ValueError('Noktalar sonlu Nx3 dizi olmali')
    relative = points - surface.origin
    depth = relative @ surface.normal
    mask = np.ones(len(points), dtype=bool)
    if surface.depth_min is not None:
        mask &= depth >= surface.depth_min
    if surface.depth_max is not None:
        mask &= depth <= surface.depth_max
    selected = np.flatnonzero(mask)
    if not len(selected):
        raise ValueError(f'{surface.name}: secilen derinlikte nokta yok')
    uv = relative[selected] @ np.column_stack((surface.u, surface.v))
    cells = np.floor(uv / surface.resolution_m + 1e-09).astype(np.int64)
    order = np.argsort(-depth[selected], kind='stable')
    unique, first, counts = np.unique(cells[order], axis=0, return_index=True, return_counts=True)
    winner = order[first]
    indices = selected[winner]
    colors = None
    if rgb is not None:
        rgb = np.asarray(rgb)
        if rgb.shape != points.shape:
            raise ValueError('RGB ile nokta sayisi uyusmuyor')
        colors = rgb[indices]
    return {'uv': uv[winner], 'cells': unique, 'rgb': colors, 'depth': depth[indices], 'source_indices': indices, 'support_counts': counts, 'source_count': len(points), 'selected_count': len(selected)}

def surfaces_in_world(placement: Any, metadata: dict) -> list[ProjectionSurface]:
    config = metadata.get('rolove', {})
    current = placed_matrix(placement, metadata)
    reference = rigid_matrix(config.get('surface_frame', current))
    delta = current @ np.linalg.inv(reference)
    return [replace(surface, origin=tuple(transform_points(surface.origin, delta)), u=tuple(delta[:3, :3] @ surface.u), v=tuple(delta[:3, :3] @ surface.v)) for surface in (ProjectionSurface.from_dict(row) for row in config.get('surfaces', []))]

@dataclass(frozen=True)
class PlaneFit:
    origin: tuple[float, float, float]
    normal: tuple[float, float, float]
    rms_m: float
    n_points: int
    radius_m: float

    @property
    def horizontal(self) -> bool:
        return abs(self.normal[2]) > 0.85

    def project(self, point: Any) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Izdusurulecek nokta sonlu XYZ olmali')
        normal = np.asarray(self.normal)
        return point - (point - self.origin) @ normal * normal

def _pca_plane(points: np.ndarray):
    centroid = points.mean(axis=0)
    centered = points - centroid
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    spread = float(np.sqrt(max(values[1], 0.0)))
    if spread < 0.03:
        return None
    normal = vectors[:, 0] / np.linalg.norm(vectors[:, 0])
    return (centroid, normal, float(np.sqrt(max(values[0], 0.0))), spread)

def fit_plane_at(points: Any, pick: Any, *, radius_m: float=0.6, min_points: int=120, fallback_radius_m: float=1.2, max_rms_m: float=0.08) -> PlaneFit:
    points = np.asarray(points, dtype=float)
    pick = np.asarray(pick, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or (not np.isfinite(points).all()):
        raise ValueError('Noktalar sonlu Nx3 dizi olmali')
    if pick.shape != (3,) or not np.isfinite(pick).all():
        raise ValueError('Tiklanan nokta sonlu XYZ olmali')
    if min_points < 12 or radius_m <= 0 or fallback_radius_m < radius_m:
        raise ValueError('Yaricap pozitif ve en az 12 nokta istenmeli')
    distance = np.linalg.norm(points - pick, axis=1)
    radius = radius_m
    within = np.flatnonzero(distance <= radius)
    if len(within) < min_points:
        radius = fallback_radius_m
        within = np.flatnonzero(distance <= radius)
    if len(within) < min_points:
        raise ValueError(f'Tiklanan noktanin {radius:g} m cevresinde {len(within)} nokta var; en az {min_points} gerekli. Daha yogun bir yuzeye tiklayin.')
    local = points[within]
    order = np.argsort(distance[within], kind='stable')

    def grow(seed):
        fit = _pca_plane(seed)
        count = 0
        for _ in range(3):
            if fit is None:
                return None
            centroid, normal, rms, _spread = fit
            tolerance = min(max(3 * rms, 0.02), 0.05)
            keep = np.abs((local - centroid) @ normal) <= tolerance
            count = int(keep.sum())
            if count < min_points:
                return None
            fit = _pca_plane(local[keep])
        return None if fit is None else (*fit, count)
    result = grow(local[order[:max(min_points // 2, 24)]]) or grow(local)
    if result is None:
        raise ValueError('Tiklanan cevrede tek bir duzlem bulunamadi; yuzeyin ortasina tiklayin.')
    centroid, normal, rms, spread, count = result
    if rms > max_rms_m or rms > 0.2 * spread:
        raise ValueError(f'Tiklanan bolge duzlem degil (rms {rms * 100:.1f} cm). Kose ve kenardan uzak, yuzeyin ortasina tiklayin.')
    origin = pick - (pick - centroid) @ normal * normal
    return PlaneFit(tuple(origin), tuple(normal), rms, count, radius)

def fit_surface_at(points: Any, pick: Any, *, eye: Any, screen_up: Any, radius_m: float=0.6, min_points: int=120, fallback_radius_m: float=1.2, max_rms_m: float=0.08, name: str='Yuzey', **kwargs: Any) -> tuple[ProjectionSurface, PlaneFit]:
    fit = fit_plane_at(points, pick, radius_m=radius_m, min_points=min_points, fallback_radius_m=fallback_radius_m, max_rms_m=max_rms_m)
    eye = np.asarray(eye, dtype=float)
    screen_up = np.asarray(screen_up, dtype=float)
    if eye.shape != (3,) or screen_up.shape != (3,) or (not np.isfinite([eye, screen_up]).all()):
        raise ValueError('Kamera konumu ve ekran yukarisi sonlu XYZ olmali')
    normal = np.asarray(fit.normal)
    if (eye - fit.origin) @ normal < 0:
        normal = -normal
    if abs(normal[2]) > 0.85:
        candidates = (screen_up, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    else:
        candidates = (np.array([0.0, 0.0, 1.0]),)
    v = None
    for candidate in candidates:
        projected = candidate - candidate @ normal * normal
        if np.linalg.norm(projected) > 1e-06:
            v = projected / np.linalg.norm(projected)
            break
    if v is None:
        raise ValueError('Ekran yukarisi duzleme dik; kamerayi biraz dondurun')
    u = np.cross(v, normal)
    u /= np.linalg.norm(u)
    surface = ProjectionSurface(name, fit.origin, tuple(u), tuple(v), **kwargs)
    return (surface, replace(fit, normal=tuple(normal)))
