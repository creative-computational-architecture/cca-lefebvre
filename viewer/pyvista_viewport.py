
from __future__ import annotations
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import numpy as np
from geometry.rolove import placed_matrix, surfaces_in_world
from viewer.brand import SKIN_C_DARK
from viewer.camera_controller import CameraController, CameraState
from viewer.colour_gap import COLOUR_GAP_DISPLAY_HEX, COLOUR_GAP_DISPLAY_RGB, COLOUR_GAP_FIELD, COLOUR_GAP_PLACEHOLDER_RGB, COLOUR_GAP_SCALAR, COLOUR_GAP_UI_LABEL, COLOUR_GAP_UNASSIGNED
from viewer.colour_gap import apply_colour_gap_display as _apply_colour_gap_display
from viewer.colour_gap import colour_gap_column as _colour_gap_column
from viewer.depth_fill import DEPTH_FILL_FIELD, DEPTH_FILL_LEGEND_SHARE, DEPTH_FILL_SCALAR, DEPTH_FILL_UI_LABEL
from viewer.depth_fill import apply_depth_fill_display as _apply_depth_fill_display
from viewer.depth_fill import depth_fill_column as _depth_fill_column
from viewer.depth_fill import labelled_count as _depth_fill_labelled_count
from viewer.depth_fill import validated_fill_share as _validated_fill_share
from viewer.evidence_grade import CONF_MEAN_FIELD, N_VIEWS_FIELD, EvidenceGradeError, EvidenceGradeResult, EvidenceGradeState, EvidenceGradeSummary, display_alpha, resolve_grade, summarize_grade
from viewer.floating_votes import DEFAULT_FLOATING_THRESHOLD, FLOATING_THRESHOLD_MAX, FLOATING_THRESHOLD_MIN, load_floating_votes
from viewer.floating_votes import floating_hidden as _floating_hidden
from viewer.layer_controller import ActorRequest, ColorStrategy, DisplayMode, DisplayModeUnavailableError, LayerControllerError, RenderPlan, resolve_display_mode, resolve_render_plan
from viewer.point_display import FLOATING_UI_LABEL, FLOATING_UNAVAILABLE_REASON, SKY_FILL_UI_LABEL, SKY_FILL_UNAVAILABLE_REASON, SPARSE_UI_LABEL, SPARSE_UNAVAILABLE_REASON, SURFACE_SUPPORT_SIZE_NOTICE, SizeBucket
from viewer.point_display import size_buckets as _size_buckets
from viewer.point_display import sky_fill_hidden as _sky_fill_hidden
from viewer.project_model import ProjectModel
from viewer.qt_shell import QtModules
from viewer.sparse_votes import DEFAULT_SPARSE_THRESHOLD_M, SPARSE_THRESHOLD_MAX_M, SPARSE_THRESHOLD_MIN_M, load_sparse_sidecar, load_sparse_stats
from viewer.sparse_votes import sparse_hidden as _sparse_hidden

def _run_sparse_default(source: Path) -> float | None:
    stats = load_sparse_stats(source)
    return None if stats is None else stats.default_threshold_m
PLAN_COLOR = '#AEBAC0'
POINT_FALLBACK_COLOR = SKIN_C_DARK['muted']
VIEWPORT_BACKGROUND = '#0B1016'
DEFAULT_POINT_SIZE = 4.0
SCAN_LAYOUT_GAP_M = 1.0
GHOST_SURFACE_OPACITY = 0.22
SOLID_SURFACE_OPACITY = 0.78
PLAN_OPACITY = 0.22
PLAN_LINE_WIDTH = 0.75
EVIDENCE_GRADE_COLORMAP = 'plasma'
EVIDENCE_GRADE_SCALAR = 'cca_evidence_grade'
RECOVERY_SHARE_FIELD = 'recovery_share'
SURFACE_DISPLAY_MODES = ('POINTS ONLY', 'GHOST SURFACE', 'SOLID SURFACE')
TRANSFORM_MODES = ('SELECT', 'MOVE', 'ROTATE Z')
MESH_SOURCE_SUFFIXES = frozenset({'.obj', '.stl'})
MOVE_CONSTRAINTS = ('XY', 'X', 'Y', 'Z')
GIZMO_AXIS_COLORS = {'X': '#E05252', 'Y': '#5CC76B', 'Z': '#4C87FF'}

def _packaged_mesh_texture(source: Path) -> Path | None:
    if source.name != 'textured_output.obj':
        return None
    texture = source.with_suffix('.jpg')
    return texture if texture.is_file() else None
GIZMO_CENTER_COLOR = '#F2F2F2'
GIZMO_ROTATE_COLOR = '#F0B44D'
SELECTION_COLOR = '#F28C28'

class PyVistaViewportError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class LoadedScan:
    scan_id: str
    actor_name: str
    source_path: Path
    source_point_count: int
    displayed_point_count: int
    band: str = 'core'
    display_mode: DisplayMode = DisplayMode.PHOTO_RGB
    representation: str = 'POINTS'
    triangle_count: int = 0
    placement_cell_size_m: float | None = None

@dataclass(frozen=True, slots=True)
class _PlacementGeometry:
    key: tuple
    vertices: np.ndarray
    colors: np.ndarray | None
    faces: np.ndarray | None
    source: Path
    source_count: int
    representation: str
    bounds: tuple[tuple[float, ...], tuple[float, ...]]
    texture_coordinates: np.ndarray | None = None
    texture_rgb: np.ndarray | None = None
    cell_size_m: float | None = None
    colour_gap_count: int | None = None
    colour_gap_total: int | None = None
    grade: np.ndarray | None = None
    cell_grade: np.ndarray | None = None
    texture_grade: np.ndarray | None = None
    texture_covered: np.ndarray | None = None
    range_summary: dict[str, float] | None = None
    depth_fill_count: int | None = None
    depth_fill_total: int | None = None
    depth_fill_labelled: np.ndarray | None = None
    sky_hidden: np.ndarray | None = None
    floating_through: np.ndarray | None = None
    floating_confirm: np.ndarray | None = None
    sparse_kdist: np.ndarray | None = None
    sparse_default_m: float | None = None
    cell_sky_hidden: np.ndarray | None = None
    patch_sky_hidden: np.ndarray | None = None
    cell_floating_t: np.ndarray | None = None
    patch_floating_t: np.ndarray | None = None
    cell_sparse_m: np.ndarray | None = None
    patch_sparse_m: np.ndarray | None = None
    surface_support: np.ndarray | None = None

@dataclass(frozen=True, slots=True)
class DrawnActor:
    name: str
    actor: Any
    kind: str = 'POINTS'
    size_factor: float = 1.0
    select: Any = None

@dataclass(frozen=True, slots=True)
class SkyFillSummary:
    point_count: int
    hidden_count: int
    visible: bool
    ui_label: str = SKY_FILL_UI_LABEL

    @property
    def hidden_fraction(self) -> float:
        return self.hidden_count / self.point_count if self.point_count else 0.0

@dataclass(frozen=True, slots=True)
class FloatingSummary:
    point_count: int
    hidden_count: int
    visible: bool
    threshold: float = DEFAULT_FLOATING_THRESHOLD
    ui_label: str = FLOATING_UI_LABEL

    @property
    def hidden_fraction(self) -> float:
        return self.hidden_count / self.point_count if self.point_count else 0.0

@dataclass(frozen=True, slots=True)
class SparseSummary:
    point_count: int
    hidden_count: int
    visible: bool
    threshold_m: float = DEFAULT_SPARSE_THRESHOLD_M
    ui_label: str = SPARSE_UI_LABEL

    @property
    def hidden_fraction(self) -> float:
        return self.hidden_count / self.point_count if self.point_count else 0.0
RANGE_FIELD = 'range_m'

def _range_summary(values: np.ndarray) -> dict[str, float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    return {'median_m': float(np.median(finite)), 'p99_m': float(np.percentile(finite, 99.0)), 'max_m': float(finite.max()), 'count': float(len(finite))}

def _atlas_coverage_mask(texture_coordinates: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = (int(shape[0]), int(shape[1]))
    flat = np.asarray(texture_coordinates, dtype=np.float64).reshape(-1, 2)
    if flat.size == 0 or len(flat) % 4:
        return np.ones((height, width), dtype=bool)
    uv = flat.reshape(-1, 4, 2)
    col0 = np.clip(np.floor(uv[:, :, 0].min(axis=1) * width), 0, width).astype(np.int64)
    col1 = np.clip(np.ceil(uv[:, :, 0].max(axis=1) * width), 0, width).astype(np.int64)
    row0 = np.clip(np.floor((1.0 - uv[:, :, 1].max(axis=1)) * height), 0, height).astype(np.int64)
    row1 = np.clip(np.ceil((1.0 - uv[:, :, 1].min(axis=1)) * height), 0, height).astype(np.int64)
    delta = np.zeros((height + 1, width + 1), dtype=np.int64)
    np.add.at(delta, (row0, col0), 1)
    np.add.at(delta, (row0, col1), -1)
    np.add.at(delta, (row1, col0), -1)
    np.add.at(delta, (row1, col1), 1)
    return delta.cumsum(axis=0).cumsum(axis=1)[:height, :width] > 0

@dataclass(frozen=True, slots=True)
class ColourGapSummary:
    point_count: int
    gap_count: int
    display_hex: str = COLOUR_GAP_DISPLAY_HEX

    @property
    def gap_fraction(self) -> float:
        return self.gap_count / self.point_count if self.point_count else 0.0

@dataclass(frozen=True, slots=True)
class DepthFillSummary:
    point_count: int
    labelled_count: int
    legend_share: float = DEPTH_FILL_LEGEND_SHARE
    ui_label: str = DEPTH_FILL_UI_LABEL

    @property
    def labelled_fraction(self) -> float:
        return self.labelled_count / self.point_count if self.point_count else 0.0

@dataclass(frozen=True, slots=True)
class PlacementDensityFallback:
    scan_id: str
    requested_cca: float
    source_point_count: int
    reason: str

    def message(self) -> str:
        return f'AYRINTI DUSURULDU / {self.scan_id} / istenen {self.requested_cca:g} CCA/m bu kaynaga artik sigmiyor ({self.source_point_count:,} nokta; ayar daha kucuk bir kaynak icin secilmisti) / otomatik ayrinti cizildi / proje ayari degistirilmedi, Uygula ile yeniden denenebilir. {self.reason}'
POINTS_BYTES_PER_POINT = 96
POINTS_FIXED_HEADROOM_BYTES = 128 * 1024 * 1024

@dataclass(frozen=True, slots=True)
class DetailCapacitySkip:
    scan_id: str
    source_point_count: int
    display_point_count: int
    needed_bytes: int
    remaining_bytes: int

    def message(self) -> str:
        return f'{self.scan_id}: {self.display_point_count:,} nokta icin ~{self.needed_bytes / 1024 ** 3:.2f} GB gerekiyor, sirasi geldiginde kalan guvenli butce {max(self.remaining_bytes, 0) / 1024 ** 3:.2f} GB / hafif temsil olarak kaldi, kaynak noktalar degismedi'

@dataclass(frozen=True, slots=True)
class DetailCapacityPlan:
    opened: tuple[str, ...]
    skipped: tuple[DetailCapacitySkip, ...]
    budget_bytes: int | None

@dataclass(frozen=True, slots=True)
class _PreparedRenderPlan:
    project_id: str
    detailed_scan_ids: frozenset[str]
    maximum: int | None
    geometries: tuple[tuple[str, _PlacementGeometry], ...]
    density_fallbacks: tuple[PlacementDensityFallback, ...] = ()

@dataclass(frozen=True, slots=True)
class ProjectLoadResult:
    scans: tuple[LoadedScan, ...]
    plan_actor_name: str | None
    plan_segment_count: int
    density_fallbacks: tuple[PlacementDensityFallback, ...] = ()

    @property
    def scan_count(self) -> int:
        return len({scan.scan_id for scan in self.scans})

    @property
    def displayed_point_count(self) -> int:
        return sum((scan.displayed_point_count for scan in self.scans))

@dataclass(frozen=True, slots=True)
class ManualTransformEvent:
    scan_id: str
    phase: str
    mode: str
    transform: tuple[tuple[float, ...], ...]
    original_transform: tuple[tuple[float, ...], ...]
    target_kind: str = 'scan'

@dataclass(slots=True)
class _ManualDrag:
    scan_id: str
    mode: str
    constraint: str
    start_position: tuple[float, float]
    original_transform: np.ndarray
    preview_transform: np.ndarray
    pivot: np.ndarray
    start_xy: np.ndarray | None = None
    start_angle: float | None = None
    target_kind: str = 'scan'

def _load_renderer_dependencies() -> tuple[Any, Any, Any]:
    try:
        import pyvista
        import vtk
        from pyvistaqt import QtInteractor
    except (ImportError, ModuleNotFoundError) as exc:
        raise PyVistaViewportError('PyVista, PyVistaQt, and VTK are required for the viewport') from exc
    return (pyvista, vtk, QtInteractor)

def _resolve_project_path(project: ProjectModel, value: str | None, label: str) -> Path:
    if not value:
        raise PyVistaViewportError(f'{label} path is missing')
    source = Path(value)
    if not source.is_absolute():
        if project.source_manifest:
            source = Path(project.source_manifest).expanduser().resolve().parent / source
        else:
            source = Path.cwd() / source
    source = source.expanduser().resolve()
    if not source.is_file():
        raise PyVistaViewportError(f'{label} does not exist: {source}')
    return source

def _rgb_values(dataset: Any, expected_count: int) -> np.ndarray | None:
    names = {str(name).lower(): str(name) for name in dataset.point_data}
    candidate: np.ndarray | None = None
    for name in ('rgb', 'rgba', 'colors', 'color'):
        actual = names.get(name)
        if actual is None:
            continue
        values = np.asarray(dataset.point_data[actual])
        if values.ndim == 2 and values.shape[0] == expected_count and (values.shape[1] >= 3):
            candidate = values[:, :3]
            break
    if candidate is None and all((channel in names for channel in ('red', 'green', 'blue'))):
        candidate = np.column_stack([np.asarray(dataset.point_data[names[channel]]) for channel in ('red', 'green', 'blue')])
    if candidate is None or candidate.shape != (expected_count, 3):
        return None
    values = np.asarray(candidate, dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    if values.size and float(values.max()) <= 1.0:
        values = values * 255.0
    return np.clip(np.rint(values), 0.0, 255.0).astype(np.uint8)

def _sample_indices(point_count: int, maximum: int | None) -> np.ndarray:
    if maximum is None or point_count <= maximum:
        return np.arange(point_count, dtype=np.int64)
    return np.linspace(0, point_count - 1, maximum, dtype=np.int64)

def _evidence_schema_columns(schema: Any) -> dict[str, str]:
    names = {prop.name.lower(): prop.name for prop in schema.vertex.properties}
    return {field: names[field] for field in (CONF_MEAN_FIELD, N_VIEWS_FIELD) if field in names}

def _read_evidence_columns(source: Path, columns: Mapping[str, str]) -> dict[str, np.ndarray]:
    from geometry.ply_preserve import iter_ply_vertex_chunks
    parts: dict[str, list[np.ndarray]] = {field: [] for field in columns}
    for chunk in iter_ply_vertex_chunks(source):
        for field, prop_name in columns.items():
            parts[field].append(np.asarray(chunk[prop_name], dtype=np.float64))
    return {field: np.concatenate(values) if values else np.empty(0, dtype=np.float64) for field, values in parts.items()}

@dataclass(frozen=True, slots=True)
class _SourceDisplayEvidence:
    grade: EvidenceGradeResult
    gap: np.ndarray | None
    range_summary: dict[str, float] | None
    fill_share: np.ndarray | None

def _effective_fill_share(fill_share: np.ndarray, recovery_share: np.ndarray | None=None, fill_recovery_share: np.ndarray | None=None) -> np.ndarray:
    fill = _validated_fill_share(fill_share).astype(np.float32)
    if fill_recovery_share is not None:
        if recovery_share is None:
            raise ValueError('fill_recovery_share requires recovery_share')
        recovery = _validated_fill_share(recovery_share)
        intersection = _validated_fill_share(fill_recovery_share)
        if np.any(intersection > fill + 1e-06) or np.any(intersection > recovery + 1e-06):
            raise ValueError('fill_recovery_share must not exceed fill_share or recovery_share')
        return np.maximum(0.0, fill - intersection).astype(np.float32)
    if recovery_share is not None:
        recovery = _validated_fill_share(recovery_share)
        return np.minimum(fill, 1.0 - recovery).astype(np.float32)
    return fill

def _resolve_source_display(source: Path, point_count: int) -> _SourceDisplayEvidence:
    from geometry.ply_preserve import read_ply_schema
    schema = read_ply_schema(source)
    if schema.vertex.count != point_count:
        return _SourceDisplayEvidence(EvidenceGradeResult(available=False, unavailable_reason='evidence column count does not match point geometry', grade=None, point_count=point_count), None, None, None)
    columns = dict(_evidence_schema_columns(schema))
    gap_column = _colour_gap_column(schema)
    if gap_column is not None:
        columns[COLOUR_GAP_FIELD] = gap_column
    fill_column = _depth_fill_column(schema)
    if fill_column is not None:
        columns[DEPTH_FILL_FIELD] = fill_column
    recovery_column = next((prop.name for prop in schema.vertex.properties if prop.name.lower() == RECOVERY_SHARE_FIELD), None)
    if recovery_column is not None:
        columns[RECOVERY_SHARE_FIELD] = recovery_column
    fill_recovery_column = next((prop.name for prop in schema.vertex.properties if prop.name.lower() == 'fill_recovery_share'), None)
    if fill_recovery_column is not None:
        columns['fill_recovery_share'] = fill_recovery_column
    range_column = next((prop.name for prop in schema.vertex.properties if prop.name.lower() == RANGE_FIELD), None)
    if range_column is not None:
        columns[RANGE_FIELD] = range_column
    scalars: dict[str, np.ndarray] = {'__point_count__': np.empty(point_count)}
    gap: np.ndarray | None = None
    fill: np.ndarray | None = None
    range_summary: dict[str, float] | None = None
    if columns:
        values = _read_evidence_columns(source, columns)
        if gap_column is not None:
            gap = values.pop(COLOUR_GAP_FIELD) == COLOUR_GAP_UNASSIGNED
        if fill_column is not None:
            recovery = values.pop(RECOVERY_SHARE_FIELD, None)
            fill_recovery = values.pop('fill_recovery_share', None)
            fill = _effective_fill_share(values[DEPTH_FILL_FIELD], recovery, fill_recovery)
            values[DEPTH_FILL_FIELD] = fill
        else:
            values.pop(RECOVERY_SHARE_FIELD, None)
            values.pop('fill_recovery_share', None)
        if range_column is not None:
            range_summary = _range_summary(values.pop(RANGE_FIELD))
        scalars.update(values)
    return _SourceDisplayEvidence(resolve_grade(scalars), gap, range_summary, fill)

def _resolve_source_evidence(source: Path, point_count: int) -> tuple[EvidenceGradeResult, np.ndarray | None, dict[str, float] | None]:
    evidence = _resolve_source_display(source, point_count)
    return (evidence.grade, evidence.gap, evidence.range_summary)

def _resolve_evidence_grade(source: Path, point_count: int) -> EvidenceGradeResult:
    return _resolve_source_evidence(source, point_count)[0]

def _sample_grade(result: EvidenceGradeResult, selected: np.ndarray) -> EvidenceGradeResult:
    if not result.available:
        return EvidenceGradeResult(available=False, unavailable_reason=result.unavailable_reason, grade=None, point_count=int(len(selected)))
    grade = result.grade[selected]
    return EvidenceGradeResult(available=True, unavailable_reason=None, grade=grade, point_count=int(len(grade)))

def _validated_maximum(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PyVistaViewportError('max_points_per_scan must be a positive integer')
    return value

def _source_vertex_count(source: Path) -> int:
    from geometry.ply_preserve import read_ply_schema
    try:
        return int(read_ply_schema(source).vertex.count)
    except (OSError, ValueError):
        return 0

def _matrix(value: object, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PyVistaViewportError(f'{label} transform is invalid') from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise PyVistaViewportError(f'{label} transform is invalid')
    return matrix

def _polydata_bounds(dataset: Any) -> tuple[float, float, float, float, float, float]:
    bounds = tuple((float(value) for value in dataset.bounds))
    if len(bounds) != 6 or not all((math.isfinite(value) for value in bounds)):
        raise PyVistaViewportError('render geometry has invalid bounds')
    return bounds

def points_to_working_axis(points: np.ndarray, source_axis: str) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or (not np.isfinite(values).all()):
        raise PyVistaViewportError('point geometry must be finite XYZ')
    if source_axis == 'CAD_Z_up':
        return values.copy()
    if source_axis == 'ARKit_Y_up':
        return np.column_stack((values[:, 0], -values[:, 2], values[:, 1]))
    raise PyVistaViewportError(f'unsupported source axis: {source_axis}')

class PyVistaViewportAdapter:

    def __init__(self, controller: CameraController | None=None, *, off_screen: bool=False, point_size: float=DEFAULT_POINT_SIZE) -> None:
        if isinstance(point_size, bool) or not isinstance(point_size, (int, float)):
            raise PyVistaViewportError('point_size must be positive')
        if not math.isfinite(float(point_size)) or float(point_size) <= 0.0:
            raise PyVistaViewportError('point_size must be positive')
        self.controller = controller or CameraController()
        self.off_screen = bool(off_screen)
        self.point_size = float(point_size)
        self._plotter: Any | None = None
        self._widget: Any | None = None
        self._view_cube: Any | None = None
        self._pyvista: Any | None = None
        self._vtk: Any | None = None
        self._scan_actors: dict[str, Any] = {}
        self._layer_actors: dict[str, Any] = {}
        self._actor_groups: dict[str, tuple[DrawnActor, ...]] = {}
        self._actor_requests: dict[str, ActorRequest] = {}
        self._actor_base_transforms: dict[str, np.ndarray] = {}
        self._scan_display_transforms: dict[str, np.ndarray] = {}
        self._surface_actors: dict[str, Any] = {}
        self._surface_base_transforms: dict[str, np.ndarray] = {}
        self._scan_bounds: dict[str, tuple[float, float, float, float, float, float]] = {}
        self._actor_bounds: dict[str, tuple[float, float, float, float, float, float]] = {}
        self._plan_actor: Any | None = None
        self._plan_actor_name: str | None = None
        self._plan_base_transform: np.ndarray | None = None
        self._scan_layout_label_actor_name: str | None = None
        self._selected_scan_id: str | None = None
        self._selected_plan = False
        self._selection_cleared_callback = None
        self._right_click_callback = None
        self._middle_button_mode = 'orbit'
        self._pickbox_px = 8
        self._selection_box_enabled = True
        self._plan_selection_callback: Callable[[], None] | None = None
        self._selection_highlight_enabled = True
        self._selection_box_actors: dict[str, Any] = {}
        self._selection_actor_state: dict[str, tuple[int, tuple[float, float, float]]] = {}
        self._transform_gizmo_actors: dict[str, Any] = {}
        self._surface_display_mode = 'GHOST SURFACE'
        self._color_mode = 'rgb'
        self._drag_position: tuple[float, float] | None = None
        self._transform_mode = 'SELECT'
        self._orbit_mode = False
        self._orbit_mode_return = 'SELECT'
        self._move_constraint = 'XY'
        self._manual_drag: _ManualDrag | None = None
        self._manual_transform_callback: Callable[[ManualTransformEvent], None] | None = None
        self._scan_selection_callback: Callable[[str], None] | None = None
        self._selection_provider: Callable[[], Sequence[str]] | None = None
        self._camera_changed_callback: Callable[[CameraState], None] | None = None
        self._transform_mode_callback: Callable[[str, str], None] | None = None
        self._last_load_result: ProjectLoadResult | None = None
        self._render_plan: RenderPlan | None = None
        self._project: ProjectModel | None = None
        self._maximum_points_per_actor: int | None = None
        self._plan_segment_count = 0
        self._rolove_pick_callback = None
        self._rolove_pick_scan_id = None
        self._rolove_actor_names: list[str] = []
        self._rolove_surface_scan_id = None
        self._placement_cache_root: Path | None = None
        self._detailed_scan_ids: frozenset[str] = frozenset()
        self._placement_geometry_cache: dict[tuple, _PlacementGeometry] = {}
        self._placement_actor_geometry: dict[str, _PlacementGeometry] = {}
        self._placement_generation = 0
        self._placement_cca_by_scan: dict[str, float] = {}
        self._placement_cca_fallbacks: dict[str, PlacementDensityFallback] = {}
        self._placement_actor_grade: dict[str, tuple[float | None, bool | None, tuple[bool, float] | None, tuple[bool, float] | None]] = {}
        self._sky_fill_visible = False
        self._sky_fill_counts: dict[str, tuple[int, int]] = {}
        self._floating_visible = False
        self._floating_threshold = DEFAULT_FLOATING_THRESHOLD
        self._floating_counts: dict[str, tuple[int, int]] = {}
        self._floating_mask_cache: dict[tuple, np.ndarray] = {}
        self._sparse_visible = False
        self._sparse_threshold = DEFAULT_SPARSE_THRESHOLD_M
        self._sparse_counts: dict[str, tuple[int, int]] = {}
        self._sparse_mask_cache: dict[tuple, np.ndarray] = {}
        self._sparse_defaults: dict[str, float] = {}
        self._evidence_grade_state = EvidenceGradeState()
        self._evidence_grade_results: dict[str, EvidenceGradeResult] = {}
        self._colour_gap_counts: dict[str, tuple[int, int]] = {}
        self._depth_fill_counts: dict[str, tuple[int, int]] = {}
        self._range_summaries: dict[str, dict[str, float] | None] = {}

    @property
    def placement_enabled(self) -> bool:
        return self._placement_cache_root is not None

    @property
    def detailed_scan_ids(self) -> frozenset[str]:
        return self._detailed_scan_ids

    def configure_placement_mode(self, cache_root: Path) -> None:
        self._placement_cache_root = Path(cache_root).expanduser().resolve()
        self._detailed_scan_ids = frozenset()

    def is_scan_detailed(self, scan_id: str) -> bool:
        return not self.placement_enabled or scan_id in self._detailed_scan_ids

    def set_placement_quality(self, values: Mapping[str, float]) -> None:
        clean = {}
        for scan_id, value in values.items():
            if not isinstance(scan_id, str) or isinstance(value, bool) or (not isinstance(value, (float, int))) or (not math.isfinite(value)) or (value <= 0):
                raise PyVistaViewportError('CCA degeri pozitif ve sonlu olmali.')
            clean[scan_id] = float(value)
        self._placement_cca_by_scan = clean
        self._placement_cca_fallbacks = {scan_id: fallback for scan_id, fallback in self._placement_cca_fallbacks.items() if clean.get(scan_id) == fallback.requested_cca}

    def retry_placement_density(self, scan_ids) -> None:
        for scan_id in scan_ids:
            self._placement_cca_fallbacks.pop(scan_id, None)

    @property
    def placement_density_fallbacks(self) -> tuple[PlacementDensityFallback, ...]:
        return tuple((self._placement_cca_fallbacks[key] for key in sorted(self._placement_cca_fallbacks)))

    def _effective_placement_cca(self, scan_id: str) -> float | None:
        requested = self._placement_cca_by_scan.get(scan_id)
        fallback = self._placement_cca_fallbacks.get(scan_id)
        if requested is None or (fallback is not None and fallback.requested_cca == requested):
            return None
        return requested

    def scan_view_distances(self, scan_ids) -> dict[str, float]:
        pivot = np.asarray(self.controller.state.pivot, dtype=np.float64)
        distances: dict[str, float] = {}
        for scan_id in scan_ids:
            bounds = self._scan_bounds.get(scan_id)
            if bounds is None:
                continue
            center = np.asarray([(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0])
            if not np.all(np.isfinite(center)):
                continue
            distances[scan_id] = float(np.linalg.norm(center - pivot))
        return distances

    def _source_count_for_request(self, request: ActorRequest) -> int:
        geometry = self._placement_actor_geometry.get(request.actor_id)
        if geometry is not None:
            return int(geometry.source_count)
        if request.source_path is None:
            return 0
        return _source_vertex_count(request.source_path.resolve())

    def plan_detail_capacity(self, plan: RenderPlan, ordered_scan_ids) -> DetailCapacityPlan:
        from viewer.placement_proxy import _detail_memory_budget
        if not self.placement_enabled:
            raise PyVistaViewportError('placement mode is not configured')
        maximum = self._maximum_points_per_actor
        budget = _detail_memory_budget()
        remaining = None if budget is None else budget - POINTS_FIXED_HEADROOM_BYTES
        by_scan: dict[str, list[ActorRequest]] = {}
        for request in plan.visible_actors:
            if request.available:
                by_scan.setdefault(request.scan_id, []).append(request)
        opened: list[str] = []
        skipped: list[DetailCapacitySkip] = []
        for scan_id in dict.fromkeys(ordered_scan_ids):
            requests = by_scan.get(scan_id)
            if not requests:
                continue
            try:
                resident = all((self._placement_key(request, frozenset({scan_id}), maximum) in self._placement_geometry_cache for request in requests))
            except (OSError, PyVistaViewportError):
                resident = False
            if resident:
                opened.append(scan_id)
                continue
            source_count = display_count = 0
            for request in requests:
                count = self._source_count_for_request(request)
                source_count += count
                display_count += count if maximum is None else min(count, maximum)
            needed = display_count * POINTS_BYTES_PER_POINT
            if remaining is None or needed <= remaining:
                opened.append(scan_id)
                if remaining is not None:
                    remaining -= needed
            else:
                skipped.append(DetailCapacitySkip(scan_id, source_count, display_count, needed, remaining))
        return DetailCapacityPlan(tuple(opened), tuple(skipped), budget)

    def _require_detailed_scan(self, scan_id: str) -> None:
        if not self.is_scan_detailed(scan_id):
            raise PyVistaViewportError("Temsil geometri olcum icin kullanilamaz. Taramayi secip HESAPLA'ya basin.")

    def _placement_key(self, request: ActorRequest, details: frozenset[str], maximum: int | None) -> tuple:
        if not request.available or request.source_path is None:
            raise PyVistaViewportError(f'render source is unavailable: {request.actor_id}')
        source = request.source_path.resolve()
        stat = source.stat()
        representation = 'MESH' if source.suffix.lower() in MESH_SOURCE_SUFFIXES else 'POINTS' if request.scan_id in details else 'PROXY'
        return (str(source), stat.st_size, stat.st_mtime_ns, request.source_axis, representation, maximum if representation == 'POINTS' else self._effective_placement_cca(request.scan_id))

    def render_plan_is_prepared(self, plan: RenderPlan) -> bool:
        if not self.placement_enabled:
            return False
        try:
            return all((self._placement_key(request, self._detailed_scan_ids, self._maximum_points_per_actor) in self._placement_geometry_cache for request in plan.visible_actors))
        except OSError:
            return False

    def prepare_render_plan(self, plan: RenderPlan, detailed_scan_ids: frozenset[str] | None=None, progress: Callable[[int, int, str], None] | None=None) -> _PreparedRenderPlan:
        return self._prepare_placement_plan(plan, self._detailed_scan_ids if detailed_scan_ids is None else frozenset(detailed_scan_ids), self._maximum_points_per_actor, progress=progress)

    def _prepare_placement_plan(self, plan: RenderPlan, details: frozenset[str], maximum: int | None, *, progress: Callable[[int, int, str], None] | None=None) -> _PreparedRenderPlan:
        if not self.placement_enabled:
            raise PyVistaViewportError('placement mode is not configured')
        plan.require_available()
        prepared = []
        pending_geometries = {}
        fallbacks: list[PlacementDensityFallback] = []
        requests = plan.visible_actors
        for index, request in enumerate(requests):
            key = self._placement_key(request, details, maximum)
            geometry = self._placement_geometry_cache.get(key) or pending_geometries.get(key)
            if geometry is None and key[4] == 'PROXY':
                source = request.source_path.resolve()
                geometry, key = self._prepare_proxy_geometry(request, source, key, pending_geometries, fallbacks)
            if geometry is None:
                source = request.source_path.resolve()
                if key[4] == 'MESH':
                    vertices, colors, faces, texture_coordinates, texture_rgb = self._read_placement_mesh(source, request.source_axis)
                    geometry = _PlacementGeometry(key, vertices, colors, faces, source, len(vertices), 'MESH', (tuple(vertices.min(axis=0)), tuple(vertices.max(axis=0))), texture_coordinates=texture_coordinates, texture_rgb=texture_rgb)
                    pending_geometries[key] = geometry
                    prepared.append((request.actor_id, geometry))
                    if progress is not None:
                        progress(index + 1, len(requests), request.scan_id)
                    continue
                points, colors, source_count, gap_count, fill_count, fill_labelled = self._read_placement_points(source, request.source_axis, maximum)
                source_evidence = _resolve_source_display(source, source_count)
                grade_full, range_summary = (source_evidence.grade, source_evidence.range_summary)
                grade_result = _sample_grade(grade_full, _sample_indices(source_count, maximum))
                grade = grade_result.grade if grade_result.available else None
                sky_hidden = _sky_fill_hidden(fill_labelled, grade) if fill_labelled is not None and grade is not None else None
                votes = load_floating_votes(source, source_count)
                floating_through = floating_confirm = None
                if votes is not None:
                    picked = _sample_indices(source_count, maximum)
                    floating_through = votes.n_through[picked]
                    floating_confirm = votes.n_confirm[picked]
                kdist = load_sparse_sidecar(source, source_count)
                sparse_kdist = None if kdist is None else kdist[_sample_indices(source_count, maximum)]
                sparse_stats = None if kdist is None else load_sparse_stats(source, source_count)
                sparse_default_m = None if sparse_stats is None else sparse_stats.default_threshold_m
                from geometry.surface_structure import surface_support
                support = surface_support(points) if source_evidence.fill_share is None else surface_support(points, reference_mask=source_evidence.fill_share[_sample_indices(source_count, maximum)] < 1)
                geometry = _PlacementGeometry(key, points, colors, None, source, source_count, 'POINTS', (tuple(points.min(axis=0)), tuple(points.max(axis=0))), colour_gap_count=gap_count, colour_gap_total=len(points) if gap_count is not None else None, grade=grade, range_summary=range_summary, depth_fill_count=fill_count, depth_fill_total=len(points) if fill_count is not None else None, depth_fill_labelled=fill_labelled, sky_hidden=sky_hidden, floating_through=floating_through, floating_confirm=floating_confirm, sparse_kdist=sparse_kdist, sparse_default_m=sparse_default_m, surface_support=support)
                pending_geometries[key] = geometry
            if request.color.strategy is ColorStrategy.CONFIDENCE_RGB and geometry.colors is None:
                raise PyVistaViewportError(f'confidence RGB is missing: {geometry.source}')
            prepared.append((request.actor_id, geometry))
            if progress is not None:
                progress(index + 1, len(requests), request.scan_id)
        return _PreparedRenderPlan(plan.project_id, details, maximum, tuple(prepared), tuple(fallbacks))

    def _prepare_proxy_geometry(self, request: ActorRequest, source: Path, key: tuple, pending: dict[tuple, _PlacementGeometry], fallbacks: list[PlacementDensityFallback]) -> tuple[_PlacementGeometry, tuple]:
        from viewer.placement_proxy import PlacementProxyCapacityError, load_or_build_proxy
        geometry = self._placement_geometry_cache.get(key) or pending.get(key)
        if geometry is not None:
            return (geometry, key)
        cca = key[5]
        options = {} if cca is None else {'cca_per_m': cca}
        try:
            proxy = load_or_build_proxy(source, request.source_axis, self._placement_cache_root, **options)
        except PlacementProxyCapacityError as error:
            if cca is None:
                raise
            fallbacks.append(PlacementDensityFallback(request.scan_id, cca, _source_vertex_count(source), str(error)))
            key = key[:5] + (None,)
            geometry = self._placement_geometry_cache.get(key) or pending.get(key)
            if geometry is not None:
                return (geometry, key)
            try:
                proxy = load_or_build_proxy(source, request.source_axis, self._placement_cache_root)
            except (PlacementProxyCapacityError, MemoryError) as auto_error:
                raise PlacementProxyCapacityError(f'{request.scan_id}: {cca:g} CCA/m sigmadi ve otomatik ayrinti bile bellege sigmiyor. Diger uygulamalari kapatin. Kaynak noktalar degismedi. {auto_error}') from auto_error
        geometry = self._proxy_geometry(key, proxy, source)
        del proxy
        pending[key] = geometry
        return (geometry, key)

    @staticmethod
    def _proxy_view_states(proxy) -> dict[str, np.ndarray | None]:
        from viewer.placement_proxy import majority_threshold
        patches = len(proxy.vertices) // 4
        states: dict[str, np.ndarray | None] = {'cell_sky_hidden': None, 'patch_sky_hidden': None, 'cell_floating_t': None, 'patch_floating_t': None, 'cell_sparse_m': None, 'patch_sparse_m': None}
        if proxy.sample_fill_share is not None and proxy.sample_grade is not None:
            cell_sky = _sky_fill_hidden(proxy.sample_fill_share >= DEPTH_FILL_LEGEND_SHARE, proxy.sample_grade)
            states['cell_sky_hidden'] = cell_sky
            states['patch_sky_hidden'] = majority_threshold(cell_sky.astype(np.float64), proxy.sample_patch_ids, patches) > 0.5
        if proxy.sample_floating_t is not None:
            states['cell_floating_t'] = proxy.sample_floating_t
            states['patch_floating_t'] = majority_threshold(proxy.sample_floating_t, proxy.sample_patch_ids, patches)
        if proxy.sample_sparse_m is not None:
            states['cell_sparse_m'] = proxy.sample_sparse_m
            states['patch_sparse_m'] = majority_threshold(proxy.sample_sparse_m, proxy.sample_patch_ids, patches, maximum=None)
        return states

    @staticmethod
    def _proxy_geometry(key: tuple, proxy, source: Path) -> _PlacementGeometry:
        gap_count = gap_total = None
        if proxy.sample_gap_fraction is not None:
            gap_count = int(np.count_nonzero(proxy.sample_gap_fraction >= 1.0))
            gap_total = int(len(proxy.sample_gap_fraction))
        fill_count = fill_total = None
        if proxy.sample_fill_share is not None:
            fill_count = _depth_fill_labelled_count(proxy.sample_fill_share)
            fill_total = int(len(proxy.sample_fill_share))
        return _PlacementGeometry(key, proxy.vertices, proxy.colors, proxy.faces, source, proxy.source_point_count, 'PROXY', (tuple(proxy.vertices.min(axis=0)), tuple(proxy.vertices.max(axis=0))), texture_coordinates=proxy.texture_coordinates, texture_rgb=proxy.texture_rgb, cell_size_m=proxy.cell_size_m, colour_gap_count=gap_count, colour_gap_total=gap_total, grade=proxy.patch_grade, cell_grade=proxy.sample_grade, texture_grade=proxy.texture_grade, texture_covered=_atlas_coverage_mask(proxy.texture_coordinates, proxy.texture_grade.shape) if proxy.texture_grade is not None and proxy.texture_rgb is not None else None, depth_fill_count=fill_count, depth_fill_total=fill_total, sparse_default_m=None if proxy.sample_sparse_m is None else _run_sparse_default(source), **PyVistaViewportAdapter._proxy_view_states(proxy))

    @staticmethod
    def _read_placement_points(source: Path, source_axis: str, maximum: int | None) -> tuple[np.ndarray, np.ndarray | None, int, int | None, int | None, np.ndarray | None]:
        from geometry.ply_preserve import iter_ply_vertex_chunks, read_ply_schema
        schema = read_ply_schema(source)
        count = schema.vertex.count
        if not count:
            raise PyVistaViewportError(f'scan has no point geometry: {source}')
        output_count = count if maximum is None else min(count, maximum)
        selected = None if output_count == count else _sample_indices(count, maximum)
        points = np.empty((output_count, 3), dtype=np.float64)
        names = {prop.name.lower(): prop.name for prop in schema.vertex.properties}
        channels = next((tuple((names[channel] for channel in candidate)) for candidate in (('red', 'green', 'blue'), ('r', 'g', 'b'), ('diffuse_red', 'diffuse_green', 'diffuse_blue')) if all((channel in names for channel in candidate))), None)
        colors = np.empty((output_count, 3), dtype=np.uint8) if channels is not None else None
        gap_column = _colour_gap_column(schema)
        gap_count: int | None = 0 if gap_column is not None else None
        fill_column = _depth_fill_column(schema)
        fill_count: int | None = 0 if fill_column is not None else None
        recovery_column = names.get(RECOVERY_SHARE_FIELD)
        fill_recovery_column = names.get('fill_recovery_share')
        fill_labelled = np.zeros(output_count, dtype=bool) if fill_column is not None else None
        offset = 0
        for chunk in iter_ply_vertex_chunks(source):
            end = offset + len(chunk)
            if selected is None:
                destination = slice(offset, end)
                local = slice(None)
            else:
                left, right = np.searchsorted(selected, (offset, end))
                destination = slice(left, right)
                local = selected[left:right] - offset
            xyz = np.column_stack([chunk[name] for name in ('x', 'y', 'z')])
            points[destination] = points_to_working_axis(xyz, source_axis)[local]
            if colors is not None:
                rgb = np.column_stack([chunk[channel][local] for channel in channels])
                if rgb.dtype.kind == 'f' and np.nanmax(rgb, initial=0.0) <= 1.0:
                    rgb = rgb * 255.0
                block = np.rint(np.clip(np.nan_to_num(rgb, nan=0.0), 0, 255)).astype(np.uint8)
                if gap_column is not None:
                    gap = np.asarray(chunk[gap_column][local]) == COLOUR_GAP_UNASSIGNED
                    gap_count += int(np.count_nonzero(gap))
                    block = _apply_colour_gap_display(block, gap)
                if fill_column is not None:
                    fill = _effective_fill_share(np.asarray(chunk[fill_column][local]), None if recovery_column is None else np.asarray(chunk[recovery_column][local]), None if fill_recovery_column is None else np.asarray(chunk[fill_recovery_column][local]))
                    labelled = fill >= DEPTH_FILL_LEGEND_SHARE
                    fill_labelled[destination] = labelled
                    fill_count += int(np.count_nonzero(labelled))
                    block = _apply_depth_fill_display(block, fill)
                colors[destination] = block
            else:
                if gap_column is not None:
                    gap_count += int(np.count_nonzero(np.asarray(chunk[gap_column][local]) == COLOUR_GAP_UNASSIGNED))
                if fill_column is not None:
                    fill = _effective_fill_share(np.asarray(chunk[fill_column][local]), None if recovery_column is None else np.asarray(chunk[recovery_column][local]), None if fill_recovery_column is None else np.asarray(chunk[fill_recovery_column][local]))
                    labelled = fill >= DEPTH_FILL_LEGEND_SHARE
                    fill_labelled[destination] = labelled
                    fill_count += int(np.count_nonzero(labelled))
            offset = end
        return (points, colors, count, gap_count, fill_count, fill_labelled)

    @staticmethod
    def _read_placement_mesh(source: Path, source_axis: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray | None]:
        pyvista, _vtk, _interactor = _load_renderer_dependencies()
        try:
            mesh = pyvista.read(source).extract_surface().triangulate()
        except Exception as error:
            raise PyVistaViewportError(f'legacy mesh cannot be read: {source.name}') from error
        raw_vertices = np.asarray(mesh.points, dtype=np.float64)
        vertices = raw_vertices.copy() if source_axis == 'SOURCE_LOCAL' else points_to_working_axis(raw_vertices, source_axis)
        packed = np.asarray(mesh.faces, dtype=np.int64)
        if len(vertices) < 3 or packed.size == 0 or packed.size % 4:
            raise PyVistaViewportError(f'legacy mesh has no triangle surface: {source.name}')
        packed = packed.reshape(-1, 4)
        if not np.all(packed[:, 0] == 3):
            raise PyVistaViewportError(f'legacy mesh has non-triangle faces: {source.name}')
        faces = packed[:, 1:].copy()
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise PyVistaViewportError(f'legacy mesh has invalid face indices: {source.name}')
        colours = None
        for candidate in ('RGB', 'rgb', 'Color', 'color'):
            if candidate in mesh.point_data:
                values = np.asarray(mesh.point_data[candidate])
                if values.ndim == 2 and values.shape == (len(vertices), 3):
                    colours = np.clip(np.rint(values * (255 if values.dtype.kind == 'f' and values.max(initial=0) <= 1 else 1)), 0, 255).astype(np.uint8)
                    break
        texture_coordinates: np.ndarray | None = None
        texture_rgb: np.ndarray | None = None
        texture_path = _packaged_mesh_texture(source)
        raw_coordinates = getattr(mesh, 'active_texture_coordinates', None)
        if texture_path is not None and raw_coordinates is not None:
            candidate_coordinates = np.asarray(raw_coordinates, dtype=np.float32)
            if candidate_coordinates.shape == (len(vertices), 2):
                try:
                    import cv2
                    image = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
                    if image is not None and image.ndim == 3 and (image.shape[2] == 3):
                        texture_coordinates = candidate_coordinates.copy()
                        texture_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except (ImportError, OSError, ValueError):
                    pass
        return (vertices, colours, faces, texture_coordinates, texture_rgb)

    @property
    def plotter(self) -> Any:
        if self._plotter is None:
            raise PyVistaViewportError('viewport widget has not been created')
        return self._plotter

    @property
    def widget(self) -> Any:
        if self._widget is None:
            raise PyVistaViewportError('viewport widget has not been created')
        return self._widget

    @property
    def scan_actors(self) -> dict[str, Any]:
        return dict(self._scan_actors)

    @property
    def drawn_actors(self) -> dict[str, tuple[DrawnActor, ...]]:
        return dict(self._actor_groups)

    @property
    def layer_actors(self) -> dict[str, Any]:
        return dict(self._layer_actors)

    @property
    def plan_actor(self) -> Any | None:
        return self._plan_actor

    @property
    def surface_actors(self) -> dict[str, Any]:
        return dict(self._surface_actors)

    @property
    def surface_display_mode(self) -> str:
        return self._surface_display_mode

    @property
    def last_load_result(self) -> ProjectLoadResult | None:
        return self._last_load_result

    @property
    def selected_scan_id(self) -> str | None:
        return self._selected_scan_id

    @property
    def selection_highlight_enabled(self) -> bool:
        return self._selection_highlight_enabled

    @property
    def transform_gizmo_actors(self) -> dict[str, Any]:
        return dict(self._transform_gizmo_actors)

    @property
    def color_mode(self) -> str:
        return self._color_mode

    @property
    def render_plan(self) -> RenderPlan | None:
        return self._render_plan

    @property
    def scan_layout_active(self) -> bool:
        return bool(self._scan_display_transforms)

    @property
    def transform_mode(self) -> str:
        return self._transform_mode

    @property
    def orbit_mode(self) -> bool:
        return self._orbit_mode

    @property
    def move_constraint(self) -> str:
        return self._move_constraint

    @property
    def manual_drag_active(self) -> bool:
        return self._manual_drag is not None

    def create_widget(self, parent: object, qt: QtModules) -> object:
        if self._widget is not None:
            raise PyVistaViewportError('viewport widget already exists')
        pyvista, vtk, QtInteractor = _load_renderer_dependencies()
        adapter = self
        QtCore = qt.QtCore

        class CCAPointsInteractor(QtInteractor):
            _navigation_shift_held = False

            def resizeEvent(self, event: object) -> None:
                super().resizeEvent(event)
                cube = adapter._view_cube
                if cube is not None:
                    cube.move(max(self.width() - cube.width() - 12, 0), 12)
                    cube.raise_()

            def _navigation_has_shift(self, event: object) -> bool:
                modifiers = event.modifiers() | qt.QtWidgets.QApplication.queryKeyboardModifiers()
                self._navigation_shift_held = self._navigation_shift_held or bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)
                return self._navigation_shift_held

            def mousePressEvent(self, event: object) -> None:
                self._navigation_has_shift(event)
                if adapter._rolove_pick_callback is not None:
                    if event.button() == QtCore.Qt.MouseButton.RightButton:
                        self._consume_right_release = True
                        self._right_press = None
                        adapter.stop_rolove_pick(cancel=True)
                        event.accept()
                        return
                    if event.button() == QtCore.Qt.MouseButton.LeftButton:
                        point = adapter.pick_rolove_point(*adapter._event_position(event))
                        if point is not None:
                            adapter._rolove_pick_callback(point)
                        self.setFocus()
                        event.accept()
                        return
                if event.button() == QtCore.Qt.MouseButton.MiddleButton:
                    adapter.begin_navigation_drag(adapter._event_position(event), 'middle')
                    self.setFocus()
                    event.accept()
                    return
                if event.button() == QtCore.Qt.MouseButton.RightButton:
                    self._consume_right_release = False
                    self._right_press = adapter._event_position(event)
                    if not adapter.manual_drag_active:
                        adapter.begin_navigation_drag(adapter._event_position(event), 'pan')
                    self.setFocus()
                    event.accept()
                    return
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    position = adapter._event_position(event)
                    if adapter.orbit_mode:
                        adapter.begin_navigation_drag(position, 'orbit')
                        self.setFocus()
                        event.accept()
                        return
                    if adapter.transform_mode == 'SELECT':
                        adapter.begin_navigation_drag(position, 'select')
                        adapter._pending_select = adapter.pick_object(*position)
                        self.setFocus()
                        event.accept()
                        return
                    elif adapter.begin_manual_drag(position):
                        self.setFocus()
                        event.accept()
                        return
                    adapter.begin_navigation_drag(position, 'select')
                    self.setFocus()
                    event.accept()
                    return
                super().mousePressEvent(event)

            def mouseMoveEvent(self, event: object) -> None:
                if adapter.manual_drag_active:
                    adapter.update_manual_drag(adapter._event_position(event))
                    event.accept()
                    return
                if adapter._drag_position is not None:
                    current = adapter._event_position(event)
                    dx = current[0] - adapter._drag_position[0]
                    dy = current[1] - adapter._drag_position[1]
                    adapter._drag_position = current
                    origin = getattr(adapter, '_drag_origin', current)
                    if math.hypot(current[0] - origin[0], current[1] - origin[1]) > 3.0:
                        adapter._pending_select = None
                    shifted = self._navigation_has_shift(event)
                    if adapter._drag_mode == 'select':
                        pass
                    elif adapter._drag_mode == 'middle':
                        orbit = (adapter._middle_button_mode == 'orbit') != shifted
                        if orbit and adapter.controller.state.mode != 'top':
                            adapter.orbit_pixels(dx, dy)
                        else:
                            adapter.pan_pixels(dx, dy)
                    elif adapter._drag_mode == 'orbit':
                        if shifted or adapter.controller.state.mode == 'top':
                            adapter.pan_pixels(dx, dy)
                        else:
                            adapter.orbit_pixels(dx, dy)
                    elif shifted or getattr(adapter, '_drag_mode', 'orbit') == 'pan':
                        adapter.pan_pixels(dx, dy)
                    else:
                        adapter.orbit_pixels(dx, dy)
                    event.accept()
                    return
                super().mouseMoveEvent(event)

            def mouseReleaseEvent(self, event: object) -> None:
                self._navigation_shift_held = False
                if event.button() == QtCore.Qt.MouseButton.RightButton:
                    if getattr(self, '_consume_right_release', False):
                        self._consume_right_release = False
                        event.accept()
                        return
                    position = adapter._event_position(event)
                    origin = getattr(self, '_right_press', None)
                    self._right_press = None
                    adapter.end_navigation_drag()
                    if origin is not None and math.dist(position, origin) <= 3.0 and (adapter._right_click_callback is not None):
                        adapter._right_click_callback(position)
                    event.accept()
                    return
                if event.button() == QtCore.Qt.MouseButton.LeftButton and adapter.manual_drag_active:
                    adapter.update_manual_drag(adapter._event_position(event))
                    adapter.commit_manual_drag()
                    event.accept()
                    return
                if adapter._drag_position is not None:
                    pending = getattr(adapter, '_pending_select', None)
                    click = math.dist(adapter._event_position(event), adapter._drag_origin) <= 3.0
                    adapter.end_navigation_drag()
                    if pending is not None:
                        adapter._select_object_from_viewport(pending)
                    elif click and event.button() == QtCore.Qt.MouseButton.LeftButton and (adapter.transform_mode == 'SELECT'):
                        adapter.clear_selection()
                    event.accept()
                    return
                super().mouseReleaseEvent(event)

            def wheelEvent(self, event: object) -> None:
                if adapter.manual_drag_active:
                    event.accept()
                    return
                angle = event.angleDelta().y()
                steps = float(angle) / 120.0 if angle else 0.0
                if not steps and hasattr(event, 'pixelDelta'):
                    steps = float(event.pixelDelta().y()) / 120.0
                adapter.zoom_at(steps, *adapter._event_position(event))
                event.accept()

            def keyPressEvent(self, event: object) -> None:
                if event.key() == QtCore.Qt.Key.Key_Shift:
                    self._navigation_shift_held = True
                    event.accept()
                    return
                if adapter.orbit_mode:
                    arrow = {QtCore.Qt.Key.Key_Left: (-1.0, 0.0), QtCore.Qt.Key.Key_Right: (1.0, 0.0), QtCore.Qt.Key.Key_Up: (0.0, 1.0), QtCore.Qt.Key.Key_Down: (0.0, -1.0)}.get(event.key())
                    if arrow is not None:
                        modifiers = event.modifiers()
                        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
                            step = 1.0
                        elif modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
                            step = 15.0
                        else:
                            step = 5.0
                        adapter.orbit_degrees(arrow[0] * step, arrow[1] * step)
                        event.accept()
                        return
                keys = {QtCore.Qt.Key.Key_F: 'f', QtCore.Qt.Key.Key_Home: 'home', QtCore.Qt.Key.Key_0: '0', QtCore.Qt.Key.Key_1: '1', QtCore.Qt.Key.Key_3: '3', QtCore.Qt.Key.Key_5: '5', QtCore.Qt.Key.Key_7: '7', QtCore.Qt.Key.Key_G: 'g', QtCore.Qt.Key.Key_R: 'r', QtCore.Qt.Key.Key_X: 'x', QtCore.Qt.Key.Key_Y: 'y', QtCore.Qt.Key.Key_Z: 'z', QtCore.Qt.Key.Key_Escape: 'escape', QtCore.Qt.Key.Key_Return: 'enter', QtCore.Qt.Key.Key_Enter: 'enter'}
                shortcut = keys.get(event.key())
                if shortcut is not None:
                    adapter.handle_shortcut(shortcut)
                    event.accept()
                    return
                super().keyPressEvent(event)

            def keyReleaseEvent(self, event: object) -> None:
                if event.key() == QtCore.Qt.Key.Key_Shift:
                    self._navigation_shift_held = False
                    event.accept()
                    return
                super().keyReleaseEvent(event)
        widget = CCAPointsInteractor(parent=parent, off_screen=self.off_screen, auto_update=False, multi_samples=0 if self.off_screen else None)
        widget.setMinimumSize(520, 360)
        widget.setMouseTracking(True)
        widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        widget.setToolTip(SURFACE_SUPPORT_SIZE_NOTICE)
        widget.set_background(VIEWPORT_BACKGROUND)
        widget.add_axes(line_width=2, labels_off=False)
        widget.enable_eye_dome_lighting()
        self._widget = widget
        self._plotter = widget
        self._pyvista = pyvista
        self._vtk = vtk
        from viewer.view_cube import create_view_cube
        self._view_cube = create_view_cube(widget, qt, self.set_named_view, self.fit_scene, self.fit_selection)
        self._view_cube.move(max(widget.width() - self._view_cube.width() - 12, 0), 12)
        self._view_cube.show()
        self._view_cube.raise_()
        self.apply_camera(render=False)
        return widget

    @staticmethod
    def _event_position(event: object) -> tuple[float, float]:
        position = event.position() if hasattr(event, 'position') else event.pos()
        return (float(position.x()), float(position.y()))

    def set_manual_transform_callback(self, callback: Callable[[ManualTransformEvent], None] | None) -> None:
        self._manual_transform_callback = callback

    def set_scan_selection_callback(self, callback: Callable[[str], None] | None) -> None:
        self._scan_selection_callback = callback

    def set_selection_provider(self, callback: Callable[[], Sequence[str]] | None) -> None:
        self._selection_provider = callback

    def set_plan_selection_callback(self, callback: Callable[[], None] | None) -> None:
        self._plan_selection_callback = callback

    def set_camera_changed_callback(self, callback: Callable[[CameraState], None] | None) -> None:
        self._camera_changed_callback = callback
        if callback is not None:
            callback(self.controller.state)

    def set_transform_mode_callback(self, callback: Callable[[str, str], None] | None) -> None:
        self._transform_mode_callback = callback

    @staticmethod
    def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
        return tuple((tuple((float(value) for value in row)) for row in matrix))

    def _emit_transform_event(self, drag: _ManualDrag, phase: str, transform: np.ndarray) -> None:
        callback = self._manual_transform_callback
        if callback is None:
            return
        callback(ManualTransformEvent(scan_id=drag.scan_id, phase=phase, mode=drag.mode, transform=self._matrix_tuple(transform), original_transform=self._matrix_tuple(drag.original_transform), target_kind=drag.target_kind))

    def _emit_transform_mode(self) -> None:
        callback = self._transform_mode_callback
        if callback is not None:
            callback(self._transform_mode, self._move_constraint)

    def _remove_transform_gizmo(self) -> None:
        if self._plotter is None:
            self._transform_gizmo_actors.clear()
            return
        for actor_name in tuple(self._transform_gizmo_actors):
            self.plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._transform_gizmo_actors.clear()

    def _refresh_selection_box(self, *, render: bool=False) -> None:
        if self._plotter is None:
            self._selection_box_actors.clear()
            return
        for name in tuple(self._selection_box_actors):
            self.plotter.remove_actor(name, reset_camera=False, render=False)
        self._selection_box_actors.clear()
        if not self._selection_box_enabled:
            return
        scan_id = self._selected_scan_id
        bounds = self._actor_bounds.get(self._plan_actor_name) if self._selected_plan else self._scan_bounds.get(scan_id) if scan_id is not None else None
        if bounds is not None:
            pyvista = self._pyvista
            if pyvista is None:
                pyvista, _vtk, _interactor = _load_renderer_dependencies()
                self._pyvista = pyvista
            name = f"cca-selection-box::{('plan' if self._selected_plan else scan_id)}"
            actor = self.plotter.add_mesh(pyvista.Box(bounds=bounds), name=name, style='wireframe', color=SELECTION_COLOR, line_width=3.0, lighting=False, pickable=False, reset_camera=False, render=False)
            self._selection_box_actors[name] = actor
        if render and (not self.off_screen):
            self.plotter.render()

    def _set_selection_tint(self, name: str, actor: Any, selected: bool) -> None:
        mapper = actor.GetMapper()
        property_ = actor.GetProperty()
        if selected:
            if name not in self._selection_actor_state:
                self._selection_actor_state[name] = (int(mapper.GetScalarVisibility()), tuple((float(value) for value in property_.GetColor())))
            mapper.SetScalarVisibility(False)
            property_.SetColor(242.0 / 255.0, 140.0 / 255.0, 40.0 / 255.0)
            return
        state = self._selection_actor_state.pop(name, None)
        if state is not None:
            scalar_visibility, color = state
            mapper.SetScalarVisibility(scalar_visibility)
            property_.SetColor(*color)

    def _refresh_transform_gizmo(self, *, render: bool=True) -> None:
        if self._plotter is None:
            return
        self._remove_transform_gizmo()
        self._refresh_selection_box(render=False)
        if self._transform_mode == 'SELECT' or (self._selected_scan_id is None and (not self._selected_plan)):
            if render and (not self.off_screen):
                self.plotter.render()
            return
        bounds = self._actor_bounds.get(self._plan_actor_name) if self._selected_plan else self._scan_bounds.get(self._selected_scan_id)
        if bounds is None:
            return
        pyvista = self._pyvista
        if pyvista is None:
            pyvista, _vtk, _interactor = _load_renderer_dependencies()
            self._pyvista = pyvista
        pivot = np.asarray(((bounds[0] + bounds[1]) * 0.5, (bounds[2] + bounds[3]) * 0.5, (bounds[4] + bounds[5]) * 0.5), dtype=np.float64)
        diagonal = float(np.linalg.norm(np.asarray((bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]), dtype=np.float64)))
        length = max(diagonal * 0.16, self.controller.state.distance * 0.035, 0.25)

        def add_actor(name: str, mesh: Any, color: str, line_width: float=5.0) -> None:
            actor_name = f'cca-transform-gizmo::{name.lower()}'
            actor = self.plotter.add_mesh(mesh, name=actor_name, color=color, line_width=line_width, render_lines_as_tubes=True, lighting=False, pickable=False, reset_camera=False, render=False)
            self._transform_gizmo_actors[actor_name] = actor
        if self._transform_mode == 'MOVE':
            directions = {'X': np.asarray((1.0, 0.0, 0.0)), 'Y': np.asarray((0.0, 1.0, 0.0)), 'Z': np.asarray((0.0, 0.0, 1.0))}
            for axis, direction in directions.items():
                width = 8.0 if self._move_constraint in {'XY', axis} else 3.0
                add_actor(axis, pyvista.Line(pivot, pivot + direction * length), GIZMO_AXIS_COLORS[axis], width)
        else:
            angles = np.linspace(0.0, 2.0 * math.pi, 97)
            points = np.column_stack((pivot[0] + np.cos(angles) * length, pivot[1] + np.sin(angles) * length, np.full_like(angles, pivot[2])))
            add_actor('rotate-z', pyvista.lines_from_points(points, close=False), GIZMO_ROTATE_COLOR, 6.0)
        center_name = 'cca-transform-gizmo::pivot'
        center = self.plotter.add_mesh(pyvista.Sphere(radius=max(length * 0.055, 0.03), center=pivot, theta_resolution=16, phi_resolution=12), name=center_name, color=GIZMO_CENTER_COLOR, lighting=False, pickable=False, reset_camera=False, render=False)
        self._transform_gizmo_actors[center_name] = center
        if render and (not self.off_screen):
            self.plotter.render()

    def set_transform_mode(self, mode: str) -> str:
        normalized = str(mode).strip().upper().replace('ROTATE_Z', 'ROTATE Z')
        if normalized == 'ROTATE':
            normalized = 'ROTATE Z'
        if normalized not in TRANSFORM_MODES:
            raise PyVistaViewportError(f"transform mode must be one of: {', '.join(TRANSFORM_MODES)}")
        if self._manual_drag is not None:
            self.cancel_manual_drag()
        self._transform_mode = normalized
        self._apply_scan_highlight()
        if normalized == 'MOVE' and self._move_constraint not in MOVE_CONSTRAINTS:
            self._move_constraint = 'XY'
        self._refresh_transform_gizmo()
        self._emit_transform_mode()
        return self._transform_mode

    def set_move_constraint(self, constraint: str) -> str:
        normalized = str(constraint).strip().upper()
        if normalized not in MOVE_CONSTRAINTS:
            raise PyVistaViewportError(f"move constraint must be one of: {', '.join(MOVE_CONSTRAINTS)}")
        self._move_constraint = normalized
        if self._manual_drag is not None and self._manual_drag.mode == 'MOVE':
            self._manual_drag.constraint = normalized
        self._refresh_transform_gizmo()
        self._emit_transform_mode()
        return self._move_constraint

    @staticmethod
    def _same_vtk_prop(first: object, second: object) -> bool:
        if first is second:
            return True
        try:
            return first.GetAddressAsString('') == second.GetAddressAsString('')
        except AttributeError:
            return False

    def pick_scan(self, x: float, y: float) -> str | None:
        if self._vtk is None or self._widget is None:
            return None
        vtk_x, vtk_y = self._vtk_display_position(x, y)

        def resolve(prop: object) -> str | None:
            if prop is None:
                return None
            for actor_id, group in self._actor_groups.items():
                if any((self._same_vtk_prop(prop, drawn.actor) for drawn in group)):
                    return self._actor_requests[actor_id].scan_id
            for actor_name, actor in self._surface_actors.items():
                if self._same_vtk_prop(prop, actor):
                    parts = actor_name.split('::')
                    return parts[1] if len(parts) >= 3 else None
            return None
        points_only = any((d.actor.GetVisibility() for g in self._actor_groups.values() for d in g)) and all((d.kind == 'POINTS' for g in self._actor_groups.values() for d in g if d.actor.GetVisibility())) and (not any((a.GetVisibility() for a in self._surface_actors.values())))
        if not points_only:
            cell_picker = self._vtk.vtkCellPicker()
            cell_picker.PickFromListOn()
            for group in self._actor_groups.values():
                for drawn in group:
                    cell_picker.AddPickList(drawn.actor)
            for actor in self._surface_actors.values():
                cell_picker.AddPickList(actor)
            if cell_picker.Pick(vtk_x, vtk_y, 0.0, self.plotter.renderer):
                picked = resolve(cell_picker.GetActor())
                if picked is not None:
                    return picked
        point_picker = self._vtk.vtkPointPicker()
        point_picker.PickFromListOn()
        for group in self._actor_groups.values():
            for drawn in group:
                point_picker.AddPickList(drawn.actor)
        if point_picker.Pick(vtk_x, vtk_y, 0.0, self.plotter.renderer):
            picked = resolve(point_picker.GetActor())
            if picked is not None:
                return picked
        return None

    def _select_from_viewport(self, scan_id: str) -> str:
        self.select_scan(scan_id)
        callback = self._scan_selection_callback
        if callback is not None:
            callback(scan_id)
        return scan_id

    def pick_object(self, x: float, y: float) -> tuple[str, str] | None:
        if self.controller.state.mode == 'top' and self._pick_plan(x, y):
            return ('plan', 'dxf')
        scan_id = self.pick_scan(x, y)
        if scan_id is not None:
            return ('scan', scan_id)
        return ('plan', 'dxf') if self._pick_plan(x, y) else None

    def _pick_plan(self, x: float, y: float) -> bool:
        actor = self._plan_actor
        if actor is None or not actor.GetVisibility() or self._vtk is None:
            return False
        picker = self._vtk.vtkCellPicker()
        picker.SetTolerance(self._pickbox_px / max(math.hypot(self.widget.width(), self.widget.height()), 1))
        picker.PickFromListOn()
        picker.AddPickList(actor)
        previous = actor.GetPickable()
        try:
            actor.SetPickable(True)
            hit = picker.Pick(*self._vtk_display_position(x, y), 0.0, self.plotter.renderer)
        finally:
            actor.SetPickable(previous)
        return bool(hit and self._same_vtk_prop(picker.GetActor(), actor))

    def _select_object_from_viewport(self, target: tuple[str, str]) -> str:
        if target[0] == 'scan':
            return self._select_from_viewport(target[1])
        self.select_plan()
        if self._plan_selection_callback is not None:
            self._plan_selection_callback()
        return 'plan::dxf'

    def select_plan(self) -> None:
        if self._plan_actor is None:
            raise PyVistaViewportError('load a DXF underlay before selecting it')
        self.select_scan(None)
        self._selected_plan = True
        self._apply_scan_highlight()
        self._refresh_transform_gizmo()

    def clear_selection(self) -> None:
        self.select_scan(None)
        if self._selection_cleared_callback is not None:
            self._selection_cleared_callback()

    def select_at_cursor(self, x: float, y: float) -> str | None:
        if self.transform_mode != 'SELECT':
            return None
        target = self.pick_object(x, y)
        if target is None:
            return None
        return self._select_object_from_viewport(target)

    def _vtk_display_position(self, x: float, y: float) -> tuple[float, float]:
        width, height = self.plotter.render_window.GetSize()
        widget_width, widget_height = (max(self.widget.width(), 1), max(self.widget.height(), 1))
        width = width if width > 0 else widget_width
        height = height if height > 0 else widget_height
        return (float(x) * width / widget_width, max(widget_height - float(y), 0.0) * height / widget_height)

    def _world_ray(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray] | None:
        renderer = self.plotter.renderer
        vtk_x, vtk_y = self._vtk_display_position(x, y)
        points: list[np.ndarray] = []
        for depth in (0.0, 1.0):
            renderer.SetDisplayPoint(vtk_x, vtk_y, depth)
            renderer.DisplayToWorld()
            homogeneous = np.asarray(renderer.GetWorldPoint(), dtype=np.float64)
            if homogeneous.shape != (4,) or not np.isfinite(homogeneous).all():
                return None
            if abs(float(homogeneous[3])) <= 1e-12:
                return None
            points.append(homogeneous[:3] / homogeneous[3])
        direction = points[1] - points[0]
        if float(np.linalg.norm(direction)) <= 1e-12:
            return None
        return (points[0], direction)

    def _xy_point_from_display(self, position: tuple[float, float], z_value: float) -> np.ndarray | None:
        ray = self._world_ray(*position)
        if ray is None:
            return None
        origin, direction = ray
        if abs(float(direction[2])) <= 1e-09:
            return None
        factor = (float(z_value) - float(origin[2])) / float(direction[2])
        point = origin + direction * factor
        return point if np.isfinite(point).all() else None

    def _metres_per_pixel(self) -> float:
        height = max(int(self.widget.height()), 1)
        camera = self.plotter.renderer.GetActiveCamera()
        if camera.GetParallelProjection():
            return max(float(camera.GetParallelScale()) * 2.0 / height, 1e-09)
        tangent = math.tan(math.radians(float(camera.GetViewAngle())) * 0.5)
        if camera.GetUseHorizontalViewAngle():
            width, frame_height = self.plotter.render_window.GetSize()
            aspect = max(float(width), 1.0) / max(float(frame_height), 1.0)
            tangent /= aspect
        return max(2.0 * self.controller.state.distance * tangent / height, 1e-09)

    def _screen_xy_delta(self, start: tuple[float, float], current: tuple[float, float]) -> np.ndarray:
        camera = self.plotter.renderer.GetActiveCamera()
        eye = np.asarray(camera.GetPosition(), dtype=np.float64)
        target = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        up = np.asarray(camera.GetViewUp(), dtype=np.float64)
        forward = target - eye
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        right = np.cross(forward, up)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        dx = float(current[0] - start[0])
        dy = float(current[1] - start[1])
        delta = (dx * right - dy * up) * self._metres_per_pixel()
        delta[2] = 0.0
        return delta

    def _display_angle(self, position: tuple[float, float], pivot: np.ndarray) -> float | None:
        renderer = self.plotter.renderer
        renderer.SetWorldPoint(float(pivot[0]), float(pivot[1]), float(pivot[2]), 1.0)
        renderer.WorldToDisplay()
        display = np.asarray(renderer.GetDisplayPoint(), dtype=np.float64)
        if display.shape != (3,) or not np.isfinite(display).all():
            return None
        event_x, event_y = self._vtk_display_position(*position)
        dx = event_x - float(display[0])
        dy = event_y - float(display[1])
        if math.hypot(dx, dy) < 3.0:
            return None
        return math.atan2(dy, dx)

    def begin_manual_drag(self, position: tuple[float, float]) -> bool:
        if self._transform_mode == 'SELECT' or (self._selected_scan_id is None and (not self._selected_plan)):
            return False
        if self._project is None:
            return False
        scan = self._project.scan(self._selected_scan_id) if self._selected_scan_id else None
        target = self._project.dxf_plan if self._selected_plan else scan
        bounds = self._actor_bounds.get(self._plan_actor_name) if self._selected_plan else self._scan_bounds.get(self._selected_scan_id)
        if target is None or bounds is None:
            return False
        if self._manual_drag is not None:
            self.cancel_manual_drag()
        original = _matrix(target.transform, 'manual object transform').copy()
        pivot = np.asarray(((bounds[0] + bounds[1]) * 0.5, (bounds[2] + bounds[3]) * 0.5, (bounds[4] + bounds[5]) * 0.5), dtype=np.float64)
        start_xy = None
        start_angle = None
        if self._transform_mode == 'MOVE' and self._move_constraint != 'Z':
            start_xy = self._xy_point_from_display(position, float(pivot[2]))
        elif self._transform_mode == 'ROTATE Z':
            start_angle = self._display_angle(position, pivot)
            if start_angle is None:
                return False
        drag = _ManualDrag(scan_id='' if self._selected_plan else scan.scan_id, mode=self._transform_mode, constraint=self._move_constraint, start_position=(float(position[0]), float(position[1])), original_transform=original, preview_transform=original.copy(), pivot=pivot, start_xy=start_xy, start_angle=start_angle, target_kind='plan' if self._selected_plan else 'scan')
        self._manual_drag = drag
        self._emit_transform_event(drag, 'begin', original)
        return True

    def update_manual_drag(self, position: tuple[float, float]) -> bool:
        drag = self._manual_drag
        if drag is None:
            return False
        current = (float(position[0]), float(position[1]))
        target = drag.original_transform.copy()
        if drag.mode == 'MOVE':
            if drag.constraint == 'Z':
                delta = np.asarray((0.0, 0.0, -(current[1] - drag.start_position[1]) * self._metres_per_pixel()), dtype=np.float64)
            else:
                current_xy = self._xy_point_from_display(current, float(drag.pivot[2]))
                if drag.start_xy is not None and current_xy is not None:
                    delta = current_xy - drag.start_xy
                else:
                    delta = self._screen_xy_delta(drag.start_position, current)
                delta[2] = 0.0
                if drag.constraint == 'X':
                    delta[1] = 0.0
                elif drag.constraint == 'Y':
                    delta[0] = 0.0
            target[:3, 3] = drag.original_transform[:3, 3] + delta
        else:
            angle = self._display_angle(current, drag.pivot)
            if angle is None or drag.start_angle is None:
                return False
            delta_angle = math.atan2(math.sin(angle - drag.start_angle), math.cos(angle - drag.start_angle))
            cosine = math.cos(delta_angle)
            sine = math.sin(delta_angle)
            rotation = np.eye(4, dtype=np.float64)
            rotation[:2, :2] = ((cosine, -sine), (sine, cosine))
            to_pivot = np.eye(4, dtype=np.float64)
            to_pivot[:3, 3] = drag.pivot
            from_pivot = np.eye(4, dtype=np.float64)
            from_pivot[:3, 3] = -drag.pivot
            target = to_pivot @ rotation @ from_pivot @ drag.original_transform
        target[3] = (0.0, 0.0, 0.0, 1.0)
        drag.preview_transform = target
        if drag.target_kind == 'plan':
            self.update_plan_transform(target)
        else:
            self.update_scan_transform(drag.scan_id, target)
        self._emit_transform_event(drag, 'preview', target)
        return True

    def commit_manual_drag(self) -> bool:
        drag = self._manual_drag
        if drag is None:
            return False
        self._manual_drag = None
        self._emit_transform_event(drag, 'commit', drag.preview_transform)
        return True

    def cancel_manual_drag(self) -> bool:
        drag = self._manual_drag
        if drag is None:
            return False
        self._manual_drag = None
        if drag.target_kind == 'plan':
            self.update_plan_transform(drag.original_transform)
        else:
            self.update_scan_transform(drag.scan_id, drag.original_transform)
        self._emit_transform_event(drag, 'cancel', drag.original_transform)
        return True

    def _visible_navigation_bounds(self, scan_ids: Sequence[str] | None=None, *, include_plan: bool=True) -> tuple[tuple[float, float, float, float, float, float], ...]:
        wanted = None if scan_ids is None else frozenset(scan_ids)
        bounds = []
        for actor_id, actor in self._layer_actors.items():
            request = self._actor_requests.get(actor_id)
            item = self._actor_bounds.get(actor_id)
            if request is not None and item is not None and actor.GetVisibility() and (wanted is None or request.scan_id in wanted):
                bounds.append(item)
        if include_plan and wanted is None and (self._plan_actor is not None) and self._plan_actor.GetVisibility():
            item = self._actor_bounds.get(self._plan_actor_name)
            if item is not None:
                bounds.append(item)
        return tuple((item for item in bounds if len(item) == 6 and all((math.isfinite(value) for value in item)) and all((item[index] <= item[index + 1] for index in (0, 2, 4)))))

    def _update_camera_clipping(self) -> None:
        pose = self.controller.pose()
        eye, target = (np.asarray(pose.eye), np.asarray(pose.target))
        forward = target - eye
        forward /= np.linalg.norm(forward)
        bounds = self._visible_navigation_bounds()
        depths = [float(np.dot(np.asarray((x, y, z)) - eye, forward)) for item in bounds for x in item[:2] for y in item[2:4] for z in item[4:]]
        if depths:
            margin = max((max(depths) - min(depths)) * 0.02, 0.01)
            far = max(max(depths) + margin, 1.0)
            near = max(min(depths) - margin, far * 1e-07, 0.0001)
        else:
            near, far = (0.001, max(self.controller.state.distance * 10, 1.0))
        self.plotter.renderer.GetActiveCamera().SetClippingRange(near, max(far, near + 1))

    @property
    def view_cube(self) -> Any | None:
        return self._view_cube

    def set_cad_crosshair(self, x, y, *, visible, percent, pickbox_px) -> None:
        if self._plotter is None:
            return
        crosshair = getattr(self, '_cad_crosshair', None)
        if not visible or not percent:
            if crosshair is not None and crosshair.actor.GetVisibility():
                crosshair.actor.VisibilityOff()
                if not self.off_screen:
                    self.plotter.render()
            return
        if crosshair is None:
            from viewer.cad_crosshair import CadCrosshair
            self._cad_crosshair = crosshair = CadCrosshair(self._vtk)
        width, height = self.plotter.render_window.GetSize()
        if width <= 0 or height <= 0:
            return
        crosshair.update(x=x, y=y, width=width, height=height, logical_width=self.widget.width(), logical_height=self.widget.height(), percent=percent, pickbox_px=pickbox_px)
        crosshair.attach(self.plotter.render_window)
        if not self.off_screen:
            self.plotter.render()

    def apply_camera(self, *, render: bool=True) -> None:
        pose = self.controller.pose()
        camera = self.plotter.renderer.GetActiveCamera()
        camera.SetPosition(*pose.eye)
        camera.SetFocalPoint(*pose.target)
        camera.SetViewUp(*pose.screen_up)
        if self.controller.state.projection == 'orthographic':
            camera.ParallelProjectionOn()
            camera.SetParallelScale(max(self.controller.state.distance * 0.5, 1e-06))
        else:
            camera.ParallelProjectionOff()
        self._update_camera_clipping()
        if self._view_cube is not None:
            self._view_cube.set_camera_pose(pose)
        if self._camera_changed_callback is not None:
            self._camera_changed_callback(self.controller.state)
        if render and (not self.off_screen):
            self.plotter.render()

    def _adopt_renderer_look_at(self) -> None:
        camera = self.plotter.renderer.GetActiveCamera()
        pose = self.controller.pose()
        if np.allclose(camera.GetPosition(), pose.eye, rtol=0, atol=1e-09) and np.allclose(camera.GetFocalPoint(), pose.target, rtol=0, atol=1e-09):
            return
        self.controller.sync_look_at(camera.GetPosition(), camera.GetFocalPoint(), mode=self.controller.state.mode)

    def begin_navigation_drag(self, position: tuple[float, float], mode: str) -> None:
        self._adopt_renderer_look_at()
        self._drag_position = position
        self._drag_origin = position
        self._pending_select = None
        self._drag_mode = mode if mode in {'pan', 'select', 'middle'} else 'orbit'
        self._orbit_center = self.pick_cursor(*position) if self._drag_mode in {'orbit', 'middle'} else None

    def set_orbit_mode(self, enabled: bool) -> bool:
        requested = bool(enabled)
        if requested == self._orbit_mode:
            return self._orbit_mode
        if requested:
            self._orbit_mode_return = self._transform_mode
            self._orbit_mode = True
            self.set_transform_mode('SELECT')
        else:
            self._orbit_mode = False
            self.set_transform_mode(self._orbit_mode_return)
        return self._orbit_mode

    def _orbit_anchor(self) -> tuple[float, float, float] | None:
        scan_id = self._selected_scan_id
        bounds = self._scan_bounds.get(scan_id) if scan_id is not None else None
        if bounds is None:
            return None
        box = np.asarray(bounds, dtype=float).reshape(3, 2)
        if box.shape != (3, 2) or not np.isfinite(box).all():
            return None
        center = box.mean(axis=1)
        return (float(center[0]), float(center[1]), float(center[2]))

    def orbit_degrees(self, azimuth_degrees: float, elevation_degrees: float=0.0) -> bool:
        yaw = math.radians(float(azimuth_degrees))
        pitch = math.radians(float(elevation_degrees))
        if self.controller.state.mode == 'top':
            return False
        if yaw == 0.0 and pitch == 0.0:
            return True
        self._adopt_renderer_look_at()
        center = self._orbit_anchor()
        if center is not None:
            self.controller.orbit_about(yaw, pitch, center)
        else:
            self.controller.orbit(yaw, pitch)
        self.apply_camera()
        return True

    def end_navigation_drag(self) -> None:
        self._drag_position = None
        self._drag_origin = None
        self._pending_select = None
        self._drag_mode = 'orbit'
        self._orbit_center = None

    def orbit_pixels(self, dx: float, dy: float) -> None:
        self._adopt_renderer_look_at()
        yaw = -float(dx) * 0.006
        pitch = float(dy) * 0.006
        center = getattr(self, '_orbit_center', None)
        if center is not None:
            self.controller.orbit_about(yaw, pitch, center)
        else:
            self.controller.orbit(yaw, pitch)
        self.apply_camera()

    def pan_pixels(self, dx: float, dy: float) -> None:
        self._adopt_renderer_look_at()
        metres_per_pixel = self._metres_per_pixel()
        self.controller.pan(-float(dx) * metres_per_pixel, float(dy) * metres_per_pixel)
        self.apply_camera()

    def zoom_steps(self, steps: float) -> None:
        self._adopt_renderer_look_at()
        self.controller.zoom(float(steps))
        self.apply_camera()

    def zoom_at(self, steps: float, x: float, y: float) -> None:
        self._adopt_renderer_look_at()
        anchor = self.pick_cursor(x, y) if steps else None
        if anchor is None:
            self.controller.zoom(float(steps))
        else:
            self.controller.zoom_towards(float(steps), anchor)
        self.apply_camera()

    def pick_cursor(self, x: float, y: float) -> tuple[float, float, float] | None:
        if self._vtk is None or self._widget is None:
            return None
        vtk_x, vtk_y = self._vtk_display_position(x, y)
        hits = []
        for use_cells in (True, False):
            picker = self._vtk.vtkCellPicker() if use_cells else self._vtk.vtkPointPicker()
            picker.SetTolerance(0.015)
            picker.PickFromListOn()
            actors = 0
            for group in self._actor_groups.values():
                for drawn in group:
                    is_surface = drawn.kind == 'SURFACE'
                    if bool(drawn.actor.GetVisibility()) and is_surface == use_cells:
                        picker.AddPickList(drawn.actor)
                        actors += 1
            if not actors or not picker.Pick(vtk_x, vtk_y, 0.0, self.plotter.renderer):
                continue
            point = tuple((float(value) for value in picker.GetPickPosition()))
            if len(point) == 3 and all((math.isfinite(value) for value in point)):
                hits.append(point)
        if not hits:
            return None
        eye = np.asarray(self.controller.pose().eye)
        return min(hits, key=lambda point: float(np.linalg.norm(np.asarray(point) - eye)))

    def start_rolove_pick(self, scan_id: str, callback: Callable) -> None:
        self._require_detailed_scan(scan_id)
        self.cancel_manual_drag()
        self.reset_scan_layout()
        self.set_transform_mode('SELECT')
        self._rolove_pick_scan_id = scan_id
        self._rolove_pick_callback = callback

    def stop_rolove_pick(self, *, cancel: bool=False) -> None:
        callback = self._rolove_pick_callback
        self._rolove_pick_callback = None
        self._rolove_pick_scan_id = None
        if cancel and callback is not None:
            callback(None)

    def pick_rolove_point(self, x: float, y: float) -> tuple | None:
        if self._vtk is None or self._rolove_pick_scan_id is None:
            return None
        picker = self._vtk.vtkPointPicker()
        picker.SetTolerance(0.01)
        picker.PickFromListOn()
        for name, request in self._actor_requests.items():
            if request.scan_id != self._rolove_pick_scan_id:
                continue
            for drawn in self._actor_groups.get(name, ()):
                if drawn.kind == 'POINTS' and drawn.actor.GetVisibility():
                    picker.AddPickList(drawn.actor)
        width, height = self.plotter.render_window.GetSize()
        px = x * width / max(self.widget.width(), 1)
        py = (self.widget.height() - y - 1) * height / max(self.widget.height(), 1)
        if not picker.Pick(px, py, 0, self.plotter.renderer):
            return None
        return tuple((float(value) for value in picker.GetPickPosition()))

    def rolove_display_points(self, scan_id: str) -> np.ndarray:
        self._require_detailed_scan(scan_id)
        return self._scan_display_points(scan_id)

    def rolove_source_layer(self, scan_id: str) -> str:
        self._require_detailed_scan(scan_id)
        for name, request in self._actor_requests.items():
            if request.scan_id == scan_id and self._layer_actors[name].GetVisibility():
                return request.source_layer_key
        raise ValueError('Taramanin gorunur bir nokta katmanini acin')

    def show_rolove_surfaces(self, scan_id: str, surfaces: list, picks: list | None=None) -> None:
        self.clear_rolove_surfaces(render=False)
        self._rolove_surface_scan_id = scan_id
        points = self._scan_display_points(scan_id)
        lower, upper = (points.min(axis=0), points.max(axis=0))
        corners = np.asarray([[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
        for number, surface in enumerate(surfaces):
            relative = corners - surface.origin
            us, vs = (relative @ surface.u, relative @ surface.v)
            u0, u1 = (float(us.min()) - 0.2, float(us.max()) + 0.2)
            v0, v1 = (float(vs.min()) - 0.2, float(vs.max()) + 0.2)
            vertices = np.asarray([np.asarray(surface.origin) + u * np.asarray(surface.u) + v * np.asarray(surface.v) for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])
            name = f'rolove-plane-{number}'
            self.plotter.add_mesh(self._pyvista.PolyData(vertices, [4, 0, 1, 2, 3]), name=name, color='#C8102E', opacity=0.28, show_edges=True, edge_color='#FF4D5E', line_width=4, pickable=False, reset_camera=False, render=False)
            self._rolove_actor_names.append(name)
            center = vertices.mean(axis=0)
            arrow_name = f'rolove-normal-{number}'
            self.plotter.add_arrows(center[None, :], surface.normal[None, :], mag=max(float(np.linalg.norm(upper - lower)) * 0.05, 0.3), name=arrow_name, color='#C8102E', pickable=False, reset_camera=False, render=False)
            self._rolove_actor_names.append(arrow_name)
        if picks:
            name = 'rolove-picked-points'
            self.plotter.add_points(np.asarray(picks), name=name, color='#E08430', point_size=16, render_points_as_spheres=True, pickable=False, reset_camera=False, render=False)
            self._rolove_actor_names.append(name)
        self.plotter.render()

    def clear_rolove_surfaces(self, *, render: bool=True) -> None:
        for name in self._rolove_actor_names:
            self.plotter.remove_actor(name, reset_camera=False, render=False)
        self._rolove_actor_names.clear()
        self._rolove_surface_scan_id = None
        if render:
            self.plotter.render()

    def handle_shortcut(self, key: str) -> bool:
        normalized = key.lower()
        if normalized == 'g':
            self.set_transform_mode('MOVE')
        elif normalized == 'r':
            self.set_transform_mode('ROTATE Z')
        elif normalized in {'escape', 'esc'}:
            if self._rolove_pick_callback is not None:
                self.stop_rolove_pick(cancel=True)
                return True
            if self._orbit_mode:
                self.set_orbit_mode(False)
                return True
            if not self.cancel_manual_drag():
                self.set_transform_mode('SELECT')
        elif normalized in {'enter', 'return'}:
            if not self.commit_manual_drag():
                return False
        elif normalized in {'x', 'y', 'z'}:
            if self._transform_mode != 'MOVE':
                return False
            self.set_move_constraint(normalized)
        elif normalized == 'f':
            self.fit_selection()
        elif normalized == 'home':
            self.fit_scene()
        elif normalized in {'0', '1', '3', '7'}:
            self.set_named_view({'0': 'perspective', '1': 'front', '3': 'right', '7': 'top'}[normalized])
        elif normalized == '5':
            projection = 'orthographic' if self.controller.state.projection == 'perspective' else 'perspective'
            self.controller.set_projection(projection)
            self.apply_camera()
        else:
            return False
        return True

    def _apply_scan_highlight(self) -> None:
        enabled = self._selection_highlight_enabled and self._transform_mode == 'SELECT'
        highlighted_scan = self._selected_scan_id if enabled else None
        if self._plan_actor is not None:
            self._set_selection_tint(self._plan_actor_name, self._plan_actor, self._selected_plan and enabled)
        for actor_id, group in self._actor_groups.items():
            request = self._actor_requests[actor_id]
            selected = highlighted_scan == request.scan_id
            for drawn in group:
                self._set_selection_tint(drawn.name, drawn.actor, selected)
                property_ = drawn.actor.GetProperty()
                if drawn.kind == 'POINTS':
                    property_.SetPointSize(self.point_size * drawn.size_factor * (1.75 if selected else 1.0))
                property_.SetOpacity(1.0 if highlighted_scan is None or selected else 0.45)
        for actor_name, actor in self._surface_actors.items():
            parts = actor_name.split('::')
            scan_id = parts[1] if len(parts) >= 3 else None
            self._set_selection_tint(actor_name, actor, scan_id == highlighted_scan)
        self._refresh_selection_box(render=False)

    def set_selection_highlight_enabled(self, enabled: bool) -> bool:
        if type(enabled) is not bool:
            raise PyVistaViewportError('selection highlight enabled must be boolean')
        self._selection_highlight_enabled = enabled
        self._apply_scan_highlight()
        if self._plotter is not None and (not self.off_screen):
            self._plotter.render()
        return self._selection_highlight_enabled

    def set_selection_box_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise PyVistaViewportError('selection box enabled must be boolean')
        self._selection_box_enabled = enabled
        self._refresh_selection_box(render=True)

    def set_point_size(self, point_size: float) -> float:
        if isinstance(point_size, bool) or not isinstance(point_size, (int, float)):
            raise PyVistaViewportError('point_size must be positive')
        normalized = float(point_size)
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise PyVistaViewportError('point_size must be positive')
        self.point_size = normalized
        self._apply_scan_highlight()
        if self._plotter is not None and (not self.off_screen):
            self.plotter.render()
        return self.point_size

    def _apply_surface_display_mode(self) -> None:
        for actor in self._surface_actors.values():
            property_ = actor.GetProperty()
            if self._surface_display_mode == 'POINTS ONLY':
                actor.SetVisibility(False)
            elif self._surface_display_mode == 'GHOST SURFACE':
                actor.SetVisibility(True)
                property_.SetRepresentationToWireframe()
                property_.SetOpacity(GHOST_SURFACE_OPACITY)
            else:
                actor.SetVisibility(True)
                property_.SetRepresentationToSurface()
                property_.SetOpacity(SOLID_SURFACE_OPACITY)

    def set_surface_display_mode(self, mode: str) -> str:
        normalized = mode.strip().upper() if isinstance(mode, str) else ''
        if normalized not in SURFACE_DISPLAY_MODES:
            choices = ', '.join(SURFACE_DISPLAY_MODES)
            raise PyVistaViewportError(f'surface display mode must be one of: {choices}')
        self._surface_display_mode = normalized
        self._apply_surface_display_mode()
        if self._plotter is not None and (not self.off_screen):
            self.plotter.render()
        return self._surface_display_mode

    def select_scan(self, scan_id: str | None) -> str | None:
        if scan_id is not None and scan_id not in self._scan_bounds:
            raise PyVistaViewportError(f'scan is not loaded: {scan_id}')
        if self._manual_drag is not None and scan_id != self._selected_scan_id:
            self.cancel_manual_drag()
        self._selected_scan_id = scan_id
        self._selected_plan = False
        self._apply_scan_highlight()
        self._refresh_transform_gizmo(render=False)
        if self._plotter is not None and (not self.off_screen):
            self.plotter.render()
        return self._selected_scan_id

    @staticmethod
    def _mode_token(mode: DisplayMode) -> str:
        return {DisplayMode.PHOTO_RGB: 'rgb', DisplayMode.CONFIDENCE: 'confcolor', DisplayMode.RUN_TINT: 'tint', DisplayMode.HEIGHT: 'height'}[mode]

    def _current_visibility(self) -> tuple[dict[str, bool], dict[str, bool]]:
        plan = self._render_plan
        if plan is None:
            return ({}, {})
        bands = {band: any((item.band == band and item.band_visible for item in plan.actors)) for band in ('core', 'body', 'shell')}
        scans = {scan_id: any((item.scan_id == scan_id and item.scan_visible for item in plan.actors)) for scan_id in {item.scan_id for item in plan.actors}}
        return (bands, scans)

    def _available_mode_tokens(self) -> str:
        if self._project is None:
            return 'none'
        bands, scans = self._current_visibility()
        available: list[str] = []
        for mode in DisplayMode:
            candidate = resolve_render_plan(self._project, mode, point_size=min(self.point_size, 10.0), band_visibility=bands or None, scan_visibility=scans or None)
            if candidate.complete:
                available.append(self._mode_token(mode))
        return ', '.join(available) or 'none'

    def set_color_mode(self, mode: str) -> str:
        try:
            resolved = resolve_display_mode(mode)
        except LayerControllerError as exc:
            raise PyVistaViewportError(str(exc)) from exc
        token = self._mode_token(resolved)
        if self._project is None:
            self._color_mode = token
            return token
        if self._render_plan is not None and resolved is self._render_plan.display_mode:
            self._color_mode = token
            return token
        bands, scans = self._current_visibility()
        try:
            plan = resolve_render_plan(self._project, resolved, point_size=min(self.point_size, 10.0), band_visibility=bands or None, scan_visibility=scans or None, require_available=True)
        except DisplayModeUnavailableError as exc:
            available = self._available_mode_tokens()
            raise PyVistaViewportError(f'color mode is not loaded in this viewport: {mode}; available: {available}') from exc
        self.apply_render_plan(plan)
        return self._color_mode

    @property
    def evidence_grade_state(self) -> EvidenceGradeState:
        return self._evidence_grade_state

    @property
    def evidence_grade_results(self) -> dict[str, EvidenceGradeResult]:
        return dict(self._evidence_grade_results)

    def evidence_grade_result(self, scan_id: str) -> EvidenceGradeResult | None:
        for actor_id, request in self._actor_requests.items():
            if request.scan_id == scan_id:
                return self._evidence_grade_results.get(actor_id)
        return None

    def scan_range_summary(self, scan_id: str) -> dict[str, float] | None:
        for actor_id, request in self._actor_requests.items():
            if request.scan_id == scan_id:
                summary = self._range_summaries.get(actor_id)
                if summary is not None:
                    return dict(summary)
        return None

    def evidence_grade_summary(self) -> EvidenceGradeSummary | None:
        grades = [result.grade for result in self._evidence_grade_results.values() if result.available]
        if not grades:
            return None
        return summarize_grade(np.concatenate(grades), self._evidence_grade_state.pivot)

    def colour_gap_summary(self) -> ColourGapSummary | None:
        counts = [value for actor_id, value in self._colour_gap_counts.items() if actor_id in self._layer_actors]
        if not counts:
            return None
        return ColourGapSummary(point_count=int(sum((total for _gap, total in counts))), gap_count=int(sum((gap for gap, _total in counts))))

    def depth_fill_summary(self) -> DepthFillSummary | None:
        counts = [value for actor_id, value in self._depth_fill_counts.items() if actor_id in self._layer_actors]
        if not counts:
            return None
        return DepthFillSummary(point_count=int(sum((total for _labelled, total in counts))), labelled_count=int(sum((labelled for labelled, _total in counts))))

    @property
    def sky_fill_visible(self) -> bool:
        return self._sky_fill_visible

    def set_sky_fill_visible(self, visible: bool) -> bool:
        if type(visible) is not bool:
            raise PyVistaViewportError('sky fill visible must be boolean')
        self._sky_fill_visible = visible
        if self._render_plan is not None:
            self.apply_render_plan(self._render_plan)
        return self._sky_fill_visible

    def sky_fill_summary(self) -> SkyFillSummary | None:
        counts = [value for actor_id, value in self._sky_fill_counts.items() if actor_id in self._layer_actors]
        if not counts:
            return None
        return SkyFillSummary(point_count=int(sum((total for _hidden, total in counts))), hidden_count=int(sum((hidden for hidden, _total in counts))), visible=self._sky_fill_visible)

    def sky_fill_unavailable_reason(self) -> str | None:
        return None if self.sky_fill_summary() is not None else SKY_FILL_UNAVAILABLE_REASON

    @property
    def floating_visible(self) -> bool:
        return self._floating_visible

    @property
    def floating_threshold(self) -> float:
        return self._floating_threshold

    def set_floating_visible(self, visible: bool) -> bool:
        if type(visible) is not bool:
            raise PyVistaViewportError('floating visible must be boolean')
        self._floating_visible = visible
        if self._render_plan is not None:
            self.apply_render_plan(self._render_plan)
        return self._floating_visible

    def set_floating_threshold(self, t: float) -> float:
        value = float(t)
        if not FLOATING_THRESHOLD_MIN - 1e-09 <= value <= FLOATING_THRESHOLD_MAX + 1e-09:
            raise PyVistaViewportError(f'floating threshold must be between {FLOATING_THRESHOLD_MIN} and {FLOATING_THRESHOLD_MAX}')
        self._floating_threshold = value
        if self._render_plan is not None:
            self.apply_render_plan(self._render_plan)
        return self._floating_threshold

    def floating_summary(self) -> FloatingSummary | None:
        counts = [value for actor_id, value in self._floating_counts.items() if actor_id in self._layer_actors]
        if not counts:
            return None
        return FloatingSummary(point_count=int(sum((total for _hidden, total in counts))), hidden_count=int(sum((hidden for hidden, _total in counts))), visible=self._floating_visible, threshold=self._floating_threshold)

    def floating_unavailable_reason(self) -> str | None:
        return None if self.floating_summary() is not None else FLOATING_UNAVAILABLE_REASON

    def scan_floating_summary(self, scan_id: str) -> FloatingSummary | None:
        counts = [self._floating_counts[actor_id] for actor_id, request in self._actor_requests.items() if request.scan_id == scan_id and actor_id in self._floating_counts and (actor_id in self._layer_actors)]
        if not counts:
            return None
        return FloatingSummary(point_count=int(sum((total for _hidden, total in counts))), hidden_count=int(sum((hidden for hidden, _total in counts))), visible=self._floating_visible, threshold=self._floating_threshold)

    @property
    def sparse_visible(self) -> bool:
        return self._sparse_visible

    @property
    def sparse_threshold(self) -> float:
        return self._sparse_threshold

    def set_sparse_visible(self, visible: bool) -> bool:
        if type(visible) is not bool:
            raise PyVistaViewportError('sparse visible must be boolean')
        self._sparse_visible = visible
        if self._render_plan is not None:
            self.apply_render_plan(self._render_plan)
        return self._sparse_visible

    def set_sparse_threshold(self, s: float) -> float:
        value = float(s)
        if not SPARSE_THRESHOLD_MIN_M - 1e-09 <= value <= SPARSE_THRESHOLD_MAX_M + 1e-09:
            raise PyVistaViewportError(f'sparse threshold must be between {SPARSE_THRESHOLD_MIN_M} and {SPARSE_THRESHOLD_MAX_M}')
        self._sparse_threshold = value
        if self._render_plan is not None:
            self.apply_render_plan(self._render_plan)
        return self._sparse_threshold

    def sparse_summary(self) -> SparseSummary | None:
        counts = [value for actor_id, value in self._sparse_counts.items() if actor_id in self._layer_actors]
        if not counts:
            return None
        return SparseSummary(point_count=int(sum((total for _hidden, total in counts))), hidden_count=int(sum((hidden for hidden, _total in counts))), visible=self._sparse_visible, threshold_m=self._sparse_threshold)

    def sparse_default_threshold(self) -> float | None:
        values = [value for actor_id, value in self._sparse_defaults.items() if actor_id in self._layer_actors]
        return max(values) if values else None

    def scan_sparse_default_threshold(self, scan_id: str) -> float | None:
        values = [self._sparse_defaults[actor_id] for actor_id, request in self._actor_requests.items() if request.scan_id == scan_id and actor_id in self._sparse_defaults and (actor_id in self._layer_actors)]
        return max(values) if values else None

    def sparse_unavailable_reason(self) -> str | None:
        return None if self.sparse_summary() is not None else SPARSE_UNAVAILABLE_REASON

    def scan_sparse_summary(self, scan_id: str) -> SparseSummary | None:
        counts = [self._sparse_counts[actor_id] for actor_id, request in self._actor_requests.items() if request.scan_id == scan_id and actor_id in self._sparse_counts and (actor_id in self._layer_actors)]
        if not counts:
            return None
        return SparseSummary(point_count=int(sum((total for _hidden, total in counts))), hidden_count=int(sum((hidden for hidden, _total in counts))), visible=self._sparse_visible, threshold_m=self._sparse_threshold)

    def scan_sky_fill_summary(self, scan_id: str) -> SkyFillSummary | None:
        counts = [self._sky_fill_counts[actor_id] for actor_id, request in self._actor_requests.items() if request.scan_id == scan_id and actor_id in self._sky_fill_counts and (actor_id in self._layer_actors)]
        if not counts:
            return None
        return SkyFillSummary(point_count=int(sum((total for _hidden, total in counts))), hidden_count=int(sum((hidden for hidden, _total in counts))), visible=self._sky_fill_visible)

    def scan_depth_fill_summary(self, scan_id: str) -> DepthFillSummary | None:
        counts = [self._depth_fill_counts[actor_id] for actor_id, request in self._actor_requests.items() if request.scan_id == scan_id and actor_id in self._depth_fill_counts and (actor_id in self._layer_actors)]
        if not counts:
            return None
        return DepthFillSummary(point_count=int(sum((total for _labelled, total in counts))), labelled_count=int(sum((labelled for labelled, _total in counts))))

    def set_evidence_grade_enabled(self, enabled: bool) -> bool:
        if type(enabled) is not bool:
            raise PyVistaViewportError('evidence grade enabled must be boolean')
        self._evidence_grade_state = replace(self._evidence_grade_state, enabled=enabled)
        if self._render_plan is not None:
            self.apply_render_plan(self._render_plan)
        return self._evidence_grade_state.enabled

    def set_evidence_grade_pivot(self, pivot: float) -> float:
        try:
            state = EvidenceGradeState(enabled=self._evidence_grade_state.enabled, pivot=pivot)
        except EvidenceGradeError as exc:
            raise PyVistaViewportError(str(exc)) from exc
        self._evidence_grade_state = state
        if self._render_plan is not None:
            self.apply_render_plan(self._render_plan)
        return self._evidence_grade_state.pivot

    def _clear_scan_actors(self) -> None:
        for name in tuple(self._selection_box_actors):
            self.plotter.remove_actor(name, reset_camera=False, render=False)
        self._selection_box_actors.clear()
        self._selection_actor_state.clear()
        for actor_id, group in tuple(self._actor_groups.items()):
            for drawn in group:
                self.plotter.remove_actor(drawn.name, reset_camera=False, render=False)
            self._actor_bounds.pop(actor_id, None)
        for actor_id in tuple(self._layer_actors):
            self.plotter.remove_actor(actor_id, reset_camera=False, render=False)
            self._actor_bounds.pop(actor_id, None)
        self._scan_actors.clear()
        self._layer_actors.clear()
        self._actor_groups.clear()
        self._actor_requests.clear()
        self._actor_base_transforms.clear()
        self._scan_display_transforms.clear()
        self._scan_bounds.clear()
        self._placement_actor_geometry.clear()
        self._placement_actor_grade.clear()
        self._evidence_grade_results.clear()
        self._colour_gap_counts.clear()
        self._depth_fill_counts.clear()
        self._sky_fill_counts.clear()
        self._floating_counts.clear()
        self._floating_mask_cache.clear()
        self._sparse_counts.clear()
        self._sparse_mask_cache.clear()
        self._sparse_defaults.clear()
        self._range_summaries.clear()

    def _clear_project_actors(self) -> None:
        self.stop_rolove_pick()
        self.clear_rolove_surfaces(render=False)
        self._manual_drag = None
        self._remove_transform_gizmo()
        self._clear_scan_actors()
        for actor_name in tuple(self._surface_actors):
            self.plotter.remove_actor(actor_name, reset_camera=False, render=False)
            self._actor_bounds.pop(actor_name, None)
        self._surface_actors.clear()
        self._surface_base_transforms.clear()
        if self._plan_actor_name is not None:
            self.plotter.remove_actor(self._plan_actor_name, reset_camera=False, render=False)
        self._actor_bounds.clear()
        self._plan_actor = None
        self._plan_actor_name = None
        self._plan_base_transform = None
        self._plan_segment_count = 0
        self._selected_scan_id = None
        self._selected_plan = False
        self._color_mode = 'rgb'
        self._last_load_result = None
        self._render_plan = None
        self._project = None

    def show_surface_preview(self, scan_id: str, result: object, *, render: bool=True) -> str:
        if self._project is None:
            raise PyVistaViewportError('load a project before showing a surface')
        scan = self._project.scan(scan_id)
        if scan is None:
            raise PyVistaViewportError(f'unknown surface scan: {scan_id}')
        mesh = getattr(result, 'mesh', None)
        if mesh is None:
            raise PyVistaViewportError('surface result has no mesh')
        vertices = np.asarray(getattr(mesh, 'vertices', ()), dtype=np.float64)
        triangles = np.asarray(getattr(mesh, 'triangles', ()), dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or (not len(vertices)) or (not np.isfinite(vertices).all()):
            raise PyVistaViewportError('surface mesh has invalid vertices')
        if triangles.ndim != 2 or triangles.shape[1] != 3 or (not len(triangles)) or (triangles.min() < 0) or (triangles.max() >= len(vertices)):
            raise PyVistaViewportError('surface mesh has invalid triangles')
        pyvista, _vtk, _interactor = _load_renderer_dependencies()
        faces = np.column_stack((np.full(len(triangles), 3, dtype=np.int64), triangles)).reshape(-1)
        polydata = pyvista.PolyData(vertices.copy(), faces)
        colors = np.asarray(getattr(mesh, 'vertex_colors', ()), dtype=np.float64)
        if colors.shape == (len(vertices), 3) and np.isfinite(colors).all():
            if colors.size and float(colors.max()) <= 1.0:
                colors = colors * 255.0
            polydata.point_data['cca_surface_rgb'] = np.clip(np.rint(colors), 0.0, 255.0).astype(np.uint8)
        transformed = polydata.transform(placed_matrix(scan.transform, scan.metadata), transform_all_input_vectors=False, inplace=False)
        actor_name = f'surface::{scan_id}::bpa'
        if actor_name in self._surface_actors:
            self._selection_actor_state.pop(actor_name, None)
            self.plotter.remove_actor(actor_name, reset_camera=False, render=False)
        options: dict[str, Any] = {'name': actor_name, 'style': 'wireframe', 'opacity': GHOST_SURFACE_OPACITY, 'line_width': 1.0, 'show_edges': True, 'smooth_shading': False, 'reset_camera': False, 'pickable': True, 'render': False}
        if 'cca_surface_rgb' in transformed.point_data:
            options.update({'scalars': 'cca_surface_rgb', 'rgb': True})
        else:
            options['color'] = POINT_FALLBACK_COLOR
        actor = self.plotter.add_mesh(transformed, **options)
        self._surface_actors[actor_name] = actor
        self._surface_base_transforms[actor_name] = placed_matrix(scan.transform, scan.metadata)
        self._actor_bounds[actor_name] = _polydata_bounds(transformed)
        self._apply_surface_display_mode()
        if render and (not self.off_screen):
            self.plotter.render()
        return actor_name

    def remove_surface_preview(self, scan_id: str, *, render: bool=True) -> bool:
        actor_name = f'surface::{scan_id}::bpa'
        if actor_name not in self._surface_actors:
            return False
        self._selection_actor_state.pop(actor_name, None)
        self.plotter.remove_actor(actor_name, reset_camera=False, render=False)
        self._surface_actors.pop(actor_name, None)
        self._surface_base_transforms.pop(actor_name, None)
        self._actor_bounds.pop(actor_name, None)
        if render and (not self.off_screen):
            self.plotter.render()
        return True

    def _request_polydata(self, request: ActorRequest, maximum: int | None) -> tuple[Any, Path, int, EvidenceGradeResult, dict[str, float] | None]:
        pyvista, _vtk, _interactor = _load_renderer_dependencies()
        if not request.available or request.source_path is None:
            raise PyVistaViewportError(f'render source is unavailable: {request.actor_id}')
        source = request.source_path
        try:
            loaded = pyvista.read(source)
        except (OSError, RuntimeError, ValueError) as exc:
            raise PyVistaViewportError(f'cannot read scan: {request.scan_id}') from exc
        points = np.asarray(loaded.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or (not len(points)):
            raise PyVistaViewportError(f'{request.scan_id} has no point geometry')
        if not np.isfinite(points).all():
            raise PyVistaViewportError(f'{request.scan_id} contains non-finite XYZ')
        if source.suffix.lower() in MESH_SOURCE_SUFFIXES:
            mesh = loaded.extract_surface().triangulate()
            if not int(mesh.n_cells):
                raise PyVistaViewportError(f'{request.scan_id} mesh has no faces')
            if request.source_axis != 'SOURCE_LOCAL':
                mesh.points = points_to_working_axis(points, request.source_axis)
            transformed_mesh = mesh.transform(_matrix(request.transform, request.scan_id), transform_all_input_vectors=False, inplace=False)
            return (transformed_mesh, source, len(points), EvidenceGradeResult(available=False, unavailable_reason='legacy mesh has no per-point measurement evidence', grade=None, point_count=len(points)), None)
        points = points_to_working_axis(points, request.source_axis)
        colors = _rgb_values(loaded, len(points))
        evidence = _resolve_source_display(source, len(points))
        range_summary = evidence.range_summary
        selected = _sample_indices(len(points), maximum)
        self._floating_counts.pop(request.actor_id, None)
        votes = load_floating_votes(source, len(points))
        if votes is not None:
            hidden = _floating_hidden(votes.n_through[selected], votes.n_confirm[selected], t=self._floating_threshold)
            self._floating_counts[request.actor_id] = (int(np.count_nonzero(hidden)), int(len(selected)))
            if not self._floating_visible and hidden.any():
                selected = selected[~hidden]
                if not len(selected):
                    raise PyVistaViewportError(f'{request.scan_id}: every point is parked by HAVADA')
        self._sparse_counts.pop(request.actor_id, None)
        self._sparse_defaults.pop(request.actor_id, None)
        kdist = load_sparse_sidecar(source, len(points))
        if kdist is not None:
            run_default = _run_sparse_default(source)
            if run_default is not None:
                self._sparse_defaults[request.actor_id] = run_default
            isolated = _sparse_hidden(kdist[selected], s=self._sparse_threshold)
            self._sparse_counts[request.actor_id] = (int(np.count_nonzero(isolated)), len(selected))
            if not self._sparse_visible and isolated.any():
                selected = selected[~isolated]
                if not len(selected):
                    raise PyVistaViewportError(f'{request.scan_id}: every point is parked by SEYREK')
        grade_result = _sample_grade(evidence.grade, selected)
        gap = None if evidence.gap is None else evidence.gap[selected]
        fill = None if evidence.fill_share is None else evidence.fill_share[selected]
        polydata = pyvista.PolyData(points[selected].copy())
        if colors is not None:
            display_colors = colors[selected].copy()
            if gap is not None:
                display_colors = _apply_colour_gap_display(display_colors, gap)
            if fill is not None:
                display_colors = _apply_depth_fill_display(display_colors, fill)
            polydata.point_data['cca_rgb'] = display_colors
        if gap is not None:
            polydata.point_data[COLOUR_GAP_SCALAR] = gap.astype(np.uint8)
        if fill is not None:
            polydata.point_data[DEPTH_FILL_SCALAR] = fill
        if request.color.strategy is ColorStrategy.CONFIDENCE_RGB and 'cca_rgb' not in polydata.point_data:
            raise PyVistaViewportError(f'confidence RGB is missing: {source}')
        if grade_result.available:
            polydata.point_data[EVIDENCE_GRADE_SCALAR] = grade_result.grade
        transformed = polydata.transform(_matrix(request.transform, request.scan_id), transform_all_input_vectors=False, inplace=False)
        if request.color.strategy is ColorStrategy.WORLD_Z:
            transformed.point_data['height_m'] = np.asarray(transformed.points[:, 2], dtype=np.float64)
        return (transformed, source, len(points), grade_result, range_summary)

    @staticmethod
    def _merge_bounds(first: tuple[float, float, float, float, float, float] | None, second: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
        if first is None:
            return second
        return (min(first[0], second[0]), max(first[1], second[1]), min(first[2], second[2]), max(first[3], second[3]), min(first[4], second[4]), max(first[5], second[5]))

    def _add_render_actor(self, request: ActorRequest, polydata: Any, source: Path, source_count: int, height_range: tuple[float, float] | None, grade_result: EvidenceGradeResult, range_summary: dict[str, float] | None=None) -> LoadedScan:
        actor_name = request.actor_id
        mesh_source = source.suffix.lower() in MESH_SOURCE_SUFFIXES
        options: dict[str, Any] = {'name': actor_name, 'style': 'surface', 'show_edges': True, 'line_width': 0.5, 'opacity': 0.78, 'reset_camera': False, 'pickable': True, 'render': False} if mesh_source else {'name': actor_name, 'style': 'points', 'point_size': self.point_size, 'render_points_as_spheres': True, 'reset_camera': False, 'pickable': True, 'render': False}
        alpha: np.ndarray | None = None
        if self._evidence_grade_state.enabled and grade_result.available:
            alpha = display_alpha(grade_result.grade, self._evidence_grade_state.pivot)
        strategy = request.color.strategy
        if strategy in {ColorStrategy.SOURCE_RGB, ColorStrategy.CONFIDENCE_RGB}:
            texture_path = _packaged_mesh_texture(source) if mesh_source else None
            texture_coordinates = getattr(polydata, 'active_texture_coordinates', None)
            if strategy is ColorStrategy.SOURCE_RGB and texture_path is not None and (texture_coordinates is not None):
                try:
                    options['texture'] = self._pyvista.read_texture(texture_path)
                    options['show_edges'] = False
                    options['opacity'] = 1.0
                except (OSError, RuntimeError, ValueError):
                    options['color'] = POINT_FALLBACK_COLOR
            elif 'cca_rgb' in polydata.point_data:
                if alpha is None:
                    options.update({'scalars': 'cca_rgb', 'rgb': True})
                else:
                    rgb = np.asarray(polydata.point_data['cca_rgb'], dtype=np.uint8)
                    alpha_channel = np.clip(np.rint(alpha * 255.0), 0.0, 255.0).astype(np.uint8)
                    polydata.point_data['cca_rgba'] = np.column_stack((rgb, alpha_channel))
                    options.update({'scalars': 'cca_rgba', 'rgb': True})
            elif strategy is ColorStrategy.CONFIDENCE_RGB:
                raise PyVistaViewportError(f'confidence RGB is missing: {request.source_path}')
            else:
                options['color'] = POINT_FALLBACK_COLOR
                if alpha is not None:
                    options['opacity'] = alpha
        elif strategy is ColorStrategy.UNIFORM_RGB:
            options['color'] = request.color.uniform_rgb
            if alpha is not None:
                options['opacity'] = alpha
        elif strategy is ColorStrategy.WORLD_Z:
            options.update({'scalars': request.color.scalar_name, 'cmap': request.color.colormap, 'clim': height_range, 'show_scalar_bar': False})
            if alpha is not None:
                options['opacity'] = alpha
        elif strategy is ColorStrategy.EVIDENCE_GRADE:
            if EVIDENCE_GRADE_SCALAR in polydata.point_data:
                options.update({'scalars': EVIDENCE_GRADE_SCALAR, 'cmap': EVIDENCE_GRADE_COLORMAP, 'clim': (0.0, 1.0), 'show_scalar_bar': False})
            else:
                options['color'] = POINT_FALLBACK_COLOR
            if alpha is not None:
                options['opacity'] = alpha
        else:
            raise PyVistaViewportError(f'unsupported color strategy: {strategy}')
        actor = self.plotter.add_mesh(polydata, **options) if mesh_source else self.plotter.add_points(polydata, **options)
        self._layer_actors[actor_name] = actor
        self._actor_groups[actor_name] = (DrawnActor(actor_name, actor, 'MESH' if mesh_source else 'POINTS', 1.0),)
        self._actor_requests[actor_name] = request
        self._actor_base_transforms[actor_name] = np.asarray(request.transform, dtype=np.float64)
        self._scan_actors.setdefault(request.scan_id, actor)
        self._evidence_grade_results[actor_name] = grade_result
        self._range_summaries[actor_name] = range_summary
        if COLOUR_GAP_SCALAR in polydata.point_data:
            self._colour_gap_counts[actor_name] = (int(np.count_nonzero(polydata.point_data[COLOUR_GAP_SCALAR])), int(polydata.n_points))
        if DEPTH_FILL_SCALAR in polydata.point_data:
            self._depth_fill_counts[actor_name] = (_depth_fill_labelled_count(np.asarray(polydata.point_data[DEPTH_FILL_SCALAR])), int(polydata.n_points))
        bounds = _polydata_bounds(polydata)
        self._scan_bounds[request.scan_id] = self._merge_bounds(self._scan_bounds.get(request.scan_id), bounds)
        self._actor_bounds[actor_name] = bounds
        return LoadedScan(scan_id=request.scan_id, actor_name=actor_name, source_path=source, source_point_count=source_count, displayed_point_count=int(polydata.n_points), band=request.band, display_mode=request.display_mode, representation='MESH' if mesh_source else 'POINTS', triangle_count=int(polydata.n_cells) if mesh_source else 0)

    def _build_plan_actor(self, project: ProjectModel, actor_name: str) -> tuple:
        pyvista, _vtk, _interactor = _load_renderer_dependencies()
        plan = project.dxf_plan
        source = _resolve_project_path(project, plan.geometry_path, 'DXF plan geometry')
        try:
            payload = json.loads(source.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise PyVistaViewportError('cannot read DXF plan geometry') from exc
        if payload.get('schema') != 'cca.plan-geometry.v1':
            raise PyVistaViewportError('unsupported DXF plan geometry schema')
        if payload.get('output_unit') != 'meter':
            raise PyVistaViewportError('DXF plan geometry must be in metres')
        segments = payload.get('segments')
        if not isinstance(segments, list) or not segments:
            raise PyVistaViewportError('DXF plan geometry has no segments')
        points: list[list[float]] = []
        lines: list[int] = []
        for index, segment in enumerate(segments):
            segment_points = segment.get('points_m') if isinstance(segment, dict) else None
            try:
                values = np.asarray(segment_points, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise PyVistaViewportError(f'DXF plan segment {index} is invalid') from exc
            if values.shape != (2, 3) or not np.isfinite(values).all():
                raise PyVistaViewportError(f'DXF plan segment {index} is invalid')
            first = len(points)
            points.extend(values.tolist())
            lines.extend((2, first, first + 1))
        polydata = pyvista.PolyData(np.asarray(points, dtype=np.float64))
        polydata.lines = np.asarray(lines, dtype=np.int64)
        transformed = polydata.transform(_matrix(plan.transform, 'DXF plan'), transform_all_input_vectors=False, inplace=False)
        actor = self.plotter.add_mesh(transformed, name=actor_name, color=PLAN_COLOR, line_width=PLAN_LINE_WIDTH, opacity=PLAN_OPACITY, lighting=False, pickable=False, reset_camera=False, render=False)
        return (actor, _polydata_bounds(transformed), len(segments))

    def _add_plan(self, project: ProjectModel) -> tuple[str, int]:
        actor_name = 'plan::dxf'
        actor, bounds, count = self._build_plan_actor(project, actor_name)
        self._plan_actor = actor
        self._plan_actor_name = actor_name
        self._plan_base_transform = np.asarray(project.dxf_plan.transform, dtype=np.float64)
        self._plan_segment_count = count
        self._actor_bounds[actor_name] = bounds
        return (actor_name, count)

    def _set_actor_transform_delta(self, actor: Any, *, base_transform: np.ndarray, target_transform: object, label: str) -> tuple[float, float, float, float, float, float]:
        target = _matrix(target_transform, label)
        if base_transform.shape != (4, 4) or not np.isfinite(base_transform).all():
            raise PyVistaViewportError(f'{label} base transform is invalid')
        try:
            delta = target @ np.linalg.inv(base_transform)
        except np.linalg.LinAlgError as exc:
            raise PyVistaViewportError(f'{label} base transform is singular') from exc
        vtk = self._vtk
        if vtk is None:
            _pyvista, vtk, _interactor = _load_renderer_dependencies()
        matrix = vtk.vtkMatrix4x4()
        for row in range(4):
            for column in range(4):
                matrix.SetElement(row, column, float(delta[row, column]))
        actor.SetUserMatrix(matrix)
        bounds = tuple((float(value) for value in actor.GetBounds()))
        if len(bounds) != 6 or not np.isfinite(np.asarray(bounds)).all():
            raise PyVistaViewportError(f'{label} produced invalid actor bounds')
        return bounds

    def update_scan_transform(self, scan_id: str, transform: object, *, render: bool=True) -> bool:
        if self._project is None or self._project.scan(scan_id) is None:
            raise PyVistaViewportError(f'unknown scan transform target: {scan_id}')
        placement_target = transform
        transform = placed_matrix(transform, self._project.scan(scan_id).metadata)
        actor_ids = [actor_id for actor_id, request in self._actor_requests.items() if request.scan_id == scan_id]
        if not actor_ids:
            return False
        merged: tuple[float, float, float, float, float, float] | None = None
        for actor_id in actor_ids:
            bounds = None
            for drawn in self._actor_groups[actor_id]:
                bounds = self._merge_bounds(bounds, self._set_actor_transform_delta(drawn.actor, base_transform=self._actor_base_transforms[actor_id], target_transform=transform, label=f'{scan_id} transform'))
            self._actor_bounds[actor_id] = bounds
            merged = self._merge_bounds(merged, bounds)
        self._scan_bounds[scan_id] = merged
        surface_name = f'surface::{scan_id}::bpa'
        surface = self._surface_actors.get(surface_name)
        if surface is not None:
            bounds = self._set_actor_transform_delta(surface, base_transform=self._surface_base_transforms[surface_name], target_transform=transform, label=f'{scan_id} surface transform')
            self._actor_bounds[surface_name] = bounds
        if self._selected_scan_id == scan_id:
            self._refresh_transform_gizmo(render=False)
        if self._rolove_surface_scan_id == scan_id and self._rolove_actor_names:
            self.show_rolove_surfaces(scan_id, surfaces_in_world(placement_target, self._project.scan(scan_id).metadata))
        if render and (not self.off_screen):
            self.plotter.render()
        return True

    @staticmethod
    def _vtk_matrix_array(matrix: Any | None) -> np.ndarray:
        if matrix is None:
            return np.eye(4, dtype=np.float64)
        return np.asarray([[matrix.GetElement(row, column) for column in range(4)] for row in range(4)], dtype=np.float64)

    def _scan_display_points(self, scan_id: str) -> np.ndarray:
        actor_ids = [actor_id for actor_id, request in self._actor_requests.items() if request.scan_id == scan_id]
        if not actor_ids:
            raise PyVistaViewportError(f'scan is not loaded: {scan_id}')
        batches: list[np.ndarray] = []
        for actor_id in actor_ids:
            geometry = self._placement_actor_geometry.get(actor_id)
            for drawn in self._actor_groups.get(actor_id, ()):
                if drawn.kind != 'POINTS' or not drawn.actor.GetVisibility():
                    continue
                if geometry is not None and geometry.representation == 'POINTS':
                    points = np.asarray(geometry.vertices, dtype=np.float64)
                    matrix = self._vtk_matrix_array(drawn.actor.GetUserMatrix())
                    if not np.isfinite(matrix).all():
                        raise PyVistaViewportError(f'{scan_id} display matrix is invalid')
                    batches.append(points @ matrix[:3, :3].T + matrix[:3, 3])
                    break
                points = np.asarray(drawn.actor.mapper.dataset.points, dtype=np.float64)
                if points.ndim != 2 or points.shape[1] != 3 or (not len(points)):
                    continue
                matrix = self._vtk_matrix_array(drawn.actor.GetUserMatrix())
                if not np.isfinite(matrix).all():
                    raise PyVistaViewportError(f'{scan_id} display matrix is invalid')
                batches.append(points @ matrix[:3, :3].T + matrix[:3, 3])
        if not batches:
            raise PyVistaViewportError(f'scan has no display points: {scan_id}')
        return np.vstack(batches)

    def scan_inspection(self, scan_id: str) -> dict[str, object]:
        points = self._scan_display_points(scan_id)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        xy = points[:, :2]
        centre_xy = xy.mean(axis=0)
        centred_xy = xy - centre_xy
        covariance = centred_xy.T @ centred_xy / max(len(centred_xy), 1)
        values, vectors = np.linalg.eigh(covariance)
        long_axis = np.asarray(vectors[:, int(np.argmax(values))], dtype=np.float64)
        norm = float(np.linalg.norm(long_axis))
        if not math.isfinite(norm) or norm <= 1e-12:
            long_axis = np.asarray((1.0, 0.0), dtype=np.float64)
        else:
            long_axis /= norm
        if long_axis[0] < 0.0 or (math.isclose(float(long_axis[0]), 0.0, abs_tol=1e-12) and long_axis[1] < 0.0):
            long_axis *= -1.0
        short_axis = np.asarray((-long_axis[1], long_axis[0]), dtype=np.float64)
        horizontal = np.column_stack((centred_xy @ long_axis, centred_xy @ short_axis))
        horizontal_min = horizontal.min(axis=0)
        horizontal_max = horizontal.max(axis=0)
        return {'point_count': len(points), 'aabb_m': {'min': tuple((float(value) for value in minimum)), 'max': tuple((float(value) for value in maximum)), 'size': tuple((float(value) for value in maximum - minimum))}, 'horizontal_pca_obb_m': {'center_xy': tuple((float(value) for value in centre_xy)), 'long_axis_xy': tuple((float(value) for value in long_axis)), 'short_axis_xy': tuple((float(value) for value in short_axis)), 'long_m': float(horizontal_max[0] - horizontal_min[0]), 'short_m': float(horizontal_max[1] - horizontal_min[1]), 'z_min_m': float(minimum[2]), 'z_max_m': float(maximum[2])}, 'range_m': self.scan_range_summary(scan_id), 'depth_fill': self._depth_fill_payload(scan_id)}

    def _depth_fill_payload(self, scan_id: str) -> dict[str, object] | None:
        summary = self.scan_depth_fill_summary(scan_id)
        if summary is None:
            return None
        return {'point_count': summary.point_count, 'labelled_count': summary.labelled_count, 'labelled_fraction': summary.labelled_fraction}

    def _apply_scan_display_transform(self, scan_id: str, display_transform: np.ndarray, *, render: bool) -> bool:
        actor_ids = [actor_id for actor_id, request in self._actor_requests.items() if request.scan_id == scan_id]
        if not actor_ids:
            return False
        merged: tuple[float, float, float, float, float, float] | None = None
        for actor_id in actor_ids:
            base = self._actor_base_transforms[actor_id]
            bounds = None
            for drawn in self._actor_groups[actor_id]:
                bounds = self._merge_bounds(bounds, self._set_actor_transform_delta(drawn.actor, base_transform=base, target_transform=display_transform @ placed_matrix(self._project.scan(scan_id).transform, self._project.scan(scan_id).metadata), label=f'{scan_id} display layout'))
            self._actor_bounds[actor_id] = bounds
            merged = self._merge_bounds(merged, bounds)
        self._scan_bounds[scan_id] = merged
        surface_name = f'surface::{scan_id}::bpa'
        surface = self._surface_actors.get(surface_name)
        if surface is not None:
            base = self._surface_base_transforms[surface_name]
            self._actor_bounds[surface_name] = self._set_actor_transform_delta(surface, base_transform=base, target_transform=display_transform @ placed_matrix(self._project.scan(scan_id).transform, self._project.scan(scan_id).metadata), label=f'{scan_id} surface display layout')
        if self._selected_scan_id == scan_id:
            self._refresh_transform_gizmo(render=False)
        if render and (not self.off_screen):
            self.plotter.render()
        return True

    def reset_scan_layout(self, *, render: bool=True) -> tuple[str, ...]:
        reset_ids = tuple(self._scan_display_transforms)
        self._clear_scan_layout_labels()
        identity = np.eye(4, dtype=np.float64)
        for scan_id in reset_ids:
            self._apply_scan_display_transform(scan_id, identity, render=False)
        self._scan_display_transforms.clear()
        if render and reset_ids and (not self.off_screen):
            self.plotter.render()
        return reset_ids

    def _clear_scan_layout_labels(self) -> None:
        name = self._scan_layout_label_actor_name
        if name is not None:
            self.plotter.remove_actor(name, reset_camera=False, render=False)
        self._scan_layout_label_actor_name = None

    def _draw_scan_layout_labels(self, points: list[np.ndarray], labels: list[str]) -> None:
        self._clear_scan_layout_labels()
        if not points:
            return
        name = 'scan-layout-chronology'
        self.plotter.add_point_labels(np.vstack(points), labels, name=name, show_points=False, always_visible=True, font_size=16, text_color='#F3F6F8', shape_opacity=0.72, shape_color='#1C2229', render=False)
        self._scan_layout_label_actor_name = name

    def apply_scan_layout(self, scan_ids: tuple[str, ...] | list[str] | None=None, *, gap_m: float=SCAN_LAYOUT_GAP_M, render: bool=True) -> tuple[str, ...]:
        if not math.isfinite(float(gap_m)) or float(gap_m) < 0.0:
            raise PyVistaViewportError('scan layout gap must be a finite non-negative metre value')
        self.stop_rolove_pick(cancel=True)
        self.clear_rolove_surfaces(render=False)
        self.reset_scan_layout(render=False)
        requested = tuple(scan_ids) if scan_ids is not None else tuple(self._scan_bounds)
        active_ids = tuple((scan_id for scan_id in requested if scan_id in self._scan_bounds))
        cursor_y = 0.0
        applied: list[str] = []
        for scan_id in active_ids:
            inspection = self.scan_inspection(scan_id)
            obb = inspection['horizontal_pca_obb_m']
            if not isinstance(obb, dict):
                raise PyVistaViewportError('scan layout inspection is invalid')
            axis = np.asarray(obb['long_axis_xy'], dtype=np.float64)
            angle = math.atan2(float(axis[1]), float(axis[0]))
            cosine = math.cos(-angle)
            sine = math.sin(-angle)
            rotation = np.asarray(((cosine, -sine, 0.0, 0.0), (sine, cosine, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)), dtype=np.float64)
            points = self._scan_display_points(scan_id)
            rotated = points @ rotation[:3, :3].T
            lower = rotated.min(axis=0)
            upper = rotated.max(axis=0)
            translation = np.eye(4, dtype=np.float64)
            translation[0, 3] = -float(lower[0])
            translation[1, 3] = cursor_y - float(lower[1])
            display = translation @ rotation
            self._apply_scan_display_transform(scan_id, display, render=False)
            self._scan_display_transforms[scan_id] = display
            cursor_y += float(upper[1] - lower[1]) + float(gap_m)
            applied.append(scan_id)
        if render and applied and (not self.off_screen):
            self.plotter.render()
        return tuple(applied)

    def apply_scan_layout_around_plan(self, scan_ids: tuple[str, ...] | list[str] | None=None, *, gap_m: float=SCAN_LAYOUT_GAP_M, render: bool=True) -> tuple[str, ...]:
        if not math.isfinite(float(gap_m)) or float(gap_m) < 0.0:
            raise PyVistaViewportError('scan layout gap must be a finite non-negative metre value')
        if self._plan_actor_name is None:
            raise PyVistaViewportError('load a DXF plan before using the around-plan scan layout')
        plan_bounds = self._actor_bounds.get(self._plan_actor_name)
        if plan_bounds is None:
            raise PyVistaViewportError('DXF plan bounds are unavailable')
        self.stop_rolove_pick(cancel=True)
        self.clear_rolove_surfaces(render=False)
        self.reset_scan_layout(render=False)
        requested = tuple(scan_ids) if scan_ids is not None else tuple(self._scan_bounds)
        active_ids = tuple((scan_id for scan_id in requested if scan_id in self._scan_bounds))
        if not active_ids:
            return ()
        points_by_scan = {scan_id: self._scan_display_points(scan_id) for scan_id in active_ids}
        half_extents = [(points[:, :2].max(axis=0) - points[:, :2].min(axis=0)) / 2.0 for points in points_by_scan.values()]
        max_half = np.max(np.vstack(half_extents), axis=0)
        xmin, xmax, ymin, ymax, _zmin, zmax = plan_bounds
        pad_x = max(5.0, float(max_half[0]) + float(gap_m))
        pad_y = max(5.0, float(max_half[1]) + float(gap_m))
        target_step = max(float(2.0 * np.max(max_half) + gap_m), 1.0)
        base_perimeter = 2.0 * (xmax - xmin + 2.0 * pad_x + (ymax - ymin) + 2.0 * pad_y)
        extra = max(0.0, (len(active_ids) * target_step - base_perimeter) / 8.0)
        left, right = (xmin - pad_x - extra, xmax + pad_x + extra)
        bottom, top = (ymin - pad_y - extra, ymax + pad_y + extra)
        width, height = (right - left, top - bottom)
        perimeter = 2.0 * (width + height)
        label_z = max(float(zmax), max((float(points[:, 2].max()) for points in points_by_scan.values()))) + max(1.0, 0.02 * max(width, height))

        def around_rectangle(distance: float) -> tuple[float, float]:
            if distance < width:
                return (left + distance, top)
            distance -= width
            if distance < height:
                return (right, top - distance)
            distance -= height
            if distance < width:
                return (right - distance, bottom)
            distance -= width
            return (left, bottom + (height - distance))
        labels: list[str] = []
        label_points: list[np.ndarray] = []
        for index, scan_id in enumerate(active_ids, start=1):
            target_x, target_y = around_rectangle(perimeter * (index - 1) / len(active_ids))
            center = points_by_scan[scan_id].mean(axis=0)
            display = np.eye(4, dtype=np.float64)
            display[0, 3] = target_x - float(center[0])
            display[1, 3] = target_y - float(center[1])
            self._apply_scan_display_transform(scan_id, display, render=False)
            self._scan_display_transforms[scan_id] = display
            labels.append(f'{index:02d}')
            label_points.append(np.asarray((target_x, target_y, label_z), dtype=np.float64))
        self._draw_scan_layout_labels(label_points, labels)
        if render and (not self.off_screen):
            self.plotter.render()
        return active_ids

    def update_plan_transform(self, transform: object, *, render: bool=True) -> bool:
        if self._plan_actor is None or self._plan_actor_name is None:
            return False
        if self._plan_base_transform is None:
            raise PyVistaViewportError('DXF plan base transform is unavailable')
        bounds = self._set_actor_transform_delta(self._plan_actor, base_transform=self._plan_base_transform, target_transform=transform, label='DXF plan transform')
        self._actor_bounds[self._plan_actor_name] = bounds
        if self._selected_plan:
            self._refresh_transform_gizmo(render=False)
        if render and (not self.off_screen):
            self.plotter.render()
        return True

    def _placement_actor(self, request: ActorRequest, geometry: _PlacementGeometry, name: str, height_range: tuple[float, float] | None, *, select: np.ndarray | None=None, point_size: float | None=None) -> Any:
        pyvista = self._pyvista
        faces = geometry.faces
        face_mask = self._placement_proxy_face_mask(geometry)
        if face_mask is not None:
            faces = faces[face_mask]
        pivot = self._placement_actor_pivot(geometry)
        alpha: np.ndarray | None = None
        if pivot is not None:
            alpha = display_alpha(geometry.grade, pivot)
            if geometry.representation == 'PROXY':
                alpha = np.repeat(alpha, 4)
        vertices = geometry.vertices
        colors = geometry.colors
        grade = geometry.grade
        if select is not None:
            if faces is not None:
                raise PyVistaViewportError('a surface representation cannot be split')
            vertices = vertices[select]
            colors = None if colors is None else colors[select]
            grade = None if grade is None else grade[select]
            alpha = None if alpha is None else alpha[select]
        if faces is None:
            data = pyvista.PolyData(vertices)
        else:
            cells = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()
            data = pyvista.PolyData(vertices, cells)
        options: dict[str, Any] = {'name': name, 'reset_camera': False, 'pickable': True, 'render': False, 'lighting': False}
        strategy = request.color.strategy
        if strategy in {ColorStrategy.SOURCE_RGB, ColorStrategy.CONFIDENCE_RGB}:
            if geometry.texture_rgb is not None:
                data.active_texture_coordinates = geometry.texture_coordinates
                image = geometry.texture_rgb
                if alpha is not None and geometry.texture_grade is not None:
                    texel_alpha = display_alpha(geometry.texture_grade.ravel(), pivot).reshape(geometry.texture_grade.shape)
                    if geometry.texture_covered is not None:
                        texel_alpha = np.where(geometry.texture_covered, texel_alpha, 0.0)
                    image = np.dstack((image, np.clip(np.rint(texel_alpha * 255.0), 0.0, 255.0).astype(np.uint8)))
                texture = pyvista.Texture(image)
                texture.interpolate = False
                texture.repeat = False
                texture.SetEdgeClamp(True)
                options['texture'] = texture
            elif colors is not None:
                if alpha is None:
                    data.point_data['cca_rgb'] = colors
                    options.update(scalars='cca_rgb', rgb=True)
                else:
                    data.point_data['cca_rgba'] = np.column_stack((colors, np.clip(np.rint(alpha * 255.0), 0.0, 255.0).astype(np.uint8)))
                    options.update(scalars='cca_rgba', rgb=True)
            else:
                options['color'] = POINT_FALLBACK_COLOR
                if alpha is not None:
                    options['opacity'] = alpha
        elif strategy is ColorStrategy.UNIFORM_RGB:
            options['color'] = request.color.uniform_rgb
            if alpha is not None:
                options['opacity'] = alpha
        elif strategy is ColorStrategy.WORLD_Z:
            transform = _matrix(request.transform, request.scan_id)
            data.point_data['height_m'] = vertices @ transform[2, :3] + transform[2, 3]
            options.update(scalars='height_m', cmap=request.color.colormap, clim=height_range, show_scalar_bar=False)
            if alpha is not None:
                options['opacity'] = alpha
        elif strategy is ColorStrategy.EVIDENCE_GRADE:
            if grade is not None:
                if geometry.representation == 'PROXY':
                    grade = np.repeat(grade, 4)
                data.point_data[EVIDENCE_GRADE_SCALAR] = np.asarray(grade, dtype=np.float32)
                options.update(scalars=EVIDENCE_GRADE_SCALAR, cmap=EVIDENCE_GRADE_COLORMAP, clim=(0.0, 1.0), show_scalar_bar=False)
            else:
                options['color'] = POINT_FALLBACK_COLOR
            if alpha is not None:
                options['opacity'] = alpha
        if geometry.representation in {'PROXY', 'MESH'}:
            return self.plotter.add_mesh(data, style='surface', **options)
        return self.plotter.add_points(data, point_size=request.point_size if point_size is None else point_size, render_points_as_spheres=True, **options)

    def _placement_actor_pivot(self, geometry: _PlacementGeometry) -> float | None:
        if geometry.grade is None or not self._evidence_grade_state.enabled:
            return None
        return self._evidence_grade_state.pivot

    def _placement_display_key(self, geometry: _PlacementGeometry) -> tuple[float | None, bool | None, tuple[bool, float] | None]:
        carries_sky = geometry.sky_hidden is not None or geometry.patch_sky_hidden is not None
        carries_floating = geometry.floating_through is not None or geometry.patch_floating_t is not None
        carries_sparse = geometry.sparse_kdist is not None or geometry.patch_sparse_m is not None
        return (self._placement_actor_pivot(geometry), self._sky_fill_visible if carries_sky else None, (self._floating_visible, self._floating_threshold) if carries_floating else None, (self._sparse_visible, self._sparse_threshold) if carries_sparse else None)

    def _placement_floating_hidden(self, geometry: _PlacementGeometry) -> np.ndarray | None:
        if geometry.representation == 'PROXY':
            return None if geometry.cell_floating_t is None else geometry.cell_floating_t > self._floating_threshold
        if geometry.representation != 'POINTS' or geometry.floating_through is None:
            return None
        cache_key = (geometry.key, float(self._floating_threshold))
        cached = self._floating_mask_cache.get(cache_key)
        if cached is None or len(cached) != len(geometry.floating_through):
            cached = _floating_hidden(geometry.floating_through, geometry.floating_confirm, t=self._floating_threshold)
            self._floating_mask_cache[cache_key] = cached
        return cached

    def _placement_sparse_hidden(self, geometry: _PlacementGeometry) -> np.ndarray | None:
        if geometry.representation == 'PROXY':
            return None if geometry.cell_sparse_m is None else geometry.cell_sparse_m > self._sparse_threshold
        if geometry.representation != 'POINTS' or geometry.sparse_kdist is None:
            return None
        cache_key = (geometry.key, float(self._sparse_threshold))
        cached = self._sparse_mask_cache.get(cache_key)
        if cached is None or len(cached) != len(geometry.sparse_kdist):
            cached = _sparse_hidden(geometry.sparse_kdist, s=self._sparse_threshold)
            self._sparse_mask_cache[cache_key] = cached
        return cached

    def _placement_proxy_face_mask(self, geometry: _PlacementGeometry) -> np.ndarray | None:
        if geometry.representation != 'PROXY' or geometry.faces is None:
            return None
        hidden: np.ndarray | None = None
        if geometry.patch_sky_hidden is not None and (not self._sky_fill_visible):
            hidden = geometry.patch_sky_hidden
        if not self._floating_visible and geometry.patch_floating_t is not None:
            floating = geometry.patch_floating_t > self._floating_threshold
            hidden = floating if hidden is None else hidden | floating
        if not self._sparse_visible and geometry.patch_sparse_m is not None:
            isolated = geometry.patch_sparse_m > self._sparse_threshold
            hidden = isolated if hidden is None else hidden | isolated
        if hidden is None or not hidden.any():
            return None
        return np.repeat(~hidden, 2)

    def _placement_drawn_mask(self, geometry: _PlacementGeometry) -> np.ndarray | None:
        if geometry.representation != 'POINTS':
            return None
        keep: np.ndarray | None = None
        if geometry.sky_hidden is not None and (not self._sky_fill_visible):
            keep = ~geometry.sky_hidden
        if not self._floating_visible:
            floating = self._placement_floating_hidden(geometry)
            if floating is not None:
                keep = ~floating if keep is None else keep & ~floating
        if not self._sparse_visible:
            isolated = self._placement_sparse_hidden(geometry)
            if isolated is not None:
                keep = ~isolated if keep is None else keep & ~isolated
        return keep

    def _placement_actor_group(self, request: ActorRequest, geometry: _PlacementGeometry, prefix: str, final: str, height_range: tuple[float, float] | None) -> tuple[DrawnActor, ...]:
        group: list[DrawnActor] = []

        def add(suffix: str, source: _PlacementGeometry, kind: str, factor: float, select: np.ndarray | None=None) -> None:
            temporary = prefix if not suffix else f'{prefix}#{suffix}'
            group.append(DrawnActor(final if not suffix else f'{final}#{suffix}', self._placement_actor(request, source, temporary, height_range, select=select, point_size=None if kind == 'SURFACE' else request.point_size * factor), kind, factor, select))
        try:
            return self._build_placement_group(request, geometry, add, group)
        except Exception:
            for drawn in group:
                self.plotter.remove_actor(drawn.actor, reset_camera=False, render=False)
            raise

    def _build_placement_group(self, request: ActorRequest, geometry: _PlacementGeometry, add: Callable[..., None], group: list[DrawnActor]) -> tuple[DrawnActor, ...]:
        if geometry.representation in {'PROXY', 'MESH'}:
            add('', geometry, 'SURFACE', 1.0)
            return tuple(group)
        keep = self._placement_drawn_mask(geometry)
        pivot = self._placement_actor_pivot(geometry)
        if pivot is None and geometry.surface_support is None:
            buckets: tuple[SizeBucket, ...] = (SizeBucket(np.ones(len(geometry.vertices), dtype=bool), 1.0),)
        else:
            buckets = _size_buckets(None if pivot is None else display_alpha(geometry.grade, pivot), surface_support=geometry.surface_support)
        single = keep is None and len(buckets) == 1
        for index, bucket in enumerate(buckets):
            mask = bucket.mask if keep is None else bucket.mask & keep
            if not mask.any():
                continue
            add('' if not group else f'b{index}', geometry, 'POINTS', float(bucket.factor), None if single else mask)
        if not group:
            add('', geometry, 'POINTS', 1.0, np.zeros(len(geometry.vertices), dtype=bool))
        return tuple(group)

    def _placement_grade_result(self, geometry: _PlacementGeometry) -> EvidenceGradeResult:
        grade = geometry.cell_grade if geometry.representation == 'PROXY' else geometry.grade
        if grade is not None:
            return EvidenceGradeResult(available=True, unavailable_reason=None, grade=grade, point_count=int(len(grade)))
        count = len(geometry.vertices) // 4 if geometry.representation == 'PROXY' else len(geometry.vertices)
        return EvidenceGradeResult(available=False, unavailable_reason=f'source has no {CONF_MEAN_FIELD}, {N_VIEWS_FIELD} evidence columns', grade=None, point_count=int(count))

    def _apply_placement_plan(self, plan: RenderPlan, payload: _PreparedRenderPlan, maximum: int | None, *, render: bool) -> ProjectLoadResult:
        if not isinstance(payload, _PreparedRenderPlan) or payload.project_id != plan.project_id:
            raise PyVistaViewportError('prepared geometry belongs to another project')
        if maximum != payload.maximum:
            raise PyVistaViewportError('prepared geometry has another display resolution')
        for fallback in payload.density_fallbacks:
            if self._placement_cca_by_scan.get(fallback.scan_id) == fallback.requested_cca:
                self._placement_cca_fallbacks[fallback.scan_id] = fallback
        geometries = dict(payload.geometries)
        desired: dict[str, tuple[ActorRequest, _PlacementGeometry]] = {}
        for request in plan.actors:
            if not request.available:
                continue
            geometry = geometries.get(request.actor_id)
            if geometry is None and (not request.visible):
                old = self._placement_actor_geometry.get(request.actor_id)
                if old is not None and old.key[4] == self._placement_key(request, payload.detailed_scan_ids, maximum)[4]:
                    geometry = old
            if geometry is None:
                if request.visible:
                    raise PyVistaViewportError(f'prepared geometry is missing: {request.actor_id}')
                continue
            if request.visible and geometry.key != self._placement_key(request, payload.detailed_scan_ids, maximum):
                raise PyVistaViewportError('scan source changed after preparation; retry HESAPLA')
            _matrix(request.transform, request.scan_id)
            desired[request.actor_id] = (request, geometry)
        height_limits = []
        for request, geometry in desired.values():
            if request.visible and request.color.strategy is ColorStrategy.WORLD_Z:
                lower, upper = geometry.bounds
                corners = np.asarray([[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
                transform = np.asarray(request.transform)
                z = corners @ transform[2, :3] + transform[2, 3]
                height_limits.extend((float(z.min()), float(z.max())))
        height_range = None
        if height_limits:
            lower, upper = (min(height_limits), max(height_limits))
            height_range = (lower - 0.5, upper + 0.5) if math.isclose(lower, upper) else (lower, upper)
        reused: dict[str, tuple[DrawnActor, ...]] = {}
        staged: dict[str, tuple[DrawnActor, ...]] = {}
        height_updates: dict[str, np.ndarray] = {}
        self._placement_generation += 1
        try:
            for actor_id, (request, geometry) in desired.items():
                old = self._placement_actor_geometry.get(actor_id)
                previous = self._actor_requests.get(actor_id)
                if old is not None and old.key == geometry.key and (previous is not None) and (previous.color == request.color) and (actor_id in self._actor_groups) and (self._placement_actor_grade.get(actor_id) == self._placement_display_key(geometry)):
                    reused[actor_id] = self._actor_groups[actor_id]
                    if request.color.strategy is ColorStrategy.WORLD_Z and previous.transform != request.transform:
                        transform = np.asarray(request.transform)
                        for drawn in reused[actor_id]:
                            vertices = geometry.vertices if drawn.select is None else geometry.vertices[drawn.select]
                            height_updates[drawn.name] = vertices @ transform[2, :3] + transform[2, 3]
                    continue
                staged[actor_id] = self._placement_actor_group(request, geometry, f'__placement_{self._placement_generation}::{actor_id}', actor_id, height_range)
        except Exception:
            for group in staged.values():
                for drawn in group:
                    self.plotter.remove_actor(drawn.actor, reset_camera=False, render=False)
            raise
        for actor_id, group in self._actor_groups.items():
            if actor_id not in reused:
                for drawn in group:
                    self._selection_actor_state.pop(drawn.name, None)
                    self.plotter.remove_actor(drawn.actor, reset_camera=False, render=False)
            self._actor_bounds.pop(actor_id, None)
        for group in staged.values():
            for drawn in group:
                self.plotter.remove_actor(drawn.actor, reset_camera=False, render=False)
                self.plotter.add_actor(drawn.actor, name=drawn.name, reset_camera=False, render=False)
        self._actor_groups = {**reused, **staged}
        self._layer_actors = {actor_id: group[0].actor for actor_id, group in self._actor_groups.items()}
        by_name = {drawn.name: drawn.actor for group in self._actor_groups.values() for drawn in group}
        for name, heights in height_updates.items():
            by_name[name].mapper.dataset.point_data['height_m'] = heights
        self._actor_requests = {key: value[0] for key, value in desired.items()}
        self._placement_actor_geometry = {key: value[1] for key, value in desired.items()}
        self._placement_actor_grade = {key: self._placement_display_key(value[1]) for key, value in desired.items()}
        self._actor_base_transforms = {key: np.eye(4) for key in desired}
        self._colour_gap_counts = {actor_id: (geometry.colour_gap_count, geometry.colour_gap_total) for actor_id, (request, geometry) in desired.items() if request.visible and geometry.colour_gap_count is not None}
        self._depth_fill_counts = {actor_id: (geometry.depth_fill_count, geometry.depth_fill_total) for actor_id, (request, geometry) in desired.items() if request.visible and geometry.depth_fill_count is not None}
        self._sky_fill_counts = {}
        for actor_id, (request, geometry) in desired.items():
            if not request.visible:
                continue
            mask = geometry.sky_hidden if geometry.sky_hidden is not None else geometry.cell_sky_hidden
            if mask is not None:
                self._sky_fill_counts[actor_id] = (int(np.count_nonzero(mask)), int(len(mask)))
        self._floating_counts = {}
        for actor_id, (request, geometry) in desired.items():
            if not request.visible:
                continue
            mask = self._placement_floating_hidden(geometry)
            if mask is not None:
                self._floating_counts[actor_id] = (int(np.count_nonzero(mask)), int(len(mask)))
        self._sparse_counts = {}
        self._sparse_defaults = {}
        for actor_id, (request, geometry) in desired.items():
            if not request.visible:
                continue
            mask = self._placement_sparse_hidden(geometry)
            if mask is not None:
                self._sparse_counts[actor_id] = (int(np.count_nonzero(mask)), len(mask))
            if geometry.sparse_default_m is not None:
                self._sparse_defaults[actor_id] = float(geometry.sparse_default_m)
        self._evidence_grade_results = {actor_id: self._placement_grade_result(geometry) for actor_id, (request, geometry) in desired.items() if request.visible}
        self._range_summaries = {actor_id: geometry.range_summary for actor_id, (request, geometry) in desired.items() if request.visible}
        self._scan_actors = {}
        self._scan_bounds = {}
        loaded = []
        for actor_id, (request, geometry) in desired.items():
            display = self._scan_display_transforms.get(request.scan_id, np.eye(4))
            bounds = None
            for drawn in self._actor_groups[actor_id]:
                bounds = self._merge_bounds(bounds, self._set_actor_transform_delta(drawn.actor, base_transform=np.eye(4), target_transform=display @ np.asarray(request.transform), label=request.scan_id))
                drawn.actor.SetVisibility(request.visible)
                if request.color.strategy is ColorStrategy.WORLD_Z and height_range is not None:
                    drawn.actor.mapper.scalar_range = height_range
            if not request.visible:
                continue
            self._scan_actors.setdefault(request.scan_id, self._layer_actors[actor_id])
            self._actor_bounds[actor_id] = bounds
            self._scan_bounds[request.scan_id] = self._merge_bounds(self._scan_bounds.get(request.scan_id), bounds)
            loaded.append(LoadedScan(request.scan_id, actor_id, geometry.source, geometry.source_count, len(geometry.vertices) if geometry.representation == 'POINTS' else 0, request.band, request.display_mode, geometry.representation, len(geometry.faces) if geometry.faces is not None else 0, geometry.cell_size_m))
        self._detailed_scan_ids = payload.detailed_scan_ids
        for name in tuple(self._surface_actors):
            scan_id = name.split('::')[1]
            if not self.is_scan_detailed(scan_id):
                self.remove_surface_preview(scan_id, render=False)
        if self._rolove_surface_scan_id and (not self.is_scan_detailed(self._rolove_surface_scan_id)):
            self.clear_rolove_surfaces(render=False)
        retained = {geometry.key for _, geometry in desired.values()}
        quality_keys = {self._placement_key(request, frozenset(), maximum) for request in plan.actors if request.available}
        self._placement_geometry_cache = {key: value for key, value in self._placement_geometry_cache.items() if key in quality_keys or key in retained}
        self._placement_geometry_cache.update({geometry.key: geometry for _, geometry in desired.values()})
        if self._rolove_pick_scan_id and (not self.is_scan_detailed(self._rolove_pick_scan_id)):
            self.stop_rolove_pick(cancel=True)
        self.point_size = plan.point_size
        if self._selected_scan_id not in self._scan_bounds:
            self._selected_scan_id = None
        self._apply_scan_highlight()
        self._refresh_transform_gizmo(render=False)
        self._render_plan = plan
        self._color_mode = self._mode_token(plan.display_mode)
        result = ProjectLoadResult(tuple(loaded), self._plan_actor_name, self._plan_segment_count, payload.density_fallbacks)
        self._last_load_result = result
        if render and (not self.off_screen):
            self.plotter.render()
        return result

    def apply_render_plan(self, plan: RenderPlan, *, max_points_per_actor: int | None=None, render: bool=True, prepared: _PreparedRenderPlan | None=None) -> ProjectLoadResult:
        if not isinstance(plan, RenderPlan):
            raise PyVistaViewportError('render plan must be a RenderPlan')
        if self._project is None:
            raise PyVistaViewportError('load a project before applying a render plan')
        if plan.project_id != self._project.project_id:
            raise PyVistaViewportError('render plan belongs to another project')
        try:
            plan.require_available()
        except DisplayModeUnavailableError as exc:
            raise PyVistaViewportError(str(exc)) from exc
        maximum = _validated_maximum(max_points_per_actor if max_points_per_actor is not None else self._maximum_points_per_actor)
        if self.placement_enabled:
            payload = prepared or self._prepare_placement_plan(plan, self._detailed_scan_ids, maximum)
            return self._apply_placement_plan(plan, payload, maximum, render=render)
        prepared_points: list[tuple[ActorRequest, Any, Path, int, EvidenceGradeResult, dict[str, float] | None]] = []
        for request in plan.visible_actors:
            polydata, source, source_count, grade_result, range_summary = self._request_polydata(request, maximum)
            prepared_points.append((request, polydata, source, source_count, grade_result, range_summary))
        heights = [np.asarray(polydata.point_data['height_m'], dtype=np.float64) for request, polydata, _source, _count, _grade, _range in prepared_points if request.color.strategy is ColorStrategy.WORLD_Z]
        height_range: tuple[float, float] | None = None
        if heights:
            lower = min((float(values.min()) for values in heights))
            upper = max((float(values.max()) for values in heights))
            if math.isclose(lower, upper, abs_tol=1e-12):
                lower -= 0.5
                upper += 0.5
            height_range = (lower, upper)
        selected_before = self._selected_scan_id
        self._clear_scan_actors()
        self.point_size = plan.point_size
        loaded: list[LoadedScan] = []
        try:
            for request, polydata, source, source_count, grade_result, range_summary in prepared_points:
                loaded.append(self._add_render_actor(request, polydata, source, source_count, height_range, grade_result, range_summary))
        except Exception:
            self._clear_scan_actors()
            raise
        self._selected_scan_id = selected_before if selected_before in self._scan_bounds else None
        self._apply_scan_highlight()
        self._refresh_transform_gizmo(render=False)
        self._render_plan = plan
        self._color_mode = self._mode_token(plan.display_mode)
        result = ProjectLoadResult(scans=tuple(loaded), plan_actor_name=self._plan_actor_name, plan_segment_count=self._plan_segment_count)
        self._last_load_result = result
        if render and (not self.off_screen):
            self.plotter.render()
        return result

    def load_project(self, project: ProjectModel, *, show_plan: bool=True, max_points_per_scan: int | None=None, render_plan: RenderPlan | None=None, prepared: _PreparedRenderPlan | None=None) -> ProjectLoadResult:
        if not isinstance(project, ProjectModel):
            raise PyVistaViewportError('project must be a ProjectModel')
        maximum = _validated_maximum(max_points_per_scan)
        if self.placement_enabled:
            selected_plan = render_plan or resolve_render_plan(project, DisplayMode.PHOTO_RGB, point_size=min(self.point_size, 10.0), require_available=True)
            payload = prepared or self._prepare_placement_plan(selected_plan, self._detailed_scan_ids, maximum)
            staged_plan = None
            if show_plan and project.dxf_plan.loaded and project.dxf_plan.geometry_path:
                staged_plan = self._build_plan_actor(project, '__placement_plan_prepare')
            previous_project = self._project
            previous_maximum = self._maximum_points_per_actor
            previous_display_transforms = self._scan_display_transforms
            previous_selection = self._selected_scan_id
            self._project = project
            self._maximum_points_per_actor = maximum
            self._scan_display_transforms = {}
            self._selected_scan_id = None
            try:
                result = self.apply_render_plan(selected_plan, prepared=payload, max_points_per_actor=maximum, render=False)
            except Exception:
                if staged_plan is not None:
                    self.plotter.remove_actor(staged_plan[0], reset_camera=False, render=False)
                self._project = previous_project
                self._maximum_points_per_actor = previous_maximum
                self._scan_display_transforms = previous_display_transforms
                self._selected_scan_id = previous_selection
                raise
            self.stop_rolove_pick()
            self.clear_rolove_surfaces(render=False)
            self._manual_drag = None
            for actor_name in tuple(self._surface_actors):
                self.plotter.remove_actor(self._surface_actors[actor_name], reset_camera=False, render=False)
                self._actor_bounds.pop(actor_name, None)
            self._surface_actors.clear()
            self._surface_base_transforms.clear()
            if self._plan_actor is not None:
                self.plotter.remove_actor(self._plan_actor, reset_camera=False, render=False)
                self._actor_bounds.pop(self._plan_actor_name, None)
                self._plan_actor = self._plan_actor_name = self._plan_base_transform = None
                self._plan_segment_count = 0
            if staged_plan is not None:
                actor, bounds, count = staged_plan
                self.plotter.remove_actor(actor, reset_camera=False, render=False)
                self.plotter.add_actor(actor, name='plan::dxf', pickable=False, reset_camera=False, render=False)
                self._plan_actor = actor
                self._plan_actor_name = 'plan::dxf'
                self._plan_base_transform = np.asarray(project.dxf_plan.transform)
                self._plan_segment_count = count
                self._actor_bounds['plan::dxf'] = bounds
            result = ProjectLoadResult(result.scans, self._plan_actor_name, self._plan_segment_count, result.density_fallbacks)
            self._last_load_result = result
            if self._scan_bounds:
                self.frame_scans(render=False)
            elif self._actor_bounds:
                self.frame_all(render=False)
            if not self.off_screen:
                self.plotter.render()
            return result
        self._clear_project_actors()
        self._project = project
        self._maximum_points_per_actor = maximum
        try:
            plan_actor_name: str | None = None
            if show_plan and project.dxf_plan.loaded and project.dxf_plan.geometry_path:
                plan_actor_name, _plan_segments = self._add_plan(project)
            selected_plan = render_plan
            if selected_plan is None:
                selected_plan = resolve_render_plan(project, DisplayMode.PHOTO_RGB, point_size=min(self.point_size, 10.0), require_available=True)
            result = self.apply_render_plan(selected_plan, max_points_per_actor=maximum, render=False)
        except Exception:
            self._clear_project_actors()
            raise
        if plan_actor_name is None and result.plan_actor_name is not None:
            raise PyVistaViewportError('unexpected plan actor state')
        self._last_load_result = result
        if self._scan_bounds:
            self.frame_scans(render=False)
        elif self._actor_bounds:
            self.frame_all(render=False)
        else:
            self.apply_camera(render=False)
        if not self.off_screen:
            self.plotter.render()
        return result

    def _frame_bound_set(self, bounds: tuple[tuple[float, float, float, float, float, float], ...], *, render: bool) -> bool:
        if not bounds:
            self.apply_camera(render=render)
            return False
        minimum = (min((item[0] for item in bounds)), min((item[2] for item in bounds)), min((item[4] for item in bounds)))
        maximum = (max((item[1] for item in bounds)), max((item[3] for item in bounds)), max((item[5] for item in bounds)))
        camera = self.plotter.renderer.GetActiveCamera()
        width, height = self.plotter.render_window.GetSize()
        if width <= 0 or height <= 0:
            width, height = (self.widget.width(), self.widget.height())
        aspect = max(float(width), 1.0) / max(float(height), 1.0)
        angle = float(camera.GetViewAngle())
        if camera.GetUseHorizontalViewAngle():
            angle = math.degrees(2 * math.atan(math.tan(math.radians(angle) / 2) / aspect))
        self.controller.frame_bounds(minimum, maximum, vertical_fov_degrees=angle, aspect_ratio=aspect)
        self.end_navigation_drag()
        self.apply_camera(render=render)
        return True

    def fit_scene(self, *, render: bool=True) -> bool:
        return self._frame_navigation_geometry(self._navigation_geometry(), render=render)

    def _navigation_geometry(self, scan_ids: Sequence[str] | None=None, *, include_plan: bool=True) -> list[tuple[np.ndarray, np.ndarray]]:
        wanted = None if scan_ids is None else frozenset(scan_ids)
        actors = []
        for actor_id, actor in self._layer_actors.items():
            request = self._actor_requests.get(actor_id)
            if request is not None and actor.GetVisibility() and (wanted is None or request.scan_id in wanted):
                geometry = self._placement_actor_geometry.get(actor_id)
                points = geometry.vertices if geometry is not None else actor.mapper.dataset.points
                actors.append((points, actor))
        if include_plan and self._plan_actor is not None and self._plan_actor.GetVisibility():
            actors.append((self._plan_actor.mapper.dataset.points, self._plan_actor))
        result = []
        for points, actor in actors:
            matrix = actor.GetMatrix()
            transform = np.asarray([[matrix.GetElement(row, column) for column in range(4)] for row in range(4)])
            values = np.asarray(points)
            if values.ndim == 2 and values.shape[1] == 3 and len(values):
                result.append((values, transform))
        return result

    def _frame_navigation_geometry(self, geometry: Sequence[tuple[np.ndarray, np.ndarray]], *, render: bool) -> bool:
        if not geometry:
            return False
        pose = self.controller.pose()
        forward = np.asarray(pose.target) - pose.eye
        forward /= np.linalg.norm(forward)
        basis = np.asarray([pose.right, pose.screen_up, forward])

        def projected_chunks():
            for points, transform in geometry:
                axes = basis @ transform[:3, :3]
                offset = basis @ transform[:3, 3]
                for start in range(0, len(points), 125000):
                    projected = points[start:start + 125000] @ axes.T + offset
                    finite = np.isfinite(projected).all(axis=1)
                    if not finite.all():
                        projected = projected[finite]
                    if len(projected):
                        yield projected
        lower, upper = (np.full(3, np.inf), np.full(3, -np.inf))
        for projected in projected_chunks():
            lower = np.minimum(lower, projected.min(axis=0))
            upper = np.maximum(upper, projected.max(axis=0))
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            return False
        center = (lower + upper) * 0.5
        camera = self.plotter.renderer.GetActiveCamera()
        width, height = self.plotter.render_window.GetSize()
        if width <= 0 or height <= 0:
            width, height = (self.widget.width(), self.widget.height())
        aspect = max(float(width), 1.0) / max(float(height), 1.0)
        tangent = math.tan(math.radians(float(camera.GetViewAngle())) * 0.5)
        if camera.GetUseHorizontalViewAngle():
            tangent /= aspect
        distance = 0.0
        padding = 1.1
        for projected in projected_chunks():
            projected -= center
            extent = np.maximum(np.abs(projected[:, 0]) / aspect, np.abs(projected[:, 1]))
            if self.controller.state.projection == 'orthographic':
                required = float(extent.max()) * 2 * padding
            else:
                required = float((extent * (padding / tangent) - projected[:, 2]).max())
            distance = max(distance, required)
        distance = max(distance, (upper[2] - lower[2]) * 0.5 + 0.01, 0.05)
        self.controller.restore(replace(self.controller.state, pivot=tuple(center @ basis), distance=distance))
        self.end_navigation_drag()
        self.apply_camera(render=render)
        return True

    def fit_selection(self, scan_ids: Sequence[str] | None=None, *, render: bool=True) -> bool:
        selected = scan_ids
        if selected is None and self._selection_provider is not None:
            selected = tuple(self._selection_provider())
        if selected is None and self._selected_scan_id is not None:
            selected = (self._selected_scan_id,)
        geometry = [] if selected is None else self._navigation_geometry(selected, include_plan=False)
        if selected is not None and len(selected) and (not geometry):
            return False
        return self._frame_navigation_geometry(geometry or self._navigation_geometry(), render=render)

    def set_named_view(self, name: str, *, fit: bool=True, render: bool=True) -> CameraState:
        self.end_navigation_drag()
        self.controller.set_named_view(name)
        if fit:
            self.fit_scene(render=render)
        else:
            self.apply_camera(render=render)
        return self.controller.state

    def frame_scan(self, scan_id: str, *, render: bool=True) -> bool:
        bounds = self._scan_bounds.get(scan_id)
        if bounds is None:
            raise PyVistaViewportError(f'scan is not loaded: {scan_id}')
        return self._frame_navigation_geometry(self._navigation_geometry((scan_id,), include_plan=False), render=render)

    def enter_plan_placement(self, scan_id: str | None=None, *, render: bool=True) -> CameraState:
        active_scan_id = scan_id or self._selected_scan_id
        if active_scan_id is None:
            raise PyVistaViewportError('select a loaded scan for plan placement')
        scan_bounds = self._scan_bounds.get(active_scan_id)
        if scan_bounds is None:
            raise PyVistaViewportError(f'scan is not loaded: {active_scan_id}')
        if self._plan_actor_name is None:
            raise PyVistaViewportError('DXF plan is not loaded')
        plan_bounds = self._actor_bounds.get(self._plan_actor_name)
        if plan_bounds is None:
            raise PyVistaViewportError('DXF plan bounds are unavailable')
        saved = self.controller.state
        if active_scan_id != self._selected_scan_id:
            self.select_scan(active_scan_id)
        self.controller.enter_top()
        self.controller.set_projection('orthographic')
        self._frame_navigation_geometry(self._navigation_geometry((active_scan_id,), include_plan=True), render=False)
        if self._manual_drag is not None:
            self.cancel_manual_drag()
        self._move_constraint = 'XY'
        self._transform_mode = 'MOVE'
        self._refresh_transform_gizmo(render=False)
        self._emit_transform_mode()
        self.apply_camera(render=render)
        return saved

    def restore_camera_state(self, state: CameraState, *, render: bool=True) -> CameraState:
        if not isinstance(state, CameraState):
            raise PyVistaViewportError('camera state must be a CameraState')
        restored = self.controller.restore(state)
        self.apply_camera(render=render)
        return restored

    def frame_scans(self, *, render: bool=True) -> bool:
        return self._frame_navigation_geometry(self._navigation_geometry(include_plan=False), render=render)

    def frame_all(self, *, render: bool=True) -> bool:
        return self.fit_scene(render=render)

    def close(self) -> None:
        crosshair = getattr(self, '_cad_crosshair', None)
        if crosshair is not None:
            crosshair.detach()
            self._cad_crosshair = None
        if self._plotter is not None:
            for name in tuple(self._selection_box_actors):
                self._plotter.remove_actor(name, reset_camera=False, render=False)
            self._selection_box_actors.clear()
            self._plotter.close()
        self._plotter = None
        self._widget = None
        self._view_cube = None
        self._vtk = None
__all__ = ['COLOUR_GAP_DISPLAY_HEX', 'COLOUR_GAP_DISPLAY_RGB', 'COLOUR_GAP_FIELD', 'COLOUR_GAP_SCALAR', 'DEPTH_FILL_SCALAR', 'DepthFillSummary', 'ColourGapSummary', 'DetailCapacityPlan', 'DetailCapacitySkip', 'LoadedScan', 'ManualTransformEvent', 'ProjectLoadResult', 'PyVistaViewportAdapter', 'PyVistaViewportError', 'SELECTION_COLOR', 'points_to_working_axis']
