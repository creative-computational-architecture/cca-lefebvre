
from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from geometry.rolove import placed_matrix
from viewer.project_model import FIXED_SCALE, WORKING_AXIS, WORKING_UNITS, Matrix4, ProjectModel, ScanEntry, ScanLayer
LAYER_BANDS = ('core', 'body', 'shell')
EVIDENCE_BANDS = (*LAYER_BANDS, 'raw', 'corrected', 'fused')
POINT_SIZE_MIN = 1.0
POINT_SIZE_MAX = 10.0
DEFAULT_POINT_SIZE = 2.0
RGB = tuple[float, float, float]
RUN_TINTS: tuple[RGB, ...] = ((1.0, 0.25, 0.03), (0.0, 0.67, 0.86), (0.38, 0.92, 0.2), (0.72, 0.34, 0.95), (1.0, 0.76, 0.04), (0.96, 0.12, 0.55), (0.1, 0.42, 1.0), (0.68, 0.9, 0.04), (0.0, 0.82, 0.6), (0.94, 0.48, 0.1), (0.45, 0.22, 1.0), (0.96, 0.24, 0.22), (0.12, 0.76, 1.0))

class LayerControllerError(ValueError):
    pass

class DisplayModeUnavailableError(LayerControllerError):

    def __init__(self, mode: DisplayMode, missing: Sequence[ActorRequest]) -> None:
        details = ', '.join((f'{item.scan_id}/{item.band} ({item.unavailable_reason})' for item in missing))
        super().__init__(f'{mode.value} unavailable for requested layers: {details}')
        self.mode = mode
        self.missing = tuple(missing)

class DisplayMode(str, Enum):
    PHOTO_RGB = 'PHOTO RGB'
    CONFIDENCE = 'CONFIDENCE'
    RUN_TINT = 'RUN TINT'
    HEIGHT = 'HEIGHT'

class EvidenceProfile(str, Enum):
    REVIEW_CORE = 'REVIEW CORE'
    NATIVE_RGB = 'NATIVE RGB'
    RAW_RGB = 'RAW RGB'
    CORRECTED_RGB = 'CORRECTED RGB'
    RAW_CONF = 'RAW CONF'
    APP_FUSED = 'PROCESSED FULL'

class ColorStrategy(str, Enum):
    SOURCE_RGB = 'SOURCE_RGB'
    CONFIDENCE_RGB = 'CONFIDENCE_RGB'
    UNIFORM_RGB = 'UNIFORM_RGB'
    WORLD_Z = 'WORLD_Z'
    EVIDENCE_GRADE = 'EVIDENCE_GRADE'

@dataclass(frozen=True, slots=True)
class ColorRule:
    strategy: ColorStrategy
    semantic: str
    uniform_rgb: RGB | None = None
    scalar_name: str | None = None
    scalar_component: int | None = None
    colormap: str | None = None
    coordinate_space: str | None = None

    def __post_init__(self) -> None:
        if self.strategy is ColorStrategy.UNIFORM_RGB:
            if self.uniform_rgb is None or len(self.uniform_rgb) != 3:
                raise LayerControllerError('uniform color rule needs one RGB triplet')
            if any((not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.uniform_rgb)):
                raise LayerControllerError('uniform RGB values must be within zero and one')
        elif self.uniform_rgb is not None:
            raise LayerControllerError('only a uniform color rule can carry uniform RGB')
        if self.strategy is ColorStrategy.WORLD_Z:
            if self.scalar_name != 'height_m' or self.scalar_component != 2 or self.coordinate_space != 'WORLD' or (not self.colormap):
                raise LayerControllerError('height color rule must use world Z in metres')
        elif any((value is not None for value in (self.scalar_name, self.scalar_component, self.colormap, self.coordinate_space))):
            raise LayerControllerError('non-height color rules cannot carry scalar fields')

@dataclass(frozen=True, slots=True)
class ActorRequest:
    actor_id: str
    scan_id: str
    group_id: str
    rail_order: int
    band: str
    display_mode: DisplayMode
    source_layer_key: str
    source_path: Path | None
    source_point_count: int | None
    source_axis: str
    transform: Matrix4
    scan_visible: bool
    band_visible: bool
    available: bool
    unavailable_reason: str | None
    point_size: float
    color: ColorRule

    def __post_init__(self) -> None:
        if self.band not in EVIDENCE_BANDS:
            raise LayerControllerError(f'unsupported layer band: {self.band}')
        if self.available and self.source_path is None:
            raise LayerControllerError('an available actor needs a source path')
        if self.available and self.unavailable_reason is not None:
            raise LayerControllerError('an available actor cannot have an unavailable reason')
        if not self.available and (not self.unavailable_reason):
            raise LayerControllerError('an unavailable actor needs a reason')

    @property
    def requested(self) -> bool:
        return self.scan_visible and self.band_visible

    @property
    def visible(self) -> bool:
        return self.requested and self.available

@dataclass(frozen=True, slots=True)
class RenderPlan:
    project_id: str
    project_name: str
    display_mode: DisplayMode
    evidence_profile: EvidenceProfile
    point_size: float
    actors: tuple[ActorRequest, ...]

    @property
    def visible_actors(self) -> tuple[ActorRequest, ...]:
        return tuple((item for item in self.actors if item.visible))

    @property
    def available_actors(self) -> tuple[ActorRequest, ...]:
        return tuple((item for item in self.actors if item.available))

    @property
    def unavailable_actors(self) -> tuple[ActorRequest, ...]:
        return tuple((item for item in self.actors if not item.available))

    @property
    def requested_unavailable(self) -> tuple[ActorRequest, ...]:
        return tuple((item for item in self.actors if item.requested and (not item.available)))

    @property
    def complete(self) -> bool:
        return not self.requested_unavailable

    def require_available(self) -> RenderPlan:
        missing = self.requested_unavailable
        partial_evidence_available = self.evidence_profile in {EvidenceProfile.REVIEW_CORE, EvidenceProfile.NATIVE_RGB, EvidenceProfile.RAW_RGB, EvidenceProfile.CORRECTED_RGB, EvidenceProfile.RAW_CONF, EvidenceProfile.APP_FUSED} and bool(self.visible_actors)
        if missing and (not partial_evidence_available):
            raise DisplayModeUnavailableError(self.display_mode, missing)
        return self
_MODE_ALIASES = {'photo rgb': DisplayMode.PHOTO_RGB, 'photo': DisplayMode.PHOTO_RGB, 'rgb': DisplayMode.PHOTO_RGB, 'confidence': DisplayMode.CONFIDENCE, 'confidence rgb': DisplayMode.CONFIDENCE, 'confcolor': DisplayMode.CONFIDENCE, 'run tint': DisplayMode.RUN_TINT, 'scan tint': DisplayMode.RUN_TINT, 'scan group': DisplayMode.RUN_TINT, 'tint': DisplayMode.RUN_TINT, 'height': DisplayMode.HEIGHT, 'elevation': DisplayMode.HEIGHT, 'z': DisplayMode.HEIGHT}
_EVIDENCE_PROFILE_ALIASES = {'review core': EvidenceProfile.REVIEW_CORE, 'review': EvidenceProfile.REVIEW_CORE, 'core': EvidenceProfile.REVIEW_CORE, 'native rgb': EvidenceProfile.NATIVE_RGB, 'native': EvidenceProfile.NATIVE_RGB, 'source rgb': EvidenceProfile.NATIVE_RGB, 'source': EvidenceProfile.NATIVE_RGB, 'kaynak': EvidenceProfile.NATIVE_RGB, 'raw rgb': EvidenceProfile.RAW_RGB, 'rgb raw': EvidenceProfile.RAW_RGB, 'raw': EvidenceProfile.RAW_RGB, 'ham': EvidenceProfile.RAW_RGB, 'ham tum noktalar': EvidenceProfile.RAW_RGB, 'corrected': EvidenceProfile.CORRECTED_RGB, 'corrected rgb': EvidenceProfile.CORRECTED_RGB, 'rgb corrected': EvidenceProfile.CORRECTED_RGB, 'duzeltilmis': EvidenceProfile.CORRECTED_RGB, 'duzeltilmis tum noktalar': EvidenceProfile.CORRECTED_RGB, 'raw conf': EvidenceProfile.RAW_CONF, 'raw confidence': EvidenceProfile.RAW_CONF, 'confidence raw': EvidenceProfile.RAW_CONF, 'app fused': EvidenceProfile.APP_FUSED, 'fused': EvidenceProfile.APP_FUSED, 'processed full': EvidenceProfile.APP_FUSED, 'full processed': EvidenceProfile.APP_FUSED}

def resolve_display_mode(value: DisplayMode | str) -> DisplayMode:
    if isinstance(value, DisplayMode):
        return value
    if not isinstance(value, str):
        raise LayerControllerError('display mode must be text')
    normalized = ' '.join(value.strip().lower().replace('_', ' ').split())
    try:
        return _MODE_ALIASES[normalized]
    except KeyError as exc:
        raise LayerControllerError(f'unsupported display mode: {value}') from exc

def resolve_evidence_profile(value: EvidenceProfile | str) -> EvidenceProfile:
    if isinstance(value, EvidenceProfile):
        return value
    if not isinstance(value, str):
        raise LayerControllerError('evidence profile must be text')
    normalized = ' '.join(value.strip().lower().replace('_', ' ').split())
    try:
        return _EVIDENCE_PROFILE_ALIASES[normalized]
    except KeyError as exc:
        raise LayerControllerError(f'unsupported evidence profile: {value}') from exc

def _point_size(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LayerControllerError('point size must be numeric')
    output = float(value)
    if not math.isfinite(output):
        raise LayerControllerError('point size must be finite')
    if not POINT_SIZE_MIN <= output <= POINT_SIZE_MAX:
        raise LayerControllerError('point size must be between 1 and 10')
    return output

def _visibility(values: Mapping[str, bool] | None, keys: Sequence[str], label: str) -> dict[str, bool] | None:
    if values is None:
        return None
    unknown = sorted(set(values) - set(keys))
    if unknown:
        raise LayerControllerError(f"unknown {label}: {', '.join(unknown)}")
    output: dict[str, bool] = {}
    for key, value in values.items():
        if type(value) is not bool:
            raise LayerControllerError(f'{label} visibility must be boolean: {key}')
        output[key] = value
    return output

def _guard_matrix(matrix: object, label: str) -> None:
    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes, bytearray)):
        raise LayerControllerError(f'{label} transform must be a 4x4 matrix')
    if len(matrix) != 4:
        raise LayerControllerError(f'{label} transform must be a 4x4 matrix')
    rows: list[list[float]] = []
    for row in matrix:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise LayerControllerError(f'{label} transform must be a 4x4 matrix')
        if len(row) != 4:
            raise LayerControllerError(f'{label} transform must be a 4x4 matrix')
        try:
            values = [float(item) for item in row]
        except (TypeError, ValueError) as exc:
            raise LayerControllerError(f'{label} transform must be numeric') from exc
        if not all((math.isfinite(item) for item in values)):
            raise LayerControllerError(f'{label} transform must be finite')
        rows.append(values)
    if any((not math.isclose(rows[3][index], expected, abs_tol=1e-08) for index, expected in enumerate((0.0, 0.0, 0.0, 1.0)))):
        raise LayerControllerError(f'{label} transform must be affine')
    rotation = [row[:3] for row in rows[:3]]
    for first in range(3):
        for second in range(3):
            dot = sum((rotation[row][first] * rotation[row][second] for row in range(3)))
            expected = 1.0 if first == second else 0.0
            if not math.isclose(dot, expected, abs_tol=1e-05):
                raise LayerControllerError(f'{label} transform scale must remain one')
    determinant = rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1]) - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0]) + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    if not math.isclose(determinant, 1.0, abs_tol=1e-05):
        raise LayerControllerError(f'{label} transform must be proper rigid')

def _guard_project(project: ProjectModel) -> None:
    if not isinstance(project, ProjectModel):
        raise LayerControllerError('project must be a ProjectModel')
    if project.units != WORKING_UNITS or project.axis != WORKING_AXIS:
        raise LayerControllerError('project must use metre units and CAD Z-up')
    if not math.isclose(float(project.fixed_scale), FIXED_SCALE, abs_tol=1e-09):
        raise LayerControllerError('project scale must remain locked at one')
    if project.raw_points_preserved is not True:
        raise LayerControllerError('project must preserve raw source points')
    for scan in project.scan_rail:
        if scan.units != WORKING_UNITS or scan.axis != WORKING_AXIS:
            raise LayerControllerError(f'{scan.scan_id} must use metre units and CAD Z-up')
        if not math.isclose(float(scan.fixed_scale), FIXED_SCALE, abs_tol=1e-09):
            raise LayerControllerError(f'{scan.scan_id} scale must remain locked at one')
        if scan.raw_points_preserved is not True:
            raise LayerControllerError(f'{scan.scan_id} must preserve raw source points')
        _guard_matrix(scan.transform, scan.scan_id)

def _project_base(project: ProjectModel) -> Path:
    if project.source_manifest:
        return Path(project.source_manifest).expanduser().resolve(strict=False).parent
    return Path.cwd().resolve(strict=False)

def _source_path(base: Path, layer: ScanLayer | None) -> Path | None:
    if layer is None or not layer.path:
        return None
    source = Path(layer.path).expanduser()
    if not source.is_absolute():
        source = base / source
    return source.resolve(strict=False)

def _source_status(layer_key: str, layer: ScanLayer | None, source: Path | None, *, require_cad_z_up: bool=False, allowed_axes: Sequence[str] | None=None) -> tuple[bool, str | None]:
    if layer is None:
        return (False, f'missing layer {layer_key}')
    if require_cad_z_up and layer.metadata.get('source_axis') != WORKING_AXIS:
        source_axis = layer.metadata.get('source_axis')
        return (False, f'{layer_key} source_axis must be {WORKING_AXIS}, got {source_axis!r}')
    if allowed_axes is not None and layer.metadata.get('source_axis') not in allowed_axes:
        source_axis = layer.metadata.get('source_axis')
        return (False, f'{layer_key} source_axis must be one of {tuple(allowed_axes)}, got {source_axis!r}')
    if layer.point_count == 0:
        return (False, f'zero points in {layer_key}')
    if source is None:
        return (False, f'missing source path for {layer_key}')
    try:
        exists = source.is_file()
    except OSError:
        exists = False
    if not exists:
        return (False, f'source file does not exist for {layer_key}')
    return (True, None)

def _app_fused_source_status(scan: ScanEntry, base: Path, layer_key: str) -> tuple[bool, str | None]:
    layer = scan.layer(layer_key)
    source = _source_path(base, layer)
    raw_source = _source_path(base, scan.layer('rgb/raw'))
    if layer is not None and (layer.role in {'CORRECTED_RAW_RGB', 'CORRECTED_RAW_RGB_PRIMARY'} or (source is not None and source == raw_source)):
        return (False, f'raw RGB alias is not app-fused geometry for {layer_key}')
    return _source_status(layer_key, layer, source, allowed_axes=(WORKING_AXIS, 'ARKit_Y_up'))

def _source_prefix(mode: DisplayMode) -> str:
    if mode is DisplayMode.CONFIDENCE:
        return 'confcolor'
    return 'rgb'

def _color_rule(mode: DisplayMode, rail_order: int) -> ColorRule:
    if mode is DisplayMode.PHOTO_RGB:
        return ColorRule(ColorStrategy.SOURCE_RGB, 'photo_rgb')
    if mode is DisplayMode.CONFIDENCE:
        return ColorRule(ColorStrategy.CONFIDENCE_RGB, 'confidence')
    if mode is DisplayMode.RUN_TINT:
        return ColorRule(ColorStrategy.UNIFORM_RGB, 'run_identity', uniform_rgb=RUN_TINTS[rail_order % len(RUN_TINTS)])
    return ColorRule(ColorStrategy.WORLD_Z, 'height_m', scalar_name='height_m', scalar_component=2, colormap='viridis', coordinate_space='WORLD')

def _evidence_color_rule(profile: EvidenceProfile, mode: DisplayMode, rail_order: int) -> ColorRule:
    if mode in {DisplayMode.RUN_TINT, DisplayMode.HEIGHT}:
        return _color_rule(mode, rail_order)
    if profile is EvidenceProfile.RAW_CONF:
        return _color_rule(DisplayMode.CONFIDENCE, rail_order)
    return _color_rule(mode, rail_order)

def _evidence_layers(profile: EvidenceProfile, mode: DisplayMode, scan: ScanEntry, base: Path) -> tuple[tuple[str, str], ...]:
    if profile is EvidenceProfile.REVIEW_CORE:
        prefix = _source_prefix(mode)
        return tuple(((band, f'{prefix}/{band}') for band in LAYER_BANDS))
    if profile is EvidenceProfile.NATIVE_RGB:
        candidates = ('rgb/raw', 'rgb/full_zup', 'rgb/full', 'rgb/core')

        def native_available(candidate: str) -> bool:
            layer = scan.layer(candidate)
            if layer is not None and layer.role.startswith('CORRECTED_RAW_'):
                return False
            return _source_status(candidate, layer, _source_path(base, layer), allowed_axes=(WORKING_AXIS, 'ARKit_Y_up'))[0]
        key = next((candidate for candidate in candidates if native_available(candidate)), next((candidate for candidate in candidates if scan.layer(candidate) is not None and (not scan.layer(candidate).role.startswith('CORRECTED_RAW_'))), 'rgb/raw'))
        return (('core', key),)
    if profile in {EvidenceProfile.RAW_RGB, EvidenceProfile.RAW_CONF}:
        prefix = 'confcolor' if profile is EvidenceProfile.RAW_CONF or mode is DisplayMode.CONFIDENCE else 'rgb'
        return (('raw', f'{prefix}/raw'),)
    if profile is EvidenceProfile.CORRECTED_RGB:
        prefix = _source_prefix(mode)
        return (('corrected', f'{prefix}/corrected'),)
    candidates = ('confcolor/full_zup', 'confcolor/full') if mode is DisplayMode.CONFIDENCE else _APP_FUSED_LAYER_KEYS
    key = next((candidate for candidate in candidates if _app_fused_source_status(scan, base, candidate)[0]), next((candidate for candidate in candidates if scan.layer(candidate) is not None), candidates[-1]))
    return (('fused', key),)

def _default_band_visible(scan: ScanEntry, band: str) -> bool:
    layers = [layer for layer in (scan.layer(f'rgb/{band}'), scan.layer(f'confcolor/{band}')) if layer is not None]
    if any((layer.visible for layer in layers)):
        return True
    return any((layer.point_count == 0 for layer in layers))

def resolve_render_plan(project: ProjectModel, display_mode: DisplayMode | str=DisplayMode.PHOTO_RGB, *, evidence_profile: EvidenceProfile | str=EvidenceProfile.REVIEW_CORE, point_size: float=DEFAULT_POINT_SIZE, band_visibility: Mapping[str, bool] | None=None, scan_visibility: Mapping[str, bool] | None=None, require_available: bool=False) -> RenderPlan:
    _guard_project(project)
    mode = resolve_display_mode(display_mode)
    profile = resolve_evidence_profile(evidence_profile)
    size = _point_size(point_size)
    band_state = _visibility(band_visibility, LAYER_BANDS, 'layer band')
    scan_ids = [scan.scan_id for scan in project.scan_rail]
    scan_state = _visibility(scan_visibility, scan_ids, 'scan')
    base = _project_base(project)
    actors: list[ActorRequest] = []
    for scan in project.scan_rail:
        scan_actor_start = len(actors)
        is_scan_visible = scan_state.get(scan.scan_id, scan.visible) if scan_state is not None else scan.visible
        color = _evidence_color_rule(profile, mode, scan.rail_order)
        for band, layer_key in _evidence_layers(profile, mode, scan, base):
            layer = scan.layer(layer_key)
            source = _source_path(base, layer)
            if profile is EvidenceProfile.APP_FUSED:
                available, reason = _app_fused_source_status(scan, base, layer_key)
            else:
                available, reason = _source_status(layer_key, layer, source, require_cad_z_up=profile in {EvidenceProfile.RAW_RGB, EvidenceProfile.RAW_CONF, EvidenceProfile.CORRECTED_RGB})
            if profile is EvidenceProfile.REVIEW_CORE:
                is_band_visible = band_state.get(band, _default_band_visible(scan, band)) if band_state is not None else _default_band_visible(scan, band)
            else:
                is_band_visible = True
            mode_token = mode.value.lower().replace(' ', '-')
            profile_token = profile.value.lower().replace(' ', '-')
            actor_id = f'scan::{scan.scan_id}::{mode_token}::{band}'
            if profile is not EvidenceProfile.REVIEW_CORE:
                actor_id = f'scan::{scan.scan_id}::{profile_token}::{mode_token}::{band}'
            actors.append(ActorRequest(actor_id=actor_id, scan_id=scan.scan_id, group_id=scan.group_id, rail_order=scan.rail_order, band=band, display_mode=mode, source_layer_key=layer_key, source_path=source, source_point_count=layer.point_count if layer is not None else None, source_axis=str(layer.metadata.get('source_axis', WORKING_AXIS) if layer is not None else WORKING_AXIS), transform=tuple((tuple((float(v) for v in row)) for row in placed_matrix(scan.transform, scan.metadata))), scan_visible=is_scan_visible, band_visible=is_band_visible, available=available, unavailable_reason=reason, point_size=size, color=color))
        if profile is EvidenceProfile.REVIEW_CORE and project.metadata.get('review_band_fallback') is True and is_scan_visible:
            scan_actor_indexes = range(scan_actor_start, len(actors))
            requested_any = any((actors[index].band_visible for index in scan_actor_indexes))
            visible_any = any((actors[index].visible for index in scan_actor_indexes))
            if requested_any and (not visible_any):
                fallback_index = next((index for index in scan_actor_indexes if actors[index].available), None)
                if fallback_index is not None:
                    for index in scan_actor_indexes:
                        actors[index] = replace(actors[index], band_visible=False)
                    actors[fallback_index] = replace(actors[fallback_index], band_visible=True)
    plan = RenderPlan(project_id=project.project_id, project_name=project.name, display_mode=mode, evidence_profile=profile, point_size=size, actors=tuple(actors))
    return plan.require_available() if require_available else plan
_APP_FUSED_LAYER_KEYS = ('rgb/full_zup', 'rgb/full')

def initial_evidence_profile(project: ProjectModel) -> EvidenceProfile:
    visible_ids = {scan.scan_id for scan in project.scan_rail if scan.visible} or {scan.scan_id for scan in project.scan_rail}

    def covers_visible(profile: EvidenceProfile) -> bool:
        plan = resolve_render_plan(project, evidence_profile=profile)
        ready_ids = {actor.scan_id for actor in plan.visible_actors}
        return bool(visible_ids) and ready_ids == visible_ids
    configured = project.metadata.get('initial_evidence_profile')
    if isinstance(configured, str):
        try:
            requested = EvidenceProfile(configured)
        except ValueError:
            requested = None
        if requested is EvidenceProfile.RAW_CONF:
            requested = EvidenceProfile.RAW_RGB
        if requested not in {None, EvidenceProfile.REVIEW_CORE} and covers_visible(requested):
            return requested
    for profile in (EvidenceProfile.CORRECTED_RGB, EvidenceProfile.RAW_RGB, EvidenceProfile.APP_FUSED, EvidenceProfile.NATIVE_RGB, EvidenceProfile.REVIEW_CORE):
        if covers_visible(profile):
            return profile
    return EvidenceProfile.REVIEW_CORE

class LayerController:

    def __init__(self, project: ProjectModel, *, display_mode: DisplayMode | str=DisplayMode.PHOTO_RGB, evidence_profile: EvidenceProfile | str=EvidenceProfile.REVIEW_CORE, point_size: float=DEFAULT_POINT_SIZE) -> None:
        _guard_project(project)
        self._project = project
        self._display_mode = resolve_display_mode(display_mode)
        self._evidence_profile = resolve_evidence_profile(evidence_profile)
        self._point_size = _point_size(point_size)
        self._scan_visibility = {scan.scan_id: scan.visible for scan in project.scan_rail}
        self._band_visibility = {band: any((layer.visible for scan in project.scan_rail for layer in (scan.layer(f'rgb/{band}'), scan.layer(f'confcolor/{band}')) if layer is not None)) for band in LAYER_BANDS}

    @property
    def project(self) -> ProjectModel:
        return self._project

    @property
    def display_mode(self) -> DisplayMode:
        return self._display_mode

    @property
    def point_size(self) -> float:
        return self._point_size

    @property
    def evidence_profile(self) -> EvidenceProfile:
        return self._evidence_profile

    @property
    def band_visibility(self) -> dict[str, bool]:
        return dict(self._band_visibility)

    @property
    def scan_visibility(self) -> dict[str, bool]:
        return dict(self._scan_visibility)

    def set_display_mode(self, value: DisplayMode | str) -> DisplayMode:
        self._display_mode = resolve_display_mode(value)
        return self._display_mode

    def set_evidence_profile(self, value: EvidenceProfile | str) -> EvidenceProfile:
        self._evidence_profile = resolve_evidence_profile(value)
        return self._evidence_profile

    def set_point_size(self, value: float) -> float:
        self._point_size = _point_size(value)
        return self._point_size

    def set_band_visible(self, band: str, visible: bool) -> bool:
        if band not in LAYER_BANDS:
            raise LayerControllerError(f'unsupported layer band: {band}')
        if type(visible) is not bool:
            raise LayerControllerError('layer visibility must be boolean')
        self._band_visibility[band] = visible
        return visible

    def set_scan_visible(self, scan_id: str, visible: bool) -> bool:
        if self._project.scan(scan_id) is None:
            raise LayerControllerError(f'unknown scan: {scan_id}')
        if type(visible) is not bool:
            raise LayerControllerError('scan visibility must be boolean')
        self._scan_visibility[scan_id] = visible
        return visible

    def render_plan(self, *, require_available: bool=False) -> RenderPlan:
        return resolve_render_plan(self._project, self._display_mode, evidence_profile=self._evidence_profile, point_size=self._point_size, band_visibility=self._band_visibility, scan_visibility=self._scan_visibility, require_available=require_available)
__all__ = ['DEFAULT_POINT_SIZE', 'EVIDENCE_BANDS', 'LAYER_BANDS', 'POINT_SIZE_MAX', 'POINT_SIZE_MIN', 'RUN_TINTS', 'ActorRequest', 'ColorRule', 'ColorStrategy', 'DisplayMode', 'DisplayModeUnavailableError', 'EvidenceProfile', 'LayerController', 'LayerControllerError', 'RenderPlan', 'initial_evidence_profile', 'resolve_display_mode', 'resolve_evidence_profile', 'resolve_render_plan']
