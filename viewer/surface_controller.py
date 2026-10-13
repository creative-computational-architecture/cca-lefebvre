
from __future__ import annotations
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from geometry.surface_preview import ALGORITHM_VERSION, PRIMARY_ROLE, SOURCE_LAYER, SurfaceCacheKey, SurfacePreviewError, SurfacePreviewResult, SurfacePreviewSpec, SurfaceUnavailableError, build_surface_preview, make_surface_preview_spec, sha256_file, surface_cache_eviction_keys
from viewer.project_model import ProjectModel, ScanEntry, ScanLayer
VISUAL_ONLY_NOTICE = 'VISUAL PREVIEW ONLY - NOT FOR MEASUREMENT'
MEASUREMENT_POLICY = 'MEASUREMENT REQUIRES CALIBRATED TSDF'
APP_FUSED_ROLE = 'APP_FUSED_SURFACE'
APP_FUSED_ROLES = frozenset({PRIMARY_ROLE, APP_FUSED_ROLE})
SIBLING_LAYER = 'rgb/full'
RAW_ALIAS_ROLES = frozenset({'CORRECTED_RAW_RGB', 'CORRECTED_RAW_RGB_PRIMARY', 'CORRECTED_RAW_CONFIDENCE', 'RAW_FRAME_RGB_EVIDENCE', 'RAW_FRAME_CONFIDENCE_EVIDENCE', 'RAW_FRAME_EVIDENCE', 'RAW_MULTIVIEW_EVIDENCE', 'RAW_ALL_EVIDENCE'})
RAW_LAYER = 'rgb/raw'

class SurfaceControllerError(ValueError):
    pass

class SurfaceBackend(str, Enum):
    BPA = 'BPA'
    POISSON = 'POISSON'
    TSDF = 'TSDF'

class SurfaceState(str, Enum):
    IDLE = 'IDLE'
    SELECTED = 'SELECTED'
    BUILDING = 'BUILDING'
    READY = 'READY'
    CACHED = 'CACHED'
    UNAVAILABLE = 'UNAVAILABLE'
    FAILED = 'FAILED'

@dataclass(frozen=True, slots=True)
class ResolvedSurfaceInput:
    scan_id: str
    layer_key: str
    source_role: str
    path: Path
    source_sha256: str
    point_count: int
    spec: SurfacePreviewSpec
    provenance_source: str = SOURCE_LAYER

    @property
    def cache_key(self) -> SurfaceCacheKey:
        return self.spec.cache_key

@dataclass(frozen=True, slots=True)
class SurfaceStatus:
    state: SurfaceState
    code: str
    message: str
    backend: SurfaceBackend
    operator_message: str = ''
    scan_id: str | None = None
    cache_key: SurfaceCacheKey | None = None
    source_path: Path | None = None
    source_sha256_before: str | None = None
    source_sha256_after: str | None = None
    cache_hit: bool = False
    visual_only: bool = True
    measurement_eligible: bool = False
    source_preserved: bool | None = None
    source_role: str | None = None
    source_layer: str | None = None
    algorithm_version: str | None = None
    retained_triangle_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {'state': self.state.value, 'code': self.code, 'message': self.message, 'operator_message': self.operator_message, 'backend': self.backend.value, 'scan_id': self.scan_id, 'cache_key': asdict(self.cache_key) if self.cache_key else None, 'source_path': str(self.source_path) if self.source_path else None, 'source_sha256_before': self.source_sha256_before, 'source_sha256_after': self.source_sha256_after, 'cache_hit': self.cache_hit, 'visual_only': self.visual_only, 'measurement_eligible': self.measurement_eligible, 'source_preserved': self.source_preserved, 'source_role': self.source_role, 'source_layer': self.source_layer, 'algorithm_version': self.algorithm_version, 'retained_triangle_count': self.retained_triangle_count, 'measurement_policy': MEASUREMENT_POLICY, 'notice': VISUAL_ONLY_NOTICE if self.visual_only else None}

@dataclass(frozen=True, slots=True)
class SurfaceOutcome:
    status: SurfaceStatus
    result: SurfacePreviewResult | None = field(default=None, repr=False)

    @property
    def available(self) -> bool:
        return self.result is not None and self.status.state in {SurfaceState.READY, SurfaceState.CACHED}
SurfaceBuilder = Callable[[SurfacePreviewSpec], SurfacePreviewResult]

def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}

def _positive_core_count(scan: ScanEntry, layer: ScanLayer) -> int | None:
    if layer.point_count is not None:
        return layer.point_count if layer.point_count > 0 else None
    candidates = (_mapping(scan.quality.get('viewer_layer_counts')).get('core'), scan.quality.get('viewer_point_count'))
    for value in candidates:
        if type(value) is int and value > 0:
            return value
    return None

def _declared_roles(layer: ScanLayer) -> tuple[object, ...]:
    return (layer.role, layer.metadata.get('source_role'), layer.metadata.get('geometry_role'), layer.metadata.get('archive_geometry_role'))

def _is_raw_alias(scan: ScanEntry, layer: ScanLayer) -> bool:
    if any((value in RAW_ALIAS_ROLES for value in _declared_roles(layer))):
        return True
    raw = scan.layer(RAW_LAYER)
    return raw is not None and bool(layer.path) and (layer.path == raw.path)

def _is_app_fused(scan: ScanEntry, layer: ScanLayer) -> bool:
    if _is_raw_alias(scan, layer):
        return False
    return any((value in APP_FUSED_ROLES for value in _declared_roles(layer)))

def _primary_provenance_source(scan: ScanEntry, layer: ScanLayer) -> str | None:
    if any((value == PRIMARY_ROLE for value in _declared_roles(layer))):
        return layer.key
    sibling = scan.layer(SIBLING_LAYER)
    if sibling is not None and (not _is_raw_alias(scan, sibling)) and any((value == PRIMARY_ROLE for value in _declared_roles(sibling))):
        return SIBLING_LAYER
    scan_level = (scan.metadata.get('full_rgb_geometry_role'), scan.quality.get('full_rgb_geometry_role'), _mapping(scan.metadata.get('geometry_contract')).get('primary'))
    if any((value == PRIMARY_ROLE for value in scan_level)):
        return 'scan'
    return None

def _refusal(english: str, turkish: str) -> SurfaceUnavailableError:
    error = SurfaceUnavailableError(english)
    error.operator_message = turkish
    return error

def operator_text(error: BaseException, scan_id: str | None) -> str:
    carried = getattr(error, 'operator_message', None)
    if isinstance(carried, str) and carried:
        return carried
    label = scan_id or 'secili tarama yok'
    return f'YUZEY YOK / {label} / {error}'

@dataclass(frozen=True, slots=True)
class SurfaceAdmission:
    scan_id: str
    admitted: bool
    reason_code: str
    message: str
    operator_message: str
    layer_key: str | None = None
    provenance_source: str | None = None
    point_count: int | None = None

def _backend(value: SurfaceBackend | str) -> SurfaceBackend:
    if isinstance(value, SurfaceBackend):
        return value
    if isinstance(value, str):
        try:
            return SurfaceBackend(value.strip().upper())
        except ValueError as error:
            raise SurfaceControllerError(f'unknown surface backend: {value}') from error
    raise SurfaceControllerError('surface backend must be a string or SurfaceBackend')

class SurfaceController:

    def __init__(self, project: ProjectModel, *, viewer_voxel_m: float=0.1, project_root: str | Path | None=None, builder: SurfaceBuilder=build_surface_preview) -> None:
        if not isinstance(project, ProjectModel):
            raise SurfaceControllerError('project must be a ProjectModel')
        project.validate()
        if isinstance(viewer_voxel_m, bool) or not isinstance(viewer_voxel_m, (int, float)):
            raise SurfaceControllerError('viewer voxel must be a positive number')
        if not math.isfinite(float(viewer_voxel_m)) or float(viewer_voxel_m) <= 0.0:
            raise SurfaceControllerError('viewer voxel must be a positive number')
        if not callable(builder):
            raise SurfaceControllerError('surface builder must be callable')
        self.project = project
        self.viewer_voxel_m = float(viewer_voxel_m)
        if project_root is not None:
            self.project_root = Path(project_root).resolve()
        elif project.source_manifest:
            self.project_root = Path(project.source_manifest).resolve().parent
        else:
            self.project_root = None
        self._builder = builder
        self._selected_scan_id: str | None = None
        self._cache: dict[SurfaceCacheKey, SurfacePreviewResult] = {}
        self._latest_status = SurfaceStatus(state=SurfaceState.IDLE, code='NO_SCAN_SELECTED', message='Select one scan for surface preview.', operator_message='YUZEY / once bir tarama secin', backend=SurfaceBackend.BPA)

    @property
    def selected_scan_id(self) -> str | None:
        return self._selected_scan_id

    @property
    def latest_status(self) -> SurfaceStatus:
        return self._latest_status

    @property
    def cache_keys(self) -> tuple[SurfaceCacheKey, ...]:
        return tuple(sorted(self._cache, key=lambda key: (key.run_id, repr(key))))

    def select_scan(self, scan_id: str | None) -> SurfaceStatus:
        if scan_id is None:
            self._selected_scan_id = None
            return self._set_status(SurfaceStatus(state=SurfaceState.IDLE, code='NO_SCAN_SELECTED', message='Select one scan for surface preview.', operator_message='YUZEY / once bir tarama secin', backend=SurfaceBackend.BPA))
        if not isinstance(scan_id, str) or not scan_id:
            raise SurfaceControllerError('scan id must be a non-empty string')
        if self.project.scan(scan_id) is None:
            raise SurfaceControllerError(f'unknown project scan: {scan_id}')
        self._selected_scan_id = scan_id
        return self._set_status(SurfaceStatus(state=SurfaceState.SELECTED, code='SCAN_SELECTED', message=f'{scan_id} selected for surface preview.', operator_message=f'YUZEY / {scan_id} secildi', backend=SurfaceBackend.BPA, scan_id=scan_id))

    def _set_status(self, status: SurfaceStatus) -> SurfaceStatus:
        self._latest_status = status
        return status

    def _scan(self) -> ScanEntry:
        if self._selected_scan_id is None:
            raise SurfaceUnavailableError('no scan is selected')
        scan = self.project.scan(self._selected_scan_id)
        if scan is None:
            raise SurfaceUnavailableError('selected scan is no longer in the project')
        return scan

    def _resolve_path(self, scan: ScanEntry, layer: ScanLayer) -> Path:
        candidate = Path(layer.path)
        if not candidate.is_absolute():
            if self.project_root is None:
                raise _refusal(f'{scan.scan_id} rgb/core is relative but project root is unknown', f'YUZEY YOK / {scan.scan_id} / rgb/core yolu goreli, proje koku bilinmiyor')
            candidate = self.project_root / candidate
        path = candidate.resolve()
        if path.suffix.lower() != '.ply':
            raise _refusal(f'{scan.scan_id} rgb/core must be a PLY file', f'YUZEY YOK / {scan.scan_id} / rgb/core bir PLY dosyasi degil')
        if not path.is_file():
            raise _refusal(f'{scan.scan_id} rgb/core file does not exist', f'YUZEY YOK / {scan.scan_id} / rgb/core dosyasi diskte yok')
        return path

    def _admit(self, scan: ScanEntry) -> tuple[ScanLayer, Path, int, str]:
        layer = scan.layer(SOURCE_LAYER)
        if layer is None:
            raise _refusal(f'{scan.scan_id} has no rgb/core layer', f'YUZEY YOK / {scan.scan_id} / projede rgb/core katmani yok')
        if _is_raw_alias(scan, layer):
            raise _refusal(f'{scan.scan_id} rgb/core is a raw RGB alias, not app-fused geometry; PRIMARY_APP_FUSED_SURFACE is required', f'YUZEY YOK / {scan.scan_id} / rgb/core ham RGB kopyasi, uygulama birlestirmesi degil')
        if not _is_app_fused(scan, layer):
            raise _refusal(f"{scan.scan_id} rgb/core role is {layer.role or 'unset'}, not app-fused geometry; PRIMARY_APP_FUSED_SURFACE provenance on rgb/core or rgb/full is required", f"YUZEY YOK / {scan.scan_id} / rgb/core rolu '{layer.role or 'bos'}', uygulama birlestirmesi yuzey degil")
        provenance = _primary_provenance_source(scan, layer)
        if provenance is None:
            raise _refusal(f'{scan.scan_id} carries no PRIMARY_APP_FUSED_SURFACE provenance on rgb/core or rgb/full', f'YUZEY YOK / {scan.scan_id} / rgb/core veya rgb/full uzerinde PRIMARY_APP_FUSED_SURFACE kaydi yok')
        point_count = _positive_core_count(scan, layer)
        if point_count is None:
            raise _refusal(f'{scan.scan_id} has zero or unknown fused-core points', f'YUZEY YOK / {scan.scan_id} / fused-core nokta sayisi sifir veya bilinmiyor')
        path = self._resolve_path(scan, layer)
        return (layer, path, point_count, provenance)

    def check_admission(self, scan_id: str) -> SurfaceAdmission:
        if not isinstance(scan_id, str) or not scan_id:
            raise SurfaceControllerError('scan id must be a non-empty string')
        scan = self.project.scan(scan_id)
        if scan is None:
            raise SurfaceControllerError(f'unknown project scan: {scan_id}')
        try:
            layer, _path, point_count, provenance = self._admit(scan)
        except SurfaceUnavailableError as error:
            return SurfaceAdmission(scan_id=scan_id, admitted=False, reason_code='BPA_INPUT_UNAVAILABLE', message=str(error), operator_message=operator_text(error, scan_id))
        return SurfaceAdmission(scan_id=scan_id, admitted=True, reason_code='BPA_INPUT_ADMITTED', message=f'{scan_id} rgb/core is app-fused geometry with {PRIMARY_ROLE} provenance on {provenance}', operator_message=f'YUZEY HAZIR / {scan_id} / rgb/core uygulama birlestirmesi, {point_count:,} nokta', layer_key=layer.key, provenance_source=provenance, point_count=point_count)

    def admissions(self) -> tuple[SurfaceAdmission, ...]:
        return tuple((self.check_admission(scan.scan_id) for scan in sorted(self.project.scan_rail, key=lambda s: s.rail_order)))

    def admissible_scan_ids(self) -> tuple[str, ...]:
        return tuple((admission.scan_id for admission in self.admissions() if admission.admitted))

    def resolve_selected_input(self) -> ResolvedSurfaceInput:
        scan = self._scan()
        layer, path, point_count, provenance = self._admit(scan)
        source_sha256 = sha256_file(path)
        run = {'id': scan.scan_id, 'full_rgb_geometry_role': PRIMARY_ROLE, 'viewer_layer_counts': {'core': point_count}, 'viewer_layer_paths': {'rgb': {'core': path}}}
        spec = make_surface_preview_spec(run, self.viewer_voxel_m, source_sha256)
        return ResolvedSurfaceInput(scan_id=scan.scan_id, layer_key=SOURCE_LAYER, source_role=PRIMARY_ROLE, path=path, source_sha256=source_sha256, point_count=point_count, spec=spec, provenance_source=provenance)

    def _unavailable_backend(self, backend: SurfaceBackend) -> SurfaceOutcome:
        scan_id = self._selected_scan_id
        label = scan_id or 'secili tarama yok'
        if backend is SurfaceBackend.POISSON:
            message = 'Poisson backend is not implemented or validated.'
            operator = f'YUZEY YOK / {label} / Poisson arka ucu bu surumde yok (uygulanmadi ve dogrulanmadi)'
            visual_only = True
        else:
            message = 'Calibrated TSDF measurement backend is not connected.'
            operator = f'YUZEY YOK / {label} / kalibre TSDF olcum arka ucu bagli degil'
            visual_only = False
        status = self._set_status(SurfaceStatus(state=SurfaceState.UNAVAILABLE, code=f'{backend.value}_BACKEND_UNAVAILABLE', message=message, operator_message=operator, backend=backend, scan_id=scan_id, visual_only=visual_only, measurement_eligible=False))
        return SurfaceOutcome(status=status)

    def request_preview(self, backend: SurfaceBackend | str=SurfaceBackend.BPA) -> SurfaceOutcome:
        selected_backend = _backend(backend)
        if selected_backend is not SurfaceBackend.BPA:
            return self._unavailable_backend(selected_backend)
        try:
            resolved = self.resolve_selected_input()
        except (OSError, SurfacePreviewError) as error:
            status = self._set_status(SurfaceStatus(state=SurfaceState.UNAVAILABLE, code='BPA_INPUT_UNAVAILABLE', message=str(error), operator_message=operator_text(error, self._selected_scan_id), backend=SurfaceBackend.BPA, scan_id=self._selected_scan_id))
            return SurfaceOutcome(status=status)
        cached = self._cache.get(resolved.cache_key)
        if cached is not None:
            status = self._set_status(self._ready_status(resolved, cached, cache_hit=True))
            return SurfaceOutcome(status=status, result=cached)
        self._set_status(SurfaceStatus(state=SurfaceState.BUILDING, code='BPA_BUILDING', message=f'Building BPA preview for {resolved.scan_id}.', operator_message=f'YUZEY / {resolved.scan_id} BPA onizlemesi olusturuluyor', backend=SurfaceBackend.BPA, scan_id=resolved.scan_id, cache_key=resolved.cache_key, source_path=resolved.path, source_sha256_before=resolved.source_sha256, source_role=resolved.source_role, source_layer=resolved.layer_key, algorithm_version=ALGORITHM_VERSION))
        try:
            result = self._builder(resolved.spec)
            source_after = sha256_file(resolved.path)
            self._validate_result(resolved, result, source_after)
        except SurfaceUnavailableError as error:
            source_after = _safe_hash(resolved.path)
            status = self._set_status(SurfaceStatus(state=SurfaceState.UNAVAILABLE, code='BPA_GEOMETRY_UNAVAILABLE', message=str(error), operator_message=operator_text(error, resolved.scan_id), backend=SurfaceBackend.BPA, scan_id=resolved.scan_id, cache_key=resolved.cache_key, source_path=resolved.path, source_sha256_before=resolved.source_sha256, source_sha256_after=source_after, source_preserved=source_after == resolved.source_sha256, source_role=resolved.source_role, source_layer=resolved.layer_key, algorithm_version=ALGORITHM_VERSION))
            return SurfaceOutcome(status=status)
        except (OSError, RuntimeError, SurfacePreviewError) as error:
            source_after = _safe_hash(resolved.path)
            status = self._set_status(SurfaceStatus(state=SurfaceState.FAILED, code='BPA_BUILD_FAILED', message=str(error), operator_message=f'YUZEY BASARISIZ / {resolved.scan_id} / {error}', backend=SurfaceBackend.BPA, scan_id=resolved.scan_id, cache_key=resolved.cache_key, source_path=resolved.path, source_sha256_before=resolved.source_sha256, source_sha256_after=source_after, source_preserved=source_after == resolved.source_sha256, source_role=resolved.source_role, source_layer=resolved.layer_key, algorithm_version=ALGORITHM_VERSION))
            return SurfaceOutcome(status=status)
        self._remove_obsolete_scan_keys(resolved.scan_id, resolved.cache_key)
        self._cache[resolved.cache_key] = result
        status = self._set_status(self._ready_status(resolved, result, cache_hit=False))
        return SurfaceOutcome(status=status, result=result)

    def request(self, backend: SurfaceBackend | str=SurfaceBackend.BPA) -> SurfaceOutcome:
        return self.request_preview(backend)

    def _validate_result(self, resolved: ResolvedSurfaceInput, result: SurfacePreviewResult, source_after: str) -> None:
        if not isinstance(result, SurfacePreviewResult):
            raise SurfacePreviewError('BPA builder returned an invalid result')
        if result.cache_key != resolved.cache_key or result.spec != resolved.spec:
            raise SurfacePreviewError('BPA result does not match the requested cache key')
        if source_after != resolved.source_sha256 or result.source_sha256_before != resolved.source_sha256 or result.source_sha256_after != resolved.source_sha256:
            raise SurfacePreviewError('fused-core source changed during BPA build')
        if result.source_point_count_before != resolved.point_count:
            raise SurfacePreviewError('BPA result point count violates project contract')
        if result.source_point_count_after != resolved.point_count:
            raise SurfacePreviewError('BPA builder changed its source point count')

    def _ready_status(self, resolved: ResolvedSurfaceInput, result: SurfacePreviewResult, *, cache_hit: bool) -> SurfaceStatus:
        return SurfaceStatus(state=SurfaceState.CACHED if cache_hit else SurfaceState.READY, code='BPA_CACHED' if cache_hit else 'BPA_READY', message=f'BPA preview ready for {resolved.scan_id}. {VISUAL_ONLY_NOTICE}', operator_message=f'YUZEY HAZIR / {resolved.scan_id} / {result.retained_triangle_count:,} ucgen / temsil destegi {result.representation_filtered_point_count:,}/{result.representation_input_point_count:,} nokta / {VISUAL_ONLY_NOTICE}', backend=SurfaceBackend.BPA, scan_id=resolved.scan_id, cache_key=resolved.cache_key, source_path=resolved.path, source_sha256_before=result.source_sha256_before, source_sha256_after=result.source_sha256_after, cache_hit=cache_hit, visual_only=True, measurement_eligible=False, source_preserved=result.source_sha256_before == result.source_sha256_after and result.source_point_count_before == result.source_point_count_after, source_role=resolved.source_role, source_layer=resolved.layer_key, algorithm_version=ALGORITHM_VERSION, retained_triangle_count=result.retained_triangle_count)

    def _remove_obsolete_scan_keys(self, scan_id: str, retained_key: SurfaceCacheKey) -> None:
        for key in tuple(self._cache):
            if key.run_id == scan_id and key != retained_key:
                self._cache.pop(key)

    def cached_result(self, key: SurfaceCacheKey) -> SurfacePreviewResult | None:
        return self._cache.get(key)

    def evict_cache(self, *, visible_scan_ids: set[str], protected_keys: set[SurfaceCacheKey] | None=None) -> tuple[SurfaceCacheKey, ...]:
        evicted = surface_cache_eviction_keys(set(self._cache), visible_run_ids=set(visible_scan_ids), protected_keys=set(protected_keys or ()))
        for key in evicted:
            self._cache.pop(key, None)
        return evicted

def _safe_hash(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except OSError:
        return None
__all__ = ['APP_FUSED_ROLE', 'APP_FUSED_ROLES', 'MEASUREMENT_POLICY', 'RAW_ALIAS_ROLES', 'SIBLING_LAYER', 'VISUAL_ONLY_NOTICE', 'ResolvedSurfaceInput', 'SurfaceAdmission', 'SurfaceBackend', 'SurfaceController', 'SurfaceControllerError', 'SurfaceOutcome', 'SurfaceState', 'SurfaceStatus', 'operator_text']
