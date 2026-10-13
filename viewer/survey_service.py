
from __future__ import annotations
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from viewer.project_model import ScanEntry, ScanLayer
from viewer.raw_processing_service import default_processed_output_root
SURVEY_SUBDIR = 'survey'
MANIFEST_NAME = 'survey-manifest.json'
WALLS_NAME = 'walls.json'
STRUCTURES_NAME = 'structures.json'
LOW_CONFIDENCE = 'LOW'
DRAFT_LABEL = 'SURVEY DRAFT / UNCALIBRATED'
_SCAN_ID = re.compile('^[A-Za-z0-9][A-Za-z0-9._-]*$')
_STAMP = re.compile('^[0-9]{8}T[0-9]{6}Z$')
_PLY_SUFFIX = '.ply'
_CAD_Z_UP = 'CAD_Z_up'
_ARKIT_Y_UP = 'ARKit_Y_up'
_SUPPORTED_SOURCE_AXES = frozenset({'auto', _CAD_Z_UP, _ARKIT_Y_UP, 'X_up', 'X_down', 'Y_up', 'Y_down', 'Z_up', 'Z_down'})
_TURKISH_ASCII = str.maketrans({'i': 'i', 'I': 'I'})

class SurveyServiceError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class SurveyPackSummary:
    scan_id: str
    source: Path
    output_dir: Path
    manifest_path: Path
    wall_count: int
    structure_count: int
    low_confidence_count: int
    result: object

    def status_line(self) -> str:
        return f'SURVEY PACK READY / {self.wall_count} walls / {self.structure_count} structures / {self.low_confidence_count} ELLE OLC'

def utc_stamp(now: datetime | None=None) -> str:
    moment = datetime.now(UTC) if now is None else now
    if moment.tzinfo is None:
        raise SurveyServiceError('survey stamps must come from a timezone-aware time')
    return moment.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')

def _visible_ply_layers(scan: ScanEntry) -> list[ScanLayer]:
    candidates = [layer for layer in scan.layers if layer.visible and Path(layer.path).suffix.lower() == _PLY_SUFFIX]

    def priority(layer: ScanLayer) -> tuple[int, int]:
        axis = layer.metadata.get('source_axis')
        if axis == _CAD_Z_UP:
            axis_rank = 0
        elif isinstance(axis, str) and axis in _SUPPORTED_SOURCE_AXES:
            axis_rank = 1
        elif axis is None:
            axis_rank = 2
        else:
            axis_rank = 3
        colour_rank = 0 if layer.key.startswith('rgb/') else 1
        return (axis_rank, colour_rank)
    candidates.sort(key=priority)
    return candidates

def _resolve_survey_layer(scan: ScanEntry) -> tuple[ScanLayer, Path]:
    if not isinstance(scan, ScanEntry):
        raise SurveyServiceError('survey source must be a project ScanEntry')
    layers = _visible_ply_layers(scan)
    if not layers:
        keys = ', '.join((layer.key for layer in scan.layers)) or 'none'
        raise SurveyServiceError(f'scan {scan.scan_id} has no visible PLY layer (layers: {keys})')
    compatible = [layer for layer in layers if layer.metadata.get('source_axis') is None or (isinstance(layer.metadata.get('source_axis'), str) and layer.metadata.get('source_axis') in _SUPPORTED_SOURCE_AXES)]
    unsupported = [layer for layer in layers if layer not in compatible]
    missing: list[str] = []
    for layer in compatible:
        path = Path(layer.path).expanduser().resolve()
        if path.is_file():
            return (layer, path)
        missing.append(f'{layer.key} -> {path}')
    if missing:
        raise SurveyServiceError(f'visible rolove PLY layer file is missing for scan {scan.scan_id}: ' + '; '.join(missing))
    if unsupported:
        axes = ', '.join(sorted({str(layer.metadata.get('source_axis')) for layer in unsupported}))
        raise SurveyServiceError(f'scan {scan.scan_id} has no PLY with a supported source axis (source axis: {axes})')
    raise SurveyServiceError(f'scan {scan.scan_id} has no usable rolove PLY layer')

def resolve_survey_source(scan: ScanEntry) -> Path:
    return _resolve_survey_layer(scan)[1]

def survey_output_dir(scan_id: str, root: str | Path | None=None, *, stamp: str | None=None) -> Path:
    if not isinstance(scan_id, str) or not _SCAN_ID.fullmatch(scan_id):
        raise SurveyServiceError('survey output needs a safe ASCII scan id: letters, digits, dot, dash')
    stamp_value = utc_stamp() if stamp is None else stamp
    if not isinstance(stamp_value, str) or not _STAMP.fullmatch(stamp_value):
        raise SurveyServiceError(f'survey stamp must look like 20260904T101500Z, got {stamp_value!r}')
    base = default_processed_output_root() if root is None else Path(root).expanduser().resolve()
    return (base / scan_id / SURVEY_SUBDIR / stamp_value).resolve()

def _ascii_project_name(name: str, fallback: str) -> str:
    normalized = unicodedata.normalize('NFKD', str(name).translate(_TURKISH_ASCII))
    ascii_name = normalized.encode('ascii', 'ignore').decode('ascii')
    cleaned = re.sub('\\s+', ' ', ascii_name).strip()
    return cleaned or fallback

def _default_config(config_type: Any) -> Any:
    try:
        return config_type()
    except TypeError:
        from geometry.survey_frame import SurveyFrameConfig
        from geometry.survey_plan import SurveyPlanConfig
        return config_type(frame=SurveyFrameConfig(), plan=SurveyPlanConfig())

def _optional_count(result: object, name: str) -> int | None:
    value = getattr(result, name, None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        raise SurveyServiceError(f'survey pack reported a negative {name}')
    return value

def _entries(payload: object, label: str) -> list[Mapping[str, Any]]:
    items: object = payload
    if isinstance(payload, Mapping):
        lists = [value for value in payload.values() if isinstance(value, list)]
        if len(lists) != 1:
            raise SurveyServiceError(f'{label} does not hold exactly one entry list')
        items = lists[0]
    if not isinstance(items, list):
        raise SurveyServiceError(f'{label} must be a JSON list or envelope')
    return [item for item in items if isinstance(item, Mapping)]

def _read_json(path: Path, label: str) -> object:
    if not path.is_file():
        raise SurveyServiceError(f'survey pack did not produce {label}: {path}')
    try:
        return json.loads(path.read_text(encoding='ascii'))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise SurveyServiceError(f'{label} is not ASCII JSON: {error}') from error

def _counts_from_files(output_dir: Path) -> tuple[int, int, int]:
    walls = _entries(_read_json(output_dir / WALLS_NAME, WALLS_NAME), WALLS_NAME)
    structures = _entries(_read_json(output_dir / STRUCTURES_NAME, STRUCTURES_NAME), STRUCTURES_NAME)
    low = sum((1 for entry in (*walls, *structures) if str(entry.get('confidence', '')).upper() == LOW_CONFIDENCE))
    return (len(walls), len(structures), low)

def summarize_survey_pack(result: object, *, scan_id: str, source: Path, output_dir: Path) -> SurveyPackSummary:
    manifest_value = getattr(result, 'manifest_path', None)
    manifest_path = Path(manifest_value).expanduser().resolve() if manifest_value is not None else (output_dir / MANIFEST_NAME).resolve()
    output_value = getattr(result, 'output_dir', None)
    pack_dir = Path(output_value).expanduser().resolve() if output_value is not None else output_dir.resolve()
    if not manifest_path.is_file():
        raise SurveyServiceError(f'survey pack did not write its manifest: {manifest_path}')
    walls = _optional_count(result, 'wall_count')
    structures = _optional_count(result, 'structure_count')
    low = _optional_count(result, 'low_confidence_count')
    if low is None:
        low = _optional_count(result, 'low_confidence')
    if walls is None or structures is None or low is None:
        walls, structures, low = _counts_from_files(pack_dir)
    return SurveyPackSummary(scan_id=scan_id, source=source, output_dir=pack_dir, manifest_path=manifest_path, wall_count=walls, structure_count=structures, low_confidence_count=low, result=result)

def run_survey_pack(scan: ScanEntry, *, project_name: str, config: object | None=None, progress: Callable[[str], None] | None=None, stamp: str | None=None, output_root: str | Path | None=None) -> SurveyPackSummary:
    source_layer, source = _resolve_survey_layer(scan)
    output_dir = survey_output_dir(scan.scan_id, output_root, stamp=stamp)
    if output_dir.exists() and (not output_dir.is_dir()):
        raise SurveyServiceError(f'survey output path is not a folder: {output_dir}')
    if output_dir == source.parent or output_dir in source.parents:
        raise SurveyServiceError('survey output must not be the source folder')
    from export.survey_pack import SurveyPackConfig, build_survey_pack
    from geometry.survey_frame import SurveyFrameError
    pre_transform: Sequence[Sequence[float]] = tuple((tuple((float(value) for value in row)) for row in scan.transform))
    base_config = _default_config(SurveyPackConfig) if config is None else config
    source_axis = source_layer.metadata.get('source_axis') or 'auto'
    try:
        pack_config = replace(base_config, pre_transform=pre_transform, source_axis=source_axis)
    except TypeError as error:
        raise SurveyServiceError('survey config must be a dataclass with pre_transform and source_axis fields') from error
    name = _ascii_project_name(project_name, scan.scan_id)

    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    def pack_progress(stage: str, fraction: float) -> None:
        emit(f'SURVEY PACK / {scan.scan_id} / {stage.upper()} / {float(fraction):.0%}')
    emit(f'SURVEY PACK / {scan.scan_id} / {DRAFT_LABEL}')
    created_output = not output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = build_survey_pack(source, output_dir, project_name=name, config=pack_config, progress=pack_progress)
    except SurveyFrameError as error:
        if created_output:
            try:
                output_dir.rmdir()
            except OSError:
                pass
        raise SurveyServiceError(f'rolove frame could not be established from ground and wall surfaces: {error}; verify metre units or set the source up axis') from error
    summary = summarize_survey_pack(result, scan_id=scan.scan_id, source=source, output_dir=output_dir)
    emit(summary.status_line())
    return summary
__all__ = ['DRAFT_LABEL', 'MANIFEST_NAME', 'STRUCTURES_NAME', 'SURVEY_SUBDIR', 'WALLS_NAME', 'SurveyPackSummary', 'SurveyServiceError', 'resolve_survey_source', 'run_survey_pack', 'summarize_survey_pack', 'survey_output_dir', 'utc_stamp']
