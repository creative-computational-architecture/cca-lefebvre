
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import sys
import tempfile
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import numpy as np
from export.survey_dxf import SurveyDxfError, SurveyDxfResult, export_survey_dxf
from export.survey_sheets import HAND_MEASURE_LABEL, SHEET_STAMP, SheetStyle, SurveySheetError, make_box_sheet, make_elevation_sheet, make_plan_sheet, make_section_sheet, make_slice_sheets
from geometry.alignment_hypotheses import Matrix4, validate_zup_rigid_transform
from geometry.ply_preserve import PlyFormatError, iter_ply_vertex_chunks, read_ply_schema
from geometry.survey_frame import SourceAxis, SourceOrientation, SurveyFrame, SurveyFrameConfig, SurveyFrameError, apply_survey_frame, estimate_survey_frame, frame_to_dict, normalize_source_points, orientation_to_dict
from geometry.survey_plan import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM, SCHEMA_STRUCTURES, SCHEMA_WALLS, SurveyPlanConfig, SurveyPlanError, build_slices, cross_walls, facade_profile, find_facade, front_view_min_y, partitions_along_facade, segment_structures, vectorize_walls
from ingest.e57_cloud import E57ConversionError, convert_e57_to_ply
SCHEMA_PACK = 'cca.survey-pack.v1'
MANIFEST_NAME = 'survey-manifest.json'
REPORT_NAME = 'REPORT.md'
SHEETS_DIRNAME = 'sheets'
WALLS_NAME = 'walls.json'
PARTITIONS_NAME = 'partitions.json'
STRUCTURES_NAME = 'structures.json'
PROFILE_NAME = 'facade_profile.json'
PACK_STAMP = SHEET_STAMP
_UNITS = ('cm', 'mm')
_SOURCE_AXES = ('auto', 'CAD_Z_up', 'ARKit_Y_up', 'X_up', 'X_down', 'Y_up', 'Y_down', 'Z_up', 'Z_down')
_UNIT_SCALE: dict[str, float] = {'cm': 100.0, 'mm': 1000.0}
_CONFIDENCES = (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW)
_HASH_CHUNK_BYTES = 1024 * 1024
_READ_CHUNK_VERTICES = 500000
_MIN_POINTS = 1000
ProgressCallback = Callable[[str, float], None]

class SurveyPackError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class SurveyPackConfig:
    frame: SurveyFrameConfig = field(default_factory=SurveyFrameConfig)
    plan: SurveyPlanConfig = field(default_factory=SurveyPlanConfig)
    label_unit: Literal['cm', 'mm'] = 'cm'
    dxf_unit: Literal['cm', 'mm'] = 'cm'
    sheets: bool = True
    dxf: bool = True
    report: bool = True
    sheet_dpi: int = 150
    source_axis: SourceAxis = 'auto'
    pre_transform: Matrix4 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.frame, SurveyFrameConfig):
            raise SurveyPackError('frame must be a SurveyFrameConfig')
        if not isinstance(self.plan, SurveyPlanConfig):
            raise SurveyPackError('plan must be a SurveyPlanConfig')
        if self.label_unit not in _UNITS or self.dxf_unit not in _UNITS:
            raise SurveyPackError("label_unit and dxf_unit must be 'cm' or 'mm'")
        if self.source_axis not in _SOURCE_AXES:
            raise SurveyPackError('source_axis must be auto, CAD_Z_up, ARKit_Y_up, or a signed axis')
        for name in ('sheets', 'dxf', 'report'):
            if not isinstance(getattr(self, name), bool):
                raise SurveyPackError(f'{name} must be a bool')
        if not isinstance(self.sheet_dpi, int) or isinstance(self.sheet_dpi, bool) or (not 50 <= self.sheet_dpi <= 600):
            raise SurveyPackError('sheet_dpi must be an integer between 50 and 600')
        if self.pre_transform is not None:
            try:
                matrix = validate_zup_rigid_transform(self.pre_transform, label='pre_transform')
            except ValueError as exc:
                raise SurveyPackError(str(exc)) from exc
            object.__setattr__(self, 'pre_transform', matrix)

@dataclass(frozen=True, slots=True)
class SurveyPackResult:
    output_dir: Path
    manifest_path: Path
    source_path: Path
    source_sha256: str
    project_name: str
    frame: SurveyFrame
    point_count: int
    wall_count: int
    cross_wall_count: int
    structure_count: int
    partition_count: int
    low_confidence: int
    confidence_counts: dict[str, dict[str, int]]
    hand_measure: tuple[dict[str, Any], ...]
    sheet_paths: tuple[Path, ...]
    dxf_path: Path | None
    report_path: Path | None
    inventory: dict[str, dict[str, Any]]
    elapsed_s: float

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (block := stream.read(_HASH_CHUNK_BYTES)):
            digest.update(block)
    return digest.hexdigest()

def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f'{type(value).__name__} is not JSON serialisable')

def _atomic_write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path

def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    encoded = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=_json_default) + '\n').encode('ascii')
    return _atomic_write_bytes(path, encoded)

def _write_text(path: Path, text: str) -> Path:
    return _atomic_write_bytes(path, text.encode('ascii'))

def _notify(progress: ProgressCallback | None, stage: str, fraction: float) -> None:
    if progress is not None:
        progress(stage, float(fraction))

def ascii_project_name(text: str, fallback: str='SURVEY') -> str:
    normalised = unicodedata.normalize('NFKD', str(text))
    folded = ''.join((ch for ch in normalised if not unicodedata.combining(ch)))
    folded = folded.replace('i', 'i').replace('\xdf', 'ss')
    cleaned = ''.join((ch if ch.isascii() and ch.isprintable() else '_' for ch in folded)).strip()
    return cleaned or fallback

def _project_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SurveyPackError('project_name must be a non-empty string')
    if not value.isascii() or not value.isprintable():
        raise SurveyPackError('project_name must be printable ASCII (use ascii_project_name to fold it)')
    return value.strip()

def _read_points(source: Path) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    try:
        schema = read_ply_schema(source)
    except PlyFormatError as exc:
        raise SurveyPackError(f'source PLY cannot be read: {exc}') from exc
    names = [prop.name for prop in schema.vertex.properties]
    has_rgb = all((name in names for name in ('red', 'green', 'blue')))
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    colours: list[np.ndarray] = []
    try:
        for chunk in iter_ply_vertex_chunks(source, chunk_vertices=_READ_CHUNK_VERTICES):
            xs.append(np.asarray(chunk['x'], dtype=np.float64))
            ys.append(np.asarray(chunk['y'], dtype=np.float64))
            zs.append(np.asarray(chunk['z'], dtype=np.float64))
            if has_rgb:
                colours.append(np.column_stack([chunk['red'], chunk['green'], chunk['blue']]))
    except PlyFormatError as exc:
        raise SurveyPackError(f'source PLY cannot be read: {exc}') from exc
    if not xs:
        raise SurveyPackError('source PLY declares no vertices')
    points = np.column_stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)])
    if points.shape[0] < _MIN_POINTS:
        raise SurveyPackError(f'source PLY has {points.shape[0]} points; at least {_MIN_POINTS} are needed')
    if not np.isfinite(points).all():
        raise SurveyPackError('source PLY holds non-finite XYZ values; the survey frame cannot be estimated')
    rgb: np.ndarray | None = None
    if has_rgb:
        stacked = np.concatenate(colours)
        if stacked.dtype.kind == 'f':
            if float(np.nanmax(stacked)) <= 1.0:
                stacked = stacked * 255.0
            stacked = np.nan_to_num(stacked)
        rgb = np.clip(np.rint(stacked), 0, 255).astype(np.uint8)
    return (points, rgb, names)

def _apply_pre_transform(points: np.ndarray, transform: Matrix4 | None) -> np.ndarray:
    if transform is None:
        return points
    matrix = np.asarray(transform, dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]

def _count_by_confidence(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in _CONFIDENCES}
    for item in items:
        confidence = str(item.get('confidence', CONFIDENCE_LOW))
        counts[confidence] = counts.get(confidence, 0) + 1
    return counts

def _partition_confidence_counts(partitions: Mapping[str, Any]) -> dict[str, int]:
    counts = {name: 0 for name in _CONFIDENCES}
    for confidence in partitions.get('confidence', []):
        counts[str(confidence)] = counts.get(str(confidence), 0) + 1
    return counts

def _hand_measure_items(walls: Sequence[Mapping[str, Any]], partitions: Mapping[str, Any], structures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for wall in walls:
        if wall.get('confidence') == CONFIDENCE_LOW:
            items.append({'kind': 'wall', 'id': str(wall.get('id')), 'axis': str(wall.get('axis')), 'pos_m': wall.get('pos_m'), 'start_m': wall.get('start_m'), 'end_m': wall.get('end_m'), 'reason': 'single slice band support'})
    xs = [float(v) for v in partitions.get('x_positions_m', [])]
    confidences = [str(v) for v in partitions.get('confidence', [])]
    if len(confidences) == len(xs):
        for index, ((a, b), (ca, cb)) in enumerate(zip(itertools.pairwise(xs), itertools.pairwise(confidences))):
            if CONFIDENCE_LOW in (ca, cb):
                items.append({'kind': 'partition', 'id': f'P{index + 1:02d}', 'start_m': a, 'end_m': b, 'span_m': round(b - a, 3), 'reason': 'low confidence partition wall at one end'})
    for structure in structures:
        identifier = str(structure.get('id'))
        if structure.get('confidence') == CONFIDENCE_LOW:
            items.append({'kind': 'structure', 'id': identifier, 'width_m': structure.get('width_m'), 'depth_m': structure.get('depth_m'), 'reason': 'low confidence bounding wall'})
        if structure.get('eave_z_m') is None:
            items.append({'kind': 'eave', 'id': identifier, 'reason': 'no facade plane support for the eave level'})
        if structure.get('ridge_z_m') is None:
            items.append({'kind': 'ridge', 'id': identifier, 'reason': 'no roof support for the ridge level'})
    return items

def _inventory(output_dir: Path, *, skip: Sequence[Path]) -> dict[str, dict[str, Any]]:
    skipped = {path.resolve() for path in skip}
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.rglob('*')):
        if not path.is_file() or path.resolve() in skipped or path.name.startswith('.'):
            continue
        inventory[path.relative_to(output_dir).as_posix()] = {'bytes': path.stat().st_size, 'sha256': _sha256(path)}
    return inventory

def _length(value: Any, unit: str) -> str:
    if value is None:
        return HAND_MEASURE_LABEL
    return f'{float(value) * _UNIT_SCALE[unit]:.0f}'

def _level(value: Any) -> str:
    if value is None:
        return HAND_MEASURE_LABEL
    return f'+{float(value):.2f}'

def _hand_measure_line(item: Mapping[str, Any], unit: str) -> str:
    kind = item.get('kind')
    identifier = item.get('id')
    if kind == 'wall':
        axis = 'x eksenine paralel' if item.get('axis') == 'x' else 'y eksenine paralel'
        where = 'y' if item.get('axis') == 'x' else 'x'
        return f"**{identifier}** duvar ({axis}, {where} = {_length(item.get('pos_m'), unit)} {unit}, {_length(item.get('start_m'), unit)}-{_length(item.get('end_m'), unit)} {unit}): tek kesit bandinda goruldu, konumu ve varligi sahada dogrulanmali."
    if kind == 'partition':
        return f"**{identifier}** bolme araligi x = {_length(item.get('start_m'), unit)} -> {_length(item.get('end_m'), unit)} {unit} ({_length(item.get('span_m'), unit)} {unit}): uclarindan biri dusuk guvenli bolme duvari."
    if kind == 'structure':
        return f"**{identifier}** genislik {_length(item.get('width_m'), unit)} {unit} / derinlik {_length(item.get('depth_m'), unit)} {unit}: sinir duvari dusuk guvenli."
    if kind == 'eave':
        return f'**{identifier}** sacak kotu: cephe duzleminde yeterli nokta yok.'
    if kind == 'ridge':
        return f'**{identifier}** mahya kotu / cati tipi: cati uzerinde yeterli nokta yok.'
    return f"**{identifier}** {item.get('reason', '')}".strip()

def _report_text(*, project: str, source: Path, sha_before: str, sha_after: str, point_count: int, has_rgb: bool, created_utc: str, elapsed_s: float, frame: SurveyFrame, frame_config: SurveyFrameConfig, source_orientation: SourceOrientation, walls: Sequence[Mapping[str, Any]], cross: Sequence[Mapping[str, Any]], partitions: Mapping[str, Any], structures: Sequence[Mapping[str, Any]], facade: Mapping[str, Any], min_y: float, hand_measure: Sequence[Mapping[str, Any]], unit: str, dxf: SurveyDxfResult | None, dxf_unit: str, sheets: Sequence[str], inventory: Mapping[str, Mapping[str, Any]], origin: Mapping[str, Any] | None) -> str:
    counts = _count_by_confidence(walls)
    source_name = ascii_project_name(source.name, 'scan.ply')
    lines: list[str] = [f'# CCA // {project} // {PACK_STAMP}', '', '> Bu belge kalibre edilmemis bir rolove **TASLAGIDIR**. Olculer LiDAR nokta bulutundan', f'> turetilmistir; `{HAND_MEASURE_LABEL}` etiketli kalemler sahada elle olculmelidir. Hicbir nokta', '> menzil veya guven esigiyle atilmamistir; dusuk destekli ogeler silinmez, isaretlenir.', '', '## Kaynak', '', '| Alan | Deger |', '|---|---|', f'| Dosya | `{source_name}` |', f'| SHA-256 (okumadan once) | `{sha_before}` |', f'| SHA-256 (paket sonrasi) | `{sha_after}` |', f'| Nokta sayisi | {point_count:,} |', f"| RGB | {('var' if has_rgb else 'yok')} |", f'| Uretim (UTC) | {created_utc} |', f'| Sure | {elapsed_s:.1f} s |', f'| Birimler | ic hesap m (Z yukari); tablolar {unit}; DXF {dxf_unit} |', f'| Kaynak ekseni | istenen `{source_orientation.requested_axis}`; bulunan `{source_orientation.detected_up_axis}`; zemin destegi {source_orientation.ground_support_ratio:.3f} |']
    if origin:
        origin_path = ascii_project_name(str(origin.get('path', '')), '-')
        origin_kind = ascii_project_name(str(origin.get('kind', '')), '-')
        origin_hash = ascii_project_name(str(origin.get('sha256', '')), '-')
        lines.append(f'| Kaynak oncesi | `{origin_path}` ({origin_kind}, sha256 `{origin_hash}`) |')
    extents = frame.extents_m
    front_note = '180 deg cevrildi (tarayici tarafi +y idi)' if frame.flipped_180 else 'korundu (tarayici tarafi -y)'
    if frame_config.front_side != 'auto':
        front_note += f' - operator secimi `{frame_config.front_side}`'
    lines += ['', '## Cerceve (survey frame)', '', '| Alan | Deger |', '|---|---|', f'| Zemin egimi duzeltmesi | {frame.ground_tilt_deg:.2f} deg (sinir {frame_config.max_tilt_deg:.2f} deg) |', f'| Zemin sigma | {frame.ground_sigma_m * 100.0:.1f} cm |', f'| Duvar yaw | {frame.yaw_deg:.2f} deg |', f"| Uzun eksen 90 deg donduruldu | {('evet' if frame.long_axis_rotated_90 else 'hayir')} |", f'| On yuz | {front_note} |', f"| Zemin noktalari (kucuk y / buyuk y) | {frame.scanner_side_counts.get('ground_points_small_y', 0):,} / {frame.scanner_side_counts.get('ground_points_large_y', 0):,} |", f"| Cephe y | {_length(facade.get('y_m'), unit)} {unit} (x {_length(facade.get('x_start_m'), unit)}-{_length(facade.get('x_end_m'), unit)} {unit}, guven {facade.get('confidence')}) |", f'| On gorunus alt siniri | y >= {_length(min_y, unit)} {unit} |', f"| Uzanim x / y / z (m) | {extents['x'][0]:.2f}..{extents['x'][1]:.2f} / {extents['y'][0]:.2f}..{extents['y'][1]:.2f} / {extents['z'][0]:.2f}..{extents['z'][1]:.2f} |", '| Olcek | 1.000 (rijit, det +1) |', '', f'## Yapilar ({len(structures)})', '', f'| ID | x0 ({unit}) | x1 ({unit}) | Genislik ({unit}) | Derinlik ({unit}) | Sacak (m) | Mahya (m) | Cati | Guven |', '|---|---|---|---|---|---|---|---|---|']
    for structure in structures:
        lines.append(f"| {structure.get('id')} | {_length(structure.get('x_start_m'), unit)} | {_length(structure.get('x_end_m'), unit)} | {_length(structure.get('width_m'), unit)} | {_length(structure.get('depth_m'), unit)} | {_level(structure.get('eave_z_m'))} | {_level(structure.get('ridge_z_m'))} | {structure.get('roof_type', 'belirsiz')} | {structure.get('confidence')} |")
    if not structures:
        lines.append('| - | - | - | - | - | - | - | - | - |')
    xs = [float(v) for v in partitions.get('x_positions_m', [])]
    spacings = [float(v) for v in partitions.get('spacings_m', [])]
    confidences = [str(v) for v in partitions.get('confidence', [])]
    lines += ['', f'## Bolme zinciri (cephe boyunca, {len(xs)} bolme duvari)', '', f'| # | Duvar | x ({unit}) | Sonraki aralik ({unit}) | Derinlik ({unit}) | Guven |', '|---|---|---|---|---|---|']
    for index, x in enumerate(xs):
        spacing = _length(spacings[index], unit) if index < len(spacings) else '-'
        wall = cross[index] if index < len(cross) else {}
        confidence = confidences[index] if index < len(confidences) else '-'
        lines.append(f"| {index + 1} | {wall.get('id', '-')} | {_length(x, unit)} | {spacing} | {(_length(wall.get('depth_m'), unit) if wall else '-')} | {confidence} |")
    if not xs:
        lines.append('| - | - | - | - | - | - |')
    lines.append('')
    lines.append(f"Cephe bolme zinciri toplami: **{_length(partitions.get('total_m', 0.0), unit)} {unit}**")
    lines += ['', f'## Duvarlar ({len(walls)}: HIGH {counts[CONFIDENCE_HIGH]} / MEDIUM {counts[CONFIDENCE_MEDIUM]} / LOW {counts[CONFIDENCE_LOW]})', '', f'| ID | Eksen | Konum ({unit}) | Baslangic ({unit}) | Bitis ({unit}) | Uzunluk ({unit}) | Kalinlik ({unit}) | Kesit bandi | Guven |', '|---|---|---|---|---|---|---|---|---|']
    for wall in walls:
        thickness = _length(wall.get('thickness_m'), unit) if wall.get('thickness_m') is not None else '-'
        lines.append(f"| {wall.get('id')} | {wall.get('axis')} | {_length(wall.get('pos_m'), unit)} | {_length(wall.get('start_m'), unit)} | {_length(wall.get('end_m'), unit)} | {_length(wall.get('length_m'), unit)} | {thickness} | {len(wall.get('slice_heights', []))} | {wall.get('confidence')} |")
    if not walls:
        lines.append('| - | - | - | - | - | - | - | - | - |')
    lines += ['', f'## {HAND_MEASURE_LABEL} listesi ({len(hand_measure)})', '']
    if hand_measure:
        lines.extend((f'- {_hand_measure_line(item, unit)}' for item in hand_measure))
    else:
        lines.append('- Elle olculecek kalem yok; her olcu en az iki kesit bandinda desteklendi.')
    lines += ['', '## Ciktilar', '']
    if dxf is not None:
        insunits = 5 if dxf_unit == 'cm' else 4
        layer_note = ', '.join((f'{name} {count}' for name, count in sorted(dxf.entity_counts_by_layer.items()) if name.startswith('CCA_SURVEY_')))
        lines.append(f'- DXF `{Path(dxf.path).name}` ($INSUNITS {insunits}, $MEASUREMENT 1, audit hatasi {dxf.audit_errors}): {layer_note}')
    if sheets:
        lines.append(f'- Paftalar ({len(sheets)}): ' + ', '.join((f'`{name}`' for name in sheets)))
    lines += ['', '| Dosya | Boyut (B) | SHA-256 |', '|---|---|---|']
    lines.extend((f"| `{name}` | {entry['bytes']:,} | `{entry['sha256']}` |" for name, entry in inventory.items()))
    lines += ['', f'_{PACK_STAMP}. Olcum 3B goruntuleyicide kalibrasyon kapisina baglidir; bu paket cevrimdisi bir taslaktir._', '']
    return '\n'.join(lines)

def build_survey_pack(source_ply: str | Path, output_dir: str | Path, *, project_name: str, config: SurveyPackConfig | None=None, progress: ProgressCallback | None=None, origin: Mapping[str, Any] | None=None) -> SurveyPackResult:
    started = time.perf_counter()
    config = config or SurveyPackConfig()
    if not isinstance(config, SurveyPackConfig):
        raise SurveyPackError('config must be a SurveyPackConfig')
    source = Path(source_ply).expanduser().resolve()
    if source.suffix.lower() != '.ply' or not source.is_file():
        raise SurveyPackError(f'source PLY does not exist: {source}')
    output = Path(output_dir).expanduser().resolve()
    if output == source or output.is_file():
        raise SurveyPackError('output_dir must be a directory, not the source file')
    project = _project_name(project_name)
    if origin is not None and (not isinstance(origin, Mapping)):
        raise SurveyPackError('origin must be a mapping or None')
    created_utc = datetime.now(timezone.utc).isoformat(timespec='seconds')
    _notify(progress, 'hash', 0.02)
    sha_before = _sha256(source)
    _notify(progress, 'read', 0.06)
    points, rgb, property_names = _read_points(source)
    point_count = int(points.shape[0])
    working, source_orientation = normalize_source_points(points, config.source_axis, config.frame)
    working = _apply_pre_transform(working, config.pre_transform)
    del points
    _notify(progress, 'frame', 0.15)
    frame = estimate_survey_frame(working, config.frame)
    framed = apply_survey_frame(working, frame)
    del working
    _notify(progress, 'plan', 0.3)
    slices = build_slices(framed, rgb, config.plan)
    walls = vectorize_walls(framed, config.plan)
    facade = find_facade(walls)
    min_y = front_view_min_y(walls, facade, config.plan)
    cross = cross_walls(framed, walls, facade, config.plan, min_y=min_y)
    partitions = partitions_along_facade(cross, facade)
    structures = segment_structures(framed, cross, facade, config.plan)
    profile = facade_profile(framed, facade, config.plan)
    _notify(progress, 'products', 0.45)
    output.mkdir(parents=True, exist_ok=True)
    frame_dict = frame_to_dict(frame)
    orientation_dict = orientation_to_dict(source_orientation)
    source_to_zup = np.asarray(source_orientation.transform_zup_from_source, dtype=np.float64)
    project_transform = np.eye(4, dtype=np.float64) if config.pre_transform is None else np.asarray(config.pre_transform, dtype=np.float64)
    survey_transform = np.asarray(frame.transform_survey_from_zup, dtype=np.float64)
    orientation_dict['transform_survey_from_source'] = (survey_transform @ project_transform @ source_to_zup).tolist()
    frame_info = {**frame_dict, 'source_orientation': orientation_dict, 'project': project, 'source_name': ascii_project_name(source.name, 'scan.ply'), 'source_sha256': sha_before}
    _write_json(output / WALLS_NAME, {'schema': SCHEMA_WALLS, 'stamp': PACK_STAMP, 'facade': facade, 'front_view_min_y_m': min_y, 'cross_walls': cross, 'walls': walls})
    _write_json(output / PARTITIONS_NAME, {**partitions, 'stamp': PACK_STAMP})
    _write_json(output / STRUCTURES_NAME, {'schema': SCHEMA_STRUCTURES, 'stamp': PACK_STAMP, 'structures': structures})
    _write_json(output / PROFILE_NAME, {**profile, 'stamp': PACK_STAMP})
    sheet_paths: list[Path] = []
    if config.sheets:
        _notify(progress, 'sheets', 0.5)
        style = SheetStyle(label_unit=config.label_unit, dpi=config.sheet_dpi)
        sheets_dir = output / SHEETS_DIRNAME
        sheet_paths.append(make_plan_sheet(slices, walls, structures, partitions, frame_info, sheets_dir / 'plan.png', style))
        sheet_paths.append(make_elevation_sheet(framed, rgb, walls, structures, profile, frame_info, sheets_dir / 'elevation.png', style, min_y=min_y))
        sheet_paths.append(make_box_sheet(framed, structures, sheets_dir / 'boxes.png', style, frame_info=frame_info, min_y=min_y))
        sheet_paths.append(make_section_sheet(framed, structures, sheets_dir / 'sections.png', style, frame_info=frame_info, min_y=min_y))
        sheet_paths.extend(make_slice_sheets(slices, sheets_dir / 'slices', style, frame_info=frame_info))
    dxf_result: SurveyDxfResult | None = None
    dxf_path: Path | None = None
    if config.dxf:
        _notify(progress, 'dxf', 0.75)
        dxf_path = output / f'survey_{config.dxf_unit}.dxf'
        dxf_result = export_survey_dxf(slices, walls, structures, partitions, profile, dxf_path, unit=config.dxf_unit, project_name=project)
    hand_measure = _hand_measure_items(walls, partitions, structures)
    confidence_counts = {'walls': _count_by_confidence(walls), 'cross_walls': _count_by_confidence(cross), 'structures': _count_by_confidence(structures), 'partitions': _partition_confidence_counts(partitions)}
    sheet_names = [path.relative_to(output).as_posix() for path in sheet_paths]
    report_path: Path | None = None
    manifest_path = output / MANIFEST_NAME
    if config.report:
        _notify(progress, 'report', 0.85)
        report_path = output / REPORT_NAME
        inventory_before_report = _inventory(output, skip=[manifest_path, report_path])
        _write_text(report_path, _report_text(project=project, source=source, sha_before=sha_before, sha_after=_sha256(source), point_count=point_count, has_rgb=rgb is not None, created_utc=created_utc, elapsed_s=time.perf_counter() - started, frame=frame, frame_config=config.frame, source_orientation=source_orientation, walls=walls, cross=cross, partitions=partitions, structures=structures, facade=facade, min_y=min_y, hand_measure=hand_measure, unit=config.label_unit, dxf=dxf_result, dxf_unit=config.dxf_unit, sheets=sheet_names, inventory=inventory_before_report, origin=origin))
    _notify(progress, 'manifest', 0.92)
    inventory = _inventory(output, skip=[manifest_path])
    sha_after = _sha256(source)
    if sha_after != sha_before:
        raise SurveyPackError(f'source PLY changed while the pack was being built: {source}')
    elapsed = time.perf_counter() - started
    manifest = {'schema': SCHEMA_PACK, 'stamp': PACK_STAMP, 'project': project, 'created_utc': created_utc, 'elapsed_s': round(elapsed, 3), 'source': {'path': str(source), 'name': source.name, 'sha256_before': sha_before, 'sha256_after': sha_after, 'point_count': point_count, 'has_rgb': rgb is not None, 'vertex_properties': property_names}, 'origin': dict(origin) if origin is not None else None, 'units': {'internal': 'm', 'label': config.label_unit, 'dxf': config.dxf_unit, 'cad_up_axis': 'z'}, 'pre_transform': None if config.pre_transform is None else [list(row) for row in config.pre_transform], 'source_orientation': orientation_dict, 'frame': frame_dict, 'config': asdict(config), 'products': {'facade': facade, 'front_view_min_y_m': min_y, 'walls': len(walls), 'cross_walls': len(cross), 'partitions': len(partitions.get('x_positions_m', [])), 'structures': len(structures), 'facade_profile_bins': len(profile.get('x_m', [])), 'slice_bands': sorted(slices['bands']), 'files': {'walls': WALLS_NAME, 'partitions': PARTITIONS_NAME, 'structures': STRUCTURES_NAME, 'facade_profile': PROFILE_NAME, 'report': REPORT_NAME if report_path is not None else None}}, 'confidence_counts': confidence_counts, 'low_confidence': len(hand_measure), 'hand_measure': hand_measure, 'sheets': sheet_names, 'dxf': None if dxf_result is None else {'path': Path(dxf_result.path).relative_to(output).as_posix(), 'unit': dxf_result.unit, 'insunits': 5 if dxf_result.unit == 'cm' else 4, 'entity_counts_by_layer': dxf_result.entity_counts_by_layer, 'audit_errors': dxf_result.audit_errors}, 'inventory': inventory}
    _write_json(manifest_path, manifest)
    _notify(progress, 'done', 1.0)
    return SurveyPackResult(output_dir=output, manifest_path=manifest_path, source_path=source, source_sha256=sha_before, project_name=project, frame=frame, point_count=point_count, wall_count=len(walls), cross_wall_count=len(cross), structure_count=len(structures), partition_count=len(partitions.get('x_positions_m', [])), low_confidence=len(hand_measure), confidence_counts=confidence_counts, hand_measure=tuple(hand_measure), sheet_paths=tuple(sheet_paths), dxf_path=dxf_path, report_path=report_path, inventory=inventory, elapsed_s=elapsed)
_CLI_ERRORS = (SurveyPackError, SurveyFrameError, SurveyPlanError, SurveySheetError, SurveyDxfError, E57ConversionError, PlyFormatError, OSError)

def _print_progress(stage: str, fraction: float) -> None:
    print(f'[{fraction * 100.0:3.0f}%] {stage}', flush=True)

def main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m export.survey_pack', description='Build a SURVEY DRAFT / UNCALIBRATED pack (sheets, DXF, report, manifest) from one scan.')
    parser.add_argument('source', help='input scan: metre .ply with automatic ground/up-axis detection, or .e57 converted next to the outputs first')
    parser.add_argument('--out', required=True, help='output directory (created when missing)')
    parser.add_argument('--project', default=None, help='project name for title blocks (folded to ASCII); default: source stem')
    parser.add_argument('--front', choices=('auto', 'keep', 'flip'), default='auto', help='front side decision')
    parser.add_argument('--source-axis', choices=_SOURCE_AXES, default='auto', help='source up-axis provenance; auto probes ground support')
    parser.add_argument('--label-unit', choices=_UNITS, default='cm', help='dimension label unit on sheets and report')
    parser.add_argument('--dxf-unit', choices=_UNITS, default='cm', help='DXF drawing unit')
    parser.add_argument('--no-sheets', action='store_true', help='skip the PNG sheets')
    parser.add_argument('--no-dxf', action='store_true', help='skip the DXF')
    parser.add_argument('--no-report', action='store_true', help='skip REPORT.md')
    parser.add_argument('--sheet-dpi', type=int, default=150, help='PNG sheet resolution')
    parser.add_argument('--y-up', action='store_true', help='E57 only: the file is Y-up; apply (x, y, z) -> (x, -z, y)')
    parser.add_argument('--quiet', action='store_true', help='no progress lines')
    args = parser.parse_args(argv)
    source = Path(args.source).expanduser()
    output = Path(args.out).expanduser()
    project = ascii_project_name(args.project if args.project else source.stem)
    progress = None if args.quiet else _print_progress
    try:
        config = SurveyPackConfig(frame=SurveyFrameConfig(front_side=args.front), label_unit=args.label_unit, dxf_unit=args.dxf_unit, sheets=not args.no_sheets, dxf=not args.no_dxf, report=not args.no_report, sheet_dpi=args.sheet_dpi, source_axis='CAD_Z_up' if source.suffix.lower() == '.e57' else args.source_axis)
        origin: dict[str, Any] | None = None
        if source.suffix.lower() == '.e57':
            output.mkdir(parents=True, exist_ok=True)
            ply_path = output / f'{source.stem}.ply'
            if not args.quiet:
                print(f'[  0%] e57 -> ply {ply_path}', flush=True)
            conversion = convert_e57_to_ply(source, ply_path, y_up=args.y_up)
            origin = {'kind': 'e57', 'path': str(conversion.source_path), 'sha256': conversion.source_sha256, 'point_count': conversion.point_count, 'scan_count': conversion.scan_count, 'has_color': conversion.has_color, 'has_intensity': conversion.has_intensity, 'reader': conversion.reader, 'y_up': bool(args.y_up), 'ply_sha256': conversion.destination_sha256}
            source = conversion.destination_path
        result = build_survey_pack(source, output, project_name=project, config=config, progress=progress, origin=origin)
    except _CLI_ERRORS as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    print(f'SURVEY PACK READY | {result.output_dir} | {result.wall_count} walls | {result.structure_count} structures | {result.low_confidence} {HAND_MEASURE_LABEL} | {result.elapsed_s:.1f} s | {PACK_STAMP}')
    return 0
__all__ = ['MANIFEST_NAME', 'PACK_STAMP', 'REPORT_NAME', 'SCHEMA_PACK', 'SHEETS_DIRNAME', 'ProgressCallback', 'SurveyPackConfig', 'SurveyPackError', 'SurveyPackResult', 'ascii_project_name', 'build_survey_pack', 'main']
if __name__ == '__main__':
    raise SystemExit(main())
