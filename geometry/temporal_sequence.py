
from __future__ import annotations
import copy
import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo
RUN_ID_FORMAT = '%Y_%m_%d_%H_%M_%S'
RUN_ID_PATTERN = re.compile('(?P<stamp>\\d{4}_\\d{2}_\\d{2}_\\d{2}_\\d{2}_\\d{2})')
SCHEMA = 'cca.temporal-tail-head-sequence.v1'
LOCAL_TIMEZONE = ZoneInfo('Europe/Istanbul')

class TemporalSequenceError(ValueError):
    pass

@dataclass(frozen=True, slots=True)
class RunTiming:
    scan_id: str
    capture_start: datetime
    capture_end: datetime
    duration_s: float
    first_frame_index: int | None
    last_frame_index: int | None
    first_sensor_time_s: float | None
    last_sensor_time_s: float | None
    frame_count: int
    raw_path: Path | None
    timing_source: str
    first_frame_sha256: str | None
    last_frame_sha256: str | None
    frame_inventory_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {'scan_id': self.scan_id, 'capture_start': self.capture_start.isoformat(timespec='seconds'), 'capture_end': self.capture_end.isoformat(timespec='milliseconds'), 'duration_s': self.duration_s, 'first_frame_index': self.first_frame_index, 'last_frame_index': self.last_frame_index, 'first_sensor_time_s': self.first_sensor_time_s, 'last_sensor_time_s': self.last_sensor_time_s, 'frame_count': self.frame_count, 'raw_path': str(self.raw_path) if self.raw_path is not None else None, 'timing_source': self.timing_source, 'first_frame_sha256': self.first_frame_sha256, 'last_frame_sha256': self.last_frame_sha256, 'frame_inventory_sha256': self.frame_inventory_sha256}

@dataclass(frozen=True, slots=True)
class TemporalPair:
    pair_index: int
    source_scan_id: str
    target_scan_id: str
    gap_s: float
    gap_source: str
    gap_median_s: float
    gap_mad_s: float
    long_gap_cutoff_s: float
    long_gap: bool
    temporal_prior: float
    endpoint_policy: str = 'SOURCE_TAIL_TO_TARGET_HEAD'

    @property
    def pair_id(self) -> str:
        return f'T{self.pair_index:02d}_{self.source_scan_id}__{self.target_scan_id}'

    def to_dict(self) -> dict[str, Any]:
        return {'pair_id': self.pair_id, 'pair_index': self.pair_index, 'source_scan_id': self.source_scan_id, 'target_scan_id': self.target_scan_id, 'gap_s': self.gap_s, 'gap_source': self.gap_source, 'gap_median_s': self.gap_median_s, 'gap_mad_s': self.gap_mad_s, 'long_gap_cutoff_s': self.long_gap_cutoff_s, 'long_gap': self.long_gap, 'temporal_prior': self.temporal_prior, 'endpoint_policy': self.endpoint_policy, 'geometry_required': True, 'time_alone_accepts': False, 'time_alone_rejects': False}

@dataclass(frozen=True, slots=True)
class TemporalSequence:
    runs: tuple[RunTiming, ...]
    pairs: tuple[TemporalPair, ...]
    evidence_sha256: str

    @property
    def ordered_scan_ids(self) -> tuple[str, ...]:
        return tuple((item.scan_id for item in self.runs))

    def pair(self, first_scan_id: str, second_scan_id: str) -> TemporalPair | None:
        return next((item for item in self.pairs if (item.source_scan_id, item.target_scan_id) == (first_scan_id, second_scan_id)), None)

    def adjacent_pair(self, first_scan_id: str, second_scan_id: str) -> TemporalPair | None:
        direct = self.pair(first_scan_id, second_scan_id)
        if direct is not None:
            return direct
        return self.pair(second_scan_id, first_scan_id)

    def predecessor(self, scan_id: str) -> TemporalPair | None:
        return next((item for item in self.pairs if item.target_scan_id == scan_id), None)

    def successor(self, scan_id: str) -> TemporalPair | None:
        return next((item for item in self.pairs if item.source_scan_id == scan_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {'schema': SCHEMA, 'policy': {'candidate_topology': 'CHRONOLOGICAL_NEIGHBOURS_PRIMARY', 'endpoint_comparison': 'SOURCE_TAIL_TO_TARGET_HEAD', 'gap_model': 'DATA_DERIVED_MEDIAN_MAD_PRIOR', 'time_alone_accepts': False, 'time_alone_rejects': False, 'raw_points_preserved': True, 'distance_cutoff_m': None, 'fixed_scale': 1.0}, 'ordered_scan_ids': list(self.ordered_scan_ids), 'runs': [item.to_dict() for item in self.runs], 'adjacent_pairs': [item.to_dict() for item in self.pairs], 'evidence_sha256': self.evidence_sha256}

def parse_run_timestamp(scan_id: str) -> datetime | None:
    match = RUN_ID_PATTERN.search(str(scan_id))
    if match is None:
        return None
    try:
        return datetime.strptime(match.group('stamp'), RUN_ID_FORMAT).replace(tzinfo=LOCAL_TIMEZONE)
    except ValueError:
        return None

def _frame_index(path: Path) -> int:
    match = re.search('(\\d+)$', path.stem)
    if match is None:
        raise TemporalSequenceError(f'frame index missing: {path.name}')
    return int(match.group(1))

def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    output = float(value)
    return output if math.isfinite(output) else None

def _read_frame_clock(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as error:
        raise TemporalSequenceError(f'cannot read frame metadata: {path}') from error
    if not isinstance(payload, dict):
        raise TemporalSequenceError(f'frame metadata is not an object: {path}')
    return _finite_number(payload.get('time'))

def _frame_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise TemporalSequenceError(f'cannot hash frame metadata: {path}') from error

def _frame_inventory_sha256(frames: Sequence[Path]) -> str:
    try:
        inventory = [{'name': frame.name, 'size_bytes': frame.stat().st_size} for frame in frames]
    except OSError as error:
        raise TemporalSequenceError('cannot read frame inventory') from error
    encoded = json.dumps(inventory, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()

def inspect_run_timing(raw_root: str | Path, scan_id: str) -> RunTiming:
    capture_start = parse_run_timestamp(scan_id)
    if capture_start is None:
        raise TemporalSequenceError(f'scan id has no capture timestamp: {scan_id}')
    run_dir = Path(raw_root).resolve() / scan_id
    if not run_dir.is_dir():
        return RunTiming(scan_id=scan_id, capture_start=capture_start, capture_end=capture_start, duration_s=0.0, first_frame_index=None, last_frame_index=None, first_sensor_time_s=None, last_sensor_time_s=None, frame_count=0, raw_path=None, timing_source='RUN_ID_ONLY', first_frame_sha256=None, last_frame_sha256=None, frame_inventory_sha256=None)
    frames = sorted(run_dir.glob('frame_*.json'), key=lambda path: (_frame_index(path), path.name))
    if not frames:
        return RunTiming(scan_id=scan_id, capture_start=capture_start, capture_end=capture_start, duration_s=0.0, first_frame_index=None, last_frame_index=None, first_sensor_time_s=None, last_sensor_time_s=None, frame_count=0, raw_path=run_dir, timing_source='RUN_ID_ONLY', first_frame_sha256=None, last_frame_sha256=None, frame_inventory_sha256=None)
    first_clock = _read_frame_clock(frames[0])
    last_clock = _read_frame_clock(frames[-1])
    duration = 0.0
    timing_source = 'RUN_ID_PLUS_FRAME_CLOCK'
    if first_clock is not None and last_clock is not None and (last_clock >= first_clock):
        duration = last_clock - first_clock
    else:
        timing_source = 'RUN_ID_ONLY'
    return RunTiming(scan_id=scan_id, capture_start=capture_start, capture_end=capture_start + timedelta(seconds=duration), duration_s=duration, first_frame_index=_frame_index(frames[0]), last_frame_index=_frame_index(frames[-1]), first_sensor_time_s=first_clock, last_sensor_time_s=last_clock, frame_count=len(frames), raw_path=run_dir, timing_source=timing_source, first_frame_sha256=_frame_sha256(frames[0]), last_frame_sha256=_frame_sha256(frames[-1]), frame_inventory_sha256=_frame_inventory_sha256(frames))

def _pair_gap(source: RunTiming, target: RunTiming) -> tuple[float, str]:
    if source.last_sensor_time_s is not None and target.first_sensor_time_s is not None and (target.first_sensor_time_s >= source.last_sensor_time_s):
        return (target.first_sensor_time_s - source.last_sensor_time_s, 'FRAME_CLOCK')
    return (max(0.0, (target.capture_start - source.capture_end).total_seconds()), 'RUN_ID_CLOCK')

def _robust_gap_model(gaps: Sequence[float]) -> tuple[float, float, float]:
    if not gaps:
        return (0.0, 0.0, math.inf)
    center = float(median(gaps))
    mad = float(median((abs(value - center) for value in gaps)))
    spread = 1.4826 * mad
    if spread <= 1e-09:
        positive = [value for value in gaps if value > center]
        cutoff = min(positive) if positive else max(gaps)
    else:
        cutoff = center + 3.0 * spread
    return (center, mad, cutoff)

def temporal_sequence_evidence_sha256(runs: Sequence[RunTiming], pairs: Sequence[TemporalPair]) -> str:
    payload = {'runs': [{'scan_id': run.scan_id, 'first_frame_index': run.first_frame_index, 'last_frame_index': run.last_frame_index, 'frame_count': run.frame_count, 'first_sensor_time_s': run.first_sensor_time_s, 'last_sensor_time_s': run.last_sensor_time_s, 'first_frame_sha256': run.first_frame_sha256, 'last_frame_sha256': run.last_frame_sha256, 'frame_inventory_sha256': run.frame_inventory_sha256} for run in runs], 'pairs': [pair.to_dict() for pair in pairs]}
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()

def build_temporal_sequence(raw_root: str | Path, scan_ids: Iterable[str]) -> TemporalSequence:
    supplied = [str(item) for item in scan_ids]
    identifiers = list(dict.fromkeys(supplied))
    if len(identifiers) != len(supplied):
        raise TemporalSequenceError('scan ids must be unique')
    parsed: list[RunTiming] = []
    missing: list[str] = []
    for scan_id in identifiers:
        if parse_run_timestamp(scan_id) is None:
            missing.append(scan_id)
            continue
        parsed.append(inspect_run_timing(raw_root, scan_id))
    if missing:
        raise TemporalSequenceError('scan ids have no capture timestamp: ' + ', '.join(missing))
    parsed.sort(key=lambda item: (item.capture_start, item.scan_id))
    raw_gaps = [_pair_gap(source, target)[0] for source, target in pairwise(parsed)]
    center, mad, cutoff = _robust_gap_model(raw_gaps)
    scale = max(center, 1.0)
    pairs: list[TemporalPair] = []
    for index, (source, target) in enumerate(pairwise(parsed), 1):
        gap, gap_source = _pair_gap(source, target)
        prior = 1.0 / (1.0 + max(0.0, gap - center) / scale)
        pairs.append(TemporalPair(pair_index=index, source_scan_id=source.scan_id, target_scan_id=target.scan_id, gap_s=gap, gap_source=gap_source, gap_median_s=center, gap_mad_s=mad, long_gap_cutoff_s=cutoff, long_gap=bool(gap > center and gap >= cutoff), temporal_prior=prior))
    runs = tuple(parsed)
    temporal_pairs = tuple(pairs)
    return TemporalSequence(runs, temporal_pairs, temporal_sequence_evidence_sha256(runs, temporal_pairs))

def apply_temporal_order(project: Any, raw_root: str | Path) -> TemporalSequence | None:
    scans = list(getattr(project, 'scans', ()))
    if len(scans) < 2:
        return None
    if any((parse_run_timestamp(getattr(scan, 'scan_id', '')) is None for scan in scans)):
        return None
    sequence = build_temporal_sequence(raw_root, (str(scan.scan_id) for scan in scans))
    transforms_before = {str(scan.scan_id): copy.deepcopy(getattr(scan, 'transform', None)) for scan in scans}
    by_id = {str(scan.scan_id): scan for scan in scans}
    project.scans = [by_id[scan_id] for scan_id in sequence.ordered_scan_ids]
    for index, scan in enumerate(project.scans):
        scan.rail_order = index
        scan.metadata = dict(getattr(scan, 'metadata', {}))
        timing = sequence.runs[index]
        scan.metadata['capture_timing'] = timing.to_dict()
    sync_groups = getattr(project, '_sync_groups', None)
    if callable(sync_groups):
        sync_groups()
    project.metadata = dict(getattr(project, 'metadata', {}))
    project.metadata['temporal_sequence'] = sequence.to_dict()
    transforms_after = {str(scan.scan_id): getattr(scan, 'transform', None) for scan in project.scans}
    if transforms_after != transforms_before:
        raise TemporalSequenceError('temporal ordering changed a scan transform')
    return sequence

def write_temporal_sequence(sequence: TemporalSequence, output_path: str | Path) -> Path:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.tmp')
    temporary.write_text(json.dumps(sequence.to_dict(), ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='ascii')
    temporary.replace(output)
    return output
__all__ = ['SCHEMA', 'RunTiming', 'TemporalPair', 'TemporalSequence', 'TemporalSequenceError', 'apply_temporal_order', 'build_temporal_sequence', 'inspect_run_timing', 'parse_run_timestamp', 'temporal_sequence_evidence_sha256', 'write_temporal_sequence']
