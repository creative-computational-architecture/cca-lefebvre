
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
FIXED_SCALE = 1.0

class PlanPlacementError(ValueError):
    pass

class PlacementPhase(str, Enum):
    IDLE = 'IDLE'
    ACTIVE = 'ACTIVE'
    FINISHED = 'FINISHED'
    CANCELLED = 'CANCELLED'

@dataclass(frozen=True, slots=True)
class PlacementAdvance:
    current_scan_id: str
    next_scan_id: str | None
    completed: bool
    restore_visibility: dict[str, bool] | None
    fixed_scale: float = FIXED_SCALE

@dataclass(frozen=True, slots=True)
class PlacementExit:
    phase: PlacementPhase
    cancelled: bool
    placed_ids: tuple[str, ...]
    restore_visibility: dict[str, bool]
    fixed_scale: float = FIXED_SCALE

def _scan_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PlanPlacementError('scan ids must be non-empty ASCII strings')
    try:
        value.encode('ascii')
    except UnicodeEncodeError as error:
        raise PlanPlacementError('scan ids must be non-empty ASCII strings') from error
    return value

def _fixed_scale(value: object) -> float:
    if type(value) is not float or value != FIXED_SCALE:
        raise PlanPlacementError('fixed scale must be 1.0')
    return FIXED_SCALE

def _ordered_ids(values: Sequence[str]) -> tuple[str, ...]:
    ordered = tuple((_scan_id(value) for value in values))
    if not ordered:
        raise PlanPlacementError('at least one scan is required')
    if len(set(ordered)) != len(ordered):
        raise PlanPlacementError('ordered scan ids must be unique')
    return ordered

def _visibility(values: Mapping[str, bool], ordered: tuple[str, ...]) -> dict[str, bool]:
    saved: dict[str, bool] = {}
    for raw_scan_id, visible in values.items():
        scan_id = _scan_id(raw_scan_id)
        if type(visible) is not bool:
            raise PlanPlacementError(f'{scan_id} visibility must be boolean')
        saved[scan_id] = visible
    missing = [scan_id for scan_id in ordered if scan_id not in saved]
    if missing:
        raise PlanPlacementError(f'missing visibility for scan: {missing[0]}')
    return saved

class PlanPlacementSession:

    def __init__(self) -> None:
        self._phase = PlacementPhase.IDLE
        self._ordered_scan_ids: tuple[str, ...] = ()
        self._active_index: int | None = None
        self._placed_ids: list[str] = []
        self._saved_original_visibility: dict[str, bool] = {}
        self._fixed_scale = FIXED_SCALE

    @property
    def phase(self) -> PlacementPhase:
        return self._phase

    @property
    def ordered_scan_ids(self) -> tuple[str, ...]:
        return self._ordered_scan_ids

    @property
    def active_index(self) -> int | None:
        return self._active_index

    @property
    def active_scan_id(self) -> str | None:
        if self._active_index is None:
            return None
        return self._ordered_scan_ids[self._active_index]

    @property
    def placed_ids(self) -> tuple[str, ...]:
        return tuple(self._placed_ids)

    @property
    def saved_original_visibility(self) -> dict[str, bool]:
        return dict(self._saved_original_visibility)

    @property
    def fixed_scale(self) -> float:
        return self._fixed_scale

    def start(self, ordered_scan_ids: Sequence[str], original_visibility: Mapping[str, bool], *, fixed_scale: float=FIXED_SCALE) -> str:
        if self._phase is not PlacementPhase.IDLE:
            raise PlanPlacementError('placement session already started')
        scale = _fixed_scale(fixed_scale)
        ordered = _ordered_ids(ordered_scan_ids)
        visibility = _visibility(original_visibility, ordered)
        self._ordered_scan_ids = ordered
        self._saved_original_visibility = visibility
        self._fixed_scale = scale
        self._placed_ids = []
        self._active_index = 0
        self._phase = PlacementPhase.ACTIVE
        return ordered[0]

    def _require_active(self) -> int:
        if self._phase is not PlacementPhase.ACTIVE or self._active_index is None:
            raise PlanPlacementError('placement session is not active')
        return self._active_index

    def previous(self) -> str:
        index = self._require_active()
        if index > 0:
            index -= 1
            self._active_index = index
        return self._ordered_scan_ids[index]

    def place_and_next(self, *, fixed_scale: float=FIXED_SCALE) -> PlacementAdvance:
        index = self._require_active()
        _fixed_scale(fixed_scale)
        current = self._ordered_scan_ids[index]
        if current not in self._placed_ids:
            self._placed_ids.append(current)
        next_index = index + 1
        if next_index < len(self._ordered_scan_ids):
            self._active_index = next_index
            return PlacementAdvance(current_scan_id=current, next_scan_id=self._ordered_scan_ids[next_index], completed=False, restore_visibility=None)
        self._active_index = None
        self._phase = PlacementPhase.FINISHED
        return PlacementAdvance(current_scan_id=current, next_scan_id=None, completed=True, restore_visibility=self.saved_original_visibility)

    def finish(self) -> PlacementExit:
        self._require_active()
        self._active_index = None
        self._phase = PlacementPhase.FINISHED
        return self._exit(cancelled=False)

    def cancel(self) -> PlacementExit:
        self._require_active()
        self._active_index = None
        self._phase = PlacementPhase.CANCELLED
        return self._exit(cancelled=True)

    def _exit(self, *, cancelled: bool) -> PlacementExit:
        return PlacementExit(phase=self._phase, cancelled=cancelled, placed_ids=self.placed_ids, restore_visibility=self.saved_original_visibility)
__all__ = ['FIXED_SCALE', 'PlacementAdvance', 'PlacementExit', 'PlacementPhase', 'PlanPlacementError', 'PlanPlacementSession']
