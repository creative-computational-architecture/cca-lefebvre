
from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any
Vec3 = tuple[float, float, float]
WORLD_Z: Vec3 = (0.0, 0.0, 1.0)
ORBIT_MODE = 'orbit'
TOP_MODE = 'top'
WALK_MODE = 'walk'
CAMERA_MODES = frozenset({ORBIT_MODE, TOP_MODE, WALK_MODE})
PROJECTION_MODES = frozenset({'perspective', 'orthographic'})
MIN_PITCH_RADIANS = -math.pi / 2.0
MAX_PITCH_RADIANS = math.pi / 2.0

class CameraStateError(ValueError):
    pass

def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CameraStateError(f'{label} must be a finite number')
    converted = float(value)
    if not math.isfinite(converted):
        raise CameraStateError(f'{label} must be a finite number')
    return converted

def _vec3(value: object, label: str) -> Vec3:
    if isinstance(value, (str, bytes)):
        raise CameraStateError(f'{label} must contain three finite numbers')
    try:
        items = tuple(value)
    except TypeError as exc:
        raise CameraStateError(f'{label} must contain three finite numbers') from exc
    if len(items) != 3:
        raise CameraStateError(f'{label} must contain three finite numbers')
    return tuple((_finite(item, f'{label}[{index}]') for index, item in enumerate(items)))

def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def _subtract(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def _scale(value: Vec3, scalar: float) -> Vec3:
    return (value[0] * scalar, value[1] * scalar, value[2] * scalar)

def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

def _length(value: Vec3) -> float:
    return math.sqrt(_dot(value, value))

def _normalized(value: Vec3, label: str) -> Vec3:
    length = _length(value)
    if length <= 1e-12:
        raise CameraStateError(f'{label} has zero length')
    return _scale(value, 1.0 / length)

def _rotate_about_axis(value: Vec3, axis: Vec3, angle_radians: float) -> Vec3:
    unit = _normalized(axis, 'rotation axis')
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    parallel = _scale(unit, _dot(unit, value))
    perpendicular = _subtract(value, parallel)
    crossed = _cross(unit, value)
    return _add(_add(parallel, _scale(perpendicular, cosine)), _scale(crossed, sine))

def _wrap_radians(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi

@dataclass(frozen=True, slots=True)
class CameraState:
    pivot: Vec3 = (0.0, 0.0, 0.0)
    yaw_radians: float = math.radians(-45.0)
    pitch_radians: float = math.radians(35.0)
    distance: float = 120.0
    projection: str = 'perspective'
    mode: str = ORBIT_MODE

    def __post_init__(self) -> None:
        pivot = _vec3(self.pivot, 'pivot')
        yaw = _finite(self.yaw_radians, 'yaw_radians')
        pitch = _finite(self.pitch_radians, 'pitch_radians')
        distance = _finite(self.distance, 'distance')
        if not MIN_PITCH_RADIANS <= pitch <= MAX_PITCH_RADIANS:
            raise CameraStateError('pitch_radians must remain inside the roll-free range')
        if distance <= 0.0:
            raise CameraStateError('distance must be positive')
        if self.projection not in PROJECTION_MODES:
            raise CameraStateError('projection must be perspective or orthographic')
        if self.mode not in CAMERA_MODES:
            raise CameraStateError('mode must be orbit, top, or walk')
        object.__setattr__(self, 'pivot', pivot)
        object.__setattr__(self, 'yaw_radians', _wrap_radians(yaw))
        object.__setattr__(self, 'pitch_radians', pitch)
        object.__setattr__(self, 'distance', distance)

    def to_dict(self) -> dict[str, Any]:
        return {'pivot': list(self.pivot), 'yaw_radians': self.yaw_radians, 'pitch_radians': self.pitch_radians, 'distance': self.distance, 'projection': self.projection, 'mode': self.mode}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CameraState:
        if not isinstance(payload, Mapping):
            raise CameraStateError('camera payload must be an object')
        try:
            return cls(pivot=_vec3(payload['pivot'], 'pivot'), yaw_radians=_finite(payload['yaw_radians'], 'yaw_radians'), pitch_radians=_finite(payload['pitch_radians'], 'pitch_radians'), distance=_finite(payload['distance'], 'distance'), projection=str(payload['projection']), mode=str(payload['mode']))
        except KeyError as exc:
            raise CameraStateError(f'camera payload is missing {exc.args[0]}') from exc

@dataclass(frozen=True, slots=True)
class CameraPose:
    eye: Vec3
    target: Vec3
    up: Vec3
    right: Vec3
    screen_up: Vec3

class CameraController:

    def __init__(self, state: CameraState | None=None, *, minimum_distance: float=0.05, maximum_distance: float=1000000.0) -> None:
        minimum = _finite(minimum_distance, 'minimum_distance')
        maximum = _finite(maximum_distance, 'maximum_distance')
        if minimum <= 0.0 or maximum < minimum:
            raise CameraStateError('camera distance limits are invalid')
        self._minimum_distance = minimum
        self._maximum_distance = maximum
        candidate = state or CameraState()
        self._state = replace(candidate, distance=min(max(candidate.distance, minimum), maximum))

    @property
    def state(self) -> CameraState:
        return self._state

    def restore(self, state: CameraState) -> CameraState:
        if not isinstance(state, CameraState):
            raise CameraStateError('state must be a CameraState')
        self._state = replace(state, distance=min(max(state.distance, self._minimum_distance), self._maximum_distance))
        return self._state

    def pose(self) -> CameraPose:
        state = self._state
        if state.mode == TOP_MODE:
            eye = _add(state.pivot, (0.0, 0.0, state.distance))
            up = (0.0, 1.0, 0.0)
        else:
            cosine = math.cos(state.pitch_radians)
            offset = (state.distance * cosine * math.cos(state.yaw_radians), state.distance * cosine * math.sin(state.yaw_radians), state.distance * math.sin(state.pitch_radians))
            eye = _add(state.pivot, offset)
            up = WORLD_Z
        forward = _normalized(_subtract(state.pivot, eye), 'camera forward')
        right = (1.0, 0.0, 0.0) if state.mode == TOP_MODE else (-math.sin(state.yaw_radians), math.cos(state.yaw_radians), 0.0)
        screen_up = _normalized(_cross(right, forward), 'camera up')
        return CameraPose(eye, state.pivot, up, right, screen_up)

    def set_projection(self, projection: str) -> CameraState:
        if projection not in PROJECTION_MODES:
            raise CameraStateError('projection must be perspective or orthographic')
        self._state = replace(self._state, projection=projection)
        return self._state

    def enter_orbit(self) -> CameraState:
        self._state = replace(self._state, mode=ORBIT_MODE)
        return self._state

    def enter_top(self) -> CameraState:
        self._state = replace(self._state, mode=TOP_MODE)
        return self._state

    def enter_walk(self) -> CameraState:
        self._state = replace(self._state, mode=WALK_MODE)
        return self._state

    def set_named_view(self, name: str) -> CameraState:
        normalized = name.strip().lower()
        aliases = {'perspective': 'isometric', 'persp': 'isometric', 'iso': 'isometric'}
        normalized = aliases.get(normalized, normalized)
        directions = {'front': (-math.pi / 2, 0.0), 'right': (0.0, 0.0), 'back': (math.pi / 2, 0.0), 'left': (math.pi, 0.0), 'bottom': (-math.pi / 2, -math.pi / 2), 'isometric': (-math.pi / 4, math.atan(1 / math.sqrt(2)))}
        if normalized == 'top':
            self._state = replace(self._state, mode=TOP_MODE, projection='orthographic')
        elif normalized in directions:
            yaw, pitch = directions[normalized]
            self._state = replace(self._state, mode=ORBIT_MODE, yaw_radians=yaw, pitch_radians=pitch, projection='perspective' if normalized == 'isometric' else 'orthographic')
        else:
            raise CameraStateError(f'unknown named camera view: {name}')
        return self._state

    def _orbit_origin(self) -> CameraState:
        if self._state.mode == TOP_MODE:
            return replace(self._state, mode=ORBIT_MODE, yaw_radians=-math.pi / 2, pitch_radians=math.pi / 2)
        return self._state

    def orbit(self, yaw_delta_radians: float, pitch_delta_radians: float) -> CameraState:
        yaw_delta = _finite(yaw_delta_radians, 'yaw_delta_radians')
        pitch_delta = _finite(pitch_delta_radians, 'pitch_delta_radians')
        if yaw_delta == 0.0 and pitch_delta == 0.0:
            return self._state
        before = self._orbit_origin()
        pitch = min(max(before.pitch_radians + pitch_delta, MIN_PITCH_RADIANS), MAX_PITCH_RADIANS)
        if abs(pitch) == math.pi / 2 and pitch == before.pitch_radians:
            yaw_delta = 0.0
        self._state = replace(before, yaw_radians=_wrap_radians(before.yaw_radians + yaw_delta), pitch_radians=pitch, mode=ORBIT_MODE)
        return self._state

    def pan(self, right_metres: float, up_metres: float) -> CameraState:
        right_amount = _finite(right_metres, 'right_metres')
        up_amount = _finite(up_metres, 'up_metres')
        pose = self.pose()
        translation = _add(_scale(pose.right, right_amount), _scale(pose.screen_up, up_amount))
        self._state = replace(self._state, pivot=_add(self._state.pivot, translation))
        return self._state

    def orbit_about(self, yaw_delta_radians: float, pitch_delta_radians: float, center: Sequence[float]) -> CameraState:
        yaw_delta = _finite(yaw_delta_radians, 'yaw_delta_radians')
        pitch_delta = _finite(pitch_delta_radians, 'pitch_delta_radians')
        center_value = _vec3(center, 'center')
        if yaw_delta == 0.0 and pitch_delta == 0.0:
            return self._state
        before = self._orbit_origin()
        pitch = min(max(before.pitch_radians + pitch_delta, MIN_PITCH_RADIANS), MAX_PITCH_RADIANS)
        applied_pitch = pitch - before.pitch_radians
        if abs(pitch) == math.pi / 2 and applied_pitch == 0.0:
            yaw_delta = 0.0
        right = self.pose().right
        relative = _subtract(before.pivot, center_value)
        relative = _rotate_about_axis(relative, right, -applied_pitch)
        relative = _rotate_about_axis(relative, WORLD_Z, yaw_delta)
        self._state = replace(before, pivot=_add(center_value, relative), yaw_radians=_wrap_radians(before.yaw_radians + yaw_delta), pitch_radians=pitch, mode=ORBIT_MODE)
        return self._state

    def zoom_towards(self, wheel_steps: float, anchor: Sequence[float], *, factor: float=1.05) -> CameraState:
        anchor_value = _vec3(anchor, 'anchor')
        before = self._state.distance
        self.zoom(wheel_steps, factor=factor)
        ratio = self._state.distance / before if before > 0.0 else 1.0
        relative = _subtract(self._state.pivot, anchor_value)
        self._state = replace(self._state, pivot=_add(anchor_value, _scale(relative, ratio)))
        return self._state

    def zoom(self, wheel_steps: float, *, factor: float=1.05) -> CameraState:
        steps = _finite(wheel_steps, 'wheel_steps')
        zoom_factor = _finite(factor, 'factor')
        if zoom_factor <= 1.0:
            raise CameraStateError('factor must be greater than one')
        try:
            distance = self._state.distance * math.pow(zoom_factor, -steps)
        except OverflowError:
            distance = 0.0 if steps > 0.0 else math.inf
        distance = min(max(distance, self._minimum_distance), self._maximum_distance)
        self._state = replace(self._state, distance=distance)
        return self._state

    def frame_bounds(self, minimum: Sequence[float], maximum: Sequence[float], *, vertical_fov_degrees: float=60.0, aspect_ratio: float=1.0, padding: float=1.1) -> CameraState:
        lower = _vec3(minimum, 'minimum')
        upper = _vec3(maximum, 'maximum')
        if any((upper[index] < lower[index] for index in range(3))):
            raise CameraStateError('maximum bounds must not be below minimum bounds')
        fov = _finite(vertical_fov_degrees, 'vertical_fov_degrees')
        fit_padding = _finite(padding, 'padding')
        aspect = _finite(aspect_ratio, 'aspect_ratio')
        if not 1.0 < fov < 179.0:
            raise CameraStateError('vertical_fov_degrees must be between 1 and 179')
        if fit_padding < 1.0:
            raise CameraStateError('padding must be at least one')
        if aspect <= 0.0:
            raise CameraStateError('aspect_ratio must be positive')
        pivot = _scale(_add(lower, upper), 0.5)
        pose = self.pose()
        forward = _normalized(_subtract(pose.target, pose.eye), 'camera forward')
        offsets = [_subtract((x, y, z), pivot) for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])]
        projected = [(_dot(value, pose.right), _dot(value, pose.screen_up), _dot(value, forward)) for value in offsets]
        if self._state.projection == 'orthographic':
            half_height = max((max(abs(x) / aspect, abs(y)) for x, y, _z in projected))
            distance = max(2 * half_height * fit_padding, max((-z for _x, _y, z in projected)) * fit_padding)
        else:
            tangent = math.tan(math.radians(fov) * 0.5)
            distance = max((max(abs(x) / (tangent * aspect), abs(y) / tangent) * fit_padding - z for x, y, z in projected))
        distance = min(max(distance, self._minimum_distance), self._maximum_distance)
        self._state = replace(self._state, pivot=pivot, distance=distance)
        return self._state

    def sync_look_at(self, eye: Sequence[float], target: Sequence[float], *, mode: str | None=None) -> CameraState:
        eye_value = _vec3(eye, 'eye')
        target_value = _vec3(target, 'target')
        offset = _subtract(eye_value, target_value)
        raw_distance = _length(offset)
        if raw_distance <= 1e-12:
            raise CameraStateError('eye and target must be different')
        distance = min(max(raw_distance, self._minimum_distance), self._maximum_distance)
        normalized = _scale(offset, 1.0 / raw_distance)
        horizontal = math.hypot(normalized[0], normalized[1])
        yaw = self._state.yaw_radians if horizontal <= 1e-12 else math.atan2(normalized[1], normalized[0])
        pitch = math.asin(min(max(normalized[2], -1.0), 1.0))
        pitch = min(max(pitch, MIN_PITCH_RADIANS), MAX_PITCH_RADIANS)
        next_mode = self._state.mode if mode is None else mode
        if next_mode not in CAMERA_MODES:
            raise CameraStateError('mode must be orbit, top, or walk')
        self._state = CameraState(pivot=target_value, yaw_radians=yaw, pitch_radians=pitch, distance=distance, projection=self._state.projection, mode=next_mode)
        return self._state
