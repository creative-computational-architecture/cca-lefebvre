
from __future__ import annotations
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from viewer.edit_commands import EDIT_COMMANDS
_SETTINGS_ORGANIZATION = 'CCA'
_SETTINGS_APPLICATION = 'Lefebvre'
_SETTINGS_KEY = 'cad/input-v1'
_RIGHT_IDLE = frozenset({'repeat_or_settings', 'repeat', 'settings', 'menu'})
_RIGHT_ACTIVE = frozenset({'confirm', 'cancel', 'menu'})
_MIDDLE = frozenset({'pan', 'orbit'})
_COMMANDS = frozenset((item.name for item in EDIT_COMMANDS))
_TARGETS = _COMMANDS | {'SETTINGS'}
_ALIAS_RE = re.compile('^[A-Z][A-Z0-9_ ]*$')

def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f'{name} must be a boolean')
    return value

def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()

def _aliases(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError('aliases must be a mapping')
    result: dict[str, str] = {}
    for raw_key, raw_target in value.items():
        key = _text(raw_key, 'alias').upper()
        target = _text(raw_target, 'alias target').upper()
        if not _ALIAS_RE.fullmatch(key) or ':' in key or '=' in key:
            raise ValueError(f'invalid alias: {raw_key!r}')
        if target not in _TARGETS or any((char in target for char in ';:=(){}[]')):
            raise ValueError(f'invalid alias target: {raw_target!r}')
        if key in result:
            raise ValueError(f'duplicate normalized alias: {key}')
        result[key] = target
    return result

@dataclass(frozen=True)
class CadInputPreferences:
    selection_key: str = 'Space'
    confirm_key: str = 'Return'
    cancel_key: str = 'Esc'
    dynamic_input: bool = True
    crosshair_percent: int = 100
    pickbox_px: int = 8
    selection_tint: bool = True
    selection_box: bool = True
    middle_button: str = 'orbit'
    right_click_idle: str = 'repeat_or_settings'
    right_click_active: str = 'confirm'
    aliases: dict[str, str] = field(default_factory=lambda: {'M': 'MOVE', 'RO': 'ROTATE', 'R': 'ROTATE', 'G': 'MOVE', 'U': 'UNDO', 'OP': 'SETTINGS', 'ZE': 'EXTENTS', 'ZS': 'FIT SELECTED', 'OR': 'ORBIT'})

    def __post_init__(self) -> None:
        for name in ('selection_key', 'confirm_key', 'cancel_key'):
            _text(getattr(self, name), name)
        for name in ('dynamic_input', 'selection_tint', 'selection_box'):
            _bool(getattr(self, name), name)
        for name, low, high in (('crosshair_percent', 0, 100), ('pickbox_px', 2, 30)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in [{low}, {high}]')
        if self.middle_button not in _MIDDLE:
            raise ValueError('invalid middle_button')
        if self.right_click_idle not in _RIGHT_IDLE:
            raise ValueError('invalid right_click_idle')
        if self.right_click_active not in _RIGHT_ACTIVE:
            raise ValueError('invalid right_click_active')
        object.__setattr__(self, 'aliases', _aliases(self.aliases))

    def to_dict(self) -> dict[str, Any]:
        return {'selection_key': self.selection_key, 'confirm_key': self.confirm_key, 'cancel_key': self.cancel_key, 'dynamic_input': self.dynamic_input, 'crosshair_percent': self.crosshair_percent, 'pickbox_px': self.pickbox_px, 'selection_tint': self.selection_tint, 'selection_box': self.selection_box, 'middle_button': self.middle_button, 'right_click_idle': self.right_click_idle, 'right_click_active': self.right_click_active, 'aliases': dict(self.aliases)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CadInputPreferences:
        if not isinstance(value, Mapping):
            raise TypeError('preferences must be a mapping')
        defaults = cls().to_dict()
        if set(value) != set(defaults):
            raise ValueError('preferences fields do not match schema')
        return cls(**dict(value))

def _settings(qt: Any, settings: Any | None) -> Any:
    if settings is not None:
        return settings
    core = getattr(qt, 'QtCore', qt)
    return core.QSettings(_SETTINGS_ORGANIZATION, _SETTINGS_APPLICATION)

def load_preferences(qt: Any, settings: Any | None=None) -> CadInputPreferences:
    raw = _settings(qt, settings).value(_SETTINGS_KEY, None)
    if raw is None:
        return CadInputPreferences()
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        return CadInputPreferences.from_dict(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return CadInputPreferences()

def save_preferences(qt: Any, prefs: CadInputPreferences, settings: Any | None=None) -> None:
    validated = CadInputPreferences.from_dict(prefs.to_dict())
    _settings(qt, settings).setValue(_SETTINGS_KEY, json.dumps(validated.to_dict(), sort_keys=True))

def _qt_parts(qt: Any) -> tuple[Any, Any]:
    widgets = getattr(qt, 'QtWidgets', None) or qt
    gui = getattr(qt, 'QtGui', None)
    if gui is None:
        gui = __import__('PySide6.QtGui', fromlist=['QtGui'])
    return (widgets, gui)

def create_preferences_widget(qt: Any, preferences: CadInputPreferences, on_change: Callable[[CadInputPreferences], None]) -> Any:
    widgets, gui = _qt_parts(qt)
    current = CadInputPreferences.from_dict(preferences.to_dict())
    root = widgets.QWidget()
    root.setObjectName('CadInputPreferences')
    form = widgets.QFormLayout(root)
    controls: dict[str, Any] = {}

    def key_control(name: str, value: str) -> None:
        control = widgets.QKeySequenceEdit(gui.QKeySequence(value))
        control.setObjectName(name)
        controls[name] = control
        form.addRow(name.replace('_', ' ').title(), control)
    key_control('selection_key', current.selection_key)
    key_control('confirm_key', current.confirm_key)
    key_control('cancel_key', current.cancel_key)
    for name, label, value in (('dynamic_input', 'Dynamic input', current.dynamic_input), ('selection_tint', 'Selection tint', current.selection_tint), ('selection_box', 'Selection box', current.selection_box)):
        control = widgets.QCheckBox()
        control.setChecked(value)
        control.setObjectName(name)
        controls[name] = control
        form.addRow(label, control)
    for name, label, low, high, value in (('crosshair_percent', 'Crosshair %', 0, 100, current.crosshair_percent), ('pickbox_px', 'Pickbox px', 2, 30, current.pickbox_px)):
        control = widgets.QSpinBox()
        control.setRange(low, high)
        control.setValue(value)
        control.setObjectName(name)
        controls[name] = control
        form.addRow(label, control)
    for name, label, options, value in (('middle_button', 'Middle button', sorted(_MIDDLE), current.middle_button), ('right_click_idle', 'Right click idle', sorted(_RIGHT_IDLE), current.right_click_idle), ('right_click_active', 'Right click active', sorted(_RIGHT_ACTIVE), current.right_click_active)):
        control = widgets.QComboBox()
        control.addItems(options)
        control.setCurrentText(value)
        control.setObjectName(name)
        controls[name] = control
        form.addRow(label, control)
    aliases = widgets.QPlainTextEdit('\n'.join((f'{key}={target}' for key, target in current.aliases.items())))
    aliases.setMaximumHeight(100)
    aliases.setObjectName('aliases')
    form.addRow('Aliases', aliases)
    error = widgets.QLabel()
    error.setObjectName('validation_error')
    error.setStyleSheet('color: #C8102E')
    error.setWordWrap(True)
    form.addRow('', error)
    apply_button = widgets.QPushButton('Apply')
    apply_button.setObjectName('apply_preferences')
    form.addRow('', apply_button)

    def collect() -> CadInputPreferences:
        alias_map: dict[str, str] = {}
        for line in aliases.toPlainText().splitlines():
            if not line.strip() or '=' not in line:
                raise ValueError('Aliases must use KEY=COMMAND lines')
            key, target = line.split('=', 1)
            if key.upper() in alias_map:
                raise ValueError(f'duplicate normalized alias: {key}')
            alias_map[key] = target
        values: dict[str, Any] = {}
        for name, control in controls.items():
            if name.endswith('_key'):
                if control.keySequence().isEmpty() or control.keySequence().count() != 1:
                    raise ValueError('Kisayol tek tus veya tek tus birlesimi olmali')
                values[name] = control.keySequence().toString()
            elif hasattr(control, 'isChecked'):
                values[name] = control.isChecked()
            elif hasattr(control, 'value'):
                values[name] = control.value()
            else:
                values[name] = control.currentText()
        values['aliases'] = alias_map
        if len({values[k] for k in ('selection_key', 'confirm_key', 'cancel_key')}) != 3:
            raise ValueError('Secim, onay ve iptal kisayollari farkli olmali')
        return CadInputPreferences(**values)

    def apply() -> None:
        nonlocal current
        try:
            candidate = collect()
        except (TypeError, ValueError) as exc:
            error.setText(str(exc))
            return
        try:
            on_change(candidate)
        except ValueError as exc:
            error.setText(str(exc))
            return
        current = candidate
        error.clear()
    apply_button.clicked.connect(apply)
    root.controls = controls
    root.apply_button = apply_button
    root.error_label = error
    root.preferences = lambda: current
    return root
