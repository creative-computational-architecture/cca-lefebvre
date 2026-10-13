
from __future__ import annotations
import math
from copy import deepcopy
import numpy as np
from viewer.edit_commands import EDIT_COMMANDS, command_help, parse_edit_command
from viewer.status_priority import STATUS_PRIORITY_DEGRADED

class EditingBindingMixin:

    def _init_editing_controls(self):
        widgets, core = (self.qt.QtWidgets, self.qt.QtCore)
        self.command_input = self._require(widgets.QLineEdit, 'EditingCommandInput')
        self.command_feedback = self._require(widgets.QLabel, 'EditingCommandFeedback')
        self.command_prompt = self._require(widgets.QLabel, 'EditingCommandPrompt')
        self.cca_control = self._require(widgets.QDoubleSpinBox, 'PlacementCcaControl')
        self.cca_apply_button = self._require(widgets.QPushButton, 'ApplyPlacementCcaButton')
        self._command_history = []
        self._command_history_index = 0
        self._last_edit_command = ''
        self._command_history_draft = ''
        self._editing_quality_selection = None
        from viewer.desktop_app import ribbon_command_audit
        from viewer.qt_shell import _command_label
        self._extra_edit_commands = {}
        for command, state in ribbon_command_audit().items():
            if state == 'ACTIVE':
                self._extra_edit_commands[command] = command
                label = _command_label(command.split(':', 1)[1]).upper()
                self._extra_edit_commands.setdefault(label, command)
        choices = sorted({alias for item in EDIT_COMMANDS for alias in item.aliases})
        choices.extend(sorted(self._extra_edit_commands))
        self.command_completer = widgets.QCompleter(choices, self.command_input)
        self.command_completer.setCaseSensitivity(core.Qt.CaseSensitivity.CaseInsensitive)
        self.command_input.setCompleter(self.command_completer)
        for name in ('MoveTransformAction', 'RotateZTransformAction'):
            action = self.window.findChild(self.qt.QtGui.QAction, name)
            if action is not None:
                action.setShortcut(self.qt.QtGui.QKeySequence())
        self.command_input.returnPressed.connect(self._submit_edit_command)
        self.cca_apply_button.clicked.connect(lambda: self._apply_placement_cca(self.cca_control.value()))
        self.cca_control.valueChanged.connect(self._cca_draft_changed)
        from viewer.cad_input import install_cad_input
        self._cad_input = install_cad_input(self)
        self._editing_key_filter = self._cad_input

    def _focus_command_entry(self):
        self.command_feedback.setText('M: tasi / RO: dondur / ZE: tumunu sigdir / CCA: ayrinti / ?: komut yardimi')
        self.command_feedback.setToolTip(command_help())
        self.command_input.setFocus()
        self.command_input.selectAll()

    def _focus_edit_viewport(self):
        widget = getattr(self.viewport_adapter, '_widget', None)
        if widget is not None:
            widget.setFocus()

    def _cancel_edit_command(self):
        self.command_input.clear()
        self.command_completer.popup().hide()
        cad = getattr(self, '_cad_input', None)
        if cad is not None:
            cad.hide_input()
        if self._plan_placement_is_active():
            self.viewport_adapter.handle_shortcut('escape')
            return
        cancel = getattr(self.viewport_adapter, 'cancel_manual_drag', None)
        if callable(cancel):
            cancel()
        self._activate_transform_mode('SELECT')
        if cad is not None:
            cad.clear_selection()
        self._status_message('Komut iptal edildi. Secim modundasiniz.')
        self._focus_edit_viewport()

    def _recall_edit_command(self, step):
        if not self._command_history:
            return
        if self._command_history_index == len(self._command_history):
            self._command_history_draft = self.command_input.text()
        self._command_history_index = min(len(self._command_history), max(0, self._command_history_index + step))
        index = self._command_history_index
        self.command_input.setText(self._command_history[index] if index < len(self._command_history) else self._command_history_draft)

    def _submit_edit_command(self):
        text = self.command_input.text().strip()
        if text and self.execute_edit_command(text):
            self.command_input.clear()
            if text.upper() in {'?', 'HELP', 'YARDIM'}:
                return
            self.command_completer.popup().hide()
            cad = getattr(self, '_cad_input', None)
            if cad is not None:
                cad.hide_input()
            if self.qt.QtWidgets.QApplication.focusWidget() is self.command_input:
                self._focus_edit_viewport()

    def execute_edit_command(self, text: str, *, resolve_alias: bool=True) -> bool:
        text = str(text).strip()
        cad = getattr(self, '_cad_input', None)
        if cad is not None and resolve_alias:
            text = cad.resolve_alias(text)
        if text.upper() in {'SETTINGS', 'OPTIONS', 'PREFERENCES'} and cad is not None:
            cad.settings()
            return True
        if text in {'?', 'HELP', 'help', 'YARDIM', 'yardim'}:
            self._focus_command_entry()
            self.command_completer.setCompletionPrefix('')
            self.command_completer.complete()
            return True
        if self._busy_job_kind is not None:
            self._status_message('Devam eden hesap bitince komutu yeniden girin.')
            return False
        mode = getattr(self.viewport_adapter, 'transform_mode', 'SELECT')
        if text and text[0] in '0123456789+-.,' and (mode in {'MOVE', 'ROTATE Z'}):
            text = ('M ' if mode == 'MOVE' else 'RO ') + text
        try:
            try:
                parsed = parse_edit_command(text)
            except ValueError:
                command = self._extra_edit_commands.get(text.upper())
                if command is None:
                    raise
                parsed = None
            command = parsed.command_id if parsed is not None else command
            if command == 'REVIEW:CCA DENSITY':
                if parsed.arguments:
                    if not self._apply_placement_cca(parsed.arguments[0]):
                        return False
                else:
                    self.cca_control.setFocus()
                    self.cca_control.selectAll()
            elif command == 'EDIT:UNDO':
                self._undo_alignment()
            elif command == 'REVIEW:ORBIT':
                self._apply_orbit_command(parsed.arguments if parsed is not None else ())
            elif parsed is not None and parsed.arguments and (command in {'ALIGN:MOVE', 'ALIGN:ROTATE Z'}):
                self._numeric_placement(command, parsed.arguments)
            else:
                self._run_ribbon_command(command)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._status_message(str(error), 12000)
            return False
        if not self._command_history or self._command_history[-1] != text:
            self._command_history.append(text)
        self._command_history_index = len(self._command_history)
        if command in {'ALIGN:MOVE', 'ALIGN:ROTATE Z', 'REVIEW:ZOOM EXTENTS', 'REVIEW:FIT SELECTED', 'ALIGN:SELECT'}:
            self._last_edit_command = {'ALIGN:MOVE': 'MOVE', 'ALIGN:ROTATE Z': 'ROTATE'}.get(command, text)
        return True

    def _numeric_placement(self, command, arguments):
        if self._property_target is not None and self._property_target[0] == 'plan':
            self._numeric_plan_placement(command, arguments)
            return
        ids = self._selected_scan_ids()
        if not ids:
            raise ValueError('Once bir tarama veya grup secin; sonra komutu yeniden girin.')
        if self._plan_placement_is_active():
            raise ValueError('Sayisal komuttan once plan uzerine yerlestirmeyi bitirin.')
        if any((self.project.scan(scan_id).placement_locked for scan_id in ids)):
            raise ValueError('Secimde kilitli tarama var. Once kilidi acin.')
        visible = {a.scan_id for a in self.layer_controller.render_plan(require_available=False).visible_actors}
        if not set(ids).issubset(visible):
            raise ValueError('Tasiyacaginiz taramalari once gorunur yapin.')
        cancel = getattr(self.viewport_adapter, 'cancel_manual_drag', None)
        if callable(cancel):
            cancel()
        self._discard_alignment_preview_for_edit()
        self._reset_scan_layout(notice=False)
        delta = np.eye(4)
        if command == 'ALIGN:MOVE':
            delta[:3, 3] = arguments
        else:
            bounds = getattr(self.viewport_adapter, '_scan_bounds', {})
            selected_bounds = []
            for scan_id in ids:
                if scan_id not in bounds:
                    continue
                box = np.asarray(bounds[scan_id], dtype=float)
                if box.shape == (6,):
                    box = box.reshape(3, 2).T
                if box.shape == (2, 3) and np.isfinite(box).all():
                    selected_bounds.append(box)
            if selected_bounds:
                center = (np.min([b[0] for b in selected_bounds], axis=0) + np.max([b[1] for b in selected_bounds], axis=0)) / 2
            else:
                center = np.mean([np.asarray(self.project.scan(s).transform)[:3, 3] for s in ids], axis=0)
            angle = math.radians(arguments[0] % 360)
            cosine, sine = (math.cos(angle), math.sin(angle))
            delta[:3, :3] = ((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1))
            delta[:3, 3] = center - delta[:3, :3] @ center
        targets = {scan_id: delta @ np.asarray(self.project.scan(scan_id).transform) for scan_id in ids}
        self._commit_group_transforms(targets, 'COMMAND_LINE')
        self._graph_selection_changed(ids)
        label = 'tasindi (m)' if command == 'ALIGN:MOVE' else 'secim merkezi etrafinda donduruldu (derece)'
        self._status_message(f'{len(ids)} tarama {label}: ' + ' / '.join((f'{a:g}' for a in arguments)))

    def _numeric_plan_placement(self, command, arguments):
        if self._plan_placement_is_active():
            raise ValueError('Plan uzerine tarama yerlestirmeyi once bitirin.')
        plan = self.project.dxf_plan
        if plan is None or not plan.visible:
            raise ValueError('Once gorunur DXF altligi secin.')
        self.viewport_adapter.cancel_manual_drag()
        delta = np.eye(4)
        if command == 'ALIGN:MOVE':
            delta[:3, 3] = arguments
        else:
            bounds = self.viewport_adapter._actor_bounds.get(self.viewport_adapter._plan_actor_name)
            if bounds is None:
                raise ValueError('Dondurmeden once plan geometrisini yukleyin.')
            center = np.asarray(bounds, dtype=float).reshape(3, 2).mean(axis=1)
            angle = math.radians(arguments[0] % 360)
            cosine, sine = (math.cos(angle), math.sin(angle))
            delta[:3, :3] = ((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1))
            delta[:3, 3] = center - delta[:3, :3] @ center
        self._commit_plan_transform(delta @ np.asarray(plan.transform), source='COMMAND_LINE')

    def _quality_values(self):
        payload = self.project.metadata.get('placement_cca', {})
        scans = payload.get('scans', {}) if isinstance(payload, dict) else {}
        if not isinstance(scans, dict):
            return {}
        return {key: float(value) for key, value in scans.items() if self.project.scan(key) is not None and isinstance(value, (int, float)) and (not isinstance(value, bool)) and math.isfinite(value) and (value > 0)}

    def _restore_quality_settings(self):
        setter = getattr(self.viewport_adapter, 'set_placement_quality', None)
        if callable(setter):
            setter(self._quality_values())
        self._editing_quality_selection = None
        self._last_edit_command = ''

    def _sync_editing_selection(self):
        if not hasattr(self, 'cca_control'):
            return
        ids = self._selected_scan_ids()
        values = self._quality_values()
        signature = (ids, tuple((values.get(key) for key in ids)))
        if signature != self._editing_quality_selection:
            settings = {values.get(key, 0.0) for key in ids}
            value = next(iter(settings)) if len(settings) == 1 else 0.0
            self.cca_control.blockSignals(True)
            self.cca_control.setMaximum(max(10000, value))
            self.cca_control.setValue(value)
            self.cca_control.blockSignals(False)
            self._editing_quality_selection = signature
        busy = self._busy_job_kind is not None
        supported = callable(getattr(self.viewport_adapter, 'set_placement_quality', None))
        self.cca_control.setEnabled(bool(ids) and (not busy) and supported)
        self.cca_apply_button.setEnabled(bool(ids) and (not busy) and supported)
        self.cca_apply_button.setText(f'Uygula ({len(ids)})' if ids else 'Uygula')
        self.command_input.setEnabled(not busy)
        self._cca_draft_changed(self.cca_control.value())

    def _cca_draft_changed(self, value):
        ids = self._selected_scan_ids() if hasattr(self, 'scan_graph') else ()
        values = {self._quality_values().get(key, 0.0) for key in ids}
        current = 'Karisik secim' if len(values) > 1 else 'Secili tarama/grup'
        spacing = f'{1000 / value:g} mm nominal hucre' if value > 0 else 'Otomatik hafif gorunum'
        reduced = self._reduced_density_scan_ids(ids)
        if reduced:
            spacing += f' / {len(reduced)} taramada istenen ayrinti bu oturumda sigmadi, otomatik cizildi; Uygula yeniden dener'
        self.cca_control.setToolTip(f'{current} / {spacing}. Uygula ile yeniden hazirlanir. CCA/m: 1 metre icindeki nominal ornek sikligi; olcum dogrulugu degildir. Kaynak ayrintisi asilmaz. Kamera veya tasima bu hesabi tekrar baslatmaz.')

    def _reduced_density_scan_ids(self, ids):
        fallbacks = getattr(self.viewport_adapter, 'placement_density_fallbacks', ()) or ()
        reduced = {item.scan_id for item in fallbacks}
        return tuple((scan_id for scan_id in ids if scan_id in reduced))

    def _apply_placement_cca(self, value) -> bool:
        if self._busy_job_kind is not None:
            self._status_message('Ayrintiyi degistirmeden once devam eden hesabi bekleyin.')
            return False
        density = float(value)
        if not math.isfinite(density) or density < 0:
            raise ValueError('CCA sifir (otomatik) veya pozitif bir sayi olmali.')
        ids = self._selected_scan_ids()
        if not ids or self.layer_controller is None:
            self._status_message('Ayrinti icin once bir tarama veya grup secin.')
            return False
        if self._plan_placement_is_active() or getattr(self.viewport_adapter, 'manual_drag_active', False):
            self._status_message('Ayrintiyi degistirmeden once aktif yerlestirmeyi bitirin.')
            return False
        try:
            plan = self.layer_controller.render_plan(require_available=True)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._status_message(f'Ayrinti hazirlanamadi. {error}', 15000)
            return False
        visible = {actor.scan_id for actor in plan.visible_actors}
        if not set(ids).issubset(visible):
            self._status_message('Ayrintisi degisecek taramalari once gorunur yapin.')
            return False
        setter = getattr(self.viewport_adapter, 'set_placement_quality', None)
        if not callable(setter):
            return False
        previous = deepcopy(self.project.metadata.get('placement_cca'))
        old_values = self._quality_values()
        values = dict(old_values)
        for scan_id in ids:
            if density:
                values[scan_id] = density
            else:
                values.pop(scan_id, None)
        retry = getattr(self.viewport_adapter, 'retry_placement_density', None)
        if callable(retry):
            retry(ids)
        setter(values)
        project = self.project
        details = frozenset(getattr(self.viewport_adapter, 'detailed_scan_ids', ())).difference(ids)

        def ready():
            if self.project is not project:
                return
            project.metadata['placement_cca'] = {'version': 1, 'scans': values}
            self._editing_quality_selection = None
            self._sync_editing_selection()
            label = f'{density:g} CCA/m' if density else 'Otomatik'
            reduced = self._reduced_density_scan_ids(ids)
            if reduced:
                label += f' / {len(reduced)} taramada sigmadi, otomatik cizildi'
            self._status_message(f'YERLESIM HAZIR / {len(ids)} tarama / {label} / konumlar korundu', 12000)

        def failed(error):
            if self.project is not project:
                return
            setter(old_values)
            if previous is None:
                project.metadata.pop('placement_cca', None)
            else:
                project.metadata['placement_cca'] = previous
            self._editing_quality_selection = None
            self._sync_editing_selection()
            self._status_message(f'Ayrinti uygulanamadi; onceki gorunum korundu. {error}', 15000, priority=STATUS_PRIORITY_DEGRADED)
        try:
            self._apply_display_plan(plan, detailed_scan_ids=details, on_ready=ready, on_error=failed)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failed(error)
            return False
        return True
