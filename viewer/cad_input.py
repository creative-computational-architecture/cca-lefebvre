
from __future__ import annotations
from viewer.cad_preferences import CadInputPreferences, load_preferences, save_preferences
from viewer.edit_commands import EDIT_COMMANDS

def install_cad_input(binding):
    from shiboken6 import isValid
    qt = binding.qt
    core, gui, widgets = (qt.QtCore, qt.QtGui, qt.QtWidgets)
    adapter = binding.viewport_adapter
    viewport = getattr(adapter, '_widget', None) or getattr(adapter, 'widget', None)

    class Crosshair(widgets.QWidget):

        def paintEvent(self, event):
            prefs = controller.preferences
            if not controller.pointer_inside or not prefs.crosshair_percent:
                return
            painter = gui.QPainter(self)
            painter.setClipRegion(self.mask())
            painter.fillRect(self.rect(), gui.QColor('#C8C8C8'))

    class CadInput(core.QObject):

        def __init__(self):
            super().__init__(binding.window)
            self.preferences = load_preferences(qt)
            self.cursor = core.QPoint(80, 80)
            self.pointer_inside = False
            self.overlay = self.hud = self.entry = None
            self._syncing = False
            if viewport is not None:
                self.overlay = Crosshair(viewport)
                self.overlay.setObjectName('CadCrosshair')
                self.overlay.setAttribute(core.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                self.overlay.setAttribute(core.Qt.WidgetAttribute.WA_NoSystemBackground)
                self.hud = widgets.QFrame(viewport)
                self.hud.setObjectName('DynamicInputHud')
                self.hud.setStyleSheet('QFrame#DynamicInputHud {background:#22262A; border:1px solid #89949C;}QLabel {color:#CBD2D6; border:0;}QLineEdit {background:#171B1E; color:#F2F3F4; border:0; padding:4px;}')
                layout = widgets.QVBoxLayout(self.hud)
                layout.setContentsMargins(8, 5, 8, 5)
                layout.setSpacing(3)
                self.prompt = widgets.QLabel('KOMUT / Space veya Enter onay / Esc iptal')
                layout.addWidget(self.prompt)
                self.entry = widgets.QLineEdit()
                self.entry.setObjectName('DynamicCommandInput')
                self.entry.setPlaceholderText('M 1 0 0 / RO -90 / OR 30 / OP')
                layout.addWidget(self.entry)
                self.hud.setFixedWidth(310)
                self.hud.adjustSize()
                self.hud.hide()
                self.entry.textChanged.connect(self._from_cursor)
                self.entry.returnPressed.connect(self.confirm)
                self.entry.setContextMenuPolicy(core.Qt.ContextMenuPolicy.CustomContextMenu)
                self.entry.customContextMenuRequested.connect(lambda _p: self.right_click())
                binding.command_input.textChanged.connect(self._from_bar)
                self.resize()
            if adapter is not None:
                adapter._right_click_callback = self.right_click
                adapter._selection_cleared_callback = self.clear_selection

        def apply(self, prefs, *, persist=True):
            sequences = [gui.QKeySequence(getattr(prefs, k)) for k in ('selection_key', 'confirm_key', 'cancel_key')]
            if any((s.isEmpty() or s.count() != 1 for s in sequences)):
                raise ValueError('Kisayollar tek tus veya tek tus birlesimi olmali.')
            if len({s.toString() for s in sequences}) != 3:
                raise ValueError('Secim, onay ve iptal kisayollari farkli olmali.')
            self.preferences = prefs
            if adapter is not None:
                adapter._middle_button_mode = prefs.middle_button
                adapter._pickbox_px = prefs.pickbox_px
            toggle = getattr(binding, 'selection_highlight_toggle', None)
            if toggle is not None:
                blocked = toggle.blockSignals(True)
                toggle.setChecked(prefs.selection_tint)
                toggle.blockSignals(blocked)
            for name, value in (('set_selection_highlight_enabled', prefs.selection_tint), ('set_selection_box_enabled', prefs.selection_box)):
                setter = getattr(adapter, name, None)
                if callable(setter):
                    setter(value)
            if not prefs.dynamic_input:
                self.hide_input()
            if self.overlay is not None:
                self.update_crosshair()
            if persist:
                save_preferences(qt, prefs)
            binding._status_message(f'CAD girisi: {prefs.selection_key} bosta secim / Space veya {prefs.confirm_key} onay / Shift+Space bosluk / OP ayarlar')

        def update_crosshair(self):
            if self.overlay is None:
                return
            prefs = self.preferences
            if self.pointer_inside and prefs.crosshair_percent:
                viewport.setCursor(core.Qt.CursorShape.BlankCursor)
            else:
                viewport.unsetCursor()
            native_crosshair = getattr(adapter, 'set_cad_crosshair', None)
            if callable(native_crosshair):
                self.overlay.hide()
                native_crosshair(self.cursor.x(), self.cursor.y(), visible=self.pointer_inside, percent=prefs.crosshair_percent, pickbox_px=prefs.pickbox_px)
                return
            if not self.pointer_inside or not prefs.crosshair_percent:
                self.overlay.hide()
                return
            x, y = (self.cursor.x(), self.cursor.y())
            half = max(1, prefs.pickbox_px // 2)
            reach_x = round(viewport.width() * prefs.crosshair_percent / 100)
            reach_y = round(viewport.height() * prefs.crosshair_percent / 100)
            region = gui.QRegion(core.QRect(x - reach_x, y, reach_x * 2 + 1, 1))
            region |= gui.QRegion(core.QRect(x, y - reach_y, 1, reach_y * 2 + 1))
            outer = gui.QRegion(core.QRect(x - half, y - half, half * 2 + 1, half * 2 + 1))
            inner = gui.QRegion(core.QRect(x - half + 1, y - half + 1, half * 2 - 1, half * 2 - 1))
            self.overlay.setMask((region | outer) - inner)
            self.overlay.show()
            self.overlay.raise_()
            self.overlay.update()
            if self.hud is not None:
                self.hud.raise_()

        def resolve_alias(self, text):
            normalized = text.upper()
            commands = {item.name: item.aliases[0] for item in EDIT_COMMANDS}
            commands['SETTINGS'] = 'SETTINGS'
            for alias in sorted(self.preferences.aliases, key=len, reverse=True):
                if normalized == alias or normalized.startswith(alias + ' '):
                    return commands[self.preferences.aliases[alias]] + text[len(alias):]
            return text

        def _from_cursor(self, text):
            if not self._syncing:
                self._syncing = True
                binding.command_input.setText(text)
                self._syncing = False

        def _from_bar(self, text):
            if not self._syncing and self.entry is not None:
                self._syncing = True
                self.entry.setText(text)
                self._syncing = False

        def resize(self):
            if self.overlay is not None:
                self.overlay.setGeometry(viewport.rect())
                self.update_crosshair()
                self.position_input()

        def position_input(self):
            if self.hud is not None:
                x = min(self.cursor.x() + 22, viewport.width() - self.hud.width() - 4)
                y = min(self.cursor.y() + 22, viewport.height() - self.hud.height() - 4)
                self.hud.move(max(4, x), max(4, y))
                self.hud.raise_()

        def begin_input(self, text):
            if viewport is not None:
                position = viewport.mapFromGlobal(gui.QCursor.pos())
                if viewport.rect().contains(position):
                    self.cursor = position
                    self.pointer_inside = True
                    self.update_crosshair()
            target = self.entry if self.preferences.dynamic_input and self.entry is not None else binding.command_input
            if target is self.entry:
                mode = getattr(adapter, 'transform_mode', 'SELECT')
                prompt = {'MOVE': 'TASI / dx dy dz (m)', 'ROTATE Z': 'DONDUR / aci (derece)'}.get(mode, 'KOMUT')
                self.prompt.setText(prompt + f' / Space veya {self.preferences.confirm_key} onay / {self.preferences.cancel_key} iptal')
                target.setToolTip('Space: onay. Degerleri ayirmak icin Shift+Space: bosluk.')
                self.position_input()
                self.hud.show()
            target.setFocus()
            target.setText(text)
            target.setCursorPosition(len(text))

        def hide_input(self):
            if self.hud is not None:
                self.hud.hide()

        def confirm(self):
            if binding.command_input.text().strip():
                binding._submit_edit_command()
                focus = widgets.QApplication.focusWidget()
                if not binding.command_input.text() and focus in (self.entry, binding.command_input):
                    binding._focus_edit_viewport()
                return
            elif getattr(adapter, 'manual_drag_active', False):
                adapter.commit_manual_drag()
            elif binding._plan_placement_is_active():
                adapter.handle_shortcut('enter')
            elif getattr(adapter, 'transform_mode', 'SELECT') != 'SELECT':
                binding._activate_transform_mode('SELECT')
                self.hide_input()
            else:
                self.hide_input()
            binding._focus_edit_viewport()

        def clear_selection(self):
            binding._graph_selection_changed(())
            if binding._plan_placement_is_active() or binding._busy_job_kind is not None:
                return
            blocked = binding.tree.blockSignals(True)
            binding.tree.clearSelection()
            binding.tree.setCurrentItem(None)
            binding.tree.blockSignals(blocked)
            binding._sync_scan_graph()

        def settings(self):
            from viewer.qt_shell import _show_settings_dialog
            _show_settings_dialog(binding.window, qt)

        def right_click(self, position=None):
            if binding._busy_job_kind is not None:
                return
            active = bool(binding.command_input.text().strip() or getattr(adapter, 'manual_drag_active', False) or getattr(adapter, 'transform_mode', 'SELECT') != 'SELECT')
            action = self.preferences.right_click_active if active else self.preferences.right_click_idle
            if action == 'confirm':
                self.confirm()
            elif action == 'cancel':
                binding._cancel_edit_command()
            elif action in {'repeat', 'repeat_or_settings'} and binding._last_edit_command:
                binding.execute_edit_command(binding._last_edit_command, resolve_alias=False)
            elif action in {'settings', 'repeat_or_settings'}:
                self.settings()
            elif action == 'menu':
                menu = widgets.QMenu(binding.window)
                menu.setObjectName('CadContextMenu')
                for label, command in (('Tasi', 'MOVE'), ('Dondur', 'ROTATE'), ('Secim', 'SELECT')):
                    menu.addAction(label, lambda value=command: binding.execute_edit_command(value, resolve_alias=False))
                menu.addSeparator()
                menu.addAction('Ayarlar', self.settings)
                self.context_menu = menu
                point = self.cursor if position is None else core.QPoint(round(position[0]), round(position[1]))
                menu.popup(viewport.mapToGlobal(point))

        def matches(self, event, name):
            key = event.key()
            if key == core.Qt.Key.Key_Enter:
                key = core.Qt.Key.Key_Return
            sequence = gui.QKeySequence(core.QKeyCombination(event.modifiers(), core.Qt.Key(key)))
            return sequence == gui.QKeySequence(getattr(self.preferences, name))

        def space_confirmation(self, event):
            return event.key() == core.Qt.Key.Key_Space and event.modifiers() == core.Qt.KeyboardModifier.NoModifier

        def command_active(self):
            return bool(binding.command_input.text().strip() or getattr(adapter, 'manual_drag_active', False) or binding._plan_placement_is_active() or (getattr(adapter, 'transform_mode', 'SELECT') != 'SELECT'))

        def eventFilter(self, watched, event):
            if viewport is not None and (not isValid(viewport)):
                return False
            if not isValid(watched):
                return False
            event_type = event.type()
            in_viewport = viewport is not None and isinstance(watched, widgets.QWidget) and (watched is viewport or viewport.isAncestorOf(watched))
            if watched is viewport and event_type == core.QEvent.Type.Resize:
                self.resize()
            if in_viewport and event_type == core.QEvent.Type.MouseMove:
                self.cursor = viewport.mapFromGlobal(event.globalPosition().toPoint())
                self.pointer_inside = viewport.rect().contains(self.cursor)
                self.update_crosshair()
                if self.entry is None or not self.entry.hasFocus():
                    self.position_input()
            if watched is viewport and event_type in (core.QEvent.Type.Enter, core.QEvent.Type.Leave):
                self.pointer_inside = event_type == core.QEvent.Type.Enter
                if self.pointer_inside and self.preferences.crosshair_percent:
                    viewport.setCursor(core.Qt.CursorShape.BlankCursor)
                else:
                    viewport.unsetCursor()
                self.update_crosshair()
            if event_type != core.QEvent.Type.KeyPress or widgets.QApplication.activeModalWidget() is not None:
                return False
            popup = binding.command_completer.popup()
            in_command_popup = isValid(popup) and popup.isVisible() and isinstance(watched, widgets.QWidget) and (watched is popup or popup.isAncestorOf(watched))
            if watched is binding.command_input or (self.entry is not None and watched is self.entry) or in_command_popup:
                if self.matches(event, 'cancel_key'):
                    binding._cancel_edit_command()
                    return True
                if self.matches(event, 'confirm_key') or self.space_confirmation(event):
                    if not (self.space_confirmation(event) and event.isAutoRepeat()):
                        self.confirm()
                    return True
                if event.key() in {core.Qt.Key.Key_Return, core.Qt.Key.Key_Enter}:
                    return True
                if event.key() in {core.Qt.Key.Key_Up, core.Qt.Key.Key_Down}:
                    if in_command_popup or (watched is binding.command_input and popup.isVisible()):
                        return False
                    binding._recall_edit_command(-1 if event.key() == core.Qt.Key.Key_Up else 1)
                    return True
                return False
            graph = getattr(binding, 'scan_graph', None)
            in_graph = graph is not None and isValid(graph) and isinstance(watched, widgets.QWidget) and (watched is graph or graph.isAncestorOf(watched))
            if not (in_viewport or in_graph) or isinstance(watched, (widgets.QLineEdit, widgets.QTextEdit, widgets.QPlainTextEdit, widgets.QAbstractSpinBox, widgets.QComboBox, widgets.QAbstractButton)):
                return False
            if getattr(adapter, '_rolove_pick_scan_id', None):
                return False
            if self.matches(event, 'cancel_key'):
                binding._cancel_edit_command()
                return True
            if self.space_confirmation(event) and (self.command_active() or event.isAutoRepeat()):
                if not event.isAutoRepeat():
                    self.confirm()
                return True
            if self.matches(event, 'selection_key'):
                binding.command_input.clear()
                self.hide_input()
                binding._activate_transform_mode('SELECT')
                return True
            if self.matches(event, 'confirm_key'):
                self.confirm()
                return True
            if event.key() in {core.Qt.Key.Key_Return, core.Qt.Key.Key_Enter, core.Qt.Key.Key_Escape}:
                return True
            if getattr(adapter, 'manual_drag_active', False):
                return False
            if event.modifiers() & (core.Qt.KeyboardModifier.ControlModifier | core.Qt.KeyboardModifier.AltModifier | core.Qt.KeyboardModifier.MetaModifier):
                return False
            text = event.text()
            mode = getattr(adapter, 'transform_mode', 'SELECT')
            if text.lower() in {'x', 'y', 'z'} and mode == 'MOVE':
                return False
            numeric = text and text[0] in '0123456789+-.,'
            if text and (text.isalpha() or text == '?' or (numeric and mode != 'SELECT')):
                if binding._busy_job_kind is None:
                    self.begin_input(text)
                return True
            return False
    controller = CadInput()
    binding.window._cca_cad_input = controller
    widgets.QApplication.instance().installEventFilter(controller)
    try:
        controller.apply(controller.preferences, persist=False)
    except ValueError:
        controller.apply(CadInputPreferences(), persist=False)
    return controller
