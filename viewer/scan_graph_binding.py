
from __future__ import annotations
import math
import uuid
from collections.abc import Mapping, Sequence
import numpy as np
from viewer.project_model import ProjectModel, ScanEntry, ScanGroup

class ScanGraphBindingMixin:

    def _init_scan_graph(self) -> None:
        self.scan_graph = getattr(self.window, '_cca_points_scan_graph', None)
        self._graph_selected_ids: tuple[str, ...] = ()
        self._graph_selection_active = False
        self._graph_syncing = False
        self._group_drag_originals: dict[str, np.ndarray] = {}
        if self.scan_graph is not None:
            self.scan_graph.scan_selection_changed.connect(self._graph_selection_changed)
            self.scan_graph.command_requested.connect(self._graph_command)
            self.scan_graph.layout_changed.connect(self._graph_layout_changed)

    def _graph_layout_changed(self, positions: object) -> None:
        if self._graph_syncing or not isinstance(positions, Mapping):
            return
        clean = {}
        for key, value in positions.items():
            if isinstance(key, str) and isinstance(value, (list, tuple)) and (len(value) == 2) and all((isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v) for v in value)):
                clean[key] = [float(v) for v in value]
        self.project.metadata['scan_graph'] = {'version': 1, 'positions': clean}

    def _sync_scan_graph(self) -> None:
        graph = getattr(self, 'scan_graph', None)
        if graph is None or self._graph_syncing:
            return
        self._graph_syncing = True
        try:
            graph.set_project(self.project, selected_scan_ids=self._selected_scan_ids())
            visibility = getattr(graph, 'set_scan_visibility', None)
            if callable(visibility) and self.layer_controller is not None:
                visibility(self.layer_controller.scan_visibility)
        finally:
            self._graph_syncing = False

    def _graph_selection_changed(self, scan_ids: object) -> None:
        if self._graph_syncing or not isinstance(scan_ids, (list, tuple)):
            return
        if self._plan_placement_is_active() or self._busy_job_kind is not None:
            self._sync_scan_graph()
            return
        ids = tuple(dict.fromkeys((s for s in scan_ids if self.project.scan(s) is not None)))
        self._graph_selected_ids = ids
        self._graph_selection_active = True
        self._restore_rail_selection(ids)
        if ids:
            self.select_scan(ids[0])
            if len(ids) > 1:
                self.selection.setText(f'{len(ids)} TARAMA SECILI')
                self.selection_hint.setText('Ortak tasi / dondur / gizle / kadraja al')
        else:
            self._clear_properties()
            select = getattr(self.viewport_adapter, 'select_scan', None)
            if callable(select):
                select(None)
        self._sync_editing_selection()

    def _rail_graph_selection_changed(self) -> None:
        if self._graph_syncing or self.rail.signalsBlocked():
            return
        self._graph_selection_active = False
        graph = getattr(self, 'scan_graph', None)
        if graph is not None:
            self._graph_syncing = True
            try:
                graph.select_scans(self._selected_scan_ids())
            finally:
                self._graph_syncing = False
        self._sync_editing_selection()

    def _graph_fit(self, scan_ids: Sequence[str]=()) -> None:
        if scan_ids:
            fit = getattr(self.viewport_adapter, 'fit_selection', None)
            fitted = bool(fit(tuple(scan_ids))) if callable(fit) else False
        else:
            fit = getattr(self.viewport_adapter, 'fit_scene', None)
            if not callable(fit):
                fit = getattr(self.viewport_adapter, 'frame_scans', None)
            fitted = bool(fit()) if callable(fit) else False
        self._status_message(('SECIM KADRAJDA' if scan_ids else 'TUM GORUNUR TARAMALAR KADRAJDA') if fitted else 'Kadraja alinacak gorunur geometri yok.')

    def _graph_command(self, command: str, payload: object) -> None:
        if not isinstance(payload, Mapping):
            return
        raw_ids = payload.get('scan_ids', ())
        ids = tuple(dict.fromkeys((s for s in raw_ids if self.project.scan(s) is not None)))
        if command == 'properties.show':
            if ids:
                self._graph_selection_changed(ids)
            self._show_properties_panel()
            return
        if command == 'viewport.fit':
            self._graph_fit(ids)
            return
        if self._busy_job_kind is not None or self._plan_placement_is_active():
            self._status_message("Devam eden islemi bitirdikten sonra node'lari duzenleyin.")
            return
        if command in {'scan.calculate', 'scan.proxy'}:
            if command == 'scan.calculate':
                self._calculate_scan_set(ids)
            else:
                details = frozenset(getattr(self.viewport_adapter, 'detailed_scan_ids', ()))
                self._change_placement_details(details.difference(ids))
            return
        if command in {'visibility.set', 'scan.isolate'}:
            if self.layer_controller is None:
                return
            isolate = command == 'scan.isolate'
            visible = payload.get('visible', True)
            if type(visible) is not bool:
                return
            for scan in self.project.scan_rail:
                if isolate or scan.scan_id in ids:
                    self.layer_controller.set_scan_visible(scan.scan_id, (scan.scan_id in ids if isolate else visible) and (not self._scan_is_removed(scan)))
            self._apply_visible_render_plan()
            self._sync_scan_graph()
            return
        self._discard_alignment_preview_for_edit()
        try:
            if command == 'group.create':
                label = str(payload.get('label', '')).strip() or f'Grup {len(self.project.groups) + 1:02d}'
                group_id = 'group-' + uuid.uuid4().hex[:12]
                if ids:
                    self.project.assign_scans_to_group(ids, group_id, label)
                else:
                    self.project.groups.append(ScanGroup(group_id=group_id, label=label))
            elif command == 'group.assign' and ids:
                group_id = str(payload.get('group_id', ''))
                group = next((g for g in self.project.groups if g.group_id == group_id), None)
                if group is None:
                    raise ValueError('Hedef grup bulunamadi.')
                self.project.assign_scans_to_group(ids, group_id)
            elif command == 'group.rename':
                group_id = str(payload.get('group_id', ''))
                label = str(payload.get('label', '')).strip()
                group = next((g for g in self.project.groups if g.group_id == group_id), None)
                if group is None or not label:
                    return
                group.label = label
                ids = tuple(group.scan_ids)
            elif command in {'group.ungroup', 'scan.ungroup'}:
                group_ids = payload.get('group_ids', ())
                if command == 'group.ungroup':
                    ids = tuple((s.scan_id for s in self.project.scan_rail if s.group_id in group_ids))
                if ids:
                    self.project.assign_scans_to_group(ids, 'UNGROUPED', 'Ungrouped')
                if command == 'group.ungroup':
                    self.project.groups[:] = [g for g in self.project.groups if g.group_id not in group_ids or g.scan_ids]
            else:
                return
            self._graph_selected_ids = ids
            self._graph_selection_active = True
            self._refresh_groups(ids, 'TARAMA AKISI GUNCELLENDI / konumlar korundu')
            self._sync_scan_graph()
        except (RuntimeError, TypeError, ValueError) as error:
            self._status_message(str(error), 10000)
            self._sync_scan_graph()

    def _show_properties_panel(self, workflow: str='REVIEW') -> None:
        show_context = getattr(self.window, '_cca_show_context', None)
        if callable(show_context):
            action = self.window.findChild(self.qt.QtGui.QAction, 'TogglePropertiesAction')
            if action is not None:
                action.setChecked(True)
            show_context(workflow)
            return
        dock = self.window.findChild(self.qt.QtWidgets.QDockWidget, 'PROPERTIES')
        if dock is not None:
            dock.show()
            dock.raise_()

    def _calculate_scan_set(self, scan_ids: Sequence[str]) -> None:
        if self.layer_controller is None or not scan_ids:
            self._status_message('Hesapla icin once bir tarama veya grup secin.')
            return
        plan = self.layer_controller.render_plan(require_available=False)
        visible = {actor.scan_id for actor in plan.visible_actors}
        requested = frozenset(scan_ids)
        if not requested <= visible:
            self._status_message("Secili taramalari gorunur yapin, sonra Hesapla'ya basin.")
            return
        self._change_placement_details(requested)

    def _group_transform_event(self, event: object) -> bool:
        phase = str(getattr(event, 'phase', '')).lower()
        anchor = str(getattr(event, 'scan_id', ''))
        if phase == 'begin':
            selected = self._selected_scan_ids()
            self._group_drag_originals = {}
            if anchor not in selected or len(selected) < 2:
                return False
            locked = [s for s in selected if self.project.scan(s).placement_locked]
            if locked or self._busy_job_kind is not None:
                cancel = getattr(self.viewport_adapter, 'cancel_manual_drag', None)
                if callable(cancel):
                    cancel()
                self._status_message('Grupta kilitli tarama var; once kilidi acin.' if locked else 'Devam eden islemi bekleyin.')
                return True
            self._group_drag_originals = {s: np.asarray(self.project.scan(s).transform, dtype=float).copy() for s in selected}
            self._status_message(f'{len(selected)} TARAMA / ortak donusum')
            return True
        originals = self._group_drag_originals
        if not originals or anchor not in originals:
            return False
        if phase == 'cancel':
            self._restore_group_actors(originals)
            self._group_drag_originals = {}
            self._set_transform_fields(self._transform_field_values(originals[anchor]))
            self._status_message('Grup donusumu iptal edildi.')
            return True
        if phase not in {'preview', 'commit'}:
            return False
        delta = np.asarray(event.transform, dtype=float) @ np.linalg.inv(originals[anchor])
        targets = {s: delta @ matrix for s, matrix in originals.items()}
        try:
            if any((self.project.scan(s).placement_locked for s in originals)):
                raise ValueError('Grupta kilitli tarama var; donusum iptal edildi.')
            if phase == 'preview':
                self._restore_group_actors({s: t for s, t in targets.items() if s != anchor})
                self._set_transform_fields(self._transform_field_values(targets[anchor]))
                return True
            if self._busy_job_kind is not None:
                raise ValueError('Devam eden islemi bekleyin.')
            self._commit_group_transforms(targets, 'VIEWPORT_GROUP')
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._restore_group_actors(originals)
            self._status_message(str(error), 10000)
            self._group_drag_originals = {}
            return True
        self._group_drag_originals = {}
        self._set_transform_fields(self._transform_field_values(targets[anchor]))
        self._status_message(f'{len(targets)} TARAMA TASINDI / kaynaklar korundu')
        return True

    def _restore_group_actors(self, transforms: Mapping[str, object]) -> None:
        update = getattr(self.viewport_adapter, 'update_scan_transform', None)
        if not callable(update):
            return
        for scan_id, transform in transforms.items():
            update(scan_id, transform, render=False)
        plotter = getattr(self.viewport_adapter, 'plotter', None)
        if plotter is not None:
            plotter.render()

    def _commit_group_transforms(self, targets: Mapping[str, object], source: str) -> None:
        previous = ProjectModel.from_dict(self.project.to_dict())
        previous_undo = self._alignment_undo_project
        previous_history = list(self._project_undo_stack)
        previous_redo = list(self._project_redo_stack)
        replacements = {}
        for scan_id, transform in targets.items():
            scan = self.project.scan(scan_id)
            if scan is None or scan.placement_locked:
                raise ValueError(f'Tarama kilitli veya bulunamadi: {scan_id}')
            payload = scan.to_dict()
            payload['transform'] = np.asarray(transform).tolist()
            payload['metadata']['manual_placement'] = {'mode': 'FIXED_SCALE_Z_UP', 'source': source, 'transform': payload['transform']}
            replacements[scan_id] = ScanEntry.from_dict(payload, scan.rail_order)
        try:
            self._restore_group_actors(targets)
        except (OSError, RuntimeError, ValueError):
            self._restore_group_actors({s: previous.scan(s).transform for s in targets})
            raise
        try:
            for replacement in replacements.values():
                self.project.scans[replacement.rail_order] = replacement
            self.project.metadata['initial_scan_layout'] = False
            self._record_project_history(previous)
            self._reset_alignment_controls(clear_undo=False)
            self._restore_project_render()
            self._sync_scan_graph()
        except (OSError, RuntimeError, TypeError, ValueError):
            self.project.scans[:] = previous.scans
            self.project.metadata = previous.metadata
            self._alignment_undo_project = previous_undo
            self._project_undo_stack[:] = previous_history
            self._project_redo_stack[:] = previous_redo
            self._set_undo_alignment_enabled(previous_undo is not None)
            self._set_redo_alignment_enabled(bool(previous_redo))
            self._restore_group_actors({s: previous.scan(s).transform for s in targets})
            self._sync_scan_graph()
            raise
