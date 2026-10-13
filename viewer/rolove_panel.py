
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
import numpy as np
from geometry.rolove import ProjectionSurface, alignment_matrix, fit_plane_at, fit_surface_at, frame_from_three_points, placed_matrix, rigid_matrix, rotate_surface, rotation_about, snap_angle, surfaces_in_world
from viewer.project_model import ScanEntry

class RolovePanel:

    def __init__(self, binding, scan):
        self.binding = binding
        self.project = binding.project
        self.scan = scan
        self.viewport = binding.viewport_adapter
        self.history = []
        self.picks = []
        self.pick_kind = None
        self._pick_plane = None
        self.last_fit = None
        self.output = None
        self.analysis_result = None
        self.surface_history = []
        self.viewport.reset_scan_layout()
        self.layer_key = self.viewport.rolove_source_layer(scan.scan_id)
        widgets = binding.qt.QtWidgets
        core = binding.qt.QtCore
        self.dock = widgets.QDockWidget('Rolove', binding.window)
        self.dock.setObjectName('RoloveDock')
        self.dock.setMinimumWidth(320)
        self.body = widgets.QWidget()
        layout = widgets.QVBoxLayout(self.body)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        main_layout = layout
        self.dock.setWidget(self.body)
        binding.window.addDockWidget(core.Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        for dock in binding.window.findChildren(widgets.QDockWidget):
            if dock is not self.dock and dock.widget() is not None and (dock.widget().objectName() == 'PropertiesScroll'):
                binding.window.tabifyDockWidget(dock, self.dock)
                break
        self.dock.visibilityChanged.connect(self._visibility_changed)

        def label(text):
            item = widgets.QLabel(text)
            item.setWordWrap(True)
            item.setSizePolicy(widgets.QSizePolicy.Policy.Ignored, widgets.QSizePolicy.Policy.Preferred)
            layout.addWidget(item)
            return item

        def button(text, name, callback):
            item = widgets.QPushButton(text)
            item.setObjectName(name)
            item.clicked.connect(lambda _checked=False: self._safe(callback))
            layout.addWidget(item)
            return item
        heading = label(scan.scan_id)
        heading.setObjectName('RoloveScanTitle')
        label(f'{self.layer_key} / 1:1 / Kaynak korunur').setProperty('ccaRole', 'muted')
        self.steps = widgets.QTabWidget()
        self.steps.setObjectName('RoloveSteps')
        main_layout.addWidget(self.steps, 1)

        def step(title):
            scroll = widgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(core.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            page = widgets.QWidget()
            page_layout = widgets.QVBoxLayout(page)
            page_layout.setContentsMargins(0, 14, 8, 8)
            page_layout.setSpacing(10)
            scroll.setWidget(page)
            self.steps.addTab(scroll, title)
            return page_layout
        layout = step('1 / XYZ hizala')
        label('Zemini ve eksenleri kur').setObjectName('RoloveSectionTitle')
        button('Zeminden 3 nokta sec', 'RoloveGroundPick', lambda: self.start_pick('ground'))
        label('Sira: orijin > +X yonu > +Y tarafi.\nNoktalari ayni zemin yuzeyinden secin.')
        button('Zemini otomatik bul ve hizala', 'RoloveAutoAlign', self.auto_align)
        row = widgets.QHBoxLayout()
        self.angles = []
        for axis in 'XYZ':
            spin = widgets.QDoubleSpinBox()
            spin.setObjectName(f'RoloveAngle{axis}')
            spin.setRange(-180, 180)
            spin.setMinimumWidth(0)
            spin.setSizePolicy(widgets.QSizePolicy.Policy.Ignored, widgets.QSizePolicy.Policy.Fixed)
            spin.setDecimals(2)
            spin.setPrefix(f'{axis} ')
            spin.setSuffix(' deg')
            row.addWidget(spin)
            self.angles.append(spin)
        layout.addLayout(row)
        self.scan_step = widgets.QDoubleSpinBox()
        self.scan_step.setObjectName('RoloveScanStep')
        self.scan_step.setRange(0.01, 90)
        self.scan_step.setDecimals(2)
        self.scan_step.setValue(1)
        self.scan_step.setPrefix('Aci adimi ')
        self.scan_step.setSuffix(' deg')
        layout.addWidget(self.scan_step)
        self.scan_snap = widgets.QCheckBox('Taramayi aci adimina kilitle')
        self.scan_snap.setChecked(True)
        layout.addWidget(self.scan_snap)
        for spin in self.angles:
            self.scan_step.valueChanged.connect(spin.setSingleStep)
            spin.editingFinished.connect(lambda spin=spin: self._snap_spin(spin, self.scan_step, self.scan_snap))
        button('XYZ donusunu uygula', 'RoloveApplyRotation', self.rotate)
        button('Son hizalamayi geri al', 'RoloveUndo', self.undo)
        button('Yuzey secimine gec', 'RoloveNextStep', lambda: self.steps.setCurrentIndex(1))
        layout.addStretch(1)
        layout = step('2 / Plan ve gorunus')
        label('Izdusum yuzeyini tanimla').setObjectName('RoloveSectionTitle')
        button('Yuzeyleri otomatik bul', 'RoloveAutoSurfaces', self.auto_surfaces)
        label('Zemin ustune plan ve dusey yuzeylere gorunus onerir.').setProperty('ccaRole', 'muted')
        button('Yuzeyi tek tikla sec', 'RoloveSurfaceClick', lambda: self.start_pick('surface_click'))
        label('Duvarin veya zeminin ortasina bir kez tiklayin; duzlem cevredeki noktalara uydurulur.').setProperty('ccaRole', 'muted')
        button('Yuzeyi 3 noktayla sec (ileri)', 'RoloveSurfacePick', lambda: self.start_pick('surface'))
        label('Serbest yuzey: orijin > cizimde sag > cizimde yukari. Noktalar ilk noktanin duzlemine kilitlenir.').setProperty('ccaRole', 'muted')
        surface_layout = layout
        manual = widgets.QGroupBox('Dilim ayarlari / eksen duzlemi ekle')
        manual.setObjectName('RoloveManualBox')
        manual.setCheckable(True)
        manual.setChecked(True)
        manual_layout = widgets.QVBoxLayout(manual)
        manual_body = widgets.QWidget()
        manual_layout.addWidget(manual_body)
        manual.toggled.connect(manual_body.setVisible)
        surface_layout.addWidget(manual)
        layout = widgets.QVBoxLayout(manual_body)
        self.name = widgets.QLineEdit('Plan')
        self.name.setObjectName('RoloveSurfaceName')
        self.name.setToolTip('3 nokta ve eksen duzlemi icin yuzey adi. Tek tik adi otomatik verir.')
        layout.addWidget(self.name)
        self.axis = widgets.QComboBox()
        self.axis.setObjectName('RoloveSurfaceAxis')
        self.axis.addItems(['XY / Plan', 'XZ / On gorunus', 'YZ / Yan gorunus'])
        layout.addWidget(self.axis)
        self.offset = widgets.QDoubleSpinBox()
        self.offset.setObjectName('RoloveOffset')
        self.offset.setRange(-100000000.0, 100000000.0)
        self.offset.setMinimumWidth(0)
        self.offset.setSizePolicy(widgets.QSizePolicy.Policy.Ignored, widgets.QSizePolicy.Policy.Fixed)
        self.offset.setDecimals(3)
        self.offset.setValue(1.1)
        self.offset.setPrefix('Konum ')
        self.offset.setSuffix(' m')
        layout.addWidget(self.offset)
        self.slab = widgets.QCheckBox('Yalniz derinlik dilimini izdustur')
        self.slab.setChecked(True)
        layout.addWidget(self.slab)
        row = widgets.QHBoxLayout()
        self.depth_min = widgets.QDoubleSpinBox()
        self.depth_max = widgets.QDoubleSpinBox()
        for spin, value in ((self.depth_min, -0.15), (self.depth_max, 0.15)):
            spin.setRange(-100000000.0, 100000000.0)
            spin.setMinimumWidth(0)
            spin.setSizePolicy(widgets.QSizePolicy.Policy.Ignored, widgets.QSizePolicy.Policy.Fixed)
            spin.setDecimals(3)
            spin.setValue(value)
            spin.setSuffix(' m')
            row.addWidget(spin)
        self.depth_min.setPrefix('Bas ')
        self.depth_max.setPrefix('Son ')
        self.slab.toggled.connect(self.depth_min.setEnabled)
        self.slab.toggled.connect(self.depth_max.setEnabled)
        layout.addLayout(row)
        label('Derinlik, yuzeyden ok yonune mesafedir. Kirmizi ok bakis tarafini gosterir.').setProperty('ccaRole', 'muted')
        self.resolution = widgets.QDoubleSpinBox()
        self.resolution.setObjectName('RoloveResolution')
        self.resolution.setRange(0.1, 100)
        self.resolution.setValue(2)
        self.resolution.setSuffix(' cm / piksel')
        layout.addWidget(self.resolution)
        self.axis.currentIndexChanged.connect(self._axis_changed)
        button('Duzlemi ekle', 'RoloveAddAxis', self.add_axis)
        layout = surface_layout
        self.surfaces_list = widgets.QListWidget()
        self.surfaces_list.setObjectName('RoloveSurfaces')
        self.surfaces_list.setMaximumHeight(125)
        layout.addWidget(self.surfaces_list)
        self.surface_axis = widgets.QComboBox()
        self.surface_axis.setObjectName('RoloveRotationAxis')
        self.surface_axis.addItems(['v / yuzeyde yukari', 'u / yuzeyde sag', 'normal / bakis ekseni'])
        layout.addWidget(self.surface_axis)
        self.surface_step = widgets.QDoubleSpinBox()
        self.surface_step.setObjectName('RoloveSurfaceStep')
        self.surface_step.setRange(0.01, 90)
        self.surface_step.setDecimals(2)
        self.surface_step.setValue(1)
        self.surface_step.setPrefix('Aci adimi ')
        self.surface_step.setSuffix(' deg')
        layout.addWidget(self.surface_step)
        self.surface_snap = widgets.QCheckBox('Yuzey donusunu aci adimina kilitle')
        self.surface_snap.setChecked(True)
        layout.addWidget(self.surface_snap)
        self.surface_angle = widgets.QDoubleSpinBox()
        self.surface_angle.setObjectName('RoloveSurfaceAngle')
        self.surface_angle.setRange(-180, 180)
        self.surface_angle.setDecimals(2)
        self.surface_angle.setSuffix(' deg')
        self.surface_step.valueChanged.connect(self.surface_angle.setSingleStep)
        self.surface_angle.editingFinished.connect(lambda: self._snap_spin(self.surface_angle, self.surface_step, self.surface_snap))
        layout.addWidget(self.surface_angle)
        row = widgets.QHBoxLayout()
        for caption, name, callback in (('- Adim', 'RoloveSurfaceMinus', lambda: self.turn_surface(-self.surface_step.value())), ('+ Adim', 'RoloveSurfacePlus', lambda: self.turn_surface(self.surface_step.value()))):
            item = widgets.QPushButton(caption)
            item.setObjectName(name)
            item.clicked.connect(lambda _checked=False, cb=callback: self._safe(cb))
            row.addWidget(item)
        layout.addLayout(row)
        button('Secili yuzeyi dondur', 'RoloveRotateSurface', lambda: self.turn_surface(self.surface_angle.value()))
        button('Yuzey duzenlemesini geri al', 'RoloveUndoSurface', self.undo_surface)
        label('Yuzey orijini sabit. Nokta bulutu donmez.').setProperty('ccaRole', 'muted')
        button('Secili yuzeyin bakisini ters cevir', 'RoloveFlip', self.flip_surface)
        button('Secili yuzeyi kaldir', 'RoloveRemove', self.remove_surface)
        layout.addStretch(1)
        layout = step('3 / Kat plani')
        label('Elemanlari analiz et').setObjectName('RoloveSectionTitle')
        self.plan_surface = widgets.QComboBox()
        self.plan_surface.setObjectName('RolovePlanSurface')
        self.plan_surface.setToolTip('Tum kat hacmi bu yatay yuzeye gore analiz edilir. Zemin = yuzey - kesit yuksekligi.')
        layout.addWidget(self.plan_surface)
        row = widgets.QHBoxLayout()
        self.plan_cut = widgets.QDoubleSpinBox()
        self.plan_cut.setObjectName('RolovePlanCut')
        self.plan_cut.setRange(0.2, 20)
        self.plan_cut.setValue(1.1)
        self.plan_cut.setPrefix('Kesit ')
        self.plan_cut.setSuffix(' m')
        row.addWidget(self.plan_cut)
        self.plan_height = widgets.QDoubleSpinBox()
        self.plan_height.setObjectName('RoloveStoreyHeight')
        self.plan_height.setRange(0.5, 50)
        self.plan_height.setValue(3.5)
        self.plan_height.setPrefix('Kat ')
        self.plan_height.setSuffix(' m')
        row.addWidget(self.plan_height)
        layout.addLayout(row)
        button('Kolon / duvar / aciklik / merdiven bul', 'RoloveAnalyzePlan', self.analyze_plan)
        self.element_list = widgets.QTreeWidget()
        self.element_list.setObjectName('RoloveElements')
        self.element_list.setHeaderLabels(['Eleman', 'Sinif', 'Durum'])
        self.element_list.setMinimumHeight(135)
        self.element_list.setMaximumHeight(190)
        layout.addWidget(self.element_list)
        self.element_kind = widgets.QComboBox()
        self.element_kind.setObjectName('RoloveElementKind')
        self.element_kind.addItems(['DUVAR', 'KOLON', 'PENCERE', 'KAPI', 'MERDIVEN', 'BELIRSIZ'])
        layout.addWidget(self.element_kind)
        button('Secili elemanin sinifini duzelt', 'RoloveClassifyElement', self.classify_element)
        button('Secili adayi disla / geri al', 'RoloveExcludeElement', self.exclude_element)
        label('Kesik: ELLE OLC. Gorunmeyen oda ve kalinlik tamamlanmaz.').setProperty('ccaRole', 'muted')
        self.plan_cut.valueChanged.connect(self._invalidate_analysis)
        self.plan_height.valueChanged.connect(self._invalidate_analysis)
        self.surfaces_list.currentRowChanged.connect(self._invalidate_analysis)
        self.plan_surface.currentIndexChanged.connect(self._invalidate_analysis)
        layout.addStretch(1)
        layout = main_layout
        self.cancel_button = button('Nokta secimini iptal et (Esc)', 'RoloveCancelPick', lambda: self.viewport.stop_rolove_pick(cancel=True))
        self.cancel_button.hide()
        self.projection_export_button = button('Nokta izdusumu PNG + DXF', 'RoloveExport', self.export)
        self.projection_export_button.setProperty('ccaRole', 'primary')
        self.preview_button = button('Kat planini buyut', 'RolovePreviewPlan', self.preview_plan)
        self.plan_export_button = button('Kat plani PNG + DXF uret', 'RoloveExportPlan', self.export_plan)
        self.plan_export_button.setProperty('ccaRole', 'primary')
        self.open_button = button('Cikti klasorunu ac', 'RoloveOpenOutput', self.open_output)
        self.open_button.setEnabled(False)
        self.status = label('Hazir. Projeyi kaydedince hizalama ve yuzeyler de saklanir.')
        self.status.setObjectName('RoloveStatus')
        self.status.setMaximumHeight(55)
        self.steps.currentChanged.connect(self._step_changed)
        self.surfaces_list.currentRowChanged.connect(self._show_surfaces)
        for spin in self.body.findChildren(widgets.QDoubleSpinBox):
            spin.setMinimumWidth(0)
            spin.setSizePolicy(widgets.QSizePolicy.Policy.Ignored, widgets.QSizePolicy.Policy.Fixed)
        self.refresh()
        self._step_changed(self.steps.currentIndex())
        self.dock.show()
        self.dock.raise_()
        binding.window.resizeDocks([self.dock], [350], core.Qt.Orientation.Horizontal)

    def _safe(self, callback):
        try:
            self._check()
            callback()
        except (ValueError, OSError, RuntimeError) as error:
            self.message(str(error))

    def _check(self, *, edit=False):
        if self.binding.project is not self.project or self.project.scan(self.scan.scan_id) is not self.scan:
            raise ValueError('Proje degisti. Secili tarama icin paneli yeniden acin.')
        if self.binding._busy_job_kind is not None:
            raise ValueError('Devam eden islemin bitmesini bekleyin.')
        if edit and self.scan.placement_locked:
            raise ValueError('Tarama kilitli. ALIGN > Lock ile kilidi acin.')

    def message(self, text):
        self.status.setText(text)
        self.status.setToolTip(text)
        self.binding._status_message(text, 15000)

    def _step_changed(self, index):
        self.projection_export_button.setVisible(index != 2)
        self.plan_export_button.setVisible(index == 2)
        self.preview_button.setVisible(index == 2)
        self._show_surfaces()

    def _show_surfaces(self, *_args):
        surfaces = self.surfaces()
        row = self.surfaces_list.currentRow()
        visible = [surfaces[row]] if self.steps.currentIndex() != 2 and 0 <= row < len(surfaces) else []
        self.viewport.show_rolove_surfaces(self.scan.scan_id, visible, self.picks)

    def _visibility_changed(self, visible):
        if not visible:
            self.viewport.stop_rolove_pick(cancel=True)
            self.viewport.clear_rolove_surfaces()
        elif hasattr(self, 'surfaces_list'):
            self._safe(self.refresh)

    def close(self):
        self.dock.close()
        self.binding.window.removeDockWidget(self.dock)
        self.dock.deleteLater()

    def surfaces(self):
        return surfaces_in_world(self.scan.transform, self.scan.metadata)

    def refresh(self):
        surfaces = self.surfaces()
        row = self.surfaces_list.currentRow()
        self.surfaces_list.blockSignals(True)
        self.surfaces_list.clear()
        for surface in surfaces:
            depth = 'tum noktalar' if surface.depth_min is None else f'{surface.depth_min:g}..{surface.depth_max:g} m'
            self.surfaces_list.addItem(f'{surface.name} / {depth}')
        if surfaces:
            self.surfaces_list.setCurrentRow(min(row, len(surfaces) - 1) if row >= 0 else len(surfaces) - 1)
        self.surfaces_list.blockSignals(False)
        selected = self.plan_surface.currentData()
        self.plan_surface.blockSignals(True)
        self.plan_surface.clear()
        for index, surface in enumerate(surfaces):
            if surface.normal[2] >= 0.85:
                self.plan_surface.addItem(surface.name, index)
        restored = self.plan_surface.findData(selected)
        if restored >= 0:
            self.plan_surface.setCurrentIndex(restored)
        self.plan_surface.blockSignals(False)
        self._show_surfaces()

    def _save_surfaces(self, surfaces):
        self.surface_history.append((self.surfaces(), placed_matrix(self.scan.transform, self.scan.metadata).copy()))
        self._invalidate_analysis()
        config = deepcopy(self.scan.metadata.get('rolove', {}))
        config['surfaces'] = [surface.to_dict() for surface in surfaces]
        config['surface_frame'] = placed_matrix(self.scan.transform, self.scan.metadata).tolist()
        config['source_layer_key'] = self.layer_key
        self.scan.metadata = {**self.scan.metadata, 'rolove': config}
        self.refresh()

    def apply_alignment(self, delta, evidence=None):
        self._check(edit=True)
        delta = rigid_matrix(delta)
        self._invalidate_analysis()
        self.surface_history.clear()
        previous = deepcopy(self.scan.metadata)
        placement = rigid_matrix(self.scan.transform)
        aligned = np.linalg.inv(placement) @ delta @ placement @ alignment_matrix(previous)
        config = {'alignment': rigid_matrix(aligned).tolist(), 'surfaces': [], 'source_layer_key': self.layer_key}
        config['alignment_evidence'] = evidence or {'method': 'OPERATOR_XYZ', 'delta': delta.tolist()}
        self.scan.metadata = {**previous, 'rolove': config}
        try:
            self.viewport.reset_scan_layout()
            self.viewport.update_scan_transform(self.scan.scan_id, self.scan.transform)
        except Exception:
            self.scan.metadata = previous
            self.viewport.update_scan_transform(self.scan.scan_id, self.scan.transform)
            raise
        self.history.append(previous.get('rolove'))
        self.refresh()
        self.viewport.frame_scan(self.scan.scan_id)
        self.steps.setCurrentIndex(1)
        self.message('XYZ hizalandi / olcek 1:1. Eski yuzeyler sifirlandi; simdi izdusum yuzeylerini secin.')

    def rotate(self):
        self._check(edit=True)
        if not any((spin.value() for spin in self.angles)):
            self.message('Donus icin X, Y veya Z acisi girin.')
            return
        self.viewport.reset_scan_layout()
        points = self.viewport.rolove_display_points(self.scan.scan_id)
        pivot = (points.min(axis=0) + points.max(axis=0)) / 2
        angles = [spin.value() for spin in self.angles]
        if self.scan_snap.isChecked():
            angles = [snap_angle(angle, self.scan_step.value()) for angle in angles]
        self.apply_alignment(rotation_about(angles, pivot))
        for spin in self.angles:
            spin.setValue(0)

    def undo(self):
        self._check(edit=True)
        if not self.history:
            raise ValueError('Bu oturumda geri alinacak hizalama yok.')
        restored = self.history.pop()
        self._invalidate_analysis()
        self.surface_history.clear()
        self.scan.metadata = dict(self.scan.metadata)
        if restored is None:
            self.scan.metadata.pop('rolove', None)
        else:
            self.scan.metadata['rolove'] = restored
        self.viewport.reset_scan_layout()
        self.viewport.update_scan_transform(self.scan.scan_id, self.scan.transform)
        self.refresh()
        self.message('Hizalama ve onceki yuzeyler geri alindi.')

    def start_pick(self, kind):
        self._check(edit=kind == 'ground')
        self.picks = []
        self.pick_kind = kind
        self._pick_plane = None
        self.cancel_button.show()
        self.steps.setCurrentIndex(0 if kind == 'ground' else 1)
        self.viewport.start_rolove_pick(self.scan.scan_id, self.receive_pick)
        self.refresh()
        self.message('Yuzeyin ortasina bir kez tiklayin. Orta tus: orbit; Shift+orta: pan; Esc: iptal.' if kind == 'surface_click' else '1/3: Orijini secin. Orta tus: orbit; Shift+orta: pan; Esc: iptal.')

    def receive_pick(self, point):
        if point is None:
            self.cancel_button.hide()
            self.picks = []
            self._pick_plane = None
            self.refresh()
            self.message('Nokta secimi iptal edildi.')
            return
        try:
            self._check(edit=self.pick_kind == 'ground')
            if self.pick_kind == 'surface_click':
                self.viewport.stop_rolove_pick()
                self.cancel_button.hide()
                self._add_fitted_surface(point)
                return
            self.picks.append(tuple(self._coplanar_pick(point)))
            self.refresh()
            if len(self.picks) < 3:
                locked = ' (duzleme kilitli)' if self._pick_plane is not None else ''
                self.message(f'2/3: +X / cizimde sag yonu secin{locked}.' if len(self.picks) == 1 else f'3/3: +Y tarafi / cizimde yukari yonu secin{locked}.')
                return
            self.viewport.stop_rolove_pick()
            self.cancel_button.hide()
            picked = self.picks
            self.picks = []
            if self.pick_kind == 'ground':
                self.apply_alignment(frame_from_three_points(picked), {'method': 'OPERATOR_GROUND_3POINT', 'picked_world_m': picked})
            else:
                surface = ProjectionSurface.from_points(self._unique_name(self.name.text().strip()), picked, **self.options())
                self._append_surface(surface)
                self.message(f'{surface.name} eklendi (3 nokta). Kirmizi okun bakis tarafini kontrol edin.')
        except (ValueError, RuntimeError) as error:
            self.viewport.stop_rolove_pick()
            self.cancel_button.hide()
            self.picks = []
            self._pick_plane = None
            self.message(str(error))

    def _coplanar_pick(self, point):
        if not self.picks:
            try:
                self._pick_plane = fit_plane_at(self.viewport.rolove_display_points(self.scan.scan_id), point)
            except ValueError:
                self._pick_plane = None
        if self._pick_plane is None:
            return np.asarray(point, dtype=float)
        return self._pick_plane.project(point)

    def _add_fitted_surface(self, point):
        pose = self.viewport.controller.pose()
        options = self.options()
        surface, fit = fit_surface_at(self.viewport.rolove_display_points(self.scan.scan_id), point, eye=pose.eye, screen_up=pose.screen_up, resolution_m=options['resolution_m'])
        if fit.horizontal:
            surface = replace(surface, depth_min=options['depth_min'], depth_max=options['depth_max'])
        surface = replace(surface, name=self._unique_name('Plan' if fit.horizontal else 'Duvar', numbered=True))
        self.last_fit = fit
        self._append_surface(surface)
        self.message(f'{surface.name} uyduruldu: {fit.n_points} nokta, rms {fit.rms_m * 100:.1f} cm. Kirmizi ok bakis tarafi; ters ise Ters cevir.')

    def _unique_name(self, base, *, numbered=False):
        names = {surface.name for surface in self.surfaces()}
        if not numbered and base not in names:
            return base
        number = 1 if numbered else 2
        while f'{base}-{number}' in names:
            number += 1
        return f'{base}-{number}'

    def _append_surface(self, surface):
        surfaces = [*self.surfaces(), surface]
        self._save_surfaces(surfaces)
        self.surfaces_list.setCurrentRow(len(surfaces) - 1)

    def options(self):
        return {'depth_min': self.depth_min.value() if self.slab.isChecked() else None, 'depth_max': self.depth_max.value() if self.slab.isChecked() else None, 'resolution_m': self.resolution.value() / 100}

    def _axis_changed(self, index):
        self.name.setText(('Plan', 'On-Gorunus', 'Yan-Gorunus')[index])
        self.offset.setValue(1.1 if index == 0 else 0)
        self.slab.setChecked(index == 0)

    def add_axis(self):
        surface = ProjectionSurface.axis_plane(self._unique_name(self.name.text().strip()), self.axis.currentText()[:2], self.offset.value(), **self.options())
        self._append_surface(surface)
        self.message(f'{surface.name} eklendi. Yuzeyler secili taramaya bagli.')

    def flip_surface(self):
        surfaces = self.surfaces()
        row = self.surfaces_list.currentRow()
        if row < 0:
            raise ValueError('Listeden bir yuzey secin.')
        surface = surfaces[row]
        surfaces[row] = replace(surface, u=tuple(-np.asarray(surface.u)), depth_min=None if surface.depth_max is None else -surface.depth_max, depth_max=None if surface.depth_min is None else -surface.depth_min)
        self._save_surfaces(surfaces)

    def remove_surface(self):
        surfaces = self.surfaces()
        row = self.surfaces_list.currentRow()
        if row >= 0:
            del surfaces[row]
            self._save_surfaces(surfaces)

    @staticmethod
    def _snap_spin(spin, step, enabled):
        if enabled.isChecked():
            spin.setValue(snap_angle(spin.value(), step.value()))

    def turn_surface(self, angle):
        surfaces = self.surfaces()
        row = self.surfaces_list.currentRow()
        if row < 0:
            raise ValueError('Dondurmek icin listeden bir yuzey secin.')
        step = self.surface_step.value() if self.surface_snap.isChecked() else None
        axis = self.surface_axis.currentText().split(' / ')[0]
        surfaces[row] = rotate_surface(surfaces[row], angle, axis, step)
        self._save_surfaces(surfaces)
        applied = snap_angle(angle, step) if step is not None else angle
        self.message(f'{surfaces[row].name}: {axis} ekseninde {applied:g} deg dondu. Tarama degismedi.')

    def undo_surface(self):
        if not self.surface_history:
            raise ValueError('Geri alinacak yuzey duzenlemesi yok.')
        previous, frame = self.surface_history.pop()
        current = placed_matrix(self.scan.transform, self.scan.metadata)
        previous = surfaces_in_world(current, {'rolove': {'surfaces': [s.to_dict() for s in previous], 'surface_frame': frame.tolist()}})
        self._save_surfaces(previous)
        self.surface_history.pop()
        self.message('Yuzey duzenlemesi geri alindi.')

    def _invalidate_analysis(self, *_args):
        self.analysis_result = None
        if hasattr(self, 'element_list'):
            self.element_list.clear()

    def auto_surfaces(self):
        from viewer.rolove_service import automatic_surfaces
        snapshot = self._snapshot()
        self.message('Tam kaynakta zemin ve dusey yuzeyler araniyor...')
        self._job('YUZEY ARA', lambda: automatic_surfaces(snapshot, self.layer_key), self._surfaces_found)

    def _surfaces_found(self, result):
        proposals = [ProjectionSurface.from_dict(row) for row in result['surfaces']]
        if not proposals:
            self.message('Guvenilir yuzey onerisi yok. Once XYZ hizalayin veya 3 nokta secin.')
            return
        existing = self.surfaces()
        names = {surface.name for surface in existing}
        for proposal in proposals:
            name = proposal.name
            number = 2
            while name in names:
                name = f'{proposal.name}-{number}'
                number += 1
            existing.append(replace(proposal, name=name))
            names.add(name)
        self._save_surfaces(existing)
        self.scan.metadata['rolove']['surface_evidence'] = result
        self.message(f"{len(proposals)} yuzey eklendi. Zemin: {('bulundu' if result['ground_found'] else 'belirsiz; 3 nokta secin')}. Bakis oklarini kontrol edin.")

    def _analysis_inputs(self):
        from geometry.rolove_analysis import PlanAnalysisConfig
        row = self.plan_surface.currentData()
        if row is None:
            raise ValueError('Once XY plan yuzeyi ekleyin veya yuzeyleri otomatik bulun.')
        return (self.surfaces()[row], PlanAnalysisConfig(cut_height_m=self.plan_cut.value(), storey_height_m=self.plan_height.value()))

    def analyze_plan(self):
        from viewer.rolove_service import analyze_scan_plan
        surface, config = self._analysis_inputs()
        snapshot = self._snapshot()
        self._invalidate_analysis()
        self.message('Tam kat hacminde duvar, kolon, aciklik ve basamak destegi araniyor...')
        self._job('KAT PLANI', lambda: analyze_scan_plan(snapshot, surface, self.layer_key, config), self._plan_analyzed)

    def _plan_analyzed(self, result):
        self.analysis_result = result
        self.analysis_pose = placed_matrix(self.scan.transform, self.scan.metadata).copy()
        self._refresh_elements()
        counts = {}
        for element in result['analysis']['elements']:
            counts[element['kind']] = counts.get(element['kind'], 0) + 1
        message = ' / '.join((f'{kind}: {count}' for kind, count in counts.items()))
        self.message((message or 'Siniflandirilacak yeterli eleman destegi yok.') + '\nAdaylari buyutulmus planda kontrol edin; eksik eleman yok sayilmaz.')

    def _refresh_elements(self):
        self.element_list.clear()
        for element in self.analysis_result['analysis']['elements']:
            item = self.binding.qt.QtWidgets.QTreeWidgetItem([element['id'], element['kind'], element['review_status']])
            item.setToolTip(1, element['evidence']['note'])
            self.element_list.addTopLevelItem(item)
        for col in range(3):
            self.element_list.resizeColumnToContents(col)

    def _checked_analysis(self):
        if self.analysis_result is None:
            raise ValueError('Once kat plani analizi yapin. Yuzey/kot degisince yeniden analiz gerekir.')
        if not np.allclose(self.analysis_pose, placed_matrix(self.scan.transform, self.scan.metadata), rtol=0, atol=1e-08):
            self._invalidate_analysis()
            raise ValueError('Tarama tasindi. Kat planini yeniden analiz edin.')
        if Path(self._snapshot().layer(self.layer_key).path).resolve() != Path(self.analysis_result['source']).resolve():
            self._invalidate_analysis()
            raise ValueError('Kaynak katmani degisti. Kat planini yeniden analiz edin.')
        return self.analysis_result

    def _selected_element(self):
        result = self._checked_analysis()
        row = self.element_list.currentIndex().row()
        if row < 0:
            raise ValueError('Listeden bir eleman adayi secin.')
        return result['analysis']['elements'][row]

    def classify_element(self):
        element = self._selected_element()
        element.setdefault('proposed_kind', element['kind'])
        element['kind'] = self.element_kind.currentText()
        element['review_status'] = 'ELLE DUZELTILDI'
        self._refresh_elements()

    def exclude_element(self):
        element = self._selected_element()
        element['review_status'] = 'ADAY' if element['review_status'] == 'DISLA' else 'DISLA'
        self._refresh_elements()

    def preview_plan(self):
        from export.architectural_plan import KINDS
        analysis = self._checked_analysis()['analysis']
        widgets, gui, core = (self.binding.qt.QtWidgets, self.binding.qt.QtGui, self.binding.qt.QtCore)
        dialog = widgets.QDialog(self.dock)
        dialog.setWindowTitle('nokta / Kat plani adaylari / ELLE OLC')
        dialog.resize(1050, 650)
        layout = widgets.QVBoxLayout(dialog)
        layout.addWidget(widgets.QLabel('Gri: tarama / Siyah: duvar / Kirmizi: kolon / Mavi: pencere / Yesil: merdiven'))
        scene = widgets.QGraphicsScene(dialog)
        view = widgets.QGraphicsView(scene)
        view.setDragMode(widgets.QGraphicsView.DragMode.ScrollHandDrag)
        view.setBackgroundBrush(gui.QBrush(gui.QColor('#f5f5f5')))
        layout.addWidget(view, 1)
        path = gui.QPainterPath()
        for x, y in analysis['underlay_m']:
            path.moveTo(x * 100, -y * 100)
            path.lineTo(x * 100 + 0.5, -y * 100)
        scene.addPath(path, gui.QPen(gui.QColor('#bbbbbb'), 0))
        for element in analysis['elements']:
            if element['review_status'] == 'DISLA':
                continue
            pen = gui.QPen(gui.QColor(KINDS[element['kind']][1]), 1.6)
            pen.setCosmetic(True)
            if element['classification_confidence'] == 'LOW' or element['support'] == 'LOW':
                pen.setStyle(core.Qt.PenStyle.DashLine)
            for line in element['lines_m']:
                path = gui.QPainterPath()
                path.moveTo(line[0][0] * 100, -line[0][1] * 100)
                for x, y in line[1:]:
                    path.lineTo(x * 100, -y * 100)
                item = scene.addPath(path, pen)
                item.setToolTip(f"{element['id']} / {element['kind']} / {element['evidence']['note']}")
        row = widgets.QHBoxLayout()
        for title, callback in (('+', lambda: view.scale(1.4, 1.4)), ('-', lambda: view.scale(1 / 1.4, 1 / 1.4)), ('Sigdir', lambda: view.fitInView(scene.itemsBoundingRect(), core.Qt.AspectRatioMode.KeepAspectRatio))):
            button = widgets.QPushButton(title)
            button.clicked.connect(lambda _checked=False, cb=callback: cb())
            row.addWidget(button)
        layout.addLayout(row)
        dialog.show()
        view.fitInView(scene.itemsBoundingRect(), core.Qt.AspectRatioMode.KeepAspectRatio)
        self.preview_dialog = dialog

    def export_plan(self):
        from export.architectural_plan import export_architectural_plan
        result = deepcopy(self._checked_analysis())
        selected = self.binding.qt.QtWidgets.QFileDialog.getExistingDirectory(self.dock, 'Kat plani cikti klasoru')
        if not selected:
            return
        output = Path(selected) / f'nokta-kat-plani-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:6]}'
        self._job('KAT PLANI DXF', lambda: export_architectural_plan(output_dir=output, **result), self._exported)

    def _snapshot(self):
        snapshot = ScanEntry.from_dict(deepcopy(self.scan.to_dict()), self.scan.rail_order)
        for request in self.viewport.render_plan.actors:
            if request.scan_id == self.scan.scan_id and request.source_layer_key == self.layer_key:
                snapshot.layer(self.layer_key).path = str(request.source_path)
                break
        return snapshot

    def _job(self, kind, operation, completed):
        self.viewport.stop_rolove_pick()
        self.cancel_button.hide()
        self.body.setEnabled(False)

        def success(result):
            self.body.setEnabled(True)
            completed(result)

        def failed(error):
            self.body.setEnabled(True)
            self.message(f'Islem tamamlanamadi: {error}')
        started = self.binding._start_background_job(kind, f'{kind} / {self.scan.scan_id}', operation, success, failed)
        if not started:
            self.body.setEnabled(True)

    def auto_align(self):
        self._check(edit=True)
        from viewer.rolove_service import auto_alignment
        snapshot = self._snapshot()
        self._job('XYZ HIZALAMA', lambda: auto_alignment(snapshot, self.layer_key), self._apply_auto_result)

    def _apply_auto_result(self, result):

        def apply_when_idle():
            if self.binding._busy_job_kind is not None:
                self.binding.qt.QtCore.QTimer.singleShot(25, apply_when_idle)
            else:
                self._safe(lambda: self.apply_alignment(result['delta'], result))
        apply_when_idle()

    def export(self):
        if not self.surfaces():
            raise ValueError('En az bir izdusum yuzeyi secin.')
        from viewer.rolove_service import export_scan_projections
        snapshot = self._snapshot()
        selected = self.binding.qt.QtWidgets.QFileDialog.getExistingDirectory(self.dock, 'Rolove ciktilari icin klasor secin')
        if not selected:
            return
        output = Path(selected) / f'nokta-rolove-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:6]}'
        self._job('IZDUSUM', lambda: export_scan_projections(snapshot, output, self.layer_key), self._exported)

    def _exported(self, output):
        self.output = Path(output)
        self.open_button.setEnabled(True)
        self.message('PNG + DXF hazir / cm / Kaynak korundu.\nCikti klasorunu acarak kontrol edin.')
        self.open_button.setToolTip(str(output))

    def open_output(self):
        if self.output is not None:
            self.binding.qt.QtGui.QDesktopServices.openUrl(self.binding.qt.QtCore.QUrl.fromLocalFile(str(self.output)))
