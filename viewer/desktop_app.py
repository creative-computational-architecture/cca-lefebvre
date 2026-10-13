
from __future__ import annotations
import argparse
import inspect
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
from geometry.alignment_hypotheses import AlignmentHypothesis, HypothesisState, PairHypothesisSet
from geometry.plan_fit import PlanFitHypothesis, PlanFitResult, fit_scan_to_plan, validate_plan_fit_for_apply
from geometry.ply_preserve import PlyFormatError, read_ply_schema
from viewer.alignment_workflow import AlignmentEdge, AlignmentWorkflow, AlignmentWorkflowError, EdgeStatus, PairRegistrationJob, PairRegistrationResult
from viewer.brand import CREATIVE_NAME, ORGANIZATION_NAME, PRODUCT_NAME
from viewer.editing_binding import EditingBindingMixin
from viewer.editing_controls import EVIDENCE_GRADE_SLIDER_SCALE, FLOATING_SLIDER_SCALE, SPARSE_SLIDER_SCALE
from viewer.evidence_grade import DEFAULT_PIVOT, EvidenceGradeError, EvidenceGradeState
from viewer.evidence_labels import evidence_label, evidence_tooltip
from viewer.floating_votes import DEFAULT_FLOATING_THRESHOLD, FLOATING_THRESHOLD_MAX, FLOATING_THRESHOLD_MIN
from viewer.import_service import import_dxf_plan_state
from viewer.layer_controller import DisplayMode, DisplayModeUnavailableError, EvidenceProfile, LayerController, LayerControllerError, resolve_render_plan
from viewer.layer_controller import initial_evidence_profile as resolve_initial_evidence_profile
from viewer.lefebvre_mcp_bridge import publish_state as publish_mcp_state
from viewer.lefebvre_mcp_bridge import take_commands as take_mcp_commands
from viewer.plan_placement import FIXED_SCALE as PLAN_PLACEMENT_SCALE
from viewer.plan_placement import PlacementPhase, PlanPlacementError, PlanPlacementSession
from viewer.point_display import FLOATING_UI_LABEL, FLOATING_UNAVAILABLE_REASON, SKY_FILL_UNAVAILABLE_REASON, SPARSE_UI_LABEL, SPARSE_UNAVAILABLE_REASON
from viewer.project_loader import build_scanner_app_scan_entry
from viewer.project_model import DXFPlanState, ProjectModel, ScanEntry, ScanLayer
from viewer.qt_shell import PlaceholderViewportAdapter, QtModules, ViewportAdapter, create_shell_window, load_qt_modules, ribbon_command_ids
from viewer.scan_graph_binding import ScanGraphBindingMixin
from viewer.sparse_votes import DEFAULT_SPARSE_THRESHOLD_M, SPARSE_THRESHOLD_MAX_M, SPARSE_THRESHOLD_MIN_M
from viewer.status_priority import STATUS_DEGRADED_HOLD_S, STATUS_PRIORITY_DEGRADED, STATUS_PRIORITY_ROUTINE
from viewer.surface_controller import VISUAL_ONLY_NOTICE, SurfaceBackend, SurfaceController
from viewer.temporal_chain_dialog import TemporalChainDialog
_LOGGER = logging.getLogger(__name__)
SURFACE_DISPLAY_MODES = ('POINTS ONLY', 'GHOST SURFACE', 'SOLID SURFACE')
SURFACE_REBUILD_HINT = 'Yuzey sekmesinden LOCAL BPA ile yeniden olusturun'
ROOT = Path(__file__).resolve().parents[1]
_ACTIVE_RIBBON_COMMANDS = frozenset({'PROJECT:NEW', 'PROJECT:OPEN', 'PROJECT:SAVE', 'PROJECT:RECENT', 'PROJECT:SETTINGS', 'PROJECT:AUDIT', 'PROJECT:ABOUT CCA', 'IMPORT:PLY SCANS', 'IMPORT:RAW ALL DATA', 'IMPORT:DXF PLAN', 'IMPORT:APPLY CORRECTION', 'IMPORT:VALIDATE', 'IMPORT:UNIT / AXIS', 'ALIGN:SELECT', 'ALIGN:MOVE', 'ALIGN:ROTATE Z', 'ALIGN:EDIT TRANSFORM', 'ALIGN:XYZ ALIGN', 'ALIGN:PLAN PLACE', 'ALIGN:PREVIOUS SCAN', 'ALIGN:PLACE + NEXT', 'ALIGN:FINISH', 'ALIGN:AUTO ALIGN', 'ALIGN:PAIR VARIATIONS', 'ALIGN:PLAN FIT', 'ALIGN:PREVIOUS', 'ALIGN:NEXT', 'ALIGN:USE CANDIDATE', 'ALIGN:POSE GRAPH', 'ALIGN:TIME CHAIN', 'ALIGN:LOCK', 'REVIEW:PHOTO RGB', 'REVIEW:CONFIDENCE', 'REVIEW:RUN TINT', 'REVIEW:HEIGHT', 'REVIEW:PROCESSED FULL', 'REVIEW:POINT SIZE', 'REVIEW:SELECT', 'REVIEW:HIDE', 'REVIEW:ISOLATE', 'REVIEW:SHOW ALL', 'REVIEW:REMOVE', 'REVIEW:UNDO REMOVE', 'REVIEW:SCAN LAYOUT', 'REVIEW:RESET LAYOUT', 'REVIEW:SNAPSHOT', 'REVIEW:ZOOM EXTENTS', 'REVIEW:FIT SELECTED', 'REVIEW:TOP', 'REVIEW:FRONT', 'REVIEW:RIGHT', 'REVIEW:PERSPECTIVE', 'SURFACE:LOCAL BPA', 'SURFACE:POISSON', 'SURFACE:TSDF STATUS', 'EXPORT:REVIEW PNG', 'EXPORT:REVIT PACK', 'EXPORT:SURVEY PACK', 'EXPORT:PROJECTION PLANES'})
_DISABLED_RIBBON_COMMANDS = frozenset({'ALIGN:POINT PAIRS', 'ALIGN:FIND OVERLAP', 'ALIGN:ICP', 'ALIGN:COLORED ICP', 'ALIGN:C2C', 'ALIGN:C2M', 'REVIEW:CLIP', 'REVIEW:SECTION', 'REVIEW:PICKS', 'SURFACE:DENSITY', 'SURFACE:NORMALS', 'MEASURE:DISTANCE', 'MEASURE:ANGLE', 'MEASURE:AREA', 'MEASURE:ELEVATION', 'MEASURE:CLEARANCE', 'MEASURE:CONTROL POINTS', 'MEASURE:UNCERTAINTY', 'EXPORT:PLY', 'EXPORT:DXF SECTION', 'EXPORT:MESH PREVIEW', 'EXPORT:CSV', 'EXPORT:AUDIT PACKAGE'})

def ribbon_command_audit() -> dict[str, str]:
    if _ACTIVE_RIBBON_COMMANDS & _DISABLED_RIBBON_COMMANDS:
        raise RuntimeError('ribbon command policy sets overlap')
    return {command: 'ACTIVE' if command in _ACTIVE_RIBBON_COMMANDS else 'DISABLED_UNVERIFIED' if command in _DISABLED_RIBBON_COMMANDS else 'UNCLASSIFIED_DISABLED' for command in ribbon_command_ids()}

def configured_default_project() -> Path | None:
    value = os.environ.get('CCA_LEFEBVRE_PROJECT', '').strip()
    if not value:
        return None
    source = Path(value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError('The explicitly configured project file does not exist')
    return source



def cca_points_app_data_root(*, platform_name: str | None=None, home: str | Path | None=None, local_app_data: str | Path | None=None) -> Path:
    override = os.environ.get('CCA_LEFEBVRE_APP_DATA_ROOT', '').strip()
    if override:
        return Path(override).expanduser().resolve(strict=False)
    platform = sys.platform if platform_name is None else platform_name
    home_path = Path.home() if home is None else Path(home).expanduser()
    if platform == 'darwin':
        return (home_path / 'Library' / 'Application Support' / 'CCA Lefebvre').resolve(strict=False)
    if platform == 'win32':
        base_value = local_app_data or os.environ.get('LOCALAPPDATA', '')
        base = Path(base_value).expanduser() if str(base_value).strip() else home_path
        return (base / 'CCA Lefebvre').resolve(strict=False)
    xdg = os.environ.get('XDG_DATA_HOME', '').strip()
    base = Path(xdg).expanduser() if xdg else home_path / '.local' / 'share'
    return (base / 'CCA Lefebvre').resolve(strict=False)





def _ascii_scan_id(value: str, fallback: str='scan') -> str:
    normalized = unicodedata.normalize('NFKD', value)
    ascii_value = normalized.encode('ascii', 'ignore').decode('ascii')
    slug = re.sub('[^A-Za-z0-9._-]+', '-', ascii_value).strip('-._')
    return slug or fallback
MESH_SOURCE_SUFFIXES = frozenset({'.obj', '.stl'})

def _scan_entry_for_imported_ply(source: Path, rail_order: int) -> ScanEntry:
    resolved = source.expanduser().resolve()
    if resolved.suffix.lower() in MESH_SOURCE_SUFFIXES:
        return ScanEntry(scan_id=_ascii_scan_id(resolved.stem), rail_order=rail_order, source_path=str(resolved), layers=[ScanLayer(key='rgb/core', path=str(resolved), role='IMPORTED_SOURCE_MESH', visible=True, metadata={'geometry_kind': 'MESH', 'source_axis': 'SOURCE_LOCAL', 'source_note': 'legacy mesh; manually place before comparison'})], status='IMPORTED_MESH_PLACE_REQUIRED')
    if resolved.name.casefold() == 'points.ply':
        from viewer.raw_processing_service import RawProcessingError, default_processed_output_root
        try:
            scan = build_scanner_app_scan_entry(resolved.parent, processed_roots=(default_processed_output_root(),), rail_order=rail_order)
            app_fused = scan.layer('rgb/full')
            if app_fused is not None:
                app_fused.visible = True
            return scan
        except RawProcessingError:
            pass
    return ScanEntry(scan_id=_ascii_scan_id(resolved.stem), rail_order=rail_order, source_path=str(resolved), layers=[ScanLayer(key='rgb/core', path=str(resolved), role='IMPORTED_SOURCE_POINTS', visible=True)], status='IMPORTED_VALIDATE_REQUIRED')

def _cli_scan_ids(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(dict.fromkeys((part.strip() for part in value.split(',') if part.strip())))

def _should_apply_initial_scan_layout(project: ProjectModel) -> bool:
    layout_mode = project.metadata.get('initial_scan_layout')
    if layout_mode is not True and layout_mode != 'DISPLAY_ONLY_INVENTORY' or not project.scans:
        return False
    if project.dxf_plan.loaded:
        return False
    if layout_mode == 'DISPLAY_ONLY_INVENTORY':
        return True
    if any((project.metadata.get(key) for key in ('accepted_alignment_placements', 'alignment_audits', 'plan_placement', 'plan_fit_audits'))):
        return False
    identity = np.eye(4, dtype=np.float64)
    for scan in project.scan_rail:
        if scan.placement_locked:
            return False
        if not np.allclose(np.asarray(scan.transform, dtype=np.float64), identity, rtol=0.0, atol=1e-12):
            return False
        if any((scan.metadata.get(key) for key in ('accepted_global_edges', 'manual_placement', 'plan_placement', 'plan_fit'))):
            return False
        if any((token in scan.status.upper() for token in ('ALIGNED', 'LOCKED', 'PLACED'))):
            return False
    return True

class DesktopProjectBinding(EditingBindingMixin, ScanGraphBindingMixin):

    def __init__(self, window: object, qt: QtModules, project: ProjectModel | None=None, viewport_adapter: ViewportAdapter | None=None, registration_service_factory: Callable[[ProjectModel], object] | None=None, initial_evidence_profile: EvidenceProfile | str | None=None) -> None:
        self.window = window
        self.qt = qt
        self.project = project or ProjectModel.empty()
        self.viewport_adapter = viewport_adapter
        configure_placement = getattr(viewport_adapter, 'configure_placement_mode', None)
        if callable(configure_placement):
            configure_placement(cca_points_app_data_root() / 'cache' / 'placement')
        widgets = qt.QtWidgets
        self.placement_proxy_button = self._require(widgets.QPushButton, 'PlacementProxyButton')
        self.calculate_points_button = self._require(widgets.QPushButton, 'CalculatePointsButton')
        self.calculate_visible_button = self._require(widgets.QPushButton, 'CalculateVisiblePointsButton')
        self.placement_mode_status = self._require(widgets.QLabel, 'PlacementModeStatus')
        self.placement_proxy_button.clicked.connect(self._use_placement_proxies)
        self.calculate_points_button.clicked.connect(self._calculate_selected_points)
        self.calculate_visible_button.clicked.connect(self._calculate_visible_points)
        self._detail_capacity_skips: tuple = ()
        self.tree = self._require(widgets.QTreeWidget, 'ProjectTree')
        self.project_filter = self._require(widgets.QLineEdit, 'ProjectFilter')
        self.rail = self._require(widgets.QListWidget, 'ScanRailList')
        self.scan_count = self._require(widgets.QLabel, 'ScanCountLabel')
        self.pipeline_hint = self._require(widgets.QLabel, 'PipelineHint')
        self.project_status = self._require(widgets.QLabel, 'ProjectStatusLabel')
        self.selection = self._require(widgets.QLabel, 'PropertiesSelection')
        self.selection_hint = self._require(widgets.QLabel, 'PropertiesHint')
        self.quality = self._require(widgets.QLabel, 'QualitySummary')
        self.point_size = self._require(widgets.QSpinBox, 'PointSizeControl')
        self.point_size.setMaximum(10)
        self.color_mode = self._require(widgets.QComboBox, 'ColorModeControl')
        self.evidence_profile = self._require(widgets.QComboBox, 'EvidenceProfileControl')
        self.evidence_availability = self._require(widgets.QLabel, 'EvidenceAvailability')
        self.evidence_correction_button = self._require(widgets.QPushButton, 'EvidenceCorrectionButton')
        self.evidence_correction_button.clicked.connect(self._apply_selected_raw_correction)
        self.depth_model_status = self._require(widgets.QLabel, 'DepthModelStatus')
        self.depth_model_load = self._require(widgets.QPushButton, 'DepthModelLoadButton')
        self.depth_model_clear = self._require(widgets.QPushButton, 'DepthModelClearButton')
        self.depth_model_load.clicked.connect(self._choose_depth_model)
        self.depth_model_clear.clicked.connect(self._clear_depth_model)
        self.ray_recovery_button = self._require(widgets.QPushButton, 'RayRecoveryButton')
        self.ray_recovery_button.clicked.connect(self._recover_selected_raw_points)
        self.structure_search_button = self._require(widgets.QPushButton, 'StructureSearchButton')
        self.structure_search_button.clicked.connect(self._search_raw_structure)
        self.lefebvre_button = self._require(widgets.QPushButton, 'LefebvreButton')
        self.lefebvre_button.clicked.connect(self._show_lefebvre_agent)
        self._require(widgets.QPushButton, 'LefebvreAgentButton').clicked.connect(self._show_lefebvre_agent)
        self.selection_highlight_toggle = self._require(widgets.QCheckBox, 'SelectionHighlightToggle')
        self.selection_highlight_toggle.toggled.connect(self._selection_highlight_changed)
        self.point_usage_button = self._require(widgets.QPushButton, 'PointUsageAuditButton')
        self.point_usage_button.clicked.connect(self._audit_selected_point_usage)
        self.help_button = self._require(widgets.QPushButton, 'HelpButton')
        self.help_button.clicked.connect(self._show_help)
        self._require(self.qt.QtGui.QAction, 'HelpAction').triggered.connect(self._show_help)
        self._require(widgets.QPushButton, 'CorrectionFormulasButton').clicked.connect(self._show_correction_formulas)
        self.surface_display = self._require(widgets.QComboBox, 'SurfaceDisplayControl')
        self.evidence_grade_toggle = self._require(widgets.QCheckBox, 'EvidenceGradeToggle')
        self.evidence_grade_slider = self._require(widgets.QSlider, 'EvidenceGradePivotSlider')
        self.evidence_grade_faded_label = self._require(widgets.QLabel, 'EvidenceGradeFadedLabel')
        self.evidence_grade_reason_label = self._require(widgets.QLabel, 'EvidenceGradeReason')
        self.evidence_colour_gap_label = self._require(widgets.QLabel, 'EvidenceColourGapLabel')
        self.evidence_depth_fill_label = self._require(widgets.QLabel, 'EvidenceDepthFillLabel')
        self.sky_fill_toggle = self._require(widgets.QCheckBox, 'SkyFillToggle')
        self.sky_fill_legend_label = self._require(widgets.QLabel, 'SkyFillLegendLabel')
        self.floating_toggle = self._require(widgets.QCheckBox, 'FloatingToggle')
        self.floating_threshold_slider = self._require(widgets.QSlider, 'FloatingThresholdSlider')
        self.floating_legend_label = self._require(widgets.QLabel, 'FloatingLegendLabel')
        self.sparse_toggle = self._require(widgets.QCheckBox, 'SparseToggle')
        self.sparse_threshold_slider = self._require(widgets.QSlider, 'SparseThresholdSlider')
        self.sparse_legend_label = self._require(widgets.QLabel, 'SparseLegendLabel')
        self._sparse_explicit_threshold: float | None = None
        self._sparse_default_applying = False
        self.content_stack = self._require(widgets.QStackedWidget, 'MainContentStack')
        self.project_home = self._require(widgets.QWidget, 'ProjectHomePage')
        self.workspace_page = self._require(widgets.QWidget, 'WorkspacePage')
        self.scan_pulse_count = self._require(widgets.QLabel, 'ScanPulseCount')
        self.scan_pulse_mode = self._require(widgets.QLabel, 'ScanPulseMode')
        self.scan_pulse_size = self._require(widgets.QLabel, 'ScanPulseSize')
        self.scan_pulse_surface = self._require(widgets.QLabel, 'ScanPulseSurface')
        self.transform_mode_label = self._require(widgets.QLabel, 'ViewportTransformMode')
        self.plan_placement_status = self._require(widgets.QLabel, 'PlanPlacementStatus')
        self.plan_place_button = self._require(widgets.QPushButton, 'PlanPlaceButton')
        self.plan_previous_button = self._require(widgets.QPushButton, 'PlanPreviousScanButton')
        self.plan_next_button = self._require(widgets.QPushButton, 'PlanPlaceNextButton')
        self.plan_finish_button = self._require(widgets.QPushButton, 'PlanFinishButton')
        self.command_palette_button = self._require(widgets.QPushButton, 'CommandPaletteButton')
        self.import_button = self._require(widgets.QPushButton, 'ImportScanButton')
        self.new_group_button = self._require(widgets.QPushButton, 'NewGroupButton')
        self.ungroup_button = self._require(widgets.QPushButton, 'UngroupButton')
        self.auto_align_button = self._require(widgets.QPushButton, 'AutoAlignButton')
        self.review_queue_button = self._require(widgets.QPushButton, 'ReviewQueueButton')
        self.alignment_summary = self._require(widgets.QLabel, 'AlignmentPreviewSummary')
        self.accept_alignment_button = self._require(widgets.QPushButton, 'AcceptAlignmentButton')
        self.reject_alignment_button = self._require(widgets.QPushButton, 'RejectAlignmentButton')
        self.apply_alignment_button = self._require(widgets.QPushButton, 'ApplyAlignmentButton')
        self.undo_alignment_button = self._require(widgets.QPushButton, 'UndoAlignmentButton')
        self.transform_fields = {key: self._require(widgets.QDoubleSpinBox, f'Transform{key.title()}') for key in ('X', 'Y', 'Z', 'YAW')}
        self.layer_controller: LayerController | None = None
        self.surface_controller: SurfaceController | None = None
        self.registration_service_factory = registration_service_factory
        self.alignment_workflow: AlignmentWorkflow | None = None
        self._alignment_review_kind: str | None = None
        self._recent_scan_ids: list[str] = []
        self._alignment_undo_project: ProjectModel | None = None
        self._project_undo_stack: list[ProjectModel] = []
        self._project_redo_stack: list[ProjectModel] = []
        self._inspection_removal_undo: tuple[str, ScanEntry] | None = None
        self._pair_variation_set: PairHypothesisSet | None = None
        self._pair_variation_index = 0
        self._pair_variation_job: PairRegistrationJob | None = None
        self._pair_variation_service: object | None = None
        self._variation_mode: str | None = None
        self._plan_fit_result: PlanFitResult | None = None
        self._plan_fit_index = 0
        self._plan_fit_staged: PlanFitHypothesis | None = None
        self._plan_fit_accepted = False
        self._busy_job_kind: str | None = None
        self._background_job_token: str | None = None
        self._background_thread: object | None = None
        self._background_worker: object | None = None
        self._background_receiver: object | None = None
        self._busy_control_states: list[tuple[object, bool]] = []
        self._shutting_down = False
        self._close_requested = False
        self._status_hold: tuple[str, str, float] | None = None
        self._property_target: tuple[str, str] | None = None
        self._updating_transform_fields = False
        self.plan_placement_session: PlanPlacementSession | None = None
        self._plan_placement_camera_state: object | None = None
        self._plan_placement_rail_enabled = True
        self._plan_placement_tree_enabled = True
        self._plan_placement_original_locks: dict[str, bool] = {}
        self._plan_placement_skipped_scan_ids: tuple[str, ...] = ()
        self._init_editing_controls()
        self.window._cca_redo_alignment = self._redo_alignment
        self._init_scan_graph()
        self._install_close_guard()
        self._init_background_job_ui()
        self.tree.itemSelectionChanged.connect(self._tree_selection_changed)
        self.project_filter.textChanged.connect(self._filter_project_tree)
        self.rail.currentItemChanged.connect(self._rail_selection_changed)
        self.rail.itemSelectionChanged.connect(self._rail_graph_selection_changed)
        self.import_button.clicked.connect(self._choose_import_files)
        self.new_group_button.clicked.connect(self._create_group_from_selection)
        self.ungroup_button.clicked.connect(self._ungroup_selected_scans)
        self.auto_align_button.clicked.connect(self._start_auto_alignment)
        self.review_queue_button.clicked.connect(self._show_alignment_review)
        self.accept_alignment_button.clicked.connect(self._accept_alignment)
        self.reject_alignment_button.clicked.connect(self._reject_alignment)
        self.apply_alignment_button.clicked.connect(self._apply_alignment)
        self.undo_alignment_button.clicked.connect(self._undo_alignment)
        self.point_size.valueChanged.connect(self._point_size_changed)
        self.color_mode.currentTextChanged.connect(self._color_mode_changed)
        self.evidence_profile.currentIndexChanged.connect(lambda _index: self._evidence_profile_changed(self.evidence_profile.currentData()))
        self.surface_display.currentTextChanged.connect(self._surface_display_changed)
        self.evidence_grade_toggle.toggled.connect(self._evidence_grade_toggled)
        self.evidence_grade_slider.valueChanged.connect(self._evidence_grade_pivot_changed)
        self.sky_fill_toggle.toggled.connect(self._sky_fill_toggled)
        self.floating_toggle.toggled.connect(self._floating_toggled)
        self.floating_threshold_slider.valueChanged.connect(self._floating_threshold_changed)
        self.sparse_toggle.toggled.connect(self._sparse_toggled)
        self.sparse_threshold_slider.valueChanged.connect(self._sparse_threshold_changed)
        for key, field in self.transform_fields.items():
            field.valueChanged.connect(lambda _value, field_key=key: self._transform_fields_changed(field_key))
        self._connect_viewport_transform_tools()
        self._connect_ribbon_commands()
        self.command_palette_button.clicked.connect(self._show_command_palette)
        self.set_project(self.project, render=False, evidence_profile=initial_evidence_profile)
        self._mcp_timer = qt.QtCore.QTimer(self.window)
        self._mcp_timer.setInterval(400)
        self._mcp_timer.timeout.connect(self._process_mcp_commands)
        self._mcp_timer.start()
        self._publish_mcp_state()

    def _require(self, widget_type: type, object_name: str) -> Any:
        widget = self.window.findChild(widget_type, object_name)
        if widget is None:
            raise RuntimeError(f'CCA Points shell is missing {object_name}')
        return widget

    def _status_message(self, message: str, timeout_ms: int=5000, *, priority: str=STATUS_PRIORITY_ROUTINE, tooltip: str | None=None) -> bool:
        if priority not in {STATUS_PRIORITY_ROUTINE, STATUS_PRIORITY_DEGRADED}:
            raise ValueError(f'unknown status priority: {priority!r}')
        text = str(message)
        detail = text if tooltip is None else str(tooltip)
        now = time.monotonic()
        hold = getattr(self, '_status_hold', None)
        feedback = getattr(self, 'command_feedback', None)
        if priority == STATUS_PRIORITY_ROUTINE and hold is not None and (now < hold[2]):
            merged = hold[1] + '\n' + text
            self._status_hold = (hold[0], merged, hold[2])
            if feedback is not None:
                feedback.setToolTip(merged)
            return False
        self.window.statusBar().showMessage(text, timeout_ms)
        if feedback is not None:
            feedback.setText(text.splitlines()[0])
            feedback.setToolTip(detail)
        self._status_hold = (text, detail, now + STATUS_DEGRADED_HOLD_S) if priority == STATUS_PRIORITY_DEGRADED else None
        return True

    def _filter_project_tree(self, value: str) -> None:
        query = str(value).strip().casefold()

        def update(item: object) -> bool:
            own_match = not query or query in item.text(0).casefold()
            child_match = False
            for index in range(item.childCount()):
                child_match = update(item.child(index)) or child_match
            visible = own_match or child_match
            item.setHidden(not visible)
            if query and child_match:
                item.setExpanded(True)
            return visible
        for index in range(self.tree.topLevelItemCount()):
            update(self.tree.topLevelItem(index))

    def _show_command_palette(self) -> None:
        self._focus_command_entry()

    def _show_help(self) -> None:
        from viewer.help_dialog import make_help_dialog
        if getattr(self, '_help_dialog', None) is None:
            self._help_dialog = make_help_dialog(self.window)
        self._help_dialog.show()
        self._help_dialog.raise_()
        self._help_dialog.activateWindow()

    def _selection_highlight_changed(self, enabled) -> None:
        cad = getattr(self, '_cad_input', None)
        if cad is not None:
            cad.apply(replace(cad.preferences, selection_tint=bool(enabled)))
            return
        setter = getattr(self.viewport_adapter, 'set_selection_highlight_enabled', None)
        if callable(setter):
            setter(bool(enabled))
            self._status_message('Secim: turuncu boyama + kutu' if enabled else 'Secim: yalniz kutu / kaynak renkleri korunuyor')
        else:
            self._status_message('Bu gorunum secim boyamasi ayarini desteklemiyor')

    def _show_lefebvre_agent(self) -> None:
        from viewer.lefebvre_agent import make_lefebvre_agent_dialog
        existing = getattr(self, '_lefebvre_agent_dialog', None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            return

        def context():
            selected = self._selected_scan_ids()
            return f"{len(self.project.scan_rail)} tarama / secili: {(', '.join(selected) if selected else 'yok')} / {self._busy_job_kind or 'hazir'}"
        self._lefebvre_agent_dialog = make_lefebvre_agent_dialog(self.window, dispatch=self._dispatch_lefebvre_action, context=context)
        self._lefebvre_agent_dialog.show()

    def _dispatch_lefebvre_action(self, action) -> str:
        if self._busy_job_kind is not None:
            return f'{self._busy_job_kind} suruyor; yeni is baslatilmadi.'
        selected = self._selected_scan_ids()
        if action == 'sequence':
            if not isinstance(self.project.metadata.get('temporal_sequence'), Mapping) or len(self.project.scan_rail) < 2:
                return 'Zaman sirasi icin en az iki tarama ve ham zaman damgalari gerekli.'
            self._show_temporal_chain_dialog()
            return 'Zaman sirasi incelemesi acildi; proje konumlari degismedi.'
        if action == 'overlap':
            if not self.project.metadata.get('raw_data_root'):
                return 'Ortak alan aramasi icin ham tarama kaynagi gerekli.'
            self._search_rgb_alignment_candidates()
            return 'RGB ortak alan aramasi istendi; sonuclarda geometri dogrulanacak.'
        if action == 'align':
            if len(selected) != 2:
                return 'Hizalama adayi icin iki tarama secin.'
            self._start_auto_alignment()
            return 'Hizalama motoruna istek verildi. Kabul/Uygula ayri inceleme adimi; otomatik uygulanmadi.'
        if action in {'plan_place', 'plan_fit'}:
            if not self.project.dxf_plan.loaded:
                return 'Once DXF altligi ekleyin. Ham cizim degistirilmez.'
            if not self.project.scans or (action == 'plan_fit' and len(selected) != 1):
                return 'Plan yerlesimi icin tarama gerekli; uyum adayi icin tek tarama secin.'
            self._show_properties_panel('TRANSFORM' if action == 'plan_place' else 'ALIGN')
            if action == 'plan_place':
                self._start_plan_placement()
            else:
                self._start_plan_fit()
            return 'Plan araci istendi; etkin adim veya engel ana durum satirinda. Konum dogrulanmis sayilmaz.'
        if action == 'patchworking':
            if self._selected_raw_source() is None:
                return 'Patchworking icin ham depth/conf kaynagi olan bir tarama secin.'
            self._search_raw_structure(lefebvre=True)
            return 'Mekanin Uretimi / Patchworking aramasi istendi. RGB + Patch Color onizlemesi acilacak.'
        if action == 'point_audit':
            if not selected:
                return 'Nokta hesabi icin bir tarama secin.'
            self._audit_selected_point_usage()
            return 'Nokta hesabi istendi; sonuc veya engel ana durum satirinda gorunur.'
        raise ValueError('Bilinmeyen Lefebvre eylemi; islem yapilmadi')

    def _install_close_guard(self) -> None:
        qt_core = self.qt.QtCore
        binding = self

        class CloseGuard(qt_core.QObject):

            def eventFilter(self, watched: object, event: object) -> bool:
                if watched is binding.window and event.type() == qt_core.QEvent.Type.Close and (binding._busy_job_kind is not None):
                    event.ignore()
                    binding._close_requested = True
                    binding._status_message(f'{binding._busy_job_kind} is finishing; nokta will close automatically', 0)
                    return True
                return super().eventFilter(watched, event)
        self._close_guard = CloseGuard(self.window)
        self.window.installEventFilter(self._close_guard)

    def _init_background_job_ui(self) -> None:
        widgets = self.qt.QtWidgets
        self.background_job_label = widgets.QLabel('')
        self.background_job_label.setObjectName('BackgroundJobLabel')
        self.background_job_progress = widgets.QProgressBar()
        self.background_job_progress.setObjectName('BackgroundJobProgress')
        self.background_job_progress.setRange(0, 0)
        self.background_job_progress.setFixedWidth(96)
        self.background_job_label.hide()
        self.background_job_progress.hide()
        status = self.window.statusBar()
        status.addPermanentWidget(self.background_job_label)
        status.addPermanentWidget(self.background_job_progress)

    def _set_compute_controls_busy(self, busy: bool, label: str='') -> None:
        if busy:
            controls: list[object] = [self.import_button, self.new_group_button, self.ungroup_button, self.command_palette_button, self.auto_align_button, self.accept_alignment_button, self.reject_alignment_button, self.apply_alignment_button, self.undo_alignment_button, self.placement_proxy_button, self.calculate_points_button, self.calculate_visible_button, self.cca_control, self.cca_apply_button, self.command_input, self.color_mode, self.evidence_profile, self.evidence_correction_button, self.depth_model_load, self.depth_model_clear, self.ray_recovery_button, self.structure_search_button, self.lefebvre_button, self.point_usage_button, self.surface_display, self.tree, self.rail, *self.transform_fields.values()]
            if self.scan_graph is not None:
                controls.append(self.scan_graph)
            guarded_commands = {'PROJECT:NEW', 'PROJECT:OPEN', 'PROJECT:SAVE', 'IMPORT:PLY SCANS', 'IMPORT:RAW ALL DATA', 'IMPORT:DXF PLAN', 'IMPORT:APPLY CORRECTION', 'PROCESS:APPLY CORRECTION', 'ALIGN:SELECT', 'ALIGN:MOVE', 'ALIGN:ROTATE Z', 'ALIGN:XYZ ALIGN', 'ALIGN:PLAN PLACE', 'ALIGN:PREVIOUS SCAN', 'ALIGN:PLACE + NEXT', 'ALIGN:FINISH', 'ALIGN:AUTO ALIGN', 'ALIGN:PAIR VARIATIONS', 'ALIGN:PLAN FIT', 'ALIGN:PREVIOUS', 'ALIGN:NEXT', 'ALIGN:USE CANDIDATE', 'ALIGN:POSE GRAPH', 'ALIGN:TIME CHAIN', 'ALIGN:LOCK', 'REVIEW:HIDE', 'REVIEW:ISOLATE', 'REVIEW:SHOW ALL', 'REVIEW:REMOVE', 'REVIEW:UNDO REMOVE', 'REVIEW:SCAN LAYOUT', 'REVIEW:RESET LAYOUT', 'REVIEW:PHOTO RGB', 'REVIEW:CONFIDENCE', 'REVIEW:RUN TINT', 'REVIEW:HEIGHT', 'REVIEW:PROCESSED FULL', 'SURFACE:LOCAL BPA', 'SURFACE:POISSON', 'SURFACE:TSDF STATUS', 'EXPORT:SURVEY PACK', 'EXPORT:PROJECTION PLANES'}
            for button in self.window.findChildren(self.qt.QtWidgets.QAbstractButton):
                if button.property('ccaCommand') in guarded_commands:
                    controls.append(button)
            for action in self.window.findChildren(self.qt.QtGui.QAction):
                if action.property('ccaCommand') in guarded_commands:
                    controls.append(action)
            undo_action = self.window.findChild(self.qt.QtGui.QAction, 'UndoAlignmentAction')
            if undo_action is not None:
                controls.append(undo_action)
            unique_controls = list(dict.fromkeys(controls))
            self._busy_control_states = [(control, bool(control.isEnabled())) for control in unique_controls]
            for control, _enabled in self._busy_control_states:
                control.setEnabled(False)
            self.background_job_label.setText(label)
            self.background_job_progress.setRange(0, 0)
            self.background_job_label.show()
            self.background_job_progress.show()
            return
        for control, enabled in self._busy_control_states:
            control.setEnabled(enabled)
        self._busy_control_states.clear()
        self.background_job_label.hide()
        self.background_job_progress.hide()

    def _start_background_job(self, kind: str, label: str, operation: Callable[..., object], on_success: Callable[[object], None], on_error: Callable[[Exception], None], *, reports_progress: bool=False) -> bool:
        if self._shutting_down:
            self._status_message('nokta is shutting down; no new job was started')
            return False
        if self._busy_job_kind is not None:
            self._status_message(f'{self._busy_job_kind} is already running; wait for completion', 5000)
            return False
        qt_core = self.qt.QtCore
        binding = self
        token = uuid.uuid4().hex

        class Worker(qt_core.QObject):
            completed = qt_core.Signal(object, object)
            progress = qt_core.Signal(int, int, str)

            def __init__(self) -> None:
                super().__init__()

            @qt_core.Slot()
            def run(self) -> None:
                try:
                    result = operation(self.progress.emit) if reports_progress else operation()
                    self.completed.emit(result, None)
                except Exception as error:
                    self.completed.emit(None, error)

        class Receiver(qt_core.QObject):

            @qt_core.Slot(int, int, str)
            def update_progress(self, done: int, total: int, item: str) -> None:
                if token == binding._background_job_token:
                    binding.background_job_progress.setRange(0, max(total, 1))
                    binding.background_job_progress.setValue(done)
                    binding.background_job_label.setText(f'{done}/{total} / {item}')

            @qt_core.Slot(object, object)
            def deliver(self, result: object, error: object) -> None:
                if token != binding._background_job_token:
                    return
                binding._set_compute_controls_busy(False)
                if kind == 'AUTO ALIGN':
                    binding.auto_align_button.setText('AUTO ALIGN')
                try:
                    if isinstance(error, Exception):
                        on_error(error)
                    else:
                        on_success(result)
                except Exception as callback_error:
                    on_error(callback_error)
                finally:
                    thread.quit()
        thread = qt_core.QThread()
        worker = Worker()
        receiver = Receiver()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(receiver.deliver)
        worker.progress.connect(receiver.update_progress)
        worker.completed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda active_token=token: self._background_job_finished(active_token))
        self._busy_job_kind = kind
        self._background_job_token = token
        self._background_thread = thread
        self._background_worker = worker
        self._background_receiver = receiver
        self._set_compute_controls_busy(True, label)
        self._status_message(label, 0)
        thread.start()
        return True

    def shutdown_background_job(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        timer = getattr(self, '_mcp_timer', None)
        if timer is not None:
            timer.stop()
        thread = self._background_thread
        self._background_job_token = None
        is_running = getattr(thread, 'isRunning', None)
        if thread is not None and callable(is_running) and is_running():
            self._status_message(f"Waiting for {self._busy_job_kind or 'compute'} to stop safely", 0)
            request_interruption = getattr(thread, 'requestInterruption', None)
            if callable(request_interruption):
                request_interruption()
            quit_thread = getattr(thread, 'quit', None)
            if callable(quit_thread):
                quit_thread()
            event_loop = self.qt.QtCore.QEventLoop()
            thread.finished.connect(event_loop.quit)
            if is_running():
                event_loop.exec()
        self._busy_job_kind = None
        self._background_thread = None
        self._background_worker = None
        self._background_receiver = None

    def _background_job_finished(self, token: str) -> None:
        if token != self._background_job_token:
            return
        self._set_compute_controls_busy(False)
        self._busy_job_kind = None
        self._background_job_token = None
        self._background_thread = None
        self._background_worker = None
        self._background_receiver = None
        self._update_placement_mode_ui()
        self._update_plan_placement_ui()
        if self._close_requested:
            self._close_requested = False
            self.qt.QtCore.QTimer.singleShot(0, self.window.close)

    def _reset_alignment_controls(self, *, clear_undo: bool=True) -> None:
        self.alignment_workflow = None
        self._alignment_review_kind = None
        self._pair_variation_set = None
        self._pair_variation_index = 0
        self._pair_variation_job = None
        self._pair_variation_service = None
        self._variation_mode = None
        self._plan_fit_result = None
        self._plan_fit_index = 0
        self._plan_fit_staged = None
        self._plan_fit_accepted = False
        self._recent_scan_ids.clear()
        if clear_undo:
            self._alignment_undo_project = None
            self._project_undo_stack.clear()
            self._project_redo_stack.clear()
        if isinstance(self.project.metadata.get('temporal_sequence'), Mapping):
            prompt = 'Select one scan for its chronological neighbour, or select two scans explicitly for coarse-to-fine AUTO ALIGN.'
        else:
            prompt = 'Select two scans in SCAN RAIL, then run AUTO ALIGN.'
        self.alignment_summary.setText(prompt)
        self.accept_alignment_button.setText('Approve candidate')
        self.apply_alignment_button.setText('Apply to project')
        for button in (self.accept_alignment_button, self.reject_alignment_button, self.apply_alignment_button):
            button.setEnabled(False)
        self._set_undo_alignment_enabled(bool(self._project_undo_stack))
        self._set_redo_alignment_enabled(bool(self._project_redo_stack))
        self._set_variation_command_state(previous=False, next_=False, use=False)

    def _set_undo_alignment_enabled(self, enabled: bool) -> None:
        self.undo_alignment_button.setEnabled(enabled)
        action = self.window.findChild(self.qt.QtGui.QAction, 'UndoAlignmentAction')
        if action is not None:
            action.setEnabled(enabled)

    def _set_redo_alignment_enabled(self, enabled: bool) -> None:
        action = self.window.findChild(self.qt.QtGui.QAction, 'RedoAlignmentAction')
        if action is not None:
            action.setEnabled(enabled)

    def _record_project_history(self, previous: ProjectModel) -> None:
        self._project_undo_stack.append(ProjectModel.from_dict(previous.to_dict()))
        self._project_redo_stack.clear()
        self._alignment_undo_project = self._project_undo_stack[-1]
        self._set_undo_alignment_enabled(True)
        self._set_redo_alignment_enabled(False)

    def _restore_project_snapshot(self, target: ProjectModel) -> None:
        if not self._apply_project_snapshot_in_place(target):
            self.set_project(target, preserve_edit_history=True)

    @staticmethod
    def _placement_free(payload: Mapping[str, object]) -> dict[str, object]:
        stripped = dict(payload)
        stripped.pop('transform', None)
        stripped.pop('metadata', None)
        stripped.pop('quality', None)
        return stripped

    def _apply_project_snapshot_in_place(self, target: ProjectModel) -> bool:
        current = self.project
        if current is None or self.layer_controller is None:
            return False
        if [s.scan_id for s in current.scans] != [s.scan_id for s in target.scans]:
            return False
        current_top = current.to_dict()
        target_top = target.to_dict()
        for key in ('scan_rail', 'scans', 'dxf_plan', 'metadata'):
            current_top.pop(key, None)
            target_top.pop(key, None)
        if current_top != target_top:
            return False
        changed: list[tuple[ScanEntry, ScanEntry]] = []
        for cur, tgt in zip(current.scans, target.scans, strict=True):
            if self._placement_free(cur.to_dict()) != self._placement_free(tgt.to_dict()):
                return False
            if cur.transform != tgt.transform or cur.metadata != tgt.metadata:
                changed.append((cur, tgt))
        cur_plan = current.dxf_plan.to_dict() if current.dxf_plan is not None else None
        tgt_plan = target.dxf_plan.to_dict() if target.dxf_plan is not None else None
        plan_changed = cur_plan != tgt_plan
        if plan_changed:
            if cur_plan is None or tgt_plan is None:
                return False
            if self._placement_free(cur_plan) != self._placement_free(tgt_plan):
                return False
        update = getattr(self.viewport_adapter, 'update_scan_transform', None)
        update_plan = getattr(self.viewport_adapter, 'update_plan_transform', None)
        try:
            for cur, tgt in changed:
                replacement = ScanEntry.from_dict(tgt.to_dict(), tgt.rail_order)
                current.scans[replacement.rail_order] = replacement
                if callable(update) and cur.transform != replacement.transform:
                    update(replacement.scan_id, replacement.transform)
            if plan_changed:
                replacement_plan = DXFPlanState.from_dict(tgt_plan)
                current.dxf_plan = replacement_plan
                if callable(update_plan):
                    update_plan(replacement_plan.transform)
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        current.metadata = dict(target.metadata)
        self._populate_tree()
        self._populate_rail()
        self._update_status()
        target_ref = self._property_target
        if target_ref is not None and target_ref[0] == 'scan':
            scan = current.scan(target_ref[1])
            if scan is not None:
                self._set_transform_fields(self._transform_field_values(scan.transform))
        self._restore_project_render()
        sync = getattr(self, '_sync_scan_graph', None)
        if callable(sync):
            sync()
        return True

    def _discard_alignment_preview_for_edit(self) -> None:
        pending = any((self.alignment_workflow is not None, self._alignment_review_kind is not None, self._pair_variation_set is not None, self._plan_fit_staged is not None))
        if not pending:
            return
        self._restore_project_render()
        self._reset_alignment_controls(clear_undo=False)

    def _set_variation_command_state(self, *, previous: bool, next_: bool, use: bool) -> None:
        states = {'ALIGN:PREVIOUS': previous, 'ALIGN:NEXT': next_, 'ALIGN:USE CANDIDATE': use}
        for button in self.window.findChildren(self.qt.QtWidgets.QAbstractButton):
            command = button.property('ccaCommand')
            if command in states:
                button.setEnabled(states[command])
        for action in self.window.findChildren(self.qt.QtGui.QAction):
            command = action.property('ccaCommand')
            if command in states:
                action.setEnabled(states[command])

    def _connect_viewport_transform_tools(self) -> None:
        adapter = self.viewport_adapter
        if adapter is None:
            self._sync_transform_mode_ui('SELECT', 'XY')
            return
        selection = getattr(adapter, 'set_scan_selection_callback', None)
        if callable(selection):
            selection(self._viewport_scan_selected)
        plan_selection = getattr(adapter, 'set_plan_selection_callback', None)
        if callable(plan_selection):
            plan_selection(self._viewport_plan_selected)
        selection_provider = getattr(adapter, 'set_selection_provider', None)
        if callable(selection_provider):
            selection_provider(self._selected_scan_ids)
        camera_callback = getattr(adapter, 'set_camera_changed_callback', None)
        if callable(camera_callback):
            camera_callback(self._camera_view_changed)
        transform = getattr(adapter, 'set_manual_transform_callback', None)
        if callable(transform):
            transform(self._viewport_transform_event)
        mode = getattr(adapter, 'set_transform_mode_callback', None)
        if callable(mode):
            mode(self._sync_transform_mode_ui)
        self._sync_transform_mode_ui(str(getattr(adapter, 'transform_mode', 'SELECT')), str(getattr(adapter, 'move_constraint', 'XY')))

    def _camera_view_changed(self, state: object) -> None:
        label = self.window.findChild(self.qt.QtWidgets.QLabel, 'ViewportTitle')
        if label is None:
            return
        projection = 'Ortografik' if getattr(state, 'projection', '') == 'orthographic' else 'Perspektif'
        mode = {'top': 'Ust', 'walk': 'Yuruyus', 'orbit': 'Orbit'}.get(getattr(state, 'mode', ''), 'Orbit')
        text = f'{projection} / {mode} / Z yukari'
        if label.text() != text:
            label.setText(text)
        navigation = self.window.findChild(self.qt.QtWidgets.QLabel, 'ViewportNavigationHint')
        if navigation is not None:
            middle = getattr(self.viewport_adapter, '_middle_button_mode', 'orbit')
            gesture = 'ORBIT MODU / sol tus dondurur / Esc cikar' if getattr(self.viewport_adapter, 'orbit_mode', False) else 'LMB secim/duzenle   MMB pan   Orbit kilitli' if getattr(state, 'mode', '') == 'top' else 'LMB secim   MMB pan / Shift+MMB orbit' if middle == 'pan' else 'LMB secim   MMB orbit / Shift+MMB pan'
            navigation.setText(f'{gesture}   Wheel zoom   Space: komutta onay / bosta secim   Sag tik komut   OP ayarlar')

    def _sync_transform_mode_ui(self, mode: str, constraint: str='XY') -> None:
        normalized = str(mode).strip().upper()
        show_context = getattr(self.window, '_cca_show_context', None)
        if callable(show_context):
            show_context('TRANSFORM' if normalized in {'MOVE', 'ROTATE Z'} else 'REVIEW')
        if normalized in {'MOVE', 'ROTATE Z'}:
            self._reset_scan_layout(notice=False)
        adapter = getattr(self, 'viewport_adapter', None)
        orbit = bool(getattr(adapter, 'orbit_mode', False))
        label = f'{normalized} {constraint}' if normalized == 'MOVE' else normalized
        self.transform_mode_label.setText('ORBIT' if orbit else label)
        if hasattr(self, 'command_prompt'):
            self.command_prompt.setText('ORBIT' if orbit else {'MOVE': 'TASI', 'ROTATE Z': 'DONDUR'}.get(mode, 'KOMUT'))
        camera_state = getattr(getattr(adapter, 'controller', None), 'state', None)
        if camera_state is not None:
            self._camera_view_changed(camera_state)
        command_for_mode = {'SELECT': 'ALIGN:SELECT', 'MOVE': 'ALIGN:MOVE', 'ROTATE Z': 'ALIGN:ROTATE Z'}.get(normalized)
        for button in self.window.findChildren(self.qt.QtWidgets.QAbstractButton):
            command = button.property('ccaCommand')
            if command in {'ALIGN:SELECT', 'ALIGN:MOVE', 'ALIGN:ROTATE Z'}:
                button.setChecked(command == command_for_mode)

    def _apply_orbit_command(self, arguments: Sequence[float]=()) -> None:
        adapter = self.viewport_adapter
        if not (hasattr(adapter, 'orbit_degrees') and hasattr(adapter, 'set_orbit_mode')):
            raise ValueError('Bu gorunum kamera orbitini desteklemiyor.')
        if arguments:
            azimuth = float(arguments[0])
            elevation = float(arguments[1]) if len(arguments) > 1 else 0.0
            if not adapter.orbit_degrees(azimuth, elevation):
                self._status_message('UST GORUNUM duzenleme duzlemidir; orbit kilitli. Once 0 ile perspektife gecin.', 9000, priority=STATUS_PRIORITY_DEGRADED)
                return
            self._status_message(f'ORBIT / azimut {azimuth:g} derece / yukselis {elevation:g} derece')
            return
        active = bool(adapter.set_orbit_mode(not bool(getattr(adapter, 'orbit_mode', False))))
        if active:
            self._status_message('ORBIT MODU / sol tus dondurur / Esc cikar', 12000)
        else:
            mode = str(getattr(adapter, 'transform_mode', 'SELECT'))
            self._status_message(f'ORBIT MODU KAPALI / {mode} araci geri geldi')

    def _activate_transform_mode(self, mode: str) -> None:
        if self._busy_job_kind is not None:
            self._status_message('Wait for the active compute job before placement')
            return
        normalized_request = str(mode).strip().upper()
        if normalized_request in {'MOVE', 'ROTATE Z'}:
            locked = [s for s in self._selected_scan_ids() if self.project.scan(s).placement_locked]
            if locked:
                self._status_message(f"KILITLI / {', '.join(locked)} / once kilidi acin.")
                return
        selected_scan = self._selected_scan_for_inspection()
        if normalized_request in {'MOVE', 'ROTATE Z'} and selected_scan is not None and selected_scan.placement_locked:
            self._status_message(f'PLACEMENT LOCKED / {selected_scan.scan_id} / use Unlock first')
            return
        if normalized_request in {'MOVE', 'ROTATE Z'}:
            self._discard_alignment_preview_for_edit()
            self._reset_scan_layout(notice=False)
        setter = getattr(self.viewport_adapter, 'set_transform_mode', None)
        if not callable(setter):
            self._status_message('The active viewport has no manual transform tool')
            return
        try:
            normalized = str(setter(mode))
        except (RuntimeError, ValueError) as error:
            self._status_message(str(error), 10000)
            return
        constraint = str(getattr(self.viewport_adapter, 'move_constraint', 'XY'))
        self._sync_transform_mode_ui(normalized, constraint)
        widget = getattr(self.viewport_adapter, 'widget', None)
        if widget is not None and hasattr(widget, 'setFocus'):
            widget.setFocus()
        guidance = {'SELECT': 'Tarama veya grup secin. M Enter: tasi / RO Enter: dondur / ZE: tumunu sigdir', 'MOVE': 'TASI / Surukleyin; X, Y, Z: eksen / M 1 0 0: +1 m X / Esc: iptal', 'ROTATE Z': 'DONDUR / Surukleyin veya RO 15 yazin (derece) / Esc: iptal / Z yukari'}
        self._status_message(guidance.get(normalized, normalized))

    def _plan_placement_is_active(self) -> bool:
        session = self.plan_placement_session
        return session is not None and session.phase is PlacementPhase.ACTIVE

    def _set_plan_placement_command_state(self, command: str, enabled: bool) -> None:
        for button in self.window.findChildren(self.qt.QtWidgets.QAbstractButton):
            if button.property('ccaCommand') == command:
                button.setEnabled(enabled)
        for action in self.window.findChildren(self.qt.QtGui.QAction):
            if action.property('ccaCommand') == command:
                action.setEnabled(enabled)

    def _update_plan_placement_ui(self, message: str | None=None) -> None:
        session = self.plan_placement_session
        active = self._plan_placement_is_active()
        busy = self._busy_job_kind is not None
        scans = tuple((scan for scan in self.project.scan_rail if not self._scan_is_removed(scan)))
        supported = callable(getattr(self.viewport_adapter, 'enter_plan_placement', None))
        can_start = bool(not busy and (not active) and self.project.dxf_plan.loaded and scans and supported)
        if active and session is not None:
            index = int(session.active_index or 0)
            active_scan_id = session.active_scan_id or 'NO SCAN'
            default_message = f'ACTIVE {index + 1:02d}/{len(session.ordered_scan_ids):02d} / {active_scan_id}\nMOVE XY + ROTATE Z / PLACE + NEXT'
            self._set_plan_placement_command_state('ALIGN:PLAN PLACE', False)
            self._set_plan_placement_command_state('ALIGN:PREVIOUS SCAN', not busy and index > 0)
            self._set_plan_placement_command_state('ALIGN:PLACE + NEXT', not busy)
            self._set_plan_placement_command_state('ALIGN:FINISH', not busy)
        else:
            if not self.project.dxf_plan.loaded:
                default_message = 'IDLE / LOAD DXF PLAN'
            elif not scans:
                default_message = 'IDLE / IMPORT SCANS'
            elif not supported:
                default_message = 'IDLE / NATIVE VIEWPORT REQUIRED'
            else:
                plan_name = self.project.dxf_plan.source_name or 'DXF PLAN'
                default_message = f'READY / {len(scans):02d} SCANS / {plan_name}\nTOP VIEW / FIXED SCALE 1.000 / CAD Z-UP'
            self._set_plan_placement_command_state('ALIGN:PLAN PLACE', can_start)
            self._set_plan_placement_command_state('ALIGN:PREVIOUS SCAN', False)
            self._set_plan_placement_command_state('ALIGN:PLACE + NEXT', False)
            self._set_plan_placement_command_state('ALIGN:FINISH', False)
        self.plan_placement_status.setText(message or default_message)

    def _record_plan_placement_state(self, status: str) -> None:
        session = self.plan_placement_session
        if session is None:
            return
        metadata = dict(self.project.metadata)
        metadata['plan_placement'] = {'mode': 'MANUAL_DXF_SEQUENCE', 'status': str(status).upper(), 'plan_source': self.project.dxf_plan.source_name or 'DXF PLAN', 'ordered_scan_ids': list(session.ordered_scan_ids), 'placed_scan_ids': list(session.placed_ids), 'skipped_scan_ids': list(self._plan_placement_skipped_scan_ids), 'active_scan_id': session.active_scan_id, 'fixed_scale': PLAN_PLACEMENT_SCALE, 'axis': 'CAD_Z_up', 'source_geometry': 'UNCHANGED'}
        self.project.metadata = metadata

    def _restore_plan_placement_visibility(self, visibility: Mapping[str, bool]) -> None:
        if self.layer_controller is None:
            return
        for scan in self.project.scan_rail:
            if scan.scan_id not in visibility:
                continue
            self.layer_controller.set_scan_visible(scan.scan_id, bool(visibility[scan.scan_id]) and (not self._scan_is_removed(scan)))
        self._apply_visible_render_plan()

    def _refresh_plan_placement_selection(self, scan_id: str) -> None:
        self._populate_tree()
        self._populate_rail()
        self.rail.setEnabled(False)
        self.tree.setEnabled(False)
        scan = self.project.scan(scan_id)
        if scan is None:
            return
        previous = self.rail.blockSignals(True)
        try:
            self.rail.setCurrentRow(scan.rail_order)
            item = self.rail.currentItem()
            if item is not None:
                item.setSelected(True)
        finally:
            self.rail.blockSignals(previous)
        self.select_scan(scan_id)

    def _activate_plan_placement_scan(self, scan_id: str, *, sync_session_state: bool=True) -> None:
        session = self.plan_placement_session
        if session is None or session.phase is not PlacementPhase.ACTIVE:
            raise PlanPlacementError('placement session is not active')
        scan = self.project.scan(scan_id)
        if scan is None or self._scan_is_removed(scan):
            raise PlanPlacementError(f'scan is unavailable for placement: {scan_id}')
        if self.layer_controller is None:
            raise PlanPlacementError('scan display controller is unavailable')
        enter = getattr(self.viewport_adapter, 'enter_plan_placement', None)
        if not callable(enter):
            raise PlanPlacementError('native viewport does not support DXF plan placement')
        for item in self.project.scan_rail:
            self.layer_controller.set_scan_visible(item.scan_id, item.scan_id == scan_id and (not self._scan_is_removed(item)))
        self._apply_visible_render_plan()
        self._refresh_plan_placement_selection(scan_id)
        saved_camera = enter(scan_id)
        if self._plan_placement_camera_state is None:
            self._plan_placement_camera_state = saved_camera
        if scan.placement_locked:
            self.project.set_scan_placement_locked(scan_id, False)
            self._refresh_plan_placement_selection(scan_id)
        constraint = getattr(self.viewport_adapter, 'set_move_constraint', None)
        if callable(constraint):
            constraint('XY')
        self._sync_transform_mode_ui(str(getattr(self.viewport_adapter, 'transform_mode', 'MOVE')), str(getattr(self.viewport_adapter, 'move_constraint', 'XY')))
        if sync_session_state:
            self._record_plan_placement_state('ACTIVE')
            self._update_plan_placement_ui()
        widget = getattr(self.viewport_adapter, 'widget', None)
        if widget is not None and hasattr(widget, 'setFocus'):
            widget.setFocus()

    def _start_plan_placement(self) -> None:
        if self._busy_job_kind is not None:
            self._status_message('Wait for the active compute job before plan placement')
            return
        if self._plan_placement_is_active():
            self._status_message('Plan placement is already active')
            return
        if not self.project.dxf_plan.loaded:
            self._update_plan_placement_ui('IDLE / LOAD DXF PLAN FIRST')
            self._status_message('Load a DXF plan before PLAN PLACE', 10000)
            return
        candidate_scan_ids = tuple((scan.scan_id for scan in self.project.scan_rail if not self._scan_is_removed(scan)))
        if not candidate_scan_ids:
            self._update_plan_placement_ui('IDLE / IMPORT SCANS FIRST')
            self._status_message('Import scans before PLAN PLACE', 10000)
            return
        if self.layer_controller is None:
            self._status_message('Scan display controller is unavailable', 10000)
            return
        preflight = self.layer_controller.render_plan(require_available=False)
        is_prepared = getattr(self.viewport_adapter, 'render_plan_is_prepared', None)
        if getattr(self.viewport_adapter, 'placement_enabled', False) and callable(is_prepared):
            all_scans = replace(preflight, actors=tuple((replace(actor, scan_visible=True, band_visible=True) for actor in preflight.actors if actor.available)))
            if not is_prepared(all_scans):
                self._apply_display_plan(preflight, prepare_plan=all_scans, on_ready=self._start_plan_placement)
                return
        available_scan_ids = {actor.scan_id for actor in preflight.available_actors}
        ordered_scan_ids = tuple((scan_id for scan_id in candidate_scan_ids if scan_id in available_scan_ids))
        self._plan_placement_skipped_scan_ids = tuple((scan_id for scan_id in candidate_scan_ids if scan_id not in available_scan_ids))
        if not ordered_scan_ids:
            self._update_plan_placement_ui('IDLE / NO USABLE SCAN EVIDENCE')
            self._status_message('PLAN PLACE requires at least one available scan layer', 10000)
            return
        if not callable(getattr(self.viewport_adapter, 'enter_plan_placement', None)):
            self._update_plan_placement_ui('IDLE / NATIVE VIEWPORT REQUIRED')
            self._status_message('The active viewport has no DXF plan placement mode', 10000)
            return
        self._discard_alignment_preview_for_edit()
        self._reset_scan_layout(notice=False)
        original_visibility = self.layer_controller.scan_visibility
        self._plan_placement_original_locks = {scan.scan_id: scan.placement_locked for scan in self.project.scan_rail}
        self._plan_placement_rail_enabled = self.rail.isEnabled()
        self._plan_placement_tree_enabled = self.tree.isEnabled()
        self._plan_placement_camera_state = None
        session = PlanPlacementSession()
        try:
            active_scan_id = session.start(ordered_scan_ids, original_visibility, fixed_scale=PLAN_PLACEMENT_SCALE)
            self.plan_placement_session = session
            self.rail.setEnabled(False)
            self.tree.setEnabled(False)
            self._activate_plan_placement_scan(active_scan_id)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            if session.phase is PlacementPhase.ACTIVE:
                session.cancel()
            cleanup_errors = self._close_plan_placement(original_visibility, status='FAILED', notice='PLAN PLACE FAILED / PROJECT UNCHANGED')
            suffix = f" / cleanup: {'; '.join(cleanup_errors)}" if cleanup_errors else ''
            self._status_message(f'{error}{suffix}', 10000)
            return
        self._status_message(f'PLAN PLACE 01/{len(ordered_scan_ids):02d} / MOVE XY / PLACE + NEXT locks the scan / {len(self._plan_placement_skipped_scan_ids)} SKIPPED')

    def _previous_plan_scan(self) -> None:
        session = self.plan_placement_session
        if session is None or session.phase is not PlacementPhase.ACTIVE:
            self._status_message('Run PLAN PLACE first')
            return
        before = int(session.active_index or 0)
        current_scan_id = session.active_scan_id
        target_index = max(0, before - 1)
        target_scan_id = session.ordered_scan_ids[target_index]
        target = self.project.scan(target_scan_id)
        target_was_locked = bool(target is not None and target.placement_locked)
        try:
            self._activate_plan_placement_scan(target_scan_id, sync_session_state=False)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            target = self.project.scan(target_scan_id)
            if target is not None and target.placement_locked != target_was_locked:
                self.project.set_scan_placement_locked(target_scan_id, target_was_locked)
            rollback_error: Exception | None = None
            if current_scan_id is not None:
                try:
                    self._activate_plan_placement_scan(current_scan_id)
                except (OSError, RuntimeError, TypeError, ValueError) as failed:
                    rollback_error = failed
            if rollback_error is not None:
                result = session.cancel()
                self._close_plan_placement(result.restore_visibility, status='INTERRUPTED', notice='INTERRUPTED / DISPLAY RESTORED')
                self._status_message(f'Previous failed: {error}; rollback failed: {rollback_error}', 10000)
            else:
                self._status_message(str(error), 10000)
            return
        scan_id = session.previous()
        self._record_plan_placement_state('ACTIVE')
        self._update_plan_placement_ui()
        if before == 0:
            self._status_message('Already at the first chronological scan')
        else:
            self._status_message(f'PLAN PLACE PREVIOUS / {scan_id} / placement unlocked')

    def _commit_plan_placement_scan(self, scan_id: str, sequence_index: int) -> None:
        if self.project.fixed_scale != PLAN_PLACEMENT_SCALE:
            raise PlanPlacementError('project fixed scale must remain 1.0')
        scan = self.project.scan(scan_id)
        if scan is None:
            raise PlanPlacementError(f'unknown plan placement scan: {scan_id}')
        if scan.fixed_scale != PLAN_PLACEMENT_SCALE:
            raise PlanPlacementError(f'scan fixed scale must remain 1.0: {scan_id}')
        self.project.set_scan_placement_locked(scan_id, True)
        self.project.metadata = dict(self.project.metadata)
        self.project.metadata['initial_scan_layout'] = False
        committed = self.project.scan(scan_id)
        if committed is None:
            raise PlanPlacementError(f'unknown plan placement scan: {scan_id}')
        metadata = dict(committed.metadata)
        metadata['plan_placement'] = {'mode': 'MANUAL_DXF_SEQUENCE', 'plan_source': self.project.dxf_plan.source_name or 'DXF PLAN', 'sequence_index': int(sequence_index) + 1, 'fixed_scale': PLAN_PLACEMENT_SCALE, 'axis': 'CAD_Z_up', 'source_geometry': 'UNCHANGED', 'transform': [list(row) for row in committed.transform]}
        committed.metadata = metadata
        self.project.validate()

    def _close_plan_placement(self, visibility: Mapping[str, bool], *, status: str, notice: str) -> tuple[str, ...]:
        cleanup_errors: list[str] = []
        session = self.plan_placement_session
        placed_ids = set(session.placed_ids) if session is not None else set()
        if str(status).upper() != 'COMPLETE':
            for scan_id, originally_locked in self._plan_placement_original_locks.items():
                if scan_id in placed_ids:
                    continue
                scan = self.project.scan(scan_id)
                if scan is not None and scan.placement_locked != originally_locked:
                    try:
                        self.project.set_scan_placement_locked(scan_id, originally_locked)
                    except (OSError, RuntimeError, TypeError, ValueError) as error:
                        cleanup_errors.append(f'lock restore: {error}')
        try:
            self._restore_plan_placement_visibility(visibility)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            cleanup_errors.append(f'visibility restore: {error}')
        setter = getattr(self.viewport_adapter, 'set_transform_mode', None)
        if callable(setter):
            try:
                setter('SELECT')
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                cleanup_errors.append(f'select mode: {error}')
        try:
            self._restore_camera_state(self._plan_placement_camera_state)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            cleanup_errors.append(f'camera restore: {error}')
        self.rail.setEnabled(self._plan_placement_rail_enabled)
        self.tree.setEnabled(self._plan_placement_tree_enabled)
        try:
            self._record_plan_placement_state(status)
        except (RuntimeError, TypeError, ValueError) as error:
            cleanup_errors.append(f'audit state: {error}')
        self.plan_placement_session = None
        self._plan_placement_camera_state = None
        self._plan_placement_original_locks.clear()
        try:
            self._sync_transform_mode_ui(str(getattr(self.viewport_adapter, 'transform_mode', 'SELECT')), str(getattr(self.viewport_adapter, 'move_constraint', 'XY')))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            cleanup_errors.append(f'mode UI: {error}')
        self._populate_tree()
        self._populate_rail()
        final_notice = notice if not cleanup_errors else f'{notice} / VIEW RESTORE WARNING'
        self._update_plan_placement_ui(final_notice)
        self._plan_placement_skipped_scan_ids = ()
        return tuple(cleanup_errors)

    def _place_and_next_scan(self) -> None:
        session = self.plan_placement_session
        if session is None or session.phase is not PlacementPhase.ACTIVE:
            self._status_message('Run PLAN PLACE first')
            return
        scan_id = session.active_scan_id
        index = session.active_index
        if scan_id is None or index is None:
            self._status_message('Plan placement has no active scan', 10000)
            return
        advanced = False
        try:
            self._commit_plan_placement_scan(scan_id, index)
            advance = session.place_and_next(fixed_scale=PLAN_PLACEMENT_SCALE)
            advanced = True
            self._record_plan_placement_state('COMPLETE' if advance.completed else 'ACTIVE')
            if advance.completed:
                visibility = advance.restore_visibility or {}
                total = len(session.ordered_scan_ids)
                skipped = len(self._plan_placement_skipped_scan_ids)
                self._close_plan_placement(visibility, status='COMPLETE', notice=f'COMPLETE / {total:02d} SCANS PLACED + LOCKED / {skipped:02d} SKIPPED')
                self._status_message(f'PLAN PLACE COMPLETE / {total} SCANS / SCALE 1.000 / {skipped} SKIPPED / SOURCE UNCHANGED')
                return
            next_scan_id = advance.next_scan_id
            if next_scan_id is None:
                raise PlanPlacementError('next chronological scan is unavailable')
            self._activate_plan_placement_scan(next_scan_id)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            if advanced and session.phase is PlacementPhase.ACTIVE:
                result = session.cancel()
                self._close_plan_placement(result.restore_visibility, status='INTERRUPTED', notice='INTERRUPTED / DISPLAY RESTORED')
            self._status_message(str(error), 10000)
            return
        active_index = int(session.active_index or 0) + 1
        self._status_message(f'PLACED + LOCKED / {scan_id} / NEXT {active_index:02d}/{len(session.ordered_scan_ids):02d} / {next_scan_id}')

    def _finish_plan_placement(self) -> None:
        session = self.plan_placement_session
        if session is None or session.phase is not PlacementPhase.ACTIVE:
            self._status_message('Plan placement is not active')
            return
        active_scan_id = session.active_scan_id
        active_index = session.active_index
        try:
            if active_scan_id is not None and active_index is not None and (active_scan_id in session.placed_ids):
                self._commit_plan_placement_scan(active_scan_id, active_index)
            result = session.finish()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._status_message(str(error), 10000)
            return
        placed_count = len(result.placed_ids)
        self._close_plan_placement(result.restore_visibility, status='FINISHED_EARLY', notice=f'FINISHED / {placed_count:02d} SCANS PLACED + LOCKED')
        self._status_message(f'PLAN PLACE FINISHED / {placed_count} SCANS LOCKED / UNPLACED LOCK STATE RESTORED')

    def _reset_plan_placement_for_project_change(self) -> None:
        session = self.plan_placement_session
        if session is not None and session.phase is PlacementPhase.ACTIVE:
            try:
                session.cancel()
            except PlanPlacementError:
                pass
        self._restore_camera_state(self._plan_placement_camera_state)
        self.rail.setEnabled(self._plan_placement_rail_enabled)
        self.tree.setEnabled(self._plan_placement_tree_enabled)
        self.plan_placement_session = None
        self._plan_placement_camera_state = None
        self._plan_placement_original_locks.clear()
        self._plan_placement_skipped_scan_ids = ()

    def _viewport_scan_selected(self, scan_id: str) -> None:
        self._graph_selection_changed((scan_id,))
        self._sync_scan_graph()

    def _connect_ribbon_commands(self) -> None:
        unavailable_tip = 'Disabled in this build: this command has no verified runtime implementation yet.'
        for button in self.window.findChildren(self.qt.QtWidgets.QAbstractButton):
            command = button.property('ccaCommand')
            if isinstance(command, str):
                if command not in _ACTIVE_RIBBON_COMMANDS:
                    button.setEnabled(False)
                    button.setToolTip(unavailable_tip)
                button.clicked.connect(lambda _checked=False, value=command: self._run_ribbon_command(value))
        for action in self.window.findChildren(self.qt.QtGui.QAction):
            command = action.property('ccaCommand')
            if isinstance(command, str) and command not in _ACTIVE_RIBBON_COMMANDS:
                action.setEnabled(False)
                action.setToolTip(unavailable_tip)

    def _run_ribbon_command(self, command: str) -> None:
        if command == 'ALIGN:EDIT TRANSFORM':
            self._show_properties_panel('TRANSFORM')
            return
        if command == 'REVIEW:ZOOM EXTENTS':
            self._graph_fit()
            return
        if command == 'REVIEW:FIT SELECTED':
            ids = self._selected_scan_ids()
            if ids:
                self._graph_fit(ids)
            else:
                self._status_message('Kadraja almak icin bir tarama veya grup secin.')
            return
        if command in {'REVIEW:TOP', 'REVIEW:FRONT', 'REVIEW:RIGHT', 'REVIEW:PERSPECTIVE'}:
            setter = getattr(self.viewport_adapter, 'set_named_view', None)
            if callable(setter):
                setter(command.split(':', 1)[1].lower())
            return
        if command == 'PROJECT:NEW':
            self._create_new_project()
            return
        if command == 'PROJECT:OPEN':
            self._choose_project_file()
            return
        if command == 'PROJECT:SAVE':
            self._save_project_file()
            return
        if command == 'PROJECT:RECENT':
            self.content_stack.setCurrentWidget(self.project_home)
            self._status_message('RECENT PROJECTS / choose a project card')
            return
        if command == 'PROJECT:SETTINGS':
            action = self.window.findChild(self.qt.QtGui.QAction, 'SettingsAction')
            if action is not None:
                action.trigger()
            return
        if command == 'PROJECT:AUDIT':
            self._validate_project_contract('PROJECT AUDIT')
            return
        if command == 'PROJECT:ABOUT CCA':
            self.qt.QtWidgets.QMessageBox.about(self.window, PRODUCT_NAME, f'{PRODUCT_NAME}\n\n{ORGANIZATION_NAME}\n{CREATIVE_NAME}\n\nMetric point-cloud alignment for architectural survey.')
            return
        if command == 'PROJECT:HOME':
            self.content_stack.setCurrentWidget(self.project_home)
            return
        if command == 'IMPORT:PLY SCANS':
            self._choose_import_files()
            return
        if command == 'IMPORT:RAW ALL DATA':
            self._choose_raw_all_data_folder()
            return
        if command in {'IMPORT:APPLY CORRECTION', 'PROCESS:APPLY CORRECTION'}:
            self._apply_selected_raw_correction()
            return
        if command == 'IMPORT:VALIDATE':
            self._validate_project_contract('INPUT VALIDATION')
            return
        if command == 'IMPORT:UNIT / AXIS':
            self._status_message('UNIT / AXIS / METRE / CAD Z-UP / SCALE 1.000 LOCKED', 10000)
            return
        if command == 'IMPORT:DXF PLAN':
            self._choose_dxf_plan()
            return
        if command == 'REVIEW:PHOTO RGB':
            self.color_mode.setCurrentText('PHOTO RGB')
            return
        if command == 'REVIEW:CONFIDENCE':
            self.color_mode.setCurrentText('CONFIDENCE')
            return
        if command == 'REVIEW:RUN TINT':
            self.color_mode.setCurrentText('SCAN GROUP')
            return
        if command == 'REVIEW:HEIGHT':
            self.color_mode.setCurrentText('HEIGHT')
            return
        if command == 'REVIEW:PROCESSED FULL':
            self._show_properties_panel()
            self.evidence_profile.setFocus()
            self.evidence_profile.showPopup()
            return
        if command == 'REVIEW:POINT SIZE':
            self._show_properties_panel()
            self.point_size.setFocus()
            self.point_size.selectAll()
            return
        if command == 'REVIEW:SNAPSHOT':
            self._choose_snapshot_path()
            return
        if command == 'EXPORT:REVIEW PNG':
            self._choose_snapshot_path()
            return
        if command == 'EXPORT:REVIT PACK':
            self._run_revit_pack()
            return
        if command == 'EXPORT:SURVEY PACK':
            self._run_survey_pack()
            return
        if command in {'ALIGN:XYZ ALIGN', 'EXPORT:PROJECTION PLANES'}:
            self._open_rolove_panel()
            panel = getattr(self, 'rolove_panel', None)
            if panel is not None:
                panel.steps.setCurrentIndex(0 if command == 'ALIGN:XYZ ALIGN' else 1)
            return
        if command == 'REVIEW:SELECT':
            self._activate_transform_mode('SELECT')
            return
        if command == 'REVIEW:HIDE':
            self._hide_selected_scan()
            return
        if command == 'REVIEW:ISOLATE':
            self._isolate_selected_scan()
            return
        if command == 'REVIEW:SHOW ALL':
            self._show_all_scans()
            return
        if command == 'REVIEW:REMOVE':
            self._remove_selected_scan()
            return
        if command == 'REVIEW:UNDO REMOVE':
            self._undo_remove_scan()
            return
        if command == 'REVIEW:SCAN LAYOUT':
            self._apply_scan_layout()
            return
        if command == 'REVIEW:RESET LAYOUT':
            self._reset_scan_layout()
            return
        if command == 'ALIGN:SELECT':
            self._activate_transform_mode('SELECT')
            return
        if command == 'ALIGN:MOVE':
            self._activate_transform_mode('MOVE')
            return
        if command == 'ALIGN:ROTATE Z':
            self._activate_transform_mode('ROTATE Z')
            return
        if command == 'ALIGN:PLAN PLACE':
            self._start_plan_placement()
            return
        if command == 'ALIGN:PREVIOUS SCAN':
            self._previous_plan_scan()
            return
        if command == 'ALIGN:PLACE + NEXT':
            self._place_and_next_scan()
            return
        if command == 'ALIGN:FINISH':
            self._finish_plan_placement()
            return
        if command == 'ALIGN:POINT PAIRS':
            self._status_message('Select two scans in SCAN RAIL, then run AUTO ALIGN')
            return
        if command == 'ALIGN:FIND OVERLAP':
            self._search_rgb_alignment_candidates()
            return
        if command == 'ALIGN:AUTO ALIGN':
            self._show_properties_panel('ALIGN')
            self._start_auto_alignment()
            return
        if command == 'ALIGN:PAIR VARIATIONS':
            self._show_properties_panel('ALIGN')
            self._start_pair_variations()
            return
        if command == 'ALIGN:PLAN FIT':
            self._show_properties_panel('ALIGN')
            self._start_plan_fit()
            return
        if command == 'ALIGN:PREVIOUS':
            self._step_pair_variation(-1)
            return
        if command == 'ALIGN:NEXT':
            self._step_pair_variation(1)
            return
        if command == 'ALIGN:USE CANDIDATE':
            self._use_pair_candidate()
            return
        if command == 'ALIGN:POSE GRAPH':
            self._run_pose_graph()
            return
        if command == 'ALIGN:TIME CHAIN':
            self._show_temporal_chain_dialog()
            return
        if command == 'ALIGN:LOCK':
            self._toggle_scan_placement_lock()
            return
        if command == 'SURFACE:LOCAL BPA':
            self._build_surface_preview(SurfaceBackend.BPA)
            return
        if command == 'SURFACE:POISSON':
            self._build_surface_preview(SurfaceBackend.POISSON)
            return
        if command == 'SURFACE:TSDF STATUS':
            self._build_surface_preview(SurfaceBackend.TSDF)
            return
        if command in {'SURFACE:DENSITY', 'SURFACE:NORMALS'}:
            _LOGGER.warning('%s is not connected in this milestone', command)
            self._status_message(f'YUZEY YOK / {command} bu surumde bagli degil; once LOCAL BPA ile gorsel onizleme olusturun', 12000, priority=STATUS_PRIORITY_DEGRADED)
            return
        self._status_message(f'{command} is visible but not connected in this milestone')

    def _show_temporal_chain_dialog(self) -> None:
        if self._busy_job_kind is not None:
            self._status_message(f'{self._busy_job_kind} is already running; wait for completion', 5000)
            return
        sequence = self.project.metadata.get('temporal_sequence')
        if not isinstance(sequence, Mapping) or len(self.project.scan_rail) < 2:
            self._status_message('TIME CHAIN requires temporal sequence metadata and at least two scans', 8000)
            return
        existing = getattr(self, '_temporal_chain_dialog', None)
        if existing is not None and existing.dialog.isVisible():
            existing.dialog.raise_()
            existing.dialog.activateWindow()
            return
        dialog = TemporalChainDialog(self.qt, self.project, parent=self.window, on_saved=self._temporal_chain_policy_saved, on_rgb_search=self._search_rgb_alignment_candidates)
        self._temporal_chain_dialog = dialog
        dialog.dialog.finished.connect(lambda _result: setattr(self, '_temporal_chain_dialog', None))
        dialog.show()

    def _temporal_chain_policy_saved(self) -> None:
        self._discard_alignment_preview_for_edit()
        self._rgb_search_mode = 'ENDPOINT'
        self._status_message('TIME CHAIN SAVED / ALIGNMENT PREVIEW DISCARDED / PROJECT GEOMETRY UNCHANGED', 10000)

    def _search_rgb_alignment_candidates(self) -> None:
        from geometry.rgb_candidates import discover_rgb_candidates
        from viewer.alignment_scope import alignment_exclusions
        service = self._make_registration_service()
        sequence = service.temporal_sequence()
        if sequence is None:
            self._status_message('RGB aramasi icin ham tarama saatleri gerekli')
            return
        project = self.project
        scope = dict(alignment_exclusions(project))

        def completed(rows):
            if self.project is not project or dict(alignment_exclusions(project)) != scope:
                raise ValueError('project/scope changed while RGB search was running')
            from viewer.rgb_alignment_dialog import RGBAlignmentDialog
            self._rgb_candidate_dialog = RGBAlignmentDialog(self.qt, rows, parent=self.window, on_validate=self._validate_rgb_candidate)
            self._rgb_candidate_dialog.show()
        self._start_background_job('RGB SEARCH', 'RGB uc/ortak alan adaylari araniyor', lambda: discover_rgb_candidates(project.metadata['raw_data_root'], sequence, scope), completed, lambda error: self._status_message(str(error), 10000))

    def _validate_rgb_candidate(self, row) -> None:
        self._discard_alignment_preview_for_edit()
        self._rgb_search_mode = row['kind']
        self._graph_selection_active = False
        self._graph_selected_ids = ()
        self.alignment_workflow = None
        self.rail.clearSelection()
        ids = {row['source_scan_id'], row['target_scan_id']}
        for i in range(self.rail.count()):
            item = self.rail.item(i)
            item.setSelected(item.data(self.qt.QtCore.Qt.ItemDataRole.UserRole) in ids)
        self._start_pair_variations()

    def _validate_project_contract(self, label: str) -> None:
        try:
            self.project.validate()
            plan = self.layer_controller.render_plan(require_available=False) if self.layer_controller is not None else None
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._status_message(f'{label} FAILED / {error}', 12000)
            return
        available = len(plan.available_actors) if plan is not None else 0
        self._status_message(f'{label} PASS / {len(self.project.scans)} SCANS / {available} DISPLAY SOURCES / METRE / CAD Z-UP / SCALE 1.000', 12000)

    def _create_new_project(self) -> None:
        from viewer.welcome import request_new_project_name
        name, accepted = request_new_project_name(self.qt, self.window)
        if not accepted or not str(name).strip():
            return
        self.set_project(ProjectModel.new(str(name).strip()))
        self.window._cca_show_context('IMPORT')
        self._status_message('New fixed-scale project created')



    def _choose_raw_all_data_folder(self) -> None:
        folder = self.qt.QtWidgets.QFileDialog.getExistingDirectory(self.window, 'Add 3D Scanner App All Data folder', str(Path.home()))
        if not folder:
            return
        self._register_raw_all_data(Path(folder))

    def _register_raw_all_data(self, source: Path) -> None:
        from viewer.raw_processing_service import validate_all_data_folder
        project = self.project

        def completed(result: object) -> None:
            if self.project is not project:
                raise RuntimeError('project changed while raw data was validated')
            inventory = result
            required = ('run_id', 'run_dir', 'frame_count', 'depth_frame_count', 'confidence_frame_count', 'rgb_frame_count', 'eligible_rgb_frame_count')
            if not all((hasattr(inventory, name) for name in required)):
                raise RuntimeError('raw validation returned an invalid inventory')
            self._attach_raw_inventory(inventory)

        def failed(error: Exception) -> None:
            self.qt.QtWidgets.QMessageBox.critical(self.window, 'Raw All Data', str(error))
        self._start_background_job('RAW IMPORT', f'VALIDATING RAW DATA / {source.name}', lambda: validate_all_data_folder(source), completed, failed)

    def _attach_raw_inventory(self, inventory: object) -> None:
        run_id = str(inventory.run_id)
        run_dir = Path(inventory.run_dir).resolve()
        if not self.project.project_id:
            self.project = ProjectModel.new('Untitled Survey')
        scan = self.project.scan(run_id)
        raw_metadata = {'path': str(run_dir), 'source_axis': 'ARKit_Y_up', 'frame_count': int(inventory.frame_count), 'depth_frame_count': int(inventory.depth_frame_count), 'confidence_frame_count': int(inventory.confidence_frame_count), 'rgb_frame_count': int(inventory.rgb_frame_count), 'eligible_rgb_frame_count': int(inventory.eligible_rgb_frame_count), 'source_preserved': True, 'distance_cutoff_m': None}
        if scan is None:
            scan = ScanEntry(scan_id=run_id, rail_order=len(self.project.scans), source_path=str(run_dir), status='RAW_CAPTURE_READY', metadata={'raw_all_data': raw_metadata})
            self.project.import_scan(scan)
        else:
            scan.source_path = str(run_dir)
            scan.status = 'RAW_CAPTURE_READY'
            scan.metadata = dict(scan.metadata)
            scan.metadata['raw_all_data'] = raw_metadata
        self.set_project(self.project)
        self.select_scan(run_id)
        self._status_message(f"HAM VERI EKLENDI / {run_id} / {raw_metadata['eligible_rgb_frame_count']:,} RGB kare / CCA donusumunu hesapla", 12000)

    def _selected_raw_source(self) -> Path | None:
        item = self.rail.currentItem()
        scan_id = item.data(self.qt.QtCore.Qt.ItemDataRole.UserRole) if item is not None else None
        scan = self.project.scan(scan_id) if isinstance(scan_id, str) else None
        if scan is None:
            return None
        metadata = scan.metadata.get('raw_all_data', {})
        candidates = []
        if isinstance(metadata, Mapping):
            candidates.append(metadata.get('path'))
        candidates.append(scan.source_path)
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate).expanduser().resolve()
            if path.is_dir():
                return path
        return None

    def _choose_depth_model(self) -> None:
        source = self._selected_raw_source()
        scan = self._selected_scan_for_inspection()
        if source is None or scan is None or self._busy_job_kind is not None:
            self._status_message('Once ham verisi olan bir tarama secin.')
            return
        path, _ = self.qt.QtWidgets.QFileDialog.getOpenFileName(self.window, 'Deneysel menzil modeli sec', str(Path('data/depth_correction') / source.name), 'Depth model (*.json)')
        if not path:
            return
        from viewer.correction_models import load_run_model
        try:
            load_run_model(path, source.name)
        except (OSError, ValueError) as error:
            self.qt.QtWidgets.QMessageBox.warning(self.window, 'Menzil modeli', str(error))
            return
        scan.metadata['depth_correction_model'] = str(Path(path).resolve())
        self._sync_depth_model_status()
        self._status_message('DENEYSEL MODEL SECILDI / Uygulamak icin CCA donusumunu hesapla', 10000)

    def _clear_depth_model(self) -> None:
        scan = self._selected_scan_for_inspection()
        if scan is not None and self._busy_job_kind is None:
            scan.metadata.pop('depth_correction_model', None)
            self._status_message('SONRAKI DONUSUMDE MENZIL MODELI KAPALI / Mevcut bulut degismedi', 10000)
        self._sync_depth_model_status()

    def _sync_depth_model_status(self) -> None:
        from viewer.correction_models import load_run_model, model_status
        scan = self._selected_scan_for_inspection()
        source = self._selected_raw_source()
        path = scan.metadata.get('depth_correction_model') if scan is not None else None
        next_text = model_status(None)
        if path and source is not None:
            try:
                model, _ = load_run_model(path, source.name)
                next_text = model_status(model)
            except (OSError, ValueError) as error:
                next_text = f'MODEL HATASI / {error}'
        request = next((actor for actor in self.layer_controller.render_plan().visible_actors if scan is not None and actor.scan_id == scan.scan_id), None) if self.layer_controller is not None else None
        layer = scan.layer(request.source_layer_key) if scan is not None and request else None
        applied = layer.metadata.get('depth_correction') if layer is not None else None
        last_generated = scan.metadata.get('raw_correction', {}).get('depth_correction') if scan is not None else None
        if request is None:
            view_text = 'GORUNUM: SECILI TARAMA GORUNUR DEGIL'
        elif request.source_layer_key == 'rgb/corrected':
            kind = applied.get('kind') if isinstance(applied, dict) else 'KAYIT YOK'
            view_text = f'GORUNUM: DUZELTILMIS / {kind}'
        elif request.source_layer_key == 'rgb/raw':
            view_text = 'GORUNUM: HAM / menzil modeli uygulanmadi'
        else:
            view_text = f'GORUNUM: {evidence_label(self.layer_controller.evidence_profile)}'
        text = view_text + '\n' + next_text
        self.depth_model_status.setText(text)
        self.depth_model_status.setToolTip(f"{text}\nEtkin kaynak: {(request.source_layer_key if request else 'yok')}\nSonraki donusum secimi: {path or 'Model secilmedi'}\nBu secim mevcut bulutu degistirmez. CCA donusumu yeniden hesaplanmalidir.\nEtkin bulutun menzil modeli: {applied or 'Kayitli menzil modeli yok'}\nSon uretilen ham-kare sonucunun modeli: {last_generated or 'Kayitli menzil modeli yok'}")
        self.depth_model_load.setEnabled(source is not None and self._busy_job_kind is None)
        self.depth_model_clear.setEnabled(bool(path) and self._busy_job_kind is None)

    def _show_correction_formulas(self) -> None:
        from viewer.correction_models import FORMULA_TEXT
        widgets = self.qt.QtWidgets
        dialog = widgets.QDialog(self.window)
        dialog.setObjectName('CorrectionFormulasDialog')
        dialog.setWindowTitle(f'{PRODUCT_NAME} / Formuller')
        dialog.resize(800, 700)
        layout = widgets.QVBoxLayout(dialog)
        text = widgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(FORMULA_TEXT + '\nSECILI MODEL VE URETIM KAYDI\n' + self.depth_model_status.toolTip())
        layout.addWidget(text)
        close = widgets.QPushButton('Kapat')
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec()

    def _apply_selected_raw_correction(self) -> None:
        source = self._selected_raw_source()
        if source is None:
            self._status_message('CCA donusumu icin once ham verisi hazir bir tarama secin.', 10000)
            return
        from viewer.raw_processing_service import process_raw_all_data
        project = self.project
        scan = self._selected_scan_for_inspection()
        model_path = scan.metadata.get('depth_correction_model') if scan is not None else None
        options = {'depth_correction': model_path} if model_path else {}
        previous = scan.metadata.get('raw_correction', {}) if scan is not None else {}
        if previous.get('ray_recovery'):
            options['ray_recovery'] = Path(previous['manifest']).parent / 'ray-recovery.json'

        def completed(result: object) -> None:
            if self.project is not project:
                raise RuntimeError('project changed while correction was running')
            self._attach_raw_processing_result(result, source)

        def failed(error: Exception) -> None:
            self.qt.QtWidgets.QMessageBox.critical(self.window, 'CCA donusumu', str(error))
        self._start_background_job('APPLY CORRECTION', f'CCA DONUSUMU HESAPLANIYOR / {source.name} / 5 CM / CAD Z-UP', lambda: process_raw_all_data(source, **options), completed, failed)

    def _recover_selected_raw_points(self) -> None:
        source = self._selected_raw_source()
        scan = self._selected_scan_for_inspection()
        if source is None or scan is None or self._busy_job_kind is not None:
            self._status_message('Konum aramasi icin ham tarama secin')
            return
        project = self.project
        model = scan.metadata.get('depth_correction_model')

        def work():
            from viewer.raw_processing_service import process_raw_all_data
            from viewer.ray_recovery_service import build_recovery
            path = ROOT / 'data/depth_recovery' / source.name / 'ray-recovery.json'
            build_recovery(source, path)
            return process_raw_all_data(source, depth_correction=model, ray_recovery=path)

        def completed(result):
            if self.project is not project:
                raise ValueError('project changed while ray recovery was running')
            self._attach_raw_processing_result(result, source)
            count = (result.ray_recovery or {}).get('relocated_pixels', 0)
            self._status_message(f'RGB konum aramasi / {count} ham olcum yeniden konumlandi', 15000)
        self._start_background_job('RAY RECOVERY', 'Ham nokta konumlari RGB goruslerinde araniyor', work, completed, lambda error: self._status_message(str(error), 12000))

    def _search_raw_structure(self, *, lefebvre=False) -> None:
        source = self._selected_raw_source()
        if source is None or self._busy_job_kind is not None:
            self._status_message('Yapi/isin aramasi icin ham tarama secin')
            return
        project = self.project

        def work():
            from viewer.structure_recovery_service import build_structure_search
            return build_structure_search(source, ROOT / 'data/depth_recovery' / source.name / 'structure-search.json')

        def completed(report):
            if self.project is not project:
                raise ValueError('project changed while structure search was running')
            if lefebvre:
                from viewer.lefebvre_dialog import make_lefebvre_dialog
                self._lefebvre_dialog = make_lefebvre_dialog(self.window, report)
                self._lefebvre_dialog.show()
                self._status_message('LEFEBVRE / Patchworking deneyi / RGB + Patch Color / ham proje korunuyor', 15000)
                return
            from viewer.structure_dialog import make_structure_dialog
            self._structure_dialog = make_structure_dialog(self.window, report)
            self._structure_dialog.show()
            self._status_message(f"Yapi/isin / {report['counts'].get('reviewable', 0)} yer adayi / ham noktalar degismedi", 15000)
        self._start_background_job('STRUCTURE SEARCH', 'Yerel yapi ve kamera isinlari karsilastiriliyor', work, completed, lambda error: self._status_message(str(error), 12000))

    def _audit_selected_point_usage(self) -> None:
        scan = self._selected_scan_for_inspection()
        if scan is None or self._busy_job_kind is not None:
            self._status_message('Nokta hesabi icin bir tarama secin')
            return
        layers = sorted(scan.layers, key=lambda layer: (layer.key != 'rgb/raw', layer.key != 'rgb/full'))
        source = next((Path(layer.path) for layer in layers if Path(layer.path).suffix.lower() == '.ply' and Path(layer.path).is_file()), None)
        if source is None:
            self._status_message('Bu taramada okunabilir PLY yok; once CCA donusumunu hesaplayin', 12000)
            return
        project = self.project

        def work():
            from viewer.point_usage_service import audit_point_usage
            return audit_point_usage(source, ROOT / 'data/point_usage' / scan.scan_id / 'point-usage.json')

        def completed(report):
            if self.project is not project:
                raise ValueError('project changed while point audit was running')
            from viewer.point_usage_service import make_point_usage_dialog
            self._point_usage_dialog = make_point_usage_dialog(self.window, report)
            self._point_usage_dialog.show()
            self._status_message(f"{report['accounted_vertices']:,} noktanin hesabi verildi / 0 silindi", 15000)
        self._start_background_job('POINT USAGE', 'Tum noktalarin yerel geometri destegi hesaplaniyor', work, completed, lambda error: self._status_message(str(error), 12000))

    def _attach_raw_processing_result(self, result: object, source: Path) -> None:
        required = ('run_id', 'rgb_zup_path', 'confidence_zup_path', 'manifest_path', 'point_count', 'used_frame_count', 'source_preserved')
        if not all((hasattr(result, name) for name in required)):
            raise RuntimeError('raw correction returned an invalid result')
        if result.source_preserved is not True:
            raise RuntimeError('raw correction did not preserve its source')
        previous_controller = self.layer_controller
        if previous_controller is not None and previous_controller.project is not self.project:
            previous_controller = None
        previous_color = self.color_mode.currentText()
        previously_loaded = previous_controller is not None and getattr(self.viewport_adapter, 'last_load_result', None) is not None
        run_id = str(result.run_id)
        point_count = int(result.point_count)
        scan = self.project.scan(run_id)
        if scan is None:
            scan = ScanEntry(scan_id=run_id, rail_order=len(self.project.scans), source_path=str(source), status='RAW_CAPTURE_READY')
            self.project.import_scan(scan)
        depth_correction = getattr(result, 'depth_correction', None)
        family = 'corrected' if isinstance(depth_correction, Mapping) else 'raw'
        layer_keys = ('rgb/corrected', 'confcolor/corrected') if family == 'corrected' else ('rgb/raw', 'confcolor/raw')
        layer_roles = ('CORRECTED_RAW_RGB', 'CORRECTED_RAW_CONFIDENCE') if family == 'corrected' else ('RAW_FRAME_RGB_EVIDENCE', 'RAW_FRAME_CONFIDENCE_EVIDENCE')
        retained = [layer for layer in scan.layers if layer.key not in layer_keys]
        evidence_metadata = {'source_axis': 'CAD_Z_up', 'source_unit': 'metre', 'source_preserved': True, 'distance_cutoff_m': None, 'processing_manifest': str(Path(result.manifest_path)), 'depth_correction': depth_correction, 'ray_recovery': getattr(result, 'ray_recovery', None), 'evidence_family': family}
        retained.extend((ScanLayer(key=layer_keys[0], path=str(Path(result.rgb_zup_path).resolve()), role=layer_roles[0], point_count=point_count, visible=True, metadata=evidence_metadata), ScanLayer(key=layer_keys[1], path=str(Path(result.confidence_zup_path).resolve()), role=layer_roles[1], point_count=point_count, visible=True, metadata=evidence_metadata)))
        scan.layers = retained
        scan.source_path = str(source.resolve())
        scan.status = 'CORRECTION_READY'
        scan.metadata = dict(scan.metadata)
        scan.metadata['raw_correction'] = {'manifest': str(Path(result.manifest_path).resolve()), 'point_count': point_count, 'used_frame_count': int(result.used_frame_count), 'voxel_m': 0.05, 'source_preserved': True, 'distance_cutoff_m': None, 'output_axis': 'CAD_Z_up', 'depth_correction': getattr(result, 'depth_correction', None), 'ray_recovery': getattr(result, 'ray_recovery', None)}
        self.project.metadata = dict(self.project.metadata)
        manifests = list(self.project.metadata.get('raw_processing_manifests', []))
        manifest_text = str(Path(result.manifest_path).resolve())
        if manifest_text not in manifests:
            manifests.append(manifest_text)
        self.project.metadata['raw_processing_manifests'] = manifests
        profile = EvidenceProfile.CORRECTED_RGB if family == 'corrected' else EvidenceProfile.RAW_RGB
        if previous_controller is not None:
            previous_plan = previous_controller.render_plan(require_available=False)
            try:
                if previous_plan.available_actors:
                    previous_plan.require_available()
                    profile = previous_controller.evidence_profile
            except LayerControllerError:
                pass
        self.set_project(self.project, render=False, evidence_profile=profile)
        if previous_controller is not None:
            for scan_id, visible in previous_controller.scan_visibility.items():
                self.layer_controller.set_scan_visible(scan_id, visible)
            for band, visible in previous_controller.band_visibility.items():
                self.layer_controller.set_band_visible(band, visible)
            self.layer_controller.set_display_mode(previous_controller.display_mode)
            self.color_mode.blockSignals(True)
            self.color_mode.setCurrentText(previous_color)
            self.color_mode.blockSignals(False)
        self._apply_display_plan(self.layer_controller.render_plan(require_available=True), reset_project=not previously_loaded, on_ready=lambda: self.select_scan(run_id))
        self.select_scan(run_id)
        self._status_message(f'CCA DONUSUMU HAZIR / {run_id} / {point_count:,} PTS / SENSOR YONU + KAMERA OLCEGI', 15000)

    def _open_rolove_panel(self) -> None:
        scan = self._selected_scan_for_inspection()
        if scan is None:
            self._status_message('Once soldaki listeden bir tarama secin.', 10000)
            return
        if self._busy_job_kind is not None:
            self._status_message('Devam eden islemin bitmesini bekleyin.', 10000)
            return
        is_detailed = getattr(self.viewport_adapter, 'is_scan_detailed', None)
        if callable(is_detailed) and (not is_detailed(scan.scan_id)):
            self._status_message("Once Hesapla'ya basin. Yuzey secimi asil noktalari kullanir.", 10000)
            return
        from viewer.rolove_panel import RolovePanel
        previous = getattr(self, 'rolove_panel', None)
        if previous is not None and previous.scan is scan:
            previous.dock.show()
            previous.dock.raise_()
            return
        if previous is not None:
            previous.close()
            self.rolove_panel = None
        try:
            self.rolove_panel = RolovePanel(self, scan)
        except (ValueError, RuntimeError) as error:
            self._status_message(f'ROLOVE / {error}', 15000)

    def _run_revit_pack(self) -> None:
        from export.revit_pack import export_revit_pack, resolve_source
        if not self.project.scans:
            self._status_message('REVIT / Once tarama yukleyin', 10000)
            return
        if self._busy_job_kind is not None:
            self._status_message('REVIT / Mevcut is bitince yeniden deneyin', 10000)
            return
        widgets = self.qt.QtWidgets
        selected = self._selected_scan_ids()
        options = [f'Tum taramalar ({len(self.project.scans)})']
        if selected:
            options.insert(0, f'Secili taramalar ({len(selected)})')
        choice, accepted = widgets.QInputDialog.getItem(self.window, 'Revit Paketi', 'Ayri E57 dosyalari; birlestirme veya Revit hizasi yapilmaz.', options, 0, False)
        if not accepted:
            return
        scan_ids = selected if choice.startswith('Secili') else None
        snapshot = self.project.to_dict()
        try:
            for scan in snapshot['scan_rail']:
                if scan_ids is None or scan['scan_id'] in scan_ids:
                    resolve_source(scan, ROOT)
        except (OSError, ValueError, KeyError) as error:
            self._status_message(f'REVIT / {error}', 15000)
            return
        folder = widgets.QFileDialog.getExistingDirectory(self.window, 'Revit paketi ust klasoru', str(ROOT / 'data' / 'exports'))
        if not folder:
            return
        output = Path(folder) / f'CCA-Lefebvre-Revit-{uuid.uuid4().hex[:10]}'

        def completed(result: object) -> None:
            self._last_revit_pack = result
            self._status_message(f"E57 HAZIR / {result['point_count']:,} nokta / {result['output_dir']}", 30000)
            widgets.QMessageBox.information(self.window, 'Revit Paketi', f"{result['output_dir']}\n\nE57 ve kaynak alanlari dogrulandi.\nReCap'te RCP/RCS indexleme gerekir. Revit hizasi henuz yok.\nSensor dolgusu ayri; taramalar kendiliginden birlestirilmedi.\nBASLA.txt dosyasini acin.")
        self._start_background_job('REVIT PACK', 'REVIT / E57 + TAM KAYNAK + KOORDINAT KAYDI', lambda notify: export_revit_pack(snapshot, output, base=ROOT, scan_ids=scan_ids, progress=notify), completed, lambda error: widgets.QMessageBox.critical(self.window, 'Revit Paketi', str(error)), reports_progress=True)

    def _run_survey_pack(self) -> None:
        scan = self._selected_scan_for_inspection()
        if scan is None:
            self._status_message('Select a scan with a visible PLY layer before ROLOVE PACK', 10000)
            return
        from viewer.survey_service import SurveyServiceError, resolve_survey_source, run_survey_pack
        try:
            resolve_survey_source(scan)
        except SurveyServiceError as error:
            self._status_message(f'ROLOVE PACK / {error}', 10000)
            return
        project = self.project
        scan_id = scan.scan_id
        project_name = project.name or scan_id

        def completed(result: object) -> None:
            if self.project is not project:
                raise RuntimeError('project changed while survey pack was running')
            self._attach_survey_pack_result(result, scan_id)

        def failed(error: Exception) -> None:
            self.qt.QtWidgets.QMessageBox.critical(self.window, 'Rolove Plan + Gorunus', str(error))
        self._start_background_job('ROLOVE PACK', f'ROLOVE PACK / {scan_id} / PLAN + GORUNUS', lambda: run_survey_pack(scan, project_name=project_name), completed, failed)

    def _attach_survey_pack_result(self, result: object, scan_id: str) -> None:
        required = ('manifest_path', 'output_dir', 'wall_count', 'structure_count', 'low_confidence_count')
        if not all((hasattr(result, name) for name in required)):
            raise RuntimeError('survey pack returned an invalid result')
        scan = self.project.scan(scan_id)
        if scan is None:
            raise RuntimeError(f'scan {scan_id} left the project during survey pack')
        manifest = Path(result.manifest_path).resolve()
        output_dir = Path(result.output_dir).resolve()
        if not manifest.is_file():
            raise RuntimeError(f'survey pack manifest is missing: {manifest}')
        walls = int(result.wall_count)
        structures = int(result.structure_count)
        low_confidence = int(result.low_confidence_count)
        scan.metadata = dict(scan.metadata)
        scan.metadata['survey_pack'] = {'manifest': str(manifest), 'output_dir': str(output_dir), 'walls': walls, 'structures': structures, 'low_confidence': low_confidence}
        self.project.metadata = dict(self.project.metadata)
        manifests = list(self.project.metadata.get('survey_pack_manifests', []))
        if str(manifest) not in manifests:
            manifests.append(str(manifest))
        self.project.metadata['survey_pack_manifests'] = manifests
        self.set_project(self.project, evidence_profile=self.evidence_profile.currentData())
        self.select_scan(scan_id)
        self._status_message(f'SURVEY PACK READY / {walls} walls / {structures} structures / {low_confidence} ELLE OLC', 15000)
        self._offer_survey_pack_folder(output_dir)

    def _offer_survey_pack_folder(self, output_dir: Path) -> bool:
        message_box = self.qt.QtWidgets.QMessageBox
        buttons = message_box.StandardButton
        answer = message_box.question(self.window, 'Rolove Plan + Gorunus', f'SURVEY DRAFT / UNCALIBRATED written to:\n{output_dir}\n\nOpen the folder?', buttons.Yes | buttons.No, buttons.No)
        if answer != buttons.Yes:
            return False
        url = self.qt.QtCore.QUrl.fromLocalFile(str(output_dir))
        return bool(self.qt.QtGui.QDesktopServices.openUrl(url))

    def _choose_project_file(self) -> None:
        path, _selected_filter = self.qt.QtWidgets.QFileDialog.getOpenFileName(self.window, 'Open nokta project', '', 'nokta project (*.ccapoints.json *.json)')
        if not path:
            return
        try:
            self.set_project(load_desktop_project(path))
        except (OSError, RuntimeError, ValueError) as error:
            self.qt.QtWidgets.QMessageBox.critical(self.window, PRODUCT_NAME, str(error))

    def _save_project_file(self) -> None:
        default_name = f"{(self.project.name or 'project').replace(' ', '-')}.ccapoints.json"
        path, _selected_filter = self.qt.QtWidgets.QFileDialog.getSaveFileName(self.window, 'Save nokta project', default_name, 'nokta project (*.ccapoints.json)')
        if not path:
            return
        try:
            saved = self.project.save(path)
        except (OSError, RuntimeError, ValueError) as error:
            self.qt.QtWidgets.QMessageBox.critical(self.window, PRODUCT_NAME, str(error))
            return
        self._status_message(f'Saved {saved.name}')

    def _choose_snapshot_path(self) -> None:
        path, _selected_filter = self.qt.QtWidgets.QFileDialog.getSaveFileName(self.window, 'Save nokta review image', 'cca-points-review.png', 'PNG image (*.png)')
        if not path:
            return
        try:
            self.capture_png(Path(path))
        except (OSError, RuntimeError, ValueError) as error:
            self.qt.QtWidgets.QMessageBox.critical(self.window, PRODUCT_NAME, str(error))

    def capture_png(self, path: Path) -> Path:
        output = path.resolve()
        if output.suffix.lower() != '.png':
            output = output.with_suffix('.png')
        output.parent.mkdir(parents=True, exist_ok=True)
        plotter = getattr(self.viewport_adapter, 'plotter', None)
        render = getattr(plotter, 'render', None)
        if callable(render):
            render()
        app = self.qt.QtWidgets.QApplication.instance()
        if app is not None:
            app.processEvents()
        handle = self.window.windowHandle()
        screen = handle.screen() if handle is not None else None
        image = screen.grabWindow(int(self.window.winId())) if screen is not None else self.window.grab()
        if not image.save(str(output), 'PNG'):
            raise RuntimeError('could not save CCA Points review image')
        self._status_message(f'Saved {output.name}')
        return output

    @staticmethod
    def _initial_evidence_profile(project: ProjectModel) -> EvidenceProfile:
        return resolve_initial_evidence_profile(project)

    def set_project(self, project: ProjectModel, *, render: bool=True, evidence_profile: EvidenceProfile | str | None=None, preserve_edit_history: bool=False) -> None:
        rolove = getattr(self, 'rolove_panel', None)
        if rolove is not None:
            rolove.close()
            self.rolove_panel = None
        self._reset_plan_placement_for_project_change()
        project.validate()
        self.project = project
        self._restore_quality_settings()
        self._restore_evidence_grade_settings()
        self._restore_sky_fill_settings()
        self._restore_floating_settings()
        self._restore_sparse_settings()
        self._graph_selection_active = False
        self._graph_selected_ids = ()
        self._group_drag_originals = {}
        self._reset_alignment_controls(clear_undo=not preserve_edit_history)
        initial_evidence = evidence_profile or self._initial_evidence_profile(project)
        self.layer_controller = LayerController(project, display_mode=DisplayMode.PHOTO_RGB, evidence_profile=initial_evidence, point_size=self.point_size.value())
        self.surface_controller = SurfaceController(project)
        configured_bands = project.metadata.get('initial_band_visibility')
        for band in ('core', 'body', 'shell'):
            configured_visible = configured_bands.get(band) if isinstance(configured_bands, Mapping) else None
            initially_visible = configured_visible if type(configured_visible) is bool else any((layer.visible for scan in project.scan_rail if scan.visible for layer in (scan.layer(f'rgb/{band}'), scan.layer(f'confcolor/{band}')) if layer is not None))
            self.layer_controller.set_band_visible(band, initially_visible)
        self.color_mode.blockSignals(True)
        self.color_mode.setCurrentText('PHOTO RGB')
        self.color_mode.blockSignals(False)
        self.evidence_profile.blockSignals(True)
        self.evidence_profile.setCurrentText(self.layer_controller.evidence_profile.value)
        self.evidence_profile.blockSignals(False)
        self._restore_surface_settings()
        if project.project_id or project.name or project.scans:
            self.content_stack.setCurrentWidget(self.workspace_page)
        else:
            self.content_stack.setCurrentWidget(self.project_home)
        tab_name = {'IMPORT': 'IMPORT' if project.scans else 'PROJECT', 'VALIDATE': 'IMPORT', 'REVIEW': 'REVIEW', 'ALIGN': 'ALIGN', 'QC': 'REVIEW', 'DXF FIT': 'TRANSFORM', 'SURFACE': 'SURFACE', 'EXPORT': 'EXPORT'}[project.pipeline.current.value]
        tabs = getattr(self.window, '_cca_points_tabs', None)
        if tabs is not None:
            tab_index = next((index for index in range(tabs.count()) if (tabs.tabData(index) or tabs.tabText(index)) == tab_name), -1)
            if tab_index >= 0:
                tabs.setCurrentIndex(tab_index)
        self._populate_tree()
        self._populate_rail()
        self._update_status()
        self._update_scan_pulse()
        self._clear_properties()
        if self.project.scans:
            preferred = next((scan for scan in self.project.scans if scan.visible), self.project.scans[0])
            self.rail.setCurrentRow(preferred.rail_order)
            self.select_scan(preferred.scan_id)
        if render:
            self._render_project(self._apply_scan_layout if _should_apply_initial_scan_layout(project) else None)
        self._update_plan_placement_ui()
        self._sync_scan_graph()
        self._publish_mcp_state()

    def _publish_mcp_state(self) -> None:
        controller = self.layer_controller
        if controller is None:
            return
        plan = controller.render_plan(require_available=False)
        visible = {item.scan_id: item for item in plan.visible_actors}
        unavailable = {item.scan_id: item for item in plan.requested_unavailable}
        selected = self._selected_scan_for_inspection()
        payload = {'project': {'id': self.project.project_id, 'name': self.project.name, 'scan_count': len(self.project.scans)}, 'view': {'source_profile': controller.evidence_profile.value, 'color_mode': controller.display_mode.value, 'selected_scan_id': selected.scan_id if selected is not None else None}, 'counts': {'stored_visible': sum(controller.scan_visibility.values()), 'drawn_scans': len(visible), 'requested_missing': len(plan.requested_unavailable)}, 'scans': [{'scan_id': scan.scan_id, 'group_id': scan.group_id, 'status': scan.status, 'visible': controller.scan_visibility.get(scan.scan_id, False), 'drawn': scan.scan_id in visible, 'source_layer': visible[scan.scan_id].source_layer_key if scan.scan_id in visible else None, 'unavailable_reason': unavailable[scan.scan_id].unavailable_reason if scan.scan_id in unavailable else None} for scan in self.project.scans]}
        try:
            publish_mcp_state(payload)
        except OSError as error:
            _LOGGER.warning('Lefebvre MCP state was not published: %s', error)

    def _process_mcp_commands(self) -> None:
        if self._shutting_down or self.layer_controller is None:
            return
        commands = take_mcp_commands()
        if not commands:
            return
        for command in commands:
            payload = command['payload']
            action = command['action']
            if action == 'select_scan':
                self.select_scan(str(payload['scan_id']))
            elif action == 'set_source':
                self.evidence_profile.setCurrentText(str(payload['profile']))
            elif action == 'set_visibility':
                scan_ids = tuple((str(item) for item in payload['scan_ids']))
                previous = self.layer_controller.scan_visibility
                for scan_id in scan_ids:
                    self.layer_controller.set_scan_visible(scan_id, bool(payload['visible']))
                try:
                    plan = self.layer_controller.render_plan(require_available=True)
                except DisplayModeUnavailableError:
                    for scan_id, was_visible in previous.items():
                        self.layer_controller.set_scan_visible(scan_id, was_visible)
                else:
                    self._apply_display_plan(plan)
                    self._sync_scan_graph()
        self._publish_mcp_state()

    def _populate_tree(self) -> None:
        widgets = self.qt.QtWidgets
        core = self.qt.QtCore
        self.tree.clear()
        root_name = (self.project.name or 'Untitled Project').upper()
        root = widgets.QTreeWidgetItem([root_name])
        root.setData(0, core.Qt.ItemDataRole.UserRole, ('project', self.project.project_id))
        plan_group = widgets.QTreeWidgetItem(['DXF PLAN'])
        if self.project.dxf_plan.loaded:
            source = self.project.dxf_plan.source_name or 'PLAN'
            unit = (self.project.dxf_plan.source_unit or 'unknown').upper()
            item = widgets.QTreeWidgetItem([f'{source}  /  {unit} -> M'])
            item.setData(0, core.Qt.ItemDataRole.UserRole, ('plan', source))
            plan_group.addChild(item)
        else:
            plan_group.addChild(widgets.QTreeWidgetItem(['No plan loaded']))
        root.addChild(plan_group)
        scans_group = widgets.QTreeWidgetItem([f'SCANS  ({len(self.project.scans)})'])
        for group in self.project.groups:
            group_item = widgets.QTreeWidgetItem([f'{group.group_id}  ({len(group.scan_ids)})'])
            for scan_id in group.scan_ids:
                scan = self.project.scan(scan_id)
                if scan is None:
                    continue
                removed = self._scan_is_removed(scan)
                label = f'TARAMA {scan.rail_order + 1:02d}  /  {scan.scan_id[-8:]}'
                title = f'{label}  /  REMOVED  /  UNDO' if removed else f'{label}  /  LOCKED' if scan.placement_locked else label
                scan_item = widgets.QTreeWidgetItem([title])
                scan_item.setData(0, core.Qt.ItemDataRole.UserRole, ('scan', scan.scan_id))
                for layer in scan.layers:
                    layer_item = widgets.QTreeWidgetItem([f'{layer.key}  /  {layer.point_count or 0:,} pts'])
                    layer_item.setData(0, core.Qt.ItemDataRole.UserRole, ('layer', scan.scan_id, layer.key))
                    scan_item.addChild(layer_item)
                group_item.addChild(scan_item)
            scans_group.addChild(group_item)
        root.addChild(scans_group)
        for title, description in (('ALIGNMENTS', 'Registration graph'), ('SURFACES', 'Visual previews'), ('MEASUREMENTS', 'Calibration-gated results'), ('EXPORTS', 'Audit packages')):
            group_item = widgets.QTreeWidgetItem([title])
            group_item.addChild(widgets.QTreeWidgetItem([description]))
            root.addChild(group_item)
        self.tree.addTopLevelItem(root)
        root.setExpanded(True)
        plan_group.setExpanded(self.project.dxf_plan.loaded)
        scans_group.setExpanded(True)

    def _populate_rail(self) -> None:
        widgets = self.qt.QtWidgets
        core = self.qt.QtCore
        self.rail.clear()
        self.rail.setIconSize(core.QSize(84, 42))
        temporal = self.project.metadata.get('temporal_sequence', {})
        pair_rows = temporal.get('adjacent_pairs', []) if isinstance(temporal, Mapping) else []
        incoming: dict[str, Mapping[str, object]] = {str(row.get('target_scan_id')): row for row in pair_rows if isinstance(row, Mapping)}
        for index, scan in enumerate(self.project.scan_rail, start=1):
            short_id = scan.scan_id[-8:] if len(scan.scan_id) > 8 else scan.scan_id
            pair = incoming.get(scan.scan_id)
            gap = 'START'
            if pair is not None:
                gap_value = float(pair.get('gap_s', 0.0))
                gap = f'+{gap_value:.0f}s'
                if pair.get('long_gap') is True:
                    gap += '  LONG GAP'
            label = f"TARAMA {index:02d}  /  {short_id}  /  {gap}\n{scan.group_id}  /  {self._short_status(scan.status)}{('  /  LOCKED' if scan.placement_locked else '')}"
            item = widgets.QListWidgetItem(label)
            item.setData(core.Qt.ItemDataRole.UserRole, scan.scan_id)
            item.setSizeHint(core.QSize(152, 50))
            self.rail.addItem(item)
        if not self.project.scans:
            placeholder = widgets.QListWidgetItem('NO SCANS  /  USE IMPORT > PLY SCANS')
            placeholder.setFlags(placeholder.flags() & ~core.Qt.ItemFlag.ItemIsSelectable)
            self.rail.addItem(placeholder)

    @staticmethod
    def _short_status(status: str) -> str:
        normalized = status.upper()
        if 'REJECT' in normalized or 'BROKEN' in normalized:
            return 'REJECTED'
        if 'REVIEW' in normalized or 'UNPLACED' in normalized:
            return 'REVIEW'
        if 'ACCEPT' in normalized or 'ALIGNED' in normalized:
            return 'ACCEPTED'
        return normalized[:12] or 'IMPORTED'

    @staticmethod
    def _scan_is_removed(scan: ScanEntry) -> bool:
        return scan.status.strip().upper() == 'REMOVED'

    def _selected_scan_for_inspection(self) -> ScanEntry | None:
        if getattr(self, '_graph_selection_active', False) and (not self._graph_selected_ids):
            return None
        if self._property_target is not None and self._property_target[0] == 'scan':
            return self.project.scan(self._property_target[1])
        current = self.rail.currentItem()
        if current is None:
            return None
        scan_id = current.data(self.qt.QtCore.Qt.ItemDataRole.UserRole)
        return self.project.scan(scan_id) if isinstance(scan_id, str) else None

    def _selected_scan_ids(self) -> tuple[str, ...]:
        if getattr(self, '_graph_selection_active', False):
            return tuple((s for s in self._graph_selected_ids if self.project.scan(s) is not None))
        selected: list[str] = []
        for item in self.rail.selectedItems():
            scan_id = item.data(self.qt.QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(scan_id, str) and self.project.scan(scan_id) is not None:
                selected.append(scan_id)
        if not selected:
            scan = self._selected_scan_for_inspection()
            if scan is not None:
                selected.append(scan.scan_id)
        return tuple(sorted(dict.fromkeys(selected), key=lambda scan_id: self.project.scan(scan_id).rail_order))

    def _restore_rail_selection(self, scan_ids: Sequence[str]) -> None:
        requested = set(scan_ids)
        first_row: int | None = None
        blocked = self.rail.blockSignals(True)
        for row in range(self.rail.count()):
            item = self.rail.item(row)
            scan_id = item.data(self.qt.QtCore.Qt.ItemDataRole.UserRole)
            selected = isinstance(scan_id, str) and scan_id in requested
            item.setSelected(selected)
            if selected and first_row is None:
                first_row = row
        if first_row is not None:
            self.rail.setCurrentRow(first_row, self.qt.QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
        else:
            self.rail.setCurrentRow(-1, self.qt.QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
            self.rail.clearSelection()
        self.rail.blockSignals(blocked)

    def _refresh_groups(self, scan_ids: Sequence[str], notice: str) -> None:
        self._populate_tree()
        self._populate_rail()
        self._restore_rail_selection(scan_ids)
        degraded = self._apply_visible_render_plan()
        if scan_ids:
            self.select_scan(scan_ids[0])
        if not degraded:
            self._status_message(notice)
        self._sync_scan_graph()

    def _create_group_from_selection(self) -> None:
        if self._busy_job_kind is not None:
            self._status_message('Wait for the active compute job before grouping')
            return
        scan_ids = self._selected_scan_ids()
        if not scan_ids:
            self._status_message('Select one or more scans before New group')
            return
        default_name = f'Group {len(self.project.groups) + 1:02d}'
        name, accepted = self.qt.QtWidgets.QInputDialog.getText(self.window, 'New scan group', 'Group name', text=default_name)
        if not accepted or not name.strip():
            return
        label = _ascii_scan_id(name, default_name.replace(' ', '-'))
        group_id = label.upper()
        self._discard_alignment_preview_for_edit()
        try:
            self.project.assign_scans_to_group(scan_ids, group_id, label)
        except (RuntimeError, ValueError) as error:
            self._status_message(str(error), 10000)
            return
        self._refresh_groups(scan_ids, f'GROUPED / {len(scan_ids)} SCANS / {group_id}')

    def _ungroup_selected_scans(self) -> None:
        if self._busy_job_kind is not None:
            self._status_message('Wait for the active compute job before ungrouping')
            return
        scan_ids = self._selected_scan_ids()
        if not scan_ids:
            self._status_message('Select one or more scans before Ungroup')
            return
        self._discard_alignment_preview_for_edit()
        try:
            self.project.assign_scans_to_group(scan_ids, 'UNGROUPED', 'Ungrouped')
        except (RuntimeError, ValueError) as error:
            self._status_message(str(error), 10000)
            return
        self._refresh_groups(scan_ids, f'UNGROUPED / {len(scan_ids)} SCANS / TRANSFORMS UNCHANGED')

    def _apply_visible_render_plan(self, on_ready: Callable[[], None] | None=None) -> bool:
        if self.layer_controller is None:
            return False
        apply_plan = getattr(self.viewport_adapter, 'apply_render_plan', None)
        if not callable(apply_plan):
            return False
        try:
            plan = self.layer_controller.render_plan(require_available=True)
        except DisplayModeUnavailableError as error:
            self._status_message(f'FUSED UNAVAILABLE / {error}', 15000, priority=STATUS_PRIORITY_DEGRADED)
            self._sync_evidence_availability()
            self._publish_mcp_state()
            return True
        self._apply_display_plan(plan, on_ready=on_ready)
        self._publish_mcp_state()
        return False

    def _hide_selected_scan(self) -> None:
        if self._graph_selection_active and len(self._graph_selected_ids) > 1:
            self._graph_command('visibility.set', {'scan_ids': self._graph_selected_ids, 'visible': False})
            return
        if self._plan_placement_is_active():
            self._status_message('Use PLACE + NEXT or FINISH while PLAN PLACE is active')
            return
        scan = self._selected_scan_for_inspection()
        if scan is None or self.layer_controller is None:
            self._status_message('Select a scan before Hide')
            return
        if self._scan_is_removed(scan):
            self._status_message('Removed scans stay out of the display; use Undo Remove')
            return
        self.layer_controller.set_scan_visible(scan.scan_id, False)
        degraded = self._apply_visible_render_plan()
        self._sync_scan_graph()
        if not degraded:
            self._status_message(f'HIDDEN / {scan.scan_id} / display only')

    def _isolate_selected_scan(self) -> None:
        if self._graph_selection_active and len(self._graph_selected_ids) > 1:
            self._graph_command('scan.isolate', {'scan_ids': self._graph_selected_ids})
            return
        if self._plan_placement_is_active():
            self._status_message('The active plan-placement scan is already isolated')
            return
        scan = self._selected_scan_for_inspection()
        if scan is None or self.layer_controller is None:
            self._status_message('Select a scan before Isolate')
            return
        if self._scan_is_removed(scan):
            self._status_message('Removed scans cannot be isolated; use Undo Remove')
            return
        for item in self.project.scan_rail:
            self.layer_controller.set_scan_visible(item.scan_id, item.scan_id == scan.scan_id and (not self._scan_is_removed(item)))
        degraded = self._apply_visible_render_plan()
        self._sync_scan_graph()
        if not degraded:
            self._status_message(f'ISOLATED / {scan.scan_id} / display only')

    def _show_all_scans(self, on_ready: Callable[[], None] | None=None) -> None:
        if self._plan_placement_is_active():
            self._status_message('Finish PLAN PLACE before showing all scans')
            return
        if self.layer_controller is None:
            return
        for scan in self.project.scan_rail:
            self.layer_controller.set_scan_visible(scan.scan_id, not self._scan_is_removed(scan))
        degraded = self._apply_visible_render_plan(on_ready)
        self._sync_scan_graph()
        if not degraded:
            self._status_message('SHOW ALL / removed scans remain out of project')

    def _remove_selected_scan(self) -> None:
        if self._plan_placement_is_active():
            self._status_message('Finish PLAN PLACE before removing a scan')
            return
        if self._busy_job_kind is not None:
            self._status_message('Wait for the active compute job before Remove')
            return
        scan = self._selected_scan_for_inspection()
        if scan is None or self.layer_controller is None:
            self._status_message('Select a scan before Remove')
            return
        if self._scan_is_removed(scan):
            self._status_message('Scan is already removed; use Undo Remove')
            return
        self._discard_alignment_preview_for_edit()
        self._reset_scan_layout(notice=False)
        previous = ScanEntry.from_dict(scan.to_dict(), scan.rail_order)
        payload = scan.to_dict()
        payload['status'] = 'REMOVED'
        payload['visible'] = False
        metadata = dict(payload.get('metadata', {}))
        metadata['inspection_removal'] = {'mode': 'SOFT_REMOVE', 'raw_sources': 'PRESERVED', 'undo_available': True}
        payload['metadata'] = metadata
        replacement = ScanEntry.from_dict(payload, scan.rail_order)
        self.project.scans[scan.rail_order] = replacement
        self._inspection_removal_undo = (scan.scan_id, previous)
        self.layer_controller.set_scan_visible(scan.scan_id, False)
        self._populate_tree()
        self._populate_rail()
        self._update_status()
        degraded = self._apply_visible_render_plan()
        self.select_scan(scan.scan_id)
        if not degraded:
            self._status_message(f'REMOVED / {scan.scan_id} / RAW SOURCES PRESERVED / UNDO AVAILABLE')

    def _undo_remove_scan(self) -> None:
        if self._busy_job_kind is not None:
            self._status_message('Wait for the active compute job before Undo Remove')
            return
        undo = self._inspection_removal_undo
        if undo is None or self.layer_controller is None:
            self._status_message('No removed scan to undo')
            return
        scan_id, previous = undo
        current = self.project.scan(scan_id)
        if current is None:
            self._status_message('Removed scan is unavailable for undo')
            return
        self._discard_alignment_preview_for_edit()
        self.project.scans[current.rail_order] = previous
        self.layer_controller.set_scan_visible(scan_id, previous.visible)
        self._inspection_removal_undo = None
        self._populate_tree()
        self._populate_rail()
        self._update_status()
        degraded = self._apply_visible_render_plan()
        self.select_scan(scan_id)
        if not degraded:
            self._status_message(f'REMOVE UNDONE / {scan_id} / RAW SOURCES PRESERVED')

    def _apply_scan_layout(self) -> None:
        if self._plan_placement_is_active():
            self._status_message('Finish PLAN PLACE before using SCAN LAYOUT')
            return
        if self.layer_controller is None:
            return
        self._show_all_scans(self._layout_loaded_scans)

    def _layout_loaded_scans(self) -> None:
        layout = getattr(self.viewport_adapter, 'apply_scan_layout', None)
        if not callable(layout):
            self._status_message('The active viewport has no Scan Layout tool')
            return
        scan_ids = tuple((scan.scan_id for scan in self.project.scan_rail if not self._scan_is_removed(scan)))
        try:
            around_plan = getattr(self.viewport_adapter, 'apply_scan_layout_around_plan', None)
            if self.project.dxf_plan.loaded and callable(around_plan):
                applied = around_plan(scan_ids, gap_m=1.0)
                layout_note = f'AROUND DXF / 01..{len(applied):02d} TIME ORDER'
            else:
                applied = layout(scan_ids, gap_m=1.0)
                layout_note = '+X LONG AXIS'
        except (OSError, RuntimeError, ValueError) as error:
            self._status_message(str(error), 10000)
            return
        self._status_message(f'SCAN LAYOUT / {len(applied)} SCANS / {layout_note} / 1.000 m GAP / DISPLAY ONLY')

    def _reset_scan_layout(self, *, notice: bool=True) -> None:
        reset = getattr(self.viewport_adapter, 'reset_scan_layout', None)
        if not callable(reset):
            return
        try:
            scan_ids = reset()
        except (OSError, RuntimeError, ValueError) as error:
            self._status_message(str(error), 10000)
            return
        if notice and scan_ids:
            self._status_message('SCAN LAYOUT RESET / PROJECT TRANSFORMS UNCHANGED')

    def _update_status(self) -> None:
        name = self.project.name or 'NO PROJECT'
        self.project_status.setText(f'{name.upper()}  /  {len(self.project.scans)} SCANS  /  SCALE 1.000  /  Z-UP')
        self.scan_count.setText(f'{len(self.project.scans)} SCANS')
        current = self.project.pipeline.current.value
        self.pipeline_hint.setText(f'IMPORT -> VALIDATE -> REVIEW -> ALIGN -> QC  /  CURRENT: {current}')

    def _update_scan_pulse(self, scan_id: str | None=None) -> None:
        point_count = 0
        if self.layer_controller is not None:
            plan = self.layer_controller.render_plan(require_available=False)
            point_count = sum((int(actor.source_point_count or 0) for actor in plan.visible_actors if scan_id is None or actor.scan_id == scan_id))
        loaded = getattr(getattr(self.viewport_adapter, 'last_load_result', None), 'scans', ())
        loaded = tuple((item for item in loaded if scan_id is None or item.scan_id == scan_id))
        if getattr(self.viewport_adapter, 'placement_enabled', False):
            triangles = sum((int(getattr(item, 'triangle_count', 0)) for item in loaded))
            points = sum((int(item.displayed_point_count) for item in loaded if getattr(item, 'representation', 'POINTS') == 'POINTS'))
            count_label = f'{triangles:,} TRI' if not points else f'{points:,} PTS'
            self.scan_pulse_count.setText(count_label)
            self.scan_pulse_count.setToolTip(f'Kaynak: {point_count:,} nokta / temsil olcum verisi degildir')
        else:
            self.scan_pulse_count.setText(f'{point_count:,} PTS')
        profile = self.evidence_profile.currentData()
        self.scan_pulse_mode.setText(evidence_label(profile))
        selected = self.project.scan(scan_id) if scan_id else self._selected_scan_for_inspection()
        requests = getattr(getattr(self.viewport_adapter, 'render_plan', None), 'visible_actors', ())
        key = next((item.source_layer_key for item in requests if selected is not None and item.scan_id == selected.scan_id), None)
        description = evidence_tooltip(profile, selected, key)
        self.scan_pulse_mode.setToolTip(description)
        self.evidence_profile.setToolTip(description)
        self.scan_pulse_size.setText(f'{self.point_size.value()} PX')
        self.scan_pulse_surface.setText(self.surface_display.currentText())
        if getattr(self.viewport_adapter, 'placement_enabled', False) and loaded and all((getattr(item, 'representation', 'POINTS') == 'PROXY' for item in loaded)):
            self.scan_pulse_size.setText('YERLESIM')
            self.scan_pulse_surface.setText('HAFIF YERLESIM')
        self._update_placement_mode_ui()
        self._sync_evidence_availability()
        self._sync_evidence_grade_ui()

    def _sync_evidence_availability(self) -> None:
        controller = self.layer_controller
        if controller is None:
            return
        current_plan = controller.render_plan(require_available=False)
        selected = self._selected_scan_for_inspection()
        visible_ids = {key for key, visible in controller.scan_visibility.items() if visible}
        prepared_ids = {item.scan_id for item in current_plan.visible_actors}
        total = len(visible_ids)
        raw_ready_ids = set()
        tooltip_role = self.qt.QtCore.Qt.ItemDataRole.ToolTipRole
        for index in range(self.evidence_profile.count()):
            key = self.evidence_profile.itemData(index)
            candidate = resolve_render_plan(self.project, controller.display_mode, evidence_profile=key, band_visibility=controller.band_visibility, scan_visibility=controller.scan_visibility)
            count = len({item.scan_id for item in candidate.visible_actors})
            if key == EvidenceProfile.RAW_RGB.value:
                raw_ready_ids = {item.scan_id for item in candidate.available_actors}
            usable = count > 0
            try:
                candidate.require_available()
            except LayerControllerError:
                usable = False
            self.evidence_profile.model().item(index).setEnabled(usable)
            note = f'{count}/{total} gorunur taramada kaynak hazir.'
            if not usable:
                note += ' Bu proje/gorunum icin kullanilabilir katman yok.'
            self.evidence_profile.setItemData(index, evidence_tooltip(key) + '\n' + note, tooltip_role)
        text = f'{len(prepared_ids)}/{total} tarama hazir'
        missing = visible_ids - prepared_ids
        if missing:
            text += f' / {len(missing)} eksik'
        selected_request = next((item for item in current_plan.visible_actors if selected is not None and item.scan_id == selected.scan_id), None)
        if selected is not None:
            if selected_request is None:
                text += '\nSecili taramada bu kaynak hazir degil.'
            else:
                layer = selected.layer(selected_request.source_layer_key)
                if layer is not None and layer.role.startswith('CORRECTED_RAW_'):
                    text += '\nCCA sensor yonu ve kamera olcegi duzeltmesi.'
                elif layer is not None and 'APP_FUSED' in layer.role:
                    text += '\nTarama uygulamasinin hazir bulutu.'
        self.evidence_availability.setText(text)
        self.evidence_availability.setToolTip(text + '\nHesapla hazir noktalari yukler; gurultu/bias duzeltmesi uygulamaz.')
        raw_ready = selected is not None and selected.scan_id in raw_ready_ids
        self.evidence_correction_button.setVisible(True)
        self.evidence_correction_button.setText('CCA donusumunu yeniden hesapla' if raw_ready else 'CCA donusumunu hesapla')
        self.evidence_correction_button.setEnabled(self._busy_job_kind is None and self._selected_raw_source() is not None)
        self._sync_depth_model_status()

    def _sync_placement_lock_ui(self, scan: ScanEntry | None) -> None:
        locked = bool(scan is not None and scan.placement_locked)
        if self._graph_selection_active and self._graph_selected_ids:
            locked = any((self.project.scan(s).placement_locked for s in self._graph_selected_ids))
        for button in self.window.findChildren(self.qt.QtWidgets.QPushButton):
            if button.property('ccaCommand') != 'ALIGN:LOCK':
                continue
            button.blockSignals(True)
            button.setEnabled(scan is not None and (not self._plan_placement_is_active()))
            button.setChecked(locked)
            button.setText('Unlock' if locked else 'Lock')
            button.setToolTip('Unlock fixed-scale placement' if locked else 'Lock fixed-scale placement')
            button.blockSignals(False)
        has_transform_target = self._property_target is not None
        for key, field in self.transform_fields.items():
            field.setEnabled(has_transform_target and (not locked) and (not (self._plan_placement_is_active() and key == 'Z')))

    def _toggle_scan_placement_lock(self) -> None:
        if self._plan_placement_is_active():
            self._status_message('PLACE + NEXT controls locking in PLAN PLACE mode')
            return
        if self._busy_job_kind is not None:
            self._status_message('Wait for the active compute job before Lock or Unlock')
            return
        scan = self._selected_scan_for_inspection()
        if scan is None:
            self._status_message('Select a scan before Lock or Unlock')
            return
        self._discard_alignment_preview_for_edit()
        scan_ids = self._graph_selected_ids if self._graph_selection_active else (scan.scan_id,)
        locked = not any((self.project.scan(s).placement_locked for s in scan_ids))
        try:
            for scan_id in scan_ids:
                self.project.set_scan_placement_locked(scan_id, locked)
        except (RuntimeError, ValueError) as error:
            self._status_message(str(error), 10000)
            return
        if locked:
            setter = getattr(self.viewport_adapter, 'set_transform_mode', None)
            if callable(setter):
                try:
                    setter('SELECT')
                except (RuntimeError, ValueError):
                    pass
        self._populate_tree()
        self._populate_rail()
        self._restore_rail_selection(scan_ids)
        current = self.project.scan(scan.scan_id)
        if current is not None:
            self.select_scan(current.scan_id)
        state = 'LOCKED' if locked else 'UNLOCKED'
        self._status_message(f'PLACEMENT {state} / {len(scan_ids)} TARAMA / SOURCE AND SCALE UNCHANGED')
        self._sync_scan_graph()

    def _clear_properties(self) -> None:
        self._property_target = None
        self.selection.setText('NO SELECTION')
        self.selection_hint.setText('Select a scan, plan, or result.')
        self.quality.setText('Scale: locked 1.000\nAxis: Z-up\nAlignment: select a scan\nMeasurement: calibration required')
        self._set_transform_fields((0.0, 0.0, 0.0, 0.0))
        self._sync_placement_lock_ui(None)

    def _set_transform_fields(self, values: Sequence[float]) -> None:
        self._updating_transform_fields = True
        try:
            for key, value in zip(('X', 'Y', 'Z', 'YAW'), values, strict=True):
                self.transform_fields[key].setValue(float(value))
        finally:
            self._updating_transform_fields = False

    @staticmethod
    def _transform_field_values(transform: Sequence[Sequence[float]]) -> tuple[float, ...]:
        yaw = math.degrees(math.atan2(transform[1][0], transform[0][0]))
        return (float(transform[0][3]), float(transform[1][3]), float(transform[2][3]), yaw)

    def _tree_selection_changed(self) -> None:
        selected = self.tree.selectedItems()
        if not selected:
            return
        payload = selected[0].data(0, self.qt.QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(payload, (tuple, list)) and len(payload) >= 2:
            if payload[0] in {'scan', 'layer'}:
                self.select_scan(str(payload[1]))
            elif payload[0] == 'plan':
                self._viewport_plan_selected()

    def _rail_selection_changed(self, current: object, _previous: object) -> None:
        if current is None:
            return
        scan_id = current.data(self.qt.QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(scan_id, str):
            if scan_id in self._recent_scan_ids:
                self._recent_scan_ids.remove(scan_id)
            self._recent_scan_ids.append(scan_id)
            self._recent_scan_ids = self._recent_scan_ids[-2:]
            if self._alignment_review_kind is None:
                try:
                    source_id, target_id = self._selected_alignment_pair()
                except AlignmentWorkflowError:
                    pass
                else:
                    temporal = self._temporal_pair_record(source_id, target_id)
                    detail = ''
                    if temporal is not None:
                        gap = float(temporal.get('gap_s', 0.0))
                        gap_class = 'LONG GAP' if temporal.get('long_gap') is True else 'ADJACENT'
                        detail = f' / {gap:.1f}s / {gap_class} / TAIL -> HEAD'
                    self.alignment_summary.setText(f'PAIR {source_id} -> {target_id}{detail}\nReady for AUTO ALIGN / scale 1.000 / CAD Z-up.')
            self.select_scan(scan_id)

    @staticmethod
    def _format_metres(values: Sequence[float]) -> str:
        return '(' + ', '.join((f'{float(value):.3f}' for value in values)) + ') m'

    def _scan_evidence_properties(self, scan: ScanEntry) -> tuple[str, int, str, list[str]]:
        request = None
        if self.layer_controller is not None:
            plan = self.layer_controller.render_plan(require_available=False)
            request = next((actor for actor in plan.actors if actor.scan_id == scan.scan_id and actor.available), None)
        layer_name = request.actor_id if request is not None else 'NO DISPLAY LAYER'
        source = str(request.source_path) if request is not None and request.source_path else 'UNAVAILABLE'
        point_count = int(request.source_point_count or 0) if request is not None else 0
        provenance = 'not declared'
        has_range_column = False
        if source != 'UNAVAILABLE':
            try:
                schema = read_ply_schema(source)
                properties = {item.name.lower() for item in schema.vertex.properties}
                has_range_column = 'range_m' in properties
                confidence_fields = [name for name in ('conf_mean', 'conf_min', 'confidence', 'n_views', 'n_hits') if name in properties]
                provenance = 'PLY ' + ', '.join(confidence_fields) if confidence_fields else 'PLY has no confidence scalar'
                if not point_count:
                    point_count = schema.vertex.count
            except (OSError, PlyFormatError):
                provenance = 'PLY header unavailable'
        metrics: list[str] = []
        range_line = 'Menzil (range_m): sutun var / goruntu yuklenince ozetlenir' if has_range_column else 'Menzil (range_m): kaynakta sutun yok'
        inspect = getattr(self.viewport_adapter, 'scan_inspection', None)
        if callable(inspect):
            try:
                payload = inspect(scan.scan_id)
                if getattr(self.viewport_adapter, 'placement_enabled', False):
                    is_detailed = self.viewport_adapter.is_scan_detailed(scan.scan_id)
                    metrics.append('Goruntu: ' + ('ASIL NOKTALAR' if is_detailed else 'TEMSIL / yaklasik boyutlar'))
                range_summary = payload.get('range_m') if isinstance(payload, Mapping) else None
                if isinstance(range_summary, Mapping):
                    range_line = f"Menzil (range_m): medyan {float(range_summary.get('median_m', 0.0)):.2f} m / p99 {float(range_summary.get('p99_m', 0.0)):.2f} m / maks {float(range_summary.get('max_m', 0.0)):.2f} m / tum kaynak, esik degil"
                depth_fill = payload.get('depth_fill') if isinstance(payload, Mapping) else None
                if isinstance(depth_fill, Mapping):
                    labelled = int(depth_fill.get('labelled_count', 0))
                    total = int(depth_fill.get('point_count', 0))
                    fraction = float(depth_fill.get('labelled_fraction', 0.0))
                    metrics.append(f'Sensor dolgu: %{fraction * 100:.1f} ({labelled:,} / {total:,}) / gokyuzu dolgu derinligi, fill_share >= 0.5; esik degil, nokta silinmez')
                aabb = payload.get('aabb_m', {}) if isinstance(payload, Mapping) else {}
                obb = payload.get('horizontal_pca_obb_m', {}) if isinstance(payload, Mapping) else {}
                if isinstance(aabb, Mapping):
                    metrics.append('AABB m: ' + self._format_metres(aabb.get('min', (0.0, 0.0, 0.0))) + ' -> ' + self._format_metres(aabb.get('max', (0.0, 0.0, 0.0))))
                if isinstance(obb, Mapping):
                    metrics.append(f"Horizontal PCA OBB m: long {float(obb.get('long_m', 0.0)):.3f} / short {float(obb.get('short_m', 0.0)):.3f} / axis " + self._format_metres((*obb.get('long_axis_xy', (1.0, 0.0)), 0.0)))
            except (OSError, RuntimeError, ValueError):
                metrics.append('AABB/OBB: viewport evidence not loaded')
        return (layer_name, point_count, source, [f'Confidence provenance: {provenance}', range_line, *metrics])

    def select_scan(self, scan_id: str) -> None:
        scan = self.project.scan(scan_id)
        if scan is None:
            return
        if self._graph_selection_active and scan_id not in self._graph_selected_ids:
            self._graph_selection_active = False
        if self.scan_graph is not None and (not self._graph_syncing):
            self._graph_syncing = True
            try:
                self.scan_graph.select_scans(self._graph_selected_ids if self._graph_selection_active else (scan_id,))
            finally:
                self._graph_syncing = False
        if not (self._graph_selection_active and scan_id in self._graph_selected_ids):
            current_rail_ids = {item.data(self.qt.QtCore.Qt.ItemDataRole.UserRole) for item in self.rail.selectedItems()}
            if scan_id not in current_rail_ids:
                self._restore_rail_selection((scan_id,))
        if self.surface_controller is not None:
            try:
                self.surface_controller.select_scan(scan_id)
            except ValueError as error:
                _LOGGER.warning('surface controller rejected %s: %s', scan_id, error)
                self._status_message(f'YUZEY SECIMI YAPILAMADI / {scan_id} / {error}', 12000, priority=STATUS_PRIORITY_DEGRADED)
        matrix = scan.transform
        self._property_target = ('scan', scan.scan_id)
        self._set_transform_fields(self._transform_field_values(matrix))
        layer_name, point_count, source, evidence_lines = self._scan_evidence_properties(scan)
        self.selection.setText(scan.scan_id)
        self.selection_hint.setText(f'GROUP {scan.group_id}  /  {scan.status}\nLAYER {layer_name}\nSOURCE {source}')
        self.quality.setText(f"Scale: locked 1.000\nAxis: Z-up\nPlacement: {('LOCKED' if scan.placement_locked else 'UNLOCKED')}\nPoints: {point_count:,}\nLayers: {len(scan.layers)}\nAlignment: {self._short_status(scan.status)}\n" + '\n'.join(evidence_lines) + '\nMeasurement: calibration required')
        self._sync_placement_lock_ui(scan)
        self._update_scan_pulse(scan_id)
        self._sync_editing_selection()
        select = getattr(self.viewport_adapter, 'select_scan', None)
        load_result = getattr(self.viewport_adapter, 'last_load_result', None)
        if callable(select) and load_result is not None:
            loaded_ids = {item.scan_id for item in getattr(load_result, 'scans', ())}
            if scan_id in loaded_ids:
                select(scan_id)
            else:
                select(None)

    def _show_plan_properties(self) -> None:
        plan = self.project.dxf_plan
        select = getattr(self.viewport_adapter, 'select_plan', None)
        if callable(select):
            select()
        matrix = plan.transform
        self._property_target = ('plan', plan.source_name or 'DXF PLAN')
        self._set_transform_fields(self._transform_field_values(matrix))
        self.selection.setText(plan.source_name or 'DXF PLAN')
        self.selection_hint.setText(f"SOURCE {str(plan.source_unit or 'unknown').upper()} -> INTERNAL M")
        self.quality.setText(f'Scale: locked 1.000\nAxis: Z-up\nSegments: {plan.segment_count or 0:,}\nFit: {plan.fit_status}')
        self._sync_placement_lock_ui(None)
        self._show_properties_panel('TRANSFORM')

    def _viewport_plan_selected(self) -> None:
        if self._plan_placement_is_active() or self._busy_job_kind is not None:
            active = self.plan_placement_session.active_scan_id if self._plan_placement_is_active() else None
            self.viewport_adapter.select_scan(active)
            self._status_message('DXF duzenlemek icin etkin yerlestirme/islemi bitirin')
            return
        self._graph_selection_active = False
        self._graph_selected_ids = ()
        self.rail.clearSelection()
        self.rail.setCurrentItem(None)
        iterator = self.qt.QtWidgets.QTreeWidgetItemIterator(self.tree)
        signals_were_blocked = self.tree.blockSignals(True)
        try:
            self.tree.clearSelection()
            while iterator.value() is not None:
                item = iterator.value()
                data = item.data(0, self.qt.QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(data, (list, tuple)) and data and (data[0] == 'plan'):
                    self.tree.setCurrentItem(item)
                    item.setSelected(True)
                    break
                iterator += 1
        finally:
            self.tree.blockSignals(signals_were_blocked)
        self._show_plan_properties()
        self._status_message('DXF ALTLIK SECILI / Tasi veya Z dondur / olcek 1.000')

    def _commit_plan_transform(self, transform: Sequence[Sequence[float]], *, source: str) -> None:
        if self._plan_placement_is_active() or self._busy_job_kind is not None:
            raise ValueError('DXF duzenlemek icin etkin yerlestirme/islemi bitirin')
        previous = ProjectModel.from_dict(self.project.to_dict())
        old_plan = self.project.dxf_plan
        payload = old_plan.to_dict()
        payload['transform'] = [list(row) for row in transform]
        payload['quality'] = {**payload.get('quality', {}), 'manual_placement': {'mode': 'FIXED_SCALE_Z_UP', 'source': source, 'transform': [list(row) for row in transform]}}
        replacement = DXFPlanState.from_dict(payload)
        update = getattr(self.viewport_adapter, 'update_plan_transform', None)
        try:
            if callable(update):
                update(replacement.transform)
        except (OSError, RuntimeError, ValueError):
            if callable(update):
                update(old_plan.transform)
            raise
        self.project.dxf_plan = replacement
        self._record_project_history(previous)
        self._reset_alignment_controls(clear_undo=False)
        self._show_plan_properties()
        self._status_message('DXF ALTLIK GUNCELLENDI / ham cizim ayni / Geri Al kullanilabilir')

    def _viewport_plan_transform_event(self, event: object) -> None:
        phase = str(getattr(event, 'phase', '')).lower()
        transform = getattr(event, 'transform', None)
        if not isinstance(transform, Sequence):
            return
        if self._busy_job_kind is not None or self._plan_placement_is_active():
            self.viewport_adapter.update_plan_transform(self.project.dxf_plan.transform)
            self._status_message('DXF duzenlemek icin etkin yerlestirme/islemi bitirin')
            return
        self._property_target = ('plan', self.project.dxf_plan.source_name or 'DXF PLAN')
        if phase in {'begin', 'cancel'}:
            self._set_transform_fields(self._transform_field_values(self.project.dxf_plan.transform))
        elif phase == 'preview':
            self._set_transform_fields(self._transform_field_values(transform))
        elif phase == 'commit':
            try:
                self._commit_plan_transform(transform, source='VIEWPORT')
            except (OSError, RuntimeError, ValueError) as error:
                self.viewport_adapter.update_plan_transform(self.project.dxf_plan.transform)
                self._set_transform_fields(self._transform_field_values(self.project.dxf_plan.transform))
                self._status_message(str(error), 10000)

    def _replace_scan_transform(self, scan_id: str, transform: Sequence[Sequence[float]], *, source: str, update_viewport: bool) -> tuple[tuple[float, ...], ...]:
        self.project.validate()
        scan = self.project.scan(scan_id)
        if scan is None:
            raise ValueError(f'unknown manual placement scan: {scan_id}')
        if scan.placement_locked:
            raise ValueError(f'placement is locked for {scan_id}; use Unlock first')
        previous = scan.transform
        payload = scan.to_dict()
        payload['transform'] = [list(row) for row in transform]
        metadata = dict(payload.get('metadata', {}))
        metadata['manual_placement'] = {'mode': 'FIXED_SCALE_Z_UP', 'source': str(source).upper(), 'transform': [list(row) for row in transform]}
        payload['metadata'] = metadata
        replacement = ScanEntry.from_dict(payload, scan.rail_order)
        update = getattr(self.viewport_adapter, 'update_scan_transform', None)
        if update_viewport and callable(update):
            try:
                update(scan_id, replacement.transform)
            except (OSError, RuntimeError, ValueError):
                try:
                    update(scan_id, previous)
                except (OSError, RuntimeError, ValueError):
                    pass
                raise
        self.project.scans[scan.rail_order] = replacement
        self.project.metadata = dict(self.project.metadata)
        self.project.metadata['initial_scan_layout'] = False
        return previous

    def _viewport_transform_event(self, event: object) -> None:
        if getattr(event, 'target_kind', 'scan') == 'plan':
            self._viewport_plan_transform_event(event)
            return
        if self._group_transform_event(event):
            return
        scan_id = str(getattr(event, 'scan_id', ''))
        phase = str(getattr(event, 'phase', '')).lower()
        mode = str(getattr(event, 'mode', 'MANUAL'))
        transform = getattr(event, 'transform', None)
        scan = self.project.scan(scan_id)
        if scan is None or not isinstance(transform, Sequence):
            return
        if scan.placement_locked:
            rollback = getattr(self.viewport_adapter, 'update_scan_transform', None)
            if phase in {'preview', 'commit'} and callable(rollback):
                try:
                    rollback(scan_id, scan.transform)
                except (OSError, RuntimeError, ValueError):
                    pass
            self._set_transform_fields(self._transform_field_values(scan.transform))
            self._status_message(f'PLACEMENT LOCKED / {scan_id} / use Unlock before moving')
            return
        if phase == 'begin':
            self._property_target = ('scan', scan_id)
            self._set_transform_fields(self._transform_field_values(scan.transform))
            self._status_message(f'{mode} preview started / scale 1.000 / CAD Z-up')
            return
        if phase == 'preview':
            self._property_target = ('scan', scan_id)
            self._set_transform_fields(self._transform_field_values(transform))
            return
        if phase == 'cancel':
            self._set_transform_fields(self._transform_field_values(scan.transform))
            self._status_message('Manual placement cancelled')
            return
        if phase != 'commit':
            return
        if self._busy_job_kind is not None:
            rollback = getattr(self.viewport_adapter, 'update_scan_transform', None)
            if callable(rollback):
                rollback(scan_id, scan.transform)
            self._set_transform_fields(self._transform_field_values(scan.transform))
            self._status_message('Wait for the active compute job before placement')
            return
        previous_project = ProjectModel.from_dict(self.project.to_dict())
        try:
            self._replace_scan_transform(scan_id, transform, source=f"VIEWPORT_{mode.replace(' ', '_')}", update_viewport=False)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            rollback = getattr(self.viewport_adapter, 'update_scan_transform', None)
            if callable(rollback):
                try:
                    rollback(scan_id, scan.transform)
                except (OSError, RuntimeError, ValueError):
                    pass
            self._set_transform_fields(self._transform_field_values(scan.transform))
            self._status_message(str(error), 10000)
            return
        committed = self.project.scan(scan_id)
        if committed is not None:
            self._set_transform_fields(self._transform_field_values(committed.transform))
        self._record_project_history(previous_project)
        self._reset_alignment_controls(clear_undo=False)
        self._restore_project_render()
        self._status_message(f'{mode} committed for {scan_id} / scale 1.000 / source unchanged')

    def _transform_fields_changed(self, _field_key: str) -> None:
        if self._updating_transform_fields or self._property_target is None:
            return
        if self._busy_job_kind is not None:
            self._status_message('Wait for the active compute job before placement')
            return
        x, y, z, yaw_degrees = (self.transform_fields[key].value() for key in ('X', 'Y', 'Z', 'YAW'))
        yaw = math.radians(yaw_degrees)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        transform = ((cosine, -sine, 0.0, x), (sine, cosine, 0.0, y), (0.0, 0.0, 1.0, z), (0.0, 0.0, 0.0, 1.0))
        target_kind, target_id = self._property_target
        if target_kind == 'plan':
            try:
                self._commit_plan_transform(transform, source='PROPERTIES')
            except (OSError, RuntimeError, ValueError) as error:
                self._set_transform_fields(self._transform_field_values(self.project.dxf_plan.transform))
                self._status_message(str(error), 10000)
            return
        if target_kind == 'scan' and self._graph_selection_active and (len(self._graph_selected_ids) > 1):
            anchor = self.project.scan(target_id)
            if anchor is None:
                return
            original = anchor.transform
            try:
                self._reset_scan_layout(notice=False)
                delta = np.asarray(transform) @ np.linalg.inv(np.asarray(original))
                self._commit_group_transforms({s: delta @ np.asarray(self.project.scan(s).transform) for s in self._graph_selected_ids}, 'PROPERTIES_GROUP')
                self._status_message(f'{len(self._graph_selected_ids)} TARAMA / ortak donusum uygulandi')
            except (OSError, RuntimeError, ValueError) as error:
                self._set_transform_fields(self._transform_field_values(original))
                self._status_message(str(error), 10000)
            return
        if target_kind == 'scan':
            scan = self.project.scan(target_id)
            if scan is not None and scan.placement_locked:
                self._set_transform_fields(self._transform_field_values(scan.transform))
                self._status_message(f'PLACEMENT LOCKED / {target_id} / use Unlock before editing')
                return
            self._reset_scan_layout(notice=False)
        rollback: Callable[[], object] | None = None
        previous_transform: Sequence[Sequence[float]] | None = None
        previous_project = ProjectModel.from_dict(self.project.to_dict())
        try:
            self.project.validate()
            if target_kind == 'scan':
                scan = self.project.scan(target_id)
                if scan is None:
                    raise ValueError(f'unknown manual placement scan: {target_id}')
                previous_transform = scan.transform
                self._replace_scan_transform(target_id, transform, source='PROPERTIES', update_viewport=True)
                notice = f'Manual placement updated for {target_id}'
            else:
                raise ValueError('unknown manual placement target')
            self._record_project_history(previous_project)
            self._reset_alignment_controls(clear_undo=False)
            self._restore_project_render()
            self._status_message(f'{notice} / scale 1.000 / source unchanged')
        except (OSError, RuntimeError, ValueError) as error:
            if previous_transform is not None:
                self._set_transform_fields(self._transform_field_values(previous_transform))
            if rollback is not None:
                try:
                    rollback()
                except (OSError, RuntimeError, ValueError):
                    self._status_message('Manual placement failed and viewport rollback also failed', 10000)
                    return
            self._status_message(str(error), 10000)

    def _temporal_pair_record(self, source_scan_id: str, target_scan_id: str) -> Mapping[str, object] | None:
        temporal = self.project.metadata.get('temporal_sequence')
        if not isinstance(temporal, Mapping):
            return None
        rows = temporal.get('adjacent_pairs', ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            return None
        return next((row for row in rows if isinstance(row, Mapping) and row.get('source_scan_id') == source_scan_id and (row.get('target_scan_id') == target_scan_id)), None)

    def _selected_alignment_pair(self) -> tuple[str, str]:
        selected: list[str] = []
        for item in self.rail.selectedItems():
            value = item.data(self.qt.QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(value, str) and value not in selected:
                selected.append(value)
        selected.sort(key=lambda scan_id: self.project.scan(scan_id).rail_order if self.project.scan(scan_id) is not None else len(self.project.scans))
        temporal_enabled = isinstance(self.project.metadata.get('temporal_sequence'), Mapping)
        if temporal_enabled:
            if len(selected) == 2:
                return (selected[0], selected[1])
            current = selected[-1] if selected else None
            if current is None and self.rail.currentItem() is not None:
                value = self.rail.currentItem().data(self.qt.QtCore.Qt.ItemDataRole.UserRole)
                current = value if isinstance(value, str) else None
            if current is None and self._recent_scan_ids:
                current = self._recent_scan_ids[-1]
            scan = self.project.scan(current) if current is not None else None
            if scan is not None:
                if scan.rail_order > 0:
                    return (self.project.scan_rail[scan.rail_order - 1].scan_id, scan.scan_id)
                if len(self.project.scan_rail) > 1:
                    return (scan.scan_id, self.project.scan_rail[1].scan_id)
            raise AlignmentWorkflowError('select a scan with a chronological neighbour before AUTO ALIGN')
        if len(selected) == 2:
            return (selected[0], selected[1])
        if len(self._recent_scan_ids) == 2:
            return (self._recent_scan_ids[0], self._recent_scan_ids[1])
        raise AlignmentWorkflowError('select two scans in SCAN RAIL before AUTO ALIGN')

    def _require_unlocked_scans(self, scan_ids: Sequence[str], action: str) -> None:
        locked = [scan_id for scan_id in scan_ids if (scan := self.project.scan(scan_id)) is not None and scan.placement_locked]
        if locked:
            raise AlignmentWorkflowError(f'{action} cannot move locked scan {locked[0]}; use Unlock first')

    def _make_registration_service(self) -> object:
        if self.registration_service_factory is not None:
            return self.registration_service_factory(self.project)
        from viewer.registration_service import ProjectRegistrationService
        return ProjectRegistrationService(self.project, search_mode=getattr(self, '_rgb_search_mode', 'ENDPOINT'))

    def _ensure_alignment_workflow(self) -> AlignmentWorkflow:
        if self.alignment_workflow is None:
            if len(self.project.scans) < 2:
                raise AlignmentWorkflowError('alignment needs at least two scans')
            self.alignment_workflow = AlignmentWorkflow(tuple((scan.scan_id for scan in self.project.scan_rail)), self.project.to_dict(), self._make_registration_service())
        return self.alignment_workflow

    def _preview_alignment_transforms(self, transforms: Mapping[str, object]) -> None:
        if self.layer_controller is None:
            return
        apply_plan = getattr(self.viewport_adapter, 'apply_render_plan', None)
        if not callable(apply_plan):
            return
        payload = self.project.to_dict()
        rows = payload.get('scan_rail', [])
        if not isinstance(rows, list):
            raise AlignmentWorkflowError('project scan rail is invalid')
        for row in rows:
            if not isinstance(row, dict):
                continue
            scan_id = row.get('scan_id')
            if isinstance(scan_id, str) and scan_id in transforms:
                row['transform'] = transforms[scan_id]
        preview = ProjectModel.from_dict(payload)
        plan = resolve_render_plan(preview, self.layer_controller.display_mode, evidence_profile=self.layer_controller.evidence_profile, point_size=self.layer_controller.point_size, band_visibility=self.layer_controller.band_visibility, scan_visibility=self.layer_controller.scan_visibility, require_available=True)
        apply_plan(plan)

    def _restore_project_render(self) -> None:
        if self.layer_controller is None:
            return
        apply_plan = getattr(self.viewport_adapter, 'apply_render_plan', None)
        if callable(apply_plan):
            apply_plan(self.layer_controller.render_plan(require_available=True))

    def _set_edge_review(self, edge: AlignmentEdge) -> None:
        self._alignment_review_kind = 'pair'
        quality = edge.result.quality.summary() if edge.result is not None else edge.message
        self.alignment_summary.setText(f'{edge.source_scan_id} -> {edge.target_scan_id}\n{edge.status.value} / {quality}\nPREVIEW ONLY / SCALE 1.000 / CAD Z-UP')
        eligible = bool(edge.result is not None and edge.result.eligible_for_apply)
        self.accept_alignment_button.setEnabled(eligible and edge.status is EdgeStatus.REVIEW)
        self.reject_alignment_button.setEnabled(edge.status in {EdgeStatus.REVIEW, EdgeStatus.ACCEPTED})
        self.apply_alignment_button.setEnabled(eligible and edge.status is EdgeStatus.ACCEPTED)

    def _current_pair_hypothesis(self) -> AlignmentHypothesis | None:
        pair_set = self._pair_variation_set
        if pair_set is None or not pair_set.hypotheses:
            return None
        return pair_set.hypotheses[self._pair_variation_index]

    @staticmethod
    def _variation_metric(value: object, *, digits: int=3) -> str:
        if value is None:
            return 'n/a'
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 'n/a'
        return f'{float(value):.{digits}f}'

    def _show_pair_variation(self, index: int) -> None:
        pair_set = self._pair_variation_set
        if pair_set is None or not pair_set.hypotheses:
            self._status_message('Run PAIR VARIATIONS for a chronological pair first')
            return
        self._pair_variation_index = index % len(pair_set.hypotheses)
        hypothesis = pair_set.hypotheses[self._pair_variation_index]
        self._alignment_review_kind = 'variation_preview'
        for button in (self.accept_alignment_button, self.reject_alignment_button, self.apply_alignment_button):
            button.setEnabled(False)
        transform = hypothesis.transform_target_from_source_zup
        if transform is None:
            self._restore_project_render()
        else:
            target_scan = self.project.scan(hypothesis.target_scan_id)
            if target_scan is None:
                raise AlignmentWorkflowError('variation target scan is missing')
            target_pose = np.asarray(target_scan.transform, dtype=np.float64)
            target_from_source = np.asarray(transform, dtype=np.float64)
            self._preview_alignment_transforms({hypothesis.source_scan_id: (target_pose @ target_from_source).tolist(), hypothesis.target_scan_id: target_pose.tolist()})
        features = hypothesis.features
        detail = 'NO TRANSFORM'
        place_evidence = 'MEKAN KANITI / Henuz yok'
        if features is not None:
            place_evidence = f'ORTAK GEOMETRI ADAYI / {features.depth_inliers or 0} depth eslesmesi / {features.independent_frame_pairs or 0} bagimsiz kare cifti'
            detail = f'FIT {self._variation_metric(features.symmetric_fitness_025m)} / OVERLAP {self._variation_metric(features.bidirectional_overlap)} / RMSE {self._variation_metric(features.rmse_m)} m / CORR {features.correspondences:,} / RGB {self._variation_metric(features.rgb_matched_to_shuffled_ratio, digits=2)}'
        reasons = ' / '.join(hypothesis.rejection_reasons) if hypothesis.rejection_reasons else 'NO REJECTION'
        self.alignment_summary.setText(f'{pair_set.pair.pair_id}\n{hypothesis.hypothesis_id} / HIZALAMA: {hypothesis.state.value} / {self._pair_variation_index + 1} OF {len(pair_set.hypotheses)}\n{place_evidence}\n{detail}\n{reasons}\nHizalama reddi farkli mekan demek degildir.\nDISPLAY PREVIEW / PROJECT UNCHANGED')
        selectable = hypothesis.state is HypothesisState.ELIGIBLE_FOR_REVIEW
        self._set_variation_command_state(previous=len(pair_set.hypotheses) > 1, next_=len(pair_set.hypotheses) > 1, use=selectable)
        self._status_message(f'PAIR VARIATION {hypothesis.hypothesis_id} / {hypothesis.state.value}', 10000)

    def _rolove_registration_guard(self) -> bool:
        from geometry.rolove import alignment_matrix
        corrected = [scan.scan_id for scan in self.project.scans if not np.allclose(alignment_matrix(scan.metadata), np.eye(4), atol=1e-09)]
        if not corrected:
            return False
        self._status_message('XYZ rolove duzeltmesi mevcut. Eski Z-up birlestirme bu duzeltmeyi kullanamaz; rolove panelinden plan/gorunus uretin veya once hizalamayi geri alin.', 15000)
        return True

    def _start_pair_variations(self) -> None:
        if self._rolove_registration_guard():
            return
        if self._busy_job_kind is not None:
            self._status_message(f'{self._busy_job_kind} is already running; wait for completion')
            return
        try:
            source_scan_id, target_scan_id = self._selected_alignment_pair()
            self._require_unlocked_scans((source_scan_id, target_scan_id), 'PAIR VARIATIONS')
            workflow = self._ensure_alignment_workflow()
            edge = workflow.select_pair(source_scan_id, target_scan_id)
            if edge.status is EdgeStatus.ACCEPTED:
                raise AlignmentWorkflowError('accepted alignment must be applied, rejected, or undone first')
            if edge.status in {EdgeStatus.REVIEW, EdgeStatus.REJECTED}:
                edge = workflow.cancel(edge.edge_id)
            if edge.status is not EdgeStatus.CANDIDATE:
                raise AlignmentWorkflowError(f'alignment edge is busy: {edge.status.value}')
            service = workflow.service
            register = getattr(service, 'register_pair_hypotheses', None)
            if not callable(register):
                raise AlignmentWorkflowError('the active registration service has no pair variations')
            job = PairRegistrationJob(job_id=f'variation-{uuid.uuid4().hex}', edge_id=edge.edge_id, source_scan_id=source_scan_id, target_scan_id=target_scan_id)
            project = self.project

            def completed(result: object) -> None:
                if self.project is not project or self.alignment_workflow is not workflow:
                    raise AlignmentWorkflowError('project changed while variations were running')
                if not isinstance(result, PairHypothesisSet):
                    raise AlignmentWorkflowError('pair variation service returned invalid evidence')
                self._pair_variation_set = result
                self._pair_variation_index = 0
                self._pair_variation_job = job
                self._pair_variation_service = service
                self._variation_mode = 'pair'
                self._plan_fit_result = None
                self._plan_fit_staged = None
                self._plan_fit_accepted = False
                self._show_pair_variation(0)

            def failed(error: Exception) -> None:
                self._pair_variation_set = None
                self._pair_variation_job = None
                self._pair_variation_service = None
                self._variation_mode = None
                self._set_variation_command_state(previous=False, next_=False, use=False)
                self.alignment_summary.setText(f'PAIR VARIATIONS FAILED\n{error}')
                self._restore_project_render()
                self._status_message(str(error), 10000)
            if self._start_background_job('PAIR VARIATIONS', f'PAIR VARIATIONS RUNNING / {source_scan_id} -> {target_scan_id}', lambda: register(job), completed, failed):
                self.alignment_summary.setText(f'{source_scan_id} -> {target_scan_id}\nGENERATING H0 + H1...Hn / PROJECT UNCHANGED')
        except (OSError, RuntimeError, ValueError) as error:
            self.alignment_summary.setText(f'PAIR VARIATIONS FAILED\n{error}')
            self._status_message(str(error), 10000)

    def _step_pair_variation(self, step: int) -> None:
        if self._variation_mode == 'plan':
            result = self._plan_fit_result
            if result is None or not result.hypotheses:
                self._status_message('Run PLAN FIT first')
                return
            self._show_plan_fit_variation(self._plan_fit_index + int(step))
            return
        if self._pair_variation_set is None:
            self._status_message('Run PAIR VARIATIONS first')
            return
        self._show_pair_variation(self._pair_variation_index + int(step))

    def _use_pair_candidate(self) -> None:
        if self._variation_mode == 'plan':
            self._stage_plan_fit_candidate()
            return
        if self._busy_job_kind is not None:
            self._status_message(f'{self._busy_job_kind} is already running; wait for completion')
            return
        pair_set = self._pair_variation_set
        hypothesis = self._current_pair_hypothesis()
        job = self._pair_variation_job
        service = self._pair_variation_service
        workflow = self.alignment_workflow
        if pair_set is None or hypothesis is None or job is None or (service is None) or (workflow is None):
            self._status_message('Run PAIR VARIATIONS and select one Hn first')
            return
        if hypothesis.state is not HypothesisState.ELIGIBLE_FOR_REVIEW:
            self._status_message('Only ELIGIBLE_FOR_REVIEW Hn can be used')
            return
        prepare = getattr(service, 'prepare_selected_hypothesis', None)
        if not callable(prepare):
            self._status_message('The active service cannot prepare a selected Hn')
            return
        project = self.project

        def completed(result: object) -> None:
            if self.project is not project or self.alignment_workflow is not workflow:
                raise AlignmentWorkflowError('project changed while candidate was prepared')
            if not isinstance(result, PairRegistrationResult):
                raise AlignmentWorkflowError('selected candidate result is invalid')
            edge = workflow.attach_pair_result(job.edge_id, result)
            self._preview_alignment_transforms(result.transforms)
            self._set_edge_review(edge)
            self._set_variation_command_state(previous=True, next_=True, use=False)
            self._status_message(f'{hypothesis.hypothesis_id} READY / ACCEPT EVIDENCE NEXT', 10000)

        def failed(error: Exception) -> None:
            self._status_message(str(error), 10000)
        self._start_background_job('USE CANDIDATE', f'PREPARING {hypothesis.hypothesis_id} / RAW SOURCES READ ONLY', lambda: prepare(job, pair_set, hypothesis.hypothesis_id), completed, failed)

    def _selected_plan_fit_scan_id(self) -> str:
        item = self.rail.currentItem()
        if item is None:
            raise AlignmentWorkflowError('select one scan before PLAN FIT')
        value = item.data(self.qt.QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(value, str) or self.project.scan(value) is None:
            raise AlignmentWorkflowError('PLAN FIT selection is not a scan')
        return value

    def _plan_geometry_path(self) -> Path:
        plan = self.project.dxf_plan
        if not plan.loaded or not plan.geometry_path:
            raise AlignmentWorkflowError('load a metric DXF plan before PLAN FIT')
        candidate = Path(plan.geometry_path)
        if not candidate.is_absolute():
            if not self.project.source_manifest:
                raise AlignmentWorkflowError('relative DXF geometry has no project root')
            candidate = Path(self.project.source_manifest).resolve().parent / candidate
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise AlignmentWorkflowError('metric DXF plan geometry does not exist')
        return resolved

    @staticmethod
    def _plan_fit_control_text(hypothesis: PlanFitHypothesis) -> str:
        proof = hypothesis.evidence
        if proof is None or proof.validation_overlap is None:
            return 'KONTROL YOK / OTOMATIK ADAY UYGULANAMAZ'
        verdict = 'INCELEMEYE UYGUN' if hypothesis.review_supported else ', '.join(proof.reasons)
        return f'AYRI KONTROL {proof.validation_cells} HUCRE / UYUM {proof.validation_overlap:.3f} / KAYDIRILMIS POZ Q90 {proof.null_overlap_q90:.3f} / FARK {proof.specificity_gain:+.3f}\n{verdict} / KALIBRE OLCUM DEGIL'

    def _show_plan_fit_variation(self, index: int) -> None:
        result = self._plan_fit_result
        if result is None or not result.hypotheses:
            self._status_message('Run PLAN FIT first')
            return
        self._plan_fit_index = index % len(result.hypotheses)
        hypothesis = result.hypotheses[self._plan_fit_index]
        self._plan_fit_staged = None
        self._plan_fit_accepted = False
        self._alignment_review_kind = 'plan_fit_preview'
        self._preview_alignment_transforms({result.scan_id: [list(row) for row in hypothesis.transform_plan_from_scan_zup]})
        for button in (self.accept_alignment_button, self.reject_alignment_button, self.apply_alignment_button):
            button.setEnabled(False)
        self._set_variation_command_state(previous=len(result.hypotheses) > 1, next_=len(result.hypotheses) > 1, use=hypothesis.review_supported)
        self.alignment_summary.setText(f'PLAN FIT / {result.scan_id}\n{hypothesis.hypothesis_id} / {self._plan_fit_index + 1} OF {len(result.hypotheses)} / YAW {hypothesis.yaw_deg:.1f} deg\nSCORE {hypothesis.rank_score:.3f} / OVERLAP {hypothesis.overlap_ratio:.3f} / MEDIAN {hypothesis.median_distance_m:.3f} m / P90 {hypothesis.p90_distance_m:.3f} m\n{self._plan_fit_control_text(hypothesis)}\nDISPLAY PREVIEW / SCALE 1.000 / PROJECT UNCHANGED')
        self._status_message(f'PLAN FIT {hypothesis.hypothesis_id} / ' + ('USE CANDIDATE TO REVIEW' if hypothesis.review_supported else 'KONTROL GECMEDI / AYRINTIYA BAK / ELLE YERLESTIRME ACIK'), 10000)

    def _start_plan_fit(self) -> None:
        if self._rolove_registration_guard():
            return
        if self._busy_job_kind is not None:
            self._status_message(f'{self._busy_job_kind} is already running; wait for completion')
            return
        try:
            scan_id = self._selected_plan_fit_scan_id()
            scan = self.project.scan(scan_id)
            if scan is None:
                raise AlignmentWorkflowError('PLAN FIT scan is missing')
            self._require_unlocked_scans((scan_id,), 'PLAN FIT')
            plan_path = self._plan_geometry_path()
            service = self._make_registration_service()
            resolve = getattr(service, 'resolve_rgb_core', None)
            if not callable(resolve):
                raise AlignmentWorkflowError('the active registration service cannot resolve rgb/core')
            geometry = resolve(scan_id)
            scan_path = getattr(geometry, 'path', None)
            if not isinstance(scan_path, Path):
                raise AlignmentWorkflowError('PLAN FIT scan geometry is invalid')
            project = self.project
            plan_transform = project.dxf_plan.transform
            scan_z_m = float(scan.transform[2][3])

            def completed(value: object) -> None:
                if self.project is not project:
                    raise AlignmentWorkflowError('project changed while PLAN FIT was running')
                if not isinstance(value, PlanFitResult):
                    raise AlignmentWorkflowError('PLAN FIT returned invalid evidence')
                self._variation_mode = 'plan'
                self._plan_fit_result = value
                self._plan_fit_index = 0
                self._plan_fit_staged = None
                self._plan_fit_accepted = False
                self._pair_variation_set = None
                self._pair_variation_job = None
                self._pair_variation_service = None
                self._show_plan_fit_variation(0)

            def failed(error: Exception) -> None:
                self._variation_mode = None
                self._plan_fit_result = None
                self._plan_fit_staged = None
                self._plan_fit_accepted = False
                self._set_variation_command_state(previous=False, next_=False, use=False)
                self.alignment_summary.setText(f'PLAN FIT FAILED\n{error}')
                self._restore_project_render()
                self._status_message(str(error), 10000)
            started = self._start_background_job('PLAN FIT', f'PLAN FIT RUNNING / {scan_id} / SOURCE READ ONLY', lambda: fit_scan_to_plan(scan_path, plan_path, scan_id=scan_id, plan_transform=plan_transform, scan_z_m=scan_z_m), completed, failed)
            if started:
                self.alignment_summary.setText(f'PLAN FIT / {scan_id}\nRANKING FIXED-SCALE XY + YAW CANDIDATES / PROJECT UNCHANGED')
        except (OSError, RuntimeError, ValueError) as error:
            self.alignment_summary.setText(f'PLAN FIT FAILED\n{error}')
            self._status_message(str(error), 10000)

    def _stage_plan_fit_candidate(self) -> None:
        result = self._plan_fit_result
        if result is None or not result.hypotheses:
            self._status_message('Run PLAN FIT first')
            return
        hypothesis = result.hypotheses[self._plan_fit_index]
        if not hypothesis.review_supported:
            self._status_message(self._plan_fit_control_text(hypothesis), 12000)
            return
        self._plan_fit_staged = hypothesis
        self._plan_fit_accepted = False
        self._alignment_review_kind = 'plan_fit'
        self.accept_alignment_button.setEnabled(True)
        self.reject_alignment_button.setEnabled(True)
        self.apply_alignment_button.setEnabled(False)
        self._set_variation_command_state(previous=len(result.hypotheses) > 1, next_=len(result.hypotheses) > 1, use=False)
        self.alignment_summary.setText(f'PLAN FIT / {result.scan_id} / {hypothesis.hypothesis_id}\nOVERLAP {hypothesis.overlap_ratio:.3f} / MEDIAN {hypothesis.median_distance_m:.3f} m / P90 {hypothesis.p90_distance_m:.3f} m\n{self._plan_fit_control_text(hypothesis)}\nREVIEW READY / ACCEPT THEN APPLY / SOURCE UNCHANGED')
        self._status_message(f'{hypothesis.hypothesis_id} READY / ACCEPT EVIDENCE NEXT', 10000)

    def _start_auto_alignment(self) -> None:
        if self._rolove_registration_guard():
            return
        if self._busy_job_kind is not None:
            self._status_message(f'{self._busy_job_kind} is already running; wait for completion')
            return
        try:
            source_scan_id, target_scan_id = self._selected_alignment_pair()
            self._require_unlocked_scans((source_scan_id, target_scan_id), 'AUTO ALIGN')
            workflow = self._ensure_alignment_workflow()
            edge = workflow.select_pair(source_scan_id, target_scan_id)
            if edge.status in {EdgeStatus.REVIEW, EdgeStatus.REJECTED}:
                edge = workflow.cancel(edge.edge_id)
            if edge.status is EdgeStatus.ACCEPTED:
                self._set_edge_review(edge)
                self._status_message('Alignment is accepted and ready to apply')
                return
            if edge.status is not EdgeStatus.CANDIDATE:
                raise AlignmentWorkflowError(f'alignment edge is busy: {edge.status.value}')
            workflow.queue(edge.edge_id)
            project = self.project

            def completed(result: object) -> None:
                if self.project is not project or self.alignment_workflow is not workflow:
                    raise AlignmentWorkflowError('project changed while alignment was running; preview discarded')
                if not isinstance(result, AlignmentEdge):
                    raise AlignmentWorkflowError('alignment queue returned an invalid result')
                completed_edge = result
                if completed_edge.result is not None and completed_edge.result.eligible_for_apply:
                    self._preview_alignment_transforms(completed_edge.result.transforms)
                else:
                    self._restore_project_render()
                self._set_edge_review(completed_edge)
                self._status_message(completed_edge.message, 10000)

            def failed(error: Exception) -> None:
                self.alignment_summary.setText(f'AUTO ALIGN FAILED\n{error}')
                self._restore_project_render()
                self._status_message(str(error), 10000)
            started = self._start_background_job('AUTO ALIGN', f'AUTO ALIGN RUNNING / {source_scan_id} -> {target_scan_id}', workflow.run_next, completed, failed)
            if not started:
                workflow.cancel(edge.edge_id)
                return
            self.auto_align_button.setText('RUNNING...')
            self.alignment_summary.setText(f'{source_scan_id} -> {target_scan_id}\nRUNNING / PREVIEW ONLY / PROJECT UNCHANGED')
        except (OSError, RuntimeError, ValueError) as error:
            self.alignment_summary.setText(f'AUTO ALIGN FAILED\n{error}')
            self._status_message(str(error), 10000)

    def _show_alignment_review(self) -> None:
        workflow = self.alignment_workflow
        if workflow is None or workflow.selected_edge_id is None:
            self._status_message('No alignment preview is queued')
            return
        self._set_edge_review(workflow.selected_edge())

    def _accept_alignment(self) -> None:
        if self._alignment_review_kind == 'plan_fit':
            if self._plan_fit_staged is None:
                self._status_message('No PLAN FIT candidate is staged')
                return
            self._plan_fit_accepted = True
            self.accept_alignment_button.setEnabled(False)
            self.reject_alignment_button.setEnabled(True)
            self.apply_alignment_button.setEnabled(True)
            self.alignment_summary.setText(self.alignment_summary.text().split('\nAPPROVED /', 1)[0] + '\nAPPROVED / APPLY TO PROJECT IS READY')
            self._status_message('PLAN FIT approved; click Apply to project to commit and lock')
            return
        workflow = self.alignment_workflow
        if workflow is None:
            self._status_message('No alignment preview to accept')
            return
        try:
            if self._alignment_review_kind == 'pose_graph':
                workflow.accept_pose_graph()
                self.accept_alignment_button.setEnabled(False)
                self.reject_alignment_button.setEnabled(True)
                self.apply_alignment_button.setEnabled(True)
                self._status_message('Pose graph accepted; APPLY remains operator-gated')
                return
            edge = workflow.accept()
            self._set_edge_review(edge)
            self._status_message('Pair accepted for pose graph or guarded apply')
        except (RuntimeError, ValueError) as error:
            self._status_message(str(error), 10000)

    def _reject_alignment(self) -> None:
        if self._alignment_review_kind == 'plan_fit':
            self._plan_fit_staged = None
            self._plan_fit_accepted = False
            self._alignment_review_kind = None
            self._restore_project_render()
            self.accept_alignment_button.setEnabled(False)
            self.apply_alignment_button.setEnabled(False)
            self.reject_alignment_button.setEnabled(False)
            result = self._plan_fit_result
            self._set_variation_command_state(previous=bool(result is not None and len(result.hypotheses) > 1), next_=bool(result is not None and len(result.hypotheses) > 1), use=result is not None and bool(result.hypotheses))
            self.alignment_summary.setText('PLAN FIT CANDIDATE REJECTED / PROJECT UNCHANGED\nChoose PREVIOUS or NEXT, then USE CANDIDATE.')
            self._status_message('PLAN FIT candidate rejected; project unchanged')
            return
        workflow = self.alignment_workflow
        if workflow is None:
            return
        try:
            if self._alignment_review_kind == 'pose_graph':
                workflow.cancel_pose_graph()
                self._alignment_review_kind = None
            else:
                edge = workflow.reject()
                self._set_edge_review(edge)
            self._restore_project_render()
            self.accept_alignment_button.setEnabled(False)
            self.apply_alignment_button.setEnabled(False)
            self.reject_alignment_button.setEnabled(False)
            self._status_message('Alignment preview rejected; project unchanged')
        except (RuntimeError, ValueError) as error:
            self._status_message(str(error), 10000)

    def _run_pose_graph(self) -> None:
        if self._rolove_registration_guard():
            return
        if self._busy_job_kind is not None:
            self._status_message(f'{self._busy_job_kind} is already running; wait for completion')
            return
        try:
            workflow = self._ensure_alignment_workflow()
            anchor = self._recent_scan_ids[-1] if self._recent_scan_ids else self.project.scan_rail[0].scan_id
            workflow.create_pose_graph_job(anchor)
            project = self.project
        except (OSError, RuntimeError, ValueError) as error:
            self.alignment_summary.setText(f'POSE GRAPH FAILED\n{error}')
            self._status_message(str(error), 10000)
            return

        def completed(_job: object) -> None:
            if self.project is not project or self.alignment_workflow is not workflow:
                raise RuntimeError('project changed while pose graph was running')
            result = workflow.pose_graph_result
            if result is None:
                raise AlignmentWorkflowError('pose graph returned no preview')
            self._preview_alignment_transforms(result.transforms)
            self._alignment_review_kind = 'pose_graph'
            self.alignment_summary.setText(f'POSE GRAPH / ANCHOR {anchor}\nRMSE {result.before_rmse_m:.3f} -> {result.after_rmse_m:.3f} m\nPREVIEW ONLY / ACCEPT THEN APPLY')
            self.accept_alignment_button.setEnabled(True)
            self.reject_alignment_button.setEnabled(True)
            self.apply_alignment_button.setEnabled(False)
            self._status_message('Pose graph preview ready for operator review')

        def failed(error: Exception) -> None:
            self.alignment_summary.setText(f'POSE GRAPH FAILED\n{error}')
            self._status_message(str(error), 10000)
        self._start_background_job('POSE GRAPH', 'POSE GRAPH RUNNING', workflow.run_pose_graph, completed, failed)

    def _restore_camera_state(self, state: object | None) -> None:
        restore_state = getattr(self.viewport_adapter, 'restore_camera_state', None)
        if state is not None and callable(restore_state):
            restore_state(state)
            return
        controller = getattr(self.viewport_adapter, 'controller', None)
        restore = getattr(controller, 'restore', None)
        apply_camera = getattr(self.viewport_adapter, 'apply_camera', None)
        if state is not None and callable(restore):
            restore(state)
            if callable(apply_camera):
                apply_camera()

    def _apply_alignment(self) -> None:
        if self._alignment_review_kind == 'plan_fit':
            self._apply_plan_fit_candidate()
            return
        workflow = self.alignment_workflow
        if workflow is None:
            self._status_message('No accepted alignment to apply')
            return
        try:
            result = workflow.apply()
            previous = ProjectModel.from_dict(self.project.to_dict())
            updated = ProjectModel.from_dict(result.project_state)
            metadata = dict(updated.metadata)
            audits = metadata.get('alignment_audits', [])
            if not isinstance(audits, list):
                audits = []
            audits.append(result.audit.to_dict())
            metadata['alignment_audits'] = audits
            updated.metadata = metadata
            controller = getattr(self.viewport_adapter, 'controller', None)
            camera_state = getattr(controller, 'state', None)
            self.set_project(updated)
            self._record_project_history(previous)
            self.alignment_summary.setText(f'APPLIED / {result.audit.audit_id}\nSOURCE FILES UNCHANGED / SCALE 1.000 / CAD Z-UP')
            self._restore_camera_state(camera_state)
            self._status_message('Alignment applied atomically; UNDO is available')
        except (OSError, RuntimeError, ValueError) as error:
            self._status_message(str(error), 10000)

    def _apply_plan_fit_candidate(self) -> None:
        hypothesis = self._plan_fit_staged
        result = self._plan_fit_result
        if not self._plan_fit_accepted or hypothesis is None or result is None:
            self._status_message('Accept the staged PLAN FIT candidate before APPLY')
            return
        try:
            validate_plan_fit_for_apply(result, hypothesis, plan_transform=self.project.dxf_plan.transform)
        except (OSError, ValueError) as error:
            self._plan_fit_accepted = False
            self.apply_alignment_button.setEnabled(False)
            self._status_message(str(error), 12000)
            return
        scan = self.project.scan(result.scan_id)
        if scan is None:
            self._status_message('PLAN FIT scan no longer exists')
            return
        previous = ProjectModel.from_dict(self.project.to_dict())
        updated = ProjectModel.from_dict(self.project.to_dict())
        updated_scan = updated.scan(result.scan_id)
        if updated_scan is None:
            self._status_message('PLAN FIT scan no longer exists')
            return
        payload = updated_scan.to_dict()
        payload['transform'] = [list(row) for row in hypothesis.transform_plan_from_scan_zup]
        payload['placement_locked'] = True
        metadata = dict(payload.get('metadata', {}))
        metadata['plan_fit'] = {'schema': 'cca.plan-fit-apply.v1', 'hypothesis': hypothesis.to_dict(), 'scan_sha256': result.scan_sha256, 'plan_sha256': result.plan_sha256, 'source_point_count': result.source_point_count, 'support_cell_count': result.support_cell_count, 'effective_grid_m': result.effective_grid_m, 'raw_points_preserved': True, 'fixed_scale': 1.0, 'operator_accepted': True, 'placement_locked': True}
        payload['metadata'] = metadata
        updated.scans[updated_scan.rail_order] = ScanEntry.from_dict(payload, updated_scan.rail_order)
        project_metadata = dict(updated.metadata)
        audits = project_metadata.get('plan_fit_audits', [])
        if not isinstance(audits, list):
            audits = []
        audits.append({'scan_id': result.scan_id, 'hypothesis_id': hypothesis.hypothesis_id, 'scan_sha256': result.scan_sha256, 'plan_sha256': result.plan_sha256, 'raw_points_preserved': True, 'fixed_scale': 1.0, 'placement_locked': True})
        project_metadata['plan_fit_audits'] = audits
        updated.metadata = project_metadata
        controller = getattr(self.viewport_adapter, 'controller', None)
        camera_state = getattr(controller, 'state', None)
        try:
            self.set_project(updated)
            self._record_project_history(previous)
            self.alignment_summary.setText(f'PLAN FIT APPLIED / {result.scan_id} / {hypothesis.hypothesis_id}\nPLACEMENT LOCKED / SOURCE FILES UNCHANGED / SCALE 1.000 / CAD Z-UP')
            self._restore_camera_state(camera_state)
            self._status_message('PLAN FIT applied and placement locked; use Unlock to edit')
        except (OSError, RuntimeError, ValueError) as error:
            self.set_project(previous)
            self._restore_camera_state(camera_state)
            self._status_message(str(error), 10000)

    def _undo_alignment(self) -> None:
        if not self._project_undo_stack:
            self._status_message('Undo history is empty')
            return
        if self._busy_job_kind is not None:
            self._status_message('Wait for the active job before Undo')
            return
        previous = self._project_undo_stack.pop()
        self._project_redo_stack.append(ProjectModel.from_dict(self.project.to_dict()))
        controller = getattr(self.viewport_adapter, 'controller', None)
        camera_state = getattr(controller, 'state', None)
        self._restore_project_snapshot(previous)
        self._alignment_undo_project = self._project_undo_stack[-1] if self._project_undo_stack else None
        self._set_undo_alignment_enabled(bool(self._project_undo_stack))
        self._set_redo_alignment_enabled(True)
        self._restore_camera_state(camera_state)
        self.alignment_summary.setText('LAST PLACEMENT OR ALIGNMENT UNDONE / SOURCE FILES UNCHANGED')
        self._status_message('Last placement/alignment undone; Ctrl+Y restores it')

    def _redo_alignment(self) -> None:
        if not self._project_redo_stack:
            self._status_message('Redo history is empty')
            return
        if self._busy_job_kind is not None:
            self._status_message('Wait for the active job before Redo')
            return
        next_project = self._project_redo_stack.pop()
        self._project_undo_stack.append(ProjectModel.from_dict(self.project.to_dict()))
        controller = getattr(self.viewport_adapter, 'controller', None)
        camera_state = getattr(controller, 'state', None)
        self._restore_project_snapshot(next_project)
        self._alignment_undo_project = self._project_undo_stack[-1]
        self._set_undo_alignment_enabled(True)
        self._set_redo_alignment_enabled(bool(self._project_redo_stack))
        self._restore_camera_state(camera_state)
        self.alignment_summary.setText('LAST PLACEMENT OR ALIGNMENT RESTORED / SOURCE FILES UNCHANGED')
        self._status_message('Last placement/alignment restored')

    def _surface_notice(self, status: object, scan_id: str | None) -> str:
        carried = getattr(status, 'operator_message', '')
        if isinstance(carried, str) and carried:
            return carried
        detail = getattr(status, 'message', '') or getattr(status, 'code', '')
        return f"YUZEY / {scan_id or 'secili tarama yok'} / {detail}"

    def _build_surface_preview(self, backend: SurfaceBackend) -> None:
        if self._busy_job_kind is not None:
            _LOGGER.warning('surface %s refused: %s is running', backend.value, self._busy_job_kind)
            self._status_message(f'YUZEY BASLATILMADI / {self._busy_job_kind} calisiyor, bitmesini bekleyin', 10000, priority=STATUS_PRIORITY_DEGRADED)
            return
        controller = self.surface_controller
        if controller is None or controller.selected_scan_id is None:
            _LOGGER.warning('surface %s refused: no scan selected', backend.value)
            self._status_message('YUZEY BASLATILMADI / once seritten bir tarama secin', 10000, priority=STATUS_PRIORITY_DEGRADED)
            return
        scan_id = controller.selected_scan_id
        project = self.project

        def completed(result: object) -> None:
            if self.project is not project or self.surface_controller is not controller:
                raise RuntimeError('project changed while surface was running; preview discarded')
            outcome = result
            if not hasattr(outcome, 'available') or not hasattr(outcome, 'status'):
                raise RuntimeError('surface worker returned an invalid result')
            status = outcome.status
            drawn = False
            display_error: Exception | None = None
            if outcome.available and outcome.result is not None:
                show = getattr(self.viewport_adapter, 'show_surface_preview', None)
                if callable(show):
                    try:
                        show(scan_id, outcome.result)
                    except (RuntimeError, ValueError) as error:
                        display_error = error
                    else:
                        drawn = True
                        mode, _requested = self._surface_state_from_metadata()
                        if mode == 'POINTS ONLY':
                            mode = 'GHOST SURFACE'
                        self.surface_display.setCurrentText(mode)
                        self._save_surface_metadata(display_mode=mode, requested_scan_id=scan_id, backend=backend.value)
            self.quality.setText(self.quality.text() + f'\nSurface: {status.state.value} / {status.code}' + (f' / {status.retained_triangle_count:,} triangles' if status.retained_triangle_count is not None else '') + '\nSurface policy: VISUAL ONLY / NOT MEASUREMENT')
            self._sync_surface_display_ui()
            if display_error is not None:
                _LOGGER.warning('surface %s built for %s but the viewport refused it: %s', backend.value, scan_id, display_error)
                self._status_message(f'YUZEY CIZILEMEDI / {scan_id} / {display_error}', 15000, priority=STATUS_PRIORITY_DEGRADED)
                return
            if drawn:
                self._status_message(self._surface_notice(status, scan_id), 12000)
                return
            _LOGGER.warning('surface %s unavailable for %s: %s / %s', backend.value, scan_id, getattr(status, 'code', '?'), getattr(status, 'message', ''))
            self._status_message(self._surface_notice(status, scan_id), 15000, priority=STATUS_PRIORITY_DEGRADED, tooltip=f"{self._surface_notice(status, scan_id)}\n{getattr(status, 'code', '')} / {getattr(status, 'message', '')}")

        def failed(error: Exception) -> None:
            _LOGGER.warning('surface %s crashed for %s: %s', backend.value, scan_id, error)
            self._status_message(f'YUZEY BASARISIZ / {scan_id} / {error}', 15000, priority=STATUS_PRIORITY_DEGRADED)
        self._start_background_job(f'SURFACE {backend.value}', f'SURFACE {backend.value} RUNNING / {scan_id}', lambda: controller.request(backend), completed, failed)

    def import_paths(self, paths: Sequence[str | Path]) -> list[ScanEntry]:
        imported: list[ScanEntry] = []
        valid = [Path(path).resolve() for path in paths]
        for source in valid:
            if not source.is_file() or source.suffix.lower() not in {'.ply', *MESH_SOURCE_SUFFIXES}:
                raise ValueError(f'unsupported scan source: {source.name}')
        if valid and (not self.project.name):
            self.project = ProjectModel.new(_ascii_scan_id(valid[0].parent.name, 'Imported Project'), uuid.uuid4().hex)
        for source in valid:
            scan = _scan_entry_for_imported_ply(source, len(self.project.scans))
            base_id = scan.scan_id
            scan_id = base_id
            counter = 2
            while self.project.scan(scan_id) is not None:
                scan_id = f'{base_id}-{counter}'
                counter += 1
            scan.scan_id = scan_id
            imported.append(self.project.import_scan(scan))
        self.set_project(self.project)
        return imported

    def _choose_import_files(self) -> None:
        files, _selected_filter = self.qt.QtWidgets.QFileDialog.getOpenFileNames(self.window, 'Import point clouds or meshes', '', 'Lefebvre scan sources (*.ply *.obj *.stl *.glb *.gltf);;PLY point clouds (*.ply);;Meshes (*.obj *.stl *.glb *.gltf)')
        if not files:
            return
        try:
            self.import_paths(files)
        except (OSError, RuntimeError, ValueError) as error:
            self.qt.QtWidgets.QMessageBox.critical(self.window, PRODUCT_NAME, str(error))

    def import_dxf_path(self, path: str | Path, *, source_unit: str, output_dir: str | Path | None=None) -> DXFPlanState:
        if output_dir is None:
            standard_paths = self.qt.QtCore.QStandardPaths
            root = standard_paths.writableLocation(standard_paths.StandardLocation.AppLocalDataLocation)
            if not root:
                raise RuntimeError('CCA Points local application data path is unavailable')
            output_dir = Path(root) / 'plans'
        state = import_dxf_plan_state(path, source_unit=source_unit, output_dir=output_dir)
        if not self.project.project_id:
            self.project.project_id = uuid.uuid4().hex
        if not self.project.name:
            self.project.name = 'Imported Plan'
        self.project.dxf_plan = state
        self.set_project(self.project)
        self._status_message(f'DXF imported: {state.segment_count or 0:,} segments / {state.source_unit} -> metre / source unchanged', 10000)
        return state

    def _choose_dxf_plan(self) -> None:
        path, _selected_filter = self.qt.QtWidgets.QFileDialog.getOpenFileName(self.window, 'Import nokta DXF plan', '', 'DXF plans (*.dxf)')
        if not path:
            return
        unit, accepted = self.qt.QtWidgets.QInputDialog.getItem(self.window, 'DXF source unit', 'Interpret DXF coordinates as:', ('cm', 'm', 'mm'), 0, False)
        if not accepted:
            return
        try:
            self.import_dxf_path(path, source_unit=str(unit))
        except (OSError, RuntimeError, ValueError) as error:
            self.qt.QtWidgets.QMessageBox.critical(self.window, PRODUCT_NAME, str(error))

    def _report_density_fallbacks(self, fallbacks) -> None:
        if self._shutting_down:
            return
        if len(fallbacks) == 1:
            summary = fallbacks[0].message()
        else:
            summary = f'AYRINTI DUSURULDU / {len(fallbacks)} taramada istenen ayrinti kaynaga artik sigmiyor, otomatik ayrinti cizildi / proje ayari degistirilmedi: ' + '; '.join((f'{item.scan_id} {item.requested_cca:g} CCA/m' for item in fallbacks))
        self._status_message(summary, 30000, priority=STATUS_PRIORITY_DEGRADED, tooltip='\n'.join((item.message() for item in fallbacks)))

    def _update_placement_mode_ui(self) -> None:
        supported = bool(getattr(self.viewport_adapter, 'placement_enabled', False))
        busy = self._busy_job_kind is not None
        details = getattr(self.viewport_adapter, 'detailed_scan_ids', frozenset())
        loaded = getattr(getattr(self.viewport_adapter, 'last_load_result', None), 'scans', ())
        proxies = {item.scan_id for item in loaded if getattr(item, 'representation', 'POINTS') == 'PROXY'}
        triangles = sum((int(getattr(item, 'triangle_count', 0)) for item in loaded))
        self.placement_proxy_button.setChecked(supported and (not details))
        self.placement_proxy_button.setEnabled(supported and bool(self.project.scans) and (not busy))
        selected = self._selected_scan_for_inspection()
        plan = self.layer_controller.render_plan(require_available=False) if self.layer_controller else None
        target_ids = self._graph_selected_ids if self._graph_selection_active else (selected.scan_id,) if selected is not None else ()
        selected_ready = bool(target_ids) and plan is not None and (set(target_ids) <= {item.scan_id for item in plan.visible_actors})
        self.calculate_points_button.setEnabled(supported and selected_ready and (not busy))
        visible_ids = frozenset((item.scan_id for item in plan.visible_actors if item.available)) if plan is not None else frozenset()
        visible_button = getattr(self, 'calculate_visible_button', None)
        if visible_button is not None:
            if not supported:
                reason = 'Yerlesim modu kapali: her tarama zaten nokta olarak acik.'
            elif busy:
                reason = 'Devam eden islemin bitmesini bekleyin.'
            elif not visible_ids:
                reason = 'Gorunur tarama yok: once en az bir taramayi gorunur yapin.'
            elif visible_ids <= frozenset(details):
                reason = 'Gorunur taramalarin tumu zaten nokta olarak acik.'
            else:
                reason = ''
            visible_button.setEnabled(not reason)
            visible_button.setToolTip(reason or 'Su an gorunur olan her taramayi nokta olarak acar; kapsami gorunurlukle secersiniz. Bellege sigmayanlar hafif temsil kalir ve adlariyla bildirilir. Yerlesim geri alir. Gurultu/bias duzeltmesi uygulamaz.')
        if not supported:
            label = 'NOKTALAR'
        elif details:
            points = sum((int(item.displayed_point_count) for item in loaded if getattr(item, 'representation', 'POINTS') == 'POINTS'))
            label = f'NOKTALAR / {points:,} PTS / {len(proxies)} hafif temsil'
        else:
            label = f'YERLESIM / {len(proxies)} tarama / {triangles:,} ucgen'
        tooltip = 'Yerlesim hafif temsili kullanir. Hesapla hazir noktalari yukler; gurultu/bias duzeltmesi uygulamaz.'
        fallbacks = tuple(getattr(self.viewport_adapter, 'placement_density_fallbacks', ()) or ())
        if supported and fallbacks:
            label += f' / {len(fallbacks)} ayrinti otomatige dusuruldu'
            tooltip += '\n' + '\n'.join((item.message() for item in fallbacks))
        if supported and (not details):
            summarize = getattr(self.viewport_adapter, 'floating_summary', None)
            floating = summarize() if callable(summarize) else None
            if floating is not None and floating.hidden_count:
                label += f' / {floating.ui_label} {floating.hidden_count:,} hucre ' + ('cizili' if floating.visible else 'gizli')
                tooltip += f'\n{floating.ui_label}: {floating.hidden_count:,} / {floating.point_count:,} hucre, esik t={floating.threshold:.2f}. Gorunum durumu - hucreler onbellekte ve dosyada duruyor.'
            summarize = getattr(self.viewport_adapter, 'sparse_summary', None)
            sparse = summarize() if callable(summarize) else None
            if sparse is not None and sparse.hidden_count:
                label += f' / {sparse.ui_label} {sparse.hidden_count:,} hucre ' + ('cizili' if sparse.visible else 'gizli')
                tooltip += f'\n{sparse.ui_label}: {sparse.hidden_count:,} / {sparse.point_count:,} hucre, esik s={sparse.threshold_m:.2f} m ({self._sparse_threshold_source()}). Gorunum durumu - hucreler onbellekte ve dosyada duruyor.'
        skips = tuple(getattr(self, '_detail_capacity_skips', ()) or ()) if supported else ()
        if skips:
            label += f' / {len(skips)} tarama sigmadi, temsil kaldi'
            tooltip += '\n' + '\n'.join((item.message() for item in skips))
        self.placement_mode_status.setText(label)
        self.placement_mode_status.setToolTip(tooltip)
        self._sync_editing_selection()

    def _use_placement_proxies(self) -> None:
        self._change_placement_details(frozenset())

    def _visible_detail_order(self, visible_ids: Sequence[str]) -> tuple[str, ...]:
        details = frozenset(getattr(self.viewport_adapter, 'detailed_scan_ids', ()) or ())
        selected = frozenset(self._selected_scan_ids())
        measure = getattr(self.viewport_adapter, 'scan_view_distances', None)
        distances: dict[str, float] = {}
        if callable(measure):
            try:
                distances = dict(measure(tuple(visible_ids)) or {})
            except (RuntimeError, TypeError, ValueError):
                distances = {}

        def rail_order(scan_id: str) -> int:
            scan = self.project.scan(scan_id)
            return int(scan.rail_order) if scan is not None else 1 << 30
        return tuple(sorted(dict.fromkeys(visible_ids), key=lambda scan_id: (scan_id not in details, scan_id not in selected, distances.get(scan_id, math.inf), rail_order(scan_id))))

    def _calculate_visible_points(self) -> None:
        if self._busy_job_kind is not None:
            self._status_message('Devam eden islemin bitmesini bekleyin.', 10000)
            return
        if not getattr(self.viewport_adapter, 'placement_enabled', False) or self.layer_controller is None:
            self._status_message('Yerlesim modu kapali: her tarama zaten nokta olarak acik.', 10000)
            return
        if self._plan_placement_is_active():
            self._status_message('Noktalari acmadan once yerlesimi Finish ile bitirin.', 10000)
            return
        try:
            plan = self.layer_controller.render_plan(require_available=True)
        except (OSError, RuntimeError, ValueError) as error:
            self._status_message(str(error), 12000)
            return
        visible = tuple(dict.fromkeys((actor.scan_id for actor in plan.visible_actors if actor.available)))
        if not visible:
            self._status_message("Gorunur tarama yok: once en az bir taramayi gorunur yapin, sonra Gorunurleri Hesapla'ya basin.", 12000)
            return
        order = self._visible_detail_order(visible)
        planner = getattr(self.viewport_adapter, 'plan_detail_capacity', None)
        if callable(planner):
            capacity = planner(plan, order)
            opened = tuple(capacity.opened)
            skipped = tuple(capacity.skipped)
        else:
            opened, skipped = (order, ())
        details = frozenset(opened)
        if not details:
            self._report_detail_capacity_skips(skipped, opened=0)
            self._update_placement_mode_ui()
            return
        self._change_placement_details(details, capacity_skips=skipped)

    def _report_detail_capacity_skips(self, skips, *, opened: int) -> None:
        if self._shutting_down or not skips:
            return
        names = '; '.join((f'{item.scan_id} ({item.display_point_count:,} nokta)' for item in skips))
        summary = f'TEMSIL KALDI / {len(skips)} gorunur tarama bellege sigmadi, hafif temsil olarak kaldi / {opened} tarama nokta olarak acildi / kaynak noktalar degismedi, diger uygulamalari kapatip yeniden deneyin: {names}'
        self._status_message(summary, 30000, priority=STATUS_PRIORITY_DEGRADED, tooltip='\n'.join((item.message() for item in skips)))

    def _calculate_selected_points(self) -> None:
        if self._graph_selection_active:
            self._calculate_scan_set(self._graph_selected_ids)
            return
        scan = self._selected_scan_for_inspection()
        if scan is None or self.layer_controller is None:
            self._status_message('Hesapla icin once bir tarama secin.', 10000)
            return
        plan = self.layer_controller.render_plan(require_available=False)
        if not any((actor.scan_id == scan.scan_id for actor in plan.visible_actors)):
            message = "Secili taramayi gorunur yapin, sonra Hesapla'ya basin." if not self.layer_controller.scan_visibility.get(scan.scan_id, False) else 'Secili taramada bu kaynak hazir degil. Hazir bulutu secin veya CCA donusumunu hesapla kullanin.'
            self._status_message(message, 12000)
            return
        self._change_placement_details(frozenset({scan.scan_id}))

    def _change_placement_details(self, details: frozenset[str], *, capacity_skips: tuple=()) -> None:
        if self._busy_job_kind is not None:
            self._status_message('Devam eden islemin bitmesini bekleyin.', 10000)
            self._update_placement_mode_ui()
            self._sync_scan_graph()
            return
        if not getattr(self.viewport_adapter, 'placement_enabled', False) or self.layer_controller is None:
            return
        if self._plan_placement_is_active() and details:
            self._status_message('Noktalari acmadan once yerlesimi Finish ile bitirin.', 10000)
            return
        selected = self._selected_scan_for_inspection()

        def ready() -> None:
            panel = getattr(self, 'rolove_panel', None)
            if panel is not None and panel.scan.scan_id not in details:
                panel.close()
                self.rolove_panel = None
            if selected is not None:
                self.select_scan(selected.scan_id)
            self._detail_capacity_skips = tuple(capacity_skips)
            self._update_placement_mode_ui()
            self._update_scan_pulse()
            if capacity_skips:
                self._report_detail_capacity_skips(capacity_skips, opened=len(details))
                self._status_message(f'NOKTALAR HAZIR / {len(details)} gorunur tarama', 10000)
                return
            self._status_message(('NOKTALAR HAZIR / secili tarama' if len(details) == 1 else f'NOKTALAR HAZIR / {len(details)} gorunur tarama') if details else 'TEMSIL HAZIR / konumlar korundu', 10000)
        self._apply_display_plan(self.layer_controller.render_plan(require_available=True), detailed_scan_ids=details, on_ready=ready)

    def _after_display_idle(self, callback: Callable[[], None]) -> None:
        if self._shutting_down:
            return
        if self._busy_job_kind is not None:
            self.qt.QtCore.QTimer.singleShot(25, lambda: self._after_display_idle(callback))
            return
        callback()

    def _apply_display_plan(self, plan: object, *, reset_project: bool=False, detailed_scan_ids: frozenset[str] | None=None, on_ready: Callable[[], None] | None=None, on_error: Callable[[Exception], None] | None=None, prepare_plan: object | None=None) -> None:
        adapter = self.viewport_adapter
        prepare = getattr(adapter, 'prepare_render_plan', None)
        supported = bool(getattr(adapter, 'placement_enabled', False)) and callable(prepare)
        project = self.project
        if supported and self._busy_job_kind is not None:

            def deferred() -> None:
                if self.project is project:
                    self._apply_display_plan(plan, reset_project=reset_project, detailed_scan_ids=detailed_scan_ids, on_ready=on_ready, on_error=on_error, prepare_plan=prepare_plan)
            self._after_display_idle(deferred)
            return

        def apply(prepared: object=None) -> None:
            if self.project is not project:
                raise RuntimeError('project changed while display was prepared')
            options = {'prepared': prepared} if supported else {}
            if reset_project:
                result = adapter.load_project(project, show_plan=True, render_plan=plan, **options)
            else:
                result = adapter.apply_render_plan(plan, **options)
            self._update_scan_pulse()
            self._update_placement_mode_ui()
            self._status_message('GORUNTU HAZIR / ' + self.placement_mode_status.text(), 7000)
            self._sync_scan_graph()
            if on_ready is not None:
                self._after_display_idle(on_ready)
            fallbacks = tuple(getattr(result, 'density_fallbacks', ()) or ())
            if fallbacks:
                self._after_display_idle(lambda: self._report_density_fallbacks(fallbacks))

        def failed(error: Exception) -> None:
            self._status_message(f'GORUNTULEME HATASI / {error}', 15000, priority=STATUS_PRIORITY_DEGRADED)
            if on_error is not None:
                on_error(error)
            self._update_placement_mode_ui()
        is_prepared = getattr(adapter, 'render_plan_is_prepared', None)
        if not supported or (not reset_project and detailed_scan_ids is None and (prepare_plan is None) and callable(is_prepared) and is_prepared(plan)):
            apply()
            return
        label = 'TEMSILLER HAZIRLANIYOR / ilk acilis onbellegi'
        if detailed_scan_ids:
            label = 'HESAPLA / secili taramanin noktalari yukleniyor' if len(detailed_scan_ids) == 1 else f'GORUNURLERI HESAPLA / {len(detailed_scan_ids)} taramanin noktalari yukleniyor'
        reports_progress = 'progress' in inspect.signature(prepare).parameters

        def operation(progress: Callable[[int, int, str], None] | None=None) -> object:
            options = {'progress': progress} if reports_progress else {}
            return prepare(prepare_plan or plan, detailed_scan_ids=detailed_scan_ids, **options)
        self._start_background_job('DISPLAY', label, operation, apply, failed, reports_progress=reports_progress)

    def _render_project(self, on_ready: Callable[[], None] | None=None) -> None:
        loader = getattr(self.viewport_adapter, 'load_project', None)
        if callable(loader):
            options: dict[str, object] = {'show_plan': True}
            if self.layer_controller is not None and 'render_plan' in inspect.signature(loader).parameters:
                options['render_plan'] = self.layer_controller.render_plan(require_available=True)
            if getattr(self.viewport_adapter, 'placement_enabled', False):

                def ready() -> None:
                    selected = self._selected_scan_for_inspection()
                    if selected is not None:
                        self.select_scan(selected.scan_id)
                    if on_ready is not None:
                        on_ready()
                self._apply_display_plan(options['render_plan'], reset_project=True, detailed_scan_ids=frozenset(), on_ready=ready)
                return
            result = loader(self.project, **options)
            loaded = getattr(result, 'scans', ())
            if loaded:
                first_id = loaded[0].scan_id
                scan = self.project.scan(first_id)
                if scan is not None:
                    self.rail.setCurrentRow(scan.rail_order)
                self.select_scan(first_id)
                select = getattr(self.viewport_adapter, 'select_scan', None)
                if callable(select):
                    select(first_id)
            if on_ready is not None:
                on_ready()

    def _point_size_changed(self, value: int) -> None:
        try:
            if self.layer_controller is not None:
                self.layer_controller.set_point_size(value)
        except LayerControllerError as error:
            self.point_size.setToolTip(str(error))
            self._status_message(str(error))
            return
        setter = getattr(self.viewport_adapter, 'set_point_size', None)
        if callable(setter):
            setter(value)
        self.scan_pulse_size.setText(f'{value} PX')
        self.point_size.setToolTip('Display-only point size; source points unchanged')

    def _surface_state_from_metadata(self) -> tuple[str, dict[str, str]]:
        payload = self.project.metadata.get('surface_preview')
        mode = 'POINTS ONLY'
        requested: dict[str, str] = {}
        if isinstance(payload, Mapping):
            candidate = payload.get('display_mode')
            if isinstance(candidate, str) and candidate.upper() in SURFACE_DISPLAY_MODES:
                mode = candidate.upper()
            scans = payload.get('scans')
            if isinstance(scans, Mapping):
                for scan_id, state in scans.items():
                    if not isinstance(scan_id, str) or self.project.scan(scan_id) is None:
                        continue
                    if isinstance(state, Mapping) and state.get('requested') is True:
                        backend = state.get('backend')
                        requested[scan_id] = backend if isinstance(backend, str) else SurfaceBackend.BPA.value
        return (mode, requested)

    def _save_surface_metadata(self, *, display_mode: str | None=None, requested_scan_id: str | None=None, backend: str | None=None) -> None:
        mode, requested = self._surface_state_from_metadata()
        if display_mode is not None:
            mode = str(display_mode).upper()
        if requested_scan_id is not None:
            requested[requested_scan_id] = backend or SurfaceBackend.BPA.value
        self.project.metadata['surface_preview'] = {'version': 1, 'display_mode': mode, 'scans': {scan_id: {'requested': True, 'backend': value} for scan_id, value in sorted(requested.items())}}

    def _restore_surface_settings(self) -> None:
        mode, requested = self._surface_state_from_metadata()
        setter = getattr(self.viewport_adapter, 'set_surface_display_mode', None)
        if callable(setter):
            try:
                setter(mode)
            except (RuntimeError, ValueError) as error:
                _LOGGER.warning('persisted surface display %s refused: %s', mode, error)
                mode = 'POINTS ONLY'
        self.surface_display.blockSignals(True)
        self.surface_display.setCurrentText(mode)
        self.surface_display.blockSignals(False)
        self.scan_pulse_surface.setText(mode)
        self._sync_surface_display_ui()
        if not requested:
            return
        actors = getattr(self.viewport_adapter, 'surface_actors', None)
        if isinstance(actors, dict) and actors:
            return
        names = ', '.join(sorted(requested))
        _LOGGER.warning('surface intent restored without a built mesh: %s', names)
        self._status_message(f'YUZEY DURUMU GERI YUKLENDI / {mode} / {len(requested)} tarama ({names}) icin yuzey kayitli ama cizilmedi; {SURFACE_REBUILD_HINT}', 15000, priority=STATUS_PRIORITY_DEGRADED)

    def _surface_admissions(self) -> tuple[Any, ...]:
        controller = self.surface_controller
        reader = getattr(controller, 'admissions', None)
        if not callable(reader):
            return ()
        try:
            return tuple(reader())
        except (OSError, ValueError) as error:
            _LOGGER.warning('surface admission gate failed: %s', error)
            self._status_message(f'YUZEY KAPISI OKUNAMADI / {error}', 12000, priority=STATUS_PRIORITY_DEGRADED)
            return ()

    def _sync_surface_display_ui(self) -> None:
        admissions = self._surface_admissions()
        admissible = tuple((a.scan_id for a in admissions if a.admitted))
        actors = getattr(self.viewport_adapter, 'surface_actors', None)
        has_surface = bool(actors) if isinstance(actors, dict) else False
        enabled = has_surface or bool(admissible)
        self.surface_display.setEnabled(enabled)
        if enabled:
            self.surface_display.setToolTip(f'Gorsel yuzey onizlemesi / {len(admissible)} tarama uygun / {VISUAL_ONLY_NOTICE}')
            return
        reason = next((a.operator_message for a in admissions if not a.admitted), 'YUZEY YOK / uygulama birlestirmesi rgb/core tasiyan tarama yok')
        self.surface_display.setToolTip(reason)

    def _surface_display_changed(self, value: str) -> None:
        mode = str(value).upper()
        setter = getattr(self.viewport_adapter, 'set_surface_display_mode', None)
        try:
            if callable(setter):
                setter(mode)
        except (RuntimeError, ValueError) as error:
            self.surface_display.blockSignals(True)
            self.surface_display.setCurrentText(self.scan_pulse_surface.text())
            self.surface_display.blockSignals(False)
            _LOGGER.warning('surface display mode %s refused: %s', mode, error)
            self._status_message(f'YUZEY GORUNUMU DEGISMEDI / {mode} / {error}', 12000, priority=STATUS_PRIORITY_DEGRADED)
            return
        self.scan_pulse_surface.setText(mode)
        self._save_surface_metadata(display_mode=mode)
        actors = getattr(self.viewport_adapter, 'surface_actors', None)
        if mode != 'POINTS ONLY' and isinstance(actors, dict) and (not actors):
            _LOGGER.warning('surface display %s chosen with no surface built', mode)
            self._status_message(f'YUZEY GORUNUMU {mode} secildi / henuz yuzey olusturulmadi; {SURFACE_REBUILD_HINT}', 12000, priority=STATUS_PRIORITY_DEGRADED)

    def _color_mode_changed(self, value: str) -> None:
        if self.layer_controller is None:
            return
        previous = self.layer_controller.display_mode
        setter = getattr(self.viewport_adapter, 'set_color_mode', None)
        apply_plan = getattr(self.viewport_adapter, 'apply_render_plan', None)

        def restore(error: Exception) -> None:
            self.layer_controller.set_display_mode(previous)
            labels = {DisplayMode.PHOTO_RGB: 'PHOTO RGB', DisplayMode.CONFIDENCE: 'CONFIDENCE', DisplayMode.RUN_TINT: 'SCAN GROUP', DisplayMode.HEIGHT: 'HEIGHT'}
            self.color_mode.blockSignals(True)
            self.color_mode.setCurrentText(labels[previous])
            self.color_mode.blockSignals(False)
            self.color_mode.setToolTip(str(error))
            self._status_message(str(error))
        try:
            self.layer_controller.set_display_mode(value)
            plan = self.layer_controller.render_plan(require_available=True)
            if callable(apply_plan):
                self._apply_display_plan(plan, on_error=restore)
            elif callable(setter):
                setter(value)
        except (OSError, RuntimeError, ValueError) as error:
            restore(error)
            return
        self.color_mode.setToolTip(f'{self.layer_controller.display_mode.value} display; source unchanged')
        self._update_scan_pulse()
        self._status_message(f'DISPLAY {self.layer_controller.display_mode.value} | CAMERA PRESERVED')
        self._publish_mcp_state()

    def _evidence_profile_changed(self, value: str) -> None:
        if self.layer_controller is None:
            return
        previous = self.layer_controller.evidence_profile
        apply_plan = getattr(self.viewport_adapter, 'apply_render_plan', None)
        fallback_notice: str | None = None

        def restore(error: Exception) -> None:
            self.layer_controller.set_evidence_profile(previous)
            self.evidence_profile.blockSignals(True)
            self.evidence_profile.setCurrentText(previous.value)
            self.evidence_profile.blockSignals(False)
            self.evidence_profile.setToolTip(str(error))
            self._update_scan_pulse()
            self._status_message(str(error), 10000)
        try:
            selected_profile = self.layer_controller.set_evidence_profile(value)
            plan = self.layer_controller.render_plan(require_available=False)
            if selected_profile is EvidenceProfile.CORRECTED_RGB and (not plan.visible_actors):
                selected_profile = self.layer_controller.set_evidence_profile(EvidenceProfile.RAW_RGB)
                plan = self.layer_controller.render_plan(require_available=True)
                self.evidence_profile.blockSignals(True)
                self.evidence_profile.setCurrentText(selected_profile.value)
                self.evidence_profile.blockSignals(False)
                fallback_notice = 'DUZELTILMIS KAYNAK GORUNUR TARAMALARDA YOK / HAM KAYNAK ACILDI'
            else:
                plan.require_available()
            if not plan.visible_actors:
                raise ValueError('Bu kaynak icin hazir ve gorunur katman yok. Baska kaynak secin veya CCA donusumunu hesapla kullanin.')
            if callable(apply_plan):
                details = None
                if getattr(self.viewport_adapter, 'placement_enabled', False):
                    details = self.viewport_adapter.detailed_scan_ids & frozenset((item.scan_id for item in plan.visible_actors))

                def ready() -> None:
                    selected = self._selected_scan_for_inspection()
                    if selected is not None:
                        self.select_scan(selected.scan_id)
                    else:
                        self._update_scan_pulse()
                    self._status_message(fallback_notice or f'{evidence_label(selected_profile.value)} / {self.evidence_availability.text().splitlines()[0]}', 15000)
                self._apply_display_plan(plan, detailed_scan_ids=details, on_ready=ready, on_error=restore)
            self._publish_mcp_state()
        except (OSError, RuntimeError, ValueError) as error:
            restore(error)
            self._publish_mcp_state()
            return
        self.evidence_profile.setToolTip(f'{self.layer_controller.evidence_profile.value} / CAD Z-up / source unchanged')
        self._update_scan_pulse()
        self._status_message(fallback_notice or f'EVIDENCE {self.layer_controller.evidence_profile.value} | CAMERA PRESERVED')

    def _evidence_grade_state_from_metadata(self) -> EvidenceGradeState:
        payload = self.project.metadata.get('evidence_grade')
        enabled = True
        pivot = DEFAULT_PIVOT
        if isinstance(payload, dict):
            candidate_enabled = payload.get('enabled', True)
            if type(candidate_enabled) is bool:
                enabled = candidate_enabled
            candidate_pivot = payload.get('pivot', DEFAULT_PIVOT)
            if isinstance(candidate_pivot, (int, float)) and (not isinstance(candidate_pivot, bool)) and math.isfinite(candidate_pivot):
                pivot = candidate_pivot
        try:
            return EvidenceGradeState(enabled=enabled, pivot=pivot)
        except EvidenceGradeError:
            return EvidenceGradeState(enabled=True, pivot=DEFAULT_PIVOT)

    def _restore_evidence_grade_settings(self) -> None:
        state = self._evidence_grade_state_from_metadata()
        pivot_setter = getattr(self.viewport_adapter, 'set_evidence_grade_pivot', None)
        if callable(pivot_setter):
            pivot_setter(state.pivot)
        enable_setter = getattr(self.viewport_adapter, 'set_evidence_grade_enabled', None)
        if callable(enable_setter):
            enable_setter(state.enabled)
        self.evidence_grade_toggle.blockSignals(True)
        self.evidence_grade_toggle.setChecked(state.enabled)
        self.evidence_grade_toggle.blockSignals(False)
        self.evidence_grade_slider.blockSignals(True)
        self.evidence_grade_slider.setValue(round(state.pivot * EVIDENCE_GRADE_SLIDER_SCALE))
        self.evidence_grade_slider.blockSignals(False)

    def _save_evidence_grade_metadata(self, *, enabled: bool, pivot: float) -> None:
        self.project.metadata['evidence_grade'] = {'version': 1, 'enabled': bool(enabled), 'pivot': float(pivot)}

    def _evidence_grade_toggled(self, checked: bool) -> None:
        setter = getattr(self.viewport_adapter, 'set_evidence_grade_enabled', None)
        if callable(setter):
            setter(bool(checked))
        self._save_evidence_grade_metadata(enabled=bool(checked), pivot=self.evidence_grade_slider.value() / EVIDENCE_GRADE_SLIDER_SCALE)
        self._sync_evidence_grade_ui()
        self._status_message('KANIT DERECESI acik / cizilen birimler kanite gore soluklasir' if checked else 'KANIT DERECESI kapali / tum cizilen birimler tam opak')

    def _evidence_grade_pivot_changed(self, raw_value: int) -> None:
        pivot = raw_value / EVIDENCE_GRADE_SLIDER_SCALE
        setter = getattr(self.viewport_adapter, 'set_evidence_grade_pivot', None)
        if callable(setter):
            setter(pivot)
        self._save_evidence_grade_metadata(enabled=self.evidence_grade_toggle.isChecked(), pivot=pivot)
        self._sync_evidence_grade_ui()

    def _sky_fill_visible_from_metadata(self) -> bool:
        payload = self.project.metadata.get('sky_fill')
        if isinstance(payload, dict) and type(payload.get('visible')) is bool:
            return payload['visible']
        return False

    def _restore_sky_fill_settings(self) -> None:
        visible = self._sky_fill_visible_from_metadata()
        setter = getattr(self.viewport_adapter, 'set_sky_fill_visible', None)
        if callable(setter):
            setter(visible)
        self.sky_fill_toggle.blockSignals(True)
        self.sky_fill_toggle.setChecked(visible)
        self.sky_fill_toggle.blockSignals(False)

    def _sky_fill_toggled(self, checked: bool) -> None:
        setter = getattr(self.viewport_adapter, 'set_sky_fill_visible', None)
        if callable(setter):
            setter(bool(checked))
        self.project.metadata['sky_fill'] = {'version': 1, 'visible': bool(checked)}
        self._sync_sky_fill_legend()
        self._status_message('SENSOR DOLGU cizilyor / gokyuzu dolgu noktalari gorunumde' if checked else 'SENSOR DOLGU gizli / noktalar dosyada ve sayimda duruyor')

    def _sync_sky_fill_legend(self) -> None:
        label = getattr(self, 'sky_fill_legend_label', None)
        if label is None:
            return
        summarize = getattr(self.viewport_adapter, 'sky_fill_summary', None)
        sky = summarize() if callable(summarize) else None
        if sky is None:
            reason = getattr(self.viewport_adapter, 'sky_fill_unavailable_reason', None)
            text = reason() if callable(reason) else None
            self.sky_fill_toggle.setEnabled(False)
            label.setText(text or SKY_FILL_UNAVAILABLE_REASON)
            label.setVisible(True)
            return
        self.sky_fill_toggle.setEnabled(True)
        state = 'cizilyor' if sky.visible else 'gizli'
        label.setText(f'{sky.ui_label} / %{sky.hidden_fraction * 100:.1f} ({sky.hidden_count:,} / {sky.point_count:,} birim) {state} / gokyuzu dolgu derinligi (fill_share >= 0.5 ve kanit derecesi 0). Bu bir gorunum durumudur: dosya ve nokta sayisi degismedi, kutu isaretlenince geri gelir.')
        label.setVisible(True)

    def _floating_state_from_metadata(self) -> tuple[bool, float]:
        payload = self.project.metadata.get('floating_hide')
        visible = False
        threshold = DEFAULT_FLOATING_THRESHOLD
        if isinstance(payload, dict):
            if type(payload.get('visible')) is bool:
                visible = payload['visible']
            raw = payload.get('t')
            if isinstance(raw, (int, float)) and (not isinstance(raw, bool)):
                value = float(raw)
                if FLOATING_THRESHOLD_MIN <= value <= FLOATING_THRESHOLD_MAX:
                    threshold = value
        return (visible, threshold)

    def _save_floating_metadata(self, *, visible: bool, threshold: float) -> None:
        self.project.metadata['floating_hide'] = {'version': 1, 'visible': bool(visible), 't': float(threshold)}

    def _restore_floating_settings(self) -> None:
        visible, threshold = self._floating_state_from_metadata()
        set_threshold = getattr(self.viewport_adapter, 'set_floating_threshold', None)
        if callable(set_threshold):
            set_threshold(threshold)
        setter = getattr(self.viewport_adapter, 'set_floating_visible', None)
        if callable(setter):
            setter(visible)
        self.floating_toggle.blockSignals(True)
        self.floating_toggle.setChecked(visible)
        self.floating_toggle.blockSignals(False)
        self.floating_threshold_slider.blockSignals(True)
        self.floating_threshold_slider.setValue(round(threshold * FLOATING_SLIDER_SCALE))
        self.floating_threshold_slider.blockSignals(False)

    def _floating_threshold_value(self) -> float:
        return self.floating_threshold_slider.value() / FLOATING_SLIDER_SCALE

    def _floating_toggled(self, checked: bool) -> None:
        setter = getattr(self.viewport_adapter, 'set_floating_visible', None)
        if callable(setter):
            setter(bool(checked))
        self._save_floating_metadata(visible=bool(checked), threshold=self._floating_threshold_value())
        self._sync_floating_legend()
        self._update_placement_mode_ui()
        self._status_message('HAVADA cizilyor / kareler icinden baktigi noktalar gorunumde' if checked else 'HAVADA gizli / noktalar dosyada ve sayimda duruyor')

    def _floating_threshold_changed(self, raw_value: int) -> None:
        threshold = raw_value / FLOATING_SLIDER_SCALE
        setter = getattr(self.viewport_adapter, 'set_floating_threshold', None)
        if callable(setter):
            setter(threshold)
        self._save_floating_metadata(visible=self.floating_toggle.isChecked(), threshold=threshold)
        self._sync_floating_legend()
        self._update_placement_mode_ui()

    def _sync_floating_legend(self) -> None:
        label = getattr(self, 'floating_legend_label', None)
        if label is None:
            return
        summarize = getattr(self.viewport_adapter, 'floating_summary', None)
        floating = summarize() if callable(summarize) else None
        if floating is None:
            reason = getattr(self.viewport_adapter, 'floating_unavailable_reason', None)
            text = reason() if callable(reason) else None
            self.floating_toggle.setEnabled(False)
            self.floating_threshold_slider.setEnabled(False)
            label.setText(f'{FLOATING_UI_LABEL} / oy yok - {text or FLOATING_UNAVAILABLE_REASON}')
            label.setVisible(True)
            return
        self.floating_toggle.setEnabled(True)
        self.floating_threshold_slider.setEnabled(True)
        state = 'gorunur' if floating.visible else 'gizli'
        label.setText(f'{floating.ui_label} / %{floating.hidden_fraction * 100:.1f} ({floating.hidden_count:,} {state} / {floating.point_count:,} birim) / serbest-uzay oyu: n_through >= 2 ve oran > {floating.threshold:.2f}. Bu bir gorunum durumudur: dosya ve nokta sayisi degismedi, kutu isaretlenince geri gelir.')
        label.setVisible(True)

    def _sparse_state_from_metadata(self) -> tuple[bool, float | None]:
        payload = self.project.metadata.get('sparse_hide')
        visible = False
        threshold: float | None = None
        if isinstance(payload, dict):
            if type(payload.get('visible')) is bool:
                visible = payload['visible']
            raw = payload.get('s')
            if isinstance(raw, (int, float)) and (not isinstance(raw, bool)):
                value = float(raw)
                if SPARSE_THRESHOLD_MIN_M <= value <= SPARSE_THRESHOLD_MAX_M:
                    threshold = value
        return (visible, threshold)

    def _save_sparse_metadata(self, *, visible: bool, threshold: float | None=None) -> None:
        payload: dict[str, object] = {'version': 1, 'visible': bool(visible)}
        if threshold is not None:
            payload['s'] = float(threshold)
        self.project.metadata['sparse_hide'] = payload

    def _sparse_run_default(self) -> float:
        resolve = getattr(self.viewport_adapter, 'sparse_default_threshold', None)
        value = resolve() if callable(resolve) else None
        if value is None:
            return DEFAULT_SPARSE_THRESHOLD_M
        return float(min(max(float(value), SPARSE_THRESHOLD_MIN_M), SPARSE_THRESHOLD_MAX_M))

    def _apply_sparse_run_default(self) -> None:
        if self._sparse_explicit_threshold is not None:
            return
        if self._sparse_default_applying:
            return
        default = self._sparse_run_default()
        if abs(default - self._sparse_threshold_value()) < 0.0005:
            return
        setter = getattr(self.viewport_adapter, 'set_sparse_threshold', None)
        self._sparse_default_applying = True
        try:
            if callable(setter):
                setter(default)
            self.sparse_threshold_slider.blockSignals(True)
            self.sparse_threshold_slider.setValue(round(default * SPARSE_SLIDER_SCALE))
            self.sparse_threshold_slider.blockSignals(False)
        finally:
            self._sparse_default_applying = False

    def _restore_sparse_settings(self) -> None:
        visible, explicit = self._sparse_state_from_metadata()
        self._sparse_explicit_threshold = explicit
        threshold = explicit if explicit is not None else self._sparse_run_default()
        set_threshold = getattr(self.viewport_adapter, 'set_sparse_threshold', None)
        if callable(set_threshold):
            set_threshold(threshold)
        setter = getattr(self.viewport_adapter, 'set_sparse_visible', None)
        if callable(setter):
            setter(visible)
        self.sparse_toggle.blockSignals(True)
        self.sparse_toggle.setChecked(visible)
        self.sparse_toggle.blockSignals(False)
        self.sparse_threshold_slider.blockSignals(True)
        self.sparse_threshold_slider.setValue(round(threshold * SPARSE_SLIDER_SCALE))
        self.sparse_threshold_slider.blockSignals(False)

    def _sparse_threshold_value(self) -> float:
        return self.sparse_threshold_slider.value() / SPARSE_SLIDER_SCALE

    def _sparse_toggled(self, checked: bool) -> None:
        setter = getattr(self.viewport_adapter, 'set_sparse_visible', None)
        if callable(setter):
            setter(bool(checked))
        self._save_sparse_metadata(visible=bool(checked), threshold=self._sparse_explicit_threshold)
        self._sync_sparse_legend()
        self._update_placement_mode_ui()
        self._status_message('SEYREK cizilyor / yerel olarak yalniz noktalar gorunumde' if checked else 'SEYREK gizli / noktalar dosyada ve sayimda duruyor')

    def _sparse_threshold_changed(self, raw_value: int) -> None:
        threshold = raw_value / SPARSE_SLIDER_SCALE
        setter = getattr(self.viewport_adapter, 'set_sparse_threshold', None)
        if callable(setter):
            setter(threshold)
        self._sparse_explicit_threshold = threshold
        self._save_sparse_metadata(visible=self.sparse_toggle.isChecked(), threshold=threshold)
        self._sync_sparse_legend()
        self._update_placement_mode_ui()

    def _sparse_threshold_source(self) -> str:
        return 'operator' if self._sparse_explicit_threshold is not None else 'kosu varsayilani'

    def _sync_sparse_legend(self) -> None:
        label = getattr(self, 'sparse_legend_label', None)
        if label is None:
            return
        self._apply_sparse_run_default()
        summarize = getattr(self.viewport_adapter, 'sparse_summary', None)
        sparse = summarize() if callable(summarize) else None
        if sparse is None:
            reason = getattr(self.viewport_adapter, 'sparse_unavailable_reason', None)
            text = reason() if callable(reason) else None
            self.sparse_toggle.setEnabled(False)
            self.sparse_threshold_slider.setEnabled(False)
            label.setText(f'{SPARSE_UI_LABEL} / oy yok - {text or SPARSE_UNAVAILABLE_REASON}')
            label.setVisible(True)
            return
        self.sparse_toggle.setEnabled(True)
        self.sparse_threshold_slider.setEnabled(True)
        state = 'gorunur' if sparse.visible else 'gizli'
        label.setText(f'{sparse.ui_label} / %{sparse.hidden_fraction * 100:.1f} ({sparse.hidden_count:,} {state} / {sparse.point_count:,} birim) / 8 komsu ortalama mesafe > {sparse.threshold_m:.2f} m ({self._sparse_threshold_source()}). Bu bir gorunum durumudur: dosya ve nokta sayisi degismedi, kutu isaretlenince geri gelir.')
        label.setVisible(True)

    def _sync_evidence_grade_ui(self) -> None:
        results = getattr(self.viewport_adapter, 'evidence_grade_results', None) or {}
        available = any((result.available for result in results.values()))
        self.evidence_grade_toggle.setEnabled(available)
        self.evidence_grade_slider.setEnabled(available and self.evidence_grade_toggle.isChecked())
        if available:
            self.evidence_grade_reason_label.setVisible(False)
            self.evidence_grade_reason_label.setText('')
        else:
            reason = next((result.unavailable_reason for result in results.values() if not result.available and result.unavailable_reason), 'derece yok / bu katmanda conf_mean, n_views kanit sutunu yok')
            self.evidence_grade_reason_label.setText(reason)
            self.evidence_grade_reason_label.setVisible(True)
        summarize = getattr(self.viewport_adapter, 'evidence_grade_summary', None)
        summary = summarize() if callable(summarize) else None
        if summary is not None:
            self.evidence_grade_faded_label.setText(f'%{round(summary.faded_fraction * 100)} soluk')
        elif available:
            self.evidence_grade_faded_label.setText('%0 soluk')
        else:
            self.evidence_grade_faded_label.setText('derece yok')
        self._sync_colour_gap_legend()
        self._sync_depth_fill_legend()
        self._sync_sky_fill_legend()
        self._sync_floating_legend()
        self._sync_sparse_legend()

    def _sync_colour_gap_legend(self) -> None:
        label = getattr(self, 'evidence_colour_gap_label', None)
        if label is None:
            return
        summarize = getattr(self.viewport_adapter, 'colour_gap_summary', None)
        gap = summarize() if callable(summarize) else None
        if gap is None or not gap.point_count:
            label.setText('')
            label.setVisible(False)
            return
        label.setText(f'RENK YOK / %{gap.gap_fraction * 100:.1f} ({gap.gap_count:,} / {gap.point_count:,} birim) / anahtar kare gormedi, notr gri {gap.display_hex} ile cizilir; dosya ve nokta sayisi degismez')
        label.setVisible(True)

    def _sync_depth_fill_legend(self) -> None:
        label = getattr(self, 'evidence_depth_fill_label', None)
        if label is None:
            return
        summarize = getattr(self.viewport_adapter, 'depth_fill_summary', None)
        fill = summarize() if callable(summarize) else None
        if fill is None or not fill.point_count:
            label.setText('')
            label.setVisible(False)
            return
        label.setText(f'SENSOR DOLGU / %{fill.labelled_fraction * 100:.1f} ({fill.labelled_count:,} / {fill.point_count:,} birim) / gokyuzu dolgu derinligi (fill_share >= 0.5); renk paya gore notr griye karisir, derece (1 - pay) ile agirlanir; dosya ve nokta sayisi degismez')
        label.setVisible(True)





def load_desktop_project(path: str | Path | None) -> ProjectModel:
    if path is None:
        return ProjectModel.empty()
    return ProjectModel.open(Path(path).expanduser())

def _smoke_payload(project: ProjectModel) -> dict[str, object]:
    accepted = project.metadata.get('accepted_alignment_placements', {})
    accepted_edge_count = int(accepted.get('edge_count', 0)) if isinstance(accepted, Mapping) else 0
    return {'accepted_alignment_edges': accepted_edge_count, 'axis': project.axis, 'dxf_loaded': project.dxf_plan.loaded, 'name': project.name, 'pipeline': project.pipeline.current.value, 'scale': project.fixed_scale, 'scans': len(project.scans), 'status': 'desktop-contract-ok', 'units': project.units, 'visible_scans': sum((scan.visible for scan in project.scan_rail))}

def _capture_window(window: object, output: Path, app: object) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    adapter = getattr(window, '_cca_points_viewport_adapter', None)
    plotter = getattr(adapter, 'plotter', None)
    render = getattr(plotter, 'render', None)
    if callable(render):
        render()
    app.processEvents()
    window_handle = window.windowHandle()
    screen = window_handle.screen() if window_handle is not None else None
    if screen is None:
        screen = app.primaryScreen()
    image = screen.grabWindow(int(window.winId())) if screen is not None else window.grab()
    if not image.save(str(output), 'PNG'):
        raise RuntimeError('could not save native CCA Points screenshot')
    result = getattr(adapter, 'last_load_result', None)
    rendered_scans = getattr(result, 'scan_count', 0) if result is not None else 0
    displayed_points = getattr(result, 'displayed_point_count', 0) if result is not None else 0
    print(json.dumps({'displayed_points': displayed_points, 'rendered_scans': rendered_scans, 'screenshot': str(output)}, ensure_ascii=True, sort_keys=True))
    app.quit()

def _capture_when_background_idle(window: object, output: Path, app: object, binding: DesktopProjectBinding, qt: QtModules, settle_ms: int) -> None:
    if binding._busy_job_kind is not None:
        qt.QtCore.QTimer.singleShot(100, lambda: _capture_when_background_idle(window, output, app, binding, qt, settle_ms))
        return
    qt.QtCore.QTimer.singleShot(settle_ms, lambda: _capture_window(window, output, app))

def main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(prog='cca-lefebvre')
    parser.add_argument('--project', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--placeholder-viewport', action='store_true')
    parser.add_argument('--scan-id')
    parser.add_argument('--surface-bpa', action='store_true')
    parser.add_argument('--evidence-profile', choices=tuple((profile.value for profile in EvidenceProfile)))
    parser.add_argument('--color-mode', choices=('PHOTO RGB', 'CONFIDENCE', 'SCAN GROUP', 'HEIGHT'))
    parser.add_argument('--transform-mode', choices=('SELECT', 'MOVE', 'ROTATE Z'))
    parser.add_argument('--scan-layout', action='store_true')
    parser.add_argument('--frame-scans', action='store_true')
    parser.add_argument('--show-floating', action='store_true')
    parser.add_argument('--show-sparse', action='store_true')
    parser.add_argument('--calculate-points', action='store_true')
    parser.add_argument('--open-properties', action='store_true')
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--command', action='append', default=[])
    parser.add_argument('--screenshot-delay-ms', type=int, default=2500)
    args = parser.parse_args(argv)
    project = load_desktop_project(args.project or configured_default_project())
    requested_scan_ids = _cli_scan_ids(args.scan_id)
    if requested_scan_ids:
        unknown = [scan_id for scan_id in requested_scan_ids if project.scan(scan_id) is None]
        if unknown:
            parser.error(f"unknown scan id: {', '.join(unknown)}")
        requested = set(requested_scan_ids)
        for scan in project.scans:
            scan.visible = scan.scan_id in requested
    if args.smoke:
        print(json.dumps(_smoke_payload(project), ensure_ascii=True, sort_keys=True))
        return 0
    qt = load_qt_modules()
    app = qt.QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = qt.QtWidgets.QApplication([sys.argv[0]])
    app.setApplicationName(PRODUCT_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)
    if 'Fusion' in qt.QtWidgets.QStyleFactory.keys():
        app.setStyle('Fusion')
    if args.placeholder_viewport:
        viewport: ViewportAdapter = PlaceholderViewportAdapter()
    else:
        from viewer.pyvista_viewport import PyVistaViewportAdapter
        viewport = PyVistaViewportAdapter()
    window = create_shell_window(viewport_adapter=viewport, qt=qt)
    binding = DesktopProjectBinding(window, qt, project, viewport, initial_evidence_profile=args.evidence_profile)
    window._cca_points_project_binding = binding
    app.aboutToQuit.connect(binding.shutdown_background_job)
    window.showMaximized()
    app.processEvents()
    startup: list[Callable[[], None]] = []
    if args.scan_layout or (not requested_scan_ids and _should_apply_initial_scan_layout(project)):
        startup.append(binding._apply_scan_layout)
    if args.frame_scans:
        frame_scans = getattr(viewport, 'frame_scans', None)
        if callable(frame_scans):
            startup.append(frame_scans)
    if requested_scan_ids:
        startup.append(lambda: binding.select_scan(requested_scan_ids[0]))
    if args.color_mode is not None:
        startup.append(lambda: binding.color_mode.setCurrentText(args.color_mode))
    if args.transform_mode is not None:
        setter = getattr(viewport, 'set_transform_mode', None)
        if callable(setter):
            startup.append(lambda: setter(args.transform_mode))
    if args.surface_bpa:
        startup.append(lambda: binding._build_surface_preview(SurfaceBackend.BPA))
    if args.calculate_points:
        startup.append(binding._calculate_visible_points)
    if args.show_floating:
        startup.append(lambda: binding.floating_toggle.setChecked(True))
    if args.show_sparse:
        startup.append(lambda: binding.sparse_toggle.setChecked(True))
    if args.open_properties:
        startup.append(lambda: window.findChild(qt.QtWidgets.QPushButton, 'PropertiesDetailsButton').click())
    for command_text in args.command:
        startup.append(lambda text=command_text: binding.execute_edit_command(text))
    if args.screenshot is not None:
        delay = max(250, args.screenshot_delay_ms)
        startup.append(lambda: _capture_when_background_idle(window, args.screenshot, app, binding, qt, delay))

    def next_startup_action() -> None:
        if not startup:
            return
        startup.pop(0)()
        qt.QtCore.QTimer.singleShot(0, lambda: binding._after_display_idle(next_startup_action))
    binding._render_project(on_ready=next_startup_action)
    if not owns_app:
        return 0
    return int(app.exec())
if __name__ == '__main__':
    raise SystemExit(main())
__all__ = ['DesktopProjectBinding', 'cca_points_app_data_root', 'configured_default_project', 'load_desktop_project', 'main']
