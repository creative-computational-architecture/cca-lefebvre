
from __future__ import annotations
import argparse
import importlib
import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol
from viewer.brand import CREATIVE_NAME, ORGANIZATION_NAME, PANEL_TITLE, PRODUCT_NAME, SKIN_C_DARK, load_brand_assets
from viewer.qt_styles import SHELL_PANELS, SHELL_TAB_LABELS, SHELL_TABS, skin_c_stylesheet
_EXPECTED_HEADER_ASSET = 'cca-mono-dark.svg'
_EMBEDDED_WINDOWS: list[object] = []
_ICON_ROOT = Path(__file__).resolve().parent / 'assets' / 'icons'
_PLAN_PLACEMENT_COMMANDS = ('PLAN PLACE', 'PREVIOUS SCAN', 'PLACE + NEXT', 'FINISH')
_PLAN_PLACEMENT_INITIAL_DISABLED = frozenset({'ALIGN:PREVIOUS SCAN', 'ALIGN:PLACE + NEXT', 'ALIGN:FINISH'})
_RIBBON_COMMANDS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {'PROJECT': (('PROJECT', ('NEW', 'OPEN', 'SAVE', 'RECENT')), ('CONTROL', ('SETTINGS', 'AUDIT', 'ABOUT CCA'))), 'IMPORT': (('SOURCE', ('PLY SCANS', 'RAW ALL DATA', 'DXF PLAN')), ('PROCESS', ('APPLY CORRECTION', 'VALIDATE', 'UNIT / AXIS'))), 'TRANSFORM': (('ELLE YERLESIM', ('ALIGN:SELECT', 'ALIGN:MOVE', 'ALIGN:ROTATE Z', 'ALIGN:EDIT TRANSFORM', 'ALIGN:LOCK')), ('PLAN UZERINDE', tuple((f'ALIGN:{command}' for command in _PLAN_PLACEMENT_COMMANDS))), ('DUZEN', ('ALIGN:XYZ ALIGN', 'REVIEW:SCAN LAYOUT', 'REVIEW:RESET LAYOUT'))), 'ALIGN': (('OTOMATIK', ('AUTO ALIGN', 'POSE GRAPH', 'TIME CHAIN')), ('ADAYLAR', ('PAIR VARIATIONS', 'PLAN FIT', 'PREVIOUS', 'NEXT', 'USE CANDIDATE'))), 'REVIEW': (('KAMERA', ('ZOOM EXTENTS', 'FIT SELECTED', 'TOP', 'FRONT', 'RIGHT', 'PERSPECTIVE')), ('DISPLAY', ('PHOTO RGB', 'CONFIDENCE', 'RUN TINT', 'HEIGHT', 'PROCESSED FULL', 'POINT SIZE')), ('INSPECT', ('SELECT', 'HIDE', 'ISOLATE', 'SHOW ALL', 'REMOVE', 'UNDO REMOVE')), ('KAYDET', ('SNAPSHOT',))), 'SURFACE': (('ONIZLEME', ('LOCAL BPA', 'POISSON')), ('OLCUM ALTYAPISI', ('TSDF STATUS',))), 'EXPORT': (('REVIT', ('REVIT PACK',)), ('GORUNTU', ('REVIEW PNG',)), ('ROLOVE', ('PROJECTION PLANES', 'SURVEY PACK')))}
_COMMAND_LABELS = {'REVIT PACK': 'Revit Paketi', 'NEW': 'Yeni', 'OPEN': 'Ac', 'SAVE': 'Kaydet', 'RECENT': 'Son projeler', 'SETTINGS': 'Ayarlar', 'AUDIT': 'Dogrula', 'LOCK': 'Kilitle', 'EDIT TRANSFORM': 'Konum', 'ZOOM EXTENTS': 'Tumunu sigdir', 'FIT SELECTED': 'Secilene odaklan', 'TOP': 'Ust', 'FRONT': 'On', 'RIGHT': 'Sag', 'PERSPECTIVE': 'Perspektif', 'ABOUT CCA': 'About CCA', 'APPLY CORRECTION': 'CCA Donusumu', 'APP FUSED': 'App Fused', 'AUDIT PACKAGE': 'Audit Package', 'AUTO ALIGN': 'Auto Align', 'C2C': 'C2C', 'C2M': 'C2M', 'COLORED ICP': 'Colored ICP', 'CONTROL POINTS': 'Control Points', 'DXF PLAN': 'DXF Plan', 'DXF SECTION': 'DXF Section', 'FIND OVERLAP': 'Find Overlap', 'ICP': 'ICP', 'LOCAL BPA': 'Local BPA', 'MESH PREVIEW': 'Mesh Preview', 'PHOTO RGB': 'Photo RGB', 'PLY': 'PLY', 'PLY SCANS': 'PLY Scans', 'POINT PAIRS': 'Point Pairs', 'PAIR VARIATIONS': 'Pair Variations', 'PLAN FIT': 'Plan Fit', 'PLAN PLACE': 'Plana yerlestir', 'PREVIOUS SCAN': 'Onceki tarama', 'PLACE + NEXT': 'Yerlestir + ilerle', 'FINISH': 'Bitir', 'PREVIOUS': 'Previous', 'NEXT': 'Next', 'USE CANDIDATE': 'Use Candidate', 'POINT SIZE': 'Point Size', 'PROCESSED FULL': 'Veri kaynagi', 'POSE GRAPH': 'Pose Graph', 'TIME CHAIN': 'Zaman zinciri', 'RAW ALL DATA': 'Raw All Data', 'REVIEW PNG': 'Review PNG', 'RUN TINT': 'Run Tint', 'ROTATE Z': 'Z dondur', 'SELECT': 'Sec', 'HIDE': 'Hide', 'ISOLATE': 'Isolate', 'SHOW ALL': 'Show All', 'REMOVE': 'Remove', 'UNDO REMOVE': 'Undo Remove', 'SCAN LAYOUT': 'Vaziyet etrafina diz', 'RESET LAYOUT': 'Dizilimi sifirla', 'SURVEY PACK': 'Otomatik Rolove', 'PROJECTION PLANES': 'Yuzeyden Plan + Gorunus', 'XYZ ALIGN': 'Taramayi Hizala XYZ', 'TSDF STATUS': 'TSDF Status', 'UNIT / AXIS': 'Unit / Axis', 'MOVE': 'Tasi'}

def _ribbon_command_id(tab: str, command: str) -> str:
    return command if ':' in command else f'{tab}:{command}'

def ribbon_command_ids() -> tuple[str, ...]:
    return tuple((_ribbon_command_id(tab, command) for tab, groups in _RIBBON_COMMANDS.items() for _group, commands in groups for command in commands))
_COMMAND_ICONS = {'EDIT TRANSFORM': 'axis.svg', 'ZOOM EXTENTS': 'compare.svg', 'FIT SELECTED': 'align-pairs.svg', 'TOP': 'axis.svg', 'FRONT': 'axis.svg', 'RIGHT': 'axis.svg', 'PERSPECTIVE': 'surface.svg', 'NEW': 'file-new.svg', 'OPEN': 'folder-open.svg', 'SAVE': 'save.svg', 'RECENT': 'folder-open.svg', 'SETTINGS': 'settings.svg', 'AUDIT': 'validate.svg', 'ABOUT CCA': 'info.svg', 'PLY SCANS': 'import-points.svg', 'RAW ALL DATA': 'raw-frames.svg', 'APPLY CORRECTION': 'axis.svg', 'DXF PLAN': 'import-dxf.svg', 'VALIDATE': 'validate.svg', 'UNIT / AXIS': 'axis.svg', 'POINT PAIRS': 'align-pairs.svg', 'FIND OVERLAP': 'align-pairs.svg', 'AUTO ALIGN': 'align-auto.svg', 'PAIR VARIATIONS': 'align-pairs.svg', 'PLAN FIT': 'import-dxf.svg', 'PLAN PLACE': 'import-dxf.svg', 'PREVIOUS SCAN': 'undo.svg', 'PLACE + NEXT': 'align-pairs.svg', 'FINISH': 'validate.svg', 'PREVIOUS': 'undo.svg', 'NEXT': 'align-pairs.svg', 'USE CANDIDATE': 'validate.svg', 'ICP': 'refine.svg', 'COLORED ICP': 'refine.svg', 'POSE GRAPH': 'pose-graph.svg', 'TIME CHAIN': 'align-pairs.svg', 'C2C': 'compare.svg', 'C2M': 'compare.svg', 'LOCK': 'lock.svg', 'SELECT': 'align-pairs.svg', 'MOVE': 'axis.svg', 'ROTATE Z': 'axis.svg', 'PHOTO RGB': 'palette.svg', 'CONFIDENCE': 'confidence.svg', 'RUN TINT': 'palette.svg', 'HEIGHT': 'axis.svg', 'POINT SIZE': 'point-size.svg', 'CLIP': 'section.svg', 'SECTION': 'section.svg', 'PICKS': 'align-pairs.svg', 'SNAPSHOT': 'snapshot.svg', 'LOCAL BPA': 'surface.svg', 'POISSON': 'surface.svg', 'DENSITY': 'surface.svg', 'NORMALS': 'surface.svg', 'TSDF STATUS': 'surface.svg', 'DISTANCE': 'measure.svg', 'ANGLE': 'measure.svg', 'AREA': 'measure.svg', 'ELEVATION': 'measure.svg', 'CLEARANCE': 'measure.svg', 'CONTROL POINTS': 'align-pairs.svg', 'UNCERTAINTY': 'confidence.svg', 'PLY': 'export.svg', 'DXF SECTION': 'export.svg', 'MESH PREVIEW': 'export.svg', 'CSV': 'export.svg', 'REVIEW PNG': 'snapshot.svg', 'AUDIT PACKAGE': 'export.svg', 'REVIT PACK': 'export.svg', 'SURVEY PACK': 'section.svg', 'PROJECTION PLANES': 'section.svg', 'XYZ ALIGN': 'axis.svg'}

class QtUnavailableError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class QtModules:
    QtCore: ModuleType
    QtGui: ModuleType
    QtWidgets: ModuleType

@dataclass(frozen=True, slots=True)
class ShellContract:
    product: str
    tabs: tuple[str, ...]
    panels: dict[str, str]
    brand_image: Path
    colors: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {'product': self.product, 'tabs': list(self.tabs), 'panels': dict(self.panels), 'brand_image': str(self.brand_image), 'colors': dict(self.colors)}

class ViewportAdapter(Protocol):

    def create_widget(self, parent: object, qt: QtModules) -> object:
        pass

def qt_available(find_spec: Callable[[str], object | None]=importlib.util.find_spec) -> bool:
    try:
        return find_spec('PySide6') is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False

def load_qt_modules(import_module: Callable[[str], ModuleType]=importlib.import_module) -> QtModules:
    try:
        return QtModules(QtCore=import_module('PySide6.QtCore'), QtGui=import_module('PySide6.QtGui'), QtWidgets=import_module('PySide6.QtWidgets'))
    except (ImportError, ModuleNotFoundError) as error:
        raise QtUnavailableError('PySide6 is not installed. The current native CCA Points launcher is unchanged.') from error

def shell_contract() -> ShellContract:
    assets = load_brand_assets()
    if assets.company_svg.name != _EXPECTED_HEADER_ASSET:
        raise RuntimeError('unexpected CCA Points header asset')
    return ShellContract(product=PRODUCT_NAME, tabs=tuple(SHELL_TABS), panels=dict(SHELL_PANELS), brand_image=assets.company_svg.resolve(), colors=dict(SKIN_C_DARK))

def _command_label(command: str) -> str:
    command = command.rsplit(':', 1)[-1]
    return _COMMAND_LABELS.get(command, command.title())

def _icon(name: str, qt: QtModules) -> object:
    path = _ICON_ROOT / name
    if not path.is_file():
        raise RuntimeError(f'missing CCA Points icon: {name}')
    return qt.QtGui.QIcon(str(path))

def _command_icon(command: str, qt: QtModules) -> object:
    command = command.rsplit(':', 1)[-1]
    return _icon(_COMMAND_ICONS.get(command, 'command.svg'), qt)

class PlaceholderViewportAdapter:

    def create_widget(self, parent: object, qt: QtModules) -> object:
        QtCore, QtGui, QtWidgets = (qt.QtCore, qt.QtGui, qt.QtWidgets)

        class ViewportCanvas(QtWidgets.QWidget):

            def __init__(self, owner: object) -> None:
                super().__init__(owner)
                self.setMinimumSize(520, 360)
                self.setMouseTracking(True)

            def paintEvent(self, event: object) -> None:
                del event
                painter = QtGui.QPainter(self)
                painter.fillRect(self.rect(), QtGui.QColor('#121417'))
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
                painter.setPen(QtGui.QPen(QtGui.QColor('#2A2C31'), 1))
                spacing = 40
                for x in range(0, self.width(), spacing):
                    painter.drawLine(x, 0, x, self.height())
                for y in range(0, self.height(), spacing):
                    painter.drawLine(0, y, self.width(), y)
                center = self.rect().center()
                painter.setPen(QtGui.QPen(QtGui.QColor('#A8A8A3'), 1))
                painter.drawLine(center.x() - 16, center.y(), center.x() + 16, center.y())
                painter.drawLine(center.x(), center.y() - 16, center.x(), center.y() + 16)
                painter.setPen(QtGui.QColor('#B8B8BA'))
                painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, 'POINT VIEWPORT\nIMPORT A SCAN TO BEGIN')
                painter.end()
        return ViewportCanvas(parent)

def _section_label(text: str, qt: QtModules) -> object:
    label = qt.QtWidgets.QLabel(text)
    label.setObjectName('SectionLabel')
    return label

def _make_project_browser(qt: QtModules) -> object:
    QtWidgets = qt.QtWidgets
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)
    context = QtWidgets.QLabel('Survey project')
    context.setObjectName('ProjectContext')
    context_meta = QtWidgets.QLabel('METRIC  /  Z-UP  /  FIXED SCALE')
    context_meta.setObjectName('ProjectContextMeta')
    context_meta.setProperty('ccaRole', 'railMeta')
    layout.addWidget(context)
    layout.addWidget(context_meta)
    search = QtWidgets.QLineEdit()
    search.setObjectName('ProjectFilter')
    search.setPlaceholderText('Filter scans and plans')
    search.setClearButtonEnabled(True)
    layout.addWidget(search)
    tree = QtWidgets.QTreeWidget()
    tree.setObjectName('ProjectTree')
    tree.setHeaderHidden(True)
    tree.setAlternatingRowColors(True)
    tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    project = QtWidgets.QTreeWidgetItem(['UNTITLED PROJECT'])
    for title, hint in (('SCANS', 'No scans imported'), ('DXF PLAN', 'No plan loaded'), ('ALIGNMENTS', 'No alignment result'), ('SURFACES', 'No surface generated'), ('MEASUREMENTS', 'No measurements')):
        group = QtWidgets.QTreeWidgetItem([title])
        child = QtWidgets.QTreeWidgetItem([hint])
        child.setForeground(0, qt.QtGui.QColor(SKIN_C_DARK['muted']))
        group.addChild(child)
        project.addChild(group)
    tree.addTopLevelItem(project)
    project.setExpanded(True)
    layout.addWidget(tree, 1)
    actions = QtWidgets.QHBoxLayout()
    import_button = QtWidgets.QPushButton('Import')
    import_button.setObjectName('ImportScanButton')
    import_button.setProperty('ccaRole', 'primary')
    import_button.setIcon(_icon('import-points.svg', qt))
    new_group_button = QtWidgets.QPushButton('New group')
    new_group_button.setObjectName('NewGroupButton')
    new_group_button.setIcon(_icon('file-new.svg', qt))
    actions.addWidget(import_button)
    actions.addWidget(new_group_button)
    ungroup_button = QtWidgets.QPushButton('Ungroup')
    ungroup_button.setObjectName('UngroupButton')
    ungroup_button.setIcon(_icon('close.svg', qt))
    actions.addWidget(ungroup_button)
    layout.addLayout(actions)
    return panel

def _property_field(label_text: str, suffix: str, qt: QtModules) -> tuple[object, object]:
    QtWidgets = qt.QtWidgets
    label = QtWidgets.QLabel(label_text)
    label.setObjectName('PropertyKey')
    field = QtWidgets.QDoubleSpinBox()
    field.setDecimals(3)
    field.setRange(-1000000.0, 1000000.0)
    field.setSingleStep(0.05)
    field.setSuffix(suffix)
    return (label, field)

def _make_properties(qt: QtModules, window: object) -> dict[str, tuple[object, ...]]:
    QtWidgets = qt.QtWidgets
    scroll = QtWidgets.QScrollArea()
    scroll.setObjectName('PropertiesDetailsScroll')
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(qt.QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    body = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(body)
    layout.setContentsMargins(10, 10, 10, 12)
    layout.setSpacing(9)

    class ElidedLabel(QtWidgets.QLabel):

        def setText(self, text):
            super().setText(text)
            self.setToolTip(text)

        def paintEvent(self, event):
            painter = qt.QtGui.QPainter(self)
            painter.setPen(self.palette().color(self.foregroundRole()))
            text = self.fontMetrics().elidedText(self.text().replace('\n', ' / '), qt.QtCore.Qt.TextElideMode.ElideRight, self.contentsRect().width())
            painter.drawText(self.contentsRect(), qt.QtCore.Qt.AlignmentFlag.AlignVCenter, text)
    selection = ElidedLabel('NO SELECTION')
    selection.setObjectName('PropertiesSelection')
    selection.setProperty('ccaRole', 'section')
    selection_hint = QtWidgets.QLabel('Select a scan, plan, or result.')
    selection_hint.setObjectName('PropertiesHint')
    selection_hint.setProperty('ccaRole', 'muted')
    selection_hint.setWordWrap(True)
    selection_hint.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
    layout.addWidget(selection)
    layout.addWidget(selection_hint)
    selection_tint = QtWidgets.QCheckBox('Turuncu boyama')
    selection_tint.setObjectName('SelectionHighlightToggle')
    selection_tint.setChecked(True)
    selection_tint.setToolTip('Kapaliyken secim kutusu kalir; RGB ve Patch Color boyanmaz.')
    layout.addWidget(selection_tint)
    layout.addSpacing(8)
    layout.addWidget(_section_label('FIXED-SCALE TRANSFORM', qt))
    transform_grid = QtWidgets.QGridLayout()
    transform_grid.setHorizontalSpacing(8)
    transform_grid.setVerticalSpacing(7)
    for row, (key, suffix) in enumerate((('X', ' m'), ('Y', ' m'), ('Z', ' m'), ('YAW', ' deg'))):
        label, field = _property_field(key, suffix, qt)
        field.setObjectName(f'Transform{key.title()}')
        transform_grid.addWidget(label, row, 0)
        transform_grid.addWidget(field, row, 1)
    scale_key = QtWidgets.QLabel('SCALE')
    scale_key.setObjectName('PropertyKey')
    scale_value = QtWidgets.QLabel('1.000  LOCKED')
    scale_value.setObjectName('ScaleLockValue')
    scale_value.setProperty('ccaRole', 'railMeta')
    transform_grid.addWidget(scale_key, 4, 0)
    transform_grid.addWidget(scale_value, 4, 1)
    layout.addLayout(transform_grid)
    layout.addSpacing(8)
    plan_placement = QtWidgets.QFrame()
    plan_placement.setObjectName('PlanPlacementPanel')
    plan_placement.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
    plan_layout = QtWidgets.QVBoxLayout(plan_placement)
    plan_layout.setContentsMargins(9, 8, 9, 9)
    plan_layout.setSpacing(7)
    plan_layout.addWidget(_section_label('PLAN PLACEMENT', qt))
    plan_status = QtWidgets.QLabel('IDLE / SELECT SCAN + DXF PLAN')
    plan_status.setObjectName('PlanPlacementStatus')
    plan_status.setProperty('ccaRole', 'muted')
    plan_status.setWordWrap(True)
    plan_layout.addWidget(plan_status)
    plan_actions = QtWidgets.QGridLayout()
    plan_actions.setHorizontalSpacing(6)
    plan_actions.setVerticalSpacing(6)
    for label, command, object_name, row, column in (('Plan Place', 'PLAN PLACE', 'PlanPlaceButton', 0, 0), ('Previous Scan', 'PREVIOUS SCAN', 'PlanPreviousScanButton', 0, 1), ('Place + Next', 'PLACE + NEXT', 'PlanPlaceNextButton', 1, 0), ('Finish', 'FINISH', 'PlanFinishButton', 1, 1)):
        button = QtWidgets.QPushButton(label)
        button.setObjectName(object_name)
        button.setProperty('ccaCommand', f'ALIGN:{command}')
        button.setIcon(_command_icon(command, qt))
        button.setEnabled(f'ALIGN:{command}' not in _PLAN_PLACEMENT_INITIAL_DISABLED)
        if command in {'PLAN PLACE', 'PLACE + NEXT'}:
            button.setProperty('ccaRole', 'primary')
        plan_actions.addWidget(button, row, column)
    plan_layout.addLayout(plan_actions)
    layout.addWidget(plan_placement)
    layout.addSpacing(8)
    layout.addWidget(_section_label('DISPLAY', qt))
    display_grid = QtWidgets.QGridLayout()
    point_label = QtWidgets.QLabel('POINT SIZE')
    point_label.setObjectName('PropertyKey')
    point_size = QtWidgets.QSpinBox()
    point_size.setObjectName('PointSizeControl')
    point_size.setRange(1, 12)
    point_size.setValue(4)
    mode_label = QtWidgets.QLabel('COLOR')
    mode_label.setObjectName('PropertyKey')
    color_mode = QtWidgets.QComboBox()
    color_mode.setObjectName('ColorModeControl')
    color_mode.addItems(('PHOTO RGB', 'CONFIDENCE', 'SCAN GROUP', 'HEIGHT'))
    surface_label = QtWidgets.QLabel('SURFACE')
    surface_label.setObjectName('PropertyKey')
    surface_mode = QtWidgets.QComboBox()
    surface_mode.setObjectName('SurfaceDisplayControl')
    surface_mode.addItems(('POINTS ONLY', 'GHOST SURFACE', 'SOLID SURFACE'))
    evidence_label = QtWidgets.QLabel('VERI KAYNAGI')
    evidence_label.setObjectName('PropertyKey')
    from viewer.editing_controls import create_evidence_combo
    evidence_profile = create_evidence_combo(qt)
    evidence_profile.setObjectName('EvidenceProfileControl')
    display_grid.addWidget(point_label, 0, 0)
    display_grid.addWidget(point_size, 0, 1)
    display_grid.addWidget(mode_label, 1, 0)
    display_grid.addWidget(color_mode, 1, 1)
    display_grid.addWidget(surface_label, 2, 0)
    display_grid.addWidget(surface_mode, 2, 1)
    display_grid.addWidget(evidence_label, 3, 0)
    display_grid.addWidget(evidence_profile, 3, 1)
    evidence_status = QtWidgets.QLabel()
    evidence_status.setObjectName('EvidenceAvailability')
    evidence_status.setWordWrap(True)
    evidence_status.setProperty('ccaRole', 'muted')
    display_grid.addWidget(evidence_status, 4, 0, 1, 2)
    correction_button = QtWidgets.QPushButton('CCA donusumunu hesapla')
    correction_button.setObjectName('EvidenceCorrectionButton')
    correction_button.setToolTip('Secili taramanin eslesen RGB/depth/conf karelerini menzil kesmeden sensor-v2 + CAD Z-up ile birlestirir. Model kapaliysa gurultu/bias duzeltmesi uygulamaz. Secilen menzil modeli ham karelere uygulanir ve ayri ciktiya kaydedilir.')
    display_grid.addWidget(correction_button, 5, 0, 1, 2)
    model_status = ElidedLabel('MENZIL: KAPALI / sensor-v2 + CAD Z-up')
    model_status.setObjectName('DepthModelStatus')
    model_status.setWordWrap(True)
    model_status.setProperty('ccaRole', 'muted')
    display_grid.addWidget(model_status, 6, 0, 1, 2)
    model_actions = QtWidgets.QHBoxLayout()
    for name, title in (('DepthModelLoadButton', 'Model sec'), ('DepthModelClearButton', 'Kapat'), ('CorrectionFormulasButton', 'Formuller')):
        button = QtWidgets.QPushButton(title)
        button.setObjectName(name)
        model_actions.addWidget(button)
    display_grid.addLayout(model_actions, 7, 0, 1, 2)
    recovery_button = QtWidgets.QPushButton('Ham noktalarin yerini RGB ile ara')
    recovery_button.setObjectName('RayRecoveryButton')
    recovery_button.setToolTip('Ayni taramada en az uc ek gorusle dogrulanan derinlik adaylarini ayri turetilmis bulutta uygular.')
    display_grid.addWidget(recovery_button, 8, 0, 1, 2)
    structure_button = QtWidgets.QPushButton('Yapi + isin yer adaylarini ara')
    structure_button.setObjectName('StructureSearchButton')
    structure_button.setToolTip('Yerel geometrik duzeni sonlu yuzeyler ve kamera isinlariyla karsilastirir. Yer adaylarini raporlar; ham noktalari tasimaz.')
    display_grid.addWidget(structure_button, 9, 0, 1, 2)
    structure_hint = QtWidgets.QLabel('Buyuk nokta: yerel yuzey destegi. Kesin konum degil.')
    structure_hint.setWordWrap(True)
    structure_hint.setProperty('ccaRole', 'muted')
    display_grid.addWidget(structure_hint, 10, 0, 1, 2)
    usage_button = QtWidgets.QPushButton('Her noktanin hesabini cikar')
    usage_button.setObjectName('PointUsageAuditButton')
    usage_button.setToolTip('Tum PLY noktalarini yuzey aramasina katar; yalniz etiketli sensor dolgusunu ayirir. Cozulemeyen noktalar korunur.')
    display_grid.addWidget(usage_button, 11, 0, 1, 2)
    lefebvre_button = QtWidgets.QPushButton('Lefebvre / Asistan')
    lefebvre_button.setObjectName('LefebvreButton')
    lefebvre_button.setToolTip('Zaman sirasi, ortak alan, hizalama ve Mekanin Uretimi / Patchworking.')
    display_grid.addWidget(lefebvre_button, 12, 0, 1, 2)
    layout.addLayout(display_grid)
    layout.addSpacing(8)
    from viewer.editing_controls import create_evidence_grade_control
    grade_control = create_evidence_grade_control(qt, body)
    layout.addWidget(grade_control)
    layout.addSpacing(8)
    layout.addWidget(_section_label('QUALITY', qt))
    quality_card = QtWidgets.QFrame()
    quality_card.setObjectName('QualityCard')
    quality_layout = QtWidgets.QVBoxLayout(quality_card)
    quality_layout.setContentsMargins(9, 8, 9, 8)
    quality_layout.setSpacing(0)
    quality = QtWidgets.QLabel('Scale: locked 1.000\nAxis: Z-up\nAlignment: not evaluated\nMeasurement: calibration required')
    quality.setObjectName('QualitySummary')
    quality.setProperty('ccaRole', 'muted')
    quality.setWordWrap(True)
    quality.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
    quality_layout.addWidget(quality)
    layout.addWidget(quality_card)
    alignment_review = QtWidgets.QFrame()
    alignment_review.setObjectName('AlignmentReviewPanel')
    alignment_layout = QtWidgets.QVBoxLayout(alignment_review)
    alignment_layout.setContentsMargins(9, 8, 9, 9)
    alignment_layout.setSpacing(7)
    alignment_layout.addWidget(_section_label('ALIGNMENT PREVIEW', qt))
    alignment_summary = ElidedLabel('Select two scans in SCAN RAIL, then run AUTO ALIGN.')
    alignment_summary.setObjectName('AlignmentPreviewSummary')
    alignment_summary.setProperty('ccaRole', 'muted')
    alignment_summary.setWordWrap(True)
    alignment_summary.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
    alignment_layout.addWidget(alignment_summary)
    alignment_actions = QtWidgets.QGridLayout()
    for label, object_name, role, row, column, span in (('Approve candidate', 'AcceptAlignmentButton', 'primary', 0, 0, 2), ('Apply to project', 'ApplyAlignmentButton', 'primary', 1, 0, 2), ('Reject', 'RejectAlignmentButton', 'danger', 2, 0, 1), ('Undo', 'UndoAlignmentButton', 'secondary', 2, 1, 1)):
        button = QtWidgets.QPushButton(label)
        button.setObjectName(object_name)
        button.setProperty('ccaRole', role)
        button.setEnabled(False)
        alignment_actions.addWidget(button, row, column, 1, span)
    alignment_layout.addLayout(alignment_actions)
    layout.addWidget(alignment_review)
    layout.addStretch(1)
    detached = []
    for owner in (layout, display_grid):
        while owner.count():
            item = owner.takeAt(0)
            detached.append(item)
            if item.widget() is not None:
                item.widget().hide()
            if item.layout() is not None:
                item.layout().setParent(None)

    def section(title, inner, width):
        panel = QtWidgets.QWidget()
        panel.setObjectName('RibbonPropertiesGroup')
        panel.setProperty('ccaProperties', True)
        panel.setFixedWidth(width)
        column = QtWidgets.QVBoxLayout(panel)
        column.setContentsMargins(4, 0, 4, 0)
        column.setSpacing(2)
        column.addLayout(inner)
        caption = QtWidgets.QLabel(title)
        caption.setObjectName('RibbonGroupLabel')
        caption.setAlignment(qt.QtCore.Qt.AlignmentFlag.AlignCenter)
        column.addWidget(caption)
        for widget in panel.findChildren(QtWidgets.QWidget):
            widget.show()
        for widget in panel.findChildren(QtWidgets.QAbstractButton):
            widget.setProperty('ccaCompact', True)
        return panel

    def detail_dialog(title, content, object_name):
        dialog = QtWidgets.QDialog(window)
        dialog.setObjectName(object_name)
        dialog.setWindowTitle(f'{PRODUCT_NAME} / {title}')
        dialog.resize(620, 560)
        dialog_layout = QtWidgets.QVBoxLayout(dialog)
        dialog_layout.addWidget(content)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(dialog.reject)
        dialog_layout.addWidget(close)
        return dialog

    def detail_button(title, name, dialog):
        button = QtWidgets.QPushButton(title)
        button.setObjectName(name)
        button.clicked.connect(lambda: (dialog.show(), dialog.raise_()))
        return button
    layout.setContentsMargins(8, 8, 8, 8)
    for widget in (selection_hint, evidence_status, quality_card, grade_control, structure_hint):
        layout.addWidget(widget)
        widget.show()
    layout.addStretch(1)
    scroll.setWidget(body)
    details = detail_dialog('Secim / kaynak / kanit', scroll, 'PropertiesDetailsDialog')
    selection_row = QtWidgets.QGridLayout()
    selection_row.setSpacing(3)
    selection.setFixedWidth(242)
    selection_row.addWidget(selection, 0, 0, 1, 2)
    selection_row.addWidget(selection_tint, 1, 0)
    selection_row.addWidget(detail_button('Ayrinti', 'PropertiesDetailsButton', details), 1, 1)
    basic_display = QtWidgets.QGridLayout()
    basic_display.setSpacing(4)
    for index, (label, control) in enumerate(((point_label, point_size), (mode_label, color_mode), (surface_label, surface_mode), (evidence_label, evidence_profile))):
        row_index, column = divmod(index, 2)
        basic_display.addWidget(label, row_index, 2 * column)
        basic_display.addWidget(control, row_index, 2 * column + 1)
    point_label.setText('Boyut')
    mode_label.setText('Renk')
    surface_label.setText('Yuzey')
    evidence_label.setText('Kaynak')
    transform_items = [transform_grid.takeAt(0) for _ in range(transform_grid.count())]
    transform_grid.setVerticalSpacing(2)
    for index in range(4):
        row_index, column = divmod(index, 2)
        transform_grid.addWidget(transform_items[2 * index].widget(), row_index, 2 * column)
        transform_grid.addWidget(transform_items[2 * index + 1].widget(), row_index, 2 * column + 1)
    scale_key.hide()
    scale_value.hide()
    plan_dialog = detail_dialog('Plan yerlesimi', plan_placement, 'PlanPlacementDetailsDialog')
    transform_grid.addWidget(detail_button('Plan', 'PlanPlacementDetailsButton', plan_dialog), 0, 4)
    transform_grid.addWidget(scale_value, 1, 4)
    approval_items = [alignment_actions.takeAt(0) for _ in range(alignment_actions.count())]
    for index, item in enumerate(approval_items):
        alignment_actions.addWidget(item.widget(), index // 2, index % 2)
    alignment_layout.takeAt(0).widget().deleteLater()
    alignment_layout.takeAt(0)
    alignment_layout.takeAt(0)
    alignment_actions.setParent(None)
    approval_row = QtWidgets.QHBoxLayout()
    alignment_summary.setFixedWidth(220)
    alignment_summary.setToolTip(alignment_summary.text())
    approval_row.addWidget(alignment_summary)
    approval_row.addLayout(alignment_actions)
    alignment_layout.addLayout(approval_row)
    alignment_layout.setContentsMargins(0, 0, 0, 0)
    approval_group = QtWidgets.QVBoxLayout()
    approval_group.addWidget(alignment_review)
    model_grid = QtWidgets.QGridLayout()
    model_grid.setSpacing(3)
    model_status.setFixedWidth(472)
    model_grid.addWidget(model_status, 0, 0, 1, 2)
    model_grid.addLayout(model_actions, 1, 0)
    correction_button.setText('CCA hesapla')
    model_grid.addWidget(correction_button, 1, 1)
    production_grid = QtWidgets.QGridLayout()
    production_grid.setSpacing(3)
    for index, widget in enumerate((recovery_button, structure_button, usage_button, lefebvre_button)):
        production_grid.addWidget(widget, index // 2, index % 2)
    body._detached_layout_items = detached
    return {'REVIEW': (section('Secim', selection_row, 254), section('Gorunum', basic_display, 530)), 'TRANSFORM': (section('Konum / olcek 1.000', transform_grid, 590),), 'ALIGN': (section('Hizalama adayi / onay ve geri alma', approval_group, 550),), 'IMPORT': (section('Menzil modeli / ham kaynak korunur', model_grid, 488),), 'SURFACE': (section('Mekanin Uretimi / turetilmis geometri', production_grid, 570),)}

def _make_scan_rail(qt: QtModules) -> object:
    QtWidgets = qt.QtWidgets
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(panel)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(10)
    summary_card = QtWidgets.QFrame()
    summary_card.setObjectName('ScanRailSummary')
    summary = QtWidgets.QVBoxLayout(summary_card)
    summary.setContentsMargins(9, 6, 10, 6)
    summary.setSpacing(2)
    count = QtWidgets.QLabel('0 SCANS')
    count.setObjectName('ScanCountLabel')
    count.setProperty('ccaRole', 'keyMetric')
    hint = QtWidgets.QLabel('IMPORT -> VALIDATE -> REVIEW -> ALIGN -> QC')
    hint.setObjectName('PipelineHint')
    hint.setProperty('ccaRole', 'railMeta')
    summary.addWidget(count)
    summary.addWidget(hint)
    layout.addWidget(summary_card)
    rail = QtWidgets.QListWidget()
    rail.setObjectName('ScanRailList')
    rail.setFlow(QtWidgets.QListView.Flow.LeftToRight)
    rail.setWrapping(False)
    rail.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    rail.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
    rail.setMaximumHeight(68)
    placeholder = QtWidgets.QListWidgetItem('NO SCANS  /  USE IMPORT > PLY SCANS')
    placeholder.setFlags(placeholder.flags() & ~qt.QtCore.Qt.ItemFlag.ItemIsSelectable)
    rail.addItem(placeholder)
    layout.addWidget(rail, 1)
    review_button = QtWidgets.QPushButton('Review queue')
    review_button.setObjectName('ReviewQueueButton')
    review_button.setIcon(_icon('validate.svg', qt))
    align_button = QtWidgets.QPushButton('Auto align')
    align_button.setObjectName('AutoAlignButton')
    align_button.setProperty('ccaRole', 'primary')
    align_button.setIcon(_icon('align-auto.svg', qt))
    layout.addWidget(review_button)
    layout.addWidget(align_button)
    return panel

def _make_project_home(qt: QtModules) -> object:
    from viewer.welcome import create_welcome_page
    return create_welcome_page(qt)

def _make_viewport(qt: QtModules, adapter: ViewportAdapter) -> tuple[object, object]:
    QtWidgets = qt.QtWidgets
    workspace = QtWidgets.QWidget()
    workspace.setObjectName('WorkspacePage')
    workspace_layout = QtWidgets.QVBoxLayout(workspace)
    workspace_layout.setContentsMargins(0, 0, 0, 0)
    workspace_layout.setSpacing(0)
    frame = QtWidgets.QFrame()
    frame.setObjectName('ViewportFrame')
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    header = QtWidgets.QFrame()
    header.setObjectName('ViewportHeader')
    header_layout = QtWidgets.QVBoxLayout(header)
    header_layout.setContentsMargins(12, 7, 12, 6)
    header_layout.setSpacing(5)
    placement_row = QtWidgets.QHBoxLayout()
    proxy_button = QtWidgets.QPushButton('Yerlesim')
    proxy_button.setObjectName('PlacementProxyButton')
    proxy_button.setCheckable(True)
    proxy_button.setToolTip("Tum taramalari hafif renkli temsile dondurur; konumlar korunur. Gorunurleri Hesapla'nin tersi.")
    calculate_button = QtWidgets.QPushButton('Hesapla')
    calculate_button.setObjectName('CalculatePointsButton')
    calculate_button.setProperty('ccaRole', 'primary')
    calculate_button.setToolTip('Secili taramanin hazir noktalarini yukler; gurultu/bias duzeltmesi uygulamaz.')
    calculate_visible_button = QtWidgets.QPushButton('Gorunurleri Hesapla')
    calculate_visible_button.setObjectName('CalculateVisiblePointsButton')
    calculate_visible_button.setToolTip('Su an gorunur olan her taramayi nokta olarak acar; kapsami gorunurlukle secersiniz. Bellege sigmayanlar hafif temsil kalir ve adlariyla bildirilir. Yerlesim geri alir. Gurultu/bias duzeltmesi uygulamaz.')
    placement_status = QtWidgets.QLabel('TEMSIL / Yerlesim icin hazir')
    placement_status.setObjectName('PlacementModeStatus')
    placement_status.setProperty('ccaRole', 'railMeta')
    placement_status.setWordWrap(True)
    placement_status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
    placement_row.addWidget(proxy_button)
    placement_row.addWidget(calculate_button)
    placement_row.addWidget(calculate_visible_button)
    from viewer.editing_controls import create_command_bar, create_quality_control
    placement_row.addWidget(create_quality_control(qt, frame))
    placement_row.addWidget(placement_status, 1)
    header_layout.addLayout(placement_row)
    data_row = QtWidgets.QHBoxLayout()
    data_row.setSpacing(6)
    title = QtWidgets.QLabel('Perspective  /  Z-up  /  Orbit')
    title.setObjectName('ViewportTitle')
    point_count = QtWidgets.QLabel('0 POINTS')
    point_count.setObjectName('ScanPulseCount')
    point_count.setProperty('ccaRole', 'dataBadge')
    color_mode = QtWidgets.QLabel('PHOTO RGB')
    color_mode.setObjectName('ScanPulseMode')
    color_mode.setProperty('ccaRole', 'dataBadge')
    point_size = QtWidgets.QLabel('4 PX')
    point_size.setObjectName('ScanPulseSize')
    point_size.setProperty('ccaRole', 'dataBadge')
    surface_mode = QtWidgets.QLabel('POINTS ONLY')
    surface_mode.setObjectName('ScanPulseSurface')
    surface_mode.setProperty('ccaRole', 'surfaceBadge')
    data_row.addWidget(title)
    data_row.addStretch(1)
    data_row.addWidget(point_count)
    data_row.addWidget(color_mode)
    data_row.addWidget(point_size)
    data_row.addWidget(surface_mode)
    header_layout.addLayout(data_row)
    navigation_row = QtWidgets.QHBoxLayout()
    navigation_row.setSpacing(8)
    pulse_label = QtWidgets.QLabel('SCAN PULSE')
    pulse_label.setObjectName('ScanPulseLabel')
    pulse_label.setProperty('ccaRole', 'keyMetric')
    navigation = QtWidgets.QLabel('LMB secim   MMB orbit   Shift+MMB pan   Wheel zoom   Space/Enter onay   OP ayarlar')
    navigation.setObjectName('ViewportNavigationHint')
    navigation.setProperty('ccaRole', 'railMeta')
    navigation.setWordWrap(True)
    navigation.setToolTip('Left/middle drag: orbit around the clicked point. Right drag or Shift+middle drag: pan. Wheel: zoom towards the cursor. Click without dragging: select. Home: fit all. F: fit selected scan/group. 1: front. 3: right. 7: top. 0: perspective. 5: orthographic/perspective. G: move. R: rotate Z. Esc: cancel.')
    transform_mode = QtWidgets.QLabel('SELECT')
    transform_mode.setObjectName('ViewportTransformMode')
    transform_mode.setProperty('ccaRole', 'dataBadge')
    navigation_row.addWidget(pulse_label)
    navigation_row.addStretch(1)
    navigation_row.addWidget(transform_mode)
    navigation_row.addWidget(navigation)
    header_layout.addLayout(navigation_row)
    layout.addWidget(header)
    pulse = QtWidgets.QFrame()
    pulse.setObjectName('ScanPulseRail')
    layout.addWidget(pulse)
    viewport = adapter.create_widget(frame, qt)
    if not isinstance(viewport, QtWidgets.QWidget):
        raise TypeError('viewport adapter must return a QWidget')
    layout.addWidget(viewport, 1)
    layout.addWidget(create_command_bar(qt, frame))
    workspace_layout.addWidget(frame)
    return (workspace, viewport)

def _ribbon_page(tab: str, qt: QtModules, property_groups: tuple[object, ...]=()) -> object:
    QtCore, QtWidgets = (qt.QtCore, qt.QtWidgets)
    page = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(page)
    row.setContentsMargins(8, 4, 8, 3)
    row.setSpacing(5)
    row.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)
    for property_group in property_groups:
        row.addWidget(property_group)
    for group_index, (group_name, commands) in enumerate(_RIBBON_COMMANDS[tab]):
        if group_index or property_groups:
            separator = QtWidgets.QFrame()
            separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
            separator.setStyleSheet('color: #353535;')
            row.addWidget(separator)
        group = QtWidgets.QWidget()
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(1)
        buttons = QtWidgets.QGridLayout()
        buttons.setSpacing(1)
        for index, command in enumerate(commands):
            command_id = _ribbon_command_id(tab, command)
            button = QtWidgets.QPushButton(_command_label(command))
            button.setObjectName('RibbonCommand')
            button.setProperty('ccaCommand', command_id)
            button.setToolTip(f'{SHELL_TAB_LABELS[tab]} / {_command_label(command)}')
            button.setIcon(_command_icon(command, qt))
            button.setIconSize(QtCore.QSize(18, 18))
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            if command_id in _PLAN_PLACEMENT_INITIAL_DISABLED:
                button.setEnabled(False)
            if command_id in {'ALIGN:SELECT', 'ALIGN:MOVE', 'ALIGN:ROTATE Z', 'ALIGN:LOCK'}:
                button.setCheckable(True)
                button.setChecked(command_id == 'ALIGN:SELECT')
            buttons.addWidget(button, index % 2, index // 2)
        caption = QtWidgets.QLabel(group_name)
        caption.setObjectName('RibbonGroupLabel')
        caption.setAlignment(qt.QtCore.Qt.AlignmentFlag.AlignCenter)
        group_layout.addLayout(buttons)
        group_layout.addWidget(caption)
        row.addWidget(group)
    row.addStretch(1)
    scroll = QtWidgets.QScrollArea()
    scroll.setObjectName('RibbonScroll')
    scroll.setProperty('ccaWorkflow', tab)
    scroll.setAccessibleName(f'{SHELL_TAB_LABELS[tab]} komutlari')
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(page)
    scroll.setWidgetResizable(False)
    return scroll

def _add_top_ribbon(window: object, qt: QtModules, brand_image: Path) -> tuple[object, object]:
    QtCore, QtGui, QtWidgets = (qt.QtCore, qt.QtGui, qt.QtWidgets)
    ribbon = QtWidgets.QToolBar('WORKFLOW', window)
    ribbon.setObjectName('TopRibbon')
    ribbon.setMovable(False)
    ribbon.setFloatable(False)
    ribbon.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.PreventContextMenu)
    ribbon.setMinimumHeight(38)
    ribbon.setMaximumHeight(38)
    logo = QtWidgets.QLabel()
    logo_pixmap = QtGui.QPixmap(str(brand_image)).scaled(78, 22, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
    logo.setPixmap(logo_pixmap)
    logo.setFixedSize(88, 36)
    logo.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
    logo.setAccessibleName(f'CCA / {ORGANIZATION_NAME}')
    logo.setToolTip(f'CCA / {ORGANIZATION_NAME}\n{CREATIVE_NAME}\n{PRODUCT_NAME}')
    ribbon.addWidget(logo)
    divider = QtWidgets.QFrame()
    divider.setFixedSize(1, 22)
    divider.setStyleSheet('background: #46484D;')
    ribbon.addWidget(divider)
    product_title = QtWidgets.QLabel(PANEL_TITLE.upper())
    product_title.setObjectName('ProductTitle')
    product_title.setFixedWidth(144)
    product_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    ribbon.addWidget(product_title)
    brand_divider = QtWidgets.QFrame()
    brand_divider.setFixedSize(1, 22)
    brand_divider.setStyleSheet('background: #46484D;')
    ribbon.addWidget(brand_divider)
    tabs = QtWidgets.QTabBar()
    tabs.setObjectName('WorkflowTabs')
    tabs.setDrawBase(False)
    tabs.setExpanding(False)
    tabs.setUsesScrollButtons(True)
    for tab in SHELL_TABS:
        index = tabs.addTab(SHELL_TAB_LABELS[tab])
        tabs.setTabData(index, tab)
    ribbon.addWidget(tabs)
    spacer = QtWidgets.QWidget()
    spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
    ribbon.addWidget(spacer)
    command = QtWidgets.QPushButton('Command  Ctrl+K')
    command.setObjectName('CommandPaletteButton')
    command.setShortcut(QtGui.QKeySequence('Ctrl+K'))
    command.setToolTip('Open verified command palette (Ctrl+K)')
    command.setIcon(_icon('command.svg', qt))
    command.setIconSize(QtCore.QSize(16, 16))
    command.setToolTip('Open the command palette')
    ribbon.addWidget(command)
    lefebvre_agent = QtWidgets.QPushButton('Lefebvre')
    lefebvre_agent.setObjectName('LefebvreAgentButton')
    lefebvre_agent.setToolTip('CCA Lefebvre: zaman sirasi, ortak alan, hizalama ve Patchworking')
    ribbon.addWidget(lefebvre_agent)
    help_button = QtWidgets.QPushButton('Yardim  F1')
    help_button.setObjectName('HelpButton')
    help_action = QtGui.QAction('Yardim', window)
    help_action.setObjectName('HelpAction')
    help_action.setShortcut(QtGui.QKeySequence('F1'))
    window.addAction(help_action)
    help_button.setToolTip('Cevrimdisi kullanim, formuller ve kisayollar')
    ribbon.addWidget(help_button)
    window.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, ribbon)
    command_ribbon = QtWidgets.QToolBar('COMMANDS', window)
    command_ribbon.setObjectName('CommandRibbon')
    command_ribbon.setMovable(False)
    command_ribbon.setFloatable(False)
    command_ribbon.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.PreventContextMenu)
    command_ribbon.setFixedHeight(104)
    stack = QtWidgets.QStackedWidget()
    stack.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
    properties = _make_properties(qt, window)
    for tab in SHELL_TABS:
        stack.addWidget(_ribbon_page(tab, qt, properties.get(tab, ())))
    window._cca_property_groups = tuple((group for groups in properties.values() for group in groups))

    def show_context(workflow: str) -> None:
        if workflow not in SHELL_TABS:
            raise ValueError(f'Unknown workflow: {workflow}')
        tabs.setCurrentIndex(SHELL_TABS.index(workflow))
        stack.currentWidget().horizontalScrollBar().setValue(0)
    window._cca_show_context = show_context
    command_ribbon.addWidget(stack)
    window.addToolBarBreak(QtCore.Qt.ToolBarArea.TopToolBarArea)
    window.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, command_ribbon)
    tabs.currentChanged.connect(stack.setCurrentIndex)
    return (tabs, stack)

def _show_settings_dialog(window: object, qt: QtModules) -> None:
    QtWidgets = qt.QtWidgets
    existing = getattr(window, '_cca_points_settings_dialog', None)
    if existing is not None:
        existing.show()
        existing.raise_()
        existing.activateWindow()
        return
    dialog = QtWidgets.QDialog(window)
    dialog.setObjectName('SettingsDialog')
    dialog.setWindowTitle(f'{PRODUCT_NAME} / Ayarlar')
    dialog.setModal(False)
    dialog.setMinimumWidth(540)
    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)
    title = QtWidgets.QLabel('Workstation settings')
    title.setObjectName('DialogTitle')
    hint = QtWidgets.QLabel('Adjust the shell without changing metric geometry or source evidence.')
    hint.setProperty('ccaRole', 'muted')
    hint.setWordWrap(True)
    layout.addWidget(title)
    layout.addWidget(hint)
    ribbon = window.findChild(QtWidgets.QToolBar, 'CommandRibbon')
    ribbon_toggle = QtWidgets.QCheckBox('Show command ribbon')
    ribbon_toggle.setChecked(bool(ribbon and ribbon.isVisible()))
    if ribbon is not None:
        ribbon_toggle.toggled.connect(ribbon.setVisible)
    layout.addWidget(ribbon_toggle)
    navigation = window.findChild(QtWidgets.QLabel, 'ViewportNavigationHint')
    navigation_toggle = QtWidgets.QCheckBox('Show viewport navigation hints')
    navigation_toggle.setChecked(bool(navigation and navigation.isVisible()))
    if navigation is not None:
        navigation_toggle.toggled.connect(navigation.setVisible)
    layout.addWidget(navigation_toggle)
    cad = getattr(window, '_cca_cad_input', None)
    if cad is not None:
        from viewer.cad_preferences import create_preferences_widget
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(390)
        scroll.setWidget(create_preferences_widget(qt, cad.preferences, cad.apply))
        layout.addWidget(scroll)
    contract = QtWidgets.QFrame()
    contract.setObjectName('SettingsContractPanel')
    contract_layout = QtWidgets.QFormLayout(contract)
    contract_layout.setContentsMargins(10, 10, 10, 10)
    contract_layout.addRow('Internal unit', QtWidgets.QLabel('metre  /  locked'))
    contract_layout.addRow('CAD axis', QtWidgets.QLabel('Z-up  /  locked'))
    contract_layout.addRow('Scale', QtWidgets.QLabel('1.000  /  locked'))
    layout.addWidget(contract)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.finished.connect(lambda _result: setattr(window, '_cca_points_settings_dialog', None))
    window._cca_points_settings_dialog = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()

def _add_menu_bar(window: object, qt: QtModules, docks: tuple[object, object], command_ribbon: object) -> object:
    QtGui, QtWidgets = (qt.QtGui, qt.QtWidgets)
    menu_bar = window.menuBar()
    menu_bar.setObjectName('MainMenuBar')
    menu_bar.setNativeMenuBar(False)

    def menu(title: str) -> object:
        result = menu_bar.addMenu(title)
        result.setObjectName(f'{title}Menu')
        return result

    def command_action(owner: object, label: str, command: str, *, object_name: str, shortcut: str | None=None) -> object:
        action = QtGui.QAction(_command_icon(command.split(':', 1)[1], qt), label, window)
        action.setObjectName(object_name)
        action.setProperty('ccaCommand', command)
        if command in _PLAN_PLACEMENT_INITIAL_DISABLED or command.startswith('MEASURE:'):
            action.setEnabled(False)
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))

        def trigger() -> None:
            for button in window.findChildren(QtWidgets.QPushButton):
                if button.property('ccaCommand') == command:
                    button.click()
                    return
        action.triggered.connect(trigger)
        owner.addAction(action)
        return action
    file_menu = menu('File')
    command_action(file_menu, 'New Project', 'PROJECT:NEW', object_name='NewProjectAction', shortcut='Ctrl+N')
    command_action(file_menu, 'Open Project...', 'PROJECT:OPEN', object_name='OpenProjectAction', shortcut='Ctrl+O')
    command_action(file_menu, 'Save Project', 'PROJECT:SAVE', object_name='SaveProjectAction', shortcut='Ctrl+S')
    file_menu.addSeparator()
    command_action(file_menu, 'Import Point Clouds...', 'IMPORT:PLY SCANS', object_name='ImportPointCloudAction', shortcut='Ctrl+I')
    command_action(file_menu, 'Process Raw All Data...', 'IMPORT:RAW ALL DATA', object_name='RawAllDataAction')
    command_action(file_menu, 'Import DXF Plan...', 'IMPORT:DXF PLAN', object_name='ImportDxfAction')
    file_menu.addSeparator()
    exit_action = file_menu.addAction(_icon('close.svg', qt), 'Exit')
    exit_action.setObjectName('ExitAction')
    exit_action.setShortcut(QtGui.QKeySequence('Alt+F4'))
    exit_action.triggered.connect(window.close)
    edit_menu = menu('Edit')
    undo_action = edit_menu.addAction(_icon('undo.svg', qt), 'Undo Placement / Alignment')
    undo_action.setObjectName('UndoAlignmentAction')
    undo_action.setShortcut(QtGui.QKeySequence('Ctrl+Z'))
    undo_action.triggered.connect(lambda: window.findChild(QtWidgets.QPushButton, 'UndoAlignmentButton').click())
    redo_action = edit_menu.addAction(_icon('redo.svg', qt), 'Redo Placement / Alignment')
    redo_action.setObjectName('RedoAlignmentAction')
    redo_action.setShortcut(QtGui.QKeySequence('Ctrl+Y'))
    redo_action.setEnabled(False)
    redo_action.triggered.connect(lambda: getattr(window, '_cca_redo_alignment', lambda: None)())
    edit_menu.addSeparator()
    settings_action = edit_menu.addAction(_icon('settings.svg', qt), 'Settings...')
    settings_action.setObjectName('SettingsAction')
    settings_action.setShortcut(QtGui.QKeySequence('Ctrl+,'))
    settings_action.triggered.connect(lambda: _show_settings_dialog(window, qt))
    view_menu = menu('View')
    for label, dock, object_name in (('Tarama akisi / liste', docks[0], 'ToggleProjectBrowserAction'), ('Tarama seridi', docks[1], 'ToggleScanRailAction'), ('Komut seridi', command_ribbon, 'ToggleRibbonAction')):
        action = view_menu.addAction(label)
        action.setObjectName(object_name)
        action.setCheckable(True)
        action.setChecked(not dock.isHidden())
        action.toggled.connect(dock.setVisible)
        dock.visibilityChanged.connect(action.setChecked)
    properties_action = view_menu.addAction('Toolbar ozellikleri')
    properties_action.setObjectName('TogglePropertiesAction')
    properties_action.setCheckable(True)
    properties_action.setChecked(True)

    def toggle_properties(visible: bool) -> None:
        for group in window._cca_property_groups:
            group.setVisible(visible)
        if visible:
            command_ribbon.show()
            window._cca_show_context('REVIEW')
    properties_action.toggled.connect(toggle_properties)
    view_menu.addSeparator()
    for token, name in (('ZOOM EXTENTS', 'ZoomExtentsAction'), ('FIT SELECTED', 'FitSelectedAction'), ('TOP', 'TopViewAction'), ('FRONT', 'FrontViewAction'), ('RIGHT', 'RightViewAction'), ('PERSPECTIVE', 'PerspectiveViewAction')):
        command_action(view_menu, _command_label(token), f'REVIEW:{token}', object_name=name)
    view_menu.addSeparator()
    fullscreen_action = view_menu.addAction('Full Screen')
    fullscreen_action.setObjectName('FullScreenAction')
    fullscreen_action.setShortcut(QtGui.QKeySequence('F11'))
    fullscreen_action.setCheckable(True)
    fullscreen_action.toggled.connect(lambda checked: window.showFullScreen() if checked else window.showNormal())
    process_menu = menu('Process')
    command_action(process_menu, 'Secili Taramanin CCA Donusumunu Hesapla', 'IMPORT:APPLY CORRECTION', object_name='ApplyCorrectionAction')
    command_action(process_menu, 'Validate Inputs', 'IMPORT:VALIDATE', object_name='ValidateInputsAction')
    command_action(process_menu, 'Review Unit and Axis', 'IMPORT:UNIT / AXIS', object_name='UnitAxisAction')
    process_menu.addSeparator()
    command_action(process_menu, 'Local BPA Preview', 'SURFACE:LOCAL BPA', object_name='LocalBpaAction')
    command_action(process_menu, 'Poisson Preview', 'SURFACE:POISSON', object_name='PoissonAction')
    command_action(process_menu, 'TSDF Status', 'SURFACE:TSDF STATUS', object_name='TsdfStatusAction')
    align_menu = menu('Align')
    command_action(align_menu, 'Select', 'ALIGN:SELECT', object_name='SelectTransformAction')
    command_action(align_menu, 'Move', 'ALIGN:MOVE', object_name='MoveTransformAction', shortcut='G')
    command_action(align_menu, 'Rotate Z', 'ALIGN:ROTATE Z', object_name='RotateZTransformAction', shortcut='R')
    align_menu.addSeparator()
    command_action(align_menu, 'Plan Place', 'ALIGN:PLAN PLACE', object_name='PlanPlaceAction')
    command_action(align_menu, 'Previous Scan', 'ALIGN:PREVIOUS SCAN', object_name='PlanPreviousScanAction')
    command_action(align_menu, 'Place + Next', 'ALIGN:PLACE + NEXT', object_name='PlanPlaceNextAction')
    command_action(align_menu, 'Finish Plan Placement', 'ALIGN:FINISH', object_name='PlanFinishAction')
    align_menu.addSeparator()
    command_action(align_menu, 'Point Pairs', 'ALIGN:POINT PAIRS', object_name='PointPairsAction')
    command_action(align_menu, 'Find Overlap', 'ALIGN:FIND OVERLAP', object_name='FindOverlapAction')
    command_action(align_menu, 'Auto Align', 'ALIGN:AUTO ALIGN', object_name='AutoAlignAction')
    command_action(align_menu, 'Zaman zinciri', 'ALIGN:TIME CHAIN', object_name='TimeChainAction')
    command_action(align_menu, 'Pair Variations', 'ALIGN:PAIR VARIATIONS', object_name='PairVariationsAction')
    command_action(align_menu, 'Plan Fit', 'ALIGN:PLAN FIT', object_name='PlanFitAction')
    command_action(align_menu, 'Previous Variation', 'ALIGN:PREVIOUS', object_name='PreviousVariationAction')
    command_action(align_menu, 'Next Variation', 'ALIGN:NEXT', object_name='NextVariationAction')
    command_action(align_menu, 'Use Candidate', 'ALIGN:USE CANDIDATE', object_name='UseCandidateAction')
    align_menu.addSeparator()
    command_action(align_menu, 'ICP', 'ALIGN:ICP', object_name='IcpAction')
    command_action(align_menu, 'Colored ICP', 'ALIGN:COLORED ICP', object_name='ColoredIcpAction')
    command_action(align_menu, 'Pose Graph', 'ALIGN:POSE GRAPH', object_name='PoseGraphAction')
    measure_menu = menu('Measure')
    for label, command, object_name in (('Distance', 'MEASURE:DISTANCE', 'DistanceAction'), ('Angle', 'MEASURE:ANGLE', 'AngleAction'), ('Area', 'MEASURE:AREA', 'AreaAction'), ('Elevation', 'MEASURE:ELEVATION', 'ElevationAction'), ('Clearance', 'MEASURE:CLEARANCE', 'ClearanceAction')):
        command_action(measure_menu, label, command, object_name=object_name)
    help_menu = menu('Help')
    command_action(help_menu, f'{PRODUCT_NAME} hakkinda', 'PROJECT:ABOUT CCA', object_name='AboutAction')
    window._cca_points_settings_action = settings_action
    return menu_bar

def _make_dock(title: str, widget: object, area: object, window: object, qt: QtModules) -> object:
    QtWidgets = qt.QtWidgets
    dock = QtWidgets.QDockWidget(title, window)
    dock.setObjectName(title.replace(' ', '_'))
    dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
    dock.setWidget(widget)
    window.addDockWidget(area, dock)
    return dock

def create_shell_window(viewport_adapter: ViewportAdapter | None=None, qt: QtModules | None=None) -> object:
    modules = qt or load_qt_modules()
    QtCore, QtGui, QtWidgets = (modules.QtCore, modules.QtGui, modules.QtWidgets)
    contract = shell_contract()
    adapter = viewport_adapter or PlaceholderViewportAdapter()
    window = QtWidgets.QMainWindow()
    window.setObjectName('CCAPointsShell')
    window.setWindowTitle(PRODUCT_NAME)
    window.setWindowIcon(QtGui.QIcon(str(load_brand_assets().company_svg)))
    available = QtGui.QGuiApplication.primaryScreen().availableGeometry()
    window.resize(min(1440, available.width() - 40), min(900, available.height() - 60))
    window.setMinimumSize(1024, 680)
    window.setDockNestingEnabled(True)
    window.setDockOptions(QtWidgets.QMainWindow.DockOption.AllowNestedDocks | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks)
    window.setStyleSheet(skin_c_stylesheet())
    tabs, ribbon_stack = _add_top_ribbon(window, modules, contract.brand_image)
    project_home = _make_project_home(modules)
    workspace_page, viewport = _make_viewport(modules, adapter)
    content_stack = QtWidgets.QStackedWidget()
    content_stack.setObjectName('MainContentStack')
    content_stack.addWidget(project_home)
    content_stack.addWidget(workspace_page)
    content_stack.setCurrentWidget(project_home)
    window.setCentralWidget(content_stack)
    from viewer.scan_graph import create_scan_graph_widget
    browser_tabs = QtWidgets.QTabWidget()
    browser_tabs.setObjectName('ProjectBrowserTabs')
    browser_tabs.setDocumentMode(True)
    scan_graph = create_scan_graph_widget(parent=browser_tabs, qt=modules)
    browser_tabs.addTab(scan_graph, 'Akis')
    browser_tabs.addTab(_make_project_browser(modules), 'Liste')
    browser_tabs.setCurrentIndex(0)
    left_dock = _make_dock(SHELL_PANELS['left'], browser_tabs, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, window, modules)
    left_dock.setWindowTitle('Tarama akisi')
    left_dock.setMinimumWidth(300)
    left_dock.resize(380, left_dock.height())
    rail_dock = _make_dock(SHELL_PANELS['bottom'], _make_scan_rail(modules), QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, window, modules)
    rail_dock.setMinimumHeight(112)
    rail_dock.setWindowTitle('Tarama seridi')
    rail_dock.setMaximumHeight(176)
    window.resizeDocks([left_dock], [320], QtCore.Qt.Orientation.Horizontal)
    window.resizeDocks([rail_dock], [112], QtCore.Qt.Orientation.Vertical)
    command_ribbon = window.findChild(QtWidgets.QToolBar, 'CommandRibbon')
    if command_ribbon is None:
        raise RuntimeError('CCA Points command ribbon was not created')
    _add_menu_bar(window, modules, (left_dock, rail_dock), command_ribbon)
    dock_preferences = {left_dock: True, rail_dock: False}
    chrome_syncing = False

    def remember_dock_visibility(dock: object) -> None:
        if not chrome_syncing and content_stack.currentIndex() == 1:
            dock_preferences[dock] = not dock.isHidden()
    for dock in dock_preferences:
        dock.visibilityChanged.connect(lambda _visible, panel=dock: remember_dock_visibility(panel))

    def sync_workspace_chrome(index: int) -> None:
        nonlocal chrome_syncing
        chrome_syncing = True
        try:
            for dock, preferred in dock_preferences.items():
                dock.setVisible(index == 1 and preferred)
        finally:
            chrome_syncing = False
    content_stack.currentChanged.connect(sync_workspace_chrome)
    sync_workspace_chrome(content_stack.currentIndex())
    status = window.statusBar()
    mode = QtWidgets.QLabel('READY')
    mode.setObjectName('StatusMode')
    mode.setStyleSheet(f"color: {SKIN_C_DARK['approval']};")
    project = QtWidgets.QLabel('NO PROJECT')
    project.setObjectName('ProjectStatusLabel')
    project.setProperty('ccaRole', 'railMeta')
    coordinates = QtWidgets.QLabel('X 0.000 m   Y 0.000 m   Z 0.000 m')
    coordinates.setObjectName('RailMeta')
    status.addWidget(mode)
    status.addWidget(project)
    status.addPermanentWidget(coordinates)

    def update_mode(index: int) -> None:
        selected = SHELL_TABS[index] if 0 <= index < len(SHELL_TABS) else 'PROJECT'
        mode.setText(f'READY  /  {selected}')
    tabs.currentChanged.connect(update_mode)
    window._cca_points_contract = contract
    window._cca_points_tabs = tabs
    window._cca_points_ribbon_stack = ribbon_stack
    window._cca_points_viewport = viewport
    window._cca_points_viewport_adapter = adapter
    window._cca_points_content_stack = content_stack
    window._cca_points_project_home = project_home
    window._cca_points_workspace_page = workspace_page
    window._cca_points_scan_graph = scan_graph
    window._cca_points_browser_tabs = browser_tabs
    return window

def _smoke_payload() -> dict[str, object]:
    payload = shell_contract().as_dict()
    payload.update({'status': 'contract-ok', 'qt_available': qt_available(), 'style': 'skin-c-square', 'viewport': 'adapter-slot'})
    return payload

def main(argv: Sequence[str] | None=None) -> int:
    parser = argparse.ArgumentParser(prog='cca-points-qt-shell')
    parser.add_argument('--smoke', action='store_true', help='validate the shell contract without importing PySide6')
    args = parser.parse_args(argv)
    if args.smoke:
        print(json.dumps(_smoke_payload(), sort_keys=True))
        return 0
    try:
        qt = load_qt_modules()
    except QtUnavailableError as error:
        print(str(error), file=sys.stderr)
        return 2
    app = qt.QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = qt.QtWidgets.QApplication([sys.argv[0]])
    app.setApplicationName(PRODUCT_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)
    if 'Fusion' in qt.QtWidgets.QStyleFactory.keys():
        app.setStyle('Fusion')
    window = create_shell_window(qt=qt)
    window.show()
    if not owns_app:
        _EMBEDDED_WINDOWS.append(window)

        def release_window() -> None:
            if window in _EMBEDDED_WINDOWS:
                _EMBEDDED_WINDOWS.remove(window)
        window.destroyed.connect(release_window)
        return 0
    return int(app.exec())
if __name__ == '__main__':
    raise SystemExit(main())
__all__ = ['PlaceholderViewportAdapter', 'QtModules', 'QtUnavailableError', 'ShellContract', 'ViewportAdapter', 'create_shell_window', 'load_qt_modules', 'main', 'qt_available', 'shell_contract']
