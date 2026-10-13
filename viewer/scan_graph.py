
from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from viewer.brand import SKIN_C_DARK
from viewer.project_model import UNGROUPED
SCAN_WIDTH = 164.0
GROUP_WIDTH = 156.0
NODE_HEIGHT = 82.0

@dataclass(frozen=True, slots=True)
class GraphNode:
    key: str
    kind: str
    node_id: str
    label: str
    scan_ids: tuple[str, ...]
    visible: bool

@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    nodes: tuple[GraphNode, ...]
    edges: tuple[tuple[str, str], ...]
    positions: dict[str, tuple[float, float]]

def graph_snapshot(project: Any) -> GraphSnapshot:
    scans = sorted(project.scans, key=lambda scan: (scan.rail_order, scan.scan_id))
    groups = {group.group_id: group for group in project.groups if group.group_id != UNGROUPED}
    members: dict[str, list[str]] = {group_id: [] for group_id in groups}
    nodes: list[GraphNode] = []
    edges: list[tuple[str, str]] = []
    positions: dict[str, tuple[float, float]] = {}
    by_scan = {scan.scan_id: scan for scan in scans}
    for index, scan in enumerate(scans):
        key = f'scan:{scan.scan_id}'
        label = scan.metadata.get('label') or scan.metadata.get('name') or scan.scan_id
        nodes.append(GraphNode(key, 'scan', scan.scan_id, str(label), (scan.scan_id,), scan.visible))
        positions[key] = (16.0, 16.0 + index * 106.0)
        if scan.group_id in groups:
            members[scan.group_id].append(scan.scan_id)
            edges.append((key, f'group:{scan.group_id}'))
    occupied_group_y: list[float] = []
    for index, (group_id, group) in enumerate(sorted(groups.items())):
        scan_ids = tuple(members[group_id])
        key = f'group:{group_id}'
        nodes.append(GraphNode(key, 'group', group_id, group.label, scan_ids, any((by_scan[scan_id].visible for scan_id in scan_ids))))
        y = sum((positions[f'scan:{scan_id}'][1] for scan_id in scan_ids)) / len(scan_ids) if scan_ids else 16.0 + index * 106.0
        while any((abs(y - other) < 100 for other in occupied_group_y)):
            y += 106.0
        occupied_group_y.append(y)
        positions[key] = (222.0, y)
    metadata = project.metadata.get('scan_graph', {})
    saved = metadata.get('positions', {}) if isinstance(metadata, Mapping) else {}
    if isinstance(saved, Mapping):
        for key, value in saved.items():
            if key not in positions or not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                continue
            if len(value) != 2 or any((isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)):
                continue
            if all((math.isfinite(float(item)) for item in value)):
                positions[key] = (float(value[0]), float(value[1]))
    return GraphSnapshot(tuple(nodes), tuple(edges), positions)

def create_scan_graph_widget(parent: object=None, qt: object=None) -> object:
    if qt is None:
        from viewer.qt_shell import load_qt_modules
        qt = load_qt_modules()
    core, gui, widgets = (qt.QtCore, qt.QtGui, qt.QtWidgets)
    color = SKIN_C_DARK

    def path_between(start, end):
        path = gui.QPainterPath(start)
        reach = max(35.0, abs(end.x() - start.x()) * 0.5)
        path.cubicTo(start + core.QPointF(reach, 0), end - core.QPointF(reach, 0), end)
        return path

    class PortItem(widgets.QGraphicsEllipseItem):

        def __init__(self, node):
            super().__init__(-5, -5, 10, 10, node)
            self.node = node
            self.is_output = node.spec.kind == 'scan'
            self.setPos(node.width if self.is_output else 0, 60)
            self.setBrush(gui.QColor(color['surface']))
            self.setPen(gui.QPen(gui.QColor(color['ink']), 1.5))
            self.setCursor(core.Qt.CursorShape.CrossCursor)
            self.setToolTip('SCAN cikisi: grup girisine surukle' if self.is_output else 'SCAN[] girisi: bir veya daha fazla tarama')

        def shape(self):
            path = gui.QPainterPath()
            path.addEllipse(core.QRectF(-9, -9, 18, 18))
            return path

        def boundingRect(self):
            return core.QRectF(-9, -9, 18, 18)

    class NodeItem(widgets.QGraphicsObject):

        def __init__(self, owner, spec):
            super().__init__()
            self.owner, self.spec = (owner, spec)
            self.width = SCAN_WIDTH if spec.kind == 'scan' else GROUP_WIDTH
            self.member_highlight = False
            self.setFlags(widgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable | widgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | widgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
            self.setCacheMode(widgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
            self.setCursor(core.Qt.CursorShape.OpenHandCursor)
            self.port = PortItem(self)
            self.setToolTip(f'{spec.label}\n{spec.node_id}\nCift tik: ' + ('grup adini degistir' if spec.kind == 'group' else "3B'de odaklan"))

        def boundingRect(self):
            return core.QRectF(-2, -2, self.width + 4, NODE_HEIGHT + 4)

        def paint(self, painter, option, widget=None):
            del option, widget
            selected = self.isSelected()
            accent = selected or self.member_highlight
            painter.setPen(gui.QPen(gui.QColor(color['accent'] if accent else '#626266'), 2.0 if selected else 1.0))
            painter.setBrush(gui.QColor('#303034' if self.spec.kind == 'group' else '#242427'))
            painter.drawRect(core.QRectF(0, 0, self.width, NODE_HEIGHT))
            painter.fillRect(core.QRectF(1, 1, self.width - 2, 21), gui.QColor('#3B3B40' if self.spec.kind == 'group' else '#303034'))
            painter.fillRect(core.QRectF(0, 0, 3, 22), gui.QColor(color['accent']))
            painter.setPen(gui.QColor('#626266'))
            painter.drawLine(core.QPointF(0, 23), core.QPointF(self.width, 23))
            painter.setPen(gui.QPen(gui.QColor('#C8C8C8'), 1.0))
            if self.spec.kind == 'scan':
                painter.setBrush(gui.QColor('#C8C8C8'))
                for x, y in ((10, 8), (14, 8), (10, 12), (14, 12), (18, 12), (14, 16), (18, 16)):
                    painter.drawEllipse(core.QPointF(x, y), 1.0, 1.0)
            else:
                painter.setBrush(core.Qt.BrushStyle.NoBrush)
                painter.drawRect(core.QRectF(9, 7, 8, 7))
                painter.drawRect(core.QRectF(13, 10, 8, 7))
            font = gui.QFont('Consolas', 8)
            painter.setFont(font)
            painter.setPen(gui.QColor('#C0C0C3'))
            painter.drawText(core.QRectF(28, 3, self.width - 35, 18), core.Qt.AlignmentFlag.AlignVCenter, 'TARAMA / SCAN' if self.spec.kind == 'scan' else 'GRUP / SCAN[]')
            font = gui.QFont('Segoe UI', 9)
            font.setWeight(gui.QFont.Weight.DemiBold)
            painter.setFont(font)
            painter.setPen(gui.QColor('#F5F5F5'))
            title = gui.QFontMetrics(font).elidedText(self.spec.label, core.Qt.TextElideMode.ElideMiddle, int(self.width - 20))
            painter.drawText(core.QRectF(10, 29, self.width - 20, 21), core.Qt.AlignmentFlag.AlignVCenter, title)
            painter.setFont(gui.QFont('Consolas', 8))
            painter.setPen(gui.QColor('#B8B8BA'))
            if self.spec.kind == 'scan':
                visible = self.owner._visibility.get(self.spec.node_id, self.spec.visible)
                detail = 'Gorunur' if visible else 'Gizli'
            else:
                detail = f'{len(self.spec.scan_ids)} tarama'
            painter.drawText(core.QRectF(12, 53, self.width - 26, 20), core.Qt.AlignmentFlag.AlignVCenter, detail)

        def itemChange(self, change, value):
            result = super().itemChange(change, value)
            if change == widgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and (not self.owner._syncing):
                self.owner._refresh_edges()
            return result

        def mouseReleaseEvent(self, event):
            super().mouseReleaseEvent(event)
            self.owner.commit_layout()

        def mouseDoubleClickEvent(self, event):
            if self.spec.kind == 'group':
                self.owner.rename_group(self.spec.node_id)
            else:
                self.owner.emit_command('viewport.fit', {'scan_ids': list(self.spec.scan_ids)})
            event.accept()

    class CableItem(widgets.QGraphicsPathItem):

        def __init__(self, source, target):
            super().__init__()
            self.source, self.target = (source, target)
            self.setZValue(-1)
            self.setFlag(widgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            self.setToolTip('Tarama -> grup. Sec + Delete: yalniz baglantiyi ayir')
            self.refresh()

        def refresh(self):
            self.setPath(path_between(self.source.port.scenePos(), self.target.port.scenePos()))

        def shape(self):
            stroker = gui.QPainterPathStroker()
            stroker.setWidth(12)
            return stroker.createStroke(self.path())

        def paint(self, painter, option, widget=None):
            self.setPen(gui.QPen(gui.QColor(color['accent'] if self.isSelected() else '#8B8B7D'), 2.0 if self.isSelected() else 1.4))
            super().paint(painter, option, widget)

    class GraphView(widgets.QGraphicsView):

        def __init__(self, owner):
            super().__init__(owner.scene, owner)
            self.owner = owner
            self.setObjectName('ScanGraphView')
            self.setAccessibleName('Tarama ve grup dugumleri')
            self.setRenderHint(gui.QPainter.RenderHint.Antialiasing)
            self.setViewportUpdateMode(widgets.QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
            self.setDragMode(widgets.QGraphicsView.DragMode.RubberBandDrag)
            self.setTransformationAnchor(widgets.QGraphicsView.ViewportAnchor.NoAnchor)
            self.setResizeAnchor(widgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
            self.setFocusPolicy(core.Qt.FocusPolicy.StrongFocus)
            self.setFrameShape(widgets.QFrame.Shape.NoFrame)
            self.setBackgroundBrush(gui.QColor(color['background']))
            self._pan_last = None
            self._pan_button = None
            self._pan_moved = False
            self._suppress_context = False
            self._space = False
            self._source = None
            self._wire = None

        def drawBackground(self, painter, rect):
            painter.fillRect(rect, gui.QColor(color['background']))
            spacing = 24.0
            while spacing * self.transform().m11() < 18:
                spacing *= 2
            left = math.floor(rect.left() / spacing) * spacing
            top = math.floor(rect.top() / spacing) * spacing
            dots = [core.QPointF(x, y) for x in range(int(left), int(rect.right()) + 1, int(spacing)) for y in range(int(top), int(rect.bottom()) + 1, int(spacing))]
            pen = gui.QPen(gui.QColor('#48483F'))
            pen.setCosmetic(True)
            painter.setPen(pen)
            if dots:
                painter.drawPoints(gui.QPolygonF(dots))

        def wheelEvent(self, event):
            delta = event.angleDelta().y()
            if not delta:
                event.ignore()
                return
            before = self.mapToScene(event.position().toPoint())
            old = self.transform().m11()
            target = min(2.5, max(0.15, old * 1.15 ** (delta / 120)))
            self.scale(target / old, target / old)
            after = self.mapToScene(event.position().toPoint())
            self.translate(after.x() - before.x(), after.y() - before.y())
            event.accept()

        def mousePressEvent(self, event):
            self.setFocus()
            point = event.position().toPoint()
            if event.button() in (core.Qt.MouseButton.MiddleButton, core.Qt.MouseButton.RightButton) or (self._space and event.button() == core.Qt.MouseButton.LeftButton):
                self._pan_last, self._pan_moved = (point, False)
                self._pan_button = event.button()
                self.setCursor(core.Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            item = self.itemAt(point)
            if event.button() == core.Qt.MouseButton.LeftButton and isinstance(item, PortItem) and item.is_output:
                self._source = item.node
                self._wire = widgets.QGraphicsPathItem()
                self._wire.setPen(gui.QPen(gui.QColor(color['accent']), 2))
                self._wire.setZValue(3)
                self._wire.setAcceptedMouseButtons(core.Qt.MouseButton.NoButton)
                self.owner.scene.addItem(self._wire)
                self._wire.setPath(path_between(item.scenePos(), self.mapToScene(point)))
                self.owner.set_status('Grubun sol girisine birakin. Bosluga birakmak iptal eder.')
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            point = event.position().toPoint()
            if self._pan_last is not None:
                delta = point - self._pan_last
                self._pan_moved |= delta.manhattanLength() > 2
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
                self._pan_last = point
                event.accept()
                return
            if self._wire is not None:
                self._wire.setPath(path_between(self._source.port.scenePos(), self.mapToScene(point)))
                event.accept()
                return
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if self._pan_last is not None:
                self._suppress_context = self._pan_moved and self._pan_button == core.Qt.MouseButton.RightButton
                self._pan_last = None
                self._pan_button = None
                self.unsetCursor()
                event.accept()
                return
            if self._wire is not None:
                self.owner.scene.removeItem(self._wire)
                self._wire = None
                item = self.itemAt(event.position().toPoint())
                if isinstance(item, PortItem) and (not item.is_output):
                    self.owner.connect_nodes(self._source.spec.key, item.node.spec.key)
                self._source = None
                self.owner.set_status('Tarama cikisini grup girisine surukleyin.')
                event.accept()
                return
            super().mouseReleaseEvent(event)
            self.owner.commit_layout()

        def contextMenuEvent(self, event):
            if self._suppress_context:
                self._suppress_context = False
                event.accept()
                return
            item = self.itemAt(event.pos())
            if isinstance(item, PortItem):
                item = item.node
            if isinstance(item, (NodeItem, CableItem)) and (not item.isSelected()):
                self.owner.scene.clearSelection()
                item.setSelected(True)
            self.owner.show_context_menu(event.globalPos())
            event.accept()

        def keyPressEvent(self, event):
            key, modifiers = (event.key(), event.modifiers())
            if key == core.Qt.Key.Key_Space:
                self._space = True
                event.accept()
                return
            if modifiers & core.Qt.KeyboardModifier.ControlModifier:
                if key == core.Qt.Key.Key_G:
                    self.owner.create_group()
                    event.accept()
                    return
                if key == core.Qt.Key.Key_A:
                    self.owner._select_all()
                    event.accept()
                    return
            if key in (core.Qt.Key.Key_Delete, core.Qt.Key.Key_Backspace):
                self.owner.detach_selection()
            elif key == core.Qt.Key.Key_F:
                self.owner._selected_command('viewport.fit')
            elif key == core.Qt.Key.Key_H:
                self.owner.toggle_visibility()
            elif key == core.Qt.Key.Key_Home:
                self.owner.fit_all()
            elif key == core.Qt.Key.Key_Escape:
                if self._wire is not None:
                    self.owner.scene.removeItem(self._wire)
                    self._wire = self._source = None
                else:
                    self.owner.scene.clearSelection()
            else:
                super().keyPressEvent(event)
                return
            event.accept()

        def keyReleaseEvent(self, event):
            if event.key() == core.Qt.Key.Key_Space:
                self._space = False
            super().keyReleaseEvent(event)

        def focusOutEvent(self, event):
            self._space = False
            super().focusOutEvent(event)

    class ScanGraphWidget(widgets.QWidget):
        scan_selection_changed = core.Signal(object)
        command_requested = core.Signal(str, object)
        layout_changed = core.Signal(object)

        def __init__(self, parent):
            super().__init__(parent)
            self.setObjectName('ScanGraphWidget')
            self.setMinimumWidth(240)
            self._syncing = False
            self._project_id = None
            self._first_view = True
            self._last_layout = {}
            self._last_selection = ()
            self._visibility = {}
            self.nodes = {}
            self.edges = []
            self.scene = widgets.QGraphicsScene(self)
            self.view = GraphView(self)
            self.scene.selectionChanged.connect(self._selection_changed)
            layout = widgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            top = widgets.QWidget(self)
            tools_layout = widgets.QHBoxLayout(top)
            tools_layout.setContentsMargins(6, 6, 6, 6)
            tools_layout.setSpacing(4)
            self.fit_button = widgets.QPushButton('Sigdir', top)
            self.fit_button.setObjectName('ScanGraphFitButton')
            self.fit_button.setToolTip('Agi tuvale sigdir (Home)')
            self.group_button = widgets.QPushButton('Grupla', top)
            self.group_button.setObjectName('ScanGraphGroupButton')
            self.group_button.setToolTip('Secilen taramalari grupla (Ctrl+G)')
            self.ungroup_button = widgets.QPushButton('Ayir', top)
            self.ungroup_button.setObjectName('ScanGraphUngroupButton')
            self.ungroup_button.setToolTip('Grubu dagit veya baglantiyi ayir; kaynaklar korunur (Delete)')
            for button in (self.fit_button, self.group_button, self.ungroup_button):
                button.setMinimumHeight(26)
                tools_layout.addWidget(button)
            tools_layout.addStretch(1)
            self.count_label = widgets.QLabel('0 tarama', top)
            tools_layout.addWidget(self.count_label)
            self.fit_button.clicked.connect(self.fit_all)
            self.group_button.clicked.connect(self.create_group)
            self.ungroup_button.clicked.connect(self.detach_selection)
            layout.addWidget(top)
            layout.addWidget(self.view, 1)
            self.status_label = widgets.QLabel('Tarama cikisini grup girisine surukleyin.', self)
            self.status_label.setWordWrap(True)
            self.status_label.setContentsMargins(8, 5, 8, 3)
            layout.addWidget(self.status_label)
            hint = widgets.QLabel('Orta/sag tus: kaydir | Tekerlek: zoom\nCtrl+G: grup | F: 3B odak | H: gorunurluk', self)
            hint.setObjectName('ScanGraphHint')
            hint.setWordWrap(True)
            hint.setContentsMargins(8, 3, 8, 7)
            layout.addWidget(hint)
            self.setStyleSheet('QWidget#ScanGraphWidget { background:#1A1A1A; } QLabel#ScanGraphHint { color:#A8A8A3; font-size:10px; } QGraphicsView#ScanGraphView { border:0; }')
            self._update_controls()

        def set_status(self, message):
            self.status_label.setText(str(message))

        def set_project(self, project, selected_scan_ids=None):
            snapshot = graph_snapshot(project)
            project_id = getattr(project, 'project_id', id(project))
            changed_project = project_id != self._project_id
            selection = (() if changed_project else self.selected_scan_ids()) if selected_scan_ids is None else tuple(selected_scan_ids)
            selected_keys = {key for key, node in self.nodes.items() if node.isSelected()}
            keep_groups = selected_scan_ids is None and (not changed_project) and any((node.isSelected() and node.spec.kind == 'group' for node in self.nodes.values()))
            added_groups = []
            self._syncing = True
            try:
                for edge in self.edges:
                    self.scene.removeItem(edge)
                self.edges.clear()
                wanted = {spec.key for spec in snapshot.nodes}
                for key in list(self.nodes):
                    if key not in wanted or changed_project:
                        self.scene.removeItem(self.nodes.pop(key))
                for spec in snapshot.nodes:
                    if spec.key not in self.nodes:
                        node = NodeItem(self, spec)
                        self.nodes[spec.key] = node
                        self.scene.addItem(node)
                        node.setPos(*snapshot.positions[spec.key])
                        if spec.kind == 'group':
                            added_groups.append(node)
                    else:
                        node = self.nodes[spec.key]
                        node.spec = spec
                        node.setToolTip(f'{spec.label}\n{spec.node_id}\nCift tik: ' + ('grup adini degistir' if spec.kind == 'group' else "3B'de odaklan"))
                        node.update()
                        saved = project.metadata.get('scan_graph', {})
                        saved_positions = saved.get('positions', {}) if isinstance(saved, Mapping) else {}
                        if isinstance(saved_positions, Mapping) and spec.key in saved_positions:
                            node.setPos(*snapshot.positions[spec.key])
                for source, target in snapshot.edges:
                    edge = CableItem(self.nodes[source], self.nodes[target])
                    self.edges.append(edge)
                    self.scene.addItem(edge)
                self._project_id = project_id
                self._visibility = {spec.node_id: spec.visible for spec in snapshot.nodes if spec.kind == 'scan'}
                if changed_project:
                    self._first_view = True
                if selected_keys and (not changed_project):
                    for key, node in self.nodes.items():
                        node.setSelected(key in selected_keys)
                if not keep_groups and self.selected_scan_ids() != selection:
                    for node in self.nodes.values():
                        node.setSelected(node.spec.kind == 'scan' and node.spec.node_id in selection)
                self._last_layout = self.export_layout()
                self.count_label.setText(f"{sum((node.spec.kind == 'scan' for node in self.nodes.values()))} tarama")
                self._refresh_scene_rect()
            finally:
                self._syncing = False
            self._last_selection = self.selected_scan_ids()
            self._selection_changed(emit=False)
            if self.isVisible() and self._first_view:
                self._initial_view()
            elif self.isVisible() and added_groups and (not changed_project):
                self.view.ensureVisible(added_groups[-1], 8, 8)

        def set_scan_visibility(self, mapping):
            for scan_id, visible in mapping.items():
                if f'scan:{scan_id}' in self.nodes:
                    self._visibility[scan_id] = bool(visible)
                    self.nodes[f'scan:{scan_id}'].update()

        def selected_scan_ids(self):
            selected = {scan_id for node in self.nodes.values() if node.isSelected() for scan_id in node.spec.scan_ids}
            return tuple((node.spec.node_id for node in self.nodes.values() if node.spec.kind == 'scan' and node.spec.node_id in selected))

        def select_scans(self, scan_ids, emit=False):
            wanted = set(scan_ids)
            if set(self.selected_scan_ids()) == wanted:
                return
            self._syncing = True
            try:
                for edge in self.edges:
                    edge.setSelected(False)
                for node in self.nodes.values():
                    node.setSelected(node.spec.kind == 'scan' and node.spec.node_id in wanted)
            finally:
                self._syncing = False
            self._selection_changed(emit=emit)

        def _select_all(self):
            self.select_scans([node.spec.node_id for node in self.nodes.values() if node.spec.kind == 'scan'], emit=True)

        def _selection_changed(self, emit=True):
            if self._syncing:
                return
            selected = self.selected_scan_ids()
            group_members = {scan_id for node in self.nodes.values() if node.isSelected() and node.spec.kind == 'group' for scan_id in node.spec.scan_ids}
            for node in self.nodes.values():
                node.member_highlight = node.spec.kind == 'scan' and node.spec.node_id in group_members
                node.update()
            changed = selected != self._last_selection
            self._last_selection = selected
            self._update_controls()
            if emit and changed:
                self.scan_selection_changed.emit(selected)

        def _update_controls(self):
            selected = bool(self.selected_scan_ids())
            self.group_button.setEnabled(selected)
            self.ungroup_button.setEnabled(bool(self.scene.selectedItems()))
            self.fit_button.setEnabled(bool(self.nodes))

        def export_layout(self):
            return {key: [float(node.pos().x()), float(node.pos().y())] for key, node in sorted(self.nodes.items())}

        def commit_layout(self):
            if self._syncing:
                return
            positions = self.export_layout()
            if positions != self._last_layout:
                self._last_layout = positions
                self._refresh_scene_rect()
                self.layout_changed.emit(positions)

        def _refresh_edges(self):
            for edge in self.edges:
                edge.refresh()

        def _refresh_scene_rect(self):
            bounds = self.scene.itemsBoundingRect()
            self.scene.setSceneRect(bounds.adjusted(-1200, -1200, 1200, 1200))

        def fit_all(self):
            if not self.nodes:
                return
            self.view.fitInView(self.scene.itemsBoundingRect().adjusted(-24, -24, 24, 24), core.Qt.AspectRatioMode.KeepAspectRatio)
            scale = self.view.transform().m11()
            if scale > 1.15:
                self.view.scale(1.15 / scale, 1.15 / scale)
            self._first_view = False

        def _initial_view(self):
            if not self.nodes or self.view.viewport().width() < 50:
                return
            bounds = self.scene.itemsBoundingRect().adjusted(-16, -16, 16, 16)
            scale = min(1.0, max(0.55, self.view.viewport().width() / max(bounds.width(), 1)))
            self.view.resetTransform()
            self.view.scale(scale, scale)
            selected = next((node for node in self.nodes.values() if node.isSelected()), None)
            top_center = bounds.top() + self.view.viewport().height() / scale / 2
            y = max(selected.sceneBoundingRect().center().y(), top_center) if selected else top_center
            self.view.centerOn(bounds.center().x(), y)
            self._first_view = False

        def showEvent(self, event):
            super().showEvent(event)
            if self._first_view:
                core.QTimer.singleShot(0, self._initial_view)

        def can_connect(self, source_key, target_key):
            source, target = (self.nodes.get(source_key), self.nodes.get(target_key))
            return bool(source is not None and target is not None and (source.spec.kind == 'scan') and (target.spec.kind == 'group'))

        def connect_nodes(self, source_key, target_key):
            if not self.can_connect(source_key, target_key):
                return False
            source, target = (self.nodes[source_key], self.nodes[target_key])
            if source.spec.node_id in target.spec.scan_ids:
                return False
            self.emit_command('group.assign', {'scan_ids': [source.spec.node_id], 'group_id': target.spec.node_id})
            return True

        def emit_command(self, command, payload):
            self.command_requested.emit(str(command), dict(payload))

        def _selected_command(self, command):
            ids = list(self.selected_scan_ids())
            if ids:
                self.emit_command(command, {'scan_ids': ids})

        def create_group(self, allow_empty=False):
            ids = list(self.selected_scan_ids())
            if not ids and (not allow_empty):
                return
            count = sum((node.spec.kind == 'group' for node in self.nodes.values()))
            self.emit_command('group.create', {'scan_ids': ids, 'label': f'Grup {count + 1}'})

        def rename_group(self, group_id):
            node = self.nodes.get(f'group:{group_id}')
            if node is None:
                return
            label, accepted = widgets.QInputDialog.getText(self, 'Grup adi', 'Yeni ad:', text=node.spec.label)
            if accepted and label.strip() and (label.strip() != node.spec.label):
                self.emit_command('group.rename', {'group_id': group_id, 'label': label.strip()})

        def detach_selection(self):
            cables = [edge for edge in self.edges if edge.isSelected()]
            groups = [node.spec.node_id for node in self.nodes.values() if node.isSelected() and node.spec.kind == 'group']
            if cables:
                self.emit_command('scan.ungroup', {'scan_ids': sorted({edge.source.spec.node_id for edge in cables})})
            elif groups:
                self.emit_command('group.ungroup', {'group_ids': groups})
            else:
                self._selected_command('scan.ungroup')

        def toggle_visibility(self):
            ids = list(self.selected_scan_ids())
            if ids:
                visible = not all((self._visibility.get(scan_id, True) for scan_id in ids))
                self.emit_command('visibility.set', {'scan_ids': ids, 'visible': visible})

        def show_context_menu(self, position):
            menu = widgets.QMenu(self)
            ids = self.selected_scan_ids()
            if ids:
                menu.addAction('Ozellikler', lambda: self._selected_command('properties.show'))
                menu.addAction("3B'de odaklan (F)", lambda: self._selected_command('viewport.fit'))
                menu.addAction('Goster / gizle (H)', self.toggle_visibility)
                menu.addAction('Yalniz secilenleri goster', lambda: self._selected_command('scan.isolate'))
                menu.addSeparator()
                menu.addAction('Hesapla - asil noktalar', lambda: self._selected_command('scan.calculate'))
                menu.addAction('Temsile don', lambda: self._selected_command('scan.proxy'))
                menu.addSeparator()
                menu.addAction('Secilenleri grupla (Ctrl+G)', self.create_group)
            groups = [node for node in self.nodes.values() if node.isSelected() and node.spec.kind == 'group']
            if len(groups) == 1:
                menu.addAction('Grubu yeniden adlandir', lambda: self.rename_group(groups[0].spec.node_id))
            if self.scene.selectedItems():
                menu.addAction('Grubu / baglantiyi ayir (Delete)', self.detach_selection)
            menu.addSeparator()
            menu.addAction('Yeni bos grup', lambda: self.create_group(allow_empty=True))
            menu.addAction('Agi sigdir (Home)', self.fit_all)
            menu.exec(position)
    return ScanGraphWidget(parent)
