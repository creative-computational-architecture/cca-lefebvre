
from __future__ import annotations
from collections.abc import Callable
from typing import Any
import numpy as np

def create_view_cube(parent: Any, qt: Any, on_view: Callable[[str], None], on_fit: Callable[[], None], on_selection: Callable[[], None]) -> Any:
    QtCore, QtGui, QtWidgets = (qt.QtCore, qt.QtGui, qt.QtWidgets)

    def choose_view(name: str) -> None:
        on_view(name)
        parent.setFocus()

    def fit_drawing(callback: Callable[[], None]) -> None:
        callback()
        parent.setFocus()

    class ViewCube(QtWidgets.QFrame):

        def __init__(self) -> None:
            super().__init__(parent)
            self.setObjectName('ccaViewCube')
            self.setAccessibleName('Gorunum kupu')
            self.setFixedSize(150, 202)
            self.setStyleSheet('#ccaViewCube { background: #202020; border: 1px solid #626266; }#ccaViewCube QToolButton { background: #2A2A2D; color: #F5F5F5;border: 1px solid #626266; padding: 2px; font-size: 11px; }#ccaViewCube QToolButton:hover { border: 1px solid #C8102E; }#ccaViewCube QToolButton:focus { border: 2px solid #C8102E; }')
            self._pose = None
            self._faces: list[tuple[Any, str]] = []
            self.buttons: dict[str, Any] = {}
            specs = [('top', 'UST', 'Ust gorunum (7)', 5, 108, 44, 25), ('front', 'ON', 'On gorunum (1)', 53, 108, 44, 25), ('right', 'SAG', 'Sag gorunum (3)', 101, 108, 44, 25), ('perspective', 'PERSPEKTIF', 'Perspektif gorunum (0)', 5, 138, 140, 25), ('fit_scene', 'TUM', 'Tumunu sigdir (Home)', 5, 169, 67, 27), ('fit_selection', 'SECILI', 'Secileni sigdir (F)', 78, 169, 67, 27)]
            for name, label, tip, x, y, width, height in specs:
                button = QtWidgets.QToolButton(self)
                button.setObjectName(f'viewCube_{name}')
                button.setText(label)
                button.setToolTip(tip)
                button.setAccessibleName(tip)
                button.setGeometry(x, y, width, height)
                if name == 'fit_scene':
                    button.clicked.connect(lambda _checked=False: fit_drawing(on_fit))
                elif name == 'fit_selection':
                    button.clicked.connect(lambda _checked=False: fit_drawing(on_selection))
                else:
                    button.clicked.connect(lambda _checked=False, value=name: choose_view(value))
                self.buttons[name] = button

        def set_camera_pose(self, pose: Any) -> None:
            self._pose = pose
            self.update()

        def paintEvent(self, event: Any) -> None:
            super().paintEvent(event)
            if self._pose is None:
                return
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            eye = np.asarray(self._pose.eye) - self._pose.target
            eye /= max(float(np.linalg.norm(eye)), 1e-12)
            right, up = (np.asarray(self._pose.right), np.asarray(self._pose.screen_up))
            faces = [('top', 'UST', (0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]), ('bottom', 'ALT', (0, 0, -1), [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)]), ('front', 'ON', (0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]), ('back', 'ARKA', (0, 1, 0), [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)]), ('right', 'SAG', (1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]), ('left', 'SOL', (-1, 0, 0), [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)])]
            self._faces.clear()
            for name, label, normal, corners in faces:
                facing = float(np.dot(normal, eye))
                if facing <= 1e-07:
                    continue
                values = np.asarray(corners)
                projected = np.column_stack((75 + values @ right * 24, 45 - values @ up * 24))
                polygon = QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in projected])
                shade = int(45 + 24 * facing)
                painter.setBrush(QtGui.QColor(shade, shade, shade))
                painter.setPen(QtGui.QPen(QtGui.QColor('#A8A8A3'), 1.2))
                painter.drawPolygon(polygon)
                painter.setPen(QtGui.QColor('#F5F5F5'))
                painter.drawText(polygon.boundingRect(), QtCore.Qt.AlignmentFlag.AlignCenter, label)
                self._faces.append((polygon, name))
            painter.setPen(QtGui.QColor('#B8B8BA'))
            painter.drawText(QtCore.QRectF(5, 86, 140, 17), QtCore.Qt.AlignmentFlag.AlignCenter, 'Z YUKARI / CAD')
            painter.end()

        def mousePressEvent(self, event: Any) -> None:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                point = event.position()
                for polygon, name in reversed(self._faces):
                    if polygon.containsPoint(point, QtCore.Qt.FillRule.OddEvenFill):
                        choose_view(name)
                        event.accept()
                        return
            event.accept()
    return ViewCube()
