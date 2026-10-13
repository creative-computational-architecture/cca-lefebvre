
from __future__ import annotations

class CadCrosshair:

    def __init__(self, vtk):
        self.points = vtk.vtkPoints()
        self.points.SetNumberOfPoints(16)
        lines = vtk.vtkCellArray()
        for index in range(0, 16, 2):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(index)
            lines.InsertCellPoint(index + 1)
        self.data = vtk.vtkPolyData()
        self.data.SetPoints(self.points)
        self.data.SetLines(lines)
        self.mapper = vtk.vtkPolyDataMapper2D()
        self.mapper.SetInputData(self.data)
        self.mapper.ScalarVisibilityOff()
        self.actor = vtk.vtkActor2D()
        self.actor.SetMapper(self.mapper)
        self.actor.PickableOff()
        self.actor.UseBoundsOff()
        self.actor.GetProperty().SetColor(200 / 255, 200 / 255, 200 / 255)
        self.actor.GetProperty().SetLineWidth(1)
        self.actor.GetProperty().SetDisplayLocationToForeground()
        self.actor.VisibilityOff()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetInteractive(False)
        self.renderer.AddViewProp(self.actor)
        self._window = None

    def attach(self, window):
        if self._window is window and window.HasRenderer(self.renderer):
            return
        self.detach()
        layer = window.GetNumberOfLayers()
        window.SetNumberOfLayers(layer + 1)
        self.renderer.SetLayer(layer)
        self.renderer.PreserveColorBufferOn()
        self.renderer.PreserveDepthBufferOff()
        window.AddRenderer(self.renderer)
        self._window = window

    def detach(self):
        if self._window is not None:
            self._window.RemoveRenderer(self.renderer)
            self._window = None

    def update(self, *, x, y, width, height, logical_width, logical_height, percent, pickbox_px):
        sx, sy = (width / max(1, logical_width), height / max(1, logical_height))
        cx = max(0, min(width - 1, round(x * sx)))
        cy = max(0, min(height - 1, height - 1 - round(y * sy)))
        hx, hy = (max(1, round(pickbox_px // 2 * sx)), max(1, round(pickbox_px // 2 * sy)))
        rx, ry = (round(width * percent / 100), round(height * percent / 100))
        segments = (((max(0, cx - rx), cy), (cx - hx, cy)), ((cx + hx, cy), (min(width - 1, cx + rx), cy)), ((cx, max(0, cy - ry)), (cx, cy - hy)), ((cx, cy + hy), (cx, min(height - 1, cy + ry))), ((cx - hx, cy - hy), (cx + hx, cy - hy)), ((cx + hx, cy - hy), (cx + hx, cy + hy)), ((cx + hx, cy + hy), (cx - hx, cy + hy)), ((cx - hx, cy + hy), (cx - hx, cy - hy)))
        for index, (start, end) in enumerate(segments):
            self.points.SetPoint(2 * index, start[0] + 0.5, start[1] + 0.5, 0)
            self.points.SetPoint(2 * index + 1, end[0] + 0.5, end[1] + 0.5, 0)
        self.points.Modified()
        self.actor.VisibilityOn()
