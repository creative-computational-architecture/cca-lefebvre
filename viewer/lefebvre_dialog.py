
import colorsys
import hashlib
import numpy as np
from geometry.patchworking import PatchworkingConfig, reconstruct_patch

def patch_color(frame, u, v):
    key = f'{frame}:{u}:{v}'.encode('ascii')
    hue = int.from_bytes(hashlib.sha256(key).digest()[:4], 'big') / 2 ** 32
    return np.asarray(colorsys.hsv_to_rgb(hue, 0.58, 0.93))

def patch_rgb(row, generated_uv):
    rgb = row.get('measured_patch_rgb')
    if rgb is None:
        return (np.full((25, 3), 0.5), np.full((len(generated_uv), 3), 0.5))
    rgb = np.asarray(rgb, float) / 255.0
    if rgb.shape != (25, 3) or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError('RGB must be 25 actual sensor pixel colours in 0..255')
    grid = rgb.reshape(5, 5, 3)
    colors = []
    for u, v in generated_uv:
        u0, v0 = (min(int(u), 3), min(int(v), 3))
        x, y = (u - u0, v - v0)
        colors.append((1 - x) * (1 - y) * grid[v0, u0] + x * (1 - y) * grid[v0, u0 + 1] + (1 - x) * y * grid[v0 + 1, u0] + x * y * grid[v0 + 1, u0 + 1])
    return (rgb, np.asarray(colors).reshape(-1, 3))

def make_lefebvre_dialog(parent, report):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PySide6 import QtWidgets

    class FullWidthCheckBox(QtWidgets.QCheckBox):

        def hitButton(self, position):
            return self.contentsRect().contains(position)
    entries = []
    missing = 0
    for row in report['candidates']:
        if not row.get('measured_patch'):
            missing += 1
            continue
        result = reconstruct_patch(row)
        rgb, new_rgb = patch_rgb(row, result['generated_uv'])
        entries.append((row, result, rgb, new_rgb))
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName('LefebvreDialog')
    dialog.setWindowTitle('CCA Lefebvre / Mekanin Uretimi / PATCHWORKING')
    dialog.resize(1220, 820)
    dialog.setStyleSheet('QDialog#LefebvreDialog {background:#171b20;color:#ededee;} #LefebvreDialog QLabel {color:#ededee;} #LefebvreDialog QComboBox, #LefebvreDialog QPushButton {background:#252a31;color:#ededee;border:1px solid #535a63;padding:7px;} #LefebvreDialog QCheckBox {color:#ededee;padding:6px;}')
    layout = QtWidgets.QVBoxLayout(dialog)
    heading = QtWidgets.QLabel('LEFEBVRE  /  MEKANIN URETIMI  /  PATCHWORKING')
    heading.setStyleSheet('font-size:20px;font-weight:600;padding:8px;')
    layout.addWidget(heading)
    note = QtWidgets.QLabel('Deneysel uretim, kalibre olcum degil. Ham tarama ve proje degismez. RGB: kaynak fotograf rengi. Patch Color: ayni parcaya ortak kimlik rengi.\nDaire: korunan/tasinan kaynak nokta. +: uretilmis yeni nokta. Yeni RGB degeri kaynak patch icinde ara degerdir; yeni fotograf gozlemi degildir.')
    note.setWordWrap(True)
    layout.addWidget(note)
    actions = QtWidgets.QHBoxLayout()
    pick = QtWidgets.QComboBox()
    pick.setObjectName('LefebvrePatchSelector')
    pick.addItem('Tum patchler', -1)
    for i, (row, *_rest) in enumerate(entries):
        pick.addItem(f"P{i + 1:03d} / kare {row['frame_index']} / {row['u']},{row['v']}", i)
    mode = QtWidgets.QComboBox()
    mode.setObjectName('LefebvreColorMode')
    mode.addItems(['RGB + Patch Color', 'RGB', 'Patch Color'])
    production = FullWidthCheckBox('Patchworking sonucunu goster')
    production.setObjectName('LefebvreProductionToggle')
    production.setChecked(True)
    reset = QtWidgets.QPushButton('Tumunu sigdir')
    reset.setObjectName('LefebvreFitButton')
    for widget in (pick, mode, production, reset):
        actions.addWidget(widget)
    layout.addLayout(actions)
    uncertain = FullWidthCheckBox('Belirsiz gorus hipotezlerini dene / olcum ve hizalama DISI')
    uncertain.setObjectName('LefebvreUncertainVisibilityToggle')
    uncertain.setChecked(False)
    uncertain.setEnabled(any((row.get('status') == 'VISIBILITY_UNVERIFIED' for row, *_ in entries)))
    layout.addWidget(uncertain)
    figure = Figure(facecolor='#171b20')
    canvas = FigureCanvasQTAgg(figure)
    canvas.setObjectName('LefebvreCanvas')
    layout.addWidget(canvas, 1)
    summary = QtWidgets.QLabel()
    summary.setObjectName('LefebvreSummary')
    summary.setWordWrap(True)
    layout.addWidget(summary)
    close = QtWidgets.QPushButton('Kapat / ham projeye don')
    close.setObjectName('LefebvreCloseButton')
    close.clicked.connect(dialog.accept)
    layout.addWidget(close)

    def draw():
        figure.clear()
        selected = entries if pick.currentData() == -1 else [entries[pick.currentData()]]
        kinds = ['RGB', 'Patch Color'] if mode.currentIndex() == 0 else [mode.currentText()]
        hypotheses = sum((r.get('visibility_unverified_hypothesis', False) for _, r, _, _ in selected))
        all_points = []
        for _row, result, _rgb, _new_rgb in selected:
            all_points.extend([result['raw'], result['placed'], result['generated']])
        points = np.concatenate(all_points) if all_points else np.zeros((1, 3))
        center = points.mean(axis=0)
        radius = max(0.15, float(np.max(np.ptp(points, axis=0))) * 0.6)
        for i, kind in enumerate(kinds):
            ax = figure.add_subplot(1, len(kinds), i + 1, projection='3d', facecolor='#171b20')
            state = ('BELIRSIZ HIPOTEZ' if hypotheses else 'URETIM') if production.isChecked() else 'HAM'
            ax.set_title(kind + ' / ' + state, color='#ededee')
            ax.tick_params(colors='#a5a9af', labelsize=8)
            for axis, label in zip((ax.xaxis, ax.yaxis, ax.zaxis), ('X m', 'Y m', 'Z m')):
                axis.set_label_text(label, color='#a5a9af')
                axis.set_pane_color((0.09, 0.11, 0.13, 1))
            for row, result, rgb, new_rgb in selected:
                color = patch_color(row['frame_index'], row['u'], row['v'])
                shown = result['placed'] if production.isChecked() else result['raw']
                colors = rgb if kind == 'RGB' else np.tile(color, (25, 1))
                ax.scatter(*shown.T, c=colors, s=14, depthshade=False)
                if production.isChecked() and len(result['generated']):
                    nc = new_rgb if kind == 'RGB' else np.tile(color, (len(result['generated']), 1))
                    ax.scatter(*result['generated'].T, c=nc, marker='+', s=18, depthshade=False)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_box_aspect((1, 1, 1))
        moved = sum((int(r['moved'].sum()) for _, r, _, _ in selected))
        added = sum((len(r['generated']) for _, r, _, _ in selected))
        colored = sum((row.get('measured_patch_rgb') is not None for row, *_ in selected))
        summary.setText(f"{report['run']} / {len(selected)} patch / {len(selected) * 25} kaynak ornek / {moved} tasima adayi / {added} yeni sentetik nokta\nRGB bulunan patch: {colored}/{len(selected)}; digerleri gri. Eski/eksik rapor parcasi: {missing}. Koordinatlar yerel ARKit Y-up / metre. Belirsiz gorus hipotezi: {hypotheses}; olcum/hizalama girdisi DEGIL. Bu sayilar tum taramanin benzersiz nokta sayisi degildir.")
        figure.tight_layout()
        figure.subplots_adjust(top=0.91)
        canvas.draw()

    def synchronize(event):
        if event.inaxes not in figure.axes or len(figure.axes) != 2:
            return
        source = event.inaxes
        for target in figure.axes:
            if target is not source:
                target.view_init(elev=source.elev, azim=source.azim, roll=source.roll)
                target.set_xlim(source.get_xlim())
                target.set_ylim(source.get_ylim())
                target.set_zlim(source.get_zlim())
        canvas.draw_idle()
    canvas.mpl_connect('button_release_event', synchronize)

    def recompute():
        for i, (row, *_rest) in enumerate(entries):
            result = reconstruct_patch(row, config=PatchworkingConfig(allow_uncertain_visibility=uncertain.isChecked()))
            rgb, new_rgb = patch_rgb(row, result['generated_uv'])
            entries[i] = (row, result, rgb, new_rgb)
        draw()
    uncertain.toggled.connect(recompute)
    pick.currentIndexChanged.connect(draw)
    mode.currentIndexChanged.connect(draw)
    production.toggled.connect(draw)
    reset.clicked.connect(draw)
    dialog._figure, dialog._entries = (figure, entries)
    draw()
    return dialog
