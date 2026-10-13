
import numpy as np

def preview_points(row):
    points = np.asarray([row[k] for k in ('origin', 'measured_point', 'candidate_point')], float)
    if points.shape != (3, 3) or not np.isfinite(points).all():
        raise ValueError('raporda sonlu kamera/ham/aday koordinati yok')
    return points

def make_structure_preview(parent, row, report):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PySide6 import QtWidgets
    p = preview_points(row)
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName('StructurePreviewDialog')
    is_anchor = 'eligible_for_correspondence_search' in row
    target_label = 'LiDAR dayanak adayi' if is_anchor else 'Yer adayi'
    dialog.setWindowTitle(f'CCA Lefebvre / {target_label} / UYGULANMADI')
    dialog.resize(900, 700)
    layout = QtWidgets.QVBoxLayout(dialog)
    note = QtWidgets.QLabel(f"Kare {row['frame_index']} / piksel {row['u']},{row['v']} / {row['raw_depth_m']:.3f} -> {row['candidate_depth_m']:.3f} m\nTuruncu: ham olcum. Beyaz: yer adayi. Camgobegi: bagimsiz kontrol.\nSurukle: 3D dondur. Koordinatlar taramanin yerel ARKit Y-up sisteminde, metre.\nBuyuk fark ve kucuk kontrol artigi, dogru duvari tek basina kanitlamaz.")
    note.setWordWrap(True)
    if is_anchor:
        note.setText(note.text() + '\nDAYANAK: ham nokta yerinde. Fotograf-model karsiligi ve hizalama henuz dogrulanmadi.')
    layout.addWidget(note)
    figure = Figure(facecolor='#171b20')
    canvas = FigureCanvasQTAgg(figure)
    canvas.setObjectName('StructurePreviewCanvas')
    ax = figure.add_subplot(111, projection='3d', facecolor='#171b20')
    ax.tick_params(colors='#a5a9af')
    for axis, text in zip((ax.xaxis, ax.yaxis, ax.zaxis), ('X m', 'Y m', 'Z m')):
        axis.label.set_color('#ededee')
        axis.set_pane_color((0.09, 0.11, 0.13, 1))
        axis.set_label_text(text)
    ax.plot(*p[[0, 2]].T, color='#c8102e', lw=1, label='Kamera isini')
    ax.scatter(*p[0], color='#c8102e', marker='^', s=55, label='Kamera')
    ax.scatter(*p[1], color='#f28c28', marker='x', s=80, label='Ham merkez')
    ax.scatter(*p[2], color='#eeeeee', marker='+', s=100, label=target_label)
    ax.plot(*p[[1, 2]].T, color='#f28c28', lw=2)
    if row.get('measured_patch'):
        patch = np.asarray(row['measured_patch'])
        ax.scatter(*patch.T, color='#f28c28', s=9, alpha=0.6)
    for patch in row.get('support_patches', []):
        xyz = np.asarray(patch['points'])
        ax.scatter(*xyz.T, s=11, color='#66bfc0' if patch['frame'] == row['held_out_frame'] else '#a5a9af')
    if not row.get('support_patches'):
        note.setText(note.text() + '\nEski rapor: oy veren 5x5 parcalar kayitli degil; bu onizlemede yalniz isin/merkezler var.')
    legend = ax.legend(facecolor='#171b20', edgecolor='#535a63', fontsize=9)
    for text in legend.get_texts():
        text.set_color('#ededee')

    def frame(local):
        target = p[1:] if local else p
        center = target.mean(axis=0)
        radius = max(0.2, float(np.max(np.ptp(target, axis=0))) * 0.65)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        canvas.draw()
    layout.addWidget(canvas, 1)
    actions = QtWidgets.QHBoxLayout()
    for name, title, callback in (('StructurePreviewLocal', 'Nokta cevresi', lambda: frame(True)), ('StructurePreviewCamera', 'Kamera geneli', lambda: frame(False)), ('StructurePreviewClose', 'Kapat', dialog.accept)):
        button = QtWidgets.QPushButton(title)
        button.setObjectName(name)
        button.clicked.connect(callback)
        actions.addWidget(button)
    layout.addLayout(actions)
    frame(True)
    dialog._figure = figure
    return dialog
