
def make_structure_dialog(parent, report):
    from PySide6 import QtWidgets
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName('StructureSearchDialog')
    dialog.setWindowTitle('CCA Lefebvre / Yapi + isin yer adaylari')
    dialog.resize(960, 600)
    layout = QtWidgets.QVBoxLayout(dialog)
    counts = report['counts']
    label = QtWidgets.QLabel(f"{report['run']} / {counts['queries']} yerel yapi / {counts.get('reviewable', 0)} inceleme adayi / {report.get('anchor_summary', {}).get('eligible', 0)} LiDAR dayanak adayi\nKonum adayi, dogrulanmis olcum degildir. Ham noktalara dokunulmadi.\nYerel duzen + sonlu yuzey parcasi + kamera isini + bagimsiz gorus kontrolu.")
    label.setWordWrap(True)
    layout.addWidget(label)
    kind = QtWidgets.QComboBox()
    kind.setObjectName('StructureEvidenceKind')
    kind.addItem('Yer / restorasyon adaylari', 'candidates')
    kind.addItem('LiDAR dayanaklari / model eslesmesi bekliyor', 'alignment_anchors')
    layout.addWidget(kind)
    table = QtWidgets.QTableWidget(0, 7)
    table.setObjectName('StructureCandidatesTable')
    table.setHorizontalHeaderLabels(['Kare/piksel', 'Ham m', 'Aday m', 'Fark m', 'Gorus', 'Kontrol m', 'Durum'])
    statuses = {'REVIEWABLE_STRUCTURE_CANDIDATE': 'YER ADAYI / UYGULANMADI', 'FOREGROUND_UNRESOLVED': 'ONDE GOZLENMIS YUZEY VAR', 'FREE_SPACE_CONFLICT': 'BOSLUK GOZLEMI ILE CELISKI', 'VISIBILITY_UNVERIFIED': 'ZAYIF GORUS CELISKISI / BELIRSIZ HIPOTEZ', 'ALIGNMENT_ANCHOR_CANDIDATE': 'DAYANAK ADAYI / HIZALAMA YAPILMADI', 'ANCHOR_VISIBILITY_UNVERIFIED': 'DAYANAK OLAMAZ / GORUS BELIRSIZ', 'ANCHOR_FREE_SPACE_CONFLICT': 'DAYANAK OLAMAZ / BOSLUK CELISKISI'}
    rows = []
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.horizontalHeader().setStretchLastSection(True)
    table.resizeColumnsToContents()
    layout.addWidget(table)
    inspect = QtWidgets.QPushButton('Secilen noktayi ve kamera isinini 3D incele')
    inspect.setObjectName('StructureInspectButton')
    inspect.setEnabled(False)

    def selection_changed():
        inspect.setEnabled(0 <= table.currentRow() < len(rows))

    def preview():
        index = table.currentRow()
        if not 0 <= index < len(rows):
            return
        from viewer.structure_preview import make_structure_preview
        try:
            child = make_structure_preview(dialog, rows[index], report)
        except (ValueError, ImportError) as error:
            label.setText(f'Onizleme acilamadi: {error}')
            return
        dialog._structure_preview = child
        child.show()
    table.itemSelectionChanged.connect(selection_changed)
    inspect.clicked.connect(preview)
    layout.addWidget(inspect)
    lefebvre = QtWidgets.QPushButton('Lefebvre / Patchworking / RGB + Patch Color')
    lefebvre.setObjectName('StructureLefebvreButton')

    def produce():
        from viewer.lefebvre_dialog import make_lefebvre_dialog
        try:
            child = make_lefebvre_dialog(dialog, report)
        except (ValueError, ImportError) as error:
            label.setText(f'Lefebvre acilamadi: {error}')
            return
        dialog._lefebvre_dialog = child
        child.show()
    lefebvre.clicked.connect(produce)
    layout.addWidget(lefebvre)

    def populate():
        nonlocal rows
        anchor_mode = kind.currentData() == 'alignment_anchors'
        rows = sorted(report.get(kind.currentData(), []), key=lambda x: (x['status'] not in {'REVIEWABLE_STRUCTURE_CANDIDATE', 'ALIGNMENT_ANCHOR_CANDIDATE'}, x.get('anchor_max_residual_m', x['held_out_error_m']), x['frame_index']))
        table.setRowCount(len(rows))
        table.clearSelection()
        inspect.setEnabled(False)
        lefebvre.setEnabled(not anchor_mode)
        for i, row in enumerate(rows):
            values = [f"{row['frame_index']} / {row['u']},{row['v']}", f"{row['raw_depth_m']:.3f}", f"{row['candidate_depth_m']:.3f}", f"{row['displacement_m']:+.3f}", str(len(row['support_frames'])), f"{max(row.get('anchor_max_residual_m', 0), row['held_out_error_m']):.3f}", statuses.get(row['status'], row['status'])]
            for j, value in enumerate(values):
                table.setItem(i, j, QtWidgets.QTableWidgetItem(value))
        table.resizeColumnsToContents()
    kind.currentIndexChanged.connect(populate)
    populate()
    close = QtWidgets.QPushButton('Kapat')
    close.setObjectName('StructureCloseButton')
    close.clicked.connect(dialog.accept)
    layout.addWidget(close)
    return dialog
