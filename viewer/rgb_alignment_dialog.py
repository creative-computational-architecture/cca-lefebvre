
class RGBAlignmentDialog:

    def __init__(self, qt, rows, *, parent=None, on_validate=None):
        w = qt.QtWidgets
        self.dialog = w.QDialog(parent)
        self.dialog.setWindowTitle('CCA Lefebvre / RGB ortak alan adaylari')
        self.dialog.resize(1060, 630)
        self.rows = rows
        layout = w.QVBoxLayout(self.dialog)
        label = w.QLabel('Renk ve ayrinti benzerligi aday uretir. Ortak alan modu zaman komsulugu aramaz; geometri dogrulaninca bindirme onizlenir.')
        label.setWordWrap(True)
        layout.addWidget(label)
        self.table = w.QTableWidget(len(rows), 7)
        self.table.setObjectName('RGBCandidateTable')
        self.table.setHorizontalHeaderLabels(['Kaynak', 'Hedef', 'Baglanti', 'RGB benzerligi', 'Ortak ayrinti', 'Kanit', 'Skor'])
        self.table.setSelectionBehavior(w.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(w.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(w.QAbstractItemView.NoEditTriggers)
        for i, row in enumerate(rows):
            values = [row['source_scan_id'], row['target_scan_id'], 'Uc uca' if row['kind'] == 'ENDPOINT' else 'Ortak alan', f"%{100 * row['rgb_similarity']:.1f}", str(row['matching_details']), 'Renk + ayrinti' if row.get('evidence_kind') == 'RGB_DETAILS' else 'Yalniz renk', f"{row['score']:.2f}"]
            for j, value in enumerate(values):
                self.table.setItem(i, j, w.QTableWidgetItem(value))
        self.table.horizontalHeader().setSectionResizeMode(w.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.validate_button = w.QPushButton('Secili ciftin geometrisini dogrula')
        self.validate_button.setEnabled(bool(rows))
        layout.addWidget(self.validate_button)

        def validate():
            index = self.table.currentRow()
            if index >= 0 and on_validate:
                self.dialog.accept()
                on_validate(rows[index])
        self.validate_button.clicked.connect(validate)
        if rows:
            self.table.selectRow(0)

    def show(self):
        self.dialog.show()
        self.dialog.raise_()
