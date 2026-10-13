
from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any
import math
from viewer.alignment_scope import alignment_exclusions, set_alignment_exclusions

class TemporalChainDialog:
    OPTIONS = (('Dahil', None), ('Bodrum', 'BASEMENT'), ('Idari', 'ADMIN'), ('Diger', 'OTHER'))

    def __init__(self, qt: Any, project: Any, *, parent: Any=None, on_saved: Callable[[], None] | None=None, on_rgb_search: Callable[[], None] | None=None) -> None:
        self.qt, self.project, self.on_saved = (qt, project, on_saved)
        self.on_rgb_search = on_rgb_search
        widgets = qt.QtWidgets
        self.dialog = widgets.QDialog(parent)
        self.dialog.setObjectName('TemporalChainDialog')
        self.dialog.setWindowTitle('Zaman zinciri')
        self.dialog.setModal(True)
        self.dialog.resize(1180, 610)
        self.combos: dict[str, Any] = {}
        self._build()

    def _sequence_rows(self) -> list[tuple[str, Mapping[str, Any], Any]]:
        metadata = getattr(self.project, 'metadata', {})
        sequence = metadata.get('temporal_sequence', {})
        ordered = sequence.get('ordered_scan_ids', ()) if isinstance(sequence, Mapping) else ()
        scan_ids = {scan.scan_id for scan in getattr(self.project, 'scan_rail', ())}
        ordered_ids = list(dict.fromkeys((scan_id for scan_id in ordered if scan_id in scan_ids)))
        ordered_ids.extend((scan.scan_id for scan in getattr(self.project, 'scan_rail', ()) if scan.scan_id not in ordered_ids))
        runs = {item.get('scan_id'): item for item in sequence.get('runs', ()) if isinstance(item, Mapping)} if isinstance(sequence, Mapping) else {}
        gaps = {item.get('target_scan_id'): item.get('gap_s') for item in sequence.get('adjacent_pairs', ()) if isinstance(item, Mapping)} if isinstance(sequence, Mapping) else {}
        return [(scan_id, runs.get(scan_id, {}), gaps.get(scan_id)) for scan_id in ordered_ids]

    @staticmethod
    def _number(value: Any) -> str:
        try:
            return f'{float(value):.2f}s' if not isinstance(value, bool) and math.isfinite(float(value)) else '--'
        except (TypeError, ValueError):
            return '--'

    def _build(self) -> None:
        widgets = self.qt.QtWidgets
        root = widgets.QVBoxLayout(self.dialog)
        root.addWidget(widgets.QLabel('ZAMAN ZINCIRI / BITIS -> BASLANGIC'))
        note = widgets.QLabel('Zaman sadece aday cifti belirler; ortak geometri kaniti zorunludur. Son 30 sn -> ilk 30 sn aranir.\nBodrum/idari taramayi kimliginden secin: dislanan ara tarama zinciri koparir, ustunden atlanmaz. Ham veri ve kayitli konumlar degismez.')
        note.setWordWrap(True)
        root.addWidget(note)
        table = widgets.QTableWidget(self.dialog)
        table.setObjectName('TemporalChainTable')
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(('Sira', 'Tarama', 'Baslangic', 'Bitis', 'Sure / onceki ara', 'Kapsam'))
        rows = self._sequence_rows()
        table.setRowCount(len(rows))
        current = alignment_exclusions(self.project)
        for row, (scan_id, timing, gap) in enumerate(rows):
            table.setItem(row, 0, widgets.QTableWidgetItem(str(row + 1)))
            table.setItem(row, 1, widgets.QTableWidgetItem(scan_id))
            duration = timing.get('duration_s') if isinstance(timing, Mapping) else None
            table.setItem(row, 2, widgets.QTableWidgetItem(str(timing.get('capture_start', '--'))))
            table.setItem(row, 3, widgets.QTableWidgetItem(str(timing.get('capture_end', '--'))))
            table.setItem(row, 4, widgets.QTableWidgetItem(f'{self._number(duration)} / {self._number(gap)}'))
            combo = widgets.QComboBox(table)
            for label, value in self.OPTIONS:
                combo.addItem(label, value)
            value = current.get(scan_id)
            combo.setCurrentIndex(next((i for i, (_, item) in enumerate(self.OPTIONS) if item == value), 0))
            table.setCellWidget(row, 5, combo)
            self.combos[scan_id] = combo
        table.setEditTriggers(widgets.QAbstractItemView.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(widgets.QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        root.addWidget(table)
        self.table = table
        buttons = widgets.QDialogButtonBox(widgets.QDialogButtonBox.Save | widgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.cancel)
        root.addWidget(buttons)
        self.save_button = buttons.button(widgets.QDialogButtonBox.Save)
        self.cancel_button = buttons.button(widgets.QDialogButtonBox.Cancel)
        self.save_button.setText('Kapsami uygula')
        self.cancel_button.setText('Vazgec')
        self.rgb_button = widgets.QPushButton('RGB ile uc ve ortak alan adaylarini ara')
        self.rgb_button.setObjectName('RGBSearchButton')
        self.rgb_button.setEnabled(self.on_rgb_search is not None)

        def search():
            self.save()
            self.on_rgb_search()
        self.rgb_button.clicked.connect(search)
        root.addWidget(self.rgb_button)
        self.dialog.setStyleSheet('QDialog{background:#1A1A1A;color:#F5F5F5;} QTableWidget,QHeaderView::section{background:#202020;color:#F5F5F5;} QComboBox{background:#252525;color:#F5F5F5;}')

    def save(self) -> None:
        set_alignment_exclusions(self.project, {scan_id: combo.currentData() for scan_id, combo in self.combos.items() if combo.currentData() is not None})
        if self.on_saved is not None:
            self.on_saved()
        self.dialog.accept()

    def cancel(self) -> None:
        self.dialog.reject()

    def show(self) -> None:
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
