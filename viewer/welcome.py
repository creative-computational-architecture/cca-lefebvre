

def _art_canvas(qt, parent=None):

    class ArtCanvas(qt.QtWidgets.QWidget):

        def __init__(self, parent):
            super().__init__(parent)
            self.setToolTip('Open a project or import your own source files.')

        def paintEvent(self, event):
            painter = qt.QtGui.QPainter(self)
            painter.fillRect(self.rect(), qt.QtGui.QColor('#202124'))
    return ArtCanvas(parent)

def create_welcome_page(qt):
    widgets = qt.QtWidgets
    page = _art_canvas(qt)
    page.setObjectName('ProjectHomePage')
    root = widgets.QVBoxLayout(page)
    root.setContentsMargins(34, 28, 34, 26)
    root.setSpacing(18)
    heading = widgets.QLabel('CCA LEFEBVRE / PROJE ALANI')
    heading.setObjectName('HomeEyebrow')
    root.addWidget(heading)
    row = widgets.QHBoxLayout()
    panel = widgets.QFrame()
    panel.setObjectName('WelcomeActions')
    panel.setFixedWidth(360)
    panel.setStyleSheet('#WelcomeActions {background:#202124;}')
    actions = widgets.QVBoxLayout(panel)
    actions.setContentsMargins(0, 16, 18, 14)
    actions.setSpacing(10)
    title = widgets.QLabel('Taramadan\nplana.')
    title.setObjectName('HomeTitle')
    title.setStyleSheet('font-size:38px;font-weight:600;')
    actions.addWidget(title)
    hint = widgets.QLabel('Siralama, hizalama, analiz ve Mekanin Uretimi.\nHam kaynak korunur; her donusum izlenir.')
    hint.setObjectName('HomeSubtitle')
    hint.setWordWrap(True)
    actions.addWidget(hint)
    actions.addSpacing(12)
    for name, title, command in (('OpenProjectButton', 'Proje ac', 'PROJECT:OPEN'), ('NewProjectButton', 'Yeni proje', 'PROJECT:NEW'), ('HomeImportRawButton', 'Ham tarama ekle', 'IMPORT:RAW ALL DATA'), ('HomeImportPlyButton', 'PLY nokta bulutu ekle', 'IMPORT:PLY SCANS'), ('HomeImportDxfButton', 'DXF altlik ekle', 'IMPORT:DXF PLAN')):
        button = widgets.QPushButton(title)
        button.setObjectName(name)
        button.setProperty('ccaCommand', command)
        button.setMinimumHeight(32)
        if name == 'OpenProjectButton':
            button.setProperty('ccaRole', 'primary')
        actions.addWidget(button)
    row.addWidget(panel)
    row.addStretch(1)
    root.addLayout(row, 1)
    return page

def create_new_project_dialog(qt, parent):
    widgets = qt.QtWidgets
    dialog = widgets.QDialog(parent)
    dialog.setObjectName('LefebvreNewProjectDialog')
    dialog.setWindowTitle('CCA Lefebvre / Yeni proje')
    dialog.resize(800, 480)
    root = widgets.QHBoxLayout(dialog)
    form = widgets.QVBoxLayout()
    title = widgets.QLabel('Yeni proje')
    title.setStyleSheet('font-size:26px;font-weight:600;')
    form.addWidget(title)
    description = widgets.QLabel('Bir ad verin. Sonra ham tarama, PLY ve DXF altliginizi ekleyin.')
    description.setWordWrap(True)
    form.addWidget(description)
    name = widgets.QLineEdit()
    name.setObjectName('NewProjectName')
    name.setPlaceholderText('Proje adi')
    name.setMaxLength(160)
    name.setAccessibleName('Yeni proje adi')
    form.addWidget(name)
    invariant = widgets.QLabel('Metre / CAD Z-up\nOlcek 1.000\nHam dosyalar degismez')
    invariant.setProperty('ccaRole', 'muted')
    form.addWidget(invariant)
    form.addStretch(1)
    create = widgets.QPushButton('Projeyi olustur')
    create.setObjectName('CreateProjectConfirm')
    create.setProperty('ccaRole', 'primary')
    create.setEnabled(False)
    create.clicked.connect(dialog.accept)
    name.textChanged.connect(lambda value: create.setEnabled(bool(value.strip())))
    name.returnPressed.connect(lambda: dialog.accept() if name.text().strip() else None)
    cancel = widgets.QPushButton('Iptal')
    cancel.setObjectName('CreateProjectCancel')
    cancel.clicked.connect(dialog.reject)
    form.addWidget(create)
    form.addWidget(cancel)
    root.addLayout(form, 2)
    artwork = _art_canvas(qt, dialog)
    artwork.setMinimumWidth(340)
    root.addWidget(artwork, 3)
    return dialog

def request_new_project_name(qt, parent):
    dialog = create_new_project_dialog(qt, parent)
    accepted = dialog.exec() == qt.QtWidgets.QDialog.DialogCode.Accepted
    name = dialog.findChild(qt.QtWidgets.QLineEdit, 'NewProjectName').text().strip()
    dialog.deleteLater()
    return (name, accepted)
