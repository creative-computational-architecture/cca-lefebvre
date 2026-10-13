
import unicodedata
ACTIONS = (('sequence', 'Zaman sirasi', 'Zaman uclarini ve tarama rollerini incele; konumlari zorla uclara tasima.'), ('overlap', 'Ortak alan ara', 'RGB uc / ortak alan adaylarini bul; renk skoru tek basina hizalama degil.'), ('align', 'Hizalama adayi', 'Secili iki taramaya mevcut geometrik hizalama motorunu calistir.'), ('plan_place', 'Altlikta yerlestir', 'DXF uzerinde kronolojik manuel yerlesim; XY tasi, Z dondur, onayla ve ilerle.'), ('plan_fit', 'Plana uyum adayi', 'Secili tarama icin mevcut DXF plan uyum motoru; kabul ve uygulama ayri.'), ('patchworking', 'Mekanin Uretimi', 'Patchworking / RGB + Patch Color; ham veriden ayri deneysel onizleme.'), ('point_audit', 'Nokta hesabi', 'Tum PLY noktalarinin destek / dolgu / cozulmemis durumunu raporla.'))

def action_for_text(text):
    text = unicodedata.normalize('NFKD', text.casefold().replace('i', 'i'))
    words = set(''.join((c for c in text if not unicodedata.combining(c))).replace('-', ' ').split())
    matches = []
    for action, keywords in (('sequence', {'sirala', 'siralama', 'zaman', 'sequence'}), ('overlap', {'ortak', 'overlap', 'rgb'}), ('align', {'hizala', 'hizalama', 'align'}), ('patchworking', {'patch', 'patchworking', 'patchwork', 'uretim', 'uretimi', 'mekan', 'yuzey'}), ('point_audit', {'hesap', 'hesabi', 'audit'})):
        if words & keywords:
            matches.append(action)
    return matches[0] if len(matches) == 1 else None

def make_lefebvre_agent_dialog(parent, *, dispatch, context):
    from PySide6 import QtCore, QtWidgets
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName('LefebvreAgentDialog')
    dialog.setWindowTitle('CCA Lefebvre / Asistan')
    dialog.resize(700, 570)
    dialog.setStyleSheet('QDialog#LefebvreAgentDialog {background:#171b20;} #LefebvreAgentDialog QLabel {color:#ededee;} #LefebvreAgentDialog QPushButton, #LefebvreAgentDialog QLineEdit {color:#ededee;background:#252a31;border:1px solid #535a63;padding:9px;}')
    layout = QtWidgets.QVBoxLayout(dialog)
    title = QtWidgets.QLabel('CCA LEFEBVRE / ASISTAN')
    title.setStyleSheet('font-size:23px;font-weight:600;')
    layout.addWidget(title)
    capability = QtWidgets.QLabel('Yerel AI / Ollama. Kurulu kucuk model kullanilir; model indirilmez. Yalniz komut metni yerel modele gider; ham veri gonderilmez.\nSiralama, ortak alan, hizalama ve mekan uretimi ayni panelden. Hizalama sonucunu projeye uygulama ayri onay adimidir.')
    capability.setObjectName('LefebvreConnectionStatus')
    capability.setWordWrap(True)
    layout.addWidget(capability)
    status = QtWidgets.QLabel(context())
    status.setObjectName('LefebvreAgentStatus')
    status.setWordWrap(True)
    layout.addWidget(status)

    def invoke(action):
        try:
            message = dispatch(action)
            status.setText(context() + '\n' + message)
        except (ValueError, RuntimeError, KeyError) as error:
            status.setText(f'Islem baslatilamadi: {error}')
    action_grid = QtWidgets.QGridLayout()
    for index, (action, label, description) in enumerate(ACTIONS):
        button = QtWidgets.QPushButton(label)
        button.setObjectName('LefebvreTask_' + action)
        button.setToolTip(description)
        button.clicked.connect(lambda checked=False, key=action: invoke(key))
        action_grid.addWidget(button, index // 2, index % 2)
    layout.addLayout(action_grid)
    edit = QtWidgets.QLineEdit()
    edit.setObjectName('LefebvreCommandInput')
    edit.setPlaceholderText('Lefebvre: taramalari zaman akisina gore incelemek istiyorum...')
    layout.addWidget(edit)
    send = QtWidgets.QPushButton('Yerel AI ile yorumla')
    send.setObjectName('LefebvreRunCommand')
    proposal = {'action': None}
    confirm = QtWidgets.QPushButton('Onerilen araci baslat')
    confirm.setObjectName('LefebvreConfirmProposal')
    confirm.setEnabled(False)
    confirm.clicked.connect(lambda: invoke(proposal['action']) if proposal['action'] else None)
    timer = QtCore.QTimer(dialog)
    timer.setInterval(100)
    pending = {}

    def submit():
        import queue
        import threading
        from viewer.lefebvre_local_model import propose_action
        if not send.isEnabled():
            return
        if not edit.text().strip():
            status.setText('Yorumlanacak komutu yazin.')
            return
        send.setEnabled(False)
        confirm.setEnabled(False)
        proposal['action'] = None
        text = edit.text()
        replies = queue.Queue()
        pending['queue'] = replies
        status.setText('Lefebvre yerel modeli komutu yorumluyor; henuz geometri islemi yok.')

        def work():
            try:
                replies.put((propose_action(text), None))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                replies.put((None, str(error)))
        threading.Thread(target=work, daemon=True, name='LefebvreLocalPlanner').start()
        timer.start()

    def poll():
        import queue
        try:
            reply, error = pending['queue'].get_nowait()
        except queue.Empty:
            return
        timer.stop()
        send.setEnabled(True)
        if error:
            status.setText(f'Yerel AI kullanilamadi: {error}\nArac dugmelerini dogrudan kullanabilirsiniz.')
        elif reply['action'] == 'unsupported':
            status.setText('Bu istek tek bir desteklenen Lefebvre aracina karsilik gelmiyor; islem baslatilmadi.')
        else:
            proposal['action'] = reply['action']
            label = next((label for key, label, _ in ACTIONS if key == reply['action']))
            capability.setText(f"Lefebvre / {reply['model']} / YEREL AI / ham veri gonderilmedi")
            status.setText(f'Oneri: {label}. Henuz uygulanmadi; baslatmak icin asagidaki dugmeyi kullanin.')
            confirm.setEnabled(True)
    timer.timeout.connect(poll)
    edit.returnPressed.connect(submit)
    send.clicked.connect(submit)
    layout.addWidget(send)
    layout.addWidget(confirm)
    close = QtWidgets.QPushButton('Kapat')
    close.setObjectName('LefebvreAgentClose')
    close.clicked.connect(dialog.accept)
    layout.addWidget(close)
    return dialog
