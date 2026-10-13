
from __future__ import annotations
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING
from viewer.brand import FONT_MONO_CANDIDATES, FONT_SANS_CANDIDATES, PRODUCT_NAME, SKIN_C_DARK
from viewer.help_content import HELP_TOPICS, TOPICS_BY_ID, HelpTopic, search_topics
if TYPE_CHECKING:
    from PySide6.QtWidgets import QDialog, QWidget
HELP_IMAGE_PATH = Path(__file__).resolve().parent / 'assets' / 'uzam' / 'help-rays-v1.png'

def _topic_html(topic: HelpTopic) -> str:
    paragraphs = ''.join((f'<p>{escape(text)}</p>' for text in topic.paragraphs))
    references = ''.join((f'<li><a href="help:{topic.topic_id}">{escape(ref)}</a></li>' for ref in (*topic.commands, *topic.widgets)))
    related = ''.join((f'<li><a href="help:{key}">{escape(TOPICS_BY_ID[key].title)}</a></li>' for key in topic.related))
    return f'<h1>{escape(topic.title)}</h1>{paragraphs}<h2>Ilgili konular</h2><ul>{related}</ul><h2>Komut / kontrol referanslari</h2><p>Belgeleme baglantilari; islem calistirmaz. Ribbon butonlari RibbonCommand nesnesinde ccaCommand kimligiyle ayrilir.</p><ul>{references}</ul>'

def make_help_dialog(parent: QWidget | None, *, initial_query: str='') -> QDialog:
    from PySide6 import QtCore, QtGui, QtWidgets

    class OfflineBrowser(QtWidgets.QTextBrowser):

        def loadResource(self, resource_type, url):
            return None

    class ConceptBanner(QtWidgets.QLabel):

        def __init__(self, pixmap, owner):
            super().__init__(owner)
            self.original = pixmap
            self.setFixedHeight(150)
            self.setMinimumWidth(0)
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
            self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        def resizeEvent(self, event):
            self.setPixmap(self.original.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
            super().resizeEvent(event)
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName('UzamHelpDialog')
    dialog.setWindowTitle(f'{PRODUCT_NAME} / Yardim')
    dialog.setAccessibleName(f'{PRODUCT_NAME} yardim')
    dialog.setAccessibleDescription('Cevrimdisi, aranabilir kullanim kilavuzu. Proje verisini degistirmez.')
    dialog.resize(1040, 760)
    dialog.setMinimumSize(680, 480)
    palette = SKIN_C_DARK
    sans = ', '.join((f"'{font}'" for font in FONT_SANS_CANDIDATES))
    mono = ', '.join((f"'{font}'" for font in FONT_MONO_CANDIDATES))
    dialog.setStyleSheet(f"\n        QDialog#UzamHelpDialog {{ background: {palette['background']}; }}\n        #UzamHelpDialog QWidget {{ color: {palette['ink']}; font-family: {sans}; font-size: 15px; }}\n        #UzamHelpDialog QLabel {{ background: transparent; }}\n        #UzamHelpDialog #UzamHelpTitle {{ font-size: 27px; font-weight: 600; }}\n        #UzamHelpDialog #UzamHelpCaption, #UzamHelpDialog #UzamHelpResults {{ color: {palette['muted']}; }}\n        #UzamHelpSearch, #UzamHelpTopics, #UzamHelpBody {{\n            background: {palette['surface']}; border: 1px solid {palette['muted']};\n            border-radius: 0; selection-background-color: {palette['accent']};\n            selection-color: {palette['ink']};\n        }}\n        #UzamHelpSearch {{ padding: 9px; }}\n        #UzamHelpSearch:focus, #UzamHelpTopics:focus, #UzamHelpBody:focus {{ border: 2px solid {palette['ink']}; }}\n        #UzamHelpTopics {{ background: {palette['sidebar']}; padding: 5px; }}\n        #UzamHelpTopics::item {{ padding: 10px 8px; }}\n        #UzamHelpTopics::item:selected {{ background: {palette['accent']}; color: {palette['ink']}; }}\n        #UzamHelpBody {{ padding: 14px; }}\n        #UzamHelpDialog QPushButton {{ background: {palette['surface']}; border: 1px solid {palette['muted']}; padding: 8px 18px; }}\n        #UzamHelpDialog QPushButton:hover, #UzamHelpDialog QPushButton:focus {{ border: 2px solid {palette['ink']}; }}\n        #UzamHelpDialog QSplitter::handle {{ background: {palette['background']}; }}\n        #UzamHelpDialog QScrollBar:vertical {{ background: {palette['sidebar']}; width: 10px; margin: 0; }}\n        #UzamHelpDialog QScrollBar::handle:vertical {{ background: {palette['muted']}; min-height: 30px; }}\n        #UzamHelpDialog QScrollBar::add-line:vertical, #UzamHelpDialog QScrollBar::sub-line:vertical {{ height: 0; }}\n        #UzamHelpDialog QScrollBar::add-page:vertical, #UzamHelpDialog QScrollBar::sub-page:vertical {{ background: none; }}\n    ")
    root = QtWidgets.QVBoxLayout(dialog)
    root.setContentsMargins(24, 20, 24, 16)
    root.setSpacing(12)
    title = QtWidgets.QLabel(f'{PRODUCT_NAME} / Yardim')
    title.setObjectName('UzamHelpTitle')
    root.addWidget(title)
    caption = QtWidgets.QLabel('Ham veri korunur. Gorunum, hipotez ve olcum kaniti farkli seylerdir.')
    caption.setObjectName('UzamHelpCaption')
    caption.setWordWrap(True)
    root.addWidget(caption)
    search_row = QtWidgets.QHBoxLayout()
    search_label = QtWidgets.QLabel('&Ara')
    search = QtWidgets.QLineEdit()
    search.setObjectName('UzamHelpSearch')
    search.setAccessibleName('Yardimda ara')
    search.setAccessibleDescription('Konu, komut kimligi veya buton nesne adi; Ctrl+F ile odaklan.')
    search.setPlaceholderText('Konu, komut veya kisayol ara')
    search_label.setBuddy(search)
    clear = QtWidgets.QPushButton('Temizle')
    clear.setObjectName('UzamHelpClear')
    clear.setAccessibleName('Yardim aramasini temizle')
    clear.setAutoDefault(False)
    clear.clicked.connect(lambda: (search.clear(), search.setFocus()))
    search_row.addWidget(search_label)
    search_row.addWidget(search, 1)
    search_row.addWidget(clear)
    root.addLayout(search_row)
    results = QtWidgets.QLabel()
    results.setObjectName('UzamHelpResults')
    root.addWidget(results)
    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    splitter.setChildrenCollapsible(False)
    topics = QtWidgets.QListWidget()
    topics.setObjectName('UzamHelpTopics')
    topics.setAccessibleName('Yardim konulari')
    topics.setAccessibleDescription('Ok tuslari ile secin. Secilen konu sagda acilir.')
    topics.setWordWrap(True)
    topics.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    topics.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
    topics.setMinimumWidth(200)
    contents = QtWidgets.QWidget()
    contents_layout = QtWidgets.QVBoxLayout(contents)
    contents_layout.setContentsMargins(0, 0, 0, 0)
    pixmap = QtGui.QPixmap(str(HELP_IMAGE_PATH)) if HELP_IMAGE_PATH.is_file() else QtGui.QPixmap()
    if not pixmap.isNull():
        banner = ConceptBanner(pixmap, dialog)
        banner.setObjectName('UzamHelpBanner')
        banner.setAccessibleName('Kavramsal isin ve yuzey illustasyonu; olcum kaniti degildir.')
        banner.setToolTip(banner.accessibleName())
        contents_layout.addWidget(banner)
        note = QtWidgets.QLabel('Kavramsal illustasyon / olcum kaniti degildir.')
        note.setObjectName('UzamHelpCaption')
        note.setWordWrap(True)
        contents_layout.addWidget(note)
    contents_layout.addWidget(topics, 1)
    body = OfflineBrowser()
    body.setObjectName('UzamHelpBody')
    body.setAccessibleName('Secili konunun yardim metni')
    body.setOpenLinks(False)
    body.setOpenExternalLinks(False)
    body.setTabChangesFocus(True)
    body.document().setDefaultStyleSheet(f"body {{font-family: {sans}; font-size: 15px; color: {palette['ink']};}}h1 {{font-size: 24px; font-weight: 600; margin-bottom: 16px;}}h2 {{font-size: 17px; font-weight: 600; margin-top: 24px;}}p {{line-height: 145%; margin-bottom: 14px;}} li {{margin-bottom: 8px;}}a {{color: {palette['ink']}; text-decoration: underline;}}code {{font-family: {mono};}}")
    splitter.addWidget(contents)
    splitter.addWidget(body)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([270, 680])
    root.addWidget(splitter, 1)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
    close = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Close)
    close.setObjectName('UzamHelpClose')
    close.setText('Kapat')
    close.setAccessibleName('Yardimi kapat')
    close.setAutoDefault(False)
    buttons.rejected.connect(dialog.reject)
    root.addWidget(buttons)
    role = QtCore.Qt.ItemDataRole.UserRole

    def show_topic(current, previous):
        if current is not None:
            topic = TOPICS_BY_ID[current.data(role)]
            body.setHtml(_topic_html(topic))
            body.setAccessibleName(f'Yardim: {topic.title}')
            body.verticalScrollBar().setValue(0)

    def filter_topics(query):
        previous = topics.currentItem()
        selected_id = previous.data(role) if previous is not None else None
        matches = search_topics(query)
        topics.blockSignals(True)
        topics.clear()
        selected_row = 0
        for index, topic in enumerate(matches):
            item = QtWidgets.QListWidgetItem(topic.title, topics)
            item.setData(role, topic.topic_id)
            if topic.topic_id == selected_id:
                selected_row = index
        topics.blockSignals(False)
        results.setText(f'{len(matches)} / {len(HELP_TOPICS)} konu')
        clear.setEnabled(bool(query))
        if matches:
            topics.setCurrentRow(selected_row)
        else:
            body.setHtml('<h1>Sonuc bulunamadi</h1><p>Daha kisa bir terim deneyin veya Temizle ile tum konulara donun.</p>')
            body.setAccessibleName('Yardim: sonuc bulunamadi')

    def follow_reference(url):
        if url.scheme() != 'help':
            return
        topic_id = url.path()
        if topic_id not in TOPICS_BY_ID:
            return
        search.clear()
        for index in range(topics.count()):
            if topics.item(index).data(role) == topic_id:
                topics.setCurrentRow(index)
                body.setFocus()
                break
    topics.currentItemChanged.connect(show_topic)
    search.textChanged.connect(filter_topics)
    body.anchorClicked.connect(follow_reference)
    focus_search = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+F'), dialog)
    focus_search.activated.connect(lambda: (search.setFocus(), search.selectAll()))
    QtWidgets.QWidget.setTabOrder(search, clear)
    QtWidgets.QWidget.setTabOrder(clear, topics)
    QtWidgets.QWidget.setTabOrder(topics, body)
    QtWidgets.QWidget.setTabOrder(body, close)
    search.setText(initial_query)
    if not initial_query:
        filter_topics('')
    search.setFocus()
    return dialog
