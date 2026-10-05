"""검사 화면에서 재사용하는 이미지·모달 구성요소."""

from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QToolTip,
    QVBoxLayout,
)


def show_pixel_tooltip(label, source_pixmap, position):
    """표시 이미지의 커서 위치를 원본 픽셀 좌표와 밝기로 변환한다."""
    displayed_pixmap = label.pixmap()
    if source_pixmap.isNull() or displayed_pixmap is None or displayed_pixmap.isNull():
        QToolTip.hideText()
        return

    contents = label.contentsRect()
    image_left = contents.left() + (contents.width() - displayed_pixmap.width()) // 2
    image_top = contents.top() + (contents.height() - displayed_pixmap.height()) // 2
    image_rect = displayed_pixmap.rect().translated(image_left, image_top)
    if not image_rect.contains(position):
        QToolTip.hideText()
        return

    source_x = min(
        source_pixmap.width() - 1,
        max(
            0,
            int(
                (position.x() - image_left)
                * source_pixmap.width()
                / displayed_pixmap.width()
            ),
        ),
    )
    source_y = min(
        source_pixmap.height() - 1,
        max(
            0,
            int(
                (position.y() - image_top)
                * source_pixmap.height()
                / displayed_pixmap.height()
            ),
        ),
    )
    color = source_pixmap.toImage().pixelColor(source_x, source_y)
    gray = round(
        0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
    )
    QToolTip.showText(
        label.mapToGlobal(position + QPoint(16, 18)),
        f"x: {source_x} · y: {source_y} · gray: {gray}",
        label,
    )


class ClickableImageLabel(QLabel):
    """클릭 이벤트를 전달하는 이미지 라벨."""

    clicked = pyqtSignal()
    hovered = pyqtSignal(QPoint)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.hovered.emit(event.pos())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)


class ImageModal(QDialog):
    """이미지를 중앙에서 크게 확인하고 다시 클릭해 닫는 모달."""

    def __init__(self, parent, enable_pixel_tooltip=True):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self.setObjectName("imageModal")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        card = QFrame()
        card.setObjectName("imageModalCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 18)
        card_layout.setSpacing(8)
        self.title_label = QLabel()
        self.title_label.setObjectName("imageModalTitle")
        self.image_scroll = QScrollArea()
        self.image_scroll.setObjectName("imageModalScroll")
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        self.image_label = ClickableImageLabel()
        self.image_label.setObjectName("imageModalPreview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setCursor(Qt.PointingHandCursor)
        self.image_scroll.setWidget(self.image_label)

        zoom_controls = QHBoxLayout()
        zoom_controls.setSpacing(6)
        zoom_controls.addStretch()
        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.setObjectName("zoomButton")
        self.zoom_label = QLabel("100%")
        self.zoom_label.setObjectName("zoomLabel")
        self.zoom_reset_button = QPushButton("맞춤")
        self.zoom_reset_button.setObjectName("zoomResetButton")
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setObjectName("zoomButton")
        zoom_controls.addWidget(self.zoom_out_button)
        zoom_controls.addWidget(self.zoom_label)
        zoom_controls.addWidget(self.zoom_in_button)
        zoom_controls.addWidget(self.zoom_reset_button)
        zoom_controls.addStretch()

        self.hint_label = QLabel("이미지를 다시 클릭하면 닫힙니다.")
        self.hint_label.setObjectName("imageModalHint")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.image_label.clicked.connect(self.accept)
        if enable_pixel_tooltip:
            self.image_label.hovered.connect(
                lambda position: show_pixel_tooltip(
                    self.image_label, self._pixmap, position
                )
            )
        self.zoom_out_button.clicked.connect(lambda: self._change_zoom(1 / 1.25))
        self.zoom_reset_button.clicked.connect(self._reset_zoom)
        self.zoom_in_button.clicked.connect(lambda: self._change_zoom(1.25))
        card_layout.addWidget(self.title_label)
        card_layout.addWidget(self.image_scroll, stretch=1)
        card_layout.addLayout(zoom_controls)
        card_layout.addWidget(self.hint_label)
        layout.addWidget(card, stretch=1)

    def show_pixmap(self, pixmap, title):
        if pixmap.isNull():
            return

        parent_size = self.parentWidget().size()
        self.resize(
            max(420, min(round(parent_size.width() * 0.82), 1500)),
            max(360, min(round(parent_size.height() * 0.82), 1000)),
        )
        parent_center = self.parentWidget().mapToGlobal(
            self.parentWidget().rect().center()
        )
        self.move(parent_center - self.rect().center())
        self._pixmap = pixmap
        self.zoom_factor = 1.0
        self.title_label.setText(title)
        self._refresh_pixmap()
        self.exec_()

    def _refresh_pixmap(self):
        if self._pixmap.isNull():
            return
        viewport_size = self.image_scroll.viewport().size()
        if viewport_size.width() <= 1 or viewport_size.height() <= 1:
            return
        fitted = self._pixmap.scaled(
            viewport_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        scaled = self._pixmap.scaled(
            max(1, round(fitted.width() * self.zoom_factor)),
            max(1, round(fitted.height() * self.zoom_factor)),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setFixedSize(scaled.size())
        self.zoom_label.setText(f"{round(self.zoom_factor * 100)}%")

    def _change_zoom(self, factor):
        self.zoom_factor = min(5.0, max(0.25, self.zoom_factor * factor))
        self._refresh_pixmap()

    def _reset_zoom(self):
        self.zoom_factor = 1.0
        self._refresh_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_pixmap()


class InspectionInfoModal(QDialog):
    """검사 로그를 필요할 때만 확인하는 정보 모달."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("inspectionInfoModal")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(18, 18, 18, 18)
        card = QFrame()
        card.setObjectName("inspectionInfoModalCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 20)
        card_layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("검사 상세 정보")
        title.setObjectName("imageModalTitle")
        close_button = QPushButton("×")
        close_button.setObjectName("modalCloseButton")
        close_button.setToolTip("닫기")
        close_button.clicked.connect(self.accept)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(close_button)

        self.info_text = QPlainTextEdit()
        self.info_text.setObjectName("inspectionInfoText")
        self.info_text.setReadOnly(True)
        self.info_text.setFocusPolicy(Qt.NoFocus)
        self.info_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        card_layout.addLayout(header)
        card_layout.addWidget(self.info_text, stretch=1)
        outer_layout.addWidget(card, stretch=1)

    def show_text(self, text):
        parent_size = self.parentWidget().size()
        self.resize(
            max(440, min(round(parent_size.width() * 0.42), 720)),
            max(360, min(round(parent_size.height() * 0.66), 760)),
        )
        parent_center = self.parentWidget().mapToGlobal(
            self.parentWidget().rect().center()
        )
        self.move(parent_center - self.rect().center())
        self.info_text.setPlainText(text)
        self.exec_()


class NoteModal(QDialog):
    """현재 검사 이미지에 대한 간단한 메모를 CSV로 남기는 모달."""

    def __init__(self, parent, source_path, notes_repository):
        super().__init__(parent)
        self.source_path = source_path
        self.notes_repository = notes_repository
        self.path_text = str(source_path.parent)
        self.filename = source_path.name
        self.existing_note = self.notes_repository.find(source_path)
        self.setObjectName("noteModal")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(18, 18, 18, 18)
        card = QFrame()
        card.setObjectName("noteModalCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 20)
        card_layout.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("메모 수정" if self.existing_note else "메모 추가")
        title.setObjectName("imageModalTitle")
        close_button = QPushButton("×")
        close_button.setObjectName("modalCloseButton")
        close_button.setToolTip("닫기")
        close_button.clicked.connect(self.reject)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(close_button)

        file_label = QLabel(f"{self.path_text}  /  {self.filename}")
        file_label.setObjectName("noteFileLabel")
        self.note_input = QPlainTextEdit()
        self.note_input.setObjectName("noteInput")
        self.note_input.setPlaceholderText("이 이미지에 대한 메모를 입력하세요.")
        self.note_input.setTabChangesFocus(True)
        if self.existing_note:
            self.note_input.setPlainText(self.existing_note["note"])
        save_button = QPushButton("메모 저장")
        save_button.setObjectName("noteSaveButton")
        save_button.clicked.connect(self._save_note)

        card_layout.addLayout(header)
        card_layout.addWidget(file_label)
        card_layout.addWidget(self.note_input, stretch=1)
        card_layout.addWidget(save_button)
        outer_layout.addWidget(card, stretch=1)

    def show_modal(self):
        parent_size = self.parentWidget().size()
        self.resize(
            max(420, min(round(parent_size.width() * 0.36), 620)),
            max(300, min(round(parent_size.height() * 0.46), 480)),
        )
        parent_center = self.parentWidget().mapToGlobal(
            self.parentWidget().rect().center()
        )
        self.move(parent_center - self.rect().center())
        self.note_input.setFocus()
        self.exec_()

    def _save_note(self):
        note = self.note_input.toPlainText().strip()
        if not note:
            QMessageBox.information(self, "메모 저장", "메모 내용을 입력하세요.")
            return

        self.notes_repository.upsert(self.source_path, note)
        self.accept()


