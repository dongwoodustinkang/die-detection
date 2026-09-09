"""검사 화면에서 재사용하는 이미지·차트·모달 구성요소."""

import numpy as np
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from contour import MAX_COUNT_RATIO


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


class TopContourHistogram(QWidget):
    """상면 컨투어에 처음 닿는 y 좌표의 분포를 표시한다."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.coordinates = []
        self.coordinate_axis = "y"
        self.coordinate_range = None
        self.histogram_side = None
        self.log_lines = []
        self.setObjectName("topContourHistogram")
        self.setMinimumHeight(148)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_coordinates(
        self, coordinates, coordinate_axis="y", coordinate_range=None,
        histogram_side=None,
    ):
        self.coordinates = [int(coordinate) for coordinate in coordinates]
        self.coordinate_axis = coordinate_axis
        self.coordinate_range = coordinate_range
        self.histogram_side = histogram_side
        self._print_peak_coordinate_counts()
        self.update()

    def _print_peak_coordinate_counts(self):
        """최빈 접점 좌표와 인접 좌표의 접점 수를 터미널에 출력한다."""
        self.log_lines = []
        if not self.coordinates:
            return

        coordinates, counts = np.unique(self.coordinates, return_counts=True)
        max_count = int(max(counts))
        peak_coordinates = coordinates[counts == max_count]
        count_by_coordinate = dict(zip(coordinates, counts))

        ratio_count = max_count * MAX_COUNT_RATIO
        ratio_log_line = (
            f"[히스토그램] max_count_ratio={MAX_COUNT_RATIO}: "
            f"{ratio_count}개"
        )
        self.log_lines.append(ratio_log_line)
        print(ratio_log_line)
        for peak_coordinate in peak_coordinates:
            peak_coordinate = int(peak_coordinate)
            previous_count = int(count_by_coordinate.get(peak_coordinate - 1, 0))
            next_count = int(count_by_coordinate.get(peak_coordinate + 1, 0))
            coordinate_log_line = (
                f"[히스토그램] {self.coordinate_axis}={peak_coordinate}: "
                f"{max_count}개, "
                f"{self.coordinate_axis}={peak_coordinate - 1}: "
                f"{previous_count}개, "
                f"{self.coordinate_axis}={peak_coordinate + 1}: "
                f"{next_count}개"
            )
            self.log_lines.append(coordinate_log_line)
            print(coordinate_log_line)
            for coordinate, count in (
                (peak_coordinate - 1, previous_count),
                (peak_coordinate + 1, next_count),
            ):
                if count <= ratio_count:
                    continue

                print(
                    f"[히스토그램] {self.coordinate_axis}={coordinate} 접점 수({count}개)가 "
                    f"max_count_ratio 값({ratio_count}개)보다 큽니다: Merge 가능"
                )
                next_coordinate = coordinate + (1 if coordinate > peak_coordinate else -1)
                next_count = int(count_by_coordinate.get(next_coordinate, 0))
                comparison = (
                    "큽니다" if next_count > ratio_count
                    else "작습니다" if next_count < ratio_count
                    else "같습니다"
                )
                print(
                    f"[히스토그램] {self.coordinate_axis}={next_coordinate} 접점 수({next_count}개)는 "
                    f"max_count_ratio 값({ratio_count}개)보다 {comparison}."
                )

        if self.histogram_side not in {"top", "bottom"}:
            return

        merge_coordinates = set()
        for peak_coordinate in peak_coordinates:
            peak_coordinate = int(peak_coordinate)
            for direction in (-1, 1):
                adjacent_coordinate = peak_coordinate + direction
                adjacent_count = int(count_by_coordinate.get(adjacent_coordinate, 0))
                if adjacent_count <= ratio_count:
                    continue

                merge_coordinates.add(adjacent_coordinate)
                next_coordinate = adjacent_coordinate + direction
                if int(count_by_coordinate.get(next_coordinate, 0)) >= ratio_count:
                    merge_coordinates.add(next_coordinate)

        highlighted_coordinates = {
            *(int(coordinate) for coordinate in peak_coordinates),
            *merge_coordinates,
        }
        center_coordinate = (
            max(highlighted_coordinates)
            if self.histogram_side == "top"
            else min(highlighted_coordinates)
        )
        side_name = "상판" if self.histogram_side == "top" else "하판"
        print(
            f"[히스토그램] {side_name} 중심 컷 좌표 : "
            f"{self.coordinate_axis}={center_coordinate}"
        )

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(12, 10, -12, -10)

        if not self.coordinates:
            painter.setPen(QColor("#8E8E93"))
            painter.drawText(rect, Qt.AlignCenter, "상면 컨투어 접점 좌표가 없습니다.")
            return

        if self.coordinate_range is None:
            minimum, maximum = min(self.coordinates), max(self.coordinates)
        else:
            minimum, maximum = self.coordinate_range

        if minimum >= maximum:
            counts = [len(self.coordinates)]
        else:
            counts, _ = np.histogram(
                self.coordinates,
                # 기준선 내 각 정수 좌표를 하나의 독립적인 막대로 표시한다.
                # 넓은 기준선 범위를 임의 구간으로 합치면 서로 다른 첫 접점이
                # 하나의 막대에 합산되어 분포가 왜곡될 수 있다.
                bins=maximum - minimum + 1,
                range=(minimum, maximum + 1),
            )

        left = rect.left() + 30
        top = rect.top() + 8
        right = rect.right() - 4
        bottom = rect.bottom() - 24
        chart_width, chart_height = right - left, bottom - top
        if chart_width <= 0 or chart_height <= 0:
            return

        total_coordinate_count = len(self.coordinates)
        painter.setPen(QPen(QColor("#D1D1D6"), 1))
        painter.drawLine(left, bottom, right, bottom)
        painter.drawLine(left, top, left, bottom)

        bin_width = chart_width / len(counts)
        # 가장 많은 첫 접점이 모인 좌표(동률 포함)만 강조한다.
        maximum_count = int(max(counts))
        ratio_count = maximum_count * MAX_COUNT_RATIO
        merge_bar_indexes = set()
        peak_indexes = np.flatnonzero(counts == maximum_count)
        for peak_index in peak_indexes:
            for direction in (-1, 1):
                adjacent_index = peak_index + direction
                if not 0 <= adjacent_index < len(counts):
                    continue
                if counts[adjacent_index] <= ratio_count:
                    continue

                # 다음 좌표도 같은 기준을 만족할 때만 병합 색상으로 표시한다.
                merge_bar_indexes.add(adjacent_index)
                next_index = adjacent_index + direction
                if (
                    0 <= next_index < len(counts)
                    and counts[next_index] >= ratio_count
                ):
                    merge_bar_indexes.add(next_index)

        for index, count in enumerate(counts):
            bar_height = chart_height * int(count) / total_coordinate_count
            bar_left = left + index * bin_width + 1
            bar_width = max(1, bin_width - 2)
            bar_color = (
                QColor("#FF453A") if count == maximum_count
                else QColor("#FF9F0A") if index in merge_bar_indexes
                else QColor("#0A84FF")
            )
            painter.fillRect(
                int(round(bar_left)),
                int(round(bottom - bar_height)),
                int(round(bar_width)),
                int(round(bar_height)),
                bar_color,
            )

        painter.setPen(QColor("#6E6E73"))
        painter.drawText(
            0, top - 1, left - 5, 16,
            Qt.AlignRight | Qt.AlignVCenter, str(total_coordinate_count),
        )
        painter.drawText(
            0, bottom - 8, left - 5, 16, Qt.AlignRight | Qt.AlignVCenter, "0"
        )
        painter.drawText(
            left, bottom + 7, 72, 16,
            Qt.AlignLeft | Qt.AlignVCenter, f"{self.coordinate_axis}={minimum}"
        )
        painter.drawText(
            right - 72, bottom + 7, 72, 16,
            Qt.AlignRight | Qt.AlignVCenter, f"{self.coordinate_axis}={maximum}",
        )
        painter.end()


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


