"""Defect Detector의 macOS 스타일 화면과 사용자 상호작용을 정의한다."""
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from contour import (
    MAX_COUNT_RATIO,
    create_detection_visualization,
    get_primary_contact_reference_point,
)
from styles import APP_STYLESHEET


IMAGE_EXTS = {".tif", ".tiff"}
DEFAULT_DIR = "dataset/"
PREVIEW_SCALE = 0.8
# DEV_IMAGE_DIR = Path("/Users/dongwookang/diehand_cv/dataset/side/total")
DEV_IMAGE_DIR = Path("/Users/dongwookang/diehand")
CAPTURE_ROOT = Path(__file__).resolve().parent / "captures"

class ClickableImageLabel(QLabel):
    """클릭 이벤트를 전달하는 이미지 라벨."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


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
            painter.setPen(QColor("#8B95A1"))
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
        painter.setPen(QPen(QColor("#DDE3EA"), 1))
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
                QColor("#F04452") if count == maximum_count
                else QColor("#FF9200") if index in merge_bar_indexes
                else QColor("#3182F6")
            )
            painter.fillRect(
                int(round(bar_left)),
                int(round(bottom - bar_height)),
                int(round(bar_width)),
                int(round(bar_height)),
                bar_color,
            )

        painter.setPen(QColor("#6B7684"))
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

    def __init__(self, parent):
        super().__init__(parent)
        self._pixmap = QPixmap()
        self.setObjectName("imageModal")
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
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


class MainWindow(QMainWindow):
    """A/B TIFF의 B 페이지 컨투어를 확인하는 메인 창."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("불량 검출기")
        self.resize(2120, 880)
        self.setMinimumSize(1200, 720)

        self.image_paths = []
        self.current_index = -1
        self.original_pixmap = QPixmap() # 원본 이미지
        self.result_pixmap = QPixmap() # B 페이지 이미지
        self.analysis_preview_pixmap = QPixmap()
        self.source_preview_pixmap = QPixmap()
        self.ball_crop_preview_pixmap = QPixmap()
        self.current_histogram_side = "top"
        self.current_histogram_region = "all"
        self.histogram_image_width = 0
        self.histogram_center_split_x = 0
        self.histogram_measurements = []
        self.program_log_lines = []
        self.side_cutting_enabled = False
        self.capture_session_dir = None
        self.capture_btn = QPushButton()
        self.capture_btn.setObjectName("captureIconButton")
        self.capture_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.capture_btn.setToolTip("현재 화면 캡처 저장")
        self.capture_btn.clicked.connect(self.on_capture)
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(APP_STYLESHEET)

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        root.addLayout(self._create_header())
        root.addLayout(self._create_workspace(), stretch=1)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

    def _create_header(self):
        header = QHBoxLayout()
        header.setSpacing(12)

        title_group = QVBoxLayout()
        title_group.setSpacing(2)
        title = QLabel("이상탐지 프로그램")
        title.setObjectName("windowTitle")
        self.header_metadata_label = QLabel(
            "파일을 불러오면 처리 정보가 표시됩니다."
        )
        self.header_metadata_label.setObjectName("headerMetadata")
        title_group.addWidget(title)
        title_group.addWidget(self.header_metadata_label)
        header.addLayout(title_group)
        header.addStretch()

        self.prev_btn = QPushButton("‹")
        self.prev_btn.setObjectName("navigationButton")
        self.prev_btn.setToolTip("이전 이미지 (←)")
        self.next_btn = QPushButton("›")
        self.next_btn.setObjectName("navigationButton")
        self.next_btn.setToolTip("다음 이미지 (→)")
        self.index_label = QLabel("0 / 0")
        self.index_label.setObjectName("indexLabel")
        self.prev_btn.clicked.connect(self.show_prev)
        self.next_btn.clicked.connect(self.show_next)
        header.addWidget(self.prev_btn)
        header.addWidget(self.index_label)
        header.addWidget(self.next_btn)
        header.addWidget(self.capture_btn)

        return header

    def _create_workspace(self):
        """원본·분석·설정을 역할에 맞는 세 영역으로 배치한다."""
        workspace = QHBoxLayout()
        workspace.setSpacing(16)

        source_column = QVBoxLayout()
        source_column.setSpacing(16)
        source_column.addWidget(self._create_source_card(), stretch=4)
        source_column.addWidget(self._create_log_card(), stretch=1)

        workspace.addLayout(source_column, stretch=5)
        workspace.addWidget(self._create_analysis_card(), stretch=3)
        workspace.addWidget(self._create_control_sidebar(), stretch=2)
        return workspace

    def _create_source_card(self):
        card = QFrame()
        card.setObjectName("imageCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        title = QLabel("원본 이미지")
        title.setObjectName("cardTitle")
        layout.addWidget(title)
        layout.addWidget(self._create_divider())

        image_pair = QHBoxLayout()
        image_pair.setSpacing(14)

        source_panel, self.image_label = self._create_image_panel(
            "A 페이지", "A/B TIFF 이미지를 불러오세요."
        )
        result_panel, self.result_label = self._create_image_panel(
            "B 페이지 결과", "B 페이지 컨투어 분석 결과가 표시됩니다."
        )
        self.image_label.clicked.connect(
            lambda: self._show_image_modal(self.original_pixmap, "A 페이지")
        )
        self.result_label.clicked.connect(
            lambda: self._show_image_modal(self.result_pixmap, "B 페이지 결과")
        )
        image_pair.addWidget(source_panel, stretch=1)
        image_pair.addWidget(result_panel, stretch=1)
        layout.addLayout(image_pair, stretch=1)
        return card

    def _create_image_panel(self, title_text, empty_text):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel(title_text)
        title.setObjectName("imagePaneTitle")
        layout.addWidget(title)

        label = self._create_image_label(empty_text)
        layout.addWidget(label, stretch=1)
        return panel, label

    def _create_log_card(self):
        info_card = QFrame()
        info_card.setObjectName("logCard")
        info_card.setMinimumHeight(126)
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(16, 13, 16, 13)
        info_layout.setSpacing(6)
        info_title = QLabel("프로그램 로그")
        info_title.setObjectName("cardTitle")
        self.info_label = QPlainTextEdit("이미지를 불러오면 상세 정보가 표시됩니다.")
        self.info_label.setObjectName("infoLabel")
        self.info_label.setReadOnly(True)
        # 로그는 마우스로만 스크롤한다. 화살표 키는 메인 창의 이미지 이동에 쓴다.
        self.info_label.setFocusPolicy(Qt.NoFocus)
        self.info_label.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.info_label.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.info_label.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.info_label.setFixedHeight(self.info_label.fontMetrics().lineSpacing() * 4 + 8)
        info_layout.addWidget(info_title)
        info_layout.addWidget(self.info_label)
        return info_card

    def _create_analysis_card(self):
        card = QFrame()
        card.setObjectName("analysisCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        analysis_scroll = QScrollArea()
        analysis_scroll.setObjectName("analysisScroll")
        analysis_scroll.setWidgetResizable(True)
        # 컨투어 영역은 마우스로만 스크롤한다. 화살표 키는 이미지 이동에 사용한다.
        analysis_scroll.setFocusPolicy(Qt.NoFocus)
        analysis_scroll.viewport().setFocusPolicy(Qt.NoFocus)
        analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        analysis_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        title = QLabel("컨투어 이미지")
        title.setObjectName("cardTitle")
        layout.addWidget(title)
        layout.addWidget(self._create_divider())

        self.analysis_preview_label = self._create_preview_section(
            layout,
            "기둥 기준(Blue) A/B 크롭 비교",
            "기둥 기준선을 같은 좌표로 적용한 A/B 페이지 크롭 결과를 비교합니다.",
        )
        self.source_preview_label = self._create_preview_section(
            layout,
            "최상단/빈도 기준(Gray) 크롭",
            "같은 회색 기준선을 적용한 A/B 페이지 크롭 결과를 비교합니다.",
        )
        self.ball_crop_preview_label = self._create_preview_section(
            layout,
            "탐지 볼 정사각형 내부 크롭",
            "조건을 만족하는 볼이 탐지되면 정사각형 내부가 표시됩니다.",
        )
        histogram_header = QHBoxLayout()
        histogram_header.setContentsMargins(0, 0, 0, 0)
        self.histogram_title = QLabel("상면 외곽 컨투어 첫 접점 y 좌표 분포")
        self.histogram_title.setObjectName("previewTitle")
        self.top_contour_count_label = QLabel("전체 0개")
        self.top_contour_count_label.setObjectName("histogramCount")
        histogram_header.addWidget(self.histogram_title)
        histogram_header.addStretch()
        self.histogram_side_group = QButtonGroup(self)
        self.histogram_side_group.setExclusive(True)
        self.histogram_side_buttons = {}
        for side_key, label_text in (
            ("top", "상"),
            ("bottom", "하"),
            ("left", "좌"),
            ("right", "우"),
        ):
            button = QPushButton(label_text)
            button.setObjectName("histogramSideButton")
            button.setCheckable(True)
            self.histogram_side_group.addButton(button)
            self.histogram_side_buttons[side_key] = button
            histogram_header.addWidget(button)
        self.histogram_side_buttons[self.current_histogram_side].setChecked(True)
        self.histogram_side_group.buttonClicked.connect(
            self._on_histogram_side_changed
        )

        region_selector = QFrame()
        region_selector.setObjectName("histogramRegionSelector")
        region_layout = QVBoxLayout(region_selector)
        region_layout.setContentsMargins(4, 3, 4, 3)
        region_layout.setSpacing(2)
        region_title = QLabel("영역")
        region_title.setObjectName("histogramRegionTitle")
        region_title.setAlignment(Qt.AlignCenter)
        region_layout.addWidget(region_title)
        self.histogram_region_group = QButtonGroup(self)
        self.histogram_region_group.setExclusive(True)
        self.histogram_region_buttons = {}
        for region_key, label_text, tooltip in (
            ("all", "전체", "선택한 면의 전체 분포"),
            ("left", "좌측", "선택한 상·하면의 좌측 분포"),
            ("right", "우측", "선택한 상·하면의 우측 분포"),
        ):
            button = QPushButton(label_text)
            button.setObjectName("histogramRegionButton")
            button.setCheckable(True)
            button.setToolTip(tooltip)
            self.histogram_region_group.addButton(button)
            self.histogram_region_buttons[region_key] = button
            region_layout.addWidget(button)
        self.histogram_region_buttons[self.current_histogram_region].setChecked(True)
        self.histogram_region_group.buttonClicked.connect(
            self._on_histogram_region_changed
        )
        self._update_histogram_region_controls()
        histogram_header.addWidget(region_selector)
        histogram_header.addWidget(self.top_contour_count_label)
        layout.addLayout(histogram_header)
        self.top_contour_histogram = TopContourHistogram()
        layout.addWidget(self.top_contour_histogram)
        self.analysis_preview_label.clicked.connect(
            lambda: self._show_image_modal(
                self.analysis_preview_pixmap, "기둥 기준(Blue) A/B 크롭 비교"
            )
        )
        self.source_preview_label.clicked.connect(
            lambda: self._show_image_modal(
                self.source_preview_pixmap, "최상단/빈도 기준(Gray) A/B 크롭 비교"
            )
        )
        self.ball_crop_preview_label.clicked.connect(
            lambda: self._show_image_modal(
                self.ball_crop_preview_pixmap, "탐지 볼 정사각형 내부 크롭"
            )
        )
        analysis_scroll.setWidget(content)
        card_layout.addWidget(analysis_scroll)
        return card

    def _create_control_sidebar(self):
        controls = QFrame()
        controls.setObjectName("controlCard")
        controls.setMinimumWidth(280)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(18, 16, 18, 18)
        controls_layout.setSpacing(12)

        controls_title = QLabel("탐지 설정")
        controls_title.setObjectName("cardTitle")
        controls_layout.addWidget(controls_title)
        controls_layout.addWidget(self._create_divider())
        controls_layout.addWidget(self._create_readonly_setting(
            "탐지 유형", "측면 컨투어"
        ))
        controls_layout.addWidget(self._create_side_cutting_toggle())
        controls_layout.addWidget(self._create_divider())
        controls_layout.addStretch(1)

        status_box = QFrame()
        status_box.setObjectName("statusCard")
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(12, 18, 12, 18)
        status_layout.setSpacing(7)
        self.status_label = QLabel("분석 대기")
        self.status_label.setObjectName("statusValue")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_detail_label = QLabel("이미지를 불러오세요.")
        self.status_detail_label.setObjectName("statusDetail")
        self.status_detail_label.setAlignment(Qt.AlignCenter)
        self.status_detail_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_detail_label)
        controls_layout.addWidget(status_box)
        controls_layout.addStretch(1)

        self.import_btn = QPushButton("불러오기")
        self.import_btn.setObjectName("secondaryButton")
        self.detect_btn = QPushButton("검출하기")
        self.detect_btn.setObjectName("primaryButton")
        self.import_btn.clicked.connect(self.on_import)
        self.detect_btn.clicked.connect(self.on_detect)
        controls_layout.addWidget(self.import_btn)
        controls_layout.addWidget(self.detect_btn)

        return controls

    @staticmethod
    def _create_divider():
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Plain)
        return divider

    def _create_readonly_setting(self, label_text, value_text):
        setting_row = QFrame()
        setting_row.setObjectName("settingRow")
        layout = QHBoxLayout(setting_row)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        label = QLabel(label_text)
        label.setObjectName("controlLabel")
        value = QLabel(value_text)
        value.setObjectName("settingValue")
        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(value)
        return setting_row

    def _create_side_cutting_toggle(self):
        """A 페이지의 측면 커팅 표시를 켜고 끄는 스위치를 만든다."""
        setting_row = QFrame()
        setting_row.setObjectName("settingRow")
        layout = QHBoxLayout(setting_row)
        layout.setContentsMargins(10, 8, 8, 8)
        layout.setSpacing(8)

        label = QLabel("측면 커팅")
        label.setObjectName("controlLabel")
        self.side_cutting_switch = QPushButton("OFF")
        self.side_cutting_switch.setObjectName("sideCuttingSwitch")
        self.side_cutting_switch.setCheckable(True)
        self.side_cutting_switch.setChecked(False)
        self.side_cutting_switch.toggled.connect(self._on_side_cutting_toggled)

        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(self.side_cutting_switch)
        return setting_row

    def _on_side_cutting_toggled(self, enabled):
        self.side_cutting_enabled = enabled
        self.side_cutting_switch.setText("ON" if enabled else "OFF")
        if self.image_paths:
            self._detect_current_image()

    def _create_preview_section(self, parent_layout, title_text, empty_text):
        preview_section = QWidget()
        preview_layout = QVBoxLayout(preview_section)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)
        title = QLabel(title_text)
        title.setObjectName("previewTitle")
        preview_layout.addWidget(title)

        preview_label = ClickableImageLabel(empty_text)
        preview_label.setObjectName("imagePreview")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setWordWrap(True)
        preview_label.setMinimumHeight(160)
        # 미리보기 Pixmap의 원본 폭이 카드의 최소 폭으로 전파되지 않게 한다.
        # 따라서 파일마다 미리보기 크기가 달라도 세 컬럼의 폭은 유지된다.
        preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        preview_layout.addWidget(preview_label)
        parent_layout.addWidget(preview_section, stretch=1)
        return preview_label

    def _create_image_label(self, message):
        label = ClickableImageLabel(message)
        label.setObjectName("imagePreview")
        label.setAlignment(Qt.AlignCenter)
        label.setWordWrap(True)
        label.setMinimumHeight(300)
        # 이미지 자체의 크기가 레이아웃 폭을 밀어내지 않도록 가로 크기 힌트를 무시한다.
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        return label

    def _show_image_modal(self, pixmap, title):
        if not pixmap.isNull():
            ImageModal(self).show_pixmap(pixmap, title)

    # ---------------- Capture ----------------
    @staticmethod
    def _next_capture_path(capture_dir, source_path):
        """원본 파일명별 다음 캡처 순번의 JPG 파일 경로를 반환한다."""

        filename = source_path.stem
        prefix = f"{filename}_"
        sequence_numbers = [
            int(path.stem[len(prefix):])
            for path in capture_dir.glob(f"{filename}_*.jpg")
            if path.stem.startswith(prefix)
            and path.stem[len(prefix):].isdecimal()
        ]
        next_sequence = max(sequence_numbers, default=0) + 1
        return capture_dir / f"{filename}_{next_sequence}.jpg"

    def on_capture(self):
        """현재 프로그램 화면을 실행 단위의 날짜·시간 폴더에 JPG로 저장한다."""

        if not self.image_paths or self.current_index < 0:
            QMessageBox.information(self, "캡처 저장", "먼저 TIFF 이미지를 불러오세요.")
            return

        if self.capture_session_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
            self.capture_session_dir = CAPTURE_ROOT / timestamp
            self.capture_session_dir.mkdir(parents=True, exist_ok=True)

        source_path = self.image_paths[self.current_index]
        capture_path = self._next_capture_path(self.capture_session_dir, source_path)
        if not self.grab().save(str(capture_path), "JPG", quality=95):
            QMessageBox.critical(self, "캡처 저장", "화면 캡처를 저장하지 못했습니다.")
            return

        self.status_detail_label.setText(
            f"캡처 저장 완료: {capture_path.relative_to(CAPTURE_ROOT)}"
        )

    # ---------------- Import ----------------
    def on_import(self):
        # 개발용 임시 로드: 아래 블록을 주석 처리하거나 삭제하면 파일 선택 창만 사용합니다.
        if DEV_IMAGE_DIR.is_dir():
            dev_paths = sorted(
                path
                for path in DEV_IMAGE_DIR.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            )
            if dev_paths:
                self._set_image_list(dev_paths)
                return

        dialog = QFileDialog(self, "이미지 불러오기", DEFAULT_DIR)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setNameFilter("TIFF 이미지 (*.tif *.tiff)")
        dialog.setLabelText(QFileDialog.Accept, "선택")

        if dialog.exec_() != QFileDialog.Accepted:
            return

        selected = [Path(path) for path in dialog.selectedFiles()]
        if not selected:
            return

        if len(selected) == 1 and selected[0].is_dir():
            paths = sorted(
                path for path in selected[0].iterdir()
                if path.suffix.lower() in IMAGE_EXTS
            )
            if not paths:
                QMessageBox.warning(self, "불러오기", "폴더에 이미지가 없습니다.")
                return
            self._set_image_list(paths)
            return

        files = [path for path in selected if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
        if files:
            self._set_image_list(files)

    def _set_image_list(self, paths):
        self.image_paths = paths
        self.current_index = 0
        self._show_current()

    # ---------------- Navigation ----------------
    def show_prev(self):
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            self._show_current()

    def show_next(self):
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._show_current()

    def _show_current(self):
        path = self.image_paths[self.current_index]
        self.original_pixmap = QPixmap(str(path))
        self._set_scaled_pixmap(self.image_label, self.original_pixmap)
        self.index_label.setText(f"{self.current_index + 1} / {len(self.image_paths)}")
        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.image_paths) - 1)

        self._detect_current_image()

    def on_detect(self):
        if not self.image_paths:
            QMessageBox.information(self, "검출", "먼저 이미지를 불러오세요.")
            return
        self._detect_current_image()

    def _detect_current_image(self):
        """한 장의 검출 시간을 재서 결과 정보에 이미지/초 처리 속도를 표시한다."""
        path = self.image_paths[self.current_index]
        started_at = perf_counter()
        try:
            result_image, result = create_detection_visualization(
                path,
                show_side_cutting=self.side_cutting_enabled,
            )
        except ValueError as error:
            self._clear_result("검출에 실패했습니다.")
            QMessageBox.critical(self, "검출", str(error))
            return

        elapsed_seconds = perf_counter() - started_at
        images_per_second = 1 / max(elapsed_seconds, 0.000001)
        self.histogram_image_width = result_image.shape[1]
        self.histogram_center_split_x = result.center_split_x or self.histogram_image_width // 2
        self._show_original_image(result.source_visualization)
        self._show_result_image(result_image)
        self._show_analysis_preview(result.analysis_preview)
        self._show_source_preview(result.source_preview)
        self._show_ball_crop_preview(result.ball_square_crop_preview)
        self._update_info_label(path, result, elapsed_seconds, images_per_second)
        self._show_top_contour_histogram(result.measurements)

    def _show_result_image(self, bgr_image):
        self.result_pixmap = self._pixmap_from_image(bgr_image)
        self._set_scaled_pixmap(self.result_label, self.result_pixmap)

    def _show_original_image(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            return
        self.original_pixmap = self._pixmap_from_image(bgr_image)
        self._set_scaled_pixmap(self.image_label, self.original_pixmap)

    def _show_analysis_preview(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            self.analysis_preview_pixmap = QPixmap()
            self.analysis_preview_label.setPixmap(QPixmap())
            self.analysis_preview_label.setText("검출된 분석 개체가 없습니다.")
            return

        self.analysis_preview_pixmap = self._pixmap_from_image(bgr_image)
        self._set_scaled_pixmap(
            self.analysis_preview_label,
            self.analysis_preview_pixmap,
            preview_scale=PREVIEW_SCALE,
        )

    def _show_source_preview(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            self.source_preview_pixmap = QPixmap()
            self.source_preview_label.setPixmap(QPixmap())
            self.source_preview_label.setText("표시할 A 페이지 좌표가 없습니다.")
            return

        self.source_preview_pixmap = self._pixmap_from_image(bgr_image)
        self._set_scaled_pixmap(
            self.source_preview_label,
            self.source_preview_pixmap,
            preview_scale=PREVIEW_SCALE,
        )

    def _show_ball_crop_preview(self, bgr_image):
        if bgr_image is None or bgr_image.size == 0:
            self.ball_crop_preview_pixmap = QPixmap()
            self.ball_crop_preview_label.setPixmap(QPixmap())
            self.ball_crop_preview_label.setText(
                "조건을 만족하는 볼이 탐지되지 않았습니다."
            )
            return

        self.ball_crop_preview_pixmap = self._pixmap_from_image(bgr_image)
        self._set_scaled_pixmap(
            self.ball_crop_preview_label,
            self.ball_crop_preview_pixmap,
            preview_scale=PREVIEW_SCALE,
        )

    def _on_histogram_side_changed(self, button):
        self.current_histogram_side = next(
            side_key
            for side_key, side_button in self.histogram_side_buttons.items()
            if side_button is button
        )
        self._update_histogram_region_controls()
        self._show_top_contour_histogram(self.histogram_measurements)

    def _on_histogram_region_changed(self, button):
        self.current_histogram_region = next(
            region_key
            for region_key, region_button in self.histogram_region_buttons.items()
            if region_button is button
        )
        self._show_top_contour_histogram(self.histogram_measurements)

    def _update_histogram_region_controls(self):
        """좌·우 분포 필터는 상·하면 y 좌표 분포에서만 사용한다."""
        enabled = self.current_histogram_side in {"top", "bottom"}
        for button in self.histogram_region_buttons.values():
            button.setEnabled(enabled)

    def _show_top_contour_histogram(self, measurements):
        """선택한 면에 처음 닿는 모든 좌표의 분포를 표시한다."""

        self.histogram_measurements = measurements
        point_attribute, coordinate_index, title = {
            "top": ("top_points", 1, "상면 외곽 컨투어 첫 접점 y 좌표 분포"),
            "bottom": ("bottom_points", 1, "하면 외곽 컨투어 첫 접점 y 좌표 분포"),
            "left": ("left_points", 0, "좌면 외곽 컨투어 첫 접점 x 좌표 분포"),
            "right": ("right_points", 0, "우면 외곽 컨투어 첫 접점 x 좌표 분포"),
        }[self.current_histogram_side]
        coordinates = [
            point[coordinate_index]
            for measurement in measurements
            for point in getattr(measurement, point_attribute)
            if (
                self.current_histogram_side not in {"top", "bottom"}
                or self.current_histogram_region == "all"
                or (
                    point[0] < self.histogram_center_split_x
                    if self.current_histogram_region == "left"
                    else point[0] >= self.histogram_center_split_x
                )
            )
        ]
        coordinate_axis = "y" if coordinate_index == 1 else "x"
        coordinate_range = self._get_histogram_coordinate_range(
            measurements, coordinate_index
        )
        region_name = {
            "all": "전체",
            "left": "좌측",
            "right": "우측",
        }[self.current_histogram_region]
        if self.current_histogram_side not in {"top", "bottom"}:
            region_name = "전체"
        self.top_contour_histogram.set_coordinates(
            coordinates,
            coordinate_axis,
            coordinate_range,
            self.current_histogram_side,
        )
        self.info_label.setPlainText(
            "\n".join(self.program_log_lines + self.top_contour_histogram.log_lines)
        )
        self.histogram_title.setText(f"{title} · {region_name}")
        self.top_contour_count_label.setText(f"{region_name} {len(coordinates):,}개")

    @staticmethod
    def _get_histogram_coordinate_range(measurements, coordinate_index):
        """기준선 사각형 안에서 사용할 전체 x/y 좌표 범위를 구한다."""

        ranges = []
        for measurement in measurements:
            if coordinate_index == 1:
                first_point = get_primary_contact_reference_point(
                    measurement.top_points, 1, use_minimum=True
                )
                last_point = get_primary_contact_reference_point(
                    measurement.bottom_points, 1, use_minimum=False
                )
            else:
                first_point = get_primary_contact_reference_point(
                    measurement.left_points, 0, use_minimum=True
                )
                last_point = get_primary_contact_reference_point(
                    measurement.right_points, 0, use_minimum=False
                )

            if first_point is not None and last_point is not None:
                ranges.append(
                    (first_point[coordinate_index], last_point[coordinate_index])
                )

        if not ranges:
            return None
        minimum = min(first_coordinate for first_coordinate, _ in ranges)
        maximum = max(last_coordinate for _, last_coordinate in ranges)
        return (minimum, maximum) if minimum <= maximum else None

    @staticmethod
    def _pixmap_from_image(image):
        image = image.copy()
        height, width, channels = image.shape
        if channels == 4:
            rgba_image = image[:, :, [2, 1, 0, 3]].copy()
            qimage = QImage(
                rgba_image.tobytes(), width, height, width * 4, QImage.Format_RGBA8888
            )
        else:
            qimage = QImage(
                image.tobytes(), width, height, width * 3, QImage.Format_BGR888
            )
        return QPixmap.fromImage(qimage.copy())

    @staticmethod
    def _set_scaled_pixmap(label, pixmap, preview_scale=1.0):
        target_size = label.size()
        if preview_scale != 1.0:
            target_size.setWidth(round(target_size.width() * preview_scale))
            target_size.setHeight(round(target_size.height() * preview_scale))
        label.setPixmap(
            pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _clear_result(self, message):
        self.result_pixmap = QPixmap()
        self.result_label.setPixmap(QPixmap())
        self.result_label.setText(message)
        self.analysis_preview_pixmap = QPixmap()
        self.analysis_preview_label.setPixmap(QPixmap())
        self.analysis_preview_label.setText(
            "기둥 기준선을 같은 좌표로 적용한 A/B 페이지 크롭 결과를 비교합니다."
        )
        self.source_preview_pixmap = QPixmap()
        self.source_preview_label.setPixmap(QPixmap())
        self.source_preview_label.setText(
            "같은 회색 기준선을 적용한 A/B 페이지 크롭 결과를 비교합니다."
        )
        self.ball_crop_preview_pixmap = QPixmap()
        self.ball_crop_preview_label.setPixmap(QPixmap())
        self.ball_crop_preview_label.setText(
            "조건을 만족하는 볼이 탐지되면 정사각형 내부가 표시됩니다."
        )
        self.top_contour_histogram.set_coordinates(())
        self.histogram_image_width = 0
        self.histogram_center_split_x = 0
        self.histogram_measurements = []
        self.top_contour_count_label.setText("전체 0개")
        self.status_label.setText("분석 실패")
        self.status_label.setStyleSheet("color: #F04452;")
        self.status_detail_label.setText(message)

    def _update_info_label(
        self, path, result=None, elapsed_seconds=None, images_per_second=None
    ):
        lines = [
            f"1. 이미지 크기  {self.original_pixmap.width()} × {self.original_pixmap.height()} px",
        ]
        if result is not None:
            lines.append(f"2. B 페이지 외곽 컨투어 개수 : {len(result.contours)}개")
            point_count = sum(
                len(measurement.top_points)
                + len(measurement.bottom_points)
                + len(measurement.left_points)
                + len(measurement.right_points)
                for measurement in result.measurements
            )
            lines.append(f"3. 4면 첫 접점 개수 : {point_count}개")
            if result.a_ball_bottommost_y is None:
                lines.append("4. A 페이지 하면 검출점 최하단 : 없음")
            else:
                lines.append(
                    "4. A 페이지 하면 검출점 최하단 "
                    f"y={result.a_ball_bottommost_y} : "
                    f"{result.a_ball_bottommost_count}개"
                )
            for section_index, point in enumerate(
                result.a_ball_bottommost_points_by_third, start=1
            ):
                point_text = "없음" if point is None else f"({point[0]}, {point[1]})"
                lines.append(
                    f"5-{section_index}. 하면 {section_index}등분 최하단 좌표 : {point_text}"
                )
            for index, measurement in enumerate(result.measurements, start=1):
                lines.extend(self._first_contact_log_lines(measurement, index))
            lines.extend(result.density_log_lines)
        self.program_log_lines = lines
        self.info_label.setPlainText("\n".join(self.program_log_lines))

        if elapsed_seconds is not None and images_per_second is not None:
            self.header_metadata_label.setText(
                f"파일 위치  {path.name} · 처리 시간 {elapsed_seconds * 1000:.1f} ms · "
                f"{images_per_second:.0f} image/sec"
            )
        if result is not None:
            contour_count = len(result.contours)
            self.status_label.setText("분석 완료" if contour_count else "컨투어 미검출")
            self.status_label.setStyleSheet(
                "color: #3182F6;" if contour_count else "color: #F59F00;"
            )
            self.status_detail_label.setText(
                f"B 페이지 외곽 컨투어 {contour_count}개를 확인했습니다."
            )

    @staticmethod
    def _first_contact_log_lines(measurement, contour_index):
        """방향별 첫 접점 중 가장 먼저 닿는 좌표를 로그용 텍스트로 만든다."""

        point_groups = (
            ("상", "y", measurement.top_points, 1, True),
            ("하", "y", measurement.bottom_points, 1, False),
            ("좌", "x", measurement.left_points, 0, True),
            ("우", "x", measurement.right_points, 0, False),
        )

        lines = [f"컨투어 {contour_index} · 가장 먼저 닿는 좌표"]
        for direction, axis, points, coordinate_index, use_minimum in point_groups:
            reference_point = get_primary_contact_reference_point(
                points, coordinate_index, use_minimum
            )
            if reference_point is None:
                lines.append(f"{direction} : 없음")
                continue
            lines.append(
                f"{direction} : {axis}={reference_point[coordinate_index]} "
                f"· 대표점 ({reference_point[0]}, {reference_point[1]})"
            )
        return lines

    def _refresh_scaled_pixmaps(self):
        if not self.original_pixmap.isNull():
            self._set_scaled_pixmap(self.image_label, self.original_pixmap)
        if not self.result_pixmap.isNull():
            self._set_scaled_pixmap(self.result_label, self.result_pixmap)
        if not self.analysis_preview_pixmap.isNull():
            self._set_scaled_pixmap(
                self.analysis_preview_label,
                self.analysis_preview_pixmap,
                preview_scale=PREVIEW_SCALE,
            )
        if not self.source_preview_pixmap.isNull():
            self._set_scaled_pixmap(
                self.source_preview_label,
                self.source_preview_pixmap,
                preview_scale=PREVIEW_SCALE,
            )
        if not self.ball_crop_preview_pixmap.isNull():
            self._set_scaled_pixmap(
                self.ball_crop_preview_label,
                self.ball_crop_preview_pixmap,
                preview_scale=PREVIEW_SCALE,
            )
    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_scaled_pixmaps()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_scaled_pixmaps()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Left:
            self.show_prev()
        elif event.key() == Qt.Key_Right:
            self.show_next()
        else:
            super().keyPressEvent(event)
