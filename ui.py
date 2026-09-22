"""Defect Detector의 macOS 스타일 화면과 사용자 상호작용을 정의한다."""
from datetime import datetime
from pathlib import Path
from time import perf_counter

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from bottom.pipeline import run_bottom_detection
from bottom.circles import (
    MAX_CENTER_DISTANCE, MAX_CIRCLE_AREA, MIN_CIRCULARITY,
    MIN_CIRCLE_AREA, MIN_CIRCLE_ASPECT_RATIO,
)
from contour import get_primary_contact_reference_point
from general import NotesRepository, next_capture_path, to_bgr
from side.pipeline import measure_ball_arcs, run_side_detection
from styles import APP_STYLESHEET
from ui_components import (
    ClickableImageLabel,
    ImageModal,
    InspectionInfoModal,
    NoteModal,
    TopContourHistogram,
    show_pixel_tooltip, 
)


IMAGE_EXTS = {".tif", ".tiff"}
DEFAULT_DIR = "dataset/"
PREVIEW_SCALE = 0.8
# # DEV_IMAGE_DIR = Path("/Users/dongwookang/diehand_cv/dataset/side/total")
DEV_IMAGE_DIR = Path("~/Library/CloudStorage/GoogleDrive-dongwoodustinkang@gmail.com/My Drive/2_KETI/200_DieHandling/220_코드파일/die-detection/dataset/")
CAPTURE_ROOT = Path(__file__).resolve().parent / "captures"
NOTES_CSV_PATH = Path(__file__).resolve().parent / "notes.csv"

ALGORITHM_OPTIONS = {
    "side": {"label": "Side"},
    "bottom": {"label": "Bottom"},
    "top": {"label": "Top"},
}

class MainWindow(QMainWindow):
    """A/B TIFF의 모드별 검출 결과를 확인하는 메인 창."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("불량 검출기")
        self.resize(2120, 880)
        self.setMinimumSize(1200, 720)

        self.image_paths = []
        self.current_index = -1
        self.original_pixmap = QPixmap() # 원본 이미지
        self.result_pixmap = QPixmap() # B 페이지 이미지
        self.raw_original_pixmap = QPixmap()
        self.raw_result_pixmap = QPixmap()
        self.annotated_original_pixmap = QPixmap()
        self.annotated_result_pixmap = QPixmap()
        self.show_analysis_overlay = True
        self.analysis_preview_pixmap = QPixmap()
        self.source_preview_pixmap = QPixmap()
        self.ball_crop_preview_pixmap = QPixmap()
        self._last_side_result = None
        self.current_histogram_side = "top"
        self.current_histogram_region = "all"
        self.histogram_image_width = 0
        self.histogram_center_split_x = 0
        self.histogram_measurements = []
        self.program_log_lines = []
        self.inspection_info_text = "이미지를 불러오면 상세 정보가 표시됩니다."
        self.notes_repository = NotesRepository(NOTES_CSV_PATH)
        self.capture_session_dir = None
        self.active_algorithm = "side"
        self.capture_btn = QPushButton("📸")
        self.capture_btn.setObjectName("captureIconButton")
        self.capture_btn.setToolTip("현재 화면 캡처 저장")
        self.capture_btn.clicked.connect(self.on_capture)
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(APP_STYLESHEET)

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        self.central_widget = central
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(16)

        root.addLayout(self._create_header())
        root.addLayout(self._create_workspace(), stretch=1)
        self._create_floating_navigation()
        self.prev_btn.setEnabled(False)
        self.note_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.detect_btn.setEnabled(False)
        self._configure_algorithm_panels()

    def _position_floating_navigation(self):
        """Anchor the transient image controls to the lower centre of the workspace."""
        if not hasattr(self, "floating_navigation"):
            return
        navigation = self.floating_navigation
        navigation.adjustSize()
        x = max(0, (self.central_widget.width() - navigation.width()) // 2)
        y = max(0, self.central_widget.height() - navigation.height() - 20)
        navigation.move(x, y)
        navigation.raise_()

    def enterEvent(self, event):
        super().enterEvent(event)
        if hasattr(self, "floating_navigation"):
            self._position_floating_navigation()
            self.floating_navigation.show()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if hasattr(self, "floating_navigation"):
            self.floating_navigation.hide()

    def _update_algorithm_title(self):
        label = ALGORITHM_OPTIONS[self.active_algorithm]["label"]
        self.algorithm_button.setText(f"{label} Detection")

    def _on_algorithm_changed(self, _checked=False):
        """Switch the processing mode and return to the pre-import state."""
        action = self.sender()
        algorithm_key = action.data() if action is not None else None
        if algorithm_key not in ALGORITHM_OPTIONS or algorithm_key == self.active_algorithm:
            return

        self.active_algorithm = algorithm_key
        self._update_algorithm_title()
        self._reset_for_algorithm_change()

    def _reset_for_algorithm_change(self):
        """Clear an earlier run so files are never analysed with the wrong mode."""
        algorithm_label = ALGORITHM_OPTIONS[self.active_algorithm]["label"]
        self._last_side_result = None
        self.ball_arc_summary.clear()
        self.ball_arc_summary.hide()
        self.image_paths = []
        self.current_index = -1
        self.original_pixmap = QPixmap()
        self.result_pixmap = QPixmap()
        self.raw_original_pixmap = QPixmap()
        self.raw_result_pixmap = QPixmap()
        self.annotated_original_pixmap = QPixmap()
        self.annotated_result_pixmap = QPixmap()
        self.analysis_preview_pixmap = QPixmap()
        self.source_preview_pixmap = QPixmap()
        self.ball_crop_preview_pixmap = QPixmap()
        self.capture_session_dir = None
        self.histogram_image_width = 0
        self.histogram_center_split_x = 0
        self.histogram_measurements = []
        self.program_log_lines = []

        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("A/B TIFF 이미지를 불러오세요.")
        self.result_label.setPixmap(QPixmap())
        self.result_label.setText("B 페이지 컨투어 분석 결과가 표시됩니다.")
        self.analysis_preview_label.setPixmap(QPixmap())
        self.analysis_preview_label.setText("기둥 기준선을 같은 좌표로 적용한 A/B 페이지 크롭 결과를 비교합니다.")
        self.analysis_preview_label.timing_label.setText("기둥 기준 + 볼 검출 · — ms")
        self.source_preview_label.setPixmap(QPixmap())
        self.source_preview_label.setText("같은 회색 기준선을 적용한 A/B 페이지 크롭 결과를 비교합니다.")
        self.source_preview_label.timing_label.setText("최상단/빈도 + 볼 검출 · — ms")
        self.ball_crop_preview_label.setPixmap(QPixmap())
        self.ball_crop_preview_label.setText("조건을 만족하는 볼이 탐지되면 정사각형 내부가 표시됩니다.")
        self.top_contour_histogram.set_coordinates(())
        self.histogram_title.setText("상면 외곽 컨투어 첫 접점 y 좌표 분포 · 전체")
        self.top_contour_count_label.setText("전체 0개")
        self.inspection_info_text = "이미지를 불러오면 상세 정보가 표시됩니다."
        self.file_context_label.setText("Filename · 선택된 TIFF 이미지 없음")
        self.header_metadata_label.setText(f"{algorithm_label} 알고리즘 · TIFF 이미지를 불러오세요.")
        self._set_detection_state("idle")
        self.prev_btn.setEnabled(False)
        self.note_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.index_label.setText("0/00")
        self.detect_btn.setEnabled(False)
        self._configure_algorithm_panels()

    def _configure_algorithm_panels(self):
        """Bottom의 칩 위치 단계와 기존 검사 화면의 표시 항목을 구분한다."""
        is_bottom = self.active_algorithm == "bottom"
        self.analysis_card_title.setText("원 후보 · A 영상" if is_bottom else "상세 컨투어")
        self.bottom_chip_summary.setVisible(is_bottom)
        self.bottom_chip_summary.setText("칩 기준 6개 영역에서 원 후보를 찾습니다.")
        self.source_preview_label.section_widget.setVisible(True)
        self.ball_crop_preview_label.section_widget.setVisible(not is_bottom)
        self.ball_arc_switch.setVisible(self.active_algorithm == "side")
        self.histogram_controls.setVisible(not is_bottom)
        self.top_contour_histogram.setVisible(not is_bottom)
        preview = self.analysis_preview_label
        preview.title_label.setText("칩 영역 · ROI 검사" if is_bottom else preview.default_title)
        preview.modal_title = "칩 영역 · ROI 검사" if is_bottom else preview.default_modal_title
        preview.setText(
            "칩이 검출되면 6개 검색 영역과 실제 윤곽을 표시합니다."
            if is_bottom else preview.default_empty_text
        )
        preview.timing_label.setText(
            f"{'원 후보 검사' if is_bottom else preview.timing_caption} · — ms"
        )
        detail = self.source_preview_label
        detail.title_label.setText(
            "6개 영역 · C: 원형도 / A: 면적(px²)" if is_bottom else detail.default_title
        )
        detail.modal_title = "원 후보 · 6개 영역 확대" if is_bottom else detail.default_modal_title
        detail.setText("각 검색 영역의 A 영상과 실제 윤곽을 표시합니다." if is_bottom else detail.default_empty_text)
        detail.timing_label.setVisible(not is_bottom)
        detail.setMinimumHeight(360 if is_bottom else 160)
        if is_bottom:
            self.result_label.setText("B 페이지는 원본 참조용으로 표시됩니다.")
        self._refresh_ball_preview()
        self._on_overlay_toggled(self.show_analysis_overlay)

    def _create_header(self):
        header = QHBoxLayout()
        header.setSpacing(10)

        title_group = QVBoxLayout()
        title_group.setSpacing(2)
        self.algorithm_button = QToolButton()
        self.algorithm_button.setObjectName("algorithmButton")
        self.algorithm_button.setCursor(Qt.PointingHandCursor)
        self.algorithm_button.setPopupMode(QToolButton.InstantPopup)
        algorithm_menu = QMenu(self.algorithm_button)
        for algorithm_key, option in ALGORITHM_OPTIONS.items():
            action = algorithm_menu.addAction(f"{option['label']} Detection")
            action.setData(algorithm_key)
            action.triggered.connect(self._on_algorithm_changed)
        self.algorithm_button.setMenu(algorithm_menu)
        self._update_algorithm_title()
        self.header_metadata_label = QLabel(
            "측면 커팅이 자동으로 적용됩니다."
        )
        self.header_metadata_label.setObjectName("headerMetadata")
        title_group.addWidget(self.algorithm_button)
        title_group.addWidget(self.header_metadata_label)
        header.addLayout(title_group)
        header.addStretch(1)

        file_context = QFrame()
        file_context.setObjectName("fileContext")
        file_layout = QHBoxLayout(file_context)
        file_layout.setContentsMargins(12, 7, 12, 7)
        file_layout.setSpacing(7)
        file_icon = QLabel("PATH")
        file_icon.setObjectName("fileContextIcon")
        self.file_context_label = QLabel("Filename · 선택된 TIFF 이미지 없음")
        self.file_context_label.setObjectName("fileContextLabel")
        self.file_context_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        file_layout.addWidget(file_icon)
        file_layout.addWidget(self.file_context_label, stretch=1)
        file_context.setMinimumWidth(360)
        self.import_btn = QPushButton("불러오기")
        self.import_btn.setObjectName("secondaryButton")
        self.detect_btn = QPushButton("검출하기")
        self.detect_btn.setObjectName("primaryButton")
        self.import_btn.clicked.connect(self.on_import)
        self.detect_btn.clicked.connect(self.on_detect)
        action_group = QFrame()
        action_group.setObjectName("headerActionGroup")
        action_layout = QHBoxLayout(action_group)
        action_layout.setContentsMargins(5, 5, 5, 5)
        action_layout.setSpacing(6)
        action_layout.addWidget(file_context, stretch=1)
        action_layout.addWidget(self.import_btn)
        action_layout.addWidget(self.detect_btn)
        header.addWidget(action_group)

        self.detection_result_badge = QFrame()
        self.detection_result_badge.setObjectName("detectionResultBadge")
        badge_layout = QHBoxLayout(self.detection_result_badge)
        badge_layout.setContentsMargins(12, 7, 12, 7)
        self.detection_result_label = QLabel()
        self.detection_result_label.setObjectName("detectionResultLabel")
        badge_layout.addWidget(self.detection_result_label)
        header.addWidget(self.detection_result_badge)
        self._set_detection_state("idle")

        return header

    def _set_detection_state(self, state):
        state_text = {
            "idle": "대기",
            "detected": "검출",
            "not_detected": "미검출",
            "error": "오류",
        }[state]
        if self.active_algorithm == "bottom":
            state_text = {"detected": "원 후보 검출", "not_detected": "검토 필요"}.get(
                state, state_text
            )
        self.detection_result_badge.setProperty("state", state)
        self.detection_result_label.setText(state_text)
        self.detection_result_badge.style().unpolish(self.detection_result_badge)
        self.detection_result_badge.style().polish(self.detection_result_badge)

    def _create_floating_navigation(self):
        """Keep image navigation close at hand without permanently occupying workspace."""
        navigation = QFrame(self.central_widget)
        navigation.setObjectName("floatingNavigation")
        navigation.setAttribute(Qt.WA_StyledBackground, True)
        layout = QHBoxLayout(navigation)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.prev_btn = QPushButton("‹")
        self.prev_btn.setObjectName("floatingNavigationButton")
        self.prev_btn.setToolTip("이전 이미지 (←)")
        self.note_btn = QPushButton("🗒️")
        self.note_btn.setObjectName("floatingNavigationButton")
        self.note_btn.setToolTip("현재 이미지에 메모 추가")
        self.next_btn = QPushButton("›")
        self.next_btn.setObjectName("floatingNavigationButton")
        self.next_btn.setToolTip("다음 이미지 (→)")
        self.index_label = QLabel("0/00")
        self.index_label.setObjectName("floatingIndexLabel")
        self.prev_btn.clicked.connect(self.show_prev)
        self.note_btn.clicked.connect(self._show_note_modal)
        self.next_btn.clicked.connect(self.show_next)

        layout.addWidget(self.prev_btn)
        layout.addWidget(self.note_btn)
        layout.addWidget(self.index_label)
        layout.addWidget(self.capture_btn)
        layout.addWidget(self.next_btn)
        navigation.adjustSize()
        navigation.hide()
        self.floating_navigation = navigation

    def _create_workspace(self):
        """원본·분석·설정을 역할에 맞는 세 영역으로 배치한다."""
        workspace = QHBoxLayout()
        workspace.setSpacing(16)

        workspace.addWidget(self._create_source_card(), stretch=6)
        workspace.addWidget(self._create_analysis_card(), stretch=4)
        return workspace

    def _create_source_card(self):
        card = QFrame()
        card.setObjectName("imageCard")
        self._apply_glass_elevation(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("이미지 비교")
        title.setObjectName("cardTitle")
        self.overlay_switch = QPushButton("분석선")
        self.overlay_switch.setObjectName("imageOverlaySwitch")
        self.overlay_switch.setCheckable(True)
        self.overlay_switch.setChecked(True)
        self.overlay_switch.setToolTip("기준선·컨투어선 표시 전환")
        self.overlay_switch.toggled.connect(self._on_overlay_toggled)
        info_button = QPushButton("ℹ")
        info_button.setObjectName("infoButton")
        info_button.setToolTip("검사 상세 정보 보기")
        info_button.clicked.connect(self._show_inspection_info)
        title_row.addWidget(title)
        title_row.addWidget(self.overlay_switch)
        title_row.addWidget(info_button)
        title_row.addStretch()
        layout.addLayout(title_row)
        layout.addWidget(self._create_divider())

        image_pair = QHBoxLayout()
        image_pair.setSpacing(14)

        source_panel, self.image_label = self._create_image_panel(
            "Image A · 분석 오버레이", "A/B TIFF 이미지를 불러오세요."
        )
        result_panel, self.result_label = self._create_image_panel(
            "Image B · 검사 오버레이", "B 페이지 컨투어 분석 결과가 표시됩니다."
        )
        self.image_label.clicked.connect(
            lambda: self._show_image_modal(self.original_pixmap, "A 페이지")
        )
        self.result_label.clicked.connect(
            lambda: self._show_image_modal(
                self.result_pixmap,
                "B 페이지 원본" if self.active_algorithm == "bottom" else "B 페이지 결과",
            )
        )
        self.image_label.hovered.connect(
            lambda position: show_pixel_tooltip(
                self.image_label, self.original_pixmap, position
            )
        )
        self.result_label.hovered.connect(
            lambda position: show_pixel_tooltip(
                self.result_label, self.result_pixmap, position
            )
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
        label.panel_title = title
        layout.addWidget(label, stretch=1)
        return panel, label

    def _create_analysis_card(self):
        card = QFrame()
        card.setObjectName("analysisCard")
        self._apply_glass_elevation(card)
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

        title = QLabel("상세 컨투어")
        title.setObjectName("cardTitle")
        self.analysis_card_title = title
        layout.addWidget(title)
        layout.addWidget(self._create_divider())
        self.bottom_chip_summary = QLabel()
        self.bottom_chip_summary.setWordWrap(True)
        layout.addWidget(self.bottom_chip_summary)

        self.analysis_preview_label = self._create_preview_section(
            layout,
            "기둥 기준 · A/B 크롭 비교",
            "기둥 기준선을 같은 좌표로 적용한 A/B 페이지 크롭 결과를 비교합니다.",
            timing_caption="기둥 기준 + 볼 검출",
        )
        self.source_preview_label = self._create_preview_section(
            layout,
            "최상단/빈도 기준 · 크롭",
            "같은 회색 기준선을 적용한 A/B 페이지 크롭 결과를 비교합니다.",
            timing_caption="최상단/빈도 + 볼 검출",
        )
        self.ball_crop_preview_label = self._create_preview_section(
            layout,
            "볼 검출 · 상세 크롭",
            "조건을 만족하는 볼이 탐지되면 정사각형 내부가 표시됩니다.",
        )
        self.ball_arc_switch = QPushButton("원호 비교")
        self.ball_arc_switch.setObjectName("ballArcSwitch")
        self.ball_arc_switch.setCheckable(True)
        self.ball_arc_switch.setCursor(Qt.PointingHandCursor)
        self.ball_arc_switch.setToolTip(
            "끔: 기존 원본 크롭 / 켬: 약한 선명화 + 하단 접점 + 원호 피팅\n"
            "반경 10–20 px (원본 기준). 비교용 지표이며 최종 OK/NG 판정이 아닙니다."
        )
        self.ball_arc_switch.toggled.connect(self._refresh_ball_preview)
        self.ball_crop_preview_label.header_layout.addWidget(self.ball_arc_switch)
        self.ball_arc_summary = QLabel()
        self.ball_arc_summary.setObjectName("ballArcSummary")
        self.ball_arc_summary.setWordWrap(True)
        self.ball_arc_summary.setToolTip(
            "시험 기준: 원호 오차 ≤ 1.5 px, 대칭 오차 ≤ 2 px, 관측각 ≥ 70°, "
            "윤곽 열 연속성 ≥ 90%. 경계 접촉·반경 한계는 검토.\n"
            "OK/NG 데이터 분포로 기준 검증이 필요합니다. 상부 접합부는 평가하지 않습니다."
        )
        self.ball_arc_summary.hide()
        self.ball_crop_preview_label.section_widget.layout().addWidget(self.ball_arc_summary)
        self.histogram_controls = QWidget()
        histogram_header = QHBoxLayout(self.histogram_controls)
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
        layout.addWidget(self.histogram_controls)
        self.top_contour_histogram = TopContourHistogram()
        layout.addWidget(self.top_contour_histogram)
        self.analysis_preview_label.clicked.connect(
            lambda: self._show_image_modal(
                self.analysis_preview_pixmap,
                self.analysis_preview_label.modal_title,
                enable_pixel_tooltip=False,
            )
        )
        self.source_preview_label.clicked.connect(
            lambda: self._show_image_modal(
                self.source_preview_pixmap,
                self.source_preview_label.modal_title,
                enable_pixel_tooltip=False,
            )
        )
        self.ball_crop_preview_label.clicked.connect(
            lambda: self._show_image_modal(
                self.ball_crop_preview_pixmap,
                self.ball_crop_preview_label.title_label.text(),
                enable_pixel_tooltip=False,
            )
        )
        for label, modal_title in (
            (self.analysis_preview_label, "기둥 기준(Blue) A/B 크롭 비교"),
            (self.source_preview_label, "최상단/빈도 기준(Gray) A/B 크롭 비교"),
        ):
            label.default_modal_title = modal_title
            label.modal_title = modal_title
        analysis_scroll.setWidget(content)
        card_layout.addWidget(analysis_scroll)
        return card

    @staticmethod
    def _apply_glass_elevation(widget, blur_radius=28, y_offset=6):
        """Give a translucent material card a restrained macOS-style elevation."""
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(blur_radius)
        shadow.setOffset(0, y_offset)
        shadow.setColor(QColor(31, 40, 55, 26))
        widget.setGraphicsEffect(shadow)

    @staticmethod
    def _create_divider():
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Plain)
        return divider

    def _create_preview_section(
        self, parent_layout, title_text, empty_text, timing_caption=None
    ):
        preview_section = QWidget()
        preview_layout = QVBoxLayout(preview_section)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel(title_text)
        title.setObjectName("previewTitle")
        title_row.addWidget(title)
        title_row.addStretch()
        timing_label = None
        if timing_caption is not None:
            timing_label = QLabel(f"{timing_caption} · — ms")
            timing_label.setObjectName("previewTiming")
            title_row.addWidget(timing_label)
        preview_layout.addLayout(title_row)

        preview_label = ClickableImageLabel(empty_text)
        preview_label.setObjectName("imagePreview")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setWordWrap(True)
        preview_label.setMinimumHeight(160)
        # 미리보기 Pixmap의 원본 폭이 카드의 최소 폭으로 전파되지 않게 한다.
        # 따라서 파일마다 미리보기 크기가 달라도 세 컬럼의 폭은 유지된다.
        preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        preview_label.timing_label = timing_label
        preview_label.timing_caption = timing_caption
        preview_label.title_label = title
        preview_label.default_title = title_text
        preview_label.default_empty_text = empty_text
        preview_label.section_widget = preview_section
        preview_label.header_layout = title_row
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

    def _show_inspection_info(self):
        InspectionInfoModal(self).show_text(self.inspection_info_text)

    def _show_image_modal(self, pixmap, title, enable_pixel_tooltip=True):
        if not pixmap.isNull():
            ImageModal(self, enable_pixel_tooltip).show_pixmap(pixmap, title)

    def _on_overlay_toggled(self, enabled):
        self.show_analysis_overlay = enabled
        self.overlay_switch.setText("분석선" if enabled else "원본")
        display_name = "분석 오버레이" if enabled else "원본"
        self.image_label.panel_title.setText(f"Image A · {display_name}")
        self.result_label.panel_title.setText(
            "Image B · 원본 (참고)" if self.active_algorithm == "bottom"
            else f"Image B · {display_name}"
        )
        self._refresh_image_comparison()

    def _refresh_image_comparison(self):
        """Swap the comparison card between raw TIFF pages and annotated results."""
        source_pixmap = (
            self.annotated_original_pixmap
            if self.show_analysis_overlay
            else self.raw_original_pixmap
        )
        result_pixmap = (
            self.annotated_result_pixmap
            if self.show_analysis_overlay
            else self.raw_result_pixmap
        )
        if not source_pixmap.isNull():
            self.original_pixmap = source_pixmap
            self._set_scaled_pixmap(self.image_label, source_pixmap)
        if not result_pixmap.isNull():
            self.result_pixmap = result_pixmap
            self._set_scaled_pixmap(self.result_label, result_pixmap)

    # ---------------- Capture ----------------
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
        capture_path = next_capture_path(self.capture_session_dir, source_path)
        if not self.grab().save(str(capture_path), "JPG", quality=95):
            QMessageBox.critical(self, "캡처 저장", "화면 캡처를 저장하지 못했습니다.")
            return

        self.header_metadata_label.setText(
            f"캡처 저장 완료: {capture_path.relative_to(CAPTURE_ROOT)}"
        )

    def _show_note_modal(self):
        if not self.image_paths or self.current_index < 0:
            return
        source_path = self.image_paths[self.current_index]
        modal = NoteModal(self, source_path, self.notes_repository)
        modal.show_modal()
        if modal.result() == QDialog.Accepted:
            self.header_metadata_label.setText(f"메모 저장 완료: {source_path.name}")

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
        self.detect_btn.setEnabled(True)
        self.note_btn.setEnabled(True)
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
        self.file_context_label.setText(f"{path.parent.name}  /  {path.name}")
        position = f"{self.current_index + 1} / {len(self.image_paths)}"
        self.index_label.setText(position.replace(" / ", "/"))
        self.prev_btn.setToolTip(f"이전 이미지 (←) · {position}")
        self.next_btn.setToolTip(f"다음 이미지 (→) · {position}")
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
            if self.active_algorithm == "bottom":
                result = run_bottom_detection(path)
                self._show_bottom_result(path, result, perf_counter() - started_at)
                return
            result_image, result = run_side_detection(path)
        except ValueError as error:
            if self.active_algorithm == "bottom":
                self._clear_bottom_result(str(error))
            else:
                self._clear_result("검출에 실패했습니다.")
            QMessageBox.critical(self, "검출", str(error))
            return

        elapsed_seconds = perf_counter() - started_at
        images_per_second = 1 / max(elapsed_seconds, 0.000001)
        self.histogram_image_width = result_image.shape[1]
        self.histogram_center_split_x = result.center_split_x or self.histogram_image_width // 2
        self.raw_original_pixmap = self._pixmap_from_image(to_bgr(result.raw_image_a))
        self.raw_result_pixmap = self._pixmap_from_image(to_bgr(result.raw_image_b))
        self.annotated_original_pixmap = self._pixmap_from_image(
            result.source_visualization
        )
        self.annotated_result_pixmap = self._pixmap_from_image(result_image)
        self._refresh_image_comparison()
        self._show_analysis_preview(result.analysis_preview)
        self._show_source_preview(result.source_preview)
        self._last_side_result = result
        self._refresh_ball_preview()
        self._show_preview_timings(result)
        detection_state = "detected" if result.is_detected else "not_detected"
        self._set_detection_state(detection_state)
        self._update_info_label(path, result, elapsed_seconds, images_per_second)
        self._show_top_contour_histogram(result.measurements)

    def _show_bottom_result(self, path, result, elapsed_seconds):
        """A의 ROI와 실제 윤곽, 위치별 조건 검사 결과를 표시한다."""
        self.raw_original_pixmap = self._pixmap_from_image(to_bgr(result.raw_image_a))
        self.raw_result_pixmap = self._pixmap_from_image(to_bgr(result.raw_image_b))
        self.annotated_original_pixmap = self._pixmap_from_image(result.source_visualization)
        self.annotated_result_pixmap = self.raw_result_pixmap
        self._refresh_image_comparison()
        self._show_analysis_preview(result.circle_preview)
        self._show_source_preview(result.roi_preview)
        self.analysis_preview_label.timing_label.setText(
            f"원 후보 검사 · {elapsed_seconds * 1000:.1f} ms"
        )
        self._set_detection_state("detected" if result.all_regions_accepted else "not_detected")
        self.header_metadata_label.setText(
            f"Bottom 원 후보 검사 완료 · {elapsed_seconds * 1000:.1f} ms"
        )
        lines = [
            f"파일  {path.name}",
            f"A 영상 크기  {result.raw_image_a.shape[1]} × {result.raw_image_a.shape[0]} px",
            "검사 단계  칩 위치 → 6개 ROI → 검은 성분 → 원 후보 조건 검사 (A 영상)",
            "원 후보 조건 통과는 양품 판정이 아닙니다.",
            f"칩 분리 임계값  {result.threshold:.1f}",
            f"원 후보 기준  원형도 ≥ {MIN_CIRCULARITY:.2f}, 면적 {MIN_CIRCLE_AREA:g}–{MAX_CIRCLE_AREA:g} px², "
            f"예상 ROI 중심에서 ≤ {MAX_CENTER_DISTANCE:g} px, 짧은 변/긴 변 ≥ {MIN_CIRCLE_ASPECT_RATIO:.2f}",
            "검색 경계에 닿거나 적합한 성분이 여러 개이면 검토 대상으로 표시합니다.",
        ]
        chip = result.chip
        if chip is None:
            summary = "완전한 사각 칩 외곽을 찾지 못했습니다."
            self.detection_result_label.setText("칩 미검출")
            self.analysis_preview_label.setText(summary)
            self.source_preview_label.setText("칩 위치가 없어 원 후보를 검사하지 않았습니다.")
        else:
            x, y, width, height = chip.bounding_rect
            summary = (
                f"원 후보 {result.candidate_count}/6 · 검토 {result.review_count} · 없음 {result.missing_count}\n"
                f"중심 ({chip.center[0]:.1f}, {chip.center[1]:.1f}) px · "
                f"기울기 {chip.angle_degrees:.2f}°\n"
                f"초록: 원 후보 · 주황: 검토 · 빨강: 없음\n"
                f"원형도 ≥ {MIN_CIRCULARITY:.2f} · 면적 {MIN_CIRCLE_AREA:g}–{MAX_CIRCLE_AREA:g} px² · 거리 ≤ {MAX_CENTER_DISTANCE:g} px"
            )
            lines.append(f"크롭 영역  x={x}, y={y}, 너비={width}, 높이={height} px")
            for name, point in zip(("좌상", "우상", "우하", "좌하"), chip.corners):
                lines.append(f"{name} 꼭짓점  ({point[0]:.1f}, {point[1]:.1f}) px")
            review_names = []
            for region in result.circle_regions:
                status_text = {"candidate": "원 후보", "review": "검토", "missing": "없음"}[region.status]
                lines.extend(("", f"[{region.index}. {region.name}] {status_text}"))
                lines.append(f"예상 중심 ({region.expected_center[0]:.1f}, {region.expected_center[1]:.1f}) px")
                for index, component in enumerate(region.components, 1):
                    lines.append(
                        f"성분 {index}: 면적 {component.area:.1f} px² · 원형도 {component.circularity:.3f} · "
                        f"가로세로비 {component.aspect_ratio:.3f} · 중심 거리 {component.center_distance:.2f} px"
                    )
                    lines.extend(component.rejection_reasons)
                if region.review_reasons:
                    lines.extend(region.review_reasons)
                    review_names.append(f"{region.index} {region.name}")
            if review_names:
                summary += "\n확인 위치: " + ", ".join(review_names)
        self.bottom_chip_summary.setText(summary)
        lines.append(summary)
        self.program_log_lines = lines
        self.inspection_info_text = "\n".join(lines)

    def _clear_bottom_result(self, message):
        """읽기 실패 시 이전 칩의 분석선·크롭·좌표가 남지 않도록 지운다."""
        self._last_side_result = None
        self.ball_arc_summary.clear()
        self.ball_arc_summary.hide()
        for attribute in (
            "original_pixmap", "result_pixmap", "raw_original_pixmap", "raw_result_pixmap",
            "annotated_original_pixmap", "annotated_result_pixmap",
            "analysis_preview_pixmap", "source_preview_pixmap", "ball_crop_preview_pixmap",
        ):
            setattr(self, attribute, QPixmap())
        for label in (
            self.image_label, self.result_label,
            self.analysis_preview_label, self.source_preview_label,
        ):
            label.setPixmap(QPixmap())
            label.setText("이미지를 읽을 수 없습니다.")
        self.analysis_preview_label.timing_label.setText("원 후보 검사 · — ms")
        self.bottom_chip_summary.setText(message)
        self.program_log_lines = [message]
        self.inspection_info_text = message
        self.header_metadata_label.setText("Bottom 원 후보 검사 실패")
        self._set_detection_state("error")

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
            preview_scale=1.0 if self.active_algorithm == "bottom" else PREVIEW_SCALE,
        )

    def _refresh_ball_preview(self, _checked=False):
        """기존 크롭과 하단 원호를 같은 입력에서 전환한다. 재검출은 하지 않는다."""
        use_arc = self.active_algorithm == "side" and self.ball_arc_switch.isChecked()
        self.ball_crop_preview_label.title_label.setText(
            "볼 검출 · 하단 원호" if use_arc else "볼 검출 · 상세 크롭"
        )
        self.ball_arc_summary.clear()
        self.ball_arc_summary.hide()
        result = self._last_side_result
        if result is None:
            self._show_ball_crop_preview(None)
            return
        if not use_arc:
            self._show_ball_crop_preview(result.ball_square_crop_preview)
            return
        measure_ball_arcs(result)
        self._show_ball_crop_preview(result.ball_arc_preview)
        lines = [
            f"하늘색: 하단 접점 · 원: 추정 윤곽 · 추가 {result.ball_arc_ms:.1f} ms",
            "R 반경 / E 원호 오차 / S 대칭 오차 (원본 px) · 시험 기준",
        ]
        for index, arc in enumerate(result.ball_arc_measurements, 1):
            status = {"fitted": "피팅", "review": "검토", "missing": "측정 불가"}[arc.status]
            if arc.radius is not None:
                symmetry = f"{arc.symmetry_error:.2f}" if arc.symmetry_error is not None else "—"
                lines.append(
                    f"{index}번 {status} · R {arc.radius:.1f} / E {arc.rmse:.2f} / S {symmetry} px"
                )
            else:
                lines.append(f"{index}번 {status}")
            if arc.reasons:
                lines.append("  " + " · ".join(arc.reasons))
        if not result.ball_arc_measurements:
            lines.append("기존 볼 위치 검출 결과가 없어 분석할 크롭이 없습니다.")
        self.ball_arc_summary.setText("\n".join(lines))
        self.ball_arc_summary.show()

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

    def _show_preview_timings(self, result):
        self.analysis_preview_label.timing_label.setText(
            f"기둥 기준 + 볼 검출 · {result.pillar_with_ball_ms:.1f} ms"
        )
        self.source_preview_label.timing_label.setText(
            f"최상단/빈도 + 볼 검출 · {result.frequency_with_ball_ms:.1f} ms"
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
        self.inspection_info_text = "\n".join(
            self.program_log_lines + self.top_contour_histogram.log_lines
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
        self._last_side_result = None
        self.ball_arc_summary.clear()
        self.ball_arc_summary.hide()
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
        self.header_metadata_label.setText(f"검사 실패 · {message}")
        self._set_detection_state("error")

    def _update_info_label(
        self, path, result=None, elapsed_seconds=None, images_per_second=None
    ):
        algorithm_label = ALGORITHM_OPTIONS[self.active_algorithm]["label"]
        lines = [
            f"이미지 크기  {self.original_pixmap.width()} × {self.original_pixmap.height()} px",
        ]
        if result is not None:
            lines.extend(("", "[표면]"))
            for index, measurement in enumerate(result.measurements, start=1):
                lines.extend(self._first_contact_log_lines(measurement, index))
            lines.extend(result.density_log_lines)

            lines.extend(("", "[볼]"))
            if not result.selected_ball_bottommost_points:
                lines.append("검출된 볼 좌표 : 없음")
            else:
                lines.append(f"검출된 볼 개수 : {len(result.selected_ball_bottommost_points)}개")
                for index, point in enumerate(result.selected_ball_bottommost_points, start=1):
                    lines.append(f"볼 {index} 하단 좌표 : ({point[0]}, {point[1]})")
        self.program_log_lines = lines
        self.inspection_info_text = "\n".join(self.program_log_lines)

        if elapsed_seconds is not None and images_per_second is not None:
            self.header_metadata_label.setText(
                f"{algorithm_label} 자동 검사 완료 · {elapsed_seconds * 1000:.1f} ms · "
                f"{images_per_second:.0f} image/sec"
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
                preview_scale=1.0 if self.active_algorithm == "bottom" else PREVIEW_SCALE,
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
        self._position_floating_navigation()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_scaled_pixmaps()
        self._position_floating_navigation()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Left:
            self.show_prev()
        elif event.key() == Qt.Key_Right:
            self.show_next()
        else:
            super().keyPressEvent(event)
