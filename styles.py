"""Apple-inspired translucent visual system for the detector workspace."""

APP_STYLESHEET = """
    /* System materials: a cool, layered background with translucent surfaces. */
    QMainWindow, #centralWidget {
        background: qradialgradient(cx: 0.12, cy: 0.03, radius: 1.25,
            fx: 0.12, fy: 0.03, stop: 0 #EAF4FF, stop: 0.42 #F5F7FB,
            stop: 1 #E8EDF5);
        color: #1D1D1F;
        font-family: "SF Pro Display", "SF Pro Text", "Apple SD Gothic Neo", "Pretendard", sans-serif;
        font-size: 13px;
    }
    QLabel#windowTitle { color: #1D1D1F; font-size: 30px; font-weight: 700; letter-spacing: -1.1px; }
    QToolButton#algorithmButton {
        color: #1D1D1F; background: transparent; border: 0; border-radius: 10px;
        font-size: 30px; font-weight: 700; letter-spacing: -1.1px; padding: 2px 22px 2px 0;
    }
    QToolButton#algorithmButton:hover { background: rgba(10, 132, 255, 0.10); color: #007AFF; }
    QToolButton#algorithmButton::menu-indicator { subcontrol-position: right center; right: 3px; }
    QMenu {
        background: rgba(255, 255, 255, 245); border: 1px solid rgba(60, 60, 67, 0.12);
        border-radius: 12px; padding: 6px; color: #1D1D1F;
    }
    QMenu::item { border-radius: 7px; padding: 8px 28px 8px 12px; font-size: 13px; font-weight: 600; }
    QMenu::item:selected { background: rgba(10, 132, 255, 0.14); color: #007AFF; }
    QToolTip {
        background: rgba(28, 28, 30, 238); border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 8px; color: #FFFFFF; font-size: 12px; font-weight: 600; padding: 5px 8px;
    }
    QLabel#headerMetadata { color: #6E6E73; font-size: 13px; font-weight: 400; }
    QFrame#fileContext, QFrame#navigationGroup {
        background: rgba(255, 255, 255, 0.52);
        border: 1px solid rgba(255, 255, 255, 0.78);
        border-radius: 14px;
    }
    QFrame#headerActionGroup {
        background: rgba(255, 255, 255, 0.52);
        border: 1px solid rgba(255, 255, 255, 0.82);
        border-radius: 18px;
    }
    QFrame#headerActionGroup QFrame#fileContext {
        background: transparent; border: 0; border-radius: 0;
    }
    QFrame#detectionResultBadge {
        background: rgba(118, 118, 128, 0.12);
        border: 1px solid rgba(60, 60, 67, 0.10); border-radius: 16px;
    }
    QLabel#detectionResultLabel {
        color: #6E6E73; font-size: 17px; font-weight: 700; min-width: 52px;
        qproperty-alignment: AlignCenter;
    }
    QFrame#detectionResultBadge[state="detected"] {
        background: rgba(52, 199, 89, 0.14); border-color: rgba(52, 199, 89, 0.28);
    }
    QFrame#detectionResultBadge[state="detected"] QLabel#detectionResultLabel { color: #248A3D; }
    QFrame#detectionResultBadge[state="not_detected"] {
        background: rgba(255, 159, 10, 0.14); border-color: rgba(255, 159, 10, 0.28);
    }
    QFrame#detectionResultBadge[state="not_detected"] QLabel#detectionResultLabel { color: #C93400; }
    QFrame#detectionResultBadge[state="error"] {
        background: rgba(255, 69, 58, 0.13); border-color: rgba(255, 69, 58, 0.25);
    }
    QFrame#detectionResultBadge[state="error"] QLabel#detectionResultLabel { color: #D70015; }
    QLabel#fileContextIcon { color: #007AFF; font-size: 13px; font-weight: 700; }
    QLabel#fileContextLabel { color: #48484A; font-size: 12px; font-weight: 500; }

    /* Main cards use a light material, highlighted upper edge and soft shadow. */
    QFrame#imageCard, QFrame#analysisCard, QFrame#logCard, QFrame#controlCard {
        background: rgba(255, 255, 255, 205);
        border: 1px solid rgba(255, 255, 255, 225);
        border-radius: 22px;
    }
    QFrame#imageCard, QFrame#analysisCard, QFrame#controlCard { border-color: rgba(255, 255, 255, 235); }
    QLabel#cardTitle, QLabel#sectionTitle { color: #1D1D1F; font-size: 15px; font-weight: 700; letter-spacing: -0.25px; }
    QPushButton#infoButton {
        background: rgba(10, 132, 255, 0.10); border: 0; border-radius: 12px; color: #007AFF;
        font-size: 14px; font-weight: 700; min-width: 24px; min-height: 24px;
        max-width: 24px; max-height: 24px; padding: 0;
    }
    QPushButton#infoButton:hover { background: rgba(10, 132, 255, 0.20); }
    QPushButton#imageOverlaySwitch {
        background: rgba(118, 118, 128, 0.12); border: 0; border-radius: 11px; color: #6E6E73;
        font-size: 11px; font-weight: 700; min-width: 46px; min-height: 24px;
        max-height: 24px; padding: 0 8px;
    }
    QPushButton#imageOverlaySwitch:checked { background: rgba(10, 132, 255, 0.15); color: #007AFF; }
    QPushButton#imageOverlaySwitch:hover { background: rgba(10, 132, 255, 0.20); color: #007AFF; }
    QLabel#imagePaneTitle, QLabel#previewTitle { color: #48484A; font-size: 12px; font-weight: 600; padding-left: 2px; }
    QLabel#previewTiming {
        color: #007AFF; font-size: 10px; font-weight: 700;
        background: rgba(10, 132, 255, 0.10); border-radius: 7px; padding: 4px 6px;
    }
    QLabel#histogramCount { color: #6E6E73; font-size: 11px; font-weight: 500; background: transparent; padding: 0; }
    QFrame#divider { color: rgba(60, 60, 67, 0.14); max-height: 1px; }

    /* Image wells are a slightly darker material that keeps high-contrast data legible. */
    QLabel#imagePreview {
        background: rgba(245, 247, 250, 180);
        border: 1px solid rgba(60, 60, 67, 0.12);
        border-radius: 14px; color: #8E8E93; padding: 14px;
    }
    QLabel#imagePreview:hover { background: rgba(255, 255, 255, 220); border-color: rgba(10, 132, 255, 0.40); }
    QPlainTextEdit#infoLabel { color: #48484A; font-size: 12px; line-height: 1.45; background: transparent; border: 0; padding: 0; }
    QPlainTextEdit#infoLabel QScrollBar:vertical, QScrollArea#analysisScroll QScrollBar:vertical { width: 7px; background: transparent; margin: 4px 0; }
    QPlainTextEdit#infoLabel QScrollBar::handle:vertical, QScrollArea#analysisScroll QScrollBar::handle:vertical { background: rgba(60, 60, 67, 0.24); border-radius: 3px; min-height: 24px; }
    QPlainTextEdit#infoLabel QScrollBar::add-line:vertical, QPlainTextEdit#infoLabel QScrollBar::sub-line:vertical, QScrollArea#analysisScroll QScrollBar::add-line:vertical, QScrollArea#analysisScroll QScrollBar::sub-line:vertical { height: 0; }

    /* Compact controls emulate macOS segmented controls and capsules. */
    QPushButton#histogramSideButton { background: rgba(118, 118, 128, 0.12); border: 0; border-radius: 8px; color: #6E6E73; font-size: 11px; font-weight: 700; min-height: 27px; min-width: 27px; padding: 0; }
    QPushButton#histogramSideButton:hover { background: rgba(10, 132, 255, 0.14); color: #007AFF; }
    QPushButton#histogramSideButton:checked { background: #007AFF; color: #FFFFFF; }
    QFrame#histogramRegionSelector { background: rgba(118, 118, 128, 0.10); border: 1px solid rgba(60, 60, 67, 0.08); border-radius: 9px; }
    QLabel#histogramRegionTitle { color: #8E8E93; font-size: 9px; font-weight: 700; }
    QPushButton#histogramRegionButton { background: transparent; border: 0; border-radius: 6px; color: #6E6E73; font-size: 10px; font-weight: 600; min-height: 21px; min-width: 40px; padding: 0 4px; }
    QPushButton#histogramRegionButton:hover { color: #007AFF; }
    QPushButton#histogramRegionButton:checked { background: #FFFFFF; color: #007AFF; }
    QFrame#settingRow { background: rgba(118, 118, 128, 0.09); border: 1px solid rgba(60, 60, 67, 0.08); border-radius: 12px; }
    QLabel#controlLabel { color: #3A3A3C; font-size: 13px; font-weight: 600; }
    QLabel#settingValue { color: #3A3A3C; background: rgba(255, 255, 255, 190); border: 1px solid rgba(60, 60, 67, 0.10); border-radius: 7px; font-size: 12px; font-weight: 600; padding: 5px 8px; }
    QFrame#ballCountSegment { background: rgba(118, 118, 128, 0.16); border-radius: 8px; }
    QPushButton#ballCountSegmentButton { background: transparent; border: 0; border-radius: 6px; color: #6E6E73; font-size: 13px; font-weight: 700; min-height: 28px; min-width: 30px; padding: 0 5px; }
    QPushButton#ballCountSegmentButton:hover { color: #007AFF; }
    QPushButton#ballCountSegmentButton:checked { background: rgba(255, 255, 255, 235); color: #007AFF; }
    QPushButton#sideCuttingSwitch { background: rgba(120, 120, 128, 0.26); border: 0; border-radius: 14px; color: #FFFFFF; font-size: 10px; font-weight: 700; min-height: 28px; min-width: 48px; max-height: 28px; max-width: 48px; padding: 0; }
    QPushButton#sideCuttingSwitch:checked { background: #34C759; color: #FFFFFF; }
    QPushButton#sideCuttingSwitch:hover { background: rgba(120, 120, 128, 0.38); }
    QPushButton#sideCuttingSwitch:checked:hover { background: #28B446; }

    QFrame#statusCard { background: rgba(10, 132, 255, 0.09); border: 1px solid rgba(10, 132, 255, 0.18); border-radius: 16px; }
    QLabel#statusValue { color: #007AFF; font-size: 21px; font-weight: 700; letter-spacing: -0.5px; }
    QLabel#statusDetail { color: #6E6E73; font-size: 12px; }
    QLabel#statusMetric {
        color: #6E6E73; font-size: 11px; font-weight: 600;
        background: rgba(255, 255, 255, 0.52); border-radius: 7px; padding: 5px 7px;
    }

    /* Buttons: one strong action, otherwise quiet translucent controls. */
    QPushButton { border: 0; border-radius: 11px; font-size: 13px; font-weight: 600; min-height: 38px; padding: 0 16px; }
    QPushButton#primaryButton { background: #007AFF; color: #FFFFFF; }
    QPushButton#primaryButton:hover { background: #0071E3; }
    QPushButton#primaryButton:pressed { background: #0066CC; }
    QPushButton#primaryButton:disabled { background: rgba(120, 120, 128, 0.18); color: #AEAEB2; }
    QPushButton#secondaryButton { background: rgba(10, 132, 255, 0.12); color: #007AFF; }
    QPushButton#secondaryButton:hover { background: rgba(10, 132, 255, 0.20); }
    QPushButton#secondaryButton:pressed { background: rgba(10, 132, 255, 0.30); color: #0066CC; }
    QPushButton#navigationButton { background: rgba(255, 255, 255, 0.45); border: 1px solid rgba(60, 60, 67, 0.10); border-radius: 14px; color: #3A3A3C; font-size: 22px; font-weight: 400; min-height: 30px; min-width: 30px; padding: 0; }
    QPushButton#navigationButton:hover { background: rgba(255, 255, 255, 0.88); color: #007AFF; }
    QPushButton#navigationButton:disabled { background: rgba(255, 255, 255, 0.22); color: #C7C7CC; }
    QPushButton#captureIconButton { background: rgba(255, 255, 255, 0.45); border: 1px solid rgba(60, 60, 67, 0.10); border-radius: 11px; min-height: 32px; min-width: 32px; max-height: 32px; max-width: 32px; padding: 0; }
    QPushButton#captureIconButton:hover { background: rgba(10, 132, 255, 0.15); }
    QPushButton#captureIconButton:pressed { background: rgba(10, 132, 255, 0.25); }
    QFrame#floatingNavigation {
        background: rgba(255, 255, 255, 178);
        border: 1px solid rgba(255, 255, 255, 226);
        border-radius: 22px;
    }
    QPushButton#floatingNavigationButton {
        background: rgba(118, 118, 128, 0.11); border: 0; border-radius: 15px;
        color: #3A3A3C; font-size: 25px; font-weight: 400;
        min-width: 38px; min-height: 38px; max-width: 38px; max-height: 38px; padding: 0;
    }
    QPushButton#floatingNavigationButton:hover { background: rgba(10, 132, 255, 0.16); color: #007AFF; }
    QPushButton#floatingNavigationButton:disabled { color: #C7C7CC; background: rgba(118, 118, 128, 0.06); }
    QLabel#floatingIndexLabel {
        color: #6E6E73; font-size: 11px; font-weight: 700; min-width: 34px;
        qproperty-alignment: AlignCenter;
    }
    QFrame#floatingNavigation QPushButton#captureIconButton {
        min-width: 38px; min-height: 38px; max-width: 38px; max-height: 38px; border-radius: 15px;
        font-size: 17px;
    }

    QDialog#imageModal { background: transparent; border: 0; }
    QFrame#imageModalCard {
        background: rgba(248, 250, 253, 190);
        border: 1px solid rgba(255, 255, 255, 230); border-radius: 18px;
    }
    QDialog#inspectionInfoModal { background: transparent; border: 0; }
    QFrame#inspectionInfoModalCard {
        background: rgba(248, 250, 253, 205);
        border: 1px solid rgba(255, 255, 255, 230); border-radius: 18px;
    }
    QPlainTextEdit#inspectionInfoText {
        background: rgba(255, 255, 255, 0.48); border: 1px solid rgba(60, 60, 67, 0.10);
        border-radius: 12px; color: #3A3A3C; font-size: 12px; padding: 10px;
    }
    QDialog#noteModal { background: transparent; border: 0; }
    QFrame#noteModalCard {
        background: rgba(248, 250, 253, 210);
        border: 1px solid rgba(255, 255, 255, 230); border-radius: 18px;
    }
    QLabel#noteFileLabel { color: #6E6E73; font-size: 11px; font-weight: 600; }
    QPlainTextEdit#noteInput {
        background: rgba(255, 255, 255, 0.60); border: 1px solid rgba(60, 60, 67, 0.12);
        border-radius: 12px; color: #1D1D1F; font-size: 13px; padding: 10px;
    }
    QPushButton#noteSaveButton { background: #007AFF; color: #FFFFFF; }
    QPushButton#noteSaveButton:hover { background: #0071E3; }
    QPushButton#modalCloseButton {
        background: rgba(118, 118, 128, 0.12); border: 0; border-radius: 12px; color: #48484A;
        font-size: 20px; font-weight: 400; min-width: 24px; min-height: 24px;
        max-width: 24px; max-height: 24px; padding: 0;
    }
    QPushButton#modalCloseButton:hover { background: rgba(255, 69, 58, 0.14); color: #FF453A; }
    QLabel#imageModalTitle { color: #1D1D1F; font-size: 15px; font-weight: 700; }
    QLabel#imageModalPreview { background: rgba(245, 247, 250, 200); border: 1px solid rgba(60, 60, 67, 0.12); border-radius: 12px; }
    QLabel#imageModalHint, QLabel#zoomLabel { color: #8E8E93; font-size: 12px; min-width: 42px; qproperty-alignment: AlignCenter; }
    QScrollArea#imageModalScroll { background: rgba(245, 247, 250, 200); border: 1px solid rgba(60, 60, 67, 0.12); border-radius: 12px; }
    QScrollArea#imageModalScroll > QWidget > QWidget { background: transparent; }
    QScrollArea#analysisScroll { background: transparent; border: 0; }
    QScrollArea#analysisScroll > QWidget > QWidget { background: transparent; }
    QPushButton#zoomButton, QPushButton#zoomResetButton { background: rgba(118, 118, 128, 0.12); border: 0; border-radius: 8px; color: #3A3A3C; font-size: 12px; font-weight: 600; min-height: 28px; min-width: 30px; padding: 0 8px; }
    QPushButton#zoomButton:hover, QPushButton#zoomResetButton:hover { background: rgba(10, 132, 255, 0.16); color: #007AFF; }
    QCheckBox { color: #3A3A3C; font-size: 13px; spacing: 9px; }
    QCheckBox::indicator { width: 17px; height: 17px; border: 1px solid rgba(60, 60, 67, 0.26); border-radius: 5px; background: rgba(255, 255, 255, 220); }
    QCheckBox::indicator:hover { border-color: rgba(0, 122, 255, 0.60); }
    QCheckBox::indicator:checked { background: #007AFF; border-color: #007AFF; }
"""
