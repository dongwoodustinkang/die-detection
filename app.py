import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPalette, QPixmap
from PyQt5.QtWidgets import QApplication

from styles import DARK_PALETTE
from ui import MainWindow


APP_NAME = "Defect Detector"
# Windows 작업 표시줄 묶음과 Linux .desktop 파일 이름에 함께 쓴다.
APP_ID = "keti.defect-detector"
APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "detector-icon.png"
# Windows 제목 표시줄·작업 표시줄과 Linux 패널이 요구하는 크기들
APP_ICON_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def load_app_icon():
    """작은 크기에서도 뭉개지지 않도록 크기별로 미리 줄인 아이콘을 만든다."""
    source = QPixmap(str(APP_ICON_PATH))
    icon = QIcon()
    for size in APP_ICON_SIZES:
        icon.addPixmap(source.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    return icon


def set_windows_app_id():
    """python.exe 대신 이 앱 아이콘으로 작업 표시줄에 묶이게 한다."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except (AttributeError, OSError):
        pass


def apply_dark_palette(app):
    """스타일 규칙이 없는 위젯도 OS 설정과 관계없이 어둡게 그려지게 한다."""
    # macOS 기본 스타일은 팔레트를 일부만 따르므로 팔레트를 그대로 쓰는 Fusion으로 바꾼다.
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in DARK_PALETTE.items():
        palette.setColor(getattr(QPalette, role), QColor(color))
    for role in ("WindowText", "Text", "ButtonText"):
        palette.setColor(QPalette.Disabled, getattr(QPalette, role), QColor("#636366"))
    app.setPalette(palette)


def install_linux_desktop_entry():
    """GNOME(Wayland) 등 .desktop 파일로 아이콘을 찾는 Linux 패널에 앱을 등록한다."""
    data_home = Path.home() / ".local" / "share"
    icon_path = data_home / "icons" / "hicolor" / "256x256" / "apps" / f"{APP_ID}.png"
    desktop_path = data_home / "applications" / f"{APP_ID}.desktop"
    icon_path.parent.mkdir(parents=True, exist_ok=True)
    desktop_path.parent.mkdir(parents=True, exist_ok=True)
    icon_path.write_bytes(APP_ICON_PATH.read_bytes())
    desktop_path.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        f'Exec="{sys.executable}" "{Path(__file__).resolve()}"\n'
        f"Path={Path(__file__).resolve().parent}\n"
        f"Icon={APP_ID}\n"
        f"StartupWMClass={APP_NAME}\n"
        "Terminal=false\n"
        "Categories=Graphics;Science;\n",
        encoding="utf-8",
    )
    print(f"설치 완료: {desktop_path}")


def main():
    if "--install-desktop-entry" in sys.argv:
        install_linux_desktop_entry()
        return

    set_windows_app_id()
    # 레티나 화면에서 버튼 아이콘의 2배 크기 이미지를 사용하게 한다.
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    # Linux에서 창 클래스(WM_CLASS)와 Wayland app_id를 .desktop 파일과 맞춘다.
    app.setApplicationName(APP_NAME)
    app.setDesktopFileName(APP_ID)
    app.setWindowIcon(load_app_icon())
    theme = "dark" if "--dark" in sys.argv else "light"
    if theme == "dark":
        apply_dark_palette(app)
    window = MainWindow(theme=theme)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
