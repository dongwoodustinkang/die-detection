"""도메인과 UI에 독립적인 파일·이미지·메모 공통 기능."""

import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


NOTE_CSV_FIELDS = ("path", "filename", "note", "created_at", "modified_at")


def to_grayscale(image):
    """이미지를 그레이스케일로 변환한다."""
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr(image):
    """표시용 BGR 3채널 이미지로 변환한다."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def to_bgra(image):
    """투명 미리보기용 BGRA 4채널 이미지로 변환한다."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image.copy()


def load_ab_tiff_pages(image_path):
    """A/B 다중 페이지 TIFF의 앞 두 페이지를 읽는다."""
    try:
        with Image.open(image_path) as tiff:
            if getattr(tiff, "n_frames", 1) < 2:
                raise ValueError("A/B 두 페이지 TIFF가 아닙니다.")

            pages = []
            for page_index in (0, 1):
                tiff.seek(page_index)
                page = np.array(tiff.copy())
                if page.ndim not in (2, 3):
                    raise ValueError(
                        f"{page_index + 1}번째 페이지의 차원이 올바르지 않습니다: "
                        f"{page.ndim}D"
                    )
                pages.append(page)
            return pages
    except (OSError, ValueError) as error:
        raise ValueError(
            f"A/B 두 페이지 TIFF를 읽을 수 없습니다: {image_path}"
        ) from error


def next_capture_path(capture_dir, source_path):
    """원본 파일명에 이어지는 다음 캡처 JPG 경로를 반환한다."""
    filename = source_path.stem
    prefix = f"{filename}_"
    sequence_numbers = [
        int(path.stem[len(prefix):])
        for path in capture_dir.glob(f"{filename}_*.jpg")
        if path.stem.startswith(prefix) and path.stem[len(prefix):].isdecimal()
    ]
    return capture_dir / f"{filename}_{max(sequence_numbers, default=0) + 1}.jpg"


class NotesRepository:
    """CSV에 파일별 메모를 조회하고 갱신한다."""

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)

    def find(self, source_path):
        source_path = Path(source_path)
        path_text = str(source_path.parent)
        for row in reversed(self._read_all()):
            if (
                row.get("filename") == source_path.name
                and row.get("path") in {path_text, ""}
            ):
                return row
        return None

    def upsert(self, source_path, note):
        source_path = Path(source_path)
        path_text = str(source_path.parent)
        rows = self._read_all()
        matching_indexes = [
            index
            for index, row in enumerate(rows)
            if row.get("filename") == source_path.name
            and row.get("path") in {path_text, ""}
        ]
        existing = rows[matching_indexes[-1]] if matching_indexes else None
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "path": path_text,
            "filename": source_path.name,
            "note": note,
            "created_at": (existing or {}).get("created_at") or timestamp,
            "modified_at": timestamp,
        }
        rows = [
            row
            for index, row in enumerate(rows)
            if index not in matching_indexes
        ]
        rows.append(record)
        self._write_all(rows)
        return record

    def _read_all(self):
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return []
        with self.csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
            return list(csv.DictReader(csv_file))

    def _write_all(self, rows):
        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=NOTE_CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: row.get(field, "") for field in NOTE_CSV_FIELDS}
                )
