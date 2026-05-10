from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QFontMetrics, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QPolygonF

from common import ensure_parent


_QT_APPLICATION: QGuiApplication | None = None
_BASE_CANVAS_WIDTH = 1120
_BASE_CANVAS_HEIGHT = 768
_OUTPUT_SCALE = 2


def _ensure_qt_application() -> None:
    global _QT_APPLICATION
    application = QGuiApplication.instance()
    if application is None:
        _QT_APPLICATION = QGuiApplication(["o1-pipeline-render"])
    else:
        _QT_APPLICATION = application


def _load_diagram_font(size: int, bold: bool = False) -> QFont:
    """加载流程图字体，供 PyQt5 QPainter 离屏绘制使用。"""

    font = QFont("Arial", size)
    font.setBold(bold)
    return font


def _draw_centered_text(
    painter: QPainter,
    box: tuple[int, int, int, int],
    text: str,
    font: QFont,
    fill: tuple[int, int, int] = (28, 28, 32),
    line_spacing: int = 4,
) -> None:
    lines = text.split("\n")
    metrics = QFontMetrics(font)
    line_widths = [metrics.horizontalAdvance(line) for line in lines]
    line_height = metrics.height()
    total_height = len(lines) * line_height + max(0, len(lines) - 1) * line_spacing
    y_position = box[1] + max(0.0, (box[3] - box[1] - total_height) / 2.0) + metrics.ascent()

    painter.setFont(font)
    painter.setPen(QPen(QColor(*fill)))
    for line, line_width in zip(lines, line_widths):
        x_position = box[0] + max(0.0, (box[2] - box[0] - line_width) / 2.0)
        painter.drawText(QPointF(x_position, y_position), line)
        y_position += line_height + line_spacing


def _draw_centered_single_line_text(
    painter: QPainter,
    box: tuple[int, int, int, int],
    text: str,
    font: QFont,
    fill: tuple[int, int, int],
) -> None:
    metrics = QFontMetrics(font)
    text_width = metrics.horizontalAdvance(text)
    x_position = box[0] + max(0.0, (box[2] - box[0] - text_width) / 2.0)
    y_position = box[1] + max(0.0, (box[3] - box[1] - metrics.height()) / 2.0) + metrics.ascent()
    painter.setFont(font)
    painter.setPen(QPen(QColor(*fill)))
    painter.drawText(QPointF(x_position, y_position), text)


def _draw_dashed_rect(
    painter: QPainter,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    dash_length: int = 4,
    gap_length: int = 4,
    radius: int = 14,
) -> None:
    left, top, right, bottom = box
    pen = QPen(QColor(*outline), 1)
    pen.setDashPattern([float(dash_length), float(gap_length)])
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QBrush(QColor(*fill)))
    painter.drawRoundedRect(QRectF(left, top, right - left, bottom - top), radius, radius)


def _draw_rounded_rect(
    painter: QPainter,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    radius: int = 6,
    width: int = 2,
) -> None:
    left, top, right, bottom = box
    painter.setPen(QPen(QColor(*outline), width))
    painter.setBrush(QBrush(QColor(*fill)))
    painter.drawRoundedRect(QRectF(left, top, right - left, bottom - top), radius, radius)


def _draw_box(
    painter: QPainter,
    box: tuple[int, int, int, int],
    text: str,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    font: QFont,
    radius: int = 6,
    width: int = 2,
) -> None:
    _draw_rounded_rect(painter, box, fill, outline, radius=radius, width=width)
    _draw_centered_text(painter, (box[0] + 10, box[1] + 8, box[2] - 10, box[3] - 8), text, font)


def _draw_centered_two_line_text(
    painter: QPainter,
    box: tuple[int, int, int, int],
    first_line: str,
    second_line: str,
    first_font: QFont,
    second_font: QFont,
    fill: tuple[int, int, int] = (28, 28, 32),
    line_spacing: int = 4,
) -> None:
    first_metrics = QFontMetrics(first_font)
    second_metrics = QFontMetrics(second_font)
    first_width = first_metrics.horizontalAdvance(first_line)
    second_width = second_metrics.horizontalAdvance(second_line)
    first_height = first_metrics.height()
    second_height = second_metrics.height()
    total_height = first_height + line_spacing + second_height
    y_position = box[1] + max(0.0, (box[3] - box[1] - total_height) / 2.0)

    painter.setPen(QPen(QColor(*fill)))
    painter.setFont(first_font)
    painter.drawText(
        QPointF(box[0] + max(0.0, (box[2] - box[0] - first_width) / 2.0), y_position + first_metrics.ascent()),
        first_line,
    )
    painter.setFont(second_font)
    painter.drawText(
        QPointF(
            box[0] + max(0.0, (box[2] - box[0] - second_width) / 2.0),
            y_position + first_height + line_spacing + second_metrics.ascent(),
        ),
        second_line,
    )


def _box_center_x(box: tuple[int, int, int, int]) -> int:
    return int(round((box[0] + box[2]) / 2.0))


def _box_center_y(box: tuple[int, int, int, int]) -> int:
    return int(round((box[1] + box[3]) / 2.0))


def _box_left_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return box[0], _box_center_y(box)


def _box_right_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return box[2], _box_center_y(box)


def _box_top_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return _box_center_x(box), box[1]


def _box_bottom_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return _box_center_x(box), box[3]


def _offset_point(start: tuple[int, int], end: tuple[int, int], distance: float) -> QPointF:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = math.hypot(delta_x, delta_y)
    if length <= 1e-6:
        return QPointF(float(start[0]), float(start[1]))
    ratio = distance / length
    return QPointF(start[0] + delta_x * ratio, start[1] + delta_y * ratio)


def _draw_arrow(
    painter: QPainter,
    points: list[tuple[int, int]],
    fill: tuple[int, int, int] = (24, 24, 24),
    width: int = 2,
    arrow_size: int = 9,
    corner_radius: int = 12,
) -> None:
    if len(points) < 2:
        return

    pen = QPen(QColor(*fill), width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    path = QPainterPath()
    path.moveTo(QPointF(*points[0]))

    for index in range(1, len(points) - 1):
        previous_point = points[index - 1]
        current_point = points[index]
        next_point = points[index + 1]

        incoming_length = math.hypot(current_point[0] - previous_point[0], current_point[1] - previous_point[1])
        outgoing_length = math.hypot(next_point[0] - current_point[0], next_point[1] - current_point[1])
        if incoming_length <= 1e-6 or outgoing_length <= 1e-6:
            path.lineTo(QPointF(*current_point))
            continue

        incoming_unit = (
            (current_point[0] - previous_point[0]) / incoming_length,
            (current_point[1] - previous_point[1]) / incoming_length,
        )
        outgoing_unit = (
            (next_point[0] - current_point[0]) / outgoing_length,
            (next_point[1] - current_point[1]) / outgoing_length,
        )
        alignment = incoming_unit[0] * outgoing_unit[0] + incoming_unit[1] * outgoing_unit[1]
        if abs(alignment) >= 0.999:
            path.lineTo(QPointF(*current_point))
            continue

        active_radius = min(float(corner_radius), incoming_length / 2.0, outgoing_length / 2.0)
        corner_entry = _offset_point(current_point, previous_point, active_radius)
        corner_exit = _offset_point(current_point, next_point, active_radius)
        path.lineTo(corner_entry)
        path.quadTo(QPointF(*current_point), corner_exit)

    path.lineTo(QPointF(*points[-1]))
    painter.drawPath(path)

    end_x, end_y = points[-1]
    start_x, start_y = points[-2]
    angle = math.atan2(end_y - start_y, end_x - start_x)
    left_angle = angle + math.pi * 0.78
    right_angle = angle - math.pi * 0.78
    arrowhead = QPolygonF(
        [
            QPointF(end_x, end_y),
            QPointF(end_x + math.cos(left_angle) * arrow_size, end_y + math.sin(left_angle) * arrow_size),
            QPointF(end_x + math.cos(right_angle) * arrow_size, end_y + math.sin(right_angle) * arrow_size),
        ]
    )
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(*fill)))
    painter.drawPolygon(arrowhead)


def create_o1_pipeline_image(path: Path) -> None:
    """使用 PyQt5 生成 O1a 目录使用的三栏流程图。"""

    _ensure_qt_application()
    canvas = QImage(_BASE_CANVAS_WIDTH * _OUTPUT_SCALE, _BASE_CANVAS_HEIGHT * _OUTPUT_SCALE, QImage.Format_RGB32)
    canvas.fill(QColor("white"))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    if hasattr(QPainter, "HighQualityAntialiasing"):
        painter.setRenderHint(QPainter.HighQualityAntialiasing, True)
    painter.scale(_OUTPUT_SCALE, _OUTPUT_SCALE)
    title_font = _load_diagram_font(19)
    box_font = _load_diagram_font(12)
    small_font = _load_diagram_font(11)
    bold_font = _load_diagram_font(12, bold=True)
    forward_bold_font = _load_diagram_font(10, bold=True)

    blue_title = (52, 101, 211)
    purple_title = (128, 92, 207)
    gold_title = (207, 151, 33)
    left_outline = (79, 113, 205)
    middle_outline = (137, 96, 219)
    right_outline = (218, 165, 49)
    left_box = (218, 228, 249)
    middle_box = (216, 201, 252)
    right_box = (255, 239, 188)

    left_group_box = (38, 62, 437, 729)
    middle_group_box = (464, 62, 754, 729)
    right_group_box = (782, 62, 1077, 729)

    middlebury_box = (74, 156, 183, 227)
    load_im1_box = (257, 84, 365, 134)
    load_im0_box = (257, 166, 365, 217)
    load_pfm_box = (257, 248, 365, 299)
    clean_box = (228, 324, 396, 391)
    smooth_box = (210, 419, 412, 474)
    synthetic_disp_box = (222, 654, 401, 703)

    forward_box = (490, 158, 732, 226)
    forward_text_box = (508, 170, 714, 214)
    mask_box = (495, 251, 727, 525)
    diffusion_box = (518, 298, 703, 347)
    interpolation_box = (510, 376, 711, 427)
    blend_box = (520, 453, 701, 504)
    synthetic_im1_box = (508, 564, 713, 617)

    ssim_box = (854, 147, 1006, 211)
    ssim_csv_box = (872, 280, 987, 331)
    results_box = (868, 652, 991, 705)

    _draw_centered_single_line_text(painter, (38, 20, 437, 52), "Data Input & Preprocessing", title_font, blue_title)
    _draw_centered_single_line_text(painter, (464, 20, 754, 52), "Core Workflow", title_font, purple_title)
    _draw_centered_single_line_text(painter, (782, 20, 1077, 52), "Evaluation & Saved Artifacts", title_font, gold_title)

    _draw_dashed_rect(painter, left_group_box, (238, 242, 250), left_outline)
    _draw_dashed_rect(painter, middle_group_box, (238, 228, 252), middle_outline)
    _draw_dashed_rect(painter, right_group_box, (255, 245, 211), right_outline)

    _draw_box(painter, middlebury_box, "Middlebury\nScene", left_box, (79, 99, 176), box_font)
    _draw_box(painter, load_im1_box, "load im1", left_box, (79, 99, 176), box_font)
    _draw_box(painter, load_im0_box, "Load im0", left_box, (79, 99, 176), box_font)
    _draw_box(painter, load_pfm_box, "Load pfm", left_box, (79, 99, 176), box_font)
    _draw_box(painter, clean_box, "Clean disparity:\nfinite + non-negative", left_box, (79, 99, 176), small_font)
    _draw_box(painter, smooth_box, "Light Gaussian smoothing", left_box, (79, 99, 176), small_font)
    _draw_box(painter, synthetic_disp_box, "Synthetic disp0.pfm", left_box, (79, 99, 176), small_font)

    _draw_rounded_rect(painter, forward_box, middle_box, middle_outline)
    _draw_centered_two_line_text(
        painter,
        forward_text_box,
        "Forward project left RGB",
        "according to disparity",
        forward_bold_font,
        small_font,
    )
    _draw_rounded_rect(painter, mask_box, (221, 209, 251), middle_outline)
    _draw_centered_text(painter, (510, 266, 712, 290), "Occlusion / hole mask", box_font)
    _draw_box(painter, diffusion_box, "4-neighbour diffusion", middle_box, middle_outline, small_font)
    _draw_box(painter, interpolation_box, "Linear grid interpolation", middle_box, middle_outline, small_font)
    _draw_box(painter, blend_box, "Final diffusion blend", middle_box, middle_outline, small_font)
    _draw_box(painter, synthetic_im1_box, "Synthetic im1.png", middle_box, middle_outline, small_font)

    _draw_box(painter, ssim_box, "SSIM against real\nright view", right_box, right_outline, small_font)
    _draw_box(painter, ssim_csv_box, "SSIM.csv", right_box, right_outline, small_font)
    _draw_box(painter, results_box, "Results Folder", right_box, right_outline, small_font)

    left_branch_x = 219
    ssim_vertical_x = _box_center_x(ssim_box)
    synth_branch_x = 823
    left_pipeline_x = _box_center_x(smooth_box)

    _draw_arrow(
        painter,
        [
            _box_right_center(middlebury_box),
            (left_branch_x, _box_center_y(middlebury_box)),
            (left_branch_x, _box_center_y(load_im1_box)),
            _box_left_center(load_im1_box),
        ],
    )
    _draw_arrow(painter, [_box_right_center(middlebury_box), _box_left_center(load_im0_box)])
    _draw_arrow(
        painter,
        [
            _box_right_center(middlebury_box),
            (left_branch_x, _box_center_y(middlebury_box)),
            (left_branch_x, _box_center_y(load_pfm_box)),
            _box_left_center(load_pfm_box),
        ],
    )
    _draw_arrow(
        painter,
        [
            _box_right_center(load_im1_box),
            (ssim_vertical_x, _box_center_y(load_im1_box)),
            _box_top_center(ssim_box),
        ],
    )
    _draw_arrow(painter, [_box_right_center(load_im0_box), _box_left_center(forward_box)])
    _draw_arrow(painter, [_box_bottom_center(load_pfm_box), _box_top_center(clean_box)])
    _draw_arrow(painter, [_box_bottom_center(clean_box), _box_top_center(smooth_box)])
    _draw_arrow(painter, [_box_bottom_center(smooth_box), _box_top_center(synthetic_disp_box)])
    _draw_arrow(
        painter,
        [
            _box_bottom_center(smooth_box),
            (left_pipeline_x, _box_center_y(synthetic_im1_box)),
            _box_left_center(synthetic_im1_box),
        ],
    )
    _draw_arrow(painter, [_box_bottom_center(forward_box), _box_top_center(mask_box)])
    _draw_arrow(painter, [_box_bottom_center(diffusion_box), _box_top_center(interpolation_box)])
    _draw_arrow(painter, [_box_bottom_center(interpolation_box), _box_top_center(blend_box)])
    _draw_arrow(painter, [_box_bottom_center(blend_box), _box_top_center(synthetic_im1_box)])
    _draw_arrow(
        painter,
        [
            _box_right_center(synthetic_im1_box),
            (synth_branch_x, _box_center_y(synthetic_im1_box)),
            (synth_branch_x, _box_center_y(ssim_box)),
            _box_left_center(ssim_box),
        ],
    )
    _draw_arrow(
        painter,
        [
            _box_right_center(synthetic_im1_box),
            (_box_center_x(results_box), _box_center_y(synthetic_im1_box)),
            _box_top_center(results_box),
        ],
    )
    _draw_arrow(painter, [_box_bottom_center(ssim_box), _box_top_center(ssim_csv_box)])
    _draw_arrow(painter, [_box_bottom_center(ssim_csv_box), _box_top_center(results_box)])
    _draw_arrow(painter, [_box_right_center(synthetic_disp_box), _box_left_center(results_box)])
    painter.end()

    ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        saved = canvas.save(str(path), "JPEG", 100)
    else:
        saved = canvas.save(str(path), "PNG")
    if not saved:
        raise OSError(f"Failed to save O1 pipeline image: {path}")


def write_o1_pipeline_assets(pipeline_dir: Path) -> None:
    """写出 PDF 要求的 O1a JPG 流程图资产。"""

    create_o1_pipeline_image(pipeline_dir / "syn_pipeline.jpg")