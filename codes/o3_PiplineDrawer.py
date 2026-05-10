from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QFontMetrics, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QPolygonF


LOGICAL_CANVAS_WIDTH = 2048
LOGICAL_CANVAS_HEIGHT = 1138
OUTPUT_SCALE = 3

_QT_APPLICATION: QGuiApplication | None = None


def _ensure_qt_application() -> None:
    global _QT_APPLICATION
    application = QGuiApplication.instance()
    if application is None:
        _QT_APPLICATION = QGuiApplication(["o3-pipeline-render"])
    else:
        _QT_APPLICATION = application


def _load_font(size: int, bold: bool = False) -> QFont:
    font = QFont("Arial", size)
    font.setBold(bold)
    return font


def _wrap_text(text: str, metrics: QFontMetrics, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.split("\n"):
        current_line = ""
        for word in raw_line.split(" "):
            candidate = word if not current_line else f"{current_line} {word}"
            if metrics.horizontalAdvance(candidate) <= max_width:
                current_line = candidate
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
    return lines


def _draw_centered_text(
    painter: QPainter,
    box: tuple[int, int, int, int],
    text: str,
    font: QFont,
    fill: tuple[int, int, int],
    line_spacing: int = 4,
) -> None:
    metrics = QFontMetrics(font)
    lines = _wrap_text(text, metrics, max(1, box[2] - box[0]))
    line_height = metrics.height()
    total_height = len(lines) * line_height + max(0, len(lines) - 1) * line_spacing
    y_position = box[1] + max(0.0, (box[3] - box[1] - total_height) / 2.0) + metrics.ascent()

    painter.setFont(font)
    painter.setPen(QPen(QColor(*fill)))
    for line in lines:
        line_width = metrics.horizontalAdvance(line)
        x_position = box[0] + max(0.0, (box[2] - box[0] - line_width) / 2.0)
        painter.drawText(QPointF(x_position, y_position), line)
        y_position += line_height + line_spacing


def _draw_group(
    painter: QPainter,
    box: tuple[int, int, int, int],
    title: str,
    title_box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    title_font: QFont,
    title_fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    pen = QPen(QColor(*outline), 2)
    pen.setDashPattern([3.0, 5.0])
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QBrush(QColor(*fill)))
    painter.drawRoundedRect(QRectF(left, top, right - left, bottom - top), 6, 6)
    _draw_centered_text(painter, title_box, title, title_font, title_fill)


def _draw_box(
    painter: QPainter,
    box: tuple[int, int, int, int],
    label: str,
    font: QFont,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    text_fill: tuple[int, int, int] = (38, 40, 48),
    radius: int = 6,
    width: int = 2,
) -> None:
    left, top, right, bottom = box
    painter.setPen(QPen(QColor(*outline), width))
    painter.setBrush(QBrush(QColor(*fill)))
    painter.drawRoundedRect(QRectF(left, top, right - left, bottom - top), radius, radius)
    _draw_centered_text(painter, (left + 14, top + 8, right - 14, bottom - 8), label, font, text_fill)


def _center_x(box: tuple[int, int, int, int]) -> int:
    return int(round((box[0] + box[2]) / 2.0))


def _center_y(box: tuple[int, int, int, int]) -> int:
    return int(round((box[1] + box[3]) / 2.0))


def _top_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return _center_x(box), box[1]


def _bottom_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return _center_x(box), box[3]


def _left_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return box[0], _center_y(box)


def _right_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return box[2], _center_y(box)


def _draw_arrow(
    painter: QPainter,
    points: list[tuple[int, int]],
    fill: tuple[int, int, int] = (20, 20, 22),
    width: int = 2,
    arrow_size: int = 10,
    corner_radius: int = 18,
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
        previous_x, previous_y = points[index - 1]
        current_x, current_y = points[index]
        next_x, next_y = points[index + 1]
        incoming_x = previous_x - current_x
        incoming_y = previous_y - current_y
        outgoing_x = next_x - current_x
        outgoing_y = next_y - current_y
        incoming_length = math.hypot(incoming_x, incoming_y)
        outgoing_length = math.hypot(outgoing_x, outgoing_y)
        if incoming_length <= 0.0 or outgoing_length <= 0.0:
            path.lineTo(QPointF(current_x, current_y))
            continue
        cross = incoming_x * outgoing_y - incoming_y * outgoing_x
        if abs(cross) < 1e-6:
            path.lineTo(QPointF(current_x, current_y))
            continue
        radius = min(float(corner_radius), incoming_length / 2.0, outgoing_length / 2.0)
        before_corner = QPointF(
            current_x + incoming_x / incoming_length * radius,
            current_y + incoming_y / incoming_length * radius,
        )
        after_corner = QPointF(
            current_x + outgoing_x / outgoing_length * radius,
            current_y + outgoing_y / outgoing_length * radius,
        )
        path.lineTo(before_corner)
        path.quadTo(QPointF(current_x, current_y), after_corner)
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


def create_o3_pipeline_image(path: Path) -> None:
    """Use PyQt5 to draw the O3a stereo disparity pipeline diagram."""

    _ensure_qt_application()
    canvas = QImage(LOGICAL_CANVAS_WIDTH * OUTPUT_SCALE, LOGICAL_CANVAS_HEIGHT * OUTPUT_SCALE, QImage.Format_RGB32)
    canvas.fill(QColor("white"))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    painter.scale(OUTPUT_SCALE, OUTPUT_SCALE)

    title_font = _load_font(22)
    node_font = _load_font(13)
    small_font = _load_font(12)

    input_style = {"fill": (245, 240, 255), "node": (236, 229, 252), "outline": (143, 112, 222), "title": (139, 108, 218)}
    sparse_style = {"fill": (247, 250, 255), "node": (241, 246, 255), "outline": (82, 124, 209), "title": (78, 121, 209)}
    calib_style = {"fill": (236, 251, 241), "node": (229, 248, 236), "outline": (76, 158, 94), "title": (76, 158, 94)}
    dense_style = {"fill": (255, 248, 225), "node": (255, 243, 203), "outline": (218, 174, 65), "title": (218, 174, 65)}
    post_style = {"fill": (255, 245, 245), "node": (255, 228, 228), "outline": (231, 89, 89), "title": (224, 78, 78)}
    output_style = {"fill": (250, 251, 253), "node": (248, 249, 251), "outline": (105, 113, 125), "title": (146, 151, 162)}

    groups = [
        ((25, 25, 347, 1080), "Input", (40, 68, 332, 112), input_style),
        ((385, 25, 806, 705), "Sparse Prior\n(Branch 1)", (410, 48, 781, 115), sparse_style),
        ((385, 743, 806, 1080), "Calibration & Bounds\n(Branch 2)", (410, 768, 781, 835), calib_style),
        ((842, 25, 1263, 1080), "Dense Matching", (872, 50, 1233, 94), dense_style),
        ((1301, 25, 1663, 1080), "Post-processing", (1326, 50, 1638, 94), post_style),
        ((1701, 25, 2024, 1080), "Output", (1730, 50, 1995, 94), output_style),
    ]
    for group_box, title, title_box, style in groups:
        _draw_group(painter, group_box, title, title_box, style["fill"], style["outline"], title_font, style["title"])

    stereo_pair = (70, 546, 298, 611)
    image_files = (70, 142, 298, 207)
    calib_input = (72, 859, 300, 924)

    sparse_left = 464
    sparse_right = 726
    load_gray = (sparse_left, 142, sparse_right, 207)
    sift = (sparse_left, 246, sparse_right, 312)
    ratio = (sparse_left, 350, sparse_right, 416)
    stereo_filter = (sparse_left, 455, sparse_right, 520)
    seed = (sparse_left, 560, sparse_right, 625)

    read_calib = (481, 859, 709, 924)
    bounds = (455, 964, 735, 1029)

    cost = (923, 639, 1184, 704)
    sgm = (923, 470, 1184, 536)
    solve = (923, 302, 1184, 367)

    lr = (1348, 180, 1611, 245)
    support = (1348, 328, 1611, 393)
    speckle = (1348, 477, 1611, 542)
    median = (1348, 626, 1611, 691)
    gap = (1348, 775, 1611, 840)
    final = (1348, 924, 1611, 989)

    pfm = (1780, 426, 1960, 491)
    png = (1780, 560, 1960, 625)
    metrics = (1780, 693, 1960, 758)

    _draw_box(painter, image_files, "im0.png / im1.png", node_font, input_style["node"], input_style["outline"])
    _draw_box(painter, stereo_pair, "Middlebury stereo pair", node_font, input_style["node"], input_style["outline"])
    _draw_box(painter, calib_input, "calib + disparity", node_font, input_style["node"], input_style["outline"])

    for box, label in [
        (load_gray, "Load as grayscale"),
        (sift, "Manual SIFT on left and right"),
        (ratio, "Descriptor KNN + ratio test"),
        (stereo_filter, "Mutual and stereo-geometry filtering"),
        (seed, "Sparse SIFT disparity seed prior"),
    ]:
        _draw_box(painter, box, label, node_font, sparse_style["node"], sparse_style["outline"])
    _draw_box(painter, read_calib, "Read calib + disparity hints", node_font, calib_style["node"], calib_style["outline"])
    _draw_box(painter, bounds, "Scene-adaptive disparity bounds", node_font, calib_style["node"], calib_style["outline"])

    for box, label in [
        (cost, "Census + gradient matching cost"),
        (sgm, "Four-direction SGM aggregation"),
        (solve, "Left / right disparity solve"),
    ]:
        _draw_box(painter, box, label, small_font, dense_style["node"], dense_style["outline"])
    for box, label in [
        (lr, "LR consistency error + margin check"),
        (support, "Local disparity support mask"),
        (speckle, "Speckle removal"),
        (median, "Median + joint weighted median"),
        (gap, "Short horizontal / vertical gap fill"),
        (final, "Final dense disparity"),
    ]:
        _draw_box(painter, box, label, small_font, post_style["node"], post_style["outline"])
    for box, label in [(pfm, "disp0.pfm"), (png, "disp0.png"), (metrics, "metrics.csv")]:
        _draw_box(painter, box, label, node_font, output_style["node"], output_style["outline"], radius=6)

    _draw_arrow(painter, [_top_center(stereo_pair), _bottom_center(image_files)])
    _draw_arrow(painter, [_bottom_center(stereo_pair), _top_center(calib_input)])
    _draw_arrow(painter, [_right_center(image_files), _left_center(load_gray)])
    _draw_arrow(painter, [_bottom_center(load_gray), _top_center(sift)])
    _draw_arrow(painter, [_bottom_center(sift), _top_center(ratio)])
    _draw_arrow(painter, [_bottom_center(ratio), _top_center(stereo_filter)])
    _draw_arrow(painter, [_bottom_center(stereo_filter), _top_center(seed)])
    _draw_arrow(painter, [_right_center(calib_input), _left_center(read_calib)])
    _draw_arrow(painter, [_bottom_center(read_calib), _top_center(bounds)])

    _draw_arrow(painter, [_right_center(seed), (878, _center_y(seed)), (878, _center_y(cost)), _left_center(cost)])
    _draw_arrow(painter, [_right_center(bounds), (878, _center_y(bounds)), (878, _center_y(cost)), _left_center(cost)])
    _draw_arrow(painter, [_right_center(image_files), (418, _center_y(image_files)), (418, _center_y(cost)), _left_center(cost)])
    _draw_arrow(painter, [_top_center(cost), _bottom_center(sgm)])
    _draw_arrow(painter, [_top_center(sgm), _bottom_center(solve)])
    _draw_arrow(painter, [_top_center(solve), (_center_x(solve), 212), _left_center(lr)])

    _draw_arrow(painter, [_bottom_center(lr), _top_center(support)])
    _draw_arrow(painter, [_bottom_center(support), _top_center(speckle)])
    _draw_arrow(painter, [_bottom_center(speckle), _top_center(median)])
    _draw_arrow(painter, [_bottom_center(median), _top_center(gap)])
    _draw_arrow(painter, [_bottom_center(gap), _top_center(final)])
    _draw_arrow(painter, [_right_center(final), (1738, _center_y(final)), (1738, 458), _left_center(pfm)])
    _draw_arrow(painter, [(1738, 594), _left_center(png)])
    _draw_arrow(painter, [(1738, 726), _left_center(metrics)])

    painter.end()

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        saved = canvas.save(str(path), "JPEG", 100)
    else:
        saved = canvas.save(str(path), "PNG")
    if not saved:
        raise OSError(f"Failed to save O3 pipeline image: {path}")
