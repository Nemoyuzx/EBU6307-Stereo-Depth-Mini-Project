from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QFontMetrics, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QPolygonF


LOGICAL_CANVAS_WIDTH = 2048
LOGICAL_CANVAS_HEIGHT = 1138
OUTPUT_SCALE = 2

_QT_APPLICATION: QGuiApplication | None = None


def _ensure_qt_application() -> None:
    global _QT_APPLICATION
    application = QGuiApplication.instance()
    if application is None:
        _QT_APPLICATION = QGuiApplication(["o2-pipeline-render"])
    else:
        _QT_APPLICATION = application


def _load_diagram_font(size: int, bold: bool = False) -> QFont:
    font = QFont("Arial", size)
    font.setBold(bold)
    return font


def _draw_centered_text(
    painter: QPainter,
    box: tuple[int, int, int, int],
    text: str,
    font: QFont,
    fill: tuple[int, int, int] = (38, 40, 48),
    line_spacing: int = 4,
) -> None:
    metrics = QFontMetrics(font)
    max_width = max(1, box[2] - box[0])
    lines: list[str] = []
    for raw_line in text.split("\n"):
        words = raw_line.split(" ")
        current_line = ""
        for word in words:
            candidate = word if not current_line else f"{current_line} {word}"
            if metrics.horizontalAdvance(candidate) <= max_width:
                current_line = candidate
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

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


def _draw_box(
    painter: QPainter,
    box: tuple[int, int, int, int],
    text: str,
    font: QFont,
    fill: tuple[int, int, int] = (242, 245, 251),
    outline: tuple[int, int, int] = (32, 32, 35),
    radius: int = 7,
    width: int = 2,
) -> None:
    left, top, right, bottom = box
    painter.setPen(QPen(QColor(*outline), width))
    painter.setBrush(QBrush(QColor(*fill)))
    painter.drawRoundedRect(QRectF(left, top, right - left, bottom - top), radius, radius)
    _draw_centered_text(painter, (left + 12, top + 8, right - 12, bottom - 8), text, font)


def _draw_dashed_group(
    painter: QPainter,
    box: tuple[int, int, int, int],
    outline: tuple[int, int, int] = (24, 24, 26),
    radius: int = 7,
) -> None:
    left, top, right, bottom = box
    pen = QPen(QColor(*outline), 2)
    pen.setDashPattern([3.0, 5.0])
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawRoundedRect(QRectF(left, top, right - left, bottom - top), radius, radius)


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
    fill: tuple[int, int, int] = (24, 24, 26),
    width: int = 2,
    arrow_size: int = 10,
    dashed: bool = False,
    corner_radius: int = 18,
) -> None:
    if len(points) < 2:
        return

    pen = QPen(QColor(*fill), width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    if dashed:
        pen.setDashPattern([3.0, 5.0])
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

        # Straight-through points stay straight; only true direction changes receive rounded corners.
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


def create_o2_pipeline_image(path: Path) -> None:
    """Use PyQt5 to draw the O2a SIFT repeatability pipeline diagram."""

    _ensure_qt_application()
    canvas = QImage(LOGICAL_CANVAS_WIDTH * OUTPUT_SCALE, LOGICAL_CANVAS_HEIGHT * OUTPUT_SCALE, QImage.Format_RGB32)
    canvas.fill(QColor("white"))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    painter.scale(OUTPUT_SCALE, OUTPUT_SCALE)

    title_font = _load_diagram_font(24)
    box_font = _load_diagram_font(13)
    small_font = _load_diagram_font(12)
    bold_font = _load_diagram_font(13, bold=True)

    node_fill = (243, 246, 252)
    output_fill = (232, 248, 238)
    outline = (30, 31, 34)

    middlebury = (22, 138, 212, 196)
    load_rgb = (297, 126, 519, 207)
    transform = (565, 126, 845, 207)
    transformed_view = (883, 126, 1098, 207)
    original_gray = (297, 256, 519, 338)
    transformed_gray = (867, 256, 1114, 338)
    original_detector = (314, 365, 501, 423)
    transformed_detector = (897, 365, 1083, 423)
    original_kp = (297, 456, 519, 538)
    transformed_kp = (860, 456, 1119, 538)
    keypoint_vis = (611, 468, 791, 526)
    matching = (570, 569, 831, 629)
    ratio = (570, 671, 831, 730)
    homography = (537, 774, 863, 832)
    score = (606, 879, 796, 937)
    metrics = (627, 984, 774, 1042)
    match_vis = (932, 774, 1112, 832)

    manual_group = (1178, 23, 2025, 1080)
    input_gray = (1252, 126, 1514, 185)
    normalize = (1234, 223, 1530, 282)
    gaussian = (1256, 320, 1510, 379)
    dog = (1232, 417, 1532, 476)
    extrema = (1232, 514, 1532, 573)
    taylor = (1256, 611, 1510, 670)
    contrast = (1273, 708, 1493, 767)
    hessian = (1224, 805, 1542, 864)
    suppression = (1234, 902, 1530, 961)

    gradient = (1650, 126, 1978, 185)
    orient_window = (1700, 233, 1930, 292)
    orient_hist = (1668, 341, 1962, 400)
    orient_interp = (1684, 448, 1946, 506)
    rotate = (1652, 556, 1977, 614)
    cells = (1684, 664, 1946, 722)
    descriptor = (1712, 771, 1918, 829)
    normalize_desc = (1668, 879, 1962, 936)
    manual_output = (1684, 985, 1946, 1044)

    _draw_box(painter, middlebury, "Middlebury scene:  im0", box_font, node_fill, outline)
    _draw_box(painter, load_rgb, "Load RGB image", box_font, node_fill, outline)
    _draw_box(painter, transform, "Generate deterministic\nrandom transform", bold_font, node_fill, outline)
    _draw_box(painter, transformed_view, "Transformed view", box_font, node_fill, outline)
    _draw_box(painter, original_gray, "Original grayscale", box_font, node_fill, outline)
    _draw_box(painter, transformed_gray, "Transformed grayscale", box_font, node_fill, outline)
    _draw_box(painter, original_detector, "Manual SIFT detector", small_font, node_fill, outline)
    _draw_box(painter, transformed_detector, "Manual SIFT detector", small_font, node_fill, outline)
    _draw_box(painter, original_kp, "Original keypoints\n+ 128D descriptors", box_font, node_fill, outline)
    _draw_box(painter, transformed_kp, "Transformed keypoints\n+ 128D descriptors", box_font, node_fill, outline)
    _draw_box(painter, keypoint_vis, "Keypoint visualizations", small_font, output_fill, outline)
    _draw_dashed_group(painter, keypoint_vis)
    _draw_box(painter, matching, "L2 KNN descriptor matching", small_font, node_fill, outline)
    _draw_box(painter, ratio, "Lowe ratio filtering", small_font, node_fill, outline)
    _draw_box(painter, homography, "Known homography repeatability check", small_font, node_fill, outline)
    _draw_box(painter, score, "Repeatability score", small_font, node_fill, outline)
    _draw_box(painter, metrics, "Metrics CSV", small_font, output_fill, outline)
    _draw_dashed_group(painter, metrics)
    _draw_box(painter, match_vis, "Match visualizations", small_font, output_fill, outline)
    _draw_dashed_group(painter, match_vis)

    _draw_dashed_group(painter, manual_group)
    _draw_centered_text(painter, (1178, 40, 2025, 72), "Manual SIFT", title_font)
    for box, text in [
        (input_gray, "Input grayscale image"),
        (normalize, "Normalize pixels to 0~1"),
        (gaussian, "Gaussian scale space"),
        (dog, "Difference of Gaussian pyramid"),
        (extrema, "26-neighbour extrema search"),
        (taylor, "3D Taylor localization"),
        (contrast, "Low contrast rejection"),
        (hessian, "Hessian edge-response rejection"),
        (suppression, "Spatial suppression and max feature cap"),
        (gradient, "Gradient magnitude and orientation pyramid"),
        (orient_window, "16x16 orientation window"),
        (orient_hist, "36-bin dominant orientation histogram"),
        (orient_interp, "Orientation peak interpolation"),
        (rotate, "Rotate and scale-normalize descriptor window"),
        (cells, "4x4 cells times 8 orientation bins"),
        (descriptor, "128D descriptor"),
        (normalize_desc, "L2 normalize, clip at 0.2, renormalize"),
        (manual_output, "ManualKeypoint plus descriptor"),
    ]:
        _draw_box(painter, box, text, small_font, node_fill, outline)

    _draw_arrow(painter, [_right_center(middlebury), _left_center(load_rgb)])
    _draw_arrow(painter, [_right_center(load_rgb), _left_center(transform)])
    _draw_arrow(painter, [_right_center(transform), _left_center(transformed_view)])
    _draw_arrow(painter, [_bottom_center(load_rgb), _top_center(original_gray)])
    _draw_arrow(painter, [_bottom_center(original_gray), _top_center(original_detector)])
    _draw_arrow(painter, [_bottom_center(original_detector), _top_center(original_kp)])
    _draw_arrow(painter, [_bottom_center(transformed_view), _top_center(transformed_gray)])
    _draw_arrow(painter, [_bottom_center(transformed_gray), _top_center(transformed_detector)])
    _draw_arrow(painter, [_bottom_center(transformed_detector), _top_center(transformed_kp)])
    _draw_arrow(painter, [_right_center(original_kp), _left_center(keypoint_vis)], dashed=True)
    _draw_arrow(painter, [_left_center(transformed_kp), _right_center(keypoint_vis)], dashed=True)
    _draw_arrow(painter, [_bottom_center(original_kp), (_center_x(original_kp), 564), _left_center(matching)])
    _draw_arrow(painter, [_bottom_center(transformed_kp), (_center_x(transformed_kp), 564), _right_center(matching)])
    _draw_arrow(painter, [_bottom_center(matching), _top_center(ratio)])
    _draw_arrow(painter, [_bottom_center(ratio), _top_center(homography)])
    _draw_arrow(painter, [_bottom_center(homography), _top_center(score)])
    _draw_arrow(painter, [_bottom_center(score), _top_center(metrics)], dashed=True)
    _draw_arrow(painter, [_right_center(homography), _left_center(match_vis)], dashed=True)

    left_manual_flow = [input_gray, normalize, gaussian, dog, extrema, taylor, contrast, hessian, suppression]
    for first, second in zip(left_manual_flow, left_manual_flow[1:]):
        _draw_arrow(painter, [_bottom_center(first), _top_center(second)])
    _draw_arrow(
        painter,
        [_right_center(suppression), (1589, _center_y(suppression)), (1589, _center_y(gradient)), _left_center(gradient)],
    )
    right_manual_flow = [
        gradient,
        orient_window,
        orient_hist,
        orient_interp,
        rotate,
        cells,
        descriptor,
        normalize_desc,
        manual_output,
    ]
    for first, second in zip(right_manual_flow, right_manual_flow[1:]):
        _draw_arrow(painter, [_bottom_center(first), _top_center(second)])

    painter.end()

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        saved = canvas.save(str(path), "JPEG", 100)
    else:
        saved = canvas.save(str(path), "PNG")
    if not saved:
        raise OSError(f"Failed to save O2 pipeline image: {path}")


def write_o2_pipeline_assets(pipeline_dir: Path) -> None:
    """Write the O2a visual pipeline asset used by the report/PDF."""

    create_o2_pipeline_image(pipeline_dir / "sift_pipeline.jpg")
