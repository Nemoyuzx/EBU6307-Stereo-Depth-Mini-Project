from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QFontMetrics, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QPolygonF


LOGICAL_CANVAS_WIDTH = 2048
LOGICAL_CANVAS_HEIGHT = 680
REFERENCE_CANVAS_WIDTH = 2731
REFERENCE_CANVAS_HEIGHT = 916
OUTPUT_SCALE = 4

_QT_APPLICATION: QGuiApplication | None = None


def _ensure_qt_application() -> None:
    global _QT_APPLICATION
    application = QGuiApplication.instance()
    if application is None:
        _QT_APPLICATION = QGuiApplication(["o4-pipeline-render"])
    else:
        _QT_APPLICATION = application


def _font(size: int, bold: bool = False) -> QFont:
    font = QFont("Arial", size)
    font.setBold(bold)
    return font


def _wrap_text(text: str, metrics: QFontMetrics, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.split("\n"):
        current = ""
        for word in raw_line.split(" "):
            candidate = word if not current else f"{current} {word}"
            if metrics.horizontalAdvance(candidate) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def _draw_centered_text(
    painter: QPainter,
    box: tuple[int, int, int, int],
    text: str,
    font: QFont,
    fill: tuple[int, int, int] = (34, 36, 44),
    line_spacing: int = 3,
) -> None:
    metrics = QFontMetrics(font)
    lines = _wrap_text(text, metrics, max(1, box[2] - box[0]))
    line_height = metrics.height()
    total_height = len(lines) * line_height + max(0, len(lines) - 1) * line_spacing
    y = box[1] + max(0.0, (box[3] - box[1] - total_height) / 2.0) + metrics.ascent()
    painter.setFont(font)
    painter.setPen(QPen(QColor(*fill)))
    for line in lines:
        line_width = metrics.horizontalAdvance(line)
        x = box[0] + max(0.0, (box[2] - box[0] - line_width) / 2.0)
        painter.drawText(QPointF(x, y), line)
        y += line_height + line_spacing


def _draw_centered_mixed_text(
    painter: QPainter,
    box: tuple[int, int, int, int],
    text: str,
    font: QFont,
    bold_font: QFont,
    fill: tuple[int, int, int] = (34, 36, 44),
    line_spacing: int = 3,
) -> None:
    regular_metrics = QFontMetrics(font)
    bold_metrics = QFontMetrics(bold_font)
    raw_lines = _wrap_text(text, regular_metrics, max(1, box[2] - box[0]))
    if not raw_lines:
        return

    line_fonts = [bold_font] + [font for _ in raw_lines[1:]]
    line_metrics = [QFontMetrics(line_font) for line_font in line_fonts]
    total_height = sum(metric.height() for metric in line_metrics) + max(0, len(raw_lines) - 1) * line_spacing
    y = box[1] + max(0.0, (box[3] - box[1] - total_height) / 2.0)

    painter.setPen(QPen(QColor(*fill)))
    for line, line_font, metric in zip(raw_lines, line_fonts, line_metrics):
        line_width = metric.horizontalAdvance(line)
        x = box[0] + max(0.0, (box[2] - box[0] - line_width) / 2.0)
        painter.setFont(line_font)
        painter.drawText(QPointF(x, y + metric.ascent()), line)
        y += metric.height() + line_spacing


def _draw_bullet_list(
    painter: QPainter,
    box: tuple[int, int, int, int],
    lines: list[str],
    font: QFont,
    fill: tuple[int, int, int] = (34, 36, 44),
    line_spacing: int = 3,
) -> None:
    metrics = QFontMetrics(font)
    line_height = metrics.height()
    total_height = len(lines) * line_height + max(0, len(lines) - 1) * line_spacing
    y = box[1] + max(0.0, (box[3] - box[1] - total_height) / 2.0)
    bullet_x = box[0] + 8
    text_x = box[0] + 22

    painter.setFont(font)
    painter.setPen(QPen(QColor(*fill)))
    painter.setBrush(QBrush(QColor(*fill)))
    for line in lines:
        baseline = y + metrics.ascent()
        painter.drawEllipse(QPointF(bullet_x, y + line_height / 2.0), 2.0, 2.0)
        painter.drawText(QPointF(text_x, baseline), line)
        y += line_height + line_spacing


def _draw_group(
    painter: QPainter,
    box: tuple[int, int, int, int],
    title: str,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    title_fill: tuple[int, int, int],
    font: QFont,
    title_y_offset: int = 8,
) -> None:
    left, top, right, bottom = box
    pen = QPen(QColor(*outline), 2)
    pen.setDashPattern([2.0, 3.0])
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QBrush(QColor(*fill)))
    painter.drawRoundedRect(QRectF(left, top, right - left, bottom - top), 6, 6)
    if title:
        _draw_centered_text(
            painter,
            (left + 12, top + title_y_offset, right - 12, top + title_y_offset + 26),
            title,
            font,
            title_fill,
        )


def _draw_box(
    painter: QPainter,
    box: tuple[int, int, int, int],
    text: str,
    font: QFont,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    text_fill: tuple[int, int, int] = (34, 36, 44),
    radius: int = 5,
    width: int = 2,
    title_bold: bool = False,
) -> None:
    left, top, right, bottom = box
    painter.setPen(QPen(QColor(*outline), width))
    painter.setBrush(QBrush(QColor(*fill)))
    painter.drawRoundedRect(QRectF(left, top, right - left, bottom - top), radius, radius)
    text_box = (left + 8, top + 5, right - 8, bottom - 5)
    if title_bold and text:
        _draw_centered_mixed_text(painter, text_box, text, font, _font(font.pointSize(), bold=True), text_fill)
    else:
        _draw_centered_text(painter, text_box, text, font, text_fill)


def _center_x(box: tuple[int, int, int, int]) -> int:
    return int(round((box[0] + box[2]) / 2.0))


def _center_y(box: tuple[int, int, int, int]) -> int:
    return int(round((box[1] + box[3]) / 2.0))


def _left_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return box[0], _center_y(box)


def _right_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return box[2], _center_y(box)


def _top_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return _center_x(box), box[1]


def _bottom_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return _center_x(box), box[3]


def _draw_arrow(
    painter: QPainter,
    points: list[tuple[int, int]],
    fill: tuple[int, int, int] = (24, 24, 26),
    width: int = 2,
    arrow_size: int = 6,
    corner_radius: int = 14,
) -> None:
    if len(points) < 2:
        return
    end_x, end_y = points[-1]
    start_x, start_y = points[-2]
    final_dx = end_x - start_x
    final_dy = end_y - start_y
    final_length = math.hypot(final_dx, final_dy)
    if final_length > 0.0:
        shaft_gap = min(arrow_size * 0.78, final_length * 0.5)
        shaft_end = (end_x - final_dx / final_length * shaft_gap, end_y - final_dy / final_length * shaft_gap)
    else:
        shaft_end = (end_x, end_y)

    pen = QPen(QColor(*fill), width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    path = QPainterPath()
    path.moveTo(QPointF(*points[0]))
    for index in range(1, len(points) - 1):
        px, py = points[index - 1]
        cx, cy = points[index]
        nx, ny = points[index + 1]
        incoming_x = px - cx
        incoming_y = py - cy
        outgoing_x = nx - cx
        outgoing_y = ny - cy
        incoming_length = math.hypot(incoming_x, incoming_y)
        outgoing_length = math.hypot(outgoing_x, outgoing_y)
        if incoming_length <= 0.0 or outgoing_length <= 0.0:
            path.lineTo(QPointF(cx, cy))
            continue
        cross = incoming_x * outgoing_y - incoming_y * outgoing_x
        if abs(cross) < 1e-6:
            path.lineTo(QPointF(cx, cy))
            continue
        radius = min(float(corner_radius), incoming_length / 2.0, outgoing_length / 2.0)
        before = QPointF(cx + incoming_x / incoming_length * radius, cy + incoming_y / incoming_length * radius)
        after = QPointF(cx + outgoing_x / outgoing_length * radius, cy + outgoing_y / outgoing_length * radius)
        path.lineTo(before)
        path.quadTo(QPointF(cx, cy), after)
    path.lineTo(QPointF(*shaft_end))
    painter.drawPath(path)

    angle = math.atan2(final_dy, final_dx)
    arrowhead = QPolygonF(
        [
            QPointF(end_x, end_y),
            QPointF(end_x + math.cos(angle + math.pi * 0.78) * arrow_size, end_y + math.sin(angle + math.pi * 0.78) * arrow_size),
            QPointF(end_x + math.cos(angle - math.pi * 0.78) * arrow_size, end_y + math.sin(angle - math.pi * 0.78) * arrow_size),
        ]
    )
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(*fill)))
    painter.drawPolygon(arrowhead)


def _draw_line(
    painter: QPainter,
    points: list[tuple[int, int]],
    fill: tuple[int, int, int] = (24, 24, 26),
    width: int = 2,
    corner_radius: int = 14,
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
        px, py = points[index - 1]
        cx, cy = points[index]
        nx, ny = points[index + 1]
        incoming_x = px - cx
        incoming_y = py - cy
        outgoing_x = nx - cx
        outgoing_y = ny - cy
        incoming_length = math.hypot(incoming_x, incoming_y)
        outgoing_length = math.hypot(outgoing_x, outgoing_y)
        if incoming_length <= 0.0 or outgoing_length <= 0.0:
            path.lineTo(QPointF(cx, cy))
            continue
        cross = incoming_x * outgoing_y - incoming_y * outgoing_x
        if abs(cross) < 1e-6:
            path.lineTo(QPointF(cx, cy))
            continue
        radius = min(float(corner_radius), incoming_length / 2.0, outgoing_length / 2.0)
        before = QPointF(cx + incoming_x / incoming_length * radius, cy + incoming_y / incoming_length * radius)
        after = QPointF(cx + outgoing_x / outgoing_length * radius, cy + outgoing_y / outgoing_length * radius)
        path.lineTo(before)
        path.quadTo(QPointF(cx, cy), after)
    path.lineTo(QPointF(*points[-1]))
    painter.drawPath(path)


def _draw_arrowhead(
    painter: QPainter,
    tip: tuple[int, int],
    previous: tuple[int, int],
    fill: tuple[int, int, int] = (24, 24, 26),
    arrow_size: int = 6,
) -> None:
    end_x, end_y = tip
    start_x, start_y = previous
    angle = math.atan2(end_y - start_y, end_x - start_x)
    arrowhead = QPolygonF(
        [
            QPointF(end_x, end_y),
            QPointF(end_x + math.cos(angle + math.pi * 0.78) * arrow_size, end_y + math.sin(angle + math.pi * 0.78) * arrow_size),
            QPointF(end_x + math.cos(angle - math.pi * 0.78) * arrow_size, end_y + math.sin(angle - math.pi * 0.78) * arrow_size),
        ]
    )
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(*fill)))
    painter.drawPolygon(arrowhead)


def _draw_line_to_arrow_base(
    painter: QPainter,
    points: list[tuple[int, int]],
    fill: tuple[int, int, int] = (24, 24, 26),
    width: int = 2,
    arrow_size: int = 6,
    corner_radius: int = 14,
) -> None:
    if len(points) < 2:
        return
    end_x, end_y = points[-1]
    start_x, start_y = points[-2]
    final_dx = end_x - start_x
    final_dy = end_y - start_y
    final_length = math.hypot(final_dx, final_dy)
    if final_length > 0.0:
        shaft_gap = min(arrow_size * 0.78, final_length * 0.5)
        shaft_end = (int(round(end_x - final_dx / final_length * shaft_gap)), int(round(end_y - final_dy / final_length * shaft_gap)))
    else:
        shaft_end = (end_x, end_y)
    _draw_line(painter, [*points[:-1], shaft_end], fill, width, corner_radius)


def _draw_vertical_label(painter: QPainter, box: tuple[int, int, int, int], text: str, font: QFont) -> None:
    painter.save()
    painter.translate(_center_x(box), _center_y(box))
    painter.rotate(-90)
    _draw_centered_text(painter, (-(box[3] - box[1]) // 2, -12, (box[3] - box[1]) // 2, 12), text, font)
    painter.restore()


def _draw_vertical_two_weight_label(
    painter: QPainter,
    box: tuple[int, int, int, int],
    title: str,
    dimension: str,
    title_font: QFont,
    dimension_font: QFont,
) -> None:
    painter.save()
    painter.translate(_center_x(box), _center_y(box))
    painter.rotate(-90)

    title_metrics = QFontMetrics(title_font)
    dimension_metrics = QFontMetrics(dimension_font)
    line_spacing = 3
    total_height = title_metrics.height() + line_spacing + dimension_metrics.height()
    y = -total_height / 2.0
    text_width = box[3] - box[1]

    painter.setPen(QPen(QColor(34, 36, 44)))
    for line, font, metrics in [
        (title, title_font, title_metrics),
        (dimension, dimension_font, dimension_metrics),
    ]:
        painter.setFont(font)
        x = -text_width / 2.0 + (text_width - metrics.horizontalAdvance(line)) / 2.0
        painter.drawText(QPointF(x, y + metrics.ascent()), line)
        y += metrics.height() + line_spacing
    painter.restore()


def create_o4_pipeline_image(path: Path) -> None:
    """Draw the O4 baseline_sgm Transformer pipeline diagram."""

    _ensure_qt_application()
    x_scale = REFERENCE_CANVAS_WIDTH / LOGICAL_CANVAS_WIDTH
    y_scale = REFERENCE_CANVAS_HEIGHT / LOGICAL_CANVAS_HEIGHT
    canvas = QImage(REFERENCE_CANVAS_WIDTH * OUTPUT_SCALE, REFERENCE_CANVAS_HEIGHT * OUTPUT_SCALE, QImage.Format_RGB32)
    canvas.fill(QColor("white"))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    painter.scale(OUTPUT_SCALE * x_scale, OUTPUT_SCALE * y_scale)

    title_font = _font(12)
    tokenization_title_font = _font(10)
    node_font = _font(7)
    small_font = _font(6)
    bold_small = _font(6, bold=True)

    purple = (142, 112, 222)
    green = (72, 158, 94)
    blue = (82, 124, 209)
    gold = (218, 174, 65)
    red = (231, 89, 89)
    gray = (113, 122, 135)
    black = (34, 36, 44)

    _draw_group(painter, (15, 15, 209, 672), "Input", (246, 241, 255), purple, purple, title_font)
    _draw_group(painter, (231, 15, 659, 672), "Token description construction   (stereo input)", (238, 251, 243), green, green, title_font)
    _draw_group(painter, (686, 15, 1237, 672), "Stereo Token Transformer", (246, 250, 255), blue, blue, title_font)
    _draw_group(painter, (1263, 15, 1562, 672), "Dense Matching", (255, 249, 229), gold, gold, title_font)
    _draw_group(painter, (1587, 15, 1834, 672), "Post-processing", (255, 246, 246), red, red, title_font)
    _draw_group(painter, (1861, 15, 2033, 672), "Output", (251, 252, 254), gray, gray, title_font)

    load = (42, 78, 179, 118)
    stereo_pair = (42, 316, 179, 356)
    mask = (42, 570, 179, 610)
    _draw_box(painter, load, "Load grayscale images\nim0.png / im1.png", small_font, (236, 229, 252), purple)
    _draw_box(painter, stereo_pair, "Middlebury stereo pair", small_font, (236, 229, 252), purple)
    _draw_box(painter, mask, "Build content masks\nnon-black valid ROI", small_font, (236, 229, 252), purple)

    left_img = (286, 78, 406, 118)
    right_img = (485, 78, 605, 118)
    left_tok = (256, 160, 436, 352)
    right_tok = (455, 160, 635, 352)
    left_pool = (272, 198, 420, 242)
    left_patch = (272, 264, left_pool[2], 336)
    right_pool = (471, 198, 619, left_pool[3])
    right_patch = (471, 263, right_pool[2], left_patch[3])
    channels = (257, 387, 635, 435)
    formula = (257, 470, 635, 535)
    left_desc = (286, 570, 406, 610)
    right_desc = (485, 570, 605, 610)
    for box, label in [
        (left_img, "Left grayscale image\nR ^ {H x W}"),
        (right_img, "Right grayscale image\nR ^ {H x W}"),
    ]:
        _draw_box(painter, box, label, small_font, (228, 248, 236), green, title_bold=True)
    for box in [left_tok, right_tok]:
        _draw_group(
            painter,
            box,
            "Tokenization",
            (233, 250, 239),
            green,
            (34, 36, 44),
            tokenization_title_font,
            title_y_offset=5,
        )
    _draw_group(painter, formula, "", (233, 250, 239), green, green, small_font)
    for box, label in [
        (left_pool, "Average pooling\n(downsample_factor = 1)"),
        (left_patch, "Patch tokenization\n(patch_size = 2)\n-> 2x2 pixels/token\ntoken_span=2px/token"),
        (right_pool, "Average pooling\n(downsample_factor = 1)"),
        (right_patch, "Patch tokenization\n(patch_size = 2)\n-> 2x2 pixels/token\ntoken_span=2px/token"),
        (channels, "token descriptor channels (per token)\n(channels = intensity + grad_x + grad_y + 3x3 context)"),
        (formula, "Descriptor Formula (per channel)\ninput_dim = 4 x patch_size^2 + 2 = 18\n(16 local value + plus mean + std)"),
        (left_desc, "Left descriptor\nR ^ {18}"),
        (right_desc, "Right descriptor\nR ^ {18}"),
    ]:
        _draw_box(
            painter,
            box,
            label,
            small_font,
            (226, 248, 234),
            green,
            title_bold=box in {left_pool, left_patch, right_pool, right_patch, channels, formula, left_desc, right_desc},
        )

    left_builder = (702, 78, 738, 610)
    right_embedding_bar = (1181, 78, 1216, 610)
    transformer = (759, 78, 1158, 610)
    ln = (790, 114, 1090, 137)
    proj = (790, 147, 1090, 179)
    cls = (790, 189, 1090, 225)
    enc = (790, 235, 1090, 417)
    layers = (814, 261, 955, 403)
    fusion = (790, 427, 1090, 462)
    head = (790, 472, 1090, 558)
    l2 = (790, 568, 1090, 597)
    _draw_box(painter, left_builder, "", small_font, (239, 245, 255), blue)
    _draw_vertical_two_weight_label(painter, left_builder, "Left / Right descriptor builder", "R ^ {18}", bold_small, small_font)
    _draw_box(painter, right_embedding_bar, "", small_font, (239, 245, 255), blue)
    _draw_vertical_two_weight_label(painter, right_embedding_bar, "Encoded left / right token embeddings", "R ^ {96}", bold_small, small_font)
    _draw_box(painter, transformer, "", small_font, (239, 245, 255), blue)
    _draw_centered_text(
        painter,
        (transformer[0] + 16, transformer[1] + 12, transformer[2] - 16, transformer[1] + 34),
        "5-fold StereoTokenTransformer (num_folds = 5)",
        node_font,
    )
    for box, label in [
        (ln, "Input LayerNorm:  LayerNorm(18)"),
        (proj, "Linear projection:  Linear 18 -> 576\nd_{model}=6 x 96, d_{model} x sequence_length = 576"),
        (cls, "Add CLS token + learned positional embedding\nsequence_length = 6 + 1 = 7"),
        (enc, ""),
        (fusion, "Fusion: 0.5 x CLS + 0.5 x mean(non-CLS)\n(Output vector)"),
        (head, "Output head:\nLayerNorm\n-> Linear 96 -> 96\n-> GELU\n-> Linear 96 -> 96\n-> LayerNorm"),
        (l2, "L2 normalization:  96-D embedding"),
    ]:
        _draw_box(
            painter,
            box,
            label,
            small_font,
            (238, 245, 255),
            blue,
            title_bold=box in {ln, proj, cls, enc, fusion, head, l2},
        )
    _draw_centered_text(
        painter,
        (enc[0] + 12, enc[1] + 2, enc[2] - 12, enc[1] + 26),
        "Transformer Encoder x 4",
        bold_small,
    )
    _draw_group(painter, layers, "(identical layers)", (238, 245, 255), blue, black, small_font, title_y_offset=1)
    for index, y in enumerate([287, 315, 343, 371], start=1):
        _draw_box(painter, (830, y, 940, y + 20), f"Layer {index}", small_font, (238, 245, 255), blue)
    _draw_bullet_list(
        painter,
        (970, 278, 1088, 386),
        [
            "d_{model} = 96",
            "heads = 8",
            "ffn = 384",
            "activation = GELU",
            "dropout = 0",
            "norm_first = True",
            "encoder_layers = 4",
        ],
        bold_small,
    )
    for text, y in [
        ("R ^ {18}", _center_y(ln)),
        ("R ^ {6 x 96}", _center_y(proj)),
        ("R ^ {7 x 96}", _center_y(cls)),
        ("R ^ {7 x 96}", _center_y(enc)),
        ("R ^ {96}", _center_y(fusion)),
        ("R ^ {96}", _center_y(head)),
        ("R ^ {96}", _center_y(l2)),
    ]:
        _draw_centered_text(painter, (1090, y - 10, 1150, y + 10), text, small_font)

    left_emb = (1289, 78, 1404, 118)
    right_emb = (1421, 78, 1536, 118)
    sim = (1311, 149, 1513, 187)
    matrix = (1333, 219, 1491, 258)
    cost = (1333, 289, 1491, 328)
    mask_bound = (1337, 359, 1487, 398)
    sgm = (1333, 430, 1491, 469)
    solve = (1326, 500, 1498, 539)
    raw = (1353, 570, 1471, 610)
    for box, label in [
        (left_emb, "Left token embeddings\nR ^ {96}"),
        (right_emb, "Right token embeddings\nR ^ {96}"),
        (sim, "Cosine similarity x exp(logit_scale)\ninit logit_scale = 2.3025, scale = 1.0"),
        (matrix, "Calculate score matrix S\n(Z_L, Z_R^T)"),
        (cost, "Token cost volume = -S\n(higher similarity -> lower cost)"),
        (mask_bound, "apply content mask\n+ token disparity bound"),
        (sgm, "O3-style 4-direction SGM\non token grid"),
        (solve, "Left / right token disparity solve\n(Quadratic disparity regression)"),
        (raw, "Raw token diagnostics\n(raw_filtered)"),
    ]:
        _draw_box(
            painter,
            box,
            label,
            small_font,
            (255, 243, 203),
            gold,
            title_bold=box in {left_emb, right_emb, sim, matrix, cost, mask_bound, sgm, solve, raw},
        )

    lr = (1615, 108, 1808, 173)
    clean = (1633, 217, 1790, 304)
    fill = (1633, 348, 1790, 413)
    upsample = (1633, 456, 1790, 558)
    for box, label in [
        (lr, "Token LR consistency + confidence margin\n(consistency_threshold = 1.0)"),
        (clean, "Token cleanup\n(token_median_filter_size = 3\nspeckle_max_size = 150\nspeckle_max_diff = 1.0)"),
        (fill, "O3-style fill + weighted median\n(fill_invalid_passes = 2)"),
        (upsample, "Upsample token disparity\nto full resolution\n(2 px/token)"),
    ]:
        _draw_box(painter, box, label, small_font, (255, 228, 228), red, title_bold=True)

    out1 = (1895, 176, 2003, 235)
    out2 = (1895, 269, 2003, 326)
    out3 = (1895, 373, 2003, 430)
    for box, label in [
        (out1, "disp0.pfm\n/ disp0.png"),
        (out2, "confidence.png\n/ error_map.png\n/ raw disparity"),
        (out3, "metrics.csv\n/ fold_summary.csv"),
    ]:
        _draw_box(painter, box, label, small_font, (248, 249, 251), gray)

    _draw_arrow(painter, [_top_center(stereo_pair), _bottom_center(load)])
    _draw_arrow(painter, [_bottom_center(stereo_pair), _top_center(mask)])
    _draw_arrow(painter, [_right_center(load), (254, _center_y(load)), (254, 60), (_center_x(left_img), 60), _top_center(left_img)])
    _draw_arrow(painter, [_right_center(load), (254, _center_y(load)), (254, 60), (_center_x(right_img), 60), _top_center(right_img)])
    _draw_arrow(painter, [_bottom_center(left_img), _top_center(left_tok)])
    _draw_arrow(painter, [_bottom_center(right_img), _top_center(right_tok)])
    _draw_arrow(painter, [_bottom_center(left_pool), _top_center(left_patch)])
    _draw_arrow(painter, [_bottom_center(right_pool), _top_center(right_patch)])
    left_column_x = _center_x(left_desc)
    right_column_x = _center_x(right_desc)
    _draw_arrow(painter, [_bottom_center(left_tok), (left_column_x, left_tok[3] + 25), (left_column_x, channels[1])])
    _draw_arrow(painter, [_bottom_center(right_tok), (right_column_x, right_tok[3] + 25), (right_column_x, channels[1])])
    _draw_arrow(painter, [(left_column_x, channels[3]), (left_column_x, formula[1])])
    _draw_arrow(painter, [(right_column_x, channels[3]), (right_column_x, formula[1])])
    _draw_arrow(painter, [(left_column_x, formula[3]), (left_column_x, left_desc[1])])
    _draw_arrow(painter, [(right_column_x, formula[3]), (right_column_x, right_desc[1])])
    descriptor_merge_y = 632
    descriptor_route_x = left_builder[0] + 18
    descriptor_entry = (_center_x(left_builder), left_builder[3])
    _draw_line(
        painter,
        [
            _bottom_center(right_desc),
            (_center_x(right_desc), descriptor_merge_y),
            (_center_x(right_desc) + 22, descriptor_merge_y),
        ],
    )
    _draw_arrow(
        painter,
        [
            _bottom_center(left_desc),
            (_center_x(left_desc), descriptor_merge_y),
            (_center_x(right_desc), descriptor_merge_y),
            (descriptor_route_x, descriptor_merge_y),
            descriptor_entry,
        ],
    )
    _draw_arrow(painter, [_right_center(left_builder), _left_center(transformer)])
    _draw_arrow(painter, [_bottom_center(ln), _top_center(proj)])
    _draw_arrow(painter, [_bottom_center(proj), _top_center(cls)])
    _draw_arrow(painter, [_bottom_center(cls), _top_center(enc)])
    _draw_arrow(painter, [_bottom_center(enc), _top_center(fusion)])
    _draw_arrow(painter, [_bottom_center(fusion), _top_center(head)])
    _draw_arrow(painter, [_bottom_center(head), _top_center(l2)])
    _draw_arrow(painter, [_right_center(transformer), _left_center(right_embedding_bar)])
    embedding_bus_y = 60
    embedding_merge_y = 136
    embedding_bus_left_x = 1282
    _draw_line_to_arrow_base(
        painter,
        [
            _right_center(right_embedding_bar),
            (embedding_bus_left_x, _center_y(right_embedding_bar)),
            (embedding_bus_left_x, embedding_bus_y),
            (_center_x(left_emb), embedding_bus_y),
            _top_center(left_emb),
        ],
    )
    _draw_line_to_arrow_base(
        painter,
        [
            (_center_x(left_emb), embedding_bus_y),
            (_center_x(right_emb), embedding_bus_y),
            _top_center(right_emb),
        ],
    )
    _draw_arrowhead(painter, _top_center(left_emb), (_center_x(left_emb), embedding_bus_y))
    _draw_arrowhead(painter, _top_center(right_emb), (_center_x(right_emb), embedding_bus_y))
    _draw_line(painter, [_bottom_center(left_emb), (_center_x(left_emb), embedding_merge_y), (_center_x(sim), embedding_merge_y)])
    _draw_line(painter, [_bottom_center(right_emb), (_center_x(right_emb), embedding_merge_y), (_center_x(sim), embedding_merge_y)])
    _draw_arrow(painter, [(_center_x(sim), embedding_merge_y), _top_center(sim)])
    _draw_arrow(painter, [_bottom_center(sim), _top_center(matrix)])
    _draw_arrow(painter, [_bottom_center(matrix), _top_center(cost)])
    _draw_arrow(painter, [_bottom_center(cost), _top_center(mask_bound)])
    _draw_arrow(painter, [_bottom_center(mask_bound), _top_center(sgm)])
    _draw_arrow(painter, [_bottom_center(sgm), _top_center(solve)])
    _draw_arrow(painter, [_bottom_center(solve), _top_center(raw)])
    _draw_arrow(painter, [_right_center(solve), (1542, _center_y(solve)), (1542, _center_y(lr)), _left_center(lr)])
    _draw_arrow(painter, [_bottom_center(lr), _top_center(clean)])
    _draw_arrow(painter, [_bottom_center(clean), _top_center(fill)])
    _draw_arrow(painter, [_bottom_center(fill), _top_center(upsample)])
    _draw_arrow(painter, [_right_center(upsample), (1846, _center_y(upsample)), (1846, _center_y(out1)), _left_center(out1)])
    _draw_arrow(painter, [_right_center(upsample), (1875, _center_y(upsample)), (1875, _center_y(out2)), _left_center(out2)])
    _draw_arrow(painter, [_right_center(raw), (1875, _center_y(raw)), (1875, _center_y(out3)), _left_center(out3)])
    _draw_arrow(painter, [_bottom_center(mask), (_center_x(mask), 656), (1318, 656), (1318, _center_y(cost)), _left_center(cost)])

    painter.end()
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = canvas.save(str(path), "JPEG", 100) if path.suffix.lower() in {".jpg", ".jpeg"} else canvas.save(str(path), "PNG")
    if not saved:
        raise OSError(f"Failed to save O4 pipeline image: {path}")


def write_o4_pipeline_assets(pipeline_dir: Path) -> None:
    create_o4_pipeline_image(pipeline_dir / "transformer_pipeline.jpg")


if __name__ == "__main__":
    write_o4_pipeline_assets(Path("results/O4a_transformer"))
