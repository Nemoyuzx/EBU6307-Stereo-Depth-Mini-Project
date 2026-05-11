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
        _QT_APPLICATION = QGuiApplication(["o4-vit-architecture-render"])
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
    fill: tuple[int, int, int] = (35, 38, 46),
    line_spacing: int = 4,
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


def _draw_box(
    painter: QPainter,
    box: tuple[int, int, int, int],
    text: str,
    font: QFont,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    text_fill: tuple[int, int, int] = (35, 38, 46),
    radius: int = 8,
    width: int = 2,
) -> None:
    left, top, right, bottom = box
    painter.setPen(QPen(QColor(*outline), width))
    painter.setBrush(QBrush(QColor(*fill)))
    painter.drawRoundedRect(QRectF(left, top, right - left, bottom - top), radius, radius)
    _draw_centered_text(painter, (left + 14, top + 8, right - 14, bottom - 8), text, font, text_fill)


def _draw_group(
    painter: QPainter,
    box: tuple[int, int, int, int],
    title: str,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    title_fill: tuple[int, int, int],
    title_font: QFont,
) -> None:
    left, top, right, bottom = box
    pen = QPen(QColor(*outline), 2)
    pen.setDashPattern([3.0, 5.0])
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QBrush(QColor(*fill)))
    painter.drawRoundedRect(QRectF(left, top, right - left, bottom - top), 8, 8)
    _draw_centered_text(painter, (left + 18, top + 16, right - 18, top + 62), title, title_font, title_fill)


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
    fill: tuple[int, int, int] = (26, 28, 34),
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


def create_o4_vit_architecture_image(path: Path) -> None:
    """Draw the core O4 StereoTokenTransformer architecture."""

    _ensure_qt_application()
    canvas = QImage(LOGICAL_CANVAS_WIDTH * OUTPUT_SCALE, LOGICAL_CANVAS_HEIGHT * OUTPUT_SCALE, QImage.Format_RGB32)
    canvas.fill(QColor("white"))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    painter.scale(OUTPUT_SCALE, OUTPUT_SCALE)

    title_font = _font(26, bold=True)
    group_font = _font(22)
    node_font = _font(13)
    small_font = _font(12)
    tiny_font = _font(11)

    blue = (73, 117, 199)
    green = (72, 151, 91)
    gold = (205, 157, 49)
    red = (214, 83, 83)
    slate = (99, 108, 122)

    _draw_centered_text(
        painter,
        (42, 26, 2006, 66),
        "O4 Core ViT-Style StereoTokenTransformer Architecture",
        title_font,
        (28, 32, 40),
    )
    _draw_centered_text(
        painter,
        (42, 68, 2006, 98),
        "Shared encoder for left and candidate token descriptors; default O4 baseline_sgm parameters shown.",
        small_font,
        (95, 101, 114),
    )

    _draw_group(painter, (34, 120, 430, 1024), "Token Descriptor Input", (246, 250, 255), blue, blue, group_font)
    _draw_group(painter, (460, 120, 1288, 1024), "Shared ViT-Style Encoder", (246, 252, 247), green, green, group_font)
    _draw_group(painter, (1318, 120, 1698, 1024), "Embedding Head", (255, 250, 235), gold, gold, group_font)
    _draw_group(painter, (1728, 120, 2014, 1024), "Stereo Score Head", (253, 247, 247), red, red, group_font)

    descriptor = (82, 180, 382, 262)
    channels = (82, 310, 382, 430)
    stats = (82, 478, 382, 560)
    input_dim = (82, 642, 382, 724)
    fold_note = (82, 806, 382, 886)

    norm = (508, 186, 742, 252)
    projection = (508, 304, 742, 386)
    reshape = (508, 438, 742, 520)
    token_norm = (508, 572, 742, 638)

    cls_pos = (804, 240, 1080, 328)
    sequence = (804, 392, 1080, 474)
    encoder = (804, 536, 1080, 782)
    repeat = (834, 807, 1050, 867)

    fusion = (1358, 238, 1658, 330)
    output_head = (1358, 438, 1658, 568)
    l2 = (1358, 688, 1658, 770)

    left_embed = (1756, 240, 1986, 322)
    candidate_embed = (1756, 420, 1986, 502)
    cosine = (1756, 600, 1986, 682)
    score = (1756, 780, 1986, 862)

    for box, label in [
        (descriptor, "Token descriptor vector"),
        (channels, "4 local channels\nintensity + grad_x + grad_y\n+ 3x3 context"),
        (stats, "Append token statistics\nmean + std"),
        (input_dim, "input_dim = 4 x 2^2 + 2\n= 18"),
        (fold_note, "Trained per fold\nnum_folds = 5"),
    ]:
        _draw_box(painter, box, label, small_font, (236, 245, 255), blue)

    for box, label in [
        (norm, "Input LayerNorm\nLayerNorm(18)"),
        (projection, "Linear projection\n18 -> 6 x 96 = 576"),
        (reshape, "Reshape\n6 tokens x 96 dims"),
        (token_norm, "Token LayerNorm\nLayerNorm(96)"),
        (cls_pos, "Prepend CLS token\n+ learned position embedding"),
        (sequence, "Transformer input sequence\n7 tokens x 96 dims"),
    ]:
        _draw_box(painter, box, label, small_font, (237, 249, 240), green)

    _draw_box(
        painter,
        encoder,
        "TransformerEncoderLayer x4\nnorm_first = True\nMulti-head attention: 8 heads\nFFN: 96 -> 384 -> 96\nactivation = GELU, dropout = 0.0",
        tiny_font,
        (232, 247, 236),
        green,
    )
    _draw_box(painter, repeat, "repeat depth = 4", tiny_font, (232, 247, 236), green)

    _draw_box(
        painter,
        fusion,
        "Fuse encoded sequence\n0.5 x CLS + 0.5 x mean(non-CLS)",
        small_font,
        (255, 244, 212),
        gold,
    )
    _draw_box(
        painter,
        output_head,
        "Output head\nLayerNorm -> Linear 96->96\n-> GELU -> Linear 96->96\n-> LayerNorm",
        tiny_font,
        (255, 244, 212),
        gold,
    )
    _draw_box(painter, l2, "L2-normalized\n96-D embedding", small_font, (255, 244, 212), gold)

    _draw_box(painter, left_embed, "Left token embedding\nshared encoder output", small_font, (255, 234, 234), red)
    _draw_box(painter, candidate_embed, "Candidate right embeddings\nshared encoder output", small_font, (255, 234, 234), red)
    _draw_box(painter, cosine, "Cosine similarity\nleft dot candidate", small_font, (255, 234, 234), red)
    _draw_box(painter, score, "Scaled matching score\nexp(logit_scale), init approx 10\nclamp max = 50", tiny_font, (255, 234, 234), red)

    _draw_arrow(painter, [_bottom_center(descriptor), _top_center(channels)])
    _draw_arrow(painter, [_bottom_center(channels), _top_center(stats)])
    _draw_arrow(painter, [_bottom_center(stats), _top_center(input_dim)])
    _draw_arrow(painter, [_right_center(input_dim), _left_center(norm)])
    _draw_arrow(painter, [_bottom_center(norm), _top_center(projection)])
    _draw_arrow(painter, [_bottom_center(projection), _top_center(reshape)])
    _draw_arrow(painter, [_bottom_center(reshape), _top_center(token_norm)])
    _draw_arrow(painter, [_right_center(token_norm), (774, _center_y(token_norm)), (774, _center_y(cls_pos)), _left_center(cls_pos)])
    _draw_arrow(painter, [_bottom_center(cls_pos), _top_center(sequence)])
    _draw_arrow(painter, [_bottom_center(sequence), _top_center(encoder)])
    _draw_arrow(painter, [_bottom_center(encoder), _top_center(repeat)])
    _draw_arrow(painter, [_right_center(encoder), (1218, _center_y(encoder)), (1218, _center_y(fusion)), _left_center(fusion)])
    _draw_arrow(painter, [_bottom_center(fusion), _top_center(output_head)])
    _draw_arrow(painter, [_bottom_center(output_head), _top_center(l2)])
    _draw_arrow(painter, [_right_center(l2), (1712, _center_y(l2)), (1712, _center_y(left_embed)), _left_center(left_embed)])
    _draw_arrow(painter, [_right_center(l2), (1712, _center_y(l2)), (1712, _center_y(candidate_embed)), _left_center(candidate_embed)])
    _draw_arrow(painter, [_bottom_center(left_embed), (1871, 370), _top_center(cosine)])
    _draw_arrow(painter, [_bottom_center(candidate_embed), _top_center(cosine)])
    _draw_arrow(painter, [_bottom_center(cosine), _top_center(score)])

    _draw_centered_text(
        painter,
        (500, 928, 1668, 972),
        "The same encoder weights are used for left descriptors and every right-candidate descriptor.",
        small_font,
        slate,
    )

    painter.end()

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        saved = canvas.save(str(path), "JPEG", 100)
    else:
        saved = canvas.save(str(path), "PNG")
    if not saved:
        raise OSError(f"Failed to save O4 ViT architecture image: {path}")


if __name__ == "__main__":
    create_o4_vit_architecture_image(Path("results/O4a_transformer/vit_architecture.jpg"))
