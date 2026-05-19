from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import rsm


def array_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        image = np.dstack([image] * 3)
    image = np.ascontiguousarray(image.astype(np.uint8))
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("Only RGB images are supported")
    bytes_per_line = channels * width
    return QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()


class ImageViewer(QWidget):
    roiChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 420)
        self.setMouseTracking(True)
        self._image: QImage | None = None
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._roi: tuple[int, int, int, int] | None = None
        self._roi_mode = False
        self._selecting = False
        self._panning = False
        self._last_mouse = QPoint()
        self._selection_start = QPointF()
        self._selection_current = QPointF()

    def set_image_array(self, image: np.ndarray | None, reset_view: bool = True) -> None:
        self._image = array_to_qimage(image) if image is not None else None
        if reset_view:
            self.fit_to_view()
        self.update()

    def set_roi(self, roi: tuple[int, int, int, int] | None) -> None:
        self._roi = roi
        self.update()

    def roi(self) -> tuple[int, int, int, int] | None:
        return self._roi

    def set_roi_mode(self, enabled: bool) -> None:
        self._roi_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.OpenHandCursor)

    def clear_roi(self) -> None:
        self._roi = None
        self.roiChanged.emit(None)
        self.update()

    def fit_to_view(self) -> None:
        if not self._image or self._image.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        scale_x = self.width() / self._image.width()
        scale_y = self.height() / self._image.height()
        self._zoom = max(0.01, min(scale_x, scale_y) * 0.96)
        image_w = self._image.width() * self._zoom
        image_h = self._image.height() * self._zoom
        self._pan = QPointF((self.width() - image_w) / 2.0, (self.height() - image_h) / 2.0)
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._image is not None and abs(self._zoom - 1.0) < 1e-6:
            self.fit_to_view()

    def wheelEvent(self, event) -> None:  # noqa: N802
        if not self._image:
            return
        cursor = QPointF(event.position())
        before = self._to_image(cursor)
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom = float(np.clip(self._zoom * factor, 0.02, 30.0))
        self._pan = cursor - before * self._zoom
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self._image:
            return
        if event.button() == Qt.LeftButton and self._roi_mode:
            self._selecting = True
            self._selection_start = self._clamp_image_point(self._to_image(QPointF(event.position())))
            self._selection_current = self._selection_start
            self.update()
            return
        if event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._panning = True
            self._last_mouse = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._image:
            return
        if self._selecting:
            self._selection_current = self._clamp_image_point(self._to_image(QPointF(event.position())))
            self.update()
            return
        if self._panning:
            point = event.position().toPoint()
            delta = point - self._last_mouse
            self._pan += QPointF(delta)
            self._last_mouse = point
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._selecting and event.button() == Qt.LeftButton:
            rect = self._selection_rect()
            if rect.width() >= 3 and rect.height() >= 3:
                roi = (
                    int(round(rect.left())),
                    int(round(rect.top())),
                    int(round(rect.width())),
                    int(round(rect.height())),
                )
                self._roi = roi
                self.roiChanged.emit(roi)
            self._selecting = False
            self.update()
            return
        if self._panning and event.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            self._panning = False
            self.setCursor(Qt.CrossCursor if self._roi_mode else Qt.OpenHandCursor)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 32, 36))
        if not self._image:
            painter.setPen(QColor(170, 176, 184))
            painter.drawText(self.rect(), Qt.AlignCenter, "请选择图片")
            return

        painter.save()
        painter.translate(self._pan)
        painter.scale(self._zoom, self._zoom)
        painter.drawImage(0, 0, self._image)
        self._draw_roi(painter)
        painter.restore()

    def _draw_roi(self, painter: QPainter) -> None:
        pen_width = max(1.0 / max(self._zoom, 1e-6), 0.5)
        if self._roi:
            x, y, w, h = self._roi
            painter.setPen(QPen(QColor(0, 210, 255), pen_width))
            painter.drawRect(QRectF(x, y, w, h))
        if self._selecting:
            rect = self._selection_rect()
            painter.setPen(QPen(QColor(255, 205, 0), pen_width))
            painter.drawRect(rect)

    def _selection_rect(self) -> QRectF:
        x1 = min(self._selection_start.x(), self._selection_current.x())
        y1 = min(self._selection_start.y(), self._selection_current.y())
        x2 = max(self._selection_start.x(), self._selection_current.x())
        y2 = max(self._selection_start.y(), self._selection_current.y())
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    def _to_image(self, point: QPointF) -> QPointF:
        return QPointF((point.x() - self._pan.x()) / self._zoom, (point.y() - self._pan.y()) / self._zoom)

    def _clamp_image_point(self, point: QPointF) -> QPointF:
        if not self._image:
            return point
        return QPointF(
            min(max(point.x(), 0.0), float(self._image.width() - 1)),
            min(max(point.y(), 0.0), float(self._image.height() - 1)),
        )


class ResponseSurfaceCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9, 4.5), dpi=100)
        super().__init__(self.figure)
        self.setParent(parent)

    def show_message(self, message: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        self.draw_idle()

    def plot_surface(
        self,
        model: rsm.QuadraticModel,
        x_factor: str,
        y_factor: str,
        fixed_values: dict[str, float],
    ) -> None:
        x_grid, y_grid, z_grid = rsm.response_surface(model, x_factor, y_factor, fixed_values)
        self.figure.clear()

        contour_ax = self.figure.add_subplot(121)
        contour = contour_ax.contourf(x_grid, y_grid, z_grid, levels=24, cmap="viridis")
        contour_ax.contour(x_grid, y_grid, z_grid, levels=10, colors="white", linewidths=0.45, alpha=0.7)
        contour_ax.set_title(f"{model.response_name} 等高线")
        contour_ax.set_xlabel(rsm.FACTOR_LABELS[x_factor])
        contour_ax.set_ylabel(rsm.FACTOR_LABELS[y_factor])
        self.figure.colorbar(contour, ax=contour_ax, shrink=0.82)

        surface_ax = self.figure.add_subplot(122, projection="3d")
        surface_ax.plot_surface(x_grid, y_grid, z_grid, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92)
        surface_ax.set_title(f"{model.response_name} 响应面")
        surface_ax.set_xlabel(rsm.FACTOR_LABELS[x_factor])
        surface_ax.set_ylabel(rsm.FACTOR_LABELS[y_factor])
        surface_ax.set_zlabel(model.response_name)
        self.figure.tight_layout()
        self.draw_idle()
