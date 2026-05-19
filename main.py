from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import processing
import rsm
from widgets import ImageViewer, ResponseSurfaceCanvas


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent
OUTPUT_DIR = APP_DIR / "outputs"

TABLE_COLUMNS = [
    ("file_name", "文件名"),
    ("sample_id", "编号"),
    ("design_type", "设计"),
    ("f_khz", "f/kHz"),
    ("tau_ns", "τ/ns"),
    ("h_mm", "h/mm"),
    ("v_mms", "v/(mm/s)"),
    ("residual_area_percent", "R/%"),
    ("gray_diff_percent", "ΔG/%"),
    ("sample_gray_mean", "样品灰度"),
    ("base_gray_mean", "基材灰度"),
    ("roi", "ROI"),
    ("status", "状态"),
]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("激光清洗图片数据处理")
        self.resize(1480, 920)

        self.design = rsm.design_map()
        self.base_rgb: np.ndarray | None = None
        self.base_cache_path: Path | None = None
        self.current_sample_path: Path | None = None
        self.current_result: processing.ProcessResult | None = None
        self.current_roi: tuple[int, int, int, int] | None = None
        self.results: list[dict[str, object]] = []
        self.last_rsm_model: rsm.QuadraticModel | None = None

        self._build_ui()
        self._load_default_base()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)
        root_layout.addWidget(self._build_left_panel())
        root_layout.addWidget(self._build_right_panel(), 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("就绪")

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(410)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        file_group = QGroupBox("文件")
        file_layout = QGridLayout(file_group)
        self.base_path_edit = QLineEdit()
        self.base_path_edit.setPlaceholderText("选择基材图片")
        default_base = DATA_DIR / "4.Base.jpg"
        if default_base.exists():
            self.base_path_edit.setText(str(default_base))
        base_button = QPushButton("选择基材")
        base_button.clicked.connect(self.choose_base_image)
        self.sample_path_edit = QLineEdit()
        self.sample_path_edit.setPlaceholderText("选择单张样品图片")
        sample_button = QPushButton("选择单张")
        sample_button.clicked.connect(self.choose_sample_image)
        self.folder_path_edit = QLineEdit(str(DATA_DIR))
        folder_button = QPushButton("选择文件夹")
        folder_button.clicked.connect(self.choose_folder)
        file_layout.addWidget(QLabel("基材"), 0, 0)
        file_layout.addWidget(self.base_path_edit, 0, 1)
        file_layout.addWidget(base_button, 0, 2)
        file_layout.addWidget(QLabel("样品"), 1, 0)
        file_layout.addWidget(self.sample_path_edit, 1, 1)
        file_layout.addWidget(sample_button, 1, 2)
        file_layout.addWidget(QLabel("批量"), 2, 0)
        file_layout.addWidget(self.folder_path_edit, 2, 1)
        file_layout.addWidget(folder_button, 2, 2)
        content_layout.addWidget(file_group)

        process_group = QGroupBox("处理")
        process_layout = QVBoxLayout(process_group)
        self.process_current_button = QPushButton("处理当前图片")
        self.process_current_button.clicked.connect(self.process_current_image)
        self.process_batch_button = QPushButton("批量处理文件夹")
        self.process_batch_button.clicked.connect(self.process_batch_folder)
        self.export_images_check = QCheckBox("批量导出掩膜和叠加图")
        self.export_images_check.setChecked(True)
        self.export_table_button = QPushButton("导出结果表")
        self.export_table_button.clicked.connect(self.export_results_table)
        self.export_current_images_button = QPushButton("导出当前图像")
        self.export_current_images_button.clicked.connect(self.export_current_images)
        process_layout.addWidget(self.process_current_button)
        process_layout.addWidget(self.process_batch_button)
        process_layout.addWidget(self.export_images_check)
        process_layout.addWidget(self.export_table_button)
        process_layout.addWidget(self.export_current_images_button)
        content_layout.addWidget(process_group)

        view_group = QGroupBox("预览")
        view_layout = QFormLayout(view_group)
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("原图", "original")
        self.view_mode_combo.addItem("残留掩膜", "mask")
        self.view_mode_combo.addItem("叠加图", "overlay")
        self.view_mode_combo.currentIndexChanged.connect(self.refresh_preview)
        self.roi_button = QPushButton("框选ROI")
        self.roi_button.setCheckable(True)
        self.roi_button.toggled.connect(self.toggle_roi_mode)
        clear_roi_button = QPushButton("清除ROI")
        clear_roi_button.clicked.connect(self.clear_roi)
        fit_button = QPushButton("适配窗口")
        fit_button.clicked.connect(lambda: self.viewer.fit_to_view())
        self.roi_label = QLabel("全图")
        view_layout.addRow("显示", self.view_mode_combo)
        view_layout.addRow(self.roi_button, clear_roi_button)
        view_layout.addRow("ROI", self.roi_label)
        view_layout.addRow(fit_button)
        content_layout.addWidget(view_group)

        settings_group = QGroupBox("过滤参数")
        settings_layout = QFormLayout(settings_group)
        self.rgb_std_spin = self._double_spin(0.5, 8.0, 2.4, 0.1, 2)
        self.hue_tol_spin = self._double_spin(0.0, 90.0, 14.0, 1.0, 1)
        self.sv_std_spin = self._double_spin(0.5, 8.0, 2.5, 0.1, 2)
        self.lab_delta_spin = self._double_spin(1.0, 120.0, 28.0, 1.0, 1)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(0, 20000)
        self.min_area_spin.setValue(120)
        self.morph_kernel_spin = QSpinBox()
        self.morph_kernel_spin.setRange(1, 31)
        self.morph_kernel_spin.setSingleStep(2)
        self.morph_kernel_spin.setValue(3)
        self.votes_spin = QSpinBox()
        self.votes_spin.setRange(1, 3)
        self.votes_spin.setValue(2)
        self.overlay_alpha_spin = self._double_spin(0.0, 1.0, 0.45, 0.05, 2)
        settings_layout.addRow("RGB标准差倍数", self.rgb_std_spin)
        settings_layout.addRow("Hue容差", self.hue_tol_spin)
        settings_layout.addRow("SV标准差倍数", self.sv_std_spin)
        settings_layout.addRow("Lab ΔE阈值", self.lab_delta_spin)
        settings_layout.addRow("最小残留面积/px", self.min_area_spin)
        settings_layout.addRow("形态学核", self.morph_kernel_spin)
        settings_layout.addRow("基材判定票数", self.votes_spin)
        settings_layout.addRow("叠加透明度", self.overlay_alpha_spin)
        content_layout.addWidget(settings_group)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Vertical)
        self.viewer = ImageViewer()
        self.viewer.roiChanged.connect(self.on_roi_changed)
        splitter.addWidget(self.viewer)
        splitter.addWidget(self._build_bottom_tabs())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)
        return panel

    def _build_bottom_tabs(self) -> QWidget:
        tabs = QTabWidget()
        self.results_table = QTableWidget(0, len(TABLE_COLUMNS))
        self.results_table.setHorizontalHeaderLabels([label for _, label in TABLE_COLUMNS])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSortingEnabled(True)
        tabs.addTab(self.results_table, "结果表")
        tabs.addTab(self._build_rsm_tab(), "响应面")
        return tabs

    def _build_rsm_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QGridLayout()
        self.response_combo = QComboBox()
        for label in rsm.RESPONSE_COLUMNS:
            self.response_combo.addItem(label, label)
        self.x_factor_combo = QComboBox()
        self.y_factor_combo = QComboBox()
        for column in rsm.FACTOR_COLUMNS:
            self.x_factor_combo.addItem(rsm.FACTOR_LABELS[column], column)
            self.y_factor_combo.addItem(rsm.FACTOR_LABELS[column], column)
        self.y_factor_combo.setCurrentIndex(1)
        controls.addWidget(QLabel("响应"), 0, 0)
        controls.addWidget(self.response_combo, 0, 1)
        controls.addWidget(QLabel("X因子"), 0, 2)
        controls.addWidget(self.x_factor_combo, 0, 3)
        controls.addWidget(QLabel("Y因子"), 0, 4)
        controls.addWidget(self.y_factor_combo, 0, 5)

        self.fixed_spins: dict[str, QDoubleSpinBox] = {}
        for idx, column in enumerate(rsm.FACTOR_COLUMNS):
            low, high = rsm.FACTOR_RANGES[column]
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setValue((low + high) / 2.0)
            spin.setDecimals(4 if column == "h_mm" else 2)
            spin.setSingleStep(0.001 if column == "h_mm" else (25.0 if column == "tau_ns" else 10.0))
            self.fixed_spins[column] = spin
            controls.addWidget(QLabel(rsm.FACTOR_LABELS[column]), 1 + idx // 2, (idx % 2) * 2)
            controls.addWidget(spin, 1 + idx // 2, (idx % 2) * 2 + 1)

        fit_button = QPushButton("拟合并绘图")
        fit_button.clicked.connect(self.fit_and_plot_rsm)
        export_button = QPushButton("导出响应图")
        export_button.clicked.connect(self.export_rsm_figure)
        controls.addWidget(fit_button, 3, 4)
        controls.addWidget(export_button, 3, 5)
        layout.addLayout(controls)

        self.rsm_canvas = ResponseSurfaceCanvas()
        self.rsm_canvas.show_message("批量处理后可拟合响应面")
        self.rsm_info = QTextEdit()
        self.rsm_info.setReadOnly(True)
        self.rsm_info.setMinimumHeight(120)
        layout.addWidget(self.rsm_canvas, 1)
        layout.addWidget(self.rsm_info)
        return tab

    def _double_spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        return spin

    def _load_default_base(self) -> None:
        path = Path(self.base_path_edit.text().strip())
        if path.exists():
            try:
                self.load_base_image(path, display=True)
            except Exception as exc:
                self.statusBar().showMessage(f"默认基材加载失败: {exc}")

    def current_settings(self) -> processing.ProcessingSettings:
        return processing.ProcessingSettings(
            rgb_std_factor=self.rgb_std_spin.value(),
            hue_tolerance=self.hue_tol_spin.value(),
            sv_std_factor=self.sv_std_spin.value(),
            lab_delta_e=self.lab_delta_spin.value(),
            min_component_area=self.min_area_spin.value(),
            morph_kernel=self.morph_kernel_spin.value(),
            votes_required=self.votes_spin.value(),
            overlay_alpha=self.overlay_alpha_spin.value(),
        )

    def choose_base_image(self) -> None:
        start = self.base_path_edit.text().strip() or str(DATA_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择基材图片",
            start,
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if path:
            self.base_path_edit.setText(path)
            self.load_base_image(Path(path), display=True)

    def choose_sample_image(self) -> None:
        start = self.sample_path_edit.text().strip() or str(DATA_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择样品图片",
            start,
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if path:
            self.sample_path_edit.setText(path)
            self.load_sample_preview(Path(path))

    def choose_folder(self) -> None:
        start = self.folder_path_edit.text().strip() or str(DATA_DIR)
        path = QFileDialog.getExistingDirectory(self, "选择批量处理文件夹", start)
        if path:
            self.folder_path_edit.setText(path)

    def load_base_image(self, path: Path, display: bool = False) -> np.ndarray:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if self.base_rgb is None or self.base_cache_path != path:
            self.base_rgb = processing.read_rgb_image(path)
            self.base_cache_path = path
        if display and self.current_sample_path is None:
            self.viewer.set_image_array(self.base_rgb)
            self.statusBar().showMessage(f"已加载基材: {path.name}")
        return self.base_rgb

    def ensure_base_image(self) -> np.ndarray:
        text = self.base_path_edit.text().strip()
        if not text:
            raise ValueError("请先选择基材图片")
        return self.load_base_image(Path(text), display=False)

    def load_sample_preview(self, path: Path) -> None:
        self.current_sample_path = path
        sample = processing.read_rgb_image(path)
        if self.base_rgb is not None:
            sample = processing.resize_to_shape(sample, self.base_rgb.shape)
        self.current_result = None
        self.view_mode_combo.setCurrentIndex(0)
        self.viewer.set_image_array(sample)
        self.viewer.set_roi(self.current_roi)
        self.statusBar().showMessage(f"已加载样品: {path.name}")

    def toggle_roi_mode(self, enabled: bool) -> None:
        self.viewer.set_roi_mode(enabled)
        self.roi_button.setText("结束ROI框选" if enabled else "框选ROI")

    def on_roi_changed(self, roi: object) -> None:
        self.current_roi = roi if isinstance(roi, tuple) else None
        self.roi_label.setText(processing.roi_to_text(self.current_roi) if self.current_roi else "全图")

    def clear_roi(self) -> None:
        self.current_roi = None
        self.viewer.clear_roi()
        self.roi_label.setText("全图")

    def process_current_image(self) -> None:
        try:
            if self.current_sample_path is None:
                text = self.sample_path_edit.text().strip()
                if not text:
                    raise ValueError("请先选择样品图片")
                self.current_sample_path = Path(text)
            base_rgb = self.ensure_base_image()
            result = processing.process_sample(
                self.current_sample_path,
                base_rgb,
                self.current_roi,
                self.current_settings(),
            )
            self.current_result = result
            self._upsert_result(result)
            self.refresh_preview()
            self.statusBar().showMessage(
                f"{result.sample_path.name}: R={result.metrics['residual_area_percent']:.4f}%, "
                f"ΔG={result.metrics['gray_diff_percent']:.4f}%"
            )
        except Exception as exc:
            self.show_error("处理失败", exc)

    def process_batch_folder(self) -> None:
        try:
            folder = Path(self.folder_path_edit.text().strip() or DATA_DIR)
            if not folder.exists():
                raise FileNotFoundError(folder)
            base_rgb = self.ensure_base_image()
            sample_paths = processing.discover_images(folder, self.base_cache_path)
            if not sample_paths:
                raise ValueError("所选文件夹内没有可处理图片")
            settings = self.current_settings()
            model = processing.build_base_model(base_rgb, self.current_roi, settings)

            progress = QProgressDialog("正在批量处理...", "取消", 0, len(sample_paths), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)

            for index, path in enumerate(sample_paths, start=1):
                if progress.wasCanceled():
                    break
                progress.setValue(index - 1)
                progress.setLabelText(f"正在处理 {path.name}")
                QApplication.processEvents()
                result = processing.process_sample_with_model(path, model, settings)
                self._upsert_result(result)
                self.current_sample_path = path
                self.current_result = result
                if self.export_images_check.isChecked():
                    self._write_result_images(result)
            progress.setValue(len(sample_paths))
            self.refresh_preview()
            self.statusBar().showMessage(f"批量处理完成: {len(self.results)} 条结果")
        except Exception as exc:
            self.show_error("批量处理失败", exc)

    def _result_row(self, result: processing.ProcessResult) -> dict[str, object]:
        row = dict(result.metrics)
        row["path"] = str(result.sample_path)
        design = self.design.get(result.sample_id.upper())
        if design:
            row.update(design)
        else:
            row["design_type"] = ""
            for column in rsm.FACTOR_COLUMNS:
                row[column] = np.nan
        return row

    def _upsert_result(self, result: processing.ProcessResult) -> None:
        row = self._result_row(result)
        path = row["path"]
        for idx, old in enumerate(self.results):
            if old.get("path") == path:
                self.results[idx] = row
                self.update_results_table()
                return
        self.results.append(row)
        self.update_results_table()

    def update_results_table(self) -> None:
        self.results_table.setSortingEnabled(False)
        self.results_table.setRowCount(len(self.results))
        for row_idx, row in enumerate(self.results):
            for col_idx, (key, _) in enumerate(TABLE_COLUMNS):
                item = QTableWidgetItem(self._format_value(row.get(key), key))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.results_table.setItem(row_idx, col_idx, item)
        self.results_table.setSortingEnabled(True)

    def refresh_preview(self) -> None:
        if self.current_result is None:
            return
        mode = self.view_mode_combo.currentData()
        if mode == "mask":
            image = np.zeros_like(self.current_result.sample_rgb)
            image[self.current_result.residual_mask] = np.array([255, 255, 255], dtype=np.uint8)
        elif mode == "overlay":
            image = self.current_result.overlay_rgb
        else:
            image = self.current_result.sample_rgb
        self.viewer.set_image_array(image, reset_view=False)
        self.viewer.set_roi(self.current_roi)

    def export_results_table(self) -> None:
        try:
            if not self.results:
                raise ValueError("当前没有可导出的结果")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            default = OUTPUT_DIR / f"results_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
            path, _ = QFileDialog.getSaveFileName(
                self,
                "导出结果表",
                str(default),
                "Excel (*.xlsx);;CSV (*.csv)",
            )
            if not path:
                return
            df = pd.DataFrame(self.results)
            ordered = [key for key, _ in TABLE_COLUMNS if key in df.columns]
            rest = [column for column in df.columns if column not in ordered]
            df = df[ordered + rest]
            if Path(path).suffix.lower() == ".csv":
                df.to_csv(path, index=False, encoding="utf-8-sig")
            else:
                df.to_excel(path, index=False)
            self.statusBar().showMessage(f"已导出结果表: {path}")
        except Exception as exc:
            self.show_error("导出失败", exc)

    def export_current_images(self) -> None:
        try:
            if self.current_result is None:
                raise ValueError("当前没有已处理图片")
            self._write_result_images(self.current_result)
            self.statusBar().showMessage("已导出当前掩膜和叠加图")
        except Exception as exc:
            self.show_error("导出图像失败", exc)

    def _write_result_images(self, result: processing.ProcessResult) -> None:
        mask_dir = OUTPUT_DIR / "masks"
        overlay_dir = OUTPUT_DIR / "overlays"
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        stem = result.sample_path.stem
        processing.write_mask_image(mask_dir / f"{stem}_mask.png", result.residual_mask)
        processing.write_rgb_image(overlay_dir / f"{stem}_overlay.png", result.overlay_rgb)

    def fit_and_plot_rsm(self) -> None:
        try:
            if not self.results:
                raise ValueError("请先完成批量处理")
            response_name = self.response_combo.currentData()
            x_factor = self.x_factor_combo.currentData()
            y_factor = self.y_factor_combo.currentData()
            if x_factor == y_factor:
                raise ValueError("X因子和Y因子不能相同")
            data = pd.DataFrame(self.results)
            model = rsm.fit_quadratic_model(data, response_name)
            fixed_values = {column: spin.value() for column, spin in self.fixed_spins.items()}
            self.rsm_canvas.plot_surface(model, x_factor, y_factor, fixed_values)
            self.last_rsm_model = model
            self.rsm_info.setPlainText(self._rsm_report(data, model))
        except Exception as exc:
            self.show_error("响应面分析失败", exc)

    def _rsm_report(self, data: pd.DataFrame, model: rsm.QuadraticModel) -> str:
        lines = [
            f"{model.response_name} 二次模型",
            f"样本数: {model.n_samples}, 矩阵秩: {model.rank}, R²: {model.r2:.5f}, RMSE: {model.rmse:.5f}",
            "",
            "最小值预测:",
        ]
        params, value = rsm.optimize_model(model)
        lines.append(f"{model.response_name} 最小: {value:.5f}; {rsm.format_params(params)}")

        try:
            residual_model = rsm.fit_quadratic_model(data, "R/%")
            gray_model = rsm.fit_quadratic_model(data, "ΔG/%")
            c_params, score, r_value, g_value = rsm.optimize_combined(residual_model, gray_model)
            lines.append(
                f"综合目标最小: score={score:.5f}, R={r_value:.5f}%, ΔG={g_value:.5f}%; "
                f"{rsm.format_params(c_params)}"
            )
        except Exception as exc:
            lines.append(f"综合目标暂不可用: {exc}")

        lines.append("")
        lines.append("系数:")
        for name, coef in zip(model.coefficient_names, model.coefficients, strict=False):
            lines.append(f"{name}: {coef:.8g}")
        return "\n".join(lines)

    def export_rsm_figure(self) -> None:
        try:
            if self.last_rsm_model is None:
                raise ValueError("请先拟合响应面")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            default = OUTPUT_DIR / f"rsm_{self.last_rsm_model.response_column}_{datetime.now():%Y%m%d_%H%M%S}.png"
            path, _ = QFileDialog.getSaveFileName(self, "导出响应图", str(default), "PNG (*.png);;PDF (*.pdf)")
            if not path:
                return
            self.rsm_canvas.figure.savefig(path, dpi=180, bbox_inches="tight")
            self.statusBar().showMessage(f"已导出响应图: {path}")
        except Exception as exc:
            self.show_error("导出响应图失败", exc)

    def show_error(self, title: str, exc: Exception) -> None:
        QMessageBox.critical(self, title, str(exc))
        self.statusBar().showMessage(f"{title}: {exc}")

    def _format_value(self, value: object, key: str) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except TypeError:
            pass
        if isinstance(value, float):
            if key == "h_mm":
                return f"{value:.5f}"
            if key in {"residual_area_percent", "gray_diff_percent"}:
                return f"{value:.4f}"
            return f"{value:.3f}"
        return str(value)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("激光清洗图片数据处理")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
