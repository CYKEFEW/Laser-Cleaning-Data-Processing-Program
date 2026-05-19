from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QFont
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
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QScrollArea,
    QSlider,
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
PRESETS_PATH = APP_DIR / "presets.json"
CALC_DESIGN_TEMPLATE_PATH = APP_DIR / "图片计算-实验参数表.xlsx"

BUILTIN_PRESETS = {
    "图示推荐参数": {
        "algorithm_mode": "score",
        "sensitivity": 73.0,
        "local_background_sigma": 42.0,
        "residual_color_boost": 1.65,
        "min_component_area": 2480,
        "morph_kernel": 3,
        "overlay_alpha": 0.27,
        "auto_preview": True,
    }
}

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
    ("algorithm_mode", "算法"),
    ("sensitivity", "灵敏度"),
    ("score_threshold", "评分阈值"),
    ("base_color_samples", "基材取色"),
    ("residual_color_samples", "残留取色"),
    ("calc_param_table", "计算参数表"),
    ("param_match_status", "参数匹配"),
    ("roi", "ROI"),
    ("status", "状态"),
]


class SliderControl(QWidget):
    valueChanged = Signal(float)
    sliderReleased = Signal()

    def __init__(
        self,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int = 2,
        suffix: str = "",
        integer: bool = False,
        odd: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._step = float(step)
        self._decimals = int(decimals)
        self._suffix = suffix
        self._integer = integer
        self._odd = odd

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(round((self._maximum - self._minimum) / self._step)))
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(72)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

        self.slider.valueChanged.connect(self._on_slider_value_changed)
        self.slider.sliderReleased.connect(self.sliderReleased)
        self.set_value(value)

    def value(self) -> float | int:
        value = self._minimum + self.slider.value() * self._step
        value = min(max(value, self._minimum), self._maximum)
        if self._odd:
            value = int(round(value))
            if value % 2 == 0:
                value += 1 if value < self._maximum else -1
            value = min(max(value, int(self._minimum)), int(self._maximum))
        if self._integer:
            return int(round(value))
        return round(float(value), self._decimals)

    def set_value(self, value: float) -> None:
        raw = int(round((float(value) - self._minimum) / self._step))
        self.slider.setValue(min(max(raw, self.slider.minimum()), self.slider.maximum()))
        self._update_label()

    def _on_slider_value_changed(self) -> None:
        self._update_label()
        self.valueChanged.emit(float(self.value()))

    def _update_label(self) -> None:
        value = self.value()
        if self._integer:
            text = f"{int(value)}{self._suffix}"
        else:
            text = f"{float(value):.{self._decimals}f}{self._suffix}"
        self.value_label.setText(text)


def _compact_column_name(value: object) -> str:
    text = str(value).strip().lower()
    remove = " _-/（）()[]【】%·.：:，,;；"
    for char in remove:
        text = text.replace(char, "")
    return text


DESIGN_COLUMN_ALIASES = {
    "sample_id": ["sample_id", "实验编号", "编号", "样品编号", "试验编号"],
    "design_type": ["design_type", "设计类型", "类型"],
    "f_khz": ["f_khz", "频率f/kHz", "频率", "频率f", "fkHz"],
    "tau_ns": ["tau_ns", "脉宽τ/ns", "脉宽", "脉宽tau", "tauns"],
    "h_mm": ["h_mm", "线间距h/mm", "线间距", "线间距h", "hmm"],
    "v_mms": ["v_mms", "扫描速度v/(mm/s)", "扫描速度", "速度", "vmms"],
}


def normalize_design_table(data: pd.DataFrame, table_name: str = "实验编号-参数表") -> pd.DataFrame:
    if data.empty:
        raise ValueError(f"{table_name}为空")
    compact_map = {_compact_column_name(column): column for column in data.columns}
    rename: dict[object, str] = {}
    for canonical, aliases in DESIGN_COLUMN_ALIASES.items():
        for alias in aliases:
            match = compact_map.get(_compact_column_name(alias))
            if match is not None:
                rename[match] = canonical
                break
    required = ["sample_id", *rsm.FACTOR_COLUMNS]
    missing = [column for column in required if column not in rename.values()]
    if missing:
        raise ValueError(f"{table_name}缺少列: {', '.join(missing)}")

    normalized = data.rename(columns=rename).copy()
    keep = ["sample_id", "design_type", *rsm.FACTOR_COLUMNS]
    for column in keep:
        if column not in normalized.columns:
            normalized[column] = ""
    normalized = normalized[keep]
    normalized["sample_id"] = normalized["sample_id"].astype(str).str.strip().str.upper()
    normalized = normalized[normalized["sample_id"] != ""].copy()
    for column in rsm.FACTOR_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=rsm.FACTOR_COLUMNS)
    normalized = normalized.drop_duplicates(subset=["sample_id"], keep="first")
    if normalized.empty:
        raise ValueError(f"{table_name}没有有效参数行")
    return normalized


class RsmAnalysisWindow(QMainWindow):
    def __init__(self, owner: "MainWindow") -> None:
        super().__init__(owner)
        self.owner = owner
        self.design_df: pd.DataFrame | None = None
        self.design_path: Path | None = None
        self.last_rsm_model: rsm.QuadraticModel | None = None
        self.setWindowTitle("响应面分析")
        self.resize(1180, 780)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        import_layout = QHBoxLayout()
        self.design_path_label = QLabel("未导入实验编号-参数表")
        import_button = QPushButton("导入实验编号-参数表")
        import_button.clicked.connect(self.import_design_table)
        import_layout.addWidget(import_button)
        import_layout.addWidget(self.design_path_label, 1)
        layout.addLayout(import_layout)

        controls = QGridLayout()
        self.language_combo = QComboBox()
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("English", "en")
        self.language_combo.currentIndexChanged.connect(self.refresh_rsm_language)
        self.response_combo = QComboBox()
        for label in rsm.RESPONSE_COLUMNS:
            self.response_combo.addItem(label, label)
        self.x_factor_combo = QComboBox()
        self.y_factor_combo = QComboBox()
        for column in rsm.FACTOR_COLUMNS:
            self.x_factor_combo.addItem(rsm.FACTOR_LABELS[column], column)
            self.y_factor_combo.addItem(rsm.FACTOR_LABELS[column], column)
        self.y_factor_combo.setCurrentIndex(1)
        controls.addWidget(QLabel("语言"), 0, 0)
        controls.addWidget(self.language_combo, 0, 1)
        controls.addWidget(QLabel("响应"), 0, 2)
        controls.addWidget(self.response_combo, 0, 3)
        controls.addWidget(QLabel("X因子"), 1, 0)
        controls.addWidget(self.x_factor_combo, 1, 1)
        controls.addWidget(QLabel("Y因子"), 1, 2)
        controls.addWidget(self.y_factor_combo, 1, 3)

        self.fixed_spins: dict[str, QDoubleSpinBox] = {}
        for idx, column in enumerate(rsm.FACTOR_COLUMNS):
            low, high = rsm.FACTOR_RANGES[column]
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setValue((low + high) / 2.0)
            spin.setDecimals(4 if column == "h_mm" else 2)
            spin.setSingleStep(0.001 if column == "h_mm" else (25.0 if column == "tau_ns" else 10.0))
            self.fixed_spins[column] = spin
            controls.addWidget(QLabel(rsm.FACTOR_LABELS[column]), 2 + idx // 2, (idx % 2) * 2)
            controls.addWidget(spin, 2 + idx // 2, (idx % 2) * 2 + 1)

        fit_button = QPushButton("拟合并绘图")
        fit_button.clicked.connect(self.fit_and_plot_rsm)
        export_button = QPushButton("导出响应图")
        export_button.clicked.connect(self.export_rsm_figure)
        batch_export_button = QPushButton("批量导出所有响应图")
        batch_export_button.clicked.connect(self.batch_export_rsm_figures)
        controls.addWidget(fit_button, 4, 2)
        controls.addWidget(export_button, 4, 3)
        controls.addWidget(batch_export_button, 5, 2, 1, 2)
        layout.addLayout(controls)

        self.rsm_canvas = ResponseSurfaceCanvas()
        self.rsm_canvas.show_message("请先导入实验编号-参数表，再拟合响应面")
        self.rsm_info = QTextEdit()
        self.rsm_info.setReadOnly(True)
        self.rsm_info.setFont(QFont("Microsoft YaHei", 10))
        self.rsm_info.setMinimumHeight(140)
        layout.addWidget(self.rsm_canvas, 1)
        layout.addWidget(self.rsm_info)
        self.setCentralWidget(root)

    def import_design_table(self) -> None:
        start = str(APP_DIR / "实验编号-参数表.xlsx")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入实验编号-参数表",
            start,
            "Excel (*.xlsx *.xls)",
        )
        if not path:
            return
        try:
            raw = pd.read_excel(path)
            self.design_df = self.normalize_design_table(raw)
            self.design_path = Path(path)
            self.design_path_label.setText(f"{Path(path).name}，{len(self.design_df)} 条参数")
            self.rsm_info.setPlainText(f"已导入实验编号-参数表: {path}\n参数行数: {len(self.design_df)}")
        except Exception as exc:
            self.show_error("导入实验编号-参数表失败", exc)

    def normalize_design_table(self, data: pd.DataFrame) -> pd.DataFrame:
        return normalize_design_table(data, "响应面实验编号-参数表")

    def current_language(self) -> str:
        return self.language_combo.currentData() or "zh"

    def refresh_rsm_language(self) -> None:
        language = self.current_language()
        self.rsm_info.setFont(QFont("Times New Roman" if language == "en" else "Microsoft YaHei", 10))
        if self.last_rsm_model is not None:
            try:
                data = self.merged_analysis_data()
                fixed_values = {column: spin.value() for column, spin in self.fixed_spins.items()}
                self.rsm_canvas.plot_surface(
                    self.last_rsm_model,
                    self.x_factor_combo.currentData(),
                    self.y_factor_combo.currentData(),
                    fixed_values,
                    language=language,
                )
                self.rsm_info.setPlainText(self._rsm_report(data, self.last_rsm_model))
            except Exception:
                pass

    def merged_analysis_data(self) -> pd.DataFrame:
        if self.design_df is None:
            raise ValueError("请先导入实验编号-参数表 xlsx")
        if not self.owner.results:
            raise ValueError("请先在主窗口完成单张或批量图片处理")
        results = pd.DataFrame(self.owner.results).copy()
        if "sample_id" not in results.columns:
            raise ValueError("当前结果表缺少实验编号 sample_id")
        results["sample_id"] = results["sample_id"].astype(str).str.strip().str.upper()
        drop_columns = [column for column in ["design_type", *rsm.FACTOR_COLUMNS] if column in results.columns]
        results = results.drop(columns=drop_columns)
        merged = results.merge(self.design_df, on="sample_id", how="left")
        matched = merged.dropna(subset=rsm.FACTOR_COLUMNS)
        if matched.empty:
            raise ValueError("当前结果表中的编号没有匹配到导入参数表")
        return merged

    def fit_and_plot_rsm(self) -> None:
        try:
            response_name = self.response_combo.currentData()
            x_factor = self.x_factor_combo.currentData()
            y_factor = self.y_factor_combo.currentData()
            if x_factor == y_factor:
                raise ValueError("X因子和Y因子不能相同")
            data = self.merged_analysis_data()
            model = rsm.fit_quadratic_model(data, response_name)
            fixed_values = {column: spin.value() for column, spin in self.fixed_spins.items()}
            self.rsm_canvas.plot_surface(model, x_factor, y_factor, fixed_values, language=self.current_language())
            self.last_rsm_model = model
            self.rsm_info.setPlainText(self._rsm_report(data, model))
        except Exception as exc:
            self.show_error("响应面分析失败", exc)

    def _rsm_report(self, data: pd.DataFrame, model: rsm.QuadraticModel) -> str:
        matched = data.dropna(subset=rsm.FACTOR_COLUMNS)
        language = self.current_language()
        response_label = {
            "R/%": "Residual area R/%" if language == "en" else "R/% 残留面积率",
            "ΔG/%": "Gray difference ΔG/%" if language == "en" else "ΔG/% 灰度差",
            "gray_diff_percent": "Gray difference ΔG/%" if language == "en" else "ΔG/% 灰度差",
            "residual_area_percent": "Residual area R/%" if language == "en" else "R/% 残留面积率",
        }.get(model.response_name, model.response_name)
        table_name = self.design_path.name if self.design_path else ("Unnamed" if language == "en" else "未命名")
        if language == "en":
            lines = [
                f"{response_label} quadratic model",
                f"Parameter table: {table_name}",
                f"Matched samples: {len(matched)}, model samples: {model.n_samples}, matrix rank: {model.rank}",
                f"R²: {model.r2:.5f}, RMSE: {model.rmse:.5f}",
                "",
                "Minimum prediction:",
            ]
        else:
            lines = [
                f"{response_label} 二次模型",
                f"参数表: {table_name}",
                f"匹配样本数: {len(matched)}, 模型样本数: {model.n_samples}, 矩阵秩: {model.rank}",
                f"R²: {model.r2:.5f}, RMSE: {model.rmse:.5f}",
                "",
                "最小值预测:",
            ]
        params, value = rsm.optimize_model(model)
        if language == "en":
            lines.append(f"{response_label} minimum: {value:.5f}; {rsm.format_params(params)}")
        else:
            lines.append(f"{response_label} 最小: {value:.5f}; {rsm.format_params(params)}")
        try:
            residual_model = rsm.fit_quadratic_model(data, "R/%")
            gray_model = rsm.fit_quadratic_model(data, "ΔG/%")
            c_params, score, r_value, g_value = rsm.optimize_combined(residual_model, gray_model)
            if language == "en":
                lines.append(
                    f"Combined objective minimum: score={score:.5f}, R={r_value:.5f}%, ΔG={g_value:.5f}%; "
                    f"{rsm.format_params(c_params)}"
                )
            else:
                lines.append(
                    f"综合目标最小: score={score:.5f}, R={r_value:.5f}%, ΔG={g_value:.5f}%; "
                    f"{rsm.format_params(c_params)}"
                )
        except Exception as exc:
            lines.append(f"Combined objective unavailable: {exc}" if language == "en" else f"综合目标暂不可用: {exc}")
        lines.append("")
        lines.append("Coefficients:" if language == "en" else "系数:")
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

    def batch_export_rsm_figures(self) -> None:
        """批量导出所有响应变量 × 所有因子对组合的响应面图。

        遍历 R/% 和 ΔG/% 两个响应，以及 4 个因子中所有 C(4,2)=6 种 X/Y 因子对，
        其余两个因子固定在当前界面填写的固定值，共生成最多 12 张图。
        """
        from itertools import combinations as _combinations

        try:
            data = self.merged_analysis_data()
        except Exception as exc:
            self.show_error("批量导出失败", exc)
            return

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "选择批量导出目录",
            str(OUTPUT_DIR),
        )
        if not out_dir:
            return

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        language = self.current_language()
        fixed_values = {column: spin.value() for column, spin in self.fixed_spins.items()}
        factor_pairs = list(_combinations(rsm.FACTOR_COLUMNS, 2))
        response_names = list(rsm.RESPONSE_COLUMNS.keys())  # ["R/%", "ΔG/%"]

        # 预先拟合两个模型，避免重复拟合
        models: dict[str, rsm.QuadraticModel] = {}
        for resp in response_names:
            try:
                models[resp] = rsm.fit_quadratic_model(data, resp)
            except Exception as exc:
                self.show_error(f"拟合 {resp} 模型失败", exc)
                return

        total = len(response_names) * len(factor_pairs)
        progress = QProgressDialog("正在批量导出响应图...", "取消", 0, total, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        count = 0
        errors: list[str] = []

        for resp in response_names:
            model = models[resp]
            resp_slug = model.response_column  # "residual_area_percent" / "gray_diff_percent"
            for x_factor, y_factor in factor_pairs:
                if progress.wasCanceled():
                    break
                progress.setValue(count)
                progress.setLabelText(
                    f"正在绘制 {resp}  {rsm.FACTOR_LABELS[x_factor]} × {rsm.FACTOR_LABELS[y_factor]}"
                )
                QApplication.processEvents()

                try:
                    # 用独立画布绘图，不影响当前显示的图
                    canvas = ResponseSurfaceCanvas()
                    canvas.plot_surface(model, x_factor, y_factor, fixed_values, language=language)

                    fname = f"rsm_{resp_slug}_{x_factor}_vs_{y_factor}_{timestamp}.png"
                    save_path = out_path / fname
                    canvas.figure.savefig(str(save_path), dpi=180, bbox_inches="tight")
                    canvas.figure.clf()
                except Exception as exc:
                    errors.append(f"{resp} {x_factor}×{y_factor}: {exc}")

                count += 1

            if progress.wasCanceled():
                break

        progress.setValue(total)

        if errors:
            self.show_error(
                "部分图导出失败",
                Exception("\n".join(errors)),
            )
        else:
            msg = f"已批量导出 {count} 张响应图到:\n{out_path}"
            QMessageBox.information(self, "批量导出完成", msg)
        self.statusBar().showMessage(f"批量导出完成: {count} 张图 → {out_path}")

    def show_error(self, title: str, exc: Exception) -> None:
        QMessageBox.critical(self, title, str(exc))
        self.statusBar().showMessage(f"{title}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("激光清洗图片数据处理")
        self.resize(1480, 920)

        self.calc_design_df: pd.DataFrame | None = None
        self.calc_design_path: Path | None = None
        self.base_rgb: np.ndarray | None = None
        self.base_cache_path: Path | None = None
        self.current_sample_path: Path | None = None
        self.current_result: processing.ProcessResult | None = None
        self.current_roi: tuple[int, int, int, int] | None = None
        self.color_pick_target = "base"
        self.base_color_samples: list[dict[str, object]] = []
        self.residual_color_samples: list[dict[str, object]] = []
        self.user_presets = self.load_user_presets()
        self.results: list[dict[str, object]] = []
        self.last_rsm_model: rsm.QuadraticModel | None = None
        self.rsm_window: RsmAnalysisWindow | None = None

        self._build_ui()
        self._load_default_base()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(0)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self._build_left_panel())
        self.main_splitter.addWidget(self._build_right_panel())
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([620, 1240])
        root_layout.addWidget(self.main_splitter)
        self.setCentralWidget(root)
        self._build_menu_bar()
        self.statusBar().showMessage("就绪")

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        choose_base_action = QAction("选择基材图片", self)
        choose_base_action.triggered.connect(self.choose_base_image)
        choose_sample_action = QAction("选择单张样品", self)
        choose_sample_action.triggered.connect(self.choose_sample_image)
        choose_folder_action = QAction("选择批量文件夹", self)
        choose_folder_action.triggered.connect(self.choose_folder)
        import_calc_design_action = QAction("导入图片计算参数表", self)
        import_calc_design_action.triggered.connect(self.import_calc_design_table)
        export_action = QAction("导出结果表", self)
        export_action.triggered.connect(self.export_results_table)
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(choose_base_action)
        file_menu.addAction(choose_sample_action)
        file_menu.addAction(choose_folder_action)
        file_menu.addAction(import_calc_design_action)
        file_menu.addSeparator()
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        process_menu = self.menuBar().addMenu("处理")
        process_current_action = QAction("处理当前图片", self)
        process_current_action.triggered.connect(self.process_current_image)
        process_batch_action = QAction("批量处理文件夹", self)
        process_batch_action.triggered.connect(self.process_batch_folder)
        process_menu.addAction(process_current_action)
        process_menu.addAction(process_batch_action)

        analysis_menu = self.menuBar().addMenu("分析")
        open_rsm_action = QAction("响应面分析", self)
        open_rsm_action.triggered.connect(self.open_rsm_window)
        analysis_menu.addAction(open_rsm_action)

    def open_rsm_window(self) -> None:
        if self.rsm_window is None:
            self.rsm_window = RsmAnalysisWindow(self)
        self.rsm_window.show()
        self.rsm_window.raise_()
        self.rsm_window.activateWindow()

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(520)
        content = QWidget()
        content.setMinimumWidth(500)
        content_layout = QVBoxLayout(content)

        file_group = QGroupBox("文件")
        file_layout = QGridLayout(file_group)
        self.base_path_edit = QLineEdit()
        self.base_path_edit.setMinimumWidth(320)
        self.base_path_edit.setPlaceholderText("选择基材图片")
        default_base = DATA_DIR / "4.Base.jpg"
        if default_base.exists():
            self.base_path_edit.setText(str(default_base))
            self.base_path_edit.setToolTip(str(default_base))
        base_button = QPushButton("选择基材")
        base_button.setMinimumWidth(96)
        base_button.clicked.connect(self.choose_base_image)
        self.sample_path_edit = QLineEdit()
        self.sample_path_edit.setMinimumWidth(320)
        self.sample_path_edit.setPlaceholderText("选择单张样品图片")
        sample_button = QPushButton("选择单张")
        sample_button.setMinimumWidth(96)
        sample_button.clicked.connect(self.choose_sample_image)
        self.folder_path_edit = QLineEdit(str(DATA_DIR))
        self.folder_path_edit.setMinimumWidth(320)
        self.folder_path_edit.setToolTip(str(DATA_DIR))
        folder_button = QPushButton("选择文件夹")
        folder_button.setMinimumWidth(96)
        folder_button.clicked.connect(self.choose_folder)
        file_layout.setColumnStretch(1, 1)
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

        calc_param_group = QGroupBox("图片计算参数表")
        calc_param_layout = QVBoxLayout(calc_param_group)
        self.calc_design_label = QLabel("未导入；结果表参数列将为空")
        self.calc_design_label.setWordWrap(True)
        import_calc_button = QPushButton("导入图片计算参数表")
        import_calc_button.clicked.connect(self.import_calc_design_table)
        calc_param_layout.addWidget(self.calc_design_label)
        calc_param_layout.addWidget(import_calc_button)
        content_layout.addWidget(calc_param_group)

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
        self.view_mode_combo.addItem("残留评分热力图", "heatmap")
        self.view_mode_combo.addItem("残留掩膜", "mask")
        self.view_mode_combo.addItem("叠加图", "overlay")
        self.view_mode_combo.currentIndexChanged.connect(self.refresh_preview)
        self.roi_button = QPushButton("框选ROI")
        self.roi_button.setCheckable(True)
        self.roi_button.toggled.connect(self.toggle_roi_mode)
        self.color_picker_button = QPushButton("取基材色")
        self.color_picker_button.setCheckable(True)
        self.color_picker_button.toggled.connect(self.toggle_base_color_picker_mode)
        clear_base_button = QPushButton("清除基材色")
        clear_base_button.clicked.connect(self.clear_base_colors)
        self.residual_picker_button = QPushButton("取残留色")
        self.residual_picker_button.setCheckable(True)
        self.residual_picker_button.toggled.connect(self.toggle_residual_color_picker_mode)
        clear_residual_button = QPushButton("清除残留色")
        clear_residual_button.clicked.connect(self.clear_residual_colors)
        clear_roi_button = QPushButton("清除ROI")
        clear_roi_button.clicked.connect(self.clear_roi)
        fit_button = QPushButton("适配窗口")
        fit_button.clicked.connect(lambda: self.viewer.fit_to_view())
        self.roi_label = QLabel("全图")
        self.base_color_label = QLabel("基材色: 0")
        self.residual_color_label = QLabel("残留色: 0")
        view_layout.addRow("显示", self.view_mode_combo)
        view_layout.addRow(self.roi_button, clear_roi_button)
        view_layout.addRow(self.color_picker_button, clear_base_button)
        view_layout.addRow(self.base_color_label)
        view_layout.addRow(self.residual_picker_button, clear_residual_button)
        view_layout.addRow(self.residual_color_label)
        view_layout.addRow("ROI", self.roi_label)
        view_layout.addRow(fit_button)
        content_layout.addWidget(view_group)

        settings_group = QGroupBox("算法参数")
        settings_layout = QFormLayout(settings_group)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem("残留评分算法(推荐)", "score")
        self.algorithm_combo.addItem("旧版多通道过滤", "legacy")
        self.algorithm_combo.currentIndexChanged.connect(lambda: self.preview_filter_effect(force=False))
        self.sensitivity_spin = SliderControl(0.0, 100.0, 73.0, 1.0, 0)
        self.background_sigma_spin = SliderControl(5.0, 80.0, 42.0, 1.0, 0, " px")
        self.residual_boost_spin = SliderControl(0.0, 2.0, 1.65, 0.05, 2)
        self.min_area_spin = SliderControl(0, 20000, 2480, 20, 0, " px", integer=True)
        self.morph_kernel_spin = SliderControl(1, 31, 3, 2, 0, " px", integer=True, odd=True)
        self.overlay_alpha_spin = SliderControl(0.0, 1.0, 0.27, 0.01, 2)
        self.auto_preview_check = QCheckBox("拖动滑块后自动更新过滤预览")
        self.auto_preview_check.setChecked(True)
        self.preview_filter_button = QPushButton("更新过滤预览")
        self.preview_filter_button.clicked.connect(lambda: self.preview_filter_effect(force=True))
        for control in self.filter_controls():
            control.sliderReleased.connect(lambda: self.preview_filter_effect(force=False))
        preset_widget = QWidget()
        preset_layout = QHBoxLayout(preset_widget)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        self.preset_combo = QComboBox()
        self.refresh_preset_combo()
        load_preset_button = QPushButton("加载")
        load_preset_button.clicked.connect(self.load_selected_preset)
        save_preset_button = QPushButton("保存")
        save_preset_button.clicked.connect(self.save_current_preset)
        delete_preset_button = QPushButton("删除")
        delete_preset_button.clicked.connect(self.delete_selected_preset)
        preset_layout.addWidget(self.preset_combo, 1)
        preset_layout.addWidget(load_preset_button)
        preset_layout.addWidget(save_preset_button)
        preset_layout.addWidget(delete_preset_button)
        settings_layout.addRow("参数预设", preset_widget)
        settings_layout.addRow("算法模式", self.algorithm_combo)
        settings_layout.addRow("灵敏度", self.sensitivity_spin)
        settings_layout.addRow("局部背景尺度", self.background_sigma_spin)
        settings_layout.addRow("残留色增强", self.residual_boost_spin)
        settings_layout.addRow("最小残留面积/px", self.min_area_spin)
        settings_layout.addRow("形态学核", self.morph_kernel_spin)
        settings_layout.addRow("叠加透明度", self.overlay_alpha_spin)
        settings_layout.addRow(self.auto_preview_check)
        settings_layout.addRow(self.preview_filter_button)
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
        self.viewer.colorPicked.connect(self.pick_base_color)
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
        self.results_table.cellDoubleClicked.connect(self.on_result_row_double_clicked)
        tabs.addTab(self.results_table, "结果表")
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

    def load_user_presets(self) -> dict[str, dict[str, object]]:
        if not PRESETS_PATH.exists():
            return {}
        try:
            with PRESETS_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return {str(name): dict(value) for name, value in data.items() if isinstance(value, dict)}
        except Exception:
            return {}
        return {}

    def save_user_presets(self) -> None:
        PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PRESETS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(self.user_presets, fh, ensure_ascii=False, indent=2)

    def refresh_preset_combo(self, selected_name: str | None = None) -> None:
        self.preset_combo.clear()
        for name in BUILTIN_PRESETS:
            self.preset_combo.addItem(f"{name} (内置)", ("builtin", name))
        for name in sorted(self.user_presets):
            self.preset_combo.addItem(name, ("user", name))
        if selected_name:
            for index in range(self.preset_combo.count()):
                source, name = self.preset_combo.itemData(index)
                if name == selected_name:
                    self.preset_combo.setCurrentIndex(index)
                    break

    def current_preset_data(self) -> dict[str, object]:
        return {
            "algorithm_mode": self.algorithm_combo.currentData(),
            "sensitivity": float(self.sensitivity_spin.value()),
            "local_background_sigma": float(self.background_sigma_spin.value()),
            "residual_color_boost": float(self.residual_boost_spin.value()),
            "min_component_area": int(self.min_area_spin.value()),
            "morph_kernel": int(self.morph_kernel_spin.value()),
            "overlay_alpha": float(self.overlay_alpha_spin.value()),
            "auto_preview": bool(self.auto_preview_check.isChecked()),
            "base_color_samples": self.base_color_samples,
            "residual_color_samples": self.residual_color_samples,
        }

    def apply_preset(self, preset: dict[str, object], refresh: bool = True) -> None:
        algorithm = preset.get("algorithm_mode", "score")
        for index in range(self.algorithm_combo.count()):
            if self.algorithm_combo.itemData(index) == algorithm:
                self.algorithm_combo.setCurrentIndex(index)
                break
        self.sensitivity_spin.set_value(float(preset.get("sensitivity", self.sensitivity_spin.value())))
        self.background_sigma_spin.set_value(float(preset.get("local_background_sigma", self.background_sigma_spin.value())))
        self.residual_boost_spin.set_value(float(preset.get("residual_color_boost", self.residual_boost_spin.value())))
        self.min_area_spin.set_value(float(preset.get("min_component_area", self.min_area_spin.value())))
        self.morph_kernel_spin.set_value(float(preset.get("morph_kernel", self.morph_kernel_spin.value())))
        self.overlay_alpha_spin.set_value(float(preset.get("overlay_alpha", self.overlay_alpha_spin.value())))
        self.auto_preview_check.setChecked(bool(preset.get("auto_preview", self.auto_preview_check.isChecked())))
        self.base_color_samples = self._normalize_color_samples(preset.get("base_color_samples", []))
        self.residual_color_samples = self._normalize_color_samples(preset.get("residual_color_samples", []))
        self.update_color_sample_labels()
        if refresh:
            self.preview_filter_effect(force=True)

    def _normalize_color_samples(self, value: object) -> list[dict[str, object]]:
        if not isinstance(value, list):
            return []
        normalized = []
        for item in value:
            if not isinstance(item, dict):
                continue
            try:
                normalized.append(
                    {
                        "rgb": tuple(float(v) for v in item["rgb"]),
                        "hue": float(item["hue"]),
                        "sv": tuple(float(v) for v in item["sv"]),
                        "lab": tuple(float(v) for v in item["lab"]),
                        "rgb_tol": float(item.get("rgb_tol", 34.0)),
                        "hue_tol": float(item.get("hue_tol", 12.0)),
                        "sv_tol": float(item.get("sv_tol", 42.0)),
                        "lab_tol": float(item.get("lab_tol", 32.0)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return normalized

    def update_color_sample_labels(self) -> None:
        self.base_color_label.setText(f"基材色: {len(self.base_color_samples)}")
        self.residual_color_label.setText(f"残留色: {len(self.residual_color_samples)}")

    def selected_preset(self) -> tuple[str, str, dict[str, object]] | None:
        data = self.preset_combo.currentData()
        if not data:
            return None
        source, name = data
        presets = BUILTIN_PRESETS if source == "builtin" else self.user_presets
        preset = presets.get(name)
        if preset is None:
            return None
        return source, name, dict(preset)

    def load_selected_preset(self) -> None:
        selected = self.selected_preset()
        if selected is None:
            return
        _, name, preset = selected
        self.apply_preset(preset, refresh=True)
        self.statusBar().showMessage(f"已加载参数预设: {name}")

    def save_current_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "保存参数预设", "预设名称:")
        name = name.strip()
        if not ok or not name:
            return
        if name in BUILTIN_PRESETS:
            QMessageBox.warning(self, "保存失败", "不能覆盖内置预设，请换一个名称。")
            return
        self.user_presets[name] = self.current_preset_data()
        self.save_user_presets()
        self.refresh_preset_combo(name)
        self.statusBar().showMessage(f"已保存参数预设: {name}")

    def delete_selected_preset(self) -> None:
        selected = self.selected_preset()
        if selected is None:
            return
        source, name, _ = selected
        if source == "builtin":
            QMessageBox.information(self, "不能删除", "内置预设不能删除。")
            return
        if QMessageBox.question(self, "删除参数预设", f"确定删除预设“{name}”？") != QMessageBox.Yes:
            return
        self.user_presets.pop(name, None)
        self.save_user_presets()
        self.refresh_preset_combo()
        self.statusBar().showMessage(f"已删除参数预设: {name}")

    def filter_controls(self) -> list[SliderControl]:
        return [
            self.sensitivity_spin,
            self.background_sigma_spin,
            self.residual_boost_spin,
            self.min_area_spin,
            self.morph_kernel_spin,
            self.overlay_alpha_spin,
        ]

    def current_settings(self) -> processing.ProcessingSettings:
        return processing.ProcessingSettings(
            algorithm_mode=self.algorithm_combo.currentData(),
            sensitivity=self.sensitivity_spin.value(),
            local_background_sigma=self.background_sigma_spin.value(),
            residual_color_boost=self.residual_boost_spin.value(),
            min_component_area=self.min_area_spin.value(),
            morph_kernel=self.morph_kernel_spin.value(),
            overlay_alpha=self.overlay_alpha_spin.value(),
            base_color_samples=tuple(self.base_color_samples),
            residual_color_samples=tuple(self.residual_color_samples),
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
            self.base_path_edit.setToolTip(path)
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
            self.sample_path_edit.setToolTip(path)
            self.load_sample_preview(Path(path))

    def choose_folder(self) -> None:
        start = self.folder_path_edit.text().strip() or str(DATA_DIR)
        path = QFileDialog.getExistingDirectory(self, "选择批量处理文件夹", start)
        if path:
            self.folder_path_edit.setText(path)
            self.folder_path_edit.setToolTip(path)

    def import_calc_design_table(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入图片计算参数表",
            str(CALC_DESIGN_TEMPLATE_PATH),
            "Excel (*.xlsx *.xls)",
        )
        if not path:
            return
        try:
            raw = pd.read_excel(path)
            self.calc_design_df = normalize_design_table(raw, "图片计算参数表")
            self.calc_design_path = Path(path)
            self.calc_design_label.setText(f"已导入: {Path(path).name}，{len(self.calc_design_df)} 条参数")
            self.calc_design_label.setToolTip(path)
            if self.results:
                self.refresh_result_design_columns()
            self.statusBar().showMessage(f"已导入图片计算参数表: {path}")
        except Exception as exc:
            self.show_error("导入图片计算参数表失败", exc)

    def refresh_result_design_columns(self) -> None:
        if self.calc_design_df is None:
            return
        design_lookup = self.calc_design_df.set_index("sample_id").to_dict(orient="index")
        for row in self.results:
            self.apply_calc_design_to_row(row, design_lookup)
        self.update_results_table()

    def apply_calc_design_to_row(
        self,
        row: dict[str, object],
        design_lookup: dict[str, dict[str, object]] | None = None,
    ) -> None:
        for column in ["design_type", *rsm.FACTOR_COLUMNS]:
            row[column] = np.nan if column in rsm.FACTOR_COLUMNS else ""
        row["calc_param_table"] = ""
        row["param_match_status"] = "未导入计算参数表"
        if self.calc_design_df is None:
            return
        if design_lookup is None:
            design_lookup = self.calc_design_df.set_index("sample_id").to_dict(orient="index")
        design = design_lookup.get(str(row.get("sample_id", "")).upper())
        row["calc_param_table"] = self.calc_design_path.name if self.calc_design_path else ""
        if not design:
            row["param_match_status"] = "未匹配"
            return
        row.update(design)
        row["param_match_status"] = "已匹配"

    def warn_if_no_calc_design(self) -> None:
        if self.calc_design_df is None:
            self.statusBar().showMessage("未导入图片计算参数表：仍会计算R/%和ΔG/%，但实验参数列为空")

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
        if enabled and self.color_picker_button.isChecked():
            self.color_picker_button.setChecked(False)
        if enabled and self.residual_picker_button.isChecked():
            self.residual_picker_button.setChecked(False)
        self.viewer.set_roi_mode(enabled)
        self.roi_button.setText("结束ROI框选" if enabled else "框选ROI")

    def toggle_base_color_picker_mode(self, enabled: bool) -> None:
        self.toggle_color_picker_mode(enabled, "base")

    def toggle_residual_color_picker_mode(self, enabled: bool) -> None:
        self.toggle_color_picker_mode(enabled, "residual")

    def toggle_color_picker_mode(self, enabled: bool, target: str) -> None:
        if enabled and self.roi_button.isChecked():
            self.roi_button.setChecked(False)
        if enabled and target == "base" and self.residual_picker_button.isChecked():
            self.residual_picker_button.setChecked(False)
        if enabled and target == "residual" and self.color_picker_button.isChecked():
            self.color_picker_button.setChecked(False)
        if enabled:
            self.color_pick_target = target
        self.viewer.set_color_pick_mode(enabled)
        self.color_picker_button.setText("退出取色" if enabled and target == "base" else "取基材色")
        self.residual_picker_button.setText("退出取色" if enabled and target == "residual" else "取残留色")
        if enabled:
            text = "基材" if target == "base" else "残留"
            self.statusBar().showMessage(f"请在预览图中点击一处{text}颜色区域")

    def on_roi_changed(self, roi: object) -> None:
        self.current_roi = roi if isinstance(roi, tuple) else None
        self.roi_label.setText(processing.roi_to_text(self.current_roi) if self.current_roi else "全图")

    def clear_roi(self) -> None:
        self.current_roi = None
        self.viewer.clear_roi()
        self.roi_label.setText("全图")

    def pick_base_color(self, x: int, y: int) -> None:
        if self.color_pick_target == "residual":
            self.pick_residual_color(x, y)
            return
        try:
            base_rgb = self.ensure_base_image()
            sample_rgb = self._current_sample_rgb_for_preview()
            if sample_rgb is None:
                sample_rgb = base_rgb
            sample_rgb = processing.resize_to_shape(sample_rgb, base_rgb.shape)
            x = int(np.clip(x, 0, sample_rgb.shape[1] - 1))
            y = int(np.clip(y, 0, sample_rgb.shape[0] - 1))
            self.base_color_samples.append(processing.color_sample_from_point(sample_rgb, x, y))
            self.update_color_sample_labels()
            self.color_picker_button.setChecked(False)
            self.preview_filter_effect(force=True)
            self.statusBar().showMessage(
                f"已添加基材色样本({x}, {y})，当前样本数: {len(self.base_color_samples)}"
            )
        except Exception as exc:
            self.show_error("取色失败", exc)

    def pick_residual_color(self, x: int, y: int) -> None:
        try:
            sample_rgb = self._current_sample_rgb_for_preview()
            if sample_rgb is None:
                raise ValueError("请先选择样品图片")
            if self.base_rgb is not None:
                sample_rgb = processing.resize_to_shape(sample_rgb, self.base_rgb.shape)
            x = int(np.clip(x, 0, sample_rgb.shape[1] - 1))
            y = int(np.clip(y, 0, sample_rgb.shape[0] - 1))
            self.residual_color_samples.append(processing.color_sample_from_point(sample_rgb, x, y))
            self.update_color_sample_labels()
            self.residual_picker_button.setChecked(False)
            self.preview_filter_effect(force=True)
            self.statusBar().showMessage(
                f"已添加残留色样本({x}, {y})，当前样本数: {len(self.residual_color_samples)}"
            )
        except Exception as exc:
            self.show_error("残留取色失败", exc)

    def clear_base_colors(self) -> None:
        self.base_color_samples.clear()
        self.update_color_sample_labels()
        self.preview_filter_effect(force=False)
        self.statusBar().showMessage("已清除基材色样本")

    def clear_residual_colors(self) -> None:
        self.residual_color_samples.clear()
        self.update_color_sample_labels()
        self.preview_filter_effect(force=False)
        self.statusBar().showMessage("已清除残留色样本")

    def _current_sample_rgb_for_preview(self) -> np.ndarray | None:
        if self.current_result is not None:
            return self.current_result.sample_rgb
        if self.current_sample_path is not None:
            return processing.read_rgb_image(self.current_sample_path)
        text = self.sample_path_edit.text().strip()
        if text:
            path = Path(text)
            if path.exists():
                return processing.read_rgb_image(path)
        return None

    def preview_filter_effect(self, force: bool = False) -> None:
        if not force and not self.auto_preview_check.isChecked():
            return
        try:
            if self.current_sample_path is None:
                text = self.sample_path_edit.text().strip()
                if not text:
                    if force:
                        raise ValueError("请先选择样品图片")
                    return
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
            self._select_view_mode("overlay")
            self.refresh_preview()
            self.statusBar().showMessage(
                f"过滤预览已更新: R={result.metrics['residual_area_percent']:.4f}%, "
                f"ΔG={result.metrics['gray_diff_percent']:.4f}%"
            )
        except Exception as exc:
            if force:
                self.show_error("更新过滤预览失败", exc)
            else:
                self.statusBar().showMessage(f"更新过滤预览失败: {exc}")

    def process_current_image(self) -> None:
        try:
            self.warn_if_no_calc_design()
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
            self._select_view_mode("overlay")
            self.refresh_preview()
            self.statusBar().showMessage(
                f"{result.sample_path.name}: R={result.metrics['residual_area_percent']:.4f}%, "
                f"ΔG={result.metrics['gray_diff_percent']:.4f}%"
            )
        except Exception as exc:
            self.show_error("处理失败", exc)

    def process_batch_folder(self) -> None:
        try:
            self.warn_if_no_calc_design()
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
            self._select_view_mode("overlay")
            self.refresh_preview()
            self.statusBar().showMessage(f"批量处理完成: {len(self.results)} 条结果")
        except Exception as exc:
            self.show_error("批量处理失败", exc)

    def _result_row(self, result: processing.ProcessResult) -> dict[str, object]:
        row = dict(result.metrics)
        row["path"] = str(result.sample_path)
        self.apply_calc_design_to_row(row)
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
                # 把 path 存在第0列的 UserRole，供双击时定位记录
                if col_idx == 0:
                    item.setData(Qt.UserRole, row.get("path", ""))
                self.results_table.setItem(row_idx, col_idx, item)
        self.results_table.setSortingEnabled(True)

    def on_result_row_double_clicked(self, row: int, _col: int) -> None:
        """双击结果表某行，切换预览到该行对应的样品处理结果。"""
        # 从第0列的 UserRole 取出 path（排序后行号与 self.results 索引不对应，用 path 匹配）
        item = self.results_table.item(row, 0)
        if item is None:
            return
        path_str = item.data(Qt.UserRole)
        if not path_str:
            return
        path = Path(path_str)
        if not path.exists():
            self.statusBar().showMessage(f"图片文件不存在: {path}")
            return

        # 如果当前已有该路径的处理结果，直接切换显示
        if self.current_result is not None and self.current_result.sample_path.resolve() == path.resolve():
            self._select_view_mode("overlay")
            self.refresh_preview()
            self.statusBar().showMessage(f"已切换预览: {path.name}")
            return

        # 尝试从 self.results 缓存中找到对应记录，重新处理以恢复 ProcessResult
        try:
            base_rgb = self.ensure_base_image()
            # 从缓存记录里恢复 ROI（如果有）
            cached_row = next(
                (r for r in self.results if r.get("path") == path_str),
                None,
            )
            roi: tuple[int, int, int, int] | None = self.current_roi
            if cached_row:
                roi_text = str(cached_row.get("roi", ""))
                # roi_text 格式: "x=N, y=N, w=N, h=N" 或 "全图"
                import re as _re
                m = _re.fullmatch(
                    r"x=(\d+),\s*y=(\d+),\s*w=(\d+),\s*h=(\d+)",
                    roi_text.strip(),
                )
                if m:
                    roi = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                elif roi_text in ("全图", ""):
                    roi = None

            result = processing.process_sample(path, base_rgb, roi, self.current_settings())
            self.current_result = result
            self.current_sample_path = path
            self.sample_path_edit.setText(str(path))
            self.sample_path_edit.setToolTip(str(path))
            self.viewer.set_roi(roi)
            self._select_view_mode("overlay")
            self.refresh_preview()
            self.statusBar().showMessage(
                f"已切换预览: {path.name}  R={result.metrics['residual_area_percent']:.4f}%  "
                f"ΔG={result.metrics['gray_diff_percent']:.4f}%"
            )
        except Exception as exc:
            self.show_error("切换预览失败", exc)

    def refresh_preview(self) -> None:
        if self.current_result is None:
            return
        mode = self.view_mode_combo.currentData()
        if mode == "mask":
            image = np.zeros_like(self.current_result.sample_rgb)
            image[self.current_result.residual_mask] = np.array([255, 255, 255], dtype=np.uint8)
        elif mode == "heatmap":
            image = processing.score_to_heatmap(
                self.current_result.residual_score,
                self.current_result.score_threshold,
            )
        elif mode == "overlay":
            image = self.current_result.overlay_rgb
        else:
            image = self.current_result.sample_rgb
        self.viewer.set_image_array(image, reset_view=False)
        self.viewer.set_roi(self.current_roi)

    def _select_view_mode(self, mode: str) -> None:
        for index in range(self.view_mode_combo.count()):
            if self.view_mode_combo.itemData(index) == mode:
                self.view_mode_combo.setCurrentIndex(index)
                return

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
