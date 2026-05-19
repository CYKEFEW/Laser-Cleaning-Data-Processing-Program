from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


FACTOR_COLUMNS = ["f_khz", "tau_ns", "h_mm", "v_mms"]
FACTOR_LABELS = {
    "f_khz": "频率 f / kHz",
    "tau_ns": "脉宽 τ / ns",
    "h_mm": "线间距 h / mm",
    "v_mms": "扫描速度 v / mm·s⁻¹",
}
FACTOR_RANGES = {
    "f_khz": (100.0, 130.0),
    "tau_ns": (250.0, 500.0),
    "h_mm": (0.020, 0.030),
    "v_mms": (4000.0, 5000.0),
}
RESPONSE_COLUMNS = {
    "R/%": "residual_area_percent",
    "ΔG/%": "gray_diff_percent",
}


@dataclass
class QuadraticModel:
    response_name: str
    response_column: str
    coefficient_names: list[str]
    coefficients: np.ndarray
    r2: float
    rmse: float
    rank: int
    n_samples: int
    y_min: float
    y_max: float


def _center_and_half_range(column: str) -> tuple[float, float]:
    low, high = FACTOR_RANGES[column]
    return (low + high) / 2.0, (high - low) / 2.0


def encode_factor_value(column: str, value: float) -> float:
    center, half_range = _center_and_half_range(column)
    return (float(value) - center) / half_range


def decode_factor_value(column: str, coded: float) -> float:
    center, half_range = _center_and_half_range(column)
    return center + float(coded) * half_range


def encode_array(values: np.ndarray) -> np.ndarray:
    encoded = np.zeros_like(values, dtype=float)
    for idx, column in enumerate(FACTOR_COLUMNS):
        encoded[:, idx] = [encode_factor_value(column, value) for value in values[:, idx]]
    return encoded


def decode_array(values: np.ndarray) -> np.ndarray:
    decoded = np.zeros_like(values, dtype=float)
    for idx, column in enumerate(FACTOR_COLUMNS):
        decoded[:, idx] = [decode_factor_value(column, value) for value in values[:, idx]]
    return decoded


def design_terms(encoded_values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(encoded_values, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    terms = [np.ones(x.shape[0])]
    names = ["Intercept"]
    for idx, column in enumerate(FACTOR_COLUMNS):
        terms.append(x[:, idx])
        names.append(column)
    for idx, column in enumerate(FACTOR_COLUMNS):
        terms.append(x[:, idx] ** 2)
        names.append(f"{column}^2")
    for i, j in combinations(range(len(FACTOR_COLUMNS)), 2):
        terms.append(x[:, i] * x[:, j])
        names.append(f"{FACTOR_COLUMNS[i]}:{FACTOR_COLUMNS[j]}")
    return np.column_stack(terms), names


def build_design_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    factorial = [
        ("F01", 100, 250, 0.020, 4000),
        ("F02", 130, 250, 0.020, 4000),
        ("F03", 100, 500, 0.020, 4000),
        ("F04", 130, 500, 0.020, 4000),
        ("F05", 100, 250, 0.030, 4000),
        ("F06", 130, 250, 0.030, 4000),
        ("F07", 100, 500, 0.030, 4000),
        ("F08", 130, 500, 0.030, 4000),
        ("F09", 100, 250, 0.020, 5000),
        ("F10", 130, 250, 0.020, 5000),
        ("F11", 100, 500, 0.020, 5000),
        ("F12", 130, 500, 0.020, 5000),
        ("F13", 100, 250, 0.030, 5000),
        ("F14", 130, 250, 0.030, 5000),
        ("F15", 100, 500, 0.030, 5000),
        ("F16", 130, 500, 0.030, 5000),
    ]
    axial = [
        ("A01", 100, 375, 0.025, 4500),
        ("A02", 130, 375, 0.025, 4500),
        ("A03", 115, 250, 0.025, 4500),
        ("A04", 115, 500, 0.025, 4500),
        ("A05", 115, 375, 0.020, 4500),
        ("A06", 115, 375, 0.030, 4500),
        ("A07", 115, 375, 0.025, 4000),
        ("A08", 115, 375, 0.025, 5000),
    ]
    center = [(f"M{idx:02d}", 115, 375, 0.025, 4500) for idx in range(1, 8)]

    for sample_id, f, tau, h, v in factorial:
        rows.append(_design_row(sample_id, "F", f, tau, h, v))
    for sample_id, f, tau, h, v in axial:
        rows.append(_design_row(sample_id, "A", f, tau, h, v))
    for sample_id, f, tau, h, v in center:
        rows.append(_design_row(sample_id, "M", f, tau, h, v))
    return pd.DataFrame(rows)


def _design_row(
    sample_id: str,
    design_type: str,
    f_khz: float,
    tau_ns: float,
    h_mm: float,
    v_mms: float,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "design_type": design_type,
        "f_khz": float(f_khz),
        "tau_ns": float(tau_ns),
        "h_mm": float(h_mm),
        "v_mms": float(v_mms),
    }


def design_map() -> dict[str, dict[str, object]]:
    return build_design_table().set_index("sample_id").to_dict(orient="index")


def add_design_columns(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    table = build_design_table()
    merged = results.copy()
    merged["sample_id"] = merged["sample_id"].astype(str).str.upper()
    return merged.merge(table, on="sample_id", how="left")


def fit_quadratic_model(data: pd.DataFrame, response_name: str) -> QuadraticModel:
    response_column = RESPONSE_COLUMNS.get(response_name, response_name)
    if response_column not in data.columns:
        raise ValueError(f"结果表缺少响应列: {response_column}")

    subset = data.dropna(subset=FACTOR_COLUMNS + [response_column]).copy()
    if len(subset) < 6:
        raise ValueError("至少需要6个带参数的数据点才能拟合响应面模型")

    raw_x = subset[FACTOR_COLUMNS].astype(float).to_numpy()
    y = subset[response_column].astype(float).to_numpy()
    encoded_x = encode_array(raw_x)
    matrix, names = design_terms(encoded_x)
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    pred = matrix @ coefficients
    residual = y - pred
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1e-12 else float("nan")
    rmse = float(np.sqrt(np.mean(residual**2)))
    return QuadraticModel(
        response_name=response_name,
        response_column=response_column,
        coefficient_names=names,
        coefficients=coefficients,
        r2=r2,
        rmse=rmse,
        rank=int(rank),
        n_samples=int(len(subset)),
        y_min=float(y.min()),
        y_max=float(y.max()),
    )


def predict_encoded(model: QuadraticModel, encoded_values: np.ndarray) -> np.ndarray:
    matrix, _ = design_terms(encoded_values)
    return matrix @ model.coefficients


def predict_real(model: QuadraticModel, values: dict[str, float] | np.ndarray) -> np.ndarray:
    if isinstance(values, dict):
        raw = np.array([[float(values[column]) for column in FACTOR_COLUMNS]], dtype=float)
    else:
        raw = np.asarray(values, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
    return predict_encoded(model, encode_array(raw))


def optimize_model(model: QuadraticModel) -> tuple[dict[str, float], float]:
    result = differential_evolution(
        lambda x: float(predict_encoded(model, np.asarray(x))[0]),
        bounds=[(-1.0, 1.0)] * len(FACTOR_COLUMNS),
        seed=7,
        polish=True,
    )
    decoded = decode_array(np.asarray(result.x).reshape(1, -1))[0]
    params = {column: float(decoded[idx]) for idx, column in enumerate(FACTOR_COLUMNS)}
    return params, float(result.fun)


def optimize_combined(
    residual_model: QuadraticModel,
    gray_model: QuadraticModel,
) -> tuple[dict[str, float], float, float, float]:
    r_span = max(residual_model.y_max - residual_model.y_min, 1e-9)
    g_span = max(gray_model.y_max - gray_model.y_min, 1e-9)

    def objective(x: np.ndarray) -> float:
        x = np.asarray(x)
        r_value = float(predict_encoded(residual_model, x)[0])
        g_value = float(predict_encoded(gray_model, x)[0])
        return ((r_value - residual_model.y_min) / r_span + (g_value - gray_model.y_min) / g_span) / 2.0

    result = differential_evolution(
        objective,
        bounds=[(-1.0, 1.0)] * len(FACTOR_COLUMNS),
        seed=11,
        polish=True,
    )
    decoded = decode_array(np.asarray(result.x).reshape(1, -1))[0]
    params = {column: float(decoded[idx]) for idx, column in enumerate(FACTOR_COLUMNS)}
    r_value = float(predict_encoded(residual_model, result.x)[0])
    g_value = float(predict_encoded(gray_model, result.x)[0])
    return params, float(result.fun), r_value, g_value


def response_surface(
    model: QuadraticModel,
    x_factor: str,
    y_factor: str,
    fixed_values: dict[str, float],
    grid_size: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_low, x_high = FACTOR_RANGES[x_factor]
    y_low, y_high = FACTOR_RANGES[y_factor]
    x_values = np.linspace(x_low, x_high, grid_size)
    y_values = np.linspace(y_low, y_high, grid_size)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    rows = []
    for x_value, y_value in zip(x_grid.ravel(), y_grid.ravel(), strict=False):
        row = []
        for column in FACTOR_COLUMNS:
            if column == x_factor:
                row.append(x_value)
            elif column == y_factor:
                row.append(y_value)
            else:
                default = sum(FACTOR_RANGES[column]) / 2.0
                row.append(float(fixed_values.get(column, default)))
        rows.append(row)
    z = predict_real(model, np.asarray(rows)).reshape(x_grid.shape)
    return x_grid, y_grid, z


def format_params(params: dict[str, float]) -> str:
    return (
        f"f={params['f_khz']:.3g} kHz, "
        f"τ={params['tau_ns']:.3g} ns, "
        f"h={params['h_mm']:.5g} mm, "
        f"v={params['v_mms']:.3g} mm/s"
    )
