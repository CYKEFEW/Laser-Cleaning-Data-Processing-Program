from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ProcessingSettings:
    rgb_std_factor: float = 2.4
    hue_tolerance: float = 14.0
    sv_std_factor: float = 2.5
    lab_delta_e: float = 28.0
    min_component_area: int = 120
    morph_kernel: int = 3
    votes_required: int = 2
    overlay_alpha: float = 0.45


@dataclass
class BaseColorModel:
    rgb_mean: np.ndarray
    rgb_std: np.ndarray
    hue_center: float
    sv_mean: np.ndarray
    sv_std: np.ndarray
    lab_mean: np.ndarray
    base_gray_mean: float
    image_shape: tuple[int, int, int]
    roi: tuple[int, int, int, int]


@dataclass
class ProcessResult:
    sample_path: Path
    sample_id: str
    sample_rgb: np.ndarray
    residual_mask: np.ndarray
    base_mask: np.ndarray
    overlay_rgb: np.ndarray
    metrics: dict[str, object]


def read_rgb_image(path: str | Path) -> np.ndarray:
    """Read image paths with non-ASCII characters using OpenCV decoding."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb_image(path: str | Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(suffix, image_bgr)
    if not ok:
        raise ValueError(f"无法编码图片: {path}")
    encoded.tofile(str(path))


def write_mask_image(path: str | Path, mask: np.ndarray) -> None:
    rgb = np.dstack([mask.astype(np.uint8) * 255] * 3)
    write_rgb_image(path, rgb)


def discover_images(folder: str | Path, base_path: str | Path | None = None) -> list[Path]:
    folder = Path(folder)
    base_resolved = Path(base_path).resolve() if base_path else None
    paths = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if base_resolved and path.resolve() == base_resolved:
            continue
        if path.stem.lower().endswith("base"):
            continue
        paths.append(path)
    return paths


def sample_id_from_path(path: str | Path) -> str:
    stem = Path(path).stem.upper()
    for token in reversed(re.split(r"[._\-\s]+", stem)):
        if re.fullmatch(r"[A-Z]\d{2}", token):
            return token
    match = re.search(r"([A-Z]\d{2})", stem)
    return match.group(1) if match else stem


def normalize_roi(
    roi: tuple[int, int, int, int] | None,
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    if roi is None:
        return 0, 0, width, height
    x, y, w, h = [int(round(v)) for v in roi]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def roi_to_text(roi: tuple[int, int, int, int]) -> str:
    x, y, w, h = roi
    return f"x={x}, y={y}, w={w}, h={h}"


def crop_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return image[y : y + h, x : x + w]


def resize_to_shape(image_rgb: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    target_h, target_w = shape[:2]
    if image_rgb.shape[:2] == (target_h, target_w):
        return image_rgb
    return cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)


def build_base_model(
    base_rgb: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    settings: ProcessingSettings,
) -> BaseColorModel:
    roi = normalize_roi(roi, base_rgb.shape)
    base_roi = crop_roi(base_rgb, roi)
    hsv_roi = cv2.cvtColor(base_roi, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab_roi = cv2.cvtColor(base_roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray_roi = cv2.cvtColor(base_roi, cv2.COLOR_RGB2GRAY)

    rgb_values = base_roi.reshape(-1, 3).astype(np.float32)
    hsv_values = hsv_roi.reshape(-1, 3)
    lab_values = lab_roi.reshape(-1, 3)

    hue = hsv_values[:, 0]
    angles = hue / 180.0 * 2.0 * np.pi
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    hue_center = (mean_angle % (2.0 * np.pi)) / (2.0 * np.pi) * 180.0

    rgb_std = np.maximum(rgb_values.std(axis=0), 6.0)
    sv_std = np.maximum(hsv_values[:, 1:3].std(axis=0), 6.0)

    return BaseColorModel(
        rgb_mean=rgb_values.mean(axis=0),
        rgb_std=rgb_std,
        hue_center=float(hue_center),
        sv_mean=hsv_values[:, 1:3].mean(axis=0),
        sv_std=sv_std,
        lab_mean=lab_values.mean(axis=0),
        base_gray_mean=float(gray_roi.mean()),
        image_shape=base_rgb.shape,
        roi=roi,
    )


def _hue_distance(hue: np.ndarray, center: float) -> np.ndarray:
    diff = np.abs(hue.astype(np.float32) - center)
    return np.minimum(diff, 180.0 - diff)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cleaned = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def compute_masks(
    sample_rgb: np.ndarray,
    model: BaseColorModel,
    settings: ProcessingSettings,
) -> tuple[np.ndarray, np.ndarray]:
    sample_rgb = resize_to_shape(sample_rgb, model.image_shape)
    x, y, w, h = model.roi
    sample_roi = crop_roi(sample_rgb, model.roi)

    hsv_roi = cv2.cvtColor(sample_roi, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab_roi = cv2.cvtColor(sample_roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    rgb_roi = sample_roi.astype(np.float32)

    rgb_tol = settings.rgb_std_factor * model.rgb_std + 8.0
    rgb_match = np.all(np.abs(rgb_roi - model.rgb_mean) <= rgb_tol, axis=2)

    hue_match = _hue_distance(hsv_roi[:, :, 0], model.hue_center) <= settings.hue_tolerance
    sv_tol = settings.sv_std_factor * model.sv_std + 8.0
    sv_match = np.all(np.abs(hsv_roi[:, :, 1:3] - model.sv_mean) <= sv_tol, axis=2)
    hsv_match = hue_match & sv_match

    delta_e = np.linalg.norm(lab_roi - model.lab_mean, axis=2)
    lab_match = delta_e <= settings.lab_delta_e

    votes = rgb_match.astype(np.uint8) + hsv_match.astype(np.uint8) + lab_match.astype(np.uint8)
    base_roi_mask = votes >= max(1, min(3, settings.votes_required))
    residual_roi_mask = ~base_roi_mask

    kernel_size = max(1, int(settings.morph_kernel))
    if kernel_size > 1:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        residual_u8 = residual_roi_mask.astype(np.uint8)
        residual_u8 = cv2.morphologyEx(residual_u8, cv2.MORPH_OPEN, kernel)
        residual_u8 = cv2.morphologyEx(residual_u8, cv2.MORPH_CLOSE, kernel)
        residual_roi_mask = residual_u8.astype(bool)

    residual_roi_mask = _remove_small_components(residual_roi_mask, settings.min_component_area)
    base_roi_mask = ~residual_roi_mask

    base_mask = np.zeros(sample_rgb.shape[:2], dtype=bool)
    residual_mask = np.zeros(sample_rgb.shape[:2], dtype=bool)
    base_mask[y : y + h, x : x + w] = base_roi_mask
    residual_mask[y : y + h, x : x + w] = residual_roi_mask
    return base_mask, residual_mask


def make_overlay(sample_rgb: np.ndarray, residual_mask: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = sample_rgb.copy().astype(np.float32)
    red = np.array([255.0, 32.0, 32.0], dtype=np.float32)
    overlay[residual_mask] = overlay[residual_mask] * (1.0 - alpha) + red * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def process_sample(
    sample_path: str | Path,
    base_rgb: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    settings: ProcessingSettings,
) -> ProcessResult:
    model = build_base_model(base_rgb, roi, settings)
    return process_sample_with_model(sample_path, model, settings)


def process_sample_with_model(
    sample_path: str | Path,
    model: BaseColorModel,
    settings: ProcessingSettings,
) -> ProcessResult:
    sample_path = Path(sample_path)
    sample_rgb = resize_to_shape(read_rgb_image(sample_path), model.image_shape)
    base_mask, residual_mask = compute_masks(sample_rgb, model, settings)
    overlay = make_overlay(sample_rgb, residual_mask, settings.overlay_alpha)

    x, y, w, h = model.roi
    valid_pixels = w * h
    residual_pixels = int(residual_mask.sum())
    sample_gray = cv2.cvtColor(crop_roi(sample_rgb, model.roi), cv2.COLOR_RGB2GRAY)
    sample_gray_mean = float(sample_gray.mean())
    base_gray_mean = model.base_gray_mean
    gray_diff = abs(sample_gray_mean - base_gray_mean) / max(base_gray_mean, 1e-6) * 100.0

    metrics: dict[str, object] = {
        "file_name": sample_path.name,
        "sample_id": sample_id_from_path(sample_path),
        "residual_area_percent": residual_pixels / valid_pixels * 100.0,
        "gray_diff_percent": gray_diff,
        "sample_gray_mean": sample_gray_mean,
        "base_gray_mean": base_gray_mean,
        "residual_pixels": residual_pixels,
        "valid_pixels": valid_pixels,
        "roi": roi_to_text(model.roi),
        "rgb_std_factor": settings.rgb_std_factor,
        "hue_tolerance": settings.hue_tolerance,
        "sv_std_factor": settings.sv_std_factor,
        "lab_delta_e": settings.lab_delta_e,
        "min_component_area": settings.min_component_area,
        "morph_kernel": settings.morph_kernel,
        "votes_required": settings.votes_required,
        "status": "OK",
    }
    return ProcessResult(
        sample_path=sample_path,
        sample_id=str(metrics["sample_id"]),
        sample_rgb=sample_rgb,
        residual_mask=residual_mask,
        base_mask=base_mask,
        overlay_rgb=overlay,
        metrics=metrics,
    )


def batch_process(
    sample_paths: Iterable[str | Path],
    base_rgb: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    settings: ProcessingSettings,
) -> list[ProcessResult]:
    model = build_base_model(base_rgb, roi, settings)
    return [process_sample_with_model(path, model, settings) for path in sample_paths]
