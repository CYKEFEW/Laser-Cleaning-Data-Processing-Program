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
    algorithm_mode: str = "score"
    sensitivity: float = 73.0
    local_background_sigma: float = 42.0
    residual_color_boost: float = 1.65
    rgb_std_factor: float = 2.4
    hue_tolerance: float = 14.0
    sv_std_factor: float = 2.5
    lab_delta_e: float = 28.0
    min_component_area: int = 2480
    morph_kernel: int = 3
    votes_required: int = 2
    overlay_alpha: float = 0.27
    base_color_samples: tuple[dict[str, object], ...] = ()
    residual_color_samples: tuple[dict[str, object], ...] = ()


@dataclass
class BaseColorModel:
    rgb_mean: np.ndarray
    rgb_std: np.ndarray
    hue_center: float
    sv_mean: np.ndarray
    sv_std: np.ndarray
    lab_mean: np.ndarray
    lab_std: np.ndarray
    base_lab_roi: np.ndarray
    base_hsv_roi: np.ndarray
    base_score_values: np.ndarray
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
    residual_score: np.ndarray
    score_threshold: float | None
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


def color_sample_from_patch(patch_rgb: np.ndarray) -> dict[str, object]:
    patch_rgb = np.asarray(patch_rgb, dtype=np.uint8)
    patch_rgb_values = patch_rgb.reshape(-1, 3).astype(np.float32)
    patch_hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    patch_lab = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    hue_values = patch_hsv[:, 0]
    angles = hue_values / 180.0 * 2.0 * np.pi
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    hue_center = float((mean_angle % (2.0 * np.pi)) / (2.0 * np.pi) * 180.0)

    rgb_mean = patch_rgb_values.mean(axis=0)
    sv_mean = patch_hsv[:, 1:3].mean(axis=0)
    lab_mean = patch_lab.mean(axis=0)

    rgb_tol = float(np.clip(np.percentile(np.max(np.abs(patch_rgb_values - rgb_mean), axis=1), 90) + 18.0, 18.0, 90.0))
    hue_tol = float(np.clip(np.percentile(_hue_distance(hue_values, hue_center), 90) + 6.0, 6.0, 45.0))
    sv_tol = float(np.clip(np.percentile(np.max(np.abs(patch_hsv[:, 1:3] - sv_mean), axis=1), 90) + 20.0, 20.0, 110.0))
    lab_tol = float(np.clip(np.percentile(np.linalg.norm(patch_lab - lab_mean, axis=1), 90) + 16.0, 16.0, 90.0))

    return {
        "rgb": tuple(float(v) for v in rgb_mean),
        "hue": hue_center,
        "sv": tuple(float(v) for v in sv_mean),
        "lab": tuple(float(v) for v in lab_mean),
        "rgb_tol": rgb_tol,
        "hue_tol": hue_tol,
        "sv_tol": sv_tol,
        "lab_tol": lab_tol,
    }


def color_sample_from_point(image_rgb: np.ndarray, x: int, y: int, radius: int = 8) -> dict[str, object]:
    height, width = image_rgb.shape[:2]
    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))
    patch = image_rgb[
        max(0, y - radius) : min(height, y + radius + 1),
        max(0, x - radius) : min(width, x + radius + 1),
    ]
    return color_sample_from_patch(patch)


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

    lab_std = np.maximum(lab_values.std(axis=0), 6.0)

    model = BaseColorModel(
        rgb_mean=rgb_values.mean(axis=0),
        rgb_std=rgb_std,
        hue_center=float(hue_center),
        sv_mean=hsv_values[:, 1:3].mean(axis=0),
        sv_std=sv_std,
        lab_mean=lab_values.mean(axis=0),
        lab_std=lab_std,
        base_lab_roi=lab_roi,
        base_hsv_roi=hsv_roi,
        base_score_values=np.array([], dtype=np.float32),
        base_gray_mean=float(gray_roi.mean()),
        image_shape=base_rgb.shape,
        roi=roi,
    )
    baseline_settings = ProcessingSettings(local_background_sigma=settings.local_background_sigma)
    model.base_score_values = compute_residual_score(base_roi, model, baseline_settings).reshape(-1)
    return model


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


def _sample_affinity(
    rgb_roi: np.ndarray,
    hsv_roi: np.ndarray,
    lab_roi: np.ndarray,
    sample: dict[str, object],
) -> np.ndarray:
    rgb = np.asarray(sample["rgb"], dtype=np.float32)
    sv = np.asarray(sample["sv"], dtype=np.float32)
    lab = np.asarray(sample["lab"], dtype=np.float32)
    hue = float(sample["hue"])
    rgb_tol = max(float(sample.get("rgb_tol", 34.0)), 1.0)
    hue_tol = max(float(sample.get("hue_tol", 12.0)), 1.0)
    sv_tol = max(float(sample.get("sv_tol", 42.0)), 1.0)
    lab_tol = max(float(sample.get("lab_tol", 32.0)), 1.0)

    rgb_dist = np.max(np.abs(rgb_roi - rgb), axis=2) / rgb_tol
    hue_dist = _hue_distance(hsv_roi[:, :, 0], hue) / hue_tol
    sv_dist = np.max(np.abs(hsv_roi[:, :, 1:3] - sv), axis=2) / sv_tol
    lab_dist = np.linalg.norm(lab_roi - lab, axis=2) / lab_tol
    z2 = 0.10 * rgb_dist**2 + 0.20 * hue_dist**2 + 0.25 * sv_dist**2 + 0.45 * lab_dist**2
    return np.exp(-0.5 * z2).astype(np.float32)


def compute_residual_score(
    image_roi: np.ndarray,
    model: BaseColorModel,
    settings: ProcessingSettings,
) -> np.ndarray:
    lab_roi = cv2.cvtColor(image_roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv_roi = cv2.cvtColor(image_roi, cv2.COLOR_RGB2HSV).astype(np.float32)
    rgb_roi = image_roi.astype(np.float32)

    sigma = max(float(settings.local_background_sigma), 3.0)
    l_channel = lab_roi[:, :, 0]
    a_channel = lab_roi[:, :, 1]
    b_channel = lab_roi[:, :, 2]
    s_channel = hsv_roi[:, :, 1]

    bg_l = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg_a = cv2.GaussianBlur(a_channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg_b = cv2.GaussianBlur(b_channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
    bg_s = cv2.GaussianBlur(s_channel, (0, 0), sigmaX=sigma, sigmaY=sigma)

    dark = np.maximum(bg_l - l_channel, 0.0)
    saturation_excess = np.maximum(s_channel - bg_s, 0.0)
    local_chroma = np.sqrt((a_channel - bg_a) ** 2 + (b_channel - bg_b) ** 2)
    purple_shift = np.maximum(a_channel - bg_a, 0.0) + np.maximum(bg_b - b_channel, 0.0)

    if model.base_lab_roi.shape[:2] == lab_roi.shape[:2]:
        base_delta = np.linalg.norm(lab_roi - model.base_lab_roi, axis=2)
    else:
        base_delta = np.linalg.norm((lab_roi - model.lab_mean) / model.lab_std, axis=2) * 8.0

    score = (
        0.80 * dark
        + 0.45 * saturation_excess
        + 0.55 * local_chroma
        + 0.55 * purple_shift
        + 0.18 * base_delta
    )

    for sample in settings.base_color_samples:
        score -= 30.0 * _sample_affinity(rgb_roi, hsv_roi, lab_roi, sample)
    for sample in settings.residual_color_samples:
        score += float(settings.residual_color_boost) * 35.0 * _sample_affinity(rgb_roi, hsv_roi, lab_roi, sample)

    score = np.maximum(score, 0.0)
    score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
    return score.astype(np.float32)


def _score_threshold(model: BaseColorModel, settings: ProcessingSettings) -> float:
    sensitivity = float(np.clip(settings.sensitivity, 0.0, 100.0))
    percentile = 99.8 - sensitivity / 100.0 * 2.8
    percentile = float(np.clip(percentile, 97.0, 99.8))
    values = model.base_score_values
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _compute_score_masks(
    sample_rgb: np.ndarray,
    model: BaseColorModel,
    settings: ProcessingSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sample_rgb = resize_to_shape(sample_rgb, model.image_shape)
    x, y, w, h = model.roi
    sample_roi = crop_roi(sample_rgb, model.roi)
    roi_score = compute_residual_score(sample_roi, model, settings)
    threshold = _score_threshold(model, settings)
    residual_roi_mask = roi_score > threshold
    residual_roi_mask = _postprocess_residual_mask(residual_roi_mask, settings)
    base_roi_mask = ~residual_roi_mask

    base_mask = np.zeros(sample_rgb.shape[:2], dtype=bool)
    residual_mask = np.zeros(sample_rgb.shape[:2], dtype=bool)
    score_map = np.zeros(sample_rgb.shape[:2], dtype=np.float32)
    base_mask[y : y + h, x : x + w] = base_roi_mask
    residual_mask[y : y + h, x : x + w] = residual_roi_mask
    score_map[y : y + h, x : x + w] = roi_score
    return base_mask, residual_mask, score_map, threshold


def _postprocess_residual_mask(mask: np.ndarray, settings: ProcessingSettings) -> np.ndarray:
    residual_roi_mask = mask.astype(bool)
    kernel_size = max(1, int(settings.morph_kernel))
    if kernel_size > 1:
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        residual_u8 = residual_roi_mask.astype(np.uint8)
        residual_u8 = cv2.morphologyEx(residual_u8, cv2.MORPH_OPEN, kernel)
        residual_u8 = cv2.morphologyEx(residual_u8, cv2.MORPH_CLOSE, kernel)
        residual_roi_mask = residual_u8.astype(bool)
    return _remove_small_components(residual_roi_mask, settings.min_component_area)


def _residual_color_mask(
    rgb_roi: np.ndarray,
    hsv_roi: np.ndarray,
    lab_roi: np.ndarray,
    settings: ProcessingSettings,
) -> np.ndarray:
    if not settings.residual_color_samples:
        return np.zeros(rgb_roi.shape[:2], dtype=bool)

    combined = np.zeros(rgb_roi.shape[:2], dtype=bool)
    for sample in settings.residual_color_samples:
        rgb = np.asarray(sample["rgb"], dtype=np.float32)
        sv = np.asarray(sample["sv"], dtype=np.float32)
        lab = np.asarray(sample["lab"], dtype=np.float32)
        hue = float(sample["hue"])
        rgb_tol = float(sample.get("rgb_tol", 34.0))
        hue_tol = float(sample.get("hue_tol", 12.0))
        sv_tol = float(sample.get("sv_tol", 42.0))
        lab_tol = float(sample.get("lab_tol", 32.0))

        rgb_match = np.max(np.abs(rgb_roi - rgb), axis=2) <= rgb_tol
        hue_match = _hue_distance(hsv_roi[:, :, 0], hue) <= hue_tol
        sv_match = np.max(np.abs(hsv_roi[:, :, 1:3] - sv), axis=2) <= sv_tol
        lab_match = np.linalg.norm(lab_roi - lab, axis=2) <= lab_tol
        votes = rgb_match.astype(np.uint8) + hue_match.astype(np.uint8) + sv_match.astype(np.uint8) + lab_match.astype(np.uint8)
        combined |= votes >= 2
    return combined


def compute_masks(
    sample_rgb: np.ndarray,
    model: BaseColorModel,
    settings: ProcessingSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    if settings.algorithm_mode == "score":
        return _compute_score_masks(sample_rgb, model, settings)

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
    residual_pick_mask = _residual_color_mask(rgb_roi, hsv_roi, lab_roi, settings)
    residual_roi_mask = (~base_roi_mask) | residual_pick_mask
    residual_roi_mask = _postprocess_residual_mask(residual_roi_mask, settings)
    base_roi_mask = ~residual_roi_mask

    base_mask = np.zeros(sample_rgb.shape[:2], dtype=bool)
    residual_mask = np.zeros(sample_rgb.shape[:2], dtype=bool)
    score_map = np.zeros(sample_rgb.shape[:2], dtype=np.float32)
    base_mask[y : y + h, x : x + w] = base_roi_mask
    residual_mask[y : y + h, x : x + w] = residual_roi_mask
    score_map[y : y + h, x : x + w] = residual_roi_mask.astype(np.float32)
    return base_mask, residual_mask, score_map, 0.5


def make_overlay(sample_rgb: np.ndarray, residual_mask: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = sample_rgb.copy().astype(np.float32)
    red = np.array([255.0, 32.0, 32.0], dtype=np.float32)
    overlay[residual_mask] = overlay[residual_mask] * (1.0 - alpha) + red * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def score_to_heatmap(score_map: np.ndarray, threshold: float | None = None) -> np.ndarray:
    score = np.asarray(score_map, dtype=np.float32)
    positive = score[score > 0]
    if positive.size == 0:
        scaled = np.zeros(score.shape, dtype=np.uint8)
    else:
        high = float(np.percentile(positive, 99.5))
        if threshold is not None:
            high = max(high, float(threshold) * 1.6)
        high = max(high, 1e-6)
        scaled = np.clip(score / high * 255.0, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


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
    base_mask, residual_mask, residual_score, score_threshold = compute_masks(sample_rgb, model, settings)
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
        "algorithm_mode": settings.algorithm_mode,
        "sensitivity": settings.sensitivity,
        "local_background_sigma": settings.local_background_sigma,
        "score_threshold": score_threshold,
        "base_color_samples": len(settings.base_color_samples),
        "residual_color_samples": len(settings.residual_color_samples),
        "residual_color_boost": settings.residual_color_boost,
        "status": "OK",
    }
    return ProcessResult(
        sample_path=sample_path,
        sample_id=str(metrics["sample_id"]),
        sample_rgb=sample_rgb,
        residual_mask=residual_mask,
        base_mask=base_mask,
        residual_score=residual_score,
        score_threshold=score_threshold,
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
