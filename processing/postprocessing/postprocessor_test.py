import os
from pathlib import Path
import csv
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest
import logging
from datetime import datetime

from postprocessing.postprocessor import PostProcessor, ProcessResult


def _project_root() -> Path:
    # postprocessing/ -> project root
    return Path(__file__).resolve().parents[1]


def _test_data_dir() -> Path:
    return _project_root() / "data" / "test" / "test_data_v1"


def _report_path() -> Path:
    return _project_root() / "postprocessing" / "postprocessor_test_report.txt"


def _append_report(lines: List[str]) -> None:
    rp = _report_path()
    with open(rp, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def _read_image_size(img_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to read image at {img_path}"
    h, w = img.shape[:2]
    return w, h


def _parse_yolo_seg_file(label_path: Path) -> Dict[int, np.ndarray]:
    class_to_poly: Dict[int, np.ndarray] = {}
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            assert len(coords) % 2 == 0 and len(coords) >= 8, (
                f"Expected polygon pairs for segmentation in {label_path}"
            )
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            class_to_poly[cls_id] = pts
    return class_to_poly


def _polygon_norm_to_mask(pts_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)

    # Map normalized coords to pixel coords, inclusive of borders
    pts_px = np.column_stack([pts_norm[:, 0] * (width - 1), pts_norm[:, 1] * (height - 1)])
    pts_px = np.round(pts_px).astype(np.int32)

    cv2.fillPoly(mask, [pts_px], 255)
    return mask


def _build_masks_for_range(start_idx: int, end_idx: int) -> Tuple[Dict[int, Dict[str, np.ndarray]], Tuple[int, int]]:
    data_dir = _test_data_dir()
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    # Determine image size from the first frame in range
    first_img = images_dir / f"_{start_idx:04d}.jpg"
    width, height = _read_image_size(first_img)

    masks_by_index: Dict[int, Dict[str, np.ndarray]] = {}

    for idx in range(start_idx, end_idx + 1):
        label_path = labels_dir / f"_{idx:04d}.txt"
        assert label_path.exists(), f"Missing label file: {label_path}"

        class_to_poly = _parse_yolo_seg_file(label_path)

        # Per data.yaml: 0: lane, 1: ball
        assert 0 in class_to_poly and 1 in class_to_poly, (
            f"Expected both lane(0) and ball(1) classes in {label_path}"
        )

        lane_poly = class_to_poly[0]
        ball_poly = class_to_poly[1]

        lane_mask = _polygon_norm_to_mask(lane_poly, width, height)
        ball_mask = _polygon_norm_to_mask(ball_poly, width, height)

        masks_by_index[idx] = {"ball": ball_mask, "lane": lane_mask}

    return masks_by_index, (width, height)


def _read_csv_episode(csv_path: Path, episode_idx: int) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["episode_idx"]) != episode_idx:
                continue

            fi = int(row["frame_idx"])
            out[fi] = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "t": float(row["t_sec"]),
                "vx": float(row["vx_units_per_sec"]),
                "dx_per_frame": float(row["dx_units_per_frame"]),
            }

    return out


def _best_fit_axis(bev_x: np.ndarray, bev_y: np.ndarray, csv_x: np.ndarray):
    # Try BEV x
    cx = np.polyfit(bev_x, csv_x, deg=1)
    pred_x = cx[0] * bev_x + cx[1]
    mae_x = float(np.mean(np.abs(pred_x - csv_x)))

    # Try BEV y
    cy = np.polyfit(bev_y, csv_x, deg=1)
    pred_y = cy[0] * bev_y + cy[1]
    mae_y = float(np.mean(np.abs(pred_y - csv_x)))

    if mae_x <= mae_y:
        return "x", cx, mae_x, pred_x
    else:
        return "y", cy, mae_y, pred_y


def test_postprocessor_bev_centroid_track_matches_csv_x():
    start_idx, end_idx = 30, 59  # episode 1
    csv_path = _project_root() / "data" / "test" / "constant_velocity_log.csv"

    masks_by_index, _ = _build_masks_for_range(start_idx, end_idx)

    pp = PostProcessor(out_size=(400, 800), dt=1.0 / 30.0, buffer_len=12)
    results_by_index, H = pp.process_run(masks_by_index)

    assert H.shape == (3, 3)
    assert results_by_index.keys() == set(range(start_idx, end_idx + 1))

    bev_x: List[float] = []
    bev_y: List[float] = []

    for idx in range(start_idx, end_idx + 1):
        r: ProcessResult = results_by_index[idx]
        assert r.warped_ball.ndim == 2 and r.warped_lane.ndim == 2
        assert r.warped_ball.shape == (800, 400)  # (H, W) by cv2.warpPerspective
        assert r.warped_lane.shape == (800, 400)
        assert r.bev_centroid is not None, f"Missing centroid for frame {idx}"

        bx, by = r.bev_centroid
        bev_x.append(bx)
        bev_y.append(by)

    bev_x = np.array(bev_x, dtype=np.float64)
    bev_y = np.array(bev_y, dtype=np.float64)

    csv_ep1 = _read_csv_episode(csv_path, episode_idx=1)
    csv_x = np.array([csv_ep1[i]["x"] for i in range(start_idx, end_idx + 1)], dtype=np.float64)

    axis, coeffs, mae, preds = _best_fit_axis(bev_x, bev_y, csv_x)

    logger = logging.getLogger(__name__)
    logger.info(
        "Episode 1 frames %d-%d: selected BEV axis=%s; fit a=%.6f, b=%.6f; MAE=%.3f",
        start_idx, end_idx, axis, float(coeffs[0]), float(coeffs[1]), mae
    )

    _append_report([
        f"[{datetime.utcnow().isoformat()}Z] test_postprocessor_bev_centroid_track_matches_csv_x",
        f"  frames={start_idx}-{end_idx}, axis={axis}, a={float(coeffs[0]):.6f}, b={float(coeffs[1]):.6f}, MAE={mae:.3f}",
        ""
    ])

    abs_err = np.abs(preds - csv_x)
    max_err = float(np.max(abs_err))
    p95 = float(np.percentile(abs_err, 95))
    p75 = float(np.percentile(abs_err, 75))
    mean_err = float(np.mean(abs_err))

    rows = [
        f"[{datetime.utcnow().isoformat()}Z] per-frame residuals (frames {start_idx}-{end_idx}, axis={axis})",
        f"  summary: mean={mean_err:.3f}, p75={p75:.3f}, p95={p95:.3f}, max={max_err:.3f}"
    ]
    for k, i in enumerate(range(start_idx, end_idx + 1)):
        rows.append(f"  frame={i}: csv_x={csv_x[k]:.3f}, pred={preds[k]:.3f}, abs_err={abs_err[k]:.3f}")
    rows.append("")
    _append_report(rows)

    assert mae < 0.6, f"MAE on x after calibration too large using BEV {axis}-axis: {mae:.3f} units"

    assert np.isfinite(bev_x).all() and np.isfinite(bev_y).all()


def test_postprocessor_velocity_consistent_with_csv_speed_magnitude():
    start_idx, end_idx = 30, 59  # episode 1
    csv_path = _project_root() / "data" / "test" / "constant_velocity_log.csv"

    masks_by_index, _ = _build_masks_for_range(start_idx, end_idx)

    pp = PostProcessor(out_size=(400, 800), dt=1.0 / 30.0, buffer_len=12)
    results_by_index, _ = pp.process_run(masks_by_index)

    bev_x: List[float] = []
    bev_y: List[float] = []
    vel_x: List[float] = []
    vel_y: List[float] = []

    for idx in range(start_idx, end_idx + 1):
        r: ProcessResult = results_by_index[idx]
        assert r.bev_centroid is not None

        bx, by = r.bev_centroid
        bev_x.append(bx)
        bev_y.append(by)

        if r.velocity is not None:
            vx, vy = r.velocity
            vel_x.append(vx)
            vel_y.append(vy)

    bev_x = np.array(bev_x, dtype=np.float64)
    bev_y = np.array(bev_y, dtype=np.float64)
    vel_x = np.array(vel_x, dtype=np.float64)  # length N-1 approximately
    vel_y = np.array(vel_y, dtype=np.float64)

    csv_ep1 = _read_csv_episode(csv_path, episode_idx=1)
    csv_x = np.array([csv_ep1[i]["x"] for i in range(start_idx, end_idx + 1)], dtype=np.float64)

    axis, coeffs, mae, _ = _best_fit_axis(bev_x, bev_y, csv_x)
    a = float(coeffs[0])  # scale factor only (offset cancels in velocity)

    vel_units = a * (vel_x if axis == "x" else vel_y)
    target_speed = 17.0
    median_speed = float(np.median(np.abs(vel_units)))
    n_vel = int(vel_units.size)

    logger = logging.getLogger(__name__)
    logger.info(
        "Episode 1 frames %d-%d: velocity using BEV %s-axis; median=%.3f units/s (target=%.1f); samples=%d",
        start_idx, end_idx, axis, median_speed, target_speed, n_vel
    )
    _append_report([
        f"[{datetime.utcnow().isoformat()}Z] test_postprocessor_velocity_consistent_with_csv_speed_magnitude",
        f"  frames={start_idx}-{end_idx}, axis={axis}, median_speed={median_speed:.3f}, target={target_speed:.1f}, samples={n_vel}",
        ""
    ])

    assert abs(median_speed - target_speed) < 1.0, (
        f"Velocity magnitude off using BEV {axis}-axis: {median_speed:.3f} vs {target_speed:.3f}"
    )

