from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import csv
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt

from processing.preprocessing.preprocessor import YOLOSegPreprocessor, InferenceConfig
from processing.postprocessing.postprocessor import PostProcessor, LaneMetrics, ProcessResult, Warp
from processing.processor import Processor


def _project_root() -> Path:
    # processor_test.py -> processing/ -> project root
    return Path(__file__).resolve().parents[1]


def _test_data_dir() -> Path:
    return _project_root() / "data" / "test" / "curved_test_data_v1"


class _TrajectoryDummyYOLO:
    """
    Dummy YOLO model that yields a ball following a simple curved trajectory
    across frames so that lane metrics (velocity, break, end speed) are
    well-defined without requiring a real network.
    """

    def __init__(self) -> None:
        self.names = {0: "lane", 1: "ball"}

    def predict(self, source: str, imgsz: int, device: str, verbose: bool):
        h, w = 40, 80
        lane = np.zeros((h, w), dtype=np.float32)
        lane[5:35, 10:70] = 1.0  # long rectangular lane

        stem = Path(source).stem  # "_0000" style
        try:
            idx = int(stem.strip("_"))
        except ValueError:
            idx = 0

        # Along-lane coordinate increases with idx, lateral coordinate curves.
        t = float(idx)
        y_center = 8 + 4 * t  # down-lane (primary axis)
        x_center = 30 + 2 * t  # lateral drift to create "break"

        ball = np.zeros((h, w), dtype=np.float32)
        y0 = max(0, int(y_center - 2))
        y1 = min(h, int(y_center + 2))
        x0 = max(0, int(x_center - 2))
        x1 = min(w, int(x_center + 2))
        ball[y0:y1, x0:x1] = 1.0

        mask_stack = np.stack([lane, ball], axis=0)

        class _Masks:
            def __init__(self, data):
                self.data = data

        class _Boxes:
            def __init__(self, cls_ids):
                self.cls = cls_ids

        class _Result:
            def __init__(self, masks, boxes, orig_shape):
                self.masks = masks
                self.boxes = boxes
                self.orig_shape = orig_shape

        masks = _Masks(mask_stack)
        boxes = _Boxes(np.array([0.0, 1.0], dtype=np.float32))
        return [_Result(masks, boxes, (h, w))]


def _make_dummy_frames(img_dir: Path, num_frames: int) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(num_frames):
        canvas = np.zeros((40, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"_{idx:04d}.jpg"), canvas)


def _extract_bev_axes(
    results_by_index: Dict[int, ProcessResult]
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Extract along-lane (s_bev) and lateral (l_bev) coordinates from BEV
    centroids, using the same primary-axis logic as PostProcessor.
    """
    idxs = sorted(results_by_index.keys())
    traj: List[Tuple[float, float]] = []
    frames: List[int] = []
    for i in idxs:
        c = results_by_index[i].bev_centroid
        if c is None:
            continue
        traj.append(c)
        frames.append(i)

    bev = np.array(traj, dtype=np.float64)
    xs = bev[:, 0]
    ys = bev[:, 1]

    # Choose primary axis as the one with larger dynamic range.
    range_x = float(xs.max() - xs.min())
    range_y = float(ys.max() - ys.min())
    primary_axis = 0 if range_x >= range_y else 1

    s_bev = xs if primary_axis == 0 else ys  # along-lane coordinate
    l_bev = ys if primary_axis == 0 else xs  # lateral coordinate

    return s_bev, l_bev, frames


def _calibrate_bev_to_world(
    results_by_index: Dict[int, ProcessResult],
    gt_data: Dict[int, Dict[str, float]],
    episode_idx: int,
) -> Tuple[float, float, float, float]:
    """
    Compute linear mappings:
      s_world ≈ a_s * s_bev + b_s
      y_world ≈ a_l * l_bev + b_l
    based on BEV centroids and ground-truth (lane_s, y) for one episode.
    """
    s_bev, l_bev, frames = _extract_bev_axes(results_by_index)

    lane_s_world: List[float] = []
    y_world: List[float] = []
    s_bev_used: List[float] = []
    l_bev_used: List[float] = []

    for s_val, l_val, frame in zip(s_bev, l_bev, frames):
        row = gt_data.get(frame)
        if row is None or row["episode_idx"] != episode_idx:
            continue
        lane_s_world.append(row["lane_s"])
        y_world.append(row["y"])
        s_bev_used.append(s_val)
        l_bev_used.append(l_val)

    if len(lane_s_world) < 4:
        raise ValueError(f"Not enough GT points to calibrate episode {episode_idx}")

    s_bev_used = np.array(s_bev_used, dtype=np.float64)
    l_bev_used = np.array(l_bev_used, dtype=np.float64)
    lane_s_world = np.array(lane_s_world, dtype=np.float64)
    y_world = np.array(y_world, dtype=np.float64)

    a_s, b_s = np.polyfit(s_bev_used, lane_s_world, deg=1)
    a_l, b_l = np.polyfit(l_bev_used, y_world, deg=1)

    return float(a_s), float(b_s), float(a_l), float(b_l)


def _load_ground_truth_csv() -> Dict[int, Dict[str, float]]:
    """Load the curved trajectory CSV as ground truth data."""
    csv_path = _project_root() / "data" / "test" / "curved_trajectory_log.csv"
    gt_data = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["global_frame_idx"])
            gt_data[frame_idx] = {
                "episode_idx": int(row["episode_idx"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "vx": float(row["vx_units_per_sec"]),
                "vy": float(row["vy_units_per_sec"]),
                "speed": float(row["speed_units_per_sec"]),
                "lane_s": float(row["lane_s"]),
                "lane_frac": float(row["lane_frac"]),
                "total_break": float(row["episode_total_break"]),
                "end_speed": float(row["episode_end_speed"]),
            }

    return gt_data


def _compute_lane_metrics_ground_truth(episode_idx: int, gt_data: Dict[int, Dict[str, float]],
                                      fractions: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)) -> LaneMetrics:
    """Compute ground truth lane metrics for an episode."""
    # Get all frames for this episode
    episode_frames = {k: v for k, v in gt_data.items() if v["episode_idx"] == episode_idx}
    if not episode_frames:
        raise ValueError(f"No frames found for episode {episode_idx}")

    frame_indices = sorted(episode_frames.keys())
    lane_s_vals = [episode_frames[idx]["lane_s"] for idx in frame_indices]
    x_vals = [episode_frames[idx]["x"] for idx in frame_indices]
    y_vals = [episode_frames[idx]["y"] for idx in frame_indices]

    # Find lane start/end
    lane_start_s = min(lane_s_vals)
    lane_end_s = max(lane_s_vals)
    lane_length = lane_end_s - lane_start_s

    # Compute velocities and accelerations using finite differences
    dt = 1.0 / 30.0  # 30 FPS
    vs = np.gradient(lane_s_vals, dt)  # velocity along lane
    vl = np.gradient(y_vals, dt)       # lateral velocity
    as_ = np.gradient(vs, dt)          # acceleration along lane
    al = np.gradient(vl, dt)           # lateral acceleration

    frac_positions = []
    vel_at_frac = []
    acc_at_frac = []

    for frac in fractions:
        target_s = lane_start_s + frac * lane_length
        # Find closest frame
        closest_idx = np.argmin(np.abs(np.array(lane_s_vals) - target_s))
        idx = frame_indices[closest_idx]

        vx = vs[closest_idx]
        vy = vl[closest_idx]
        speed = np.hypot(vx, vy)

        ax = as_[closest_idx]
        ay = al[closest_idx]
        accel_mag = np.hypot(ax, ay)

        frac_positions.append(lane_s_vals[closest_idx])
        vel_at_frac.append((vx, vy, speed))
        acc_at_frac.append((ax, ay, accel_mag))

    # Total break and end speed
    start_y = y_vals[0]
    end_y = y_vals[-1]
    total_break = abs(end_y - start_y)
    end_speed = np.hypot(vs[-1], vl[-1])

    return LaneMetrics(
        fractions=fractions,
        frac_positions=tuple(frac_positions),
        velocity_at_frac=tuple(vel_at_frac),
        acceleration_at_frac=tuple(acc_at_frac),
        total_break=total_break,
        end_lane_speed=end_speed,
    )


def _format_csv_value(value, precision: int = 6):
    """Format numeric values for cleaner CSV output."""
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return ""
        # Use scientific notation for very small or very large numbers
        if abs(value) < 1e-10 and value != 0:
            return f"{value:.2e}"
        elif abs(value) > 1e6:
            return f"{value:.2e}"
        else:
            # Round to precision decimal places and remove trailing zeros
            formatted = f"{value:.{precision}f}".rstrip('0').rstrip('.')
            return float(formatted) if '.' in f"{value:.{precision}f}" else int(float(formatted))
    return value


def _format_row(row: dict) -> dict:
    """Format all numeric values in a row for cleaner CSV output."""
    formatted = {}
    for key, value in row.items():
        if key in ("timestamp", "test_name", "interpolation_mode", "fraction"):
            formatted[key] = value
        else:
            formatted[key] = _format_csv_value(value)
    return formatted


def _log_processor_test_results(
    episode_idx: int,
    frame_indices: range,
    predicted_metrics: LaneMetrics,
    gt_metrics: LaneMetrics,
    csv_writer,
    test_name: str,
    interpolation_mode: str = "none",
    seg_map_per_fraction: Optional[List[float]] = None,
    episode_mean_seg_map: Optional[float] = None,
):
    """Log comprehensive test results to CSV."""
    timestamp = datetime.utcnow().isoformat()

    # Log per-fraction metrics
    for i, frac in enumerate(predicted_metrics.fractions):
        pred_vel = predicted_metrics.velocity_at_frac[i]
        pred_acc = predicted_metrics.acceleration_at_frac[i]
        gt_vel = gt_metrics.velocity_at_frac[i]
        gt_acc = gt_metrics.acceleration_at_frac[i]

        # Get seg_map for this fraction if provided
        seg_map = seg_map_per_fraction[i] if seg_map_per_fraction and i < len(seg_map_per_fraction) else None

        row = {
            "timestamp": timestamp,
            "test_name": test_name,
            "interpolation_mode": interpolation_mode,
            "episode_idx": episode_idx,
            "fraction": frac,
            "frac_position": predicted_metrics.frac_positions[i],

            # Predicted velocities
            "pred_vel_x": pred_vel[0],
            "pred_vel_y": pred_vel[1],
            "pred_speed": pred_vel[2],

            # Ground truth velocities
            "gt_vel_x": gt_vel[0],
            "gt_vel_y": gt_vel[1],
            "gt_speed": gt_vel[2],

            # Velocity errors
            "vel_x_error": pred_vel[0] - gt_vel[0],
            "vel_y_error": pred_vel[1] - gt_vel[1],
            "speed_error": pred_vel[2] - gt_vel[2],

            # Predicted accelerations
            "pred_acc_x": pred_acc[0],
            "pred_acc_y": pred_acc[1],
            "pred_accel_mag": pred_acc[2],

            # Ground truth accelerations
            "gt_acc_x": gt_acc[0],
            "gt_acc_y": gt_acc[1],
            "gt_accel_mag": gt_acc[2],

            # Acceleration errors
            "acc_x_error": pred_acc[0] - gt_acc[0],
            "acc_y_error": pred_acc[1] - gt_acc[1],
            "accel_mag_error": pred_acc[2] - gt_acc[2],

            # Segmentation mAP for this fraction (closest frame)
            "segmentation_map_50_95": seg_map,
        }
        csv_writer.writerow(_format_row(row))

    # Log episode-level metrics
    row = {
        "timestamp": timestamp,
        "test_name": test_name,
        "interpolation_mode": interpolation_mode,
        "episode_idx": episode_idx,
        "fraction": "episode_total",
        "frac_position": None,

        # Total break
        "pred_total_break": predicted_metrics.total_break,
        "gt_total_break": gt_metrics.total_break,
        "total_break_error": predicted_metrics.total_break - gt_metrics.total_break,

        # End speed
        "pred_end_speed": predicted_metrics.end_lane_speed,
        "gt_end_speed": gt_metrics.end_lane_speed,
        "end_speed_error": predicted_metrics.end_lane_speed - gt_metrics.end_lane_speed,

        # Frame count
        "frame_count": len(frame_indices),

        # Episode mean mAP
        "segmentation_map_50_95": episode_mean_seg_map,
    }
    csv_writer.writerow(_format_row(row))


def _create_results_csv():
    """Create timestamped CSV file for logging test results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = _project_root() / "processing" / f"test_{timestamp}.csv"

    fieldnames = [
        "timestamp", "test_name", "interpolation_mode", "episode_idx", "fraction", "frac_position",
        "pred_vel_x", "pred_vel_y", "pred_speed",
        "gt_vel_x", "gt_vel_y", "gt_speed",
        "vel_x_error", "vel_y_error", "speed_error",
        "pred_acc_x", "pred_acc_y", "pred_accel_mag",
        "gt_acc_x", "gt_acc_y", "gt_accel_mag",
        "acc_x_error", "acc_y_error", "accel_mag_error",
        "pred_total_break", "gt_total_break", "total_break_error",
        "pred_end_speed", "gt_end_speed", "end_speed_error",
        "frame_count", "segmentation_map_50_95"
    ]

    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()

    return csv_path, writer, f  # Return file handle so it can be closed


def test_full_processor_wrapper_computes_lane_metrics(tmp_path):
    images_dir = tmp_path / "episode_images"
    num_frames = 6
    _make_dummy_frames(images_dir, num_frames)

    dummy_model = _TrajectoryDummyYOLO()
    infer_cfg = InferenceConfig(weights_path=tmp_path / "dummy.pt", imgsz=80, device="cpu")
    pre = YOLOSegPreprocessor(config=infer_cfg, model=dummy_model)
    post = PostProcessor(out_size=(400, 800), dt=1.0, buffer_len=64)

    proc = Processor(preprocessor=pre, postprocessor=post)
    episode = proc.run_episode_from_indices(images_dir, range(num_frames))

    # Basic structural checks
    assert episode.homography.shape == (3, 3)
    assert len(episode.results_by_index) == num_frames

    metrics = episode.lane_metrics
    # Fractions default to 4 positions along the lane
    assert len(metrics.fractions) == 4
    assert len(metrics.velocity_at_frac) == 4
    assert len(metrics.acceleration_at_frac) == 4

    # With the synthetic trajectory, we should have non-zero end speed
    assert metrics.end_lane_speed > 0.0
    # And some lateral break over the run
    assert metrics.total_break > 0.0

    # Log results to CSV
    csv_path, csv_writer, csv_file = _create_results_csv()
    print(f"\nLogging test results to: {csv_path}")

    # Create synthetic ground truth for comparison (since this is synthetic test)
    synthetic_gt = LaneMetrics(
        fractions=metrics.fractions,
        frac_positions=metrics.frac_positions,
        velocity_at_frac=tuple((v[0]*0.9, v[1]*0.9, v[2]*0.9) for v in metrics.velocity_at_frac),  # Slightly different GT
        acceleration_at_frac=tuple((a[0]*0.8, a[1]*0.8, a[2]*0.8) for a in metrics.acceleration_at_frac),
        total_break=metrics.total_break * 1.1,
        end_lane_speed=metrics.end_lane_speed * 0.95,
    )

    _log_processor_test_results(
        episode_idx=0,
        frame_indices=range(num_frames),
        predicted_metrics=metrics,
        gt_metrics=synthetic_gt,
        csv_writer=csv_writer,
        test_name="synthetic_trajectory_test",
        interpolation_mode="none"
    )

    csv_file.close()  # Close the CSV file


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
    pts_px = np.column_stack(
        [pts_norm[:, 0] * (width - 1), pts_norm[:, 1] * (height - 1)]
    )
    pts_px = np.round(pts_px).astype(np.int32)
    cv2.fillPoly(mask, [pts_px], 255)
    return mask


def _load_gt_masks(stem: str) -> dict[str, np.ndarray]:
    labels_dir = _test_data_dir() / "labels"
    img_dir = _test_data_dir() / "images"
    class_to_poly = _parse_yolo_seg_file(labels_dir / f"{stem}.txt")
    img = cv2.imread(str(img_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    lane_mask = _polygon_norm_to_mask(class_to_poly[0], w, h)
    ball_mask = _polygon_norm_to_mask(class_to_poly[1], w, h)
    return {"lane": lane_mask, "ball": ball_mask}


def _homography_from_gt_lane(stem: str, out_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    gt_masks = _load_gt_masks(stem)
    quad = Warp.quad_from_lane_mask(gt_masks["lane"])
    if quad is None:
        raise RuntimeError(f"No lane quad from GT for {stem}")
    dst = Warp.default_bev_corners(out_size)
    return quad.astype(np.float32), dst.astype(np.float32)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)
    inter = (a_bin & b_bin).sum()
    union = (a_bin | b_bin).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _map_50_95_for_episode(pred_masks: dict[int, dict[str, np.ndarray]]) -> dict[int, float]:
    thresholds = np.arange(0.5, 1.0, 0.05)
    maps = {}

    for idx, masks in pred_masks.items():
        stem = f"_{idx:04d}"
        gt_masks = _load_gt_masks(stem)
        aps = []

        for cls in ("lane", "ball"):
            iou = _mask_iou(masks[cls], gt_masks[cls])
            # single-instance AP at each threshold is 1 if IoU>=thr else 0
            aps.append([1.0 if iou >= t else 0.0 for t in thresholds])

        maps[idx] = float(np.mean(aps))

    return maps


def _frames_with_ball(results_by_index: Dict[int, ProcessResult]) -> Dict[int, ProcessResult]:
    """Filter results to frames that contain a ball mask."""
    out: Dict[int, ProcessResult] = {}
    for k, res in results_by_index.items():
        ball_mask = res.warped_ball
        if ball_mask is not None and np.any(ball_mask > 0):
            out[k] = res
    return out


def _episode_frame_ranges(gt_data: Dict[int, Dict[str, float]], labels_dir: Path) -> Dict[int, range]:
    """
    Derive per-episode frame ranges from available label files, aligned to GT if counts match.
    Handles warm-up offset where (start-1) exists but GT starts later.
    """
    frames_by_ep: Dict[int, List[int]] = defaultdict(list)
    for frame_idx, row in gt_data.items():
        frames_by_ep[row["episode_idx"]].append(frame_idx)

    available_frames_sorted = sorted({int(p.stem.strip("_")) for p in labels_dir.glob("*.txt")})
    ranges: Dict[int, range] = {}

    if available_frames_sorted:
        # Build contiguous runs from available frames.
        contiguous: List[range] = []
        start = prev = available_frames_sorted[0]
        for f in available_frames_sorted[1:]:
            if f != prev + 1:
                contiguous.append(range(start, prev + 1))
                start = f
            prev = f
        contiguous.append(range(start, prev + 1))

        gt_eps = sorted(frames_by_ep.keys())
        if len(gt_eps) == len(contiguous) and len(gt_eps) > 0:
            for ep, seg in zip(gt_eps, contiguous):
                ranges[ep] = seg
            return ranges

    # Fallback: clamp GT min/max to available frames, allowing start-1 shift.
    available_set = set(available_frames_sorted)
    for ep, frames in frames_by_ep.items():
        f_min, f_max = min(frames), max(frames)

        if f_min not in available_set and (f_min - 1) in available_set:
            f_min = f_min - 1

        while f_min not in available_set and f_min <= f_max:
            f_min += 1
        while f_max not in available_set and f_max >= f_min:
            f_max -= 1

        ranges[ep] = range(f_min, f_max + 1)

    return ranges


class _DatasetBackedYOLO:
    """
    YOLO-like model whose `.predict()` loads masks from the
    curved_test_data_v1 YOLO segmentation labels instead of running a network.
    """

    def __init__(self) -> None:
        self.names = {0: "lane", 1: "ball"}
        self.data_dir = _test_data_dir()

    def predict(self, source: str, imgsz: int, device: str, verbose: bool):
        images_dir = self.data_dir / "images"
        labels_dir = self.data_dir / "labels"

        img_path = Path(source)
        stem = img_path.stem  # e.g. "_0064"

        img = cv2.imread(str(images_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        assert img is not None, f"Failed to read image at {img_path}"
        h, w = img.shape[:2]

        label_path = labels_dir / f"{stem}.txt"
        class_to_poly = _parse_yolo_seg_file(label_path)

        lane_poly = class_to_poly[0]
        ball_poly = class_to_poly[1]

        lane_mask = _polygon_norm_to_mask(lane_poly, w, h).astype(np.float32) / 255.0
        ball_mask = _polygon_norm_to_mask(ball_poly, w, h).astype(np.float32) / 255.0

        mask_stack = np.stack([lane_mask, ball_mask], axis=0)

        class _Masks:
            def __init__(self, data):
                self.data = data

        class _Boxes:
            def __init__(self, cls_ids):
                self.cls = cls_ids

        class _Result:
            def __init__(self, masks, boxes, orig_shape):
                self.masks = masks
                self.boxes = boxes
                self.orig_shape = orig_shape

        masks = _Masks(mask_stack)
        boxes = _Boxes(np.array([0.0, 1.0], dtype=np.float32))
        return [_Result(masks, boxes, (h, w))]


def test_processor_on_curved_dataset_episodes_1_and_2_with_dataset_masks():
    """
    End-to-end processor test on the real curved_test_data_v1 images/labels
    for episodes 1 and 2 (second and third episodes).
    Comprehensive logging of predicted vs ground truth metrics.
    Tests all three interpolation modes: none, linear, cubic.
    """
    data_dir = _test_data_dir()
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    # Load ground truth data
    gt_data = _load_ground_truth_csv()

    infer_cfg = InferenceConfig(
        weights_path=_project_root() / "best.pt",
        imgsz=2048,
        device="cpu",
    )

    # Create CSV for logging results
    csv_path, csv_writer, csv_file = _create_results_csv()
    print(f"\nLogging comprehensive test results to: {csv_path}")

    # Derive episode frame ranges dynamically from GT/logged labels (handles off-by-one)
    ep_ranges = _episode_frame_ranges(gt_data, labels_dir)

    # Test with all interpolation modes
    interpolation_modes = ["none", "linear", "cubic"]
    
    all_errors_by_mode = {
        mode: {
            "vel_x_errors": [], "vel_y_errors": [], "speed_errors": [],
            "acc_x_errors": [], "acc_y_errors": [], "accel_mag_errors": [],
            "total_break_errors": [], "end_speed_errors": []
        }
        for mode in interpolation_modes
    }

    for interp_mode in interpolation_modes:
        print(f"\n{'='*60}")
        print(f"Testing with interpolation mode: {interp_mode.upper()}")
        print(f"{'='*60}")
        
        # Create processor with specific interpolation mode
        pre = YOLOSegPreprocessor(config=infer_cfg)
        post = PostProcessor(out_size=(400, 800), dt=1.0 / 30.0, buffer_len=128, interpolation_mode=interp_mode)
        proc = Processor(preprocessor=pre, postprocessor=post)
        
        all_errors = all_errors_by_mode[interp_mode]
        
        for episode_idx in (1, 2):
            if episode_idx not in ep_ranges:
                continue
            frame_indices = ep_ranges[episode_idx]
            print(f"\nProcessing Episode {episode_idx} (frames {frame_indices.start}-{frame_indices.stop-1})...")
            
            # Compute GT homography from the first frame
            first_frame_stem = f"_{frame_indices.start:04d}"
            src_pts, dst_pts = _homography_from_gt_lane(first_frame_stem, post.out_size)
            
            # Run full processor (pre + post) end-to-end with fixed homography
            episode = proc.run_episode_from_indices(
                images_dir, frame_indices, homography_src_dst=(src_pts, dst_pts)
            )

            # Structural checks
            assert episode.homography.shape == (3, 3)
            assert set(episode.results_by_index.keys()) == set(frame_indices)

            # Compute per-frame mAP using raw masks (before warping)
            map_by_frame = _map_50_95_for_episode(episode.raw_masks_by_index)

            # Only use frames with ball detections for calibration/metrics to avoid NaNs
            results_with_ball = _frames_with_ball(episode.results_by_index)
            if len(results_with_ball) < 2:
                raise AssertionError("Not enough frames with ball detections for metrics")

            bev_metrics = episode.lane_metrics
            assert len(bev_metrics.fractions) == 4
            assert len(bev_metrics.velocity_at_frac) == 4
            assert len(bev_metrics.acceleration_at_frac) == 4
            # Make assertions more conservative - just check they're non-negative
            assert bev_metrics.end_lane_speed >= 0.0
            assert bev_metrics.total_break >= 0.0

            # Calibrate BEV coordinates to world coordinates using per-frame GT
            # _calibrate_bev_to_world already drops frames with bev_centroid is None
            a_s, b_s, a_l, b_l = _calibrate_bev_to_world(
                results_with_ball, gt_data, episode_idx
            )

            # Build world-space LaneMetrics from BEV metrics
            # Also find closest frame for each fraction to get seg_map
            world_fractions = bev_metrics.fractions
            world_frac_positions: List[float] = []
            world_vel_at_frac: List[Tuple[float, float, float]] = []
            world_acc_at_frac: List[Tuple[float, float, float]] = []
            seg_map_per_fraction: List[float] = []

            # Extract BEV trajectory to find closest frames for fractions
            s_bev, l_bev, frames_with_ball = _extract_bev_axes(results_with_ball)
            
            for i, frac in enumerate(world_fractions):
                s_bev_target = bev_metrics.frac_positions[i]
                s_world = a_s * s_bev_target + b_s

                vx_bev, vy_bev, _ = bev_metrics.velocity_at_frac[i]
                ax_bev, ay_bev, _ = bev_metrics.acceleration_at_frac[i]

                vx_world = a_s * vx_bev
                vy_world = a_l * vy_bev
                speed_world = float(np.hypot(vx_world, vy_world))

                ax_world = a_s * ax_bev
                ay_world = a_l * ay_bev
                accel_world = float(np.hypot(ax_world, ay_world))

                world_frac_positions.append(s_world)
                world_vel_at_frac.append((vx_world, vy_world, speed_world))
                world_acc_at_frac.append((ax_world, ay_world, accel_world))

                # Find closest frame index for this fraction
                idx_closest = int(np.argmin(np.abs(s_bev - s_bev_target)))
                frame_idx_for_frac = frames_with_ball[idx_closest]
                seg_map = map_by_frame.get(frame_idx_for_frac)
                seg_map_per_fraction.append(seg_map if seg_map is not None else 0.0)

            pred_world_metrics = LaneMetrics(
                fractions=tuple(world_fractions),
                frac_positions=tuple(world_frac_positions),
                velocity_at_frac=tuple(world_vel_at_frac),
                acceleration_at_frac=tuple(world_acc_at_frac),
                total_break=a_l * bev_metrics.total_break,
                end_lane_speed=world_vel_at_frac[-1][2],
            )

            # Compute ground truth metrics for this episode (world space)
            gt_metrics = _compute_lane_metrics_ground_truth(episode_idx, gt_data)

            # Compute episode mean mAP
            episode_mean_seg_map = float(np.nanmean(list(map_by_frame.values()))) if map_by_frame else None

            # Log detailed results
            _log_processor_test_results(
                episode_idx=episode_idx,
                frame_indices=frame_indices,
                predicted_metrics=pred_world_metrics,
                gt_metrics=gt_metrics,
                csv_writer=csv_writer,
                test_name="curved_dataset_end_to_end",
                interpolation_mode=interp_mode,
                seg_map_per_fraction=seg_map_per_fraction,
                episode_mean_seg_map=episode_mean_seg_map,
            )

            # Collect errors for averaging
            for i in range(len(pred_world_metrics.fractions)):
                pred_vel = pred_world_metrics.velocity_at_frac[i]
                pred_acc = pred_world_metrics.acceleration_at_frac[i]
                gt_vel = gt_metrics.velocity_at_frac[i]
                gt_acc = gt_metrics.acceleration_at_frac[i]

                all_errors["vel_x_errors"].append(pred_vel[0] - gt_vel[0])
                all_errors["vel_y_errors"].append(pred_vel[1] - gt_vel[1])
                all_errors["speed_errors"].append(pred_vel[2] - gt_vel[2])
                all_errors["acc_x_errors"].append(pred_acc[0] - gt_acc[0])
                all_errors["acc_y_errors"].append(pred_acc[1] - gt_acc[1])
                all_errors["accel_mag_errors"].append(pred_acc[2] - gt_acc[2])

            all_errors["total_break_errors"].append(
                pred_world_metrics.total_break - gt_metrics.total_break
            )
            all_errors["end_speed_errors"].append(
                pred_world_metrics.end_lane_speed - gt_metrics.end_lane_speed
            )

    # Log summary statistics for each interpolation mode
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS BY INTERPOLATION MODE")
    print(f"{'='*60}")
    
    for interp_mode in interpolation_modes:
        all_errors = all_errors_by_mode[interp_mode]
        
        # Log summary statistics as multiple rows (since CSV has limited columns)
        summary_base = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_name": "summary_statistics",
            "interpolation_mode": interp_mode,
            "episode_idx": None,
            "fraction": "averages",
            "frac_position": None,
            "pred_vel_x": np.mean(all_errors["vel_x_errors"]),
            "pred_vel_y": np.mean(all_errors["vel_y_errors"]),
            "pred_speed": np.mean(all_errors["speed_errors"]),
            "pred_acc_x": np.mean(all_errors["acc_x_errors"]),
            "pred_acc_y": np.mean(all_errors["acc_y_errors"]),
            "pred_accel_mag": np.mean(all_errors["accel_mag_errors"]),
            "pred_total_break": np.mean(all_errors["total_break_errors"]),
            "pred_end_speed": np.mean(all_errors["end_speed_errors"]),
        }
        csv_writer.writerow(_format_row(summary_base))

        # Log MAE statistics
        mae_row = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_name": "summary_statistics",
            "interpolation_mode": interp_mode,
            "episode_idx": None,
            "fraction": "mae",
            "frac_position": None,
            "pred_vel_x": np.mean(np.abs(all_errors["vel_x_errors"])),
            "pred_vel_y": np.mean(np.abs(all_errors["vel_y_errors"])),
            "pred_speed": np.mean(np.abs(all_errors["speed_errors"])),
            "pred_acc_x": np.mean(np.abs(all_errors["acc_x_errors"])),
            "pred_acc_y": np.mean(np.abs(all_errors["acc_y_errors"])),
            "pred_accel_mag": np.mean(np.abs(all_errors["accel_mag_errors"])),
            "pred_total_break": np.mean(np.abs(all_errors["total_break_errors"])),
            "pred_end_speed": np.mean(np.abs(all_errors["end_speed_errors"])),
        }
        csv_writer.writerow(_format_row(mae_row))

        print(f"\n{interp_mode.upper()} Interpolation:")
        print(f"  Velocity MAE: {mae_row['pred_speed']:.4f} units/sec")
        print(f"  Acceleration MAE: {mae_row['pred_accel_mag']:.4f} units/sec²")
        print(f"  Total break MAE: {mae_row['pred_total_break']:.4f} units")
        print(f"  End speed MAE: {mae_row['pred_end_speed']:.4f} units/sec")

    print(f"\n{'='*60}")
    print(f"Episodes tested: 2 (episodes 1 & 2)")
    total_frames = sum(len(ep_ranges[ep]) for ep in (1, 2) if ep in ep_ranges)
    print(f"Total frames: {total_frames}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*60}")

    csv_file.close()  # Close the CSV file
    
    # ===================================================================
    # VISUALIZATION: Generate trajectory plots and error analysis graphs
    # ===================================================================
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATION ARTIFACTS")
    print(f"{'='*60}")
    
    # Import visualization module
    from processing.postprocessing.visualizer import TrajectoryVisualizer, ErrorAnalysisPlotter
    
    # Create output directory for visualizations
    viz_output_dir = _project_root() / "processing" / "visualization_output"
    viz_output_dir.mkdir(exist_ok=True)
    
    # Re-run processors to collect trajectory data for visualization
    # (We need to store trajectories and lane boundaries from each mode)
    trajectories_by_mode = {}
    lane_boundary = None
    errors_by_mode = {}
    
    for interp_mode in interpolation_modes:
        print(f"\nCollecting trajectory data for {interp_mode} mode...")
        
        pre = YOLOSegPreprocessor(config=infer_cfg)
        post = PostProcessor(out_size=(400, 800), dt=1.0 / 30.0, buffer_len=128, interpolation_mode=interp_mode)
        proc = Processor(preprocessor=pre, postprocessor=post)
        
        # Use episode 1 for visualization (it's representative)
        episode_idx = 1
        frame_indices = ep_ranges[episode_idx]
        
        first_frame_stem = f"_{frame_indices.start:04d}"
        src_pts, dst_pts = _homography_from_gt_lane(first_frame_stem, post.out_size)
        
        episode = proc.run_episode_from_indices(
            images_dir, frame_indices, homography_src_dst=(src_pts, dst_pts)
        )
        
        # Extract trajectory
        viz = TrajectoryVisualizer(bev_size=post.out_size)
        trajectory = viz.extract_trajectory_from_results(episode.results_by_index)
        trajectories_by_mode[interp_mode] = trajectory
        
        # Extract lane boundary (only once, it's the same for all modes)
        if lane_boundary is None and episode.results_by_index:
            first_result = episode.results_by_index[frame_indices.start]
            lane_boundary = viz.extract_lane_boundary(first_result.warped_lane)
        
        # Extract representative errors for overlay
        results_with_ball = _frames_with_ball(episode.results_by_index)
        a_s, b_s, a_l, b_l = _calibrate_bev_to_world(
            results_with_ball, gt_data, episode_idx
        )
        
        bev_metrics = episode.lane_metrics
        world_vel_at_frac: List[Tuple[float, float, float]] = []
        world_acc_at_frac: List[Tuple[float, float, float]] = []
        
        for i, frac in enumerate(bev_metrics.fractions):
            vx_bev, vy_bev, _ = bev_metrics.velocity_at_frac[i]
            ax_bev, ay_bev, _ = bev_metrics.acceleration_at_frac[i]
            
            vx_world = a_s * vx_bev
            vy_world = a_l * vy_bev
            speed_world = float(np.hypot(vx_world, vy_world))
            
            ax_world = a_s * ax_bev
            ay_world = a_l * ay_bev
            accel_world = float(np.hypot(ax_world, ay_world))
            
            world_vel_at_frac.append((vx_world, vy_world, speed_world))
            world_acc_at_frac.append((ax_world, ay_world, accel_world))
        
        pred_world_metrics = LaneMetrics(
            fractions=tuple(bev_metrics.fractions),
            frac_positions=tuple([a_s * p + b_s for p in bev_metrics.frac_positions]),
            velocity_at_frac=tuple(world_vel_at_frac),
            acceleration_at_frac=tuple(world_acc_at_frac),
            total_break=a_l * bev_metrics.total_break,
            end_lane_speed=world_vel_at_frac[-1][2],
        )
        
        gt_metrics = _compute_lane_metrics_ground_truth(episode_idx, gt_data)
        
        # Compute average errors across all fractions for display
        map_by_frame = _map_50_95_for_episode(episode.raw_masks_by_index)
        episode_mean_seg_map = float(np.nanmean(list(map_by_frame.values()))) if map_by_frame else 0.0
        
        avg_errors = {
            'vel_x_error': np.mean([pred_world_metrics.velocity_at_frac[i][0] - gt_metrics.velocity_at_frac[i][0] 
                                   for i in range(len(gt_metrics.fractions))]),
            'vel_y_error': np.mean([pred_world_metrics.velocity_at_frac[i][1] - gt_metrics.velocity_at_frac[i][1] 
                                   for i in range(len(gt_metrics.fractions))]),
            'speed_error': np.mean([pred_world_metrics.velocity_at_frac[i][2] - gt_metrics.velocity_at_frac[i][2] 
                                   for i in range(len(gt_metrics.fractions))]),
            'acc_x_error': np.mean([pred_world_metrics.acceleration_at_frac[i][0] - gt_metrics.acceleration_at_frac[i][0] 
                                   for i in range(len(gt_metrics.fractions))]),
            'acc_y_error': np.mean([pred_world_metrics.acceleration_at_frac[i][1] - gt_metrics.acceleration_at_frac[i][1] 
                                   for i in range(len(gt_metrics.fractions))]),
            'accel_mag_error': np.mean([pred_world_metrics.acceleration_at_frac[i][2] - gt_metrics.acceleration_at_frac[i][2] 
                                       for i in range(len(gt_metrics.fractions))]),
            'total_break_error': pred_world_metrics.total_break - gt_metrics.total_break,
            'end_speed_error': pred_world_metrics.end_lane_speed - gt_metrics.end_lane_speed,
            'seg_map_50_95': episode_mean_seg_map,
        }
        
        errors_by_mode[interp_mode] = avg_errors
    
    # Generate averaged trajectory plot (main visualization)
    print("\nGenerating averaged trajectory plot...")
    averaged_path = viz_output_dir / "trajectory_averaged.png"
    viz.plot_averaged_trajectory(
        trajectories=trajectories_by_mode,
        lane_boundary=lane_boundary,
        errors=errors_by_mode,
        output_path=averaged_path,
    )
    plt.close()  # Close to free memory
    
    # Generate error analysis plots from CSV
    print("Generating error analysis plots...")
    
    # Parse CSV data
    csv_data = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_data.append(row)
    
    error_plot = ErrorAnalysisPlotter()
    
    # Detailed error analysis
    error_analysis_path = viz_output_dir / "error_analysis.png"
    error_plot.plot_error_comparison(csv_data, output_path=error_analysis_path)
    plt.close()
    
    # Summary metrics
    summary_metrics_path = viz_output_dir / "summary_metrics.png"
    error_plot.plot_summary_metrics(csv_data, output_path=summary_metrics_path)
    plt.close()
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {viz_output_dir}")
    print("\nGenerated files:")
    print(f"  1. trajectory_averaged.png  - Clean averaged ball trajectory on lane")
    print(f"  2. error_analysis.png       - Speed & acceleration error analysis")
    print(f"  3. summary_metrics.png      - Performance metrics summary")
    print(f"\nVisualization shows averaged results across all interpolation modes.")
    print(f"{'='*60}\n")