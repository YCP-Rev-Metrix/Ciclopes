# Processing prototype and algorithm test harness

This directory is an earlier modular lane/ball processor. It is useful for algorithm experiments and tests, but the production API implementation is under [`../Ciclopes-API/core/LaneBalls/`](../Ciclopes-API/core/LaneBalls/), and current model training is [`../pipeline_v2/`](../pipeline_v2/).

## Files

| File | Purpose |
|---|---|
| `processor.py` | Facade that runs preprocessing followed by post-processing |
| `preprocessing/preprocessor.py` | Runs YOLO segmentation and converts results into lane/ball masks |
| `postprocessing/postprocessor.py` | Perspective warp, temporal buffers, metric path, and lane metrics |
| `postprocessing/visualizer.py` | Debug/result visualization |
| `*_test.py` | Unit/integration tests, including synthetic curved-trajectory cases |
| `test_*.csv` | Small checked-in fixtures/results for regression work |

## Flow

```text
video/frame source
  -> YOLOSegPreprocessor
  -> class masks (lane and ball)
  -> PostProcessor
       -> perspective warp
       -> buffered/stabilized positions
       -> metric lane coordinates
       -> trajectory/kinematic metrics
  -> ProcessResult / visualizer
```

`YOLOSegPreprocessor` accepts an Ultralytics model plus inference configuration (`imgsz`, confidence, IoU, and device), calls prediction, and rasterizes/collects the segmentation outputs in the structure expected by the postprocessor.

## Prototype post-processing

`Warp` estimates and applies a perspective transform between the detected lane geometry and a top-down destination. `BufferCalcs` holds recent timestamped positions and performs temporal calculations rather than trusting a single noisy frame. `PostProcessor` coordinates the warp, converts image positions into lane-relative coordinates, computes `LaneMetrics`, and returns `ProcessResult` for visualization/testing.

The newer API algorithm goes further: dense boundary fitting, multiple-lane temporal tracking, explicit trapezoid quality scores, homography health metrics, motion-interval trimming, isotonic interpolation, bounded hook extrapolation, and per-quarter kinematics. Use the API README for that maintained algorithm.

## Run tests

From repository root:

```bash
python -m pytest processing -q
```

Some curved-dataset tests expect local synthetic data that is ignored by Git. A missing fixture dataset is different from an algorithm failure; inspect the test's expected `data/` path before debugging code.

When changing shared ideas, compare behavior in all three places:

1. this prototype;
2. `pipeline_v2/detection_seg_overlay_video_gen.py`; and
3. `Ciclopes-API/core/LaneBalls/`.

Do not assume a passing prototype test proves the API copy changed.
