# Android inference bridge contract

The web frontend remains shared JavaScript/HTML. Android performs the ONNX
inference in a native Android layer and returns decoded lane candidates to the
web UI. Assuming the app uses Capacitor, expose this through a custom Android
Capacitor plugin. Keep the analysis payload independent of Capacitor so the
inference core can still be reused if the packaging choice changes.

## Native capability

Expose a plugin named `CiclopesLaneAnalysis` with:

- `isAvailable()`: whether the Android ONNX runtime and compatible model are
  ready;
- `analyzeFrame({frameUri, width, height, rotation})`: run one selected video
  frame through local detection, returning normalized lane candidates, ball
  boxes, and health signals; and
- `applyLaneCorrection({corners, coordinateSpace, frameWidth, frameHeight})`:
  validate a user-adjusted normalized quadrilateral and return its homography
  into lane-meter coordinates; and
- `release()`: free the ONNX session and native buffers when analysis ends.

The JavaScript wrapper can declare this interface with Capacitor's
`registerPlugin`:

```ts
import { Capacitor, registerPlugin } from '@capacitor/core';

export interface CiclopesLaneAnalysisPlugin {
  isAvailable(): Promise<{ available: boolean }>;
  analyzeFrame(options: {
    frameUri: string;
    width: number;
    height: number;
    rotation: 0 | 90 | 180 | 270;
  }): Promise<LaneFrameResult>;
  applyLaneCorrection(options: {
    corners: Array<{ x: number; y: number }>;
    coordinateSpace: 'normalized_frame';
    frameWidth: number;
    frameHeight: number;
  }): Promise<{
    accepted: true;
    homography: number[];
    laneWidthMeters: number;
    laneLengthMeters: number;
  }>;
  release(): Promise<void>;
}

export interface LaneFrameResult {
  frameWidth: number;
  frameHeight: number;
  candidates: Array<{
    confidence: number;
    corners: Array<{ x: number; y: number }>;
  }>;
  balls: Array<{
    confidence: number;
    box: [number, number, number, number];
  }>;
  health: Record<string, number>;
}

export const CiclopesLaneAnalysis =
  registerPlugin<CiclopesLaneAnalysisPlugin>('CiclopesLaneAnalysis');

export async function canUseLocalLaneAnalysis(): Promise<boolean> {
  if (Capacitor.getPlatform() !== 'android') return false;
  return (await CiclopesLaneAnalysis.isAvailable()).available;
}
```

Call the native plugin only on Android. On iPhone, keep using the frontend's
existing analysis path; do not register an iOS stub that implies local ONNX
support exists.

The `frame` transport should avoid base64 if the chosen wrapper can pass a file
URI, shared buffer, or native camera frame. Keep video decoding and inference
off the UI thread. Return only compact candidate geometry and diagnostics to
JavaScript, not full-resolution segmentation masks unless the overlay needs
them.

## Response shape

Coordinates are normalized against the upright frame returned by
`analyzeFrame` (rotation has already been applied). Corner order is top-left,
top-right, bottom-right, bottom-left. The frontend should submit the same
`frameWidth` and `frameHeight` with the correction so native code can build a
homography from those exact frame coordinates.

```json
{
  "frameWidth": 1920,
  "frameHeight": 1080,
  "candidates": [
    {
      "confidence": 0.91,
      "corners": [
        {"x": 0.41, "y": 0.22},
        {"x": 0.59, "y": 0.22},
        {"x": 0.83, "y": 0.98},
        {"x": 0.17, "y": 0.98}
      ]
    }
  ],
  "balls": [
    {"confidence": 0.84, "box": [0.48, 0.71, 0.53, 0.78]}
  ],
  "health": {"inferenceMs": 42, "laneCount": 1, "ballCount": 1}
}
```

When detection is empty or fails a calibrated quality gate, JavaScript should
show a correction UI: choose a detected candidate when there are several, or
drag the four lane corners. `applyLaneCorrection` validates the normalized
quadrilateral and returns the homography mapping frame pixels to the regulation
lane rectangle (1.0541 m by 18.288 m). The current native slice does not yet
run ball tracking and kinematics across a video with that homography; the
frontend should keep corrections associated with the video/frame used to
define them.

## Android runtime notes

Implement the plugin using Capacitor's Android `Plugin` and `@PluginMethod`
API, and add ONNX Runtime's Android package
(`com.microsoft.onnxruntime:onnxruntime-android`) to the Android app/plugin.
Start with CPU or XNNPACK; benchmark NNAPI on
the target phones before choosing it, since accelerator performance depends on
model operators and device support. Keep the ONNX file as an Android app asset
or managed model file. Don't add the model to the watch application.

The initial Capacitor implementation lives in `mobile/capacitor-plugin`. It
uses the model's `[1,3,640,640]` RGB float32 input, letterboxes, decodes the
YOLO26 end-to-end `[1,300,38]` detection output and `[1,32,160,160]` mask
prototypes, and returns lane candidates and ball boxes in normalized upright
frame coordinates. The 300 rows are already end-to-end detections, so there is
no extra NMS step. Accuracy and performance still need comparison against the
Python pipeline and real Android phones.
