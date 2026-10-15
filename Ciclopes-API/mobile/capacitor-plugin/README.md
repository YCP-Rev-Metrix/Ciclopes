# Ciclopes Capacitor lane analysis plugin

This local plugin provides the first Android inference and manual-lane geometry
slice. It expects the host app to use Capacitor and package the ONNX model.

## Model and Android dependency

Export the fixed 640 model from the API directory:

```sh
python scripts/export_mobile_onnx.py
```

The exporter writes the ONNX model to `mobile/models` and stages a copy under
`android/src/main/assets`, where the Android library packages it with the
plugin. The model is intentionally gitignored and must be generated from the
checkpoint in each development environment. The Android module adds
`com.microsoft.onnxruntime:onnxruntime-android:1.31.0`; inference currently
uses the CPU execution provider.

Install this directory as a local npm package from the host frontend, then run
Capacitor's Android sync. Exact install commands depend on the frontend app's
package layout.

## Implemented native behavior

- Opens a frame URI, applies the requested rotation, and letterboxes RGB pixels
  into NCHW float32 640x640 input.
- Runs the model off the UI thread and decodes its YOLO26 end-to-end detections
  and 32-channel segmentation prototypes.
- Reconstructs lane masks to estimate four normalized lane corners; returns
  normalized ball boxes as well.
- Validates a user-supplied convex quadrilateral and computes a projective
  transform from upright frame pixels into regulation lane meters.

The corner estimate is an initial approximation of the Python mask geometry.
The plugin does not yet process a full video, project a ball track through the
homography, or calculate kinematics. Those are the next native-port steps.

## Android smoke tests

The `android/src/androidTest` suite checks corner validation and homography
mapping, then loads the packaged model and runs one bundled synthetic lane
frame. From the Frontend host's `android` directory, after local package install
and Capacitor sync, run:

```sh
./gradlew :revmetrix-ciclopes-lane-analysis:connectedDebugAndroidTest
```

The synthetic fixture only checks that ONNX Runtime and the decoder execute.
Use a representative bowling frame to judge detection accuracy and compare it
with the Python pipeline.
