package com.revmetrix.ciclopes.laneanalysis;

import android.net.Uri;

import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@CapacitorPlugin(name = "CiclopesLaneAnalysis")
public final class CiclopesLaneAnalysisPlugin extends Plugin {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private LaneInferenceEngine inference;
    private double[] activeHomography;

    @PluginMethod
    public void isAvailable(PluginCall call) {
        worker.execute(() -> {
            try {
                inferenceEngine();
                JSObject result = new JSObject();
                result.put("available", true);
                call.resolve(result);
            } catch (Exception error) {
                JSObject result = new JSObject();
                result.put("available", false);
                result.put("reason", error.getMessage() == null ? "Could not load the ONNX model" : error.getMessage());
                call.resolve(result);
            }
        });
    }

    @PluginMethod
    public void analyzeFrame(PluginCall call) {
        String frameUri = call.getString("frameUri");
        Integer width = call.getInt("width");
        Integer height = call.getInt("height");
        Integer rotation = call.getInt("rotation", 0);
        if (frameUri == null || frameUri.isEmpty() || width == null || height == null || width <= 0 || height <= 0) {
            call.reject("frameUri, positive width, and positive height are required");
            return;
        }
        if (rotation == null || (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270)) {
            call.reject("rotation must be 0, 90, 180, or 270 degrees");
            return;
        }
        worker.execute(() -> {
            try {
                LaneInferenceEngine.FrameResult frame = inferenceEngine().analyze(getContext(), Uri.parse(frameUri), rotation);
                JSObject result = new JSObject();
                result.put("frameWidth", frame.width);
                result.put("frameHeight", frame.height);
                JSArray candidates = new JSArray();
                for (LaneInferenceEngine.LaneCandidate candidate : frame.lanes) {
                    JSObject item = new JSObject();
                    item.put("confidence", candidate.confidence);
                    item.put("corners", points(candidate.corners));
                    candidates.put(item);
                }
                JSArray balls = new JSArray();
                for (LaneInferenceEngine.BallCandidate ball : frame.balls) {
                    JSObject item = new JSObject();
                    item.put("confidence", ball.confidence);
                    JSArray box = new JSArray();
                    for (double coordinate : ball.box) box.put(coordinate);
                    item.put("box", box);
                    balls.put(item);
                }
                result.put("candidates", candidates);
                result.put("balls", balls);
                JSObject health = new JSObject();
                health.put("inferenceMs", frame.inferenceMs);
                health.put("laneCount", frame.lanes.size());
                health.put("ballCount", frame.balls.size());
                result.put("health", health);
                call.resolve(result);
            } catch (Exception error) {
                call.reject("Frame analysis failed: " + (error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage()), error);
            }
        });
    }

    @PluginMethod
    public void applyLaneCorrection(PluginCall call) {
        JSArray corners = call.getArray("corners");
        String coordinateSpace = call.getString("coordinateSpace");
        Integer frameWidth = call.getInt("frameWidth");
        Integer frameHeight = call.getInt("frameHeight");
        if (!"normalized_frame".equals(coordinateSpace) || corners == null || frameWidth == null || frameHeight == null) {
            call.reject("corners, coordinateSpace='normalized_frame', frameWidth, and frameHeight are required");
            return;
        }
        try {
            if (corners.length() != 4) throw new IllegalArgumentException("Exactly four corners are required");
            double[][] normalized = new double[4][2];
            for (int i = 0; i < 4; i++) {
                JSONObject point = corners.getJSONObject(i);
                normalized[i][0] = point.getDouble("x");
                normalized[i][1] = point.getDouble("y");
            }
            activeHomography = LaneGeometry.homography(normalized, frameWidth, frameHeight);
            JSObject result = new JSObject();
            result.put("accepted", true);
            JSArray matrix = new JSArray();
            for (double value : activeHomography) matrix.put(value);
            result.put("homography", matrix);
            result.put("laneWidthMeters", LaneGeometry.LANE_WIDTH_METERS);
            result.put("laneLengthMeters", LaneGeometry.LANE_LENGTH_METERS);
            call.resolve(result);
        } catch (Exception error) {
            call.reject("Lane correction is invalid: " + error.getMessage(), error);
        }
    }

    @PluginMethod
    public void release(PluginCall call) {
        worker.execute(() -> {
            if (inference != null) {
                try { inference.close(); } catch (Exception ignored) { }
                inference = null;
            }
            activeHomography = null;
            call.resolve();
        });
    }

    private synchronized LaneInferenceEngine inferenceEngine() throws Exception {
        if (inference == null) inference = new LaneInferenceEngine(getContext());
        return inference;
    }

    private static JSArray points(double[][] coords) throws JSONException {
        JSArray output = new JSArray();
        for (double[] coord : coords) {
            JSObject point = new JSObject();
            point.put("x", coord[0]);
            point.put("y", coord[1]);
            output.put(point);
        }
        return output;
    }
}
