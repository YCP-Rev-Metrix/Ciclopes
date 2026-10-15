package com.revmetrix.ciclopes.laneanalysis;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.net.Uri;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** Android frame preprocessing and YOLO26 segmentation output adapter. */
final class LaneInferenceEngine implements AutoCloseable {
    static final int INPUT_SIZE = 640;
    static final int PROTO_SIZE = 160;
    static final int PROTO_CHANNELS = 32;
    static final int MAX_DETECTIONS = 300;
    static final int ROW_SIZE = 6 + PROTO_CHANNELS;
    static final float CONFIDENCE_THRESHOLD = 0.10f;
    private static final String MODEL_ASSET = "ciclopes_lane_ball.onnx";

    static final class LaneCandidate {
        final float confidence;
        final double[][] corners;
        LaneCandidate(float confidence, double[][] corners) { this.confidence = confidence; this.corners = corners; }
    }

    static final class BallCandidate {
        final float confidence;
        final double[] box;
        BallCandidate(float confidence, double[] box) { this.confidence = confidence; this.box = box; }
    }

    static final class FrameResult {
        final int width, height;
        final long inferenceMs;
        final List<LaneCandidate> lanes;
        final List<BallCandidate> balls;
        FrameResult(int width, int height, long inferenceMs, List<LaneCandidate> lanes, List<BallCandidate> balls) {
            this.width = width; this.height = height; this.inferenceMs = inferenceMs; this.lanes = lanes; this.balls = balls;
        }
    }

    private static final class Letterbox {
        final Bitmap bitmap;
        final float scale, padX, padY;
        Letterbox(Bitmap bitmap, float scale, float padX, float padY) { this.bitmap = bitmap; this.scale = scale; this.padX = padX; this.padY = padY; }
    }

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final String inputName;

    LaneInferenceEngine(Context context) throws Exception {
        File model = new File(context.getFilesDir(), MODEL_ASSET);
        if (!model.isFile()) {
            try (InputStream in = context.getAssets().open(MODEL_ASSET); FileOutputStream out = new FileOutputStream(model)) {
                byte[] buffer = new byte[1024 * 1024];
                int count;
                while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
            }
        }
        environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(Math.max(2, Runtime.getRuntime().availableProcessors() / 2));
        session = environment.createSession(model.getAbsolutePath(), options);
        options.close();
        if (session.getInputNames().size() != 1) throw new IllegalStateException("Expected one ONNX input");
        inputName = session.getInputNames().iterator().next();
        validateModelShape();
    }

    private void validateModelShape() throws Exception {
        TensorInfo info = (TensorInfo) session.getInputInfo().get(inputName).getInfo();
        long[] shape = info.getShape();
        if (shape.length != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != INPUT_SIZE || shape[3] != INPUT_SIZE) {
            throw new IllegalStateException("Expected model input [1,3,640,640], got " + java.util.Arrays.toString(shape));
        }
        if (!session.getOutputNames().contains("output0") || !session.getOutputNames().contains("output1")) {
            throw new IllegalStateException("Expected YOLO26 outputs named output0 and output1");
        }
        long[] detections = ((TensorInfo) session.getOutputInfo().get("output0").getInfo()).getShape();
        long[] prototypes = ((TensorInfo) session.getOutputInfo().get("output1").getInfo()).getShape();
        if (!java.util.Arrays.equals(detections, new long[] {1, MAX_DETECTIONS, ROW_SIZE})
                || !java.util.Arrays.equals(prototypes, new long[] {1, PROTO_CHANNELS, PROTO_SIZE, PROTO_SIZE})) {
            throw new IllegalStateException("Unexpected YOLO26 output shapes: output0="
                + java.util.Arrays.toString(detections) + ", output1=" + java.util.Arrays.toString(prototypes));
        }
    }

    FrameResult analyze(Context context, Uri uri, int rotation) throws Exception {
        Bitmap decoded;
        try (InputStream input = context.getContentResolver().openInputStream(uri)) {
            if (input == null) throw new IllegalArgumentException("Could not open frame URI");
            decoded = BitmapFactoryCompat.decode(input);
        }
        if (decoded == null) throw new IllegalArgumentException("Frame URI did not contain a decodable image");
        Bitmap upright = rotate(decoded, rotation);
        if (decoded != upright) decoded.recycle();

        Letterbox prepared = letterbox(upright);
        long start = System.nanoTime();
        float[][][] predictions;
        float[][][][] prototypes;
        try (OnnxTensor inputTensor = makeInput(prepared.bitmap); OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor))) {
            predictions = (float[][][]) result.get("output0").get().getValue();
            prototypes = (float[][][][]) result.get("output1").get().getValue();
        } finally {
            prepared.bitmap.recycle();
        }
        long elapsed = (System.nanoTime() - start) / 1_000_000L;
        List<LaneCandidate> lanes = new ArrayList<>();
        List<BallCandidate> balls = new ArrayList<>();
        decode(predictions[0], prototypes[0], prepared, upright.getWidth(), upright.getHeight(), lanes, balls);
        FrameResult frame = new FrameResult(upright.getWidth(), upright.getHeight(), elapsed, lanes, balls);
        upright.recycle();
        return frame;
    }

    private static OnnxTensor makeInput(Bitmap bitmap) throws Exception {
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        FloatBuffer input = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer();
        int plane = INPUT_SIZE * INPUT_SIZE;
        for (int i = 0; i < plane; i++) input.put(i, Color.red(pixels[i]) / 255.0f);
        for (int i = 0; i < plane; i++) input.put(plane + i, Color.green(pixels[i]) / 255.0f);
        for (int i = 0; i < plane; i++) input.put(2 * plane + i, Color.blue(pixels[i]) / 255.0f);
        input.rewind();
        return OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), input, new long[] {1, 3, INPUT_SIZE, INPUT_SIZE});
    }

    private static Letterbox letterbox(Bitmap source) {
        float scale = Math.min(INPUT_SIZE / (float) source.getWidth(), INPUT_SIZE / (float) source.getHeight());
        int resizedW = Math.round(source.getWidth() * scale);
        int resizedH = Math.round(source.getHeight() * scale);
        float padX = (INPUT_SIZE - resizedW) / 2.0f;
        float padY = (INPUT_SIZE - resizedH) / 2.0f;
        Bitmap target = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(target);
        canvas.drawColor(Color.rgb(114, 114, 114));
        Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG | Paint.DITHER_FLAG);
        canvas.drawBitmap(source, null, new android.graphics.RectF(padX, padY, padX + resizedW, padY + resizedH), paint);
        return new Letterbox(target, scale, padX, padY);
    }

    private static Bitmap rotate(Bitmap source, int degrees) {
        if (degrees == 0) return source;
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    private static void decode(float[][] rows, float[][][] proto, Letterbox transform, int frameW, int frameH,
                               List<LaneCandidate> lanes, List<BallCandidate> balls) {
        int count = Math.min(rows.length, MAX_DETECTIONS);
        for (int i = 0; i < count; i++) {
            float[] row = rows[i];
            if (row.length < ROW_SIZE) continue;
            float confidence = row[4];
            int classId = Math.round(row[5]);
            if (!Float.isFinite(confidence) || confidence < CONFIDENCE_THRESHOLD) continue;
            float x1 = clamp(row[0], 0, INPUT_SIZE), y1 = clamp(row[1], 0, INPUT_SIZE);
            float x2 = clamp(row[2], 0, INPUT_SIZE), y2 = clamp(row[3], 0, INPUT_SIZE);
            if (x2 <= x1 || y2 <= y1) continue;
            if (classId == 0) {
                balls.add(new BallCandidate(confidence, new double[] {
                    normX(x1, transform, frameW), normY(y1, transform, frameH),
                    normX(x2, transform, frameW), normY(y2, transform, frameH)
                }));
            } else if (classId == 1) {
                double[][] quad = laneQuad(row, proto, x1, y1, x2, y2, transform, frameW, frameH);
                if (quad != null) lanes.add(new LaneCandidate(confidence, quad));
            }
        }
    }

    private static double[][] laneQuad(float[] row, float[][][] proto, float x1, float y1, float x2, float y2,
                                       Letterbox transform, int frameW, int frameH) {
        int minX = Math.max(0, (int) Math.floor(x1 * PROTO_SIZE / INPUT_SIZE));
        int maxX = Math.min(PROTO_SIZE - 1, (int) Math.ceil(x2 * PROTO_SIZE / INPUT_SIZE));
        int minY = Math.max(0, (int) Math.floor(y1 * PROTO_SIZE / INPUT_SIZE));
        int maxY = Math.min(PROTO_SIZE - 1, (int) Math.ceil(y2 * PROTO_SIZE / INPUT_SIZE));
        if (maxX <= minX || maxY <= minY) return null;

        int[] left = new int[PROTO_SIZE], right = new int[PROTO_SIZE];
        java.util.Arrays.fill(left, -1);
        java.util.Arrays.fill(right, -1);
        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                float logit = 0f;
                for (int c = 0; c < PROTO_CHANNELS; c++) logit += row[6 + c] * proto[c][y][x];
                float prob = (float) (1.0 / (1.0 + Math.exp(-Math.max(-30.0, Math.min(30.0, logit)))));
                if (prob >= 0.5f) {
                    if (left[y] < 0) left[y] = x;
                    right[y] = x;
                }
            }
        }

        int top = -1, bottom = -1;
        for (int y = minY; y <= maxY; y++) if (left[y] >= 0 && right[y] > left[y]) { top = y; break; }
        for (int y = maxY; y >= minY; y--) if (left[y] >= 0 && right[y] > left[y]) { bottom = y; break; }
        if (top < 0 || bottom - top < 8) return null;
        int band = Math.max(2, Math.round((bottom - top) * 0.12f));
        double lTop = averageBoundary(left, top, Math.min(bottom, top + band));
        double rTop = averageBoundary(right, top, Math.min(bottom, top + band));
        double lBottom = averageBoundary(left, Math.max(top, bottom - band), bottom);
        double rBottom = averageBoundary(right, Math.max(top, bottom - band), bottom);
        if (lTop < 0 || rTop <= lTop || lBottom < 0 || rBottom <= lBottom) return null;

        double[][] quad = {
            sourcePoint(lTop * INPUT_SIZE / PROTO_SIZE, top * INPUT_SIZE / PROTO_SIZE, transform, frameW, frameH),
            sourcePoint(rTop * INPUT_SIZE / PROTO_SIZE, top * INPUT_SIZE / PROTO_SIZE, transform, frameW, frameH),
            sourcePoint(rBottom * INPUT_SIZE / PROTO_SIZE, bottom * INPUT_SIZE / PROTO_SIZE, transform, frameW, frameH),
            sourcePoint(lBottom * INPUT_SIZE / PROTO_SIZE, bottom * INPUT_SIZE / PROTO_SIZE, transform, frameW, frameH)
        };
        try { LaneGeometry.validateQuad(quad); } catch (IllegalArgumentException invalid) { return null; }
        return quad;
    }

    private static double averageBoundary(int[] values, int from, int to) {
        long sum = 0; int n = 0;
        for (int y = Math.max(0, from); y <= Math.min(PROTO_SIZE - 1, to); y++) if (values[y] >= 0) { sum += values[y]; n++; }
        return n == 0 ? -1 : sum / (double) n;
    }

    private static double[] sourcePoint(double modelX, double modelY, Letterbox t, int width, int height) {
        return new double[] {
            clamp((modelX - t.padX) / t.scale / width, 0, 1),
            clamp((modelY - t.padY) / t.scale / height, 0, 1)
        };
    }
    private static double normX(float x, Letterbox t, int width) { return clamp((x - t.padX) / t.scale / width, 0, 1); }
    private static double normY(float y, Letterbox t, int height) { return clamp((y - t.padY) / t.scale / height, 0, 1); }
    private static float clamp(float v, float lo, float hi) { return Math.max(lo, Math.min(hi, v)); }
    private static double clamp(double v, double lo, double hi) { return Math.max(lo, Math.min(hi, v)); }

    @Override public void close() throws Exception { session.close(); }

    /** Keeps BitmapFactory import isolated so the engine's public API remains small. */
    private static final class BitmapFactoryCompat {
        static Bitmap decode(InputStream input) { return android.graphics.BitmapFactory.decodeStream(input); }
    }
}
