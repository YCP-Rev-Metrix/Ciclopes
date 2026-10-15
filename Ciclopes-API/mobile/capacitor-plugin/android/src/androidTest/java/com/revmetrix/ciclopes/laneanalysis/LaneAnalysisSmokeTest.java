package com.revmetrix.ciclopes.laneanalysis;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.net.Uri;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

@RunWith(AndroidJUnit4.class)
public final class LaneAnalysisSmokeTest {
    @Test
    public void manualLaneCornersProduceExpectedProjectiveMapping() {
        double[][] corners = {
            {0.35, 0.18}, {0.65, 0.18}, {0.92, 0.95}, {0.08, 0.95}
        };
        double[] h = LaneGeometry.homography(corners, 640, 480);

        assertEquals(9, h.length);
        assertMapsTo(h, corners[0], 640, 480, 0.0, LaneGeometry.LANE_LENGTH_METERS);
        assertMapsTo(h, corners[1], 640, 480, LaneGeometry.LANE_WIDTH_METERS, LaneGeometry.LANE_LENGTH_METERS);
        assertMapsTo(h, corners[2], 640, 480, LaneGeometry.LANE_WIDTH_METERS, 0.0);
        assertMapsTo(h, corners[3], 640, 480, 0.0, 0.0);
    }

    @Test
    public void rejectsCrossedCorners() {
        double[][] crossed = {
            {0.2, 0.2}, {0.8, 0.8}, {0.8, 0.2}, {0.2, 0.8}
        };
        boolean rejected = false;
        try {
            LaneGeometry.homography(crossed, 640, 480);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        assertTrue("crossed corners should be rejected", rejected);
    }

    @Test
    public void onnxModelRunsOnBundledFrame() throws Exception {
        Context target = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
        File frame = new File(target.getCacheDir(), "lane-smoke-frame.png");
        try (InputStream input = testContext.getAssets().open("lane-smoke-frame.png");
             FileOutputStream output = new FileOutputStream(frame)) {
            byte[] bytes = new byte[8192];
            int count;
            while ((count = input.read(bytes)) >= 0) output.write(bytes, 0, count);
        }

        try (LaneInferenceEngine engine = new LaneInferenceEngine(target)) {
            LaneInferenceEngine.FrameResult result = engine.analyze(target, Uri.fromFile(frame), 0);
            assertEquals(640, result.width);
            assertEquals(480, result.height);
            assertTrue("inference duration should be positive", result.inferenceMs > 0);
            assertNotNull(result.lanes);
            assertNotNull(result.balls);
            for (LaneInferenceEngine.LaneCandidate lane : result.lanes) {
                assertEquals(4, lane.corners.length);
                LaneGeometry.validateQuad(lane.corners);
            }
            assertFalse("fixture itself must exist", frame.length() == 0);
        } finally {
            // Keep the fixture out of persistent app data after the smoke test.
            //noinspection ResultOfMethodCallIgnored
            frame.delete();
        }
    }

    private static void assertMapsTo(double[] h, double[] normalized, int width, int height, double expectedX, double expectedY) {
        double x = normalized[0] * width;
        double y = normalized[1] * height;
        double denominator = h[6] * x + h[7] * y + h[8];
        double mappedX = (h[0] * x + h[1] * y + h[2]) / denominator;
        double mappedY = (h[3] * x + h[4] * y + h[5]) / denominator;
        assertEquals(expectedX, mappedX, 1e-4);
        assertEquals(expectedY, mappedY, 1e-4);
    }
}
