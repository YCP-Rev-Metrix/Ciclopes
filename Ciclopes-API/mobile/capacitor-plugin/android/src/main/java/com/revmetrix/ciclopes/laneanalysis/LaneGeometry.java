package com.revmetrix.ciclopes.laneanalysis;

import java.util.Arrays;

/** Geometry utilities shared by automatic lane proposals and manual corrections. */
final class LaneGeometry {
    static final double LANE_WIDTH_METERS = 1.0541;
    static final double LANE_LENGTH_METERS = 18.288;

    private LaneGeometry() {}

    static double[] homography(double[][] src, int width, int height) {
        validateQuad(src);
        if (width <= 0 || height <= 0) throw new IllegalArgumentException("Frame dimensions must be positive");

        double[][] source = new double[4][2];
        for (int i = 0; i < 4; i++) {
            source[i][0] = src[i][0] * width;
            source[i][1] = src[i][1] * height;
        }
        double[][] target = {
            {0.0, LANE_LENGTH_METERS},
            {LANE_WIDTH_METERS, LANE_LENGTH_METERS},
            {LANE_WIDTH_METERS, 0.0},
            {0.0, 0.0}
        };

        // Solve the eight projective parameters with h[8] fixed at 1.
        double[][] a = new double[8][9];
        for (int i = 0; i < 4; i++) {
            double x = source[i][0], y = source[i][1];
            double u = target[i][0], v = target[i][1];
            int r = 2 * i;
            a[r] = new double[] {x, y, 1, 0, 0, 0, -u * x, -u * y, u};
            a[r + 1] = new double[] {0, 0, 0, x, y, 1, -v * x, -v * y, v};
        }
        double[] h = solve(a);
        double[] matrix = Arrays.copyOf(h, 9);
        matrix[8] = 1.0;
        for (double value : matrix) {
            if (!Double.isFinite(value)) throw new IllegalArgumentException("Correction produces an invalid homography");
        }
        return matrix;
    }

    static void validateQuad(double[][] points) {
        if (points == null || points.length != 4) throw new IllegalArgumentException("Exactly four corners are required");
        for (double[] p : points) {
            if (p == null || p.length != 2 || !Double.isFinite(p[0]) || !Double.isFinite(p[1])) {
                throw new IllegalArgumentException("Each corner must have finite x and y coordinates");
            }
            if (p[0] < 0 || p[0] > 1 || p[1] < 0 || p[1] > 1) {
                throw new IllegalArgumentException("Normalized corners must be within [0, 1]");
            }
        }
        double signedArea = 0.0;
        double sign = 0.0;
        for (int i = 0; i < 4; i++) {
            double[] p = points[i], q = points[(i + 1) % 4], r = points[(i + 2) % 4];
            signedArea += p[0] * q[1] - q[0] * p[1];
            double cross = (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0]);
            if (Math.abs(cross) < 1e-8) throw new IllegalArgumentException("Lane corners must form a non-degenerate quadrilateral");
            double currentSign = Math.signum(cross);
            if (sign == 0.0) sign = currentSign;
            else if (sign != currentSign) throw new IllegalArgumentException("Lane corners must form a convex quadrilateral in TL, TR, BR, BL order");
        }
        if (Math.abs(signedArea) < 1e-6) throw new IllegalArgumentException("Lane quadrilateral has near-zero area");
    }

    private static double[] solve(double[][] a) {
        int n = 8;
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
            }
            if (Math.abs(a[pivot][col]) < 1e-10) throw new IllegalArgumentException("Lane corners do not define a stable homography");
            double[] swap = a[col]; a[col] = a[pivot]; a[pivot] = swap;
            double divisor = a[col][col];
            for (int j = col; j <= n; j++) a[col][j] /= divisor;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = a[row][col];
                for (int j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = a[i][n];
        return out;
    }
}
