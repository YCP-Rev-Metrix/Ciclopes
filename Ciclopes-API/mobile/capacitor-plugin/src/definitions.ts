export interface Point {
  x: number;
  y: number;
}

export interface LaneCandidate {
  confidence: number;
  /** top-left, top-right, bottom-right, bottom-left in normalized upright-frame coordinates */
  corners: [Point, Point, Point, Point];
}

export interface BallCandidate {
  confidence: number;
  /** normalized [left, top, right, bottom] in upright-frame coordinates */
  box: [number, number, number, number];
}

export interface LaneFrameResult {
  frameWidth: number;
  frameHeight: number;
  candidates: LaneCandidate[];
  balls: BallCandidate[];
  health: {
    inferenceMs: number;
    laneCount: number;
    ballCount: number;
  };
}

export interface LaneCorrectionResult {
  accepted: true;
  /** 3x3 row-major projective transform mapping source pixels to lane meters. */
  homography: number[];
  laneWidthMeters: number;
  laneLengthMeters: number;
}

export interface CiclopesLaneAnalysisPlugin {
  isAvailable(): Promise<{ available: boolean; reason?: string }>;
  analyzeFrame(options: {
    frameUri: string;
    width: number;
    height: number;
    rotation: 0 | 90 | 180 | 270;
  }): Promise<LaneFrameResult>;
  applyLaneCorrection(options: {
    corners: [Point, Point, Point, Point];
    coordinateSpace: 'normalized_frame';
    frameWidth: number;
    frameHeight: number;
  }): Promise<LaneCorrectionResult>;
  release(): Promise<void>;
}
