"""
Parse sensor JSON from the RevMetrix smart ball to determine ball contact frame.

The sensor JSON contains:
- BufferData: list of accelerometer/gyro samples with timestamps
  - BufferData[0].Timestamp = recording start time
- AccelerometerJumpTimestamps: list of timestamps where derivative spikes
  - AccelerometerJumpTimestamps[0] = ball contact point (first derivative)

From the time delta and video FPS we derive the ball contact frame index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger("ciclopes.sensor_parser")


@dataclass
class SensorFrameInfo:
    ball_contact_frame: int
    dt_seconds: float


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp, handling variable fractional-second precision."""
    # Python's fromisoformat handles most ISO formats in 3.11+,
    # but on older versions we may need to truncate fractional digits to 6.
    # The sensor JSON uses 7-digit fractional seconds (100ns ticks).
    ts = ts.strip()
    # Split off timezone portion, truncate fractional seconds, reassemble
    if "." in ts:
        base, rest = ts.split(".", 1)
        # rest might be "8261380-04:00" or "826138-04:00"
        frac = ""
        tz = ""
        for i, ch in enumerate(rest):
            if ch in ("+", "-", "Z"):
                frac = rest[:i]
                tz = rest[i:]
                break
        else:
            frac = rest
            tz = ""
        # Truncate to 6 digits (microseconds)
        frac = frac[:6].ljust(6, "0")
        ts = f"{base}.{frac}{tz}"
    return datetime.fromisoformat(ts)


def compute_ball_contact_frame(
    sensor_data: Dict[str, Any],
    fps: float,
) -> Optional[SensorFrameInfo]:
    """
    Given parsed sensor JSON and the video FPS, compute the ball contact frame.

    Returns None if the sensor data is missing required fields.
    """
    buffer_data = sensor_data.get("BufferData")
    jump_timestamps = sensor_data.get("AccelerometerJumpTimestamps")

    if not buffer_data or not jump_timestamps:
        logger.warning("Sensor data missing BufferData or AccelerometerJumpTimestamps")
        return None

    try:
        recording_start = _parse_timestamp(buffer_data[0]["Timestamp"])
        ball_contact_time = _parse_timestamp(jump_timestamps[0])
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("Failed to parse sensor timestamps: %s", exc)
        return None

    dt = (ball_contact_time - recording_start).total_seconds()
    if dt < 0:
        logger.warning("Ball contact time is before recording start (dt=%.4fs)", dt)
        return None

    contact_frame = int(dt * fps)
    logger.info(
        "Sensor: recording_start=%s  ball_contact=%s  dt=%.4fs  fps=%.1f  contact_frame=%d",
        recording_start.isoformat(),
        ball_contact_time.isoformat(),
        dt,
        fps,
        contact_frame,
    )
    return SensorFrameInfo(ball_contact_frame=contact_frame, dt_seconds=dt)
