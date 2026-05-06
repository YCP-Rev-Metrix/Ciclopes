from __future__ import annotations

import asyncio
import logging

import cv2
from fastapi import APIRouter, HTTPException, Request

from core.LaneBalls.Kinematics import compute_kinematics_per_quarter
from core.LaneBalls.models import BallPos
from core.MockDB.mock_db_reader import load_named_runs, load_shots
from core.MockDB.mock_db_writer import default_run_name, save_named_run_section
from core.SensorData.sensor_parser import compute_ball_contact_frame
from src.modules.Aggregated.models import (
    AggQueryInput,
    AggQueryOutput,
    AggRunInput,
    AggRunOutput,
    BallPoint,
    KinematicsRow,
    SkeletonPoint,
)

logger = logging.getLogger("ciclopes.aggregated_routes")

router = APIRouter(
    prefix="/agg",
    tags=["Aggregated"],
)


def _get_engine(request: Request):
    engine = getattr(request.app.state, "inference_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="InferenceEngine is not initialized")
    return engine


def _get_settings(request: Request):
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=503, detail="Application settings are not initialized")
    return settings


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@router.post("/run", response_model=AggRunOutput)
async def run_aggregate_pipeline(request: Request, payload: AggRunInput):
    from core.VideoUtil.FrameSplit import split_video_into_frames
    from core.VideoUtil.SpacesApiClient import query_video_via_api_to_temp_file

    engine = _get_engine(request)
    settings = _get_settings(request)

    if not settings.username or not settings.password:
        raise HTTPException(
            status_code=500,
            detail="Missing API credentials. Set username/password in Ciclopes-API/.env or environment.",
        )

    logger.info(
        "Running /agg/run for video_key=%s sd_key=%s",
        payload.video_key,
        payload.sd_key,
    )

    try:
        temp_ctx = query_video_via_api_to_temp_file(
            base=settings.api_base,
            verify_api=settings.verify_api,
            username=settings.username,
            password=settings.password,
            key=payload.video_key,
            ttl_seconds=settings.presign_ttl_seconds,
            verify_presigned=settings.verify_presigned,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Video query failed: {exc}")

    with temp_ctx as temp_path:
        split_video = split_video_into_frames(str(temp_path))

    if not split_video.frames:
        return AggRunOutput(ball_points=[], kinematics_table=[], skeleton_points=[])

    fps = float(split_video.fps) if split_video.fps > 0 else 30.0
    rgb_frames = [cv2.cvtColor(vf.image, cv2.COLOR_BGR2RGB) for vf in split_video.frames]

    # ── Determine frame ranges from sensor data (or use defaults) ─────────
    use_sensor = payload.sd_key != "key"
    lb_start_frame = settings.lane_ball_start_frame
    fourdbody_frames = rgb_frames

    if use_sensor:
        from core.VideoUtil.SpacesApiClient import query_json_via_api

        logger.info("Fetching sensor JSON from bucket: sd_key=%s", payload.sd_key)
        try:
            sensor_data = query_json_via_api(
                base=settings.api_base,
                verify_api=settings.verify_api,
                username=settings.username,
                password=settings.password,
                key=payload.sd_key,
                ttl_seconds=settings.presign_ttl_seconds,
                verify_presigned=settings.verify_presigned,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Sensor JSON query failed: {exc}")

        sensor_info = compute_ball_contact_frame(sensor_data, fps)
        if sensor_info is not None:
            lb_start_frame = sensor_info.ball_contact_frame
            # fourdbody: from start up to 1 second after ball contact
            fourdbody_end = sensor_info.ball_contact_frame + int(fps)
            fourdbody_frames = rgb_frames[:fourdbody_end]
            logger.info(
                "Sensor-derived frames: laneball_start=%d  fourdbody_end=%d (of %d total)",
                lb_start_frame,
                fourdbody_end,
                len(rgb_frames),
            )
        else:
            logger.warning("Could not parse sensor data; falling back to defaults")

    # Run both pipelines concurrently — they use separate thread-pool slots.
    lane_ball_coro = engine.forward_lane_ball(
        frames_rgb=rgb_frames,
        fps=fps,
        start_frame=lb_start_frame,
        batch_size=settings.lane_ball_batch_size,
    )
    sam3d_coro = engine.forward_sam3d_body(
        frames_rgb=fourdbody_frames,
        batch_size=settings.sam3d_body_batch_size,
    )

    try:
        lane_ball_output, sam3d_output = await asyncio.gather(
            lane_ball_coro, sam3d_coro
        )
    except Exception as exc:
        logger.exception("/agg/run pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # ── Lane-ball results ──────────────────────────────────────────────────
    health = lane_ball_output.get("health", {})
    pipeline_error = health.get("error")
    if pipeline_error:
        raise HTTPException(status_code=500, detail=str(pipeline_error))

    positions = lane_ball_output.get("positions", [])
    quarters = lane_ball_output.get("kinematics", {}).get("quarters", [])

    ball_points = [
        BallPoint(x=float(point.get("x_m", 0.0)), y=float(point.get("y_m", 0.0)))
        for point in positions
    ]

    kinematics_table = [
        KinematicsRow(
            quarter=int(row.get("quarter", 0)),
            start_m=float(row.get("start_m", 0.0)),
            end_m=float(row.get("end_m", 0.0)),
            mean_speed_mps=float(row.get("mean_speed_mps", 0.0)),
            mean_acceleration_mps2=float(row.get("mean_acceleration_mps2", 0.0)),
            sample_count=int(row.get("sample_count", 0)),
        )
        for row in quarters
    ]

    # ── SAM3D Body results ─────────────────────────────────────────────────
    # sam3d_output: list[list[dict]] — outer=frames, inner=joints per frame
    skeleton_points = [
        [
            SkeletonPoint(
                joint_id=int(j["joint_id"]),
                x=float(j["x"]),
                y=float(j["y"]),
                z=float(j["z"]),
            )
            for j in frame_joints
        ]
        for frame_joints in sam3d_output
    ]

    output = AggRunOutput(
        ball_points=ball_points,
        kinematics_table=kinematics_table,
        skeleton_points=skeleton_points,
    )
    run_name = payload.save_name or default_run_name(payload.video_key)
    try:
        save_named_run_section(
            name=run_name,
            section="aggregated",
            response=_model_to_dict(output),
            video_key=payload.video_key,
            sd_key=payload.sd_key,
        )
        save_named_run_section(
            name=run_name,
            section="laneballs",
            response={
                "fps": fps,
                "ball_points": [point.dict() if hasattr(point, "dict") else point for point in ball_points],
                "kinematics_table": [row.dict() if hasattr(row, "dict") else row for row in kinematics_table],
                "is_trapezoid": bool(lane_ball_output.get("is_trapezoid", False)),
                "homography_frame": lane_ball_output.get("homography_frame"),
                "health": health,
            },
            video_key=payload.video_key,
            sd_key=payload.sd_key,
        )
        save_named_run_section(
            name=run_name,
            section="fourdbody",
            response={
                "fps": fps,
                "skeleton_points": [
                    [joint.dict() if hasattr(joint, "dict") else joint for joint in frame]
                    for frame in skeleton_points
                ],
            },
            video_key=payload.video_key,
            sd_key=payload.sd_key,
        )
    except Exception:
        logger.exception("Failed to save aggregate run name=%s", run_name)
    return output


@router.post("/query", response_model=AggQueryOutput)
async def query_aggregated_shots(payload: AggQueryInput):
    """
    Return saved lane-ball and pose data for one or more shot numbers from the mock DB.
    Pulls from the aggregated mock-DB files. Missing shot numbers are silently omitted.
    """
    records = load_shots("aggregated", payload.shot_numbers)

    shots: dict[int, AggRunOutput] = {}
    for shot_number, rec in records.items():
        fps = rec.get("fps", 30.0)
        raw_positions = rec.get("ball_positions", [])

        ball_points = [BallPoint(x=p["x"], y=p["y"]) for p in raw_positions]

        ball_pos_objs = [
            BallPos(frame_index=i, timestamp_s=i / fps, x_m=p["x"], y_m=p["y"])
            for i, p in enumerate(raw_positions)
        ]
        kinematics = compute_kinematics_per_quarter(ball_pos_objs)
        kinematics_table = [
            KinematicsRow(
                quarter=q.quarter,
                start_m=q.start_m,
                end_m=q.end_m,
                mean_speed_mps=q.mean_speed_mps,
                mean_acceleration_mps2=q.mean_acceleration_mps2,
                sample_count=q.sample_count,
            )
            for q in kinematics.quarters
        ]

        skeleton_points = [
            [
                SkeletonPoint(
                    joint_id=j["joint_id"],
                    x=j["x"],
                    y=j["y"],
                    z=j["z"],
                )
                for j in frame
            ]
            for frame in rec.get("skeleton_frames", [])
        ]
        shots[shot_number] = AggRunOutput(
            ball_points=ball_points,
            kinematics_table=kinematics_table,
            skeleton_points=skeleton_points,
        )

    runs: dict[str, AggRunOutput] = {}
    for name, rec in load_named_runs(payload.names).items():
        sections = rec.get("sections", {})
        section = sections.get("aggregated")
        if section:
            try:
                runs[name] = AggRunOutput(**section)
            except Exception:
                logger.exception("Failed to parse saved aggregated section for name=%s", name)
            continue

        lane_section = sections.get("laneballs") or {}
        body_section = sections.get("fourdbody") or {}
        if not lane_section and not body_section:
            continue
        try:
            runs[name] = AggRunOutput(
                ball_points=[
                    BallPoint(x=float(point.get("x", 0.0)), y=float(point.get("y", 0.0)))
                    for point in lane_section.get("ball_points", [])
                ],
                kinematics_table=[
                    KinematicsRow(
                        quarter=int(row.get("quarter", 0)),
                        start_m=float(row.get("start_m", 0.0)),
                        end_m=float(row.get("end_m", 0.0)),
                        mean_speed_mps=float(row.get("mean_speed_mps", 0.0)),
                        mean_acceleration_mps2=float(row.get("mean_acceleration_mps2", 0.0)),
                        sample_count=int(row.get("sample_count", 0)),
                    )
                    for row in lane_section.get("kinematics_table", [])
                ],
                skeleton_points=[
                    [
                        SkeletonPoint(
                            joint_id=int(joint.get("joint_id", 0)),
                            x=float(joint.get("x", 0.0)),
                            y=float(joint.get("y", 0.0)),
                            z=float(joint.get("z", 0.0)),
                        )
                        for joint in frame
                    ]
                    for frame in body_section.get("skeleton_points", [])
                ],
            )
        except Exception:
            logger.exception("Failed to synthesize aggregate query for name=%s", name)

    return AggQueryOutput(shots=shots, runs=runs)
