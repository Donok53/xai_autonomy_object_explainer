#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as point_cloud2
from cv_bridge import CvBridge


ALLOWED_LABELS_KO = [
    "사람",
    "자전거",
    "자동차",
    "오토바이",
    "기차",
    "트럭",
    "신호등",
    "소화전",
    "정지 표지판",
    "주차 미터기",
    "벤치",
    "고양이",
    "개",
    "가방",
    "우산",
    "손가방",
    "캐리어",
    "공",
    "병",
    "컵",
    "와인잔",
    "책",
    "시계",
    "벽",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export offline VLM teacher dataset from ROS bag."
    )
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--planner-topic", default="/xai/planner_snapshot")
    parser.add_argument("--event-topic", default="/xai/event_log")
    parser.add_argument(
        "--point-cloud-topic",
        default="/planning/linefit_ground/non_ground_cloud",
    )
    parser.add_argument("--max-image-age-s", type=float, default=0.25)
    parser.add_argument("--max-planner-age-s", type=float, default=0.75)
    parser.add_argument("--max-pointcloud-age-s", type=float, default=0.40)
    parser.add_argument("--max-pointcloud-points", type=int, default=2500)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def safe_json_loads(raw_text):
    try:
        return json.loads(raw_text)
    except Exception:
        return {}


def stamp_to_float(value, fallback=None):
    if value is None:
        return fallback
    try:
        return float(value)
    except Exception:
        return fallback


def sample_point_cloud(msg, max_points):
    points = []
    width = int(getattr(msg, "width", 0) or 0)
    height = int(getattr(msg, "height", 0) or 0)
    estimated = max(1, width * max(1, height))
    stride = max(1, int(math.ceil(float(estimated) / float(max(1, max_points)))))
    for index, point_xyz in enumerate(
        point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
    ):
        if (index % stride) != 0:
            continue
        points.append((float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])))
    return np.asarray(points, dtype=np.float32)


def obstacle_summary(event_data):
    obstacle = (event_data or {}).get("obstacle_evidence") or {}
    near_raw = obstacle.get("near_field_raw_overlay_hits") or {}
    centroid = near_raw.get("sample_centroid") or {}
    summary = {
        "near_raw_points": int(near_raw.get("reported_points") or 0),
        "near_raw_min_range_m": near_raw.get("min_range_m"),
        "near_raw_min_x_m": near_raw.get("min_x_m"),
        "near_raw_centroid_xyz": {
            "x": centroid.get("x"),
            "y": centroid.get("y"),
            "z": centroid.get("z"),
        },
    }
    return summary


def planner_reason(planner_snapshot, event_data):
    decision = ((event_data or {}).get("decision") or {})
    behavior = decision.get("behavior") or {}
    reason = behavior.get("reason")
    if reason:
        return str(reason)
    decision = ((planner_snapshot or {}).get("decision") or {})
    behavior = decision.get("behavior") or {}
    reason = behavior.get("reason")
    if reason:
        return str(reason)
    return "unknown"


def motion_state(planner_snapshot, event_data):
    control = ((event_data or {}).get("control") or {})
    state = control.get("motion_state")
    if state:
        return str(state)
    control = ((planner_snapshot or {}).get("control") or {})
    state = control.get("motion_state")
    if state:
        return str(state)
    return "unknown"


def build_teacher_prompt(sample):
    event_label = sample.get("event_label", "unknown")
    planner_reason_text = sample.get("planner_reason", "unknown")
    motion_state_text = sample.get("motion_state", "unknown")
    obstacle = sample.get("obstacle_summary") or {}
    near_range = obstacle.get("near_raw_min_range_m")
    near_centroid = obstacle.get("near_raw_centroid_xyz") or {}
    near_centroid_text = json.dumps(near_centroid, ensure_ascii=False)

    prompt_lines = [
        "너는 자율주행 XAI teacher 모델이다.",
        "목표는 planner/LiDAR 맥락과 가장 관련 있는 대표 시각 객체를 1개 고르는 것이다.",
        "설명은 보이는 것만 기준으로 하고, planner의 판단 이유를 그대로 복사하지 마라.",
        "",
        "주행 문맥:",
        "- event_label: {}".format(event_label),
        "- planner_reason: {}".format(planner_reason_text),
        "- motion_state: {}".format(motion_state_text),
        "- near_raw_min_range_m: {}".format(near_range),
        "- near_raw_centroid_xyz: {}".format(near_centroid_text),
        "",
        "허용 라벨 후보:",
        "- {}".format(", ".join(ALLOWED_LABELS_KO)),
        "",
        "규칙:",
        "- 허용 라벨에 없으면 primary_object_ko는 반드시 '벽'으로 답한다.",
        "- camera image에서 가장 관련 있는 대표 객체 1개만 고른다.",
        "- dynamic은 static, dynamic, unknown 중 하나로 답한다.",
        "- 반드시 JSON만 출력한다.",
        "",
        "출력 형식:",
        '{'
        '"primary_object_ko":"",'
        '"primary_object_en":"",'
        '"dynamic":"static|dynamic|unknown",'
        '"scene_summary_ko":"",'
        '"reasoning_link_ko":"",'
        '"confidence":0.0'
        '}',
    ]
    return "\n".join(prompt_lines)


def ensure_dirs(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "pointcloud").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)
    (output_dir / "annotations").mkdir(exist_ok=True)


def main():
    args = parse_args()
    bag_path = Path(args.bag).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dirs(output_dir)

    bridge = CvBridge()
    latest_image = None
    latest_planner = None
    latest_cloud = None
    exported = 0
    skipped = 0

    metadata_path = output_dir / "metadata" / "teacher_dataset.jsonl"
    with rosbag.Bag(str(bag_path)) as bag, metadata_path.open("w", encoding="utf-8") as metadata_file:
        for topic, msg, bag_stamp in bag.read_messages(
            topics=[
                args.image_topic,
                args.planner_topic,
                args.event_topic,
                args.point_cloud_topic,
            ]
        ):
            if topic == args.image_topic:
                latest_image = {
                    "stamp": float(msg.header.stamp.to_sec()),
                    "msg": msg,
                }
                continue

            if topic == args.planner_topic:
                snapshot = safe_json_loads(msg.data)
                latest_planner = {
                    "stamp": stamp_to_float(snapshot.get("stamp"), float(bag_stamp.to_sec())),
                    "data": snapshot,
                }
                continue

            if topic == args.point_cloud_topic:
                latest_cloud = {
                    "stamp": float(msg.header.stamp.to_sec()),
                    "frame_id": str(msg.header.frame_id or ""),
                    "msg": msg,
                }
                continue

            if topic != args.event_topic:
                continue

            event_data = safe_json_loads(msg.data)
            event_stamp = stamp_to_float(event_data.get("stamp"), float(bag_stamp.to_sec()))

            if latest_image is None:
                skipped += 1
                continue
            if abs(event_stamp - latest_image["stamp"]) > float(args.max_image_age_s):
                skipped += 1
                continue
            if latest_planner is None or abs(event_stamp - latest_planner["stamp"]) > float(args.max_planner_age_s):
                skipped += 1
                continue

            image_bgr = bridge.imgmsg_to_cv2(
                latest_image["msg"],
                desired_encoding="bgr8",
            )

            sample_id = "sample_{:05d}".format(exported)
            image_path = output_dir / "images" / "{}.jpg".format(sample_id)
            cv2.imwrite(
                str(image_path),
                image_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
            )

            pointcloud_relpath = None
            pointcloud_summary = {
                "frame_id": None,
                "stamp": None,
                "point_count": 0,
            }
            if latest_cloud is not None and abs(event_stamp - latest_cloud["stamp"]) <= float(args.max_pointcloud_age_s):
                points_xyz = sample_point_cloud(latest_cloud["msg"], args.max_pointcloud_points)
                pointcloud_path = output_dir / "pointcloud" / "{}.npz".format(sample_id)
                np.savez_compressed(pointcloud_path, points_xyz=points_xyz)
                pointcloud_relpath = str(pointcloud_path.relative_to(output_dir))
                pointcloud_summary = {
                    "frame_id": latest_cloud["frame_id"],
                    "stamp": latest_cloud["stamp"],
                    "point_count": int(points_xyz.shape[0]),
                }

            planner_snapshot = latest_planner["data"]
            event_label = str(event_data.get("event_label") or "unknown")
            sample = {
                "sample_id": sample_id,
                "stamp": event_stamp,
                "event_label": event_label,
                "event_type": str(event_data.get("event_type") or "unknown"),
                "planner_reason": planner_reason(planner_snapshot, event_data),
                "motion_state": motion_state(planner_snapshot, event_data),
                "path_blocked": bool((((event_data.get("decision") or {}).get("path_blocked") or {}).get("value"))),
                "image_path": str(image_path.relative_to(output_dir)),
                "pointcloud_path": pointcloud_relpath,
                "pointcloud_summary": pointcloud_summary,
                "obstacle_summary": obstacle_summary(event_data),
                "teacher_label_candidates_ko": list(ALLOWED_LABELS_KO),
                "planner_snapshot": planner_snapshot,
                "event_log": event_data,
            }
            sample["teacher_prompt_ko"] = build_teacher_prompt(sample)
            metadata_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

            exported += 1
            if args.limit > 0 and exported >= int(args.limit):
                break

    print("exported_samples={}".format(exported))
    print("skipped_events={}".format(skipped))
    print("metadata={}".format(metadata_path))


if __name__ == "__main__":
    main()
