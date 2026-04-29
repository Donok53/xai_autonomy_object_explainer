#!/usr/bin/env python3
import argparse
import bisect
import json
import os
from collections import Counter

import rosbag


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an event-centered JSONL index from an autonomy rosbag."
    )
    parser.add_argument("--bag", required=True, help="Input rosbag path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--lidar-topic", default="/ouster/points")
    parser.add_argument("--planner-snapshot-topic", default="/xai/planner_snapshot")
    parser.add_argument("--event-log-topic", default="/xai/event_log")
    parser.add_argument(
        "--max-match-dt-s",
        type=float,
        default=0.5,
        help="Maximum allowed time gap for nearest-topic matching",
    )
    return parser.parse_args()


def stamp_to_float(stamp):
    if stamp is None:
        return None
    if isinstance(stamp, (int, float)):
        return float(stamp)
    secs = getattr(stamp, "secs", None)
    nsecs = getattr(stamp, "nsecs", None)
    if secs is not None and nsecs is not None:
        return float(secs) + float(nsecs) * 1e-9
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    return None


def load_json_string(message):
    raw = getattr(message, "data", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def nearest_stamp(stamps, target, max_dt_s=None):
    if not stamps or target is None:
        return None, None
    idx = bisect.bisect_left(stamps, target)
    candidates = []
    if idx < len(stamps):
        candidates.append(stamps[idx])
    if idx > 0:
        candidates.append(stamps[idx - 1])
    if not candidates:
        return None, None
    best = min(candidates, key=lambda value: abs(value - target))
    delta = best - target
    if max_dt_s is not None and abs(delta) > max_dt_s:
        return None, None
    return best, delta


def shallow_decision(payload):
    decision = payload.get("decision", {})
    behavior = decision.get("behavior", {})
    return {
        "behavior_reason": behavior.get("reason"),
        "behavior_stop": behavior.get("stop"),
        "speed_limit_mps": behavior.get("speed_limit_mps"),
        "emergency_stop": decision.get("emergency_stop", {}).get("value"),
        "path_blocked": decision.get("path_blocked", {}).get("value"),
        "global_obstacle_caution": decision.get("global_obstacle_caution", {}).get("value"),
    }


def shallow_evidence(payload):
    obstacle = payload.get("obstacle_evidence", {})
    planning = payload.get("planning", {})
    near_raw = obstacle.get("near_field_raw_overlay_hits", {})
    near_stop = obstacle.get("near_field_stop_hits", {})
    overlay_boxes = obstacle.get("global_overlay_boxes", {})
    path_change = planning.get("path_change", {}).get("latest", {})
    global_path = planning.get("global_path", {})
    return {
        "near_raw_min_range_m": near_raw.get("min_range_m"),
        "near_raw_min_x_m": near_raw.get("min_x_m"),
        "near_raw_points": near_raw.get("reported_points"),
        "near_raw_sample_centroid": near_raw.get("sample_centroid"),
        "near_stop_points": near_stop.get("reported_points"),
        "near_stop_sample_centroid": near_stop.get("sample_centroid"),
        "overlay_box_count": overlay_boxes.get("box_count"),
        "overlay_nearest_box": overlay_boxes.get("nearest_box"),
        "path_change_direction": path_change.get("direction"),
        "path_change_changed": path_change.get("changed"),
        "global_path_length_m": global_path.get("length_m"),
        "global_path_points": global_path.get("points"),
    }


def classify_horizontal_region(y_value):
    if y_value is None:
        return "unknown"
    if y_value > 0.35:
        return "front_left"
    if y_value < -0.35:
        return "front_right"
    return "front_center"


def describe_horizontal_region_ko(region):
    mapping = {
        "front_left": "전방 좌측",
        "front_center": "전방 중앙",
        "front_right": "전방 우측",
        "unknown": "전방 불명확 위치",
    }
    return mapping.get(region, "전방 불명확 위치")


def classify_vertical_band(z_value):
    if z_value is None:
        return "unknown"
    if z_value < 0.25:
        return "ground_near"
    if z_value < 0.9:
        return "mid_height"
    return "upper_height"


def describe_vertical_band_ko(band):
    mapping = {
        "ground_near": "지면 가까운 낮은 위치",
        "mid_height": "중간 높이",
        "upper_height": "상대적으로 높은 위치",
        "unknown": "높이 불명확",
    }
    return mapping.get(band, "높이 불명확")


def build_visual_grounding_hint(payload):
    obstacle = payload.get("obstacle_evidence", {})
    near_stop = obstacle.get("near_field_stop_hits", {})
    near_raw = obstacle.get("near_field_raw_overlay_hits", {})
    overlay_boxes = obstacle.get("global_overlay_boxes", {})

    anchor = None
    source_kind = None
    distance_hint_m = None

    for source_name, source in (
        ("near_field_stop_hits", near_stop),
        ("near_field_raw_overlay_hits", near_raw),
    ):
        centroid = source.get("sample_centroid")
        if isinstance(centroid, dict):
            anchor = centroid
            source_kind = source_name
            distance_hint_m = source.get("min_range_m")
            break

    nearest_box = overlay_boxes.get("nearest_box")
    if anchor is None and isinstance(nearest_box, dict):
        anchor = nearest_box.get("center") or nearest_box.get("position")
        source_kind = "global_overlay_boxes"
        distance_hint_m = nearest_box.get("distance_m")

    if not isinstance(anchor, dict):
        return {
            "available": False,
            "matched_visual_region_ko": "정확한 시각적 대응 영역은 아직 추정하지 못했다.",
            "grounding_confidence": "low",
        }

    horizontal_region = classify_horizontal_region(anchor.get("y"))
    vertical_band = classify_vertical_band(anchor.get("z"))
    return {
        "available": True,
        "source_evidence_kind": source_kind,
        "matched_visual_region_ko": "{} {}".format(
            describe_horizontal_region_ko(horizontal_region),
            describe_vertical_band_ko(vertical_band),
        ),
        "grounding_confidence": "medium",
        "distance_hint_m": distance_hint_m,
        "anchor_xyz": anchor,
    }


def build_baseline_explanation(event_payload):
    decision = shallow_decision(event_payload)
    evidence = shallow_evidence(event_payload)
    grounding_hint = build_visual_grounding_hint(event_payload)

    if decision.get("emergency_stop"):
        reason = "로봇은 비상정지 상태를 선택했다."
    elif decision.get("path_blocked"):
        reason = "로봇은 현재 전역 경로가 막혔다고 판단했다."
    elif decision.get("global_obstacle_caution"):
        reason = "로봇은 장애물 주의 상태로 감속 또는 보수 주행을 택했다."
    elif decision.get("behavior_reason"):
        reason = "로봇은 behavior reason에 따라 현재 경로를 유지하거나 조정했다."
    else:
        reason = "로봇의 판단 근거가 일부만 관찰되었다."

    details = []
    if evidence.get("near_raw_min_range_m") is not None:
        details.append("근거리 raw obstacle evidence가 존재한다")
    if evidence.get("near_stop_points"):
        details.append("근접 stop hit가 기록되었다")
    if evidence.get("overlay_box_count"):
        details.append("global overlay box가 감지되었다")
    if evidence.get("path_change_changed"):
        details.append(
            "선택 경로가 {} 방향으로 바뀌었다".format(
                evidence.get("path_change_direction") or "unknown"
            )
        )
    if not details and decision.get("behavior_reason"):
        details.append("behavior reason은 '{}'이다".format(decision["behavior_reason"]))

    return {
        "planner_reason_ko": reason,
        "matched_visual_region_ko": grounding_hint.get("matched_visual_region_ko"),
        "grounding_confidence": grounding_hint.get("grounding_confidence"),
        "details_ko": details,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image_stamps = []
    lidar_stamps = []
    planner_snapshots = []
    event_payloads = []
    topic_counts = Counter()

    with rosbag.Bag(args.bag, "r") as bag:
        for topic, message, stamp in bag.read_messages(
            topics=[
                args.image_topic,
                args.lidar_topic,
                args.planner_snapshot_topic,
                args.event_log_topic,
            ]
        ):
            bag_stamp = stamp.to_sec()
            topic_counts[topic] += 1

            if topic == args.image_topic:
                image_stamps.append(bag_stamp)
            elif topic == args.lidar_topic:
                lidar_stamps.append(bag_stamp)
            elif topic == args.planner_snapshot_topic:
                planner_payload = load_json_string(message)
                planner_stamp = stamp_to_float(planner_payload.get("stamp"))
                if planner_stamp is None:
                    planner_stamp = bag_stamp
                planner_snapshots.append((planner_stamp, planner_payload))
            elif topic == args.event_log_topic:
                event_payloads.append(load_json_string(message))

    planner_stamps = [item[0] for item in planner_snapshots]
    label_counts = Counter()

    index_path = os.path.join(args.output_dir, "event_index.jsonl")
    with open(index_path, "w", encoding="utf-8") as handle:
        for event in event_payloads:
            event_stamp = stamp_to_float(event.get("stamp"))
            planner_stamp, planner_dt = nearest_stamp(
                planner_stamps, event_stamp, max_dt_s=args.max_match_dt_s
            )
            image_stamp, image_dt = nearest_stamp(
                image_stamps, event_stamp, max_dt_s=args.max_match_dt_s
            )
            lidar_stamp, lidar_dt = nearest_stamp(
                lidar_stamps, event_stamp, max_dt_s=args.max_match_dt_s
            )

            planner_payload = {}
            if planner_stamp is not None:
                planner_idx = bisect.bisect_left(planner_stamps, planner_stamp)
                if planner_idx < len(planner_snapshots):
                    planner_payload = planner_snapshots[planner_idx][1]

            event_label = event.get("event_label", "unknown")
            label_counts[event_label] += 1
            baseline = build_baseline_explanation(event)

            record = {
                "bag_path": os.path.abspath(args.bag),
                "event_seq": event.get("seq"),
                "event_type": event.get("event_type"),
                "event_label": event_label,
                "event_stamp": event_stamp,
                "decision": shallow_decision(event),
                "signature": event.get("signature", {}),
                "evidence": shallow_evidence(event),
                "planning": {
                    "global_path": event.get("planning", {}).get("global_path", {}),
                    "path_change": event.get("planning", {}).get("path_change", {}),
                },
                "nearest": {
                    "planner_snapshot_stamp": planner_stamp,
                    "planner_snapshot_dt_s": planner_dt,
                    "image_stamp": image_stamp,
                    "image_dt_s": image_dt,
                    "lidar_stamp": lidar_stamp,
                    "lidar_dt_s": lidar_dt,
                },
                "planner_snapshot_decision": shallow_decision(planner_payload),
                "baseline_explanation": baseline,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "bag_path": os.path.abspath(args.bag),
        "output_dir": os.path.abspath(args.output_dir),
        "topic_counts": dict(topic_counts),
        "event_count": len(event_payloads),
        "planner_snapshot_count": len(planner_snapshots),
        "image_count": len(image_stamps),
        "lidar_count": len(lidar_stamps),
        "max_match_dt_s": args.max_match_dt_s,
        "event_label_counts": dict(label_counts),
    }

    summary_path = os.path.join(args.output_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("wrote {}".format(index_path))
    print("wrote {}".format(summary_path))
    print("indexed {} events".format(len(event_payloads)))


if __name__ == "__main__":
    main()
