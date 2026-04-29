#!/usr/bin/env python3
import json

import rospy
from std_msgs.msg import String


def nested_get(payload, *keys, default=None):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


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


class DrivingExplainerNode:
    def __init__(self):
        self.planner_snapshot_topic = rospy.get_param(
            "~planner_snapshot_topic", "/xai/planner_snapshot"
        )
        self.event_log_topic = rospy.get_param("~event_log_topic", "/xai/event_log")
        self.output_topic = rospy.get_param(
            "~output_topic", "/xai/driving_explanations"
        )
        self.publish_snapshot_updates = bool(
            rospy.get_param("~publish_snapshot_updates", False)
        )
        self.log_combined_explanation = bool(
            rospy.get_param("~log_combined_explanation", True)
        )
        self.log_allowed_event_labels = self._parse_allowed_event_labels(
            rospy.get_param(
                "~log_allowed_event_labels",
                "path_blocked,path_update,state_update,behavior_reason",
            )
        )

        self.latest_snapshot = {}
        self.latest_event = {}
        self.last_logged_signature = None

        self.publisher = rospy.Publisher(self.output_topic, String, queue_size=20)
        self.snapshot_sub = rospy.Subscriber(
            self.planner_snapshot_topic, String, self._on_snapshot, queue_size=10
        )
        self.event_sub = rospy.Subscriber(
            self.event_log_topic, String, self._on_event, queue_size=20
        )

        rospy.loginfo(
            "xai_driving_explainer started | snapshot=%s event=%s output=%s",
            self.planner_snapshot_topic,
            self.event_log_topic,
            self.output_topic,
        )

    def _parse_allowed_event_labels(self, raw_value):
        if isinstance(raw_value, list):
            return {str(item).strip() for item in raw_value if str(item).strip()}
        if raw_value is None:
            return set()
        return {item.strip() for item in str(raw_value).split(",") if item.strip()}

    def _parse_json(self, message):
        try:
            return json.loads(message.data)
        except Exception as exc:
            rospy.logwarn("failed to parse JSON payload: %s", str(exc))
            return {}

    def _on_snapshot(self, message):
        self.latest_snapshot = self._parse_json(message)
        if self.publish_snapshot_updates:
            self._publish_bundle(trigger="planner_snapshot")

    def _on_event(self, message):
        self.latest_event = self._parse_json(message)
        self._publish_bundle(trigger="event_log")

    def _compose_planner_reason(self):
        payload = self.latest_event or self.latest_snapshot or {}
        decision = payload.get("decision", {})
        behavior_reason = nested_get(decision, "behavior", "reason")
        emergency_stop = nested_get(decision, "emergency_stop", "value")
        path_blocked = nested_get(decision, "path_blocked", "value")
        caution = nested_get(decision, "global_obstacle_caution", "value")
        speed_limit = nested_get(decision, "behavior", "speed_limit_mps")

        if emergency_stop:
            return "로봇은 현재 비상정지 상태이며, 전방 근거리 장애물 또는 정지 근거를 우선적으로 반영하고 있다."
        if path_blocked:
            return "로봇은 현재 전역 경로가 막혔다고 판단해 우회 또는 경로 재선택을 준비하고 있다."
        if caution:
            return "로봇은 장애물 주의 상태로 판단해 보수적으로 속도를 제한하고 있다."
        if behavior_reason:
            return "로봇은 현재 '{}' 판단을 기준으로 경로를 유지하거나 조정하고 있다.".format(
                behavior_reason
            )
        if speed_limit is not None:
            return "로봇은 현재 속도 제한 {:.2f}m/s 범위 안에서 주행 중이다.".format(
                float(speed_limit)
            )
        return "로봇의 주행 설명 근거를 일부만 수신했다."

    def _compose_summary(self):
        return self._compose_planner_reason()

    def _compose_evidence(self):
        payload = self.latest_snapshot or self.latest_event or {}
        obstacle = payload.get("obstacle_evidence", {})
        planning = payload.get("planning", {})
        control = payload.get("control", {})

        near_raw = obstacle.get("near_field_raw_overlay_hits", {})
        near_stop = obstacle.get("near_field_stop_hits", {})
        overlay_boxes = obstacle.get("global_overlay_boxes", {})
        path_change = nested_get(planning, "path_change", "latest", default={}) or {}
        global_path = planning.get("global_path", {})

        evidence = []
        if near_raw.get("reported_points"):
            evidence.append(
                {
                    "kind": "near_field_raw_overlay_hits",
                    "reported_points": near_raw.get("reported_points"),
                    "min_range_m": near_raw.get("min_range_m"),
                    "min_x_m": near_raw.get("min_x_m"),
                    "sample_centroid": near_raw.get("sample_centroid"),
                    "frame_id": near_raw.get("frame_id"),
                }
            )
        if near_stop.get("reported_points"):
            evidence.append(
                {
                    "kind": "near_field_stop_hits",
                    "reported_points": near_stop.get("reported_points"),
                    "sample_centroid": near_stop.get("sample_centroid"),
                    "frame_id": near_stop.get("frame_id"),
                }
            )
        if overlay_boxes.get("box_count"):
            evidence.append(
                {
                    "kind": "global_overlay_boxes",
                    "box_count": overlay_boxes.get("box_count"),
                    "nearest_box": overlay_boxes.get("nearest_box"),
                }
            )
        if path_change.get("changed"):
            evidence.append(
                {
                    "kind": "path_change",
                    "direction": path_change.get("direction"),
                    "lateral_shift_m": path_change.get("lateral_shift_m"),
                }
            )
        if global_path.get("received"):
            evidence.append(
                {
                    "kind": "global_path",
                    "length_m": global_path.get("length_m"),
                    "points": global_path.get("points"),
                }
            )
        if control.get("received"):
            evidence.append(
                {
                    "kind": "control",
                    "linear_x_mps": control.get("linear_x_mps"),
                    "angular_z_radps": control.get("angular_z_radps"),
                    "steering_direction": control.get("steering_direction"),
                    "motion_state": control.get("motion_state"),
                }
            )
        return evidence

    def _extract_visual_grounding_hint(self):
        payload = self.latest_snapshot or self.latest_event or {}
        obstacle = payload.get("obstacle_evidence", {})

        near_stop = obstacle.get("near_field_stop_hits", {})
        near_raw = obstacle.get("near_field_raw_overlay_hits", {})
        overlay_boxes = obstacle.get("global_overlay_boxes", {})

        source_kind = None
        anchor = None
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
                "note_ko": "현재 출력은 planner evidence 중심이며, camera와의 1:1 객체 대응은 보장하지 않는다.",
            }

        x_value = anchor.get("x")
        y_value = anchor.get("y")
        z_value = anchor.get("z")

        horizontal_region = classify_horizontal_region(y_value)
        vertical_band = classify_vertical_band(z_value)
        matched_region_ko = "{} {}".format(
            describe_horizontal_region_ko(horizontal_region),
            describe_vertical_band_ko(vertical_band),
        )

        confidence = "medium"
        if source_kind == "global_overlay_boxes":
            confidence = "medium"
        if source_kind == "near_field_stop_hits":
            confidence = "medium"

        return {
            "available": True,
            "source_evidence_kind": source_kind,
            "frame_id": "base_link",
            "anchor_xyz": {
                "x": x_value,
                "y": y_value,
                "z": z_value,
            },
            "distance_hint_m": distance_hint_m,
            "horizontal_region": horizontal_region,
            "vertical_band": vertical_band,
            "matched_visual_region_ko": matched_region_ko,
            "grounding_confidence": confidence,
            "note_ko": "이 힌트는 시간 정렬된 planner/LiDAR evidence를 기반으로 한 느슨한 시각 대응이며, exact object identity를 보장하지 않는다.",
        }

    def _compose_final_explanation(self, planner_reason, grounding_hint):
        if grounding_hint.get("available"):
            return "{} 카메라에서는 우선 '{}' 영역을 확인해 scene narration을 보강하는 방식이 적절하다.".format(
                planner_reason,
                grounding_hint.get("matched_visual_region_ko"),
            )
        return "{} 카메라는 장면 전반의 통로 상태와 장애물 배치를 설명하되, 특정 객체 동일시는 보수적으로 다루는 편이 적절하다.".format(
            planner_reason
        )

    def _compose_camera_scene_hint(self, grounding_hint):
        if grounding_hint.get("available"):
            return "카메라에서는 '{}' 영역의 장애물 배치와 통로 상태를 우선 확인한다.".format(
                grounding_hint.get("matched_visual_region_ko")
            )
        return "카메라에서는 전방 장면 전체의 통로 구조와 눈에 보이는 장애물 배치를 폭넓게 확인한다."

    def _compose_console_log(self, bundle):
        event_label = bundle.get("event_label") or "unknown"
        planner_reason = bundle.get("planner_reason_ko") or "-"
        camera_hint = bundle.get("camera_scene_hint_ko") or "-"
        final_explanation = bundle.get("final_explanation_ko") or "-"
        grounding_hint = bundle.get("visual_grounding_hint", {})
        matched_region = grounding_hint.get("matched_visual_region_ko") or "unknown"
        confidence = grounding_hint.get("grounding_confidence") or "unknown"
        return (
            "[XAI] event={event} | planner_reason={planner} | camera_hint={camera} | "
            "matched_region={region} | grounding_confidence={confidence} | final={final}"
        ).format(
            event=event_label,
            planner=planner_reason,
            camera=camera_hint,
            region=matched_region,
            confidence=confidence,
            final=final_explanation,
        )

    def _should_log_bundle(self, bundle):
        if not self.log_combined_explanation:
            return False

        event_label = bundle.get("event_label")
        if self.log_allowed_event_labels and event_label not in self.log_allowed_event_labels:
            return False

        signature = (
            event_label,
            bundle.get("planner_reason_ko"),
            bundle.get("camera_scene_hint_ko"),
            nested_get(bundle, "visual_grounding_hint", "matched_visual_region_ko"),
            nested_get(bundle, "visual_grounding_hint", "grounding_confidence"),
            bundle.get("final_explanation_ko"),
        )
        if signature == self.last_logged_signature:
            return False

        self.last_logged_signature = signature
        return True

    def _compose_bundle(self, trigger):
        event = self.latest_event or {}
        snapshot = self.latest_snapshot or {}
        source_stamp = event.get("stamp") or snapshot.get("stamp")
        replay_stamp = rospy.Time.now().to_sec()
        stamp = source_stamp or replay_stamp
        planner_reason = self._compose_planner_reason()
        grounding_hint = self._extract_visual_grounding_hint()
        camera_scene_hint = self._compose_camera_scene_hint(grounding_hint)
        final_explanation = self._compose_final_explanation(planner_reason, grounding_hint)
        payload = {
            "schema": "xai_driving_explainer/DrivingExplanationBundle@1",
            "trigger": trigger,
            "stamp": stamp,
            "source_stamp": source_stamp,
            "replay_stamp": replay_stamp,
            "event_label": event.get("event_label"),
            "event_type": event.get("event_type"),
            "summary_ko": planner_reason,
            "planner_reason_ko": planner_reason,
            "camera_scene_role_ko": "카메라는 장면의 통로 구조, 눈에 보이는 장애물 배치, 가시성, 그리고 사람이 이해하기 쉬운 scene context를 설명한다.",
            "camera_do_not_claim_ko": "카메라는 단일 프레임만으로 planner가 회피한 정확한 동일 객체를 단정하지 않는다.",
            "camera_scene_hint_ko": camera_scene_hint,
            "visual_grounding_hint": grounding_hint,
            "final_explanation_ko": final_explanation,
            "decision": (event or snapshot).get("decision", {}),
            "signature": event.get("signature", {}),
            "evidence": self._compose_evidence(),
            "faithfulness_policy": {
                "planner_state_is_primary": True,
                "camera_or_vlm_is_secondary_narration": True,
                "exact_visual_object_identity_not_required": True,
            },
        }
        return payload

    def _publish_bundle(self, trigger):
        bundle = self._compose_bundle(trigger)
        message = String()
        message.data = json.dumps(bundle, ensure_ascii=False)
        self.publisher.publish(message)
        if self._should_log_bundle(bundle):
            rospy.loginfo(self._compose_console_log(bundle))


def main():
    rospy.init_node("xai_driving_explainer")
    DrivingExplainerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
