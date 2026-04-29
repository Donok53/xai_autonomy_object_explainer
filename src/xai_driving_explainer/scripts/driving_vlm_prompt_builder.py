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


def short_json(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


class DrivingVlmPromptBuilder:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/xai/driving_explanations")
        self.output_topic = rospy.get_param("~output_topic", "/xai/driving_vlm_prompts")
        self.allowed_event_labels = self._parse_allowed_event_labels(
            rospy.get_param(
                "~allowed_event_labels",
                "path_blocked,path_update",
            )
        )

        self.publisher = rospy.Publisher(self.output_topic, String, queue_size=20)
        self.subscriber = rospy.Subscriber(
            self.input_topic, String, self._on_bundle, queue_size=20
        )

        rospy.loginfo(
            "driving_vlm_prompt_builder started | input=%s output=%s",
            self.input_topic,
            self.output_topic,
        )

    def _parse_allowed_event_labels(self, raw_value):
        if isinstance(raw_value, list):
            return {str(item).strip() for item in raw_value if str(item).strip()}
        if raw_value is None:
            return set()
        return {item.strip() for item in str(raw_value).split(",") if item.strip()}

    def _parse_bundle(self, message):
        try:
            return json.loads(message.data)
        except Exception as exc:
            rospy.logwarn("failed to parse driving explanation bundle: %s", str(exc))
            return {}

    def _compose_system_prompt(self):
        return (
            "당신은 자율주행 로봇의 전방 카메라 장면만 짧게 설명하는 VLM이다. "
            "planner 판단, 로봇 의도, 경로 상태는 추정하지 마라. "
            "이미지에서 실제로 보이는 장면과 물체만 아주 짧게 적어라. "
            "가능하면 반드시 한국어만 사용하라. "
            "출력은 JSON 객체 하나만 작성하라."
        )

    def _compose_user_prompt(self, bundle):
        grounding_hint = bundle.get("visual_grounding_hint", {})
        matched_region = grounding_hint.get(
            "matched_visual_region_ko",
            "정확한 시각적 대응 영역은 아직 추정하지 못했다.",
        )
        grounding_confidence = grounding_hint.get("grounding_confidence", "low")

        prompt_lines = [
            "아래 카메라 이미지를 보고 실제로 보이는 장면만 짧게 설명하라.",
            "planner, hold, clear, 경로 유지, 우회 같은 제어 표현은 쓰지 마라.",
            "판단 이유를 설명하지 말고, 물체와 장면만 묘사하라.",
            "",
            "event_label: {}".format(bundle.get("event_label")),
            "",
            "[visual grounding hint]",
            "matched_visual_region_ko: {}".format(matched_region),
            "grounding_confidence: {}".format(grounding_confidence),
            "위 영역을 우선 보되, 장면 전체에서 눈에 띄는 물체도 함께 보라.",
            "",
            "[작업]",
            "1. 장면을 아주 짧은 1문장으로 설명하라.",
            "2. 보이는 주요 물체를 0~3개만 적어라.",
            "3. visual_summary_ko도 짧은 1문장으로 적어라.",
            "4. 가능하면 반드시 한국어만 사용하라.",
            "",
            "[출력 형식]",
            '{'
            '"scene_description_ko": "...",'
            '"detected_objects_ko": ["...", "..."],'
            '"visual_summary_ko": "...",'
            '"confidence": "low|medium|high"'
            '}',
        ]
        return "\n".join(prompt_lines)

    def _compose_prompt_payload(self, bundle):
        return {
            "schema": "xai_driving_explainer/DrivingVlmPrompt@1",
            "stamp": bundle.get("stamp"),
            "event_label": bundle.get("event_label"),
            "event_type": bundle.get("event_type"),
            "source_summary_ko": bundle.get("summary_ko"),
            "planner_reason_ko": bundle.get("planner_reason_ko"),
            "camera_scene_hint_ko": bundle.get("camera_scene_hint_ko"),
            "visual_grounding_hint": bundle.get("visual_grounding_hint", {}),
            "recommended_image_topic": "/camera/color/image_raw",
            "recommended_lidar_topic": "/ouster/points",
            "system_prompt": self._compose_system_prompt(),
            "user_prompt": self._compose_user_prompt(bundle),
            "suggested_output_schema": {
                "scene_description_ko": "string",
                "detected_objects_ko": ["string"],
                "visual_summary_ko": "string",
                "confidence": "low|medium|high",
            },
        }

    def _on_bundle(self, message):
        bundle = self._parse_bundle(message)
        if not bundle:
            return
        event_label = str(bundle.get("event_label") or "").strip()
        if self.allowed_event_labels and event_label not in self.allowed_event_labels:
            return
        payload = self._compose_prompt_payload(bundle)
        outgoing = String()
        outgoing.data = json.dumps(payload, ensure_ascii=False)
        self.publisher.publish(outgoing)


def main():
    rospy.init_node("driving_vlm_prompt_builder")
    DrivingVlmPromptBuilder()
    rospy.spin()


if __name__ == "__main__":
    main()
