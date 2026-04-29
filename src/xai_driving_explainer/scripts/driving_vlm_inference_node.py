#!/usr/bin/env python3
import base64
import json
import os
import re
import ssl
import time
import threading
import urllib.error
import urllib.request
from collections import deque

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


def stamp_to_float(stamp):
    if stamp is None:
        return None
    secs = getattr(stamp, "secs", None)
    nsecs = getattr(stamp, "nsecs", None)
    if secs is not None and nsecs is not None:
        return float(secs) + float(nsecs) * 1e-9
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    if isinstance(stamp, (int, float)):
        return float(stamp)
    return None


def extract_first_json_object(text):
    if not text:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_partial_string_field(text, field_name):
    pattern = r'"{}"\s*:\s*"([^"]*)'.format(re.escape(field_name))
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def extract_partial_list_field(text, field_name):
    pattern = r'"{}"\s*:\s*\[(.*?)\]'.format(re.escape(field_name))
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    body = match.group(1)
    items = re.findall(r'"([^"]*)"', body)
    return [item.strip() for item in items if item.strip()]


def salvage_partial_vlm_json(text):
    if not text:
        return None
    partial = {}
    scene = extract_partial_string_field(text, "scene_description_ko")
    objects = extract_partial_list_field(text, "detected_objects_ko")
    visual_summary = extract_partial_string_field(text, "visual_summary_ko")
    uncertain = extract_partial_list_field(text, "uncertain_points_ko")
    confidence = extract_partial_string_field(text, "confidence")

    if scene:
        partial["scene_description_ko"] = scene
    if objects is not None:
        partial["detected_objects_ko"] = objects
    if visual_summary:
        partial["visual_summary_ko"] = visual_summary
    if uncertain is not None:
        partial["uncertain_points_ko"] = uncertain
    if confidence:
        partial["confidence"] = confidence

    if not partial:
        return None
    return partial


def response_text_from_chat_completion(payload):
    try:
        return payload["choices"][0]["message"]["content"]
    except Exception:
        return ""


def response_text_from_ollama_chat(payload):
    try:
        return payload["message"]["content"]
    except Exception:
        return ""


def make_ssl_context():
    return ssl.create_default_context()


def compute_focus_rect(width, height, hint):
    if not isinstance(hint, dict) or not hint.get("available"):
        return (0, 0, width, height)

    region = hint.get("horizontal_region", "unknown")
    band = hint.get("vertical_band", "unknown")

    if region == "front_left":
        x0, x1 = int(width * 0.03), int(width * 0.48)
    elif region == "front_right":
        x0, x1 = int(width * 0.52), int(width * 0.97)
    else:
        x0, x1 = int(width * 0.20), int(width * 0.80)

    if band == "ground_near":
        y0, y1 = int(height * 0.45), int(height * 0.98)
    elif band == "upper_height":
        y0, y1 = int(height * 0.02), int(height * 0.55)
    else:
        y0, y1 = int(height * 0.18), int(height * 0.82)

    return (x0, y0, x1, y1)


def summarize_vlm_failure(status, result, model_name=None):
    model_label = str(model_name or "지정한 Ollama 비전 모델")
    result = result or {}
    if status == "quota_exceeded":
        return (
            "OpenAI API 사용 한도를 초과해 VLM 장면 설명을 생성하지 못했다.",
            "platform.openai.com의 billing 또는 credits를 확인한 뒤 노드를 다시 시작해야 한다.",
        )
    if status == "missing_api_key":
        return (
            "OPENAI_API_KEY가 설정되지 않아 VLM 호출을 수행하지 못했다.",
            "launch를 띄운 터미널에서 OPENAI_API_KEY를 export 했는지 확인해야 한다.",
        )
    if status == "ollama_not_running":
        return (
            "로컬 Ollama 서버가 실행 중이 아니어서 VLM 장면 설명을 생성하지 못했다.",
            "ollama serve 로 서버를 띄우고 ollama pull {} 로 모델을 내려받았는지 확인해야 한다.".format(
                model_label
            ),
        )
    if status == "ollama_model_missing":
        return (
            "지정한 Ollama 비전 모델이 아직 설치되지 않았다.",
            str(
                result.get("body")
                or "ollama pull {} 를 먼저 실행해야 한다.".format(model_label)
            )[:240],
        )
    if status == "no_recent_image":
        return (
            "이벤트 시점과 가까운 카메라 프레임을 찾지 못해 VLM 호출을 건너뛰었다.",
            "bag 재생 시 /camera/color/image_raw가 함께 재생되는지와 timestamp 차이를 확인해야 한다.",
        )
    if status == "image_encode_failed":
        return (
            "카메라 프레임 JPEG 인코딩에 실패해 VLM 호출을 진행하지 못했다.",
            "cv2.imencode('.jpg', image) 단계에서 실패했다.",
        )
    if status == "request_error":
        return (
            "VLM 백엔드 요청 자체가 실패했다.",
            str(result.get("error") or "request_error"),
        )
    if status == "http_error":
        detail = result.get("body") or result.get("status_code") or "http_error"
        return (
            "VLM 백엔드가 HTTP 오류를 반환했다.",
            str(detail)[:240],
        )
    if status == "parse_error":
        return (
            "VLM 백엔드 응답을 JSON으로 파싱하지 못했다.",
            str(result.get("body") or "parse_error")[:240],
        )
    if status == "json_parse_error":
        return (
            "모델 응답은 왔지만 JSON이 일부 잘려 부분 파싱만 가능했거나 형식이 완전하지 않았다.",
            str(result.get("raw_text") or "json_parse_error")[:240],
        )
    if status == "partial_ok":
        return (
            "모델 응답이 일부 잘렸지만 핵심 시각 설명은 복구했다.",
            "scene, objects, visual summary 일부를 부분 파싱해 사용한다.",
        )
    return (
        "VLM 설명 생성에 실패했다.",
        str(status),
    )


class DrivingVlmInferenceNode:
    def __init__(self):
        self.prompt_topic = rospy.get_param("~prompt_topic", "/xai/driving_vlm_prompts")
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.output_topic = rospy.get_param(
            "~output_topic", "/xai/driving_vlm_explanations"
        )
        self.backend = str(rospy.get_param("~backend", "ollama")).strip().lower()
        self.allowed_event_labels = self._parse_allowed_event_labels(
            rospy.get_param(
                "~allowed_event_labels",
                "path_blocked,path_update",
            )
        )
        self.model = str(rospy.get_param("~model", "moondream"))
        self.endpoint = str(
            rospy.get_param("~endpoint", "http://127.0.0.1:11434/api/chat")
        )
        self.api_key = str(rospy.get_param("~api_key", "")).strip() or str(
            rospy.get_param("/openai_api_key", "")
        ).strip() or str(rospy.get_param("/OPENAI_API_KEY", "")).strip()
        if not self.api_key:
            self.api_key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
        self.max_image_dt_s = float(rospy.get_param("~max_image_dt_s", 0.5))
        self.allow_stale_image_fallback = bool(
            rospy.get_param("~allow_stale_image_fallback", True)
        )
        self.allow_arrival_time_fallback = bool(
            rospy.get_param("~allow_arrival_time_fallback", True)
        )
        self.max_past_image_fallback_s = float(
            rospy.get_param("~max_past_image_fallback_s", 15.0)
        )
        self.max_future_image_fallback_s = float(
            rospy.get_param("~max_future_image_fallback_s", 2.0)
        )
        self.max_image_arrival_age_s = float(
            rospy.get_param("~max_image_arrival_age_s", 1.5)
        )
        self.max_buffer_size = int(rospy.get_param("~max_buffer_size", 60))
        self.use_focus_crop = bool(rospy.get_param("~use_focus_crop", True))
        self.focus_crop_margin_ratio = float(
            rospy.get_param("~focus_crop_margin_ratio", 0.10)
        )
        self.max_image_side_px = int(rospy.get_param("~max_image_side_px", 288))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 55))
        self.min_request_interval_s = float(
            rospy.get_param("~min_request_interval_s", 1.5)
        )
        self.log_vlm_explanation = bool(
            rospy.get_param("~log_vlm_explanation", True)
        )
        self.request_timeout_s = float(rospy.get_param("~request_timeout_s", 20.0))
        self.image_detail = str(rospy.get_param("~image_detail", "low"))
        self.max_tokens = int(rospy.get_param("~max_tokens", 96))

        self.bridge = CvBridge()
        self.image_buffer = deque(maxlen=max(5, self.max_buffer_size))
        self.last_request_time = 0.0
        self.last_prompt_signature = None
        self.warned_missing_api_key = False
        self.quota_exceeded = False
        self.request_lock = threading.Lock()

        self.publisher = rospy.Publisher(self.output_topic, String, queue_size=20)
        self.prompt_subscriber = rospy.Subscriber(
            self.prompt_topic, String, self._on_prompt, queue_size=1
        )
        self.image_subscriber = rospy.Subscriber(
            self.image_topic, Image, self._on_image, queue_size=5
        )

        rospy.loginfo(
            "driving_vlm_inference started | backend=%s prompt=%s image=%s output=%s model=%s endpoint=%s",
            self.backend,
            self.prompt_topic,
            self.image_topic,
            self.output_topic,
            self.model,
            self.endpoint,
        )

    def _parse_allowed_event_labels(self, raw_value):
        if isinstance(raw_value, list):
            return {str(item).strip() for item in raw_value if str(item).strip()}
        if raw_value is None:
            return set()
        return {item.strip() for item in str(raw_value).split(",") if item.strip()}

    def _on_image(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn("failed to decode image: %s", str(exc))
            return
        stamp = stamp_to_float(message.header.stamp)
        self.image_buffer.append(
            {
                "stamp": stamp,
                "arrival_wall_time": time.time(),
                "frame_id": message.header.frame_id,
                "image": image,
            }
        )

    def _parse_prompt(self, message):
        try:
            return json.loads(message.data)
        except Exception as exc:
            rospy.logwarn("failed to parse vlm prompt payload: %s", str(exc))
            return {}

    def _nearest_image(self, target_stamp, prompt_arrival_wall_time=None):
        if not self.image_buffer:
            return None
        if target_stamp is None:
            latest = self.image_buffer[-1]
            if self.allow_arrival_time_fallback and prompt_arrival_wall_time is not None:
                age = prompt_arrival_wall_time - float(
                    latest.get("arrival_wall_time") or prompt_arrival_wall_time
                )
                if age <= self.max_image_arrival_age_s:
                    return latest
            return latest
        buffered = list(self.image_buffer)
        best = min(
            buffered,
            key=lambda item: abs((item.get("stamp") or 0.0) - target_stamp),
        )
        delta = abs((best.get("stamp") or 0.0) - target_stamp)
        if delta > self.max_image_dt_s:
            if not self.allow_stale_image_fallback:
                return None
            oldest = buffered[0]
            latest = buffered[-1]
            oldest_stamp = oldest.get("stamp")
            latest_stamp = latest.get("stamp")
            if oldest_stamp is not None and target_stamp < oldest_stamp:
                if (oldest_stamp - target_stamp) <= self.max_past_image_fallback_s:
                    rospy.loginfo_throttle(
                        5.0,
                        "using oldest buffered image as fallback for stale event stamp | target=%.3f oldest=%.3f",
                        float(target_stamp),
                        float(oldest_stamp),
                    )
                    return oldest
            if latest_stamp is not None and target_stamp > latest_stamp:
                if (target_stamp - latest_stamp) <= self.max_future_image_fallback_s:
                    rospy.loginfo_throttle(
                        5.0,
                        "using latest buffered image as fallback for ahead-of-buffer event stamp | target=%.3f latest=%.3f",
                        float(target_stamp),
                        float(latest_stamp),
                    )
                    return latest
            if self.allow_arrival_time_fallback and prompt_arrival_wall_time is not None:
                latest_age = prompt_arrival_wall_time - float(
                    latest.get("arrival_wall_time") or prompt_arrival_wall_time
                )
                if latest_age <= self.max_image_arrival_age_s:
                    rospy.loginfo_throttle(
                        5.0,
                        "using latest buffered image by arrival-time fallback | target=%.3f latest_stamp=%.3f latest_age=%.3f",
                        float(target_stamp),
                        float(latest_stamp or 0.0),
                        float(latest_age),
                    )
                    return latest
            return None
        return best

    def _prepare_image_for_vlm(self, prompt_payload, image_bgr):
        prepared = image_bgr
        hint = prompt_payload.get("visual_grounding_hint", {})
        if self.use_focus_crop and prepared is not None:
            h, w = prepared.shape[:2]
            x0, y0, x1, y1 = compute_focus_rect(w, h, hint)
            margin_x = int((x1 - x0) * self.focus_crop_margin_ratio)
            margin_y = int((y1 - y0) * self.focus_crop_margin_ratio)
            x0 = max(0, x0 - margin_x)
            y0 = max(0, y0 - margin_y)
            x1 = min(w, x1 + margin_x)
            y1 = min(h, y1 + margin_y)
            if x1 > x0 and y1 > y0:
                prepared = prepared[y0:y1, x0:x1]

        if prepared is None:
            return image_bgr

        height, width = prepared.shape[:2]
        longest = max(height, width)
        if self.max_image_side_px > 0 and longest > self.max_image_side_px:
            scale = float(self.max_image_side_px) / float(longest)
            resized_w = max(1, int(round(width * scale)))
            resized_h = max(1, int(round(height * scale)))
            prepared = cv2.resize(
                prepared,
                (resized_w, resized_h),
                interpolation=cv2.INTER_AREA,
            )
        return prepared

    def _encode_image_base64(self, image_bgr):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(30, min(95, self.jpeg_quality))]
        success, encoded = cv2.imencode(".jpg", image_bgr, encode_params)
        if not success:
            return None
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    def _build_openai_request_payload(self, prompt_payload, image_base64):
        system_prompt = prompt_payload.get("system_prompt") or ""
        user_prompt = prompt_payload.get("user_prompt") or ""
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,{}".format(image_base64),
                                "detail": self.image_detail,
                            },
                        },
                    ],
                },
            ],
            "max_tokens": self.max_tokens,
        }

    def _build_ollama_request_payload(self, prompt_payload, image_base64):
        system_prompt = prompt_payload.get("system_prompt") or ""
        user_prompt = prompt_payload.get("user_prompt") or ""
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_base64],
                },
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": self.max_tokens,
            },
        }

    def _request_json(self, request_bytes, headers):
        request = urllib.request.Request(
            self.endpoint,
            data=request_bytes,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(
            request,
            timeout=self.request_timeout_s,
            context=make_ssl_context() if self.endpoint.startswith("https://") else None,
        ) as response:
            status_code = getattr(response, "status", response.getcode())
            response_body = response.read().decode("utf-8", errors="replace")
        return status_code, response_body

    def _call_vlm(self, prompt_payload, image_bgr):
        if self.backend == "openai" and not self.api_key:
            if not self.warned_missing_api_key:
                rospy.logwarn(
                    "OPENAI_API_KEY is not set. driving_vlm_inference will publish fallback results without vision output."
                )
                self.warned_missing_api_key = True
            return None, "missing_api_key"

        prepared_image = self._prepare_image_for_vlm(prompt_payload, image_bgr)
        image_base64 = self._encode_image_base64(prepared_image)
        if not image_base64:
            return None, "image_encode_failed"

        if self.backend == "openai":
            request_payload = self._build_openai_request_payload(
                prompt_payload, image_base64
            )
            headers = {
                "Authorization": "Bearer {}".format(self.api_key),
                "Content-Type": "application/json",
            }
        elif self.backend == "ollama":
            request_payload = self._build_ollama_request_payload(
                prompt_payload, image_base64
            )
            headers = {
                "Content-Type": "application/json",
            }
        else:
            return {"error": "unsupported backend: {}".format(self.backend)}, "request_error"

        request_bytes = json.dumps(request_payload).encode("utf-8")
        try:
            status_code, response_body = self._request_json(request_bytes, headers)
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = str(exc)
            lowered = error_body.lower()
            if self.backend == "openai" and "exceeded your current quota" in lowered:
                self.quota_exceeded = True
                rospy.logwarn(
                    "OpenAI API quota exceeded. Further VLM requests will be skipped until this node restarts."
                )
            if self.backend == "ollama" and (
                "model" in lowered and "not found" in lowered
            ):
                return {
                    "status_code": exc.code,
                    "body": error_body,
                }, "ollama_model_missing"
            return {
                "status_code": exc.code,
                "body": error_body,
            }, "http_error"
        except urllib.error.URLError as exc:
            if self.backend == "ollama":
                return {"error": str(exc)}, "ollama_not_running"
            return {"error": str(exc)}, "request_error"
        except Exception as exc:
            return {"error": str(exc)}, "request_error"

        if status_code >= 400:
            return {"status_code": status_code, "body": response_body}, "http_error"

        try:
            payload = json.loads(response_body)
        except Exception:
            return {"body": response_body}, "parse_error"

        if self.backend == "openai":
            text = response_text_from_chat_completion(payload)
        else:
            text = response_text_from_ollama_chat(payload)
        parsed_json = extract_first_json_object(text)
        if parsed_json is None:
            partial = salvage_partial_vlm_json(text)
            if partial:
                return {
                    "parsed": partial,
                    "raw_text": text,
                    "raw_response": payload,
                }, "partial_ok"
            return {"raw_text": text, "raw_response": payload}, "json_parse_error"
        return {"parsed": parsed_json, "raw_text": text}, "ok"

    def _fallback_vlm_result(self, prompt_payload, bundle, result, status):
        grounding = bundle.get("visual_grounding_hint", {})
        matched_region = grounding.get("matched_visual_region_ko") or "정확한 시각적 대응 영역은 아직 추정하지 못했다."
        planner_reason = bundle.get("planner_reason_ko") or bundle.get("source_summary_ko")
        scene_text, error_detail = summarize_vlm_failure(status, result, self.model)
        return {
            "schema": "xai_driving_explainer/DrivingVlmExplanation@1",
            "stamp": prompt_payload.get("stamp"),
            "event_label": prompt_payload.get("event_label"),
            "event_type": prompt_payload.get("event_type"),
            "status": status,
            "status_detail_ko": error_detail,
            "backend": self.backend,
            "matched_visual_region_ko": matched_region,
            "grounding_confidence": grounding.get("grounding_confidence", "low"),
            "scene_description_ko": scene_text,
            "detected_objects_ko": [],
            "planner_reason_ko": planner_reason,
            "grounded_action_explanation_ko": "",
            "final_combined_explanation_ko": scene_text,
            "unknowns_ko": [
                "실제 VLM 결과가 없어 카메라 기반 객체 설명은 비어 있다.",
                error_detail,
            ],
            "faithfulness_notes_ko": "camera scene description 미생성",
            "confidence": "low",
        }

    def _build_output_payload(self, prompt_payload, bundle, image_entry, result, status):
        if status not in ("ok", "partial_ok"):
            payload = self._fallback_vlm_result(prompt_payload, bundle, result, status)
            payload["image_stamp"] = image_entry.get("stamp") if image_entry else None
            return payload

        parsed = result.get("parsed", {})
        grounding = prompt_payload.get("visual_grounding_hint", {})
        scene_description = parsed.get("scene_description_ko") or "카메라 장면 설명이 비어 있다."
        visual_summary = parsed.get("visual_summary_ko") or scene_description
        matched_region = (
            parsed.get("matched_visual_region_ko")
            or grounding.get("matched_visual_region_ko")
            or "정확한 시각적 대응 영역은 아직 추정하지 못했다."
        )
        grounding_confidence = (
            parsed.get("grounding_confidence")
            or grounding.get("grounding_confidence")
            or "low"
        )
        detected_objects = parsed.get("detected_objects_ko") or []
        return {
            "schema": "xai_driving_explainer/DrivingVlmExplanation@1",
            "stamp": prompt_payload.get("stamp"),
            "image_stamp": image_entry.get("stamp") if image_entry else None,
            "event_label": prompt_payload.get("event_label"),
            "event_type": prompt_payload.get("event_type"),
            "status": status,
            "status_detail_ko": "VLM visual scene description generated successfully."
            if status == "ok"
            else "VLM 응답이 일부 잘렸지만 핵심 시각 설명은 복구했다.",
            "backend": self.backend,
            "matched_visual_region_ko": matched_region,
            "grounding_confidence": grounding_confidence,
            "scene_description_ko": scene_description,
            "detected_objects_ko": detected_objects,
            "planner_reason_ko": prompt_payload.get("planner_reason_ko"),
            "grounded_action_explanation_ko": visual_summary,
            "visual_summary_ko": visual_summary,
            "final_combined_explanation_ko": visual_summary,
            "unknowns_ko": parsed.get("uncertain_points_ko", []),
            "faithfulness_notes_ko": "카메라에서 보이는 장면만 설명하도록 제한한 visual-only 출력",
            "confidence": parsed.get("confidence", "medium"),
            "raw_model_output": result.get("raw_text", ""),
        }

    def _log_vlm_payload(self, payload):
        if not self.log_vlm_explanation:
            return
        rospy.loginfo(
            "[XAI-VLM] backend=%s | event=%s | status=%s | detail=%s | scene=%s | objects=%s | final=%s",
            payload.get("backend"),
            payload.get("event_label"),
            payload.get("status"),
            payload.get("status_detail_ko"),
            payload.get("scene_description_ko"),
            payload.get("detected_objects_ko"),
            payload.get("final_combined_explanation_ko"),
        )

    def _on_prompt(self, message):
        if not self.request_lock.acquire(False):
            return
        try:
            prompt_payload = self._parse_prompt(message)
            if not prompt_payload:
                return
            event_label = str(prompt_payload.get("event_label") or "").strip()
            if self.allowed_event_labels and event_label not in self.allowed_event_labels:
                return

            now = time.time()
            bundle_stamp = prompt_payload.get("stamp")
            image_entry = self._nearest_image(bundle_stamp, now)
            bundle = {
                "visual_grounding_hint": prompt_payload.get("visual_grounding_hint", {}),
                "planner_reason_ko": prompt_payload.get("planner_reason_ko"),
                "source_summary_ko": prompt_payload.get("source_summary_ko"),
            }

            signature = (
                prompt_payload.get("event_label"),
                prompt_payload.get("planner_reason_ko"),
                json.dumps(prompt_payload.get("visual_grounding_hint", {}), ensure_ascii=False, sort_keys=True),
            )
            if signature == self.last_prompt_signature and (
                now - self.last_request_time
            ) < self.min_request_interval_s:
                return

            if image_entry is None:
                payload = self._build_output_payload(
                    prompt_payload, bundle, None, None, "no_recent_image"
                )
                outgoing = String()
                outgoing.data = json.dumps(payload, ensure_ascii=False)
                self.publisher.publish(outgoing)
                self._log_vlm_payload(payload)
                return

            if self.quota_exceeded:
                payload = self._build_output_payload(
                    prompt_payload, bundle, image_entry, None, "quota_exceeded"
                )
                outgoing = String()
                outgoing.data = json.dumps(payload, ensure_ascii=False)
                self.publisher.publish(outgoing)
                self._log_vlm_payload(payload)
                return

            result, status = self._call_vlm(prompt_payload, image_entry["image"])
            payload = self._build_output_payload(
                prompt_payload, bundle, image_entry, result, status
            )
            outgoing = String()
            outgoing.data = json.dumps(payload, ensure_ascii=False)
            self.publisher.publish(outgoing)
            self._log_vlm_payload(payload)
            self.last_prompt_signature = signature
            self.last_request_time = now
        finally:
            self.request_lock.release()


def main():
    rospy.init_node("driving_vlm_inference")
    DrivingVlmInferenceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
