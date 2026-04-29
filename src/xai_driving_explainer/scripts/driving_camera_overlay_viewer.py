#!/usr/bin/env python3
import json
import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from PIL import Image as PilImage
from PIL import ImageDraw, ImageFont
from sensor_msgs.msg import Image
from std_msgs.msg import String


def wrap_text(text, max_chars):
    if not text:
        return []
    words = str(text).split()
    if not words:
        return [str(text)]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def measure_text_width(draw, font, text):
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    if hasattr(draw, "textsize"):
        width, _ = draw.textsize(text, font=font)
        return width
    if hasattr(font, "getsize"):
        width, _ = font.getsize(text)
        return width
    return max(1, len(str(text))) * 12


def wrap_text_with_font(text, draw, font, max_width_px):
    if not text:
        return []
    text = str(text)
    words = text.split()
    if not words:
        return [text]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        width = measure_text_width(draw, font, candidate)
        if width <= max_width_px:
            current = candidate
            continue
        lines.append(current)
        current = word

    lines.append(current)
    return lines


class DrivingCameraOverlayViewer:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.explanation_topic = rospy.get_param(
            "~explanation_topic", "/xai/driving_explanations"
        )
        self.vlm_topic = rospy.get_param(
            "~vlm_topic", "/xai/driving_vlm_explanations"
        )
        self.output_topic = rospy.get_param(
            "~output_topic", "/xai/driving_camera_overlay"
        )
        self.display_window = bool(rospy.get_param("~display_window", True))
        self.window_name = str(
            rospy.get_param("~window_name", "xai_driving_camera_view")
        )
        self.font_scale = float(rospy.get_param("~font_scale", 0.55))
        self.line_height = int(rospy.get_param("~line_height", 22))
        self.font_path = str(
            rospy.get_param(
                "~font_path",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            )
        )
        self.font_size_px = int(rospy.get_param("~font_size_px", 24))
        self.focus_font_size_px = int(rospy.get_param("~focus_font_size_px", 26))
        self.vlm_image_match_tolerance_s = float(
            rospy.get_param("~vlm_image_match_tolerance_s", 0.75)
        )

        self.bridge = CvBridge()
        self.latest_explanation = {}
        self.latest_vlm = {}
        self.text_font = None
        self.focus_font = None
        self._init_fonts()

        self.publisher = rospy.Publisher(self.output_topic, Image, queue_size=5)
        self.image_subscriber = rospy.Subscriber(
            self.image_topic, Image, self._on_image, queue_size=5
        )
        self.explanation_subscriber = rospy.Subscriber(
            self.explanation_topic, String, self._on_explanation, queue_size=20
        )
        self.vlm_subscriber = rospy.Subscriber(
            self.vlm_topic, String, self._on_vlm, queue_size=20
        )

        rospy.loginfo(
            "driving_camera_overlay_viewer started | image=%s explanation=%s vlm=%s output=%s",
            self.image_topic,
            self.explanation_topic,
            self.vlm_topic,
            self.output_topic,
        )

    def _init_fonts(self):
        if not os.path.exists(self.font_path):
            rospy.logwarn(
                "overlay font path does not exist: %s | falling back to OpenCV text",
                self.font_path,
            )
            return
        try:
            self.text_font = ImageFont.truetype(self.font_path, self.font_size_px)
            self.focus_font = ImageFont.truetype(
                self.font_path, self.focus_font_size_px
            )
        except Exception as exc:
            rospy.logwarn(
                "failed to load overlay font at %s: %s | falling back to OpenCV text",
                self.font_path,
                str(exc),
            )
            self.text_font = None
            self.focus_font = None

    def _parse_json(self, message):
        try:
            return json.loads(message.data)
        except Exception as exc:
            rospy.logwarn("failed to parse overlay viewer payload: %s", str(exc))
            return {}

    def _on_explanation(self, message):
        self.latest_explanation = self._parse_json(message)

    def _on_vlm(self, message):
        self.latest_vlm = self._parse_json(message)

    def _region_to_rect(self, width, height, hint):
        if not hint.get("available"):
            return (0, 0, width - 1, height - 1)

        region = hint.get("horizontal_region", "unknown")
        band = hint.get("vertical_band", "unknown")

        if region == "front_left":
            x0, x1 = int(width * 0.05), int(width * 0.45)
        elif region == "front_right":
            x0, x1 = int(width * 0.55), int(width * 0.95)
        else:
            x0, x1 = int(width * 0.25), int(width * 0.75)

        if band == "ground_near":
            y0, y1 = int(height * 0.60), int(height * 0.95)
        elif band == "upper_height":
            y0, y1 = int(height * 0.05), int(height * 0.45)
        else:
            y0, y1 = int(height * 0.25), int(height * 0.75)

        return (x0, y0, x1, y1)

    def _draw_overlay_panel(self, image):
        panel_height = min(image.shape[0] - 10, 170)
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (image.shape[1] - 10, 10 + panel_height), (0, 0, 0), -1)
        return cv2.addWeighted(overlay, 0.45, image, 0.55, 0.0)

    def _draw_unicode_text(
        self,
        image_bgr,
        text,
        origin,
        color_bgr,
        font,
        max_width_px=None,
        line_height=None,
    ):
        if font is None:
            cv2.putText(
                image_bgr,
                str(text),
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color_bgr,
                1,
                cv2.LINE_AA,
            )
            return image_bgr

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PilImage.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        x, y = origin
        line_height = line_height or self.line_height
        max_width_px = max_width_px or (image_bgr.shape[1] - x - 10)
        fill = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        wrapped_lines = wrap_text_with_font(text, draw, font, max_width_px)
        for line in wrapped_lines:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _select_vlm_for_image(self, image_stamp):
        vlm = self.latest_vlm or {}
        if not vlm:
            return None
        if vlm.get("status") not in ("ok", "partial_ok"):
            return None
        vlm_image_stamp = vlm.get("image_stamp")
        if image_stamp is None or vlm_image_stamp is None:
            return None
        try:
            delta = abs(float(vlm_image_stamp) - float(image_stamp))
        except Exception:
            return None
        if delta > self.vlm_image_match_tolerance_s:
            return None
        return vlm

    def _annotate_image(self, image_bgr, image_stamp=None):
        explanation = self.latest_explanation or {}
        vlm = self._select_vlm_for_image(image_stamp)
        hint = explanation.get("visual_grounding_hint", {})

        annotated = self._draw_overlay_panel(image_bgr.copy())
        h, w = annotated.shape[:2]
        x0, y0, x1, y1 = self._region_to_rect(w, h, hint)

        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 2)

        lines = [
            "event: {}".format(explanation.get("event_label", "-")),
            "final: {}".format(
                (vlm or {}).get(
                    "final_combined_explanation_ko",
                    "VLM 장면 설명 대기 중",
                )
            ),
            "objects: {}".format((vlm or {}).get("detected_objects_ko", [])),
        ]

        y = 35
        for raw_line in lines:
            if self.text_font is not None:
                image_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                pil_image = PilImage.fromarray(image_rgb)
                draw = ImageDraw.Draw(pil_image)
                line_group = wrap_text_with_font(
                    raw_line,
                    draw,
                    self.text_font,
                    annotated.shape[1] - 40,
                )
                annotated = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                line_group = wrap_text(raw_line, 90)

            for line in line_group:
                annotated = self._draw_unicode_text(
                    annotated,
                    line,
                    (20, y),
                    (255, 255, 255),
                    self.text_font,
                    max_width_px=annotated.shape[1] - 40,
                    line_height=self.line_height,
                )
                y += self.line_height
                if y > 165:
                    break
            if y > 165:
                break

        return annotated

    def _on_image(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn("failed to decode overlay image: %s", str(exc))
            return

        annotated = self._annotate_image(image, message.header.stamp.to_sec())

        if self.display_window:
            try:
                cv2.imshow(self.window_name, annotated)
                cv2.waitKey(1)
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "cv2 viewer disabled due to display error: %s", str(exc))

        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out_msg.header = message.header
            self.publisher.publish(out_msg)
        except Exception as exc:
            rospy.logwarn("failed to publish overlay image: %s", str(exc))


def main():
    rospy.init_node("driving_camera_overlay_viewer")
    DrivingCameraOverlayViewer()
    rospy.spin()


if __name__ == "__main__":
    main()
