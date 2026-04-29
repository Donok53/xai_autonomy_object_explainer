#!/usr/bin/env python3
import json
import os
from collections import deque

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
            rospy.get_param("~window_name", "xai_driving_lidar_view")
        )
        self.render_mode = str(
            rospy.get_param("~render_mode", "lidar_only")
        ).strip().lower()
        self.show_detector_boxes = bool(
            rospy.get_param("~show_detector_boxes", False)
        )
        self.show_pointcloud_bbox = bool(
            rospy.get_param("~show_pointcloud_bbox", False)
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
        self.vlm_buffer_size = int(rospy.get_param("~vlm_buffer_size", 40))
        self.lidar_bev_max_forward_m = float(
            rospy.get_param("~lidar_bev_max_forward_m", 3.5)
        )
        self.lidar_bev_half_width_m = float(
            rospy.get_param("~lidar_bev_half_width_m", 2.0)
        )
        self.lidar_bev_point_radius_px = int(
            rospy.get_param("~lidar_bev_point_radius_px", 3)
        )

        self.bridge = CvBridge()
        self.latest_explanation = {}
        self.latest_vlm = {}
        self.vlm_buffer = deque(maxlen=max(5, self.vlm_buffer_size))
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
        if self.latest_vlm:
            self.vlm_buffer.append(self.latest_vlm)

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
        candidates = list(self.vlm_buffer)
        if not candidates:
            return None
        best = None
        best_delta = None
        for vlm in candidates:
            if vlm.get("status") not in ("ok", "partial_ok"):
                continue
            vlm_image_stamp = vlm.get("image_stamp")
            if image_stamp is None or vlm_image_stamp is None:
                continue
            try:
                delta = abs(float(vlm_image_stamp) - float(image_stamp))
            except Exception:
                continue
            if delta > self.vlm_image_match_tolerance_s:
                continue
            if best is None or delta < best_delta:
                best = vlm
                best_delta = delta
        return best

    def _draw_detector_boxes(self, image_bgr, payload):
        if not self.show_detector_boxes:
            return image_bgr
        detections = (
            (payload or {})
            .get("detector_details", {})
            .get("detections", [])
        )
        if not detections:
            return image_bgr
        selected = (payload or {}).get("detector_details", {}).get("selected_detection") or {}
        selected_bbox = tuple(selected.get("bbox_xyxy_full") or [])
        annotated = image_bgr
        for item in detections:
            bbox = item.get("bbox_xyxy_full") or []
            if len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in bbox]
            is_selected = tuple(bbox) == selected_bbox and len(selected_bbox) == 4
            thickness = 3 if is_selected else 2
            color = (0, 255, 255)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), color, thickness)
            label = item.get("label_ko") or item.get("label") or "object"
            confidence = item.get("confidence")
            memory_id = item.get("memory_id")
            track_key = item.get("track_key")
            id_parts = []
            if memory_id:
                id_parts.append(str(memory_id))
            if track_key:
                id_parts.append(str(track_key))
            if id_parts:
                label = "[{}] {}".format(" / ".join(id_parts), label)
            if confidence is not None:
                label = "{} {:.2f}".format(label, float(confidence))
            annotated = self._draw_unicode_text(
                annotated,
                label,
                (max(5, x0), max(20, y0 - 8)),
                color,
                self.text_font,
                max_width_px=max(100, annotated.shape[1] - x0 - 10),
                line_height=self.line_height,
                )
        return annotated

    def _draw_pointcloud_overlay(self, image_bgr, payload):
        pointcloud_visual = (
            (payload or {})
            .get("detector_details", {})
            .get("pointcloud_visual", {})
        )
        if not isinstance(pointcloud_visual, dict) or not pointcloud_visual.get("available"):
            return image_bgr

        annotated = image_bgr
        points = pointcloud_visual.get("projected_points_xy_full") or []
        bbox = pointcloud_visual.get("bbox_xyxy_full") or []
        point_radius = int(max(1, pointcloud_visual.get("point_radius_px") or 2))
        for point_xy in points:
            if len(point_xy) != 2:
                continue
            px = int(point_xy[0])
            py = int(point_xy[1])
            cv2.circle(
                annotated,
                (px, py),
                point_radius,
                (255, 255, 0),
                -1,
                lineType=cv2.LINE_AA,
            )

        if self.show_pointcloud_bbox and len(bbox) == 4:
            x0, y0, x1, y1 = [int(value) for value in bbox]
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 120, 0), 2)
            label = "LiDAR {}pts".format(int(pointcloud_visual.get("point_count") or 0))
            annotated = self._draw_unicode_text(
                annotated,
                label,
                (max(5, x0), max(20, y1 + 20)),
                (255, 180, 0),
                self.text_font,
                max_width_px=max(100, annotated.shape[1] - x0 - 10),
                line_height=self.line_height,
            )
        return annotated

    def _draw_lidar_bev(self, image_bgr, payload):
        pointcloud_visual = (
            (payload or {})
            .get("detector_details", {})
            .get("pointcloud_visual", {})
        )
        if not isinstance(pointcloud_visual, dict) or not pointcloud_visual.get("available"):
            return image_bgr

        context_points = pointcloud_visual.get("context_points_xyz") or []
        cluster_points = pointcloud_visual.get("cluster_points_xyz") or []
        if not context_points and not cluster_points:
            return image_bgr

        annotated = image_bgr
        canvas_h, canvas_w = annotated.shape[:2]
        top_margin = 185
        bottom_margin = 28
        side_margin = 32
        draw_h = max(40, canvas_h - top_margin - bottom_margin)
        draw_w = max(40, canvas_w - (2 * side_margin))
        robot_px = (canvas_w // 2, canvas_h - bottom_margin)

        max_forward_m = max(1.0, float(self.lidar_bev_max_forward_m))
        half_width_m = max(0.5, float(self.lidar_bev_half_width_m))
        point_radius_px = max(1, int(self.lidar_bev_point_radius_px))

        def world_to_canvas(point_xyz):
            forward_x = float(point_xyz[0])
            lateral_y = float(point_xyz[1])
            norm_y = lateral_y / half_width_m
            norm_x = forward_x / max_forward_m
            px = int(round((canvas_w * 0.5) - (norm_y * (draw_w * 0.5))))
            py = int(round((canvas_h - bottom_margin) - (norm_x * draw_h)))
            return px, py

        overlay = annotated.copy()
        grid_color = (35, 35, 35)
        for forward_m in (0.5, 1.0, 1.5, 2.0, 3.0):
            if forward_m >= max_forward_m:
                continue
            _, py = world_to_canvas((forward_m, 0.0, 0.0))
            cv2.line(
                overlay,
                (side_margin, py),
                (canvas_w - side_margin, py),
                grid_color,
                1,
                cv2.LINE_AA,
            )
        cv2.line(
            overlay,
            (canvas_w // 2, top_margin),
            (canvas_w // 2, canvas_h - bottom_margin),
            grid_color,
            1,
            cv2.LINE_AA,
        )
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0.0)

        for point_xyz in context_points:
            px, py = world_to_canvas(point_xyz)
            if px < side_margin or px >= (canvas_w - side_margin):
                continue
            if py < top_margin or py >= (canvas_h - bottom_margin):
                continue
            cv2.circle(
                annotated,
                (px, py),
                max(1, point_radius_px - 1),
                (80, 80, 80),
                -1,
                lineType=cv2.LINE_AA,
            )

        for point_xyz in cluster_points:
            px, py = world_to_canvas(point_xyz)
            if px < side_margin or px >= (canvas_w - side_margin):
                continue
            if py < top_margin or py >= (canvas_h - bottom_margin):
                continue
            cv2.circle(
                annotated,
                (px, py),
                point_radius_px,
                (0, 255, 255),
                -1,
                lineType=cv2.LINE_AA,
            )

        anchor_xyz = pointcloud_visual.get("anchor_xyz") or {}
        if anchor_xyz:
            anchor_px, anchor_py = world_to_canvas(
                (
                    float(anchor_xyz.get("x") or 0.0),
                    float(anchor_xyz.get("y") or 0.0),
                    float(anchor_xyz.get("z") or 0.0),
                )
            )
            cv2.circle(
                annotated,
                (anchor_px, anchor_py),
                max(3, point_radius_px + 2),
                (0, 180, 255),
                2,
                lineType=cv2.LINE_AA,
            )

        robot_triangle = np.array(
            [
                [robot_px[0], robot_px[1] - 16],
                [robot_px[0] - 10, robot_px[1] + 8],
                [robot_px[0] + 10, robot_px[1] + 8],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(annotated, robot_triangle, (120, 120, 120))
        cv2.circle(annotated, robot_px, 3, (220, 220, 220), -1, lineType=cv2.LINE_AA)

        info_text = "LiDAR {}pts | selected {}pts".format(
            int(pointcloud_visual.get("context_point_count") or len(context_points)),
            int(pointcloud_visual.get("point_count") or len(cluster_points)),
        )
        if pointcloud_visual.get("distance_hint_m") is not None:
            try:
                info_text += " | {:.1f}m".format(float(pointcloud_visual.get("distance_hint_m")))
            except Exception:
                pass
        annotated = self._draw_unicode_text(
            annotated,
            info_text,
            (20, max(190, canvas_h - 26)),
            (180, 220, 255),
            self.text_font,
            max_width_px=canvas_w - 40,
            line_height=self.line_height,
        )
        return annotated

    def _annotate_image(self, image_bgr, image_stamp=None):
        explanation = self.latest_explanation or {}
        vlm = self._select_vlm_for_image(image_stamp)
        selected = (vlm or {}).get("detector_details", {}).get("selected_detection") or {}
        memory_id = selected.get("memory_id")
        track_key = selected.get("track_key")
        selected_id_text = "-"
        selected_id_parts = []
        if memory_id:
            selected_id_parts.append(str(memory_id))
        if track_key:
            selected_id_parts.append(str(track_key))
        if selected_id_parts:
            selected_id_text = " / ".join(selected_id_parts)

        if self.render_mode == "camera":
            base_image = image_bgr.copy()
        else:
            base_image = np.zeros_like(image_bgr)

        annotated = self._draw_overlay_panel(base_image)
        if self.render_mode == "camera":
            annotated = self._draw_pointcloud_overlay(annotated, vlm)
        else:
            annotated = self._draw_lidar_bev(annotated, vlm)
        annotated = self._draw_detector_boxes(annotated, vlm)

        lines = [
            "event: {}".format(explanation.get("event_label", "-")),
            "id: {}".format(selected_id_text),
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
