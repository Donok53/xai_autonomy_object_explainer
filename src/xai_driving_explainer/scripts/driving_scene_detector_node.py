#!/usr/bin/env python3
import json
import os
import queue
import subprocess
import threading
import time
import base64
import math
from collections import deque

import cv2
import rospy
import sensor_msgs.point_cloud2 as point_cloud2
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
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


def nested_get(payload, *keys, default=None):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


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


def format_distance(distance_m):
    if distance_m is None:
        return ""
    try:
        return "약 {:.1f}m ".format(float(distance_m))
    except Exception:
        return ""


def normalize_region_phrase(region_text):
    text = str(region_text or "").strip()
    if not text:
        return "전방 통로"
    if "정확한 시각적 대응 영역은 아직 추정하지 못했다" in text:
        return "전방 통로"
    return text


COMMON_LABEL_KO = {
    "person": "사람",
    "umbrella": "우산",
    "chair": "의자",
    "bench": "벤치",
    "bottle": "병",
    "cup": "컵",
    "backpack": "가방",
    "handbag": "가방",
    "suitcase": "캐리어",
    "potted plant": "화분",
    "vase": "화병",
    "book": "책",
    "tv": "TV",
    "refrigerator": "냉장고",
    "microwave": "전자레인지",
    "oven": "오븐",
    "sink": "세면대",
    "toilet": "변기",
    "couch": "소파",
    "bed": "침대",
    "dining table": "테이블",
    "laptop": "노트북",
    "cell phone": "휴대폰",
    "keyboard": "키보드",
    "mouse": "마우스",
    "clock": "시계",
    "truck": "트럭",
    "car": "자동차",
    "bus": "버스",
    "bicycle": "자전거",
    "motorcycle": "오토바이",
}


def english_label_to_korean(label):
    label = str(label or "").strip().lower()
    if not label:
        return "장애물"
    return COMMON_LABEL_KO.get(label, label)


def choose_subject_particle(text):
    value = str(text or "").strip()
    if not value:
        return "가"
    last = value[-1]
    code = ord(last)
    if 0xAC00 <= code <= 0xD7A3:
        return "이" if ((code - 0xAC00) % 28) != 0 else "가"
    return "가"


def bbox_iou_xyxy(box_a, box_b):
    if not box_a or not box_b:
        return 0.0
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    area_a = float(max(1, ax1 - ax0) * max(1, ay1 - ay0))
    area_b = float(max(1, bx1 - bx0) * max(1, by1 - by0))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def bbox_center_distance(box_a, box_b):
    if not box_a or not box_b:
        return float("inf")
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    acx = 0.5 * (ax0 + ax1)
    acy = 0.5 * (ay0 + ay1)
    bcx = 0.5 * (bx0 + bx1)
    bcy = 0.5 * (by0 + by1)
    dx = acx - bcx
    dy = acy - bcy
    return (dx * dx + dy * dy) ** 0.5


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def yaw_from_quaternion(quaternion):
    x = float(getattr(quaternion, "x", 0.0) or 0.0)
    y = float(getattr(quaternion, "y", 0.0) or 0.0)
    z = float(getattr(quaternion, "z", 0.0) or 0.0)
    w = float(getattr(quaternion, "w", 1.0) or 1.0)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def rotate_point_by_quaternion(point_xyz, quaternion_xyzw):
    px, py, pz = point_xyz
    qx, qy, qz, qw = quaternion_xyzw
    # Quaternion rotation: p' = q * p * q_conjugate.
    ix = qw * px + qy * pz - qz * py
    iy = qw * py + qz * px - qx * pz
    iz = qw * pz + qx * py - qy * px
    iw = -qx * px - qy * py - qz * pz

    rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
    ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
    rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
    return (rx, ry, rz)


class YoloWorkerClient:
    def __init__(
        self,
        python_path,
        worker_script,
        model_name,
        request_timeout_s,
        confidence_threshold,
        iou_threshold,
        max_det,
        imgsz,
        device,
        use_track,
        tracker_config,
    ):
        self.python_path = python_path
        self.worker_script = worker_script
        self.model_name = model_name
        self.request_timeout_s = float(request_timeout_s)
        self.lock = threading.Lock()
        self.stdout_queue = queue.Queue()
        self.stderr_thread = None
        self.stdout_thread = None
        self.request_id = 0
        command = [
            self.python_path,
            self.worker_script,
            "--model",
            self.model_name,
            "--conf",
            str(confidence_threshold),
            "--iou",
            str(iou_threshold),
            "--max-det",
            str(max_det),
            "--imgsz",
            str(imgsz),
            "--device",
            str(device),
            "--tracker",
            str(tracker_config),
        ]
        if use_track:
            command.append("--use-track")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.stdout_thread = threading.Thread(
            target=self._pump_stdout,
            name="yolo-worker-stdout",
            daemon=True,
        )
        self.stderr_thread = threading.Thread(
            target=self._pump_stderr,
            name="yolo-worker-stderr",
            daemon=True,
        )
        self.stdout_thread.start()
        self.stderr_thread.start()
        self._wait_for_ready()

    def _pump_stdout(self):
        try:
            for line in self.process.stdout:
                line = line.strip()
                if line:
                    self.stdout_queue.put(line)
        finally:
            self.stdout_queue.put(
                json.dumps(
                    {
                        "type": "worker_exit",
                        "returncode": self.process.poll(),
                    }
                )
            )

    def _pump_stderr(self):
        try:
            for line in self.process.stderr:
                line = line.rstrip()
                if line:
                    rospy.loginfo_throttle(5.0, "[YOLO-WORKER] %s", line)
        except Exception:
            return

    def _wait_for_ready(self):
        deadline = time.time() + max(30.0, self.request_timeout_s)
        while time.time() < deadline:
            remaining = deadline - time.time()
            try:
                raw_line = self.stdout_queue.get(timeout=max(0.1, remaining))
            except queue.Empty:
                continue
            try:
                payload = json.loads(raw_line)
            except Exception:
                continue
            if payload.get("type") == "ready":
                return
            if payload.get("type") == "error":
                raise RuntimeError(payload.get("message") or "yolo worker failed")
            if payload.get("type") == "worker_exit":
                raise RuntimeError(
                    "yolo worker exited before ready (returncode={})".format(
                        payload.get("returncode")
                    )
                )
        raise RuntimeError("timed out while waiting for yolo worker readiness")

    def infer(self, image_bgr, jpeg_quality):
        success, encoded = cv2.imencode(
            ".jpg",
            image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not success:
            raise RuntimeError("failed to encode image for yolo worker")
        with self.lock:
            self.request_id += 1
            request_id = self.request_id
            outgoing = {
                "type": "infer",
                "request_id": request_id,
                "image_jpeg_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
            self.process.stdin.write(json.dumps(outgoing) + "\n")
            self.process.stdin.flush()
            deadline = time.time() + self.request_timeout_s
            while time.time() < deadline:
                remaining = deadline - time.time()
                try:
                    raw_line = self.stdout_queue.get(timeout=max(0.1, remaining))
                except queue.Empty:
                    continue
                try:
                    payload = json.loads(raw_line)
                except Exception:
                    continue
                if payload.get("type") == "worker_exit":
                    raise RuntimeError(
                        "yolo worker exited during inference (returncode={})".format(
                            payload.get("returncode")
                        )
                    )
                if payload.get("type") == "result" and payload.get("request_id") == request_id:
                    return payload
                if payload.get("type") == "error":
                    raise RuntimeError(payload.get("message") or "yolo worker error")
            raise RuntimeError("timed out while waiting for yolo worker response")

    def shutdown(self):
        if not self.process:
            return
        try:
            if self.process.poll() is None and self.process.stdin:
                self.process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=1.0)
                return
        except Exception:
            pass
        try:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=3.0)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


class DrivingSceneDetectorNode:
    def __init__(self):
        self.explanation_topic = rospy.get_param(
            "~explanation_topic", "/xai/driving_explanations"
        )
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.output_topic = rospy.get_param(
            "~output_topic", "/xai/driving_vlm_explanations"
        )
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.odom_topic = rospy.get_param(
            "~odom_topic", "/lio_localizer/odometry/optimization"
        )
        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera/color/camera_info"
        )
        self.point_cloud_topic = rospy.get_param(
            "~point_cloud_topic", "/planning/linefit_ground/non_ground_cloud"
        )
        self.backend = str(rospy.get_param("~backend", "yolo_worker")).strip().lower()
        self.allowed_event_labels = self._parse_allowed_event_labels(
            rospy.get_param(
                "~allowed_event_labels",
                "path_blocked,path_update,state_update",
            )
        )
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
        self.max_point_cloud_dt_s = float(
            rospy.get_param("~max_point_cloud_dt_s", 1.0)
        )
        self.max_buffer_size = int(rospy.get_param("~max_buffer_size", 60))
        self.point_cloud_buffer_size = int(
            rospy.get_param("~point_cloud_buffer_size", 12)
        )
        self.use_focus_crop = bool(rospy.get_param("~use_focus_crop", True))
        self.focus_crop_margin_ratio = float(
            rospy.get_param("~focus_crop_margin_ratio", 0.12)
        )
        self.max_image_side_px = int(rospy.get_param("~max_image_side_px", 416))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 80))
        self.min_process_interval_s = float(
            rospy.get_param("~min_process_interval_s", 0.25)
        )
        self.yolo_request_timeout_s = float(
            rospy.get_param("~yolo_request_timeout_s", 3.0)
        )
        self.yolo_confidence_threshold = float(
            rospy.get_param("~yolo_confidence_threshold", 0.25)
        )
        self.yolo_iou_threshold = float(
            rospy.get_param("~yolo_iou_threshold", 0.45)
        )
        self.yolo_max_det = int(rospy.get_param("~yolo_max_det", 8))
        self.yolo_imgsz = int(rospy.get_param("~yolo_imgsz", 320))
        self.yolo_device = str(rospy.get_param("~yolo_device", "auto"))
        self.yolo_model = str(rospy.get_param("~yolo_model", "yolo11n.pt"))
        self.yolo_use_tracker = bool(rospy.get_param("~yolo_use_tracker", True))
        self.yolo_tracker_config = str(
            rospy.get_param("~yolo_tracker_config", "bytetrack.yaml")
        )
        self.yolo_worker_python = str(
            rospy.get_param(
                "~yolo_worker_python",
                "/home/byeongjae/miniconda3/envs/vad/bin/python",
            )
        )
        self.yolo_worker_script = str(
            rospy.get_param(
                "~yolo_worker_script",
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "yolo_ultralytics_worker.py",
                ),
            )
        )
        self.hog_confidence_threshold = float(
            rospy.get_param("~hog_confidence_threshold", 0.3)
        )
        self.obstacle_min_area_ratio = float(
            rospy.get_param("~obstacle_min_area_ratio", 0.012)
        )
        self.obstacle_min_width_ratio = float(
            rospy.get_param("~obstacle_min_width_ratio", 0.08)
        )
        self.obstacle_min_height_ratio = float(
            rospy.get_param("~obstacle_min_height_ratio", 0.12)
        )
        self.track_hold_ttl_s = float(rospy.get_param("~track_hold_ttl_s", 0.9))
        self.track_iou_match_threshold = float(
            rospy.get_param("~track_iou_match_threshold", 0.20)
        )
        self.track_center_match_px = float(
            rospy.get_param("~track_center_match_px", 120.0)
        )
        self.track_bbox_alpha = float(rospy.get_param("~track_bbox_alpha", 0.35))
        self.track_label_decay = float(rospy.get_param("~track_label_decay", 0.88))
        self.track_min_hits = int(rospy.get_param("~track_min_hits", 2))
        self.use_motion_awareness = bool(
            rospy.get_param("~use_motion_awareness", True)
        )
        self.motion_linear_speed_ref_mps = float(
            rospy.get_param("~motion_linear_speed_ref_mps", 0.45)
        )
        self.motion_angular_speed_ref_radps = float(
            rospy.get_param("~motion_angular_speed_ref_radps", 0.70)
        )
        self.motion_extra_hold_ttl_s = float(
            rospy.get_param("~motion_extra_hold_ttl_s", 0.70)
        )
        self.motion_extra_center_match_px = float(
            rospy.get_param("~motion_extra_center_match_px", 100.0)
        )
        self.motion_label_decay_bonus = float(
            rospy.get_param("~motion_label_decay_bonus", 0.06)
        )
        self.motion_selected_track_bonus = float(
            rospy.get_param("~motion_selected_track_bonus", 0.35)
        )
        self.motion_selected_track_distance_px = float(
            rospy.get_param("~motion_selected_track_distance_px", 140.0)
        )
        self.semantic_memory_enabled = bool(
            rospy.get_param("~semantic_memory_enabled", True)
        )
        self.semantic_memory_match_radius_m = float(
            rospy.get_param("~semantic_memory_match_radius_m", 0.75)
        )
        self.semantic_memory_ttl_s = float(
            rospy.get_param("~semantic_memory_ttl_s", 45.0)
        )
        self.semantic_memory_person_ttl_s = float(
            rospy.get_param("~semantic_memory_person_ttl_s", 3.0)
        )
        self.semantic_memory_label_decay = float(
            rospy.get_param("~semantic_memory_label_decay", 0.985)
        )
        self.semantic_memory_min_hits = int(
            rospy.get_param("~semantic_memory_min_hits", 3)
        )
        self.semantic_memory_override_margin = float(
            rospy.get_param("~semantic_memory_override_margin", 0.08)
        )
        self.semantic_memory_position_alpha = float(
            rospy.get_param("~semantic_memory_position_alpha", 0.25)
        )
        self.point_cloud_projection_enabled = bool(
            rospy.get_param("~point_cloud_projection_enabled", True)
        )
        self.point_cloud_anchor_radius_m = float(
            rospy.get_param("~point_cloud_anchor_radius_m", 0.45)
        )
        self.point_cloud_projection_max_range_m = float(
            rospy.get_param("~point_cloud_projection_max_range_m", 6.0)
        )
        self.point_cloud_projection_max_points = int(
            rospy.get_param("~point_cloud_projection_max_points", 350)
        )
        self.point_cloud_overlay_point_radius_px = int(
            rospy.get_param("~point_cloud_overlay_point_radius_px", 2)
        )
        self.point_cloud_association_iou_threshold = float(
            rospy.get_param("~point_cloud_association_iou_threshold", 0.03)
        )
        self.point_cloud_association_center_match_px = float(
            rospy.get_param("~point_cloud_association_center_match_px", 80.0)
        )
        self.point_cloud_use_tf = bool(
            rospy.get_param("~point_cloud_use_tf", True)
        )
        self.point_cloud_fallback_transform_enabled = bool(
            rospy.get_param("~point_cloud_fallback_transform_enabled", True)
        )
        self.point_cloud_fallback_source_frame = str(
            rospy.get_param("~point_cloud_fallback_source_frame", "base_link")
        )
        self.point_cloud_fallback_target_frame = str(
            rospy.get_param(
                "~point_cloud_fallback_target_frame",
                "camera_color_optical_frame",
            )
        )
        self.point_cloud_fallback_translation_xyz = [
            float(value)
            for value in rospy.get_param(
                "~point_cloud_fallback_translation_xyz",
                [0.0, -0.05913, 0.0],
            )
        ]
        self.point_cloud_fallback_rotation_xyzw = [
            float(value)
            for value in rospy.get_param(
                "~point_cloud_fallback_rotation_xyzw",
                [-0.5, 0.5, -0.5, 0.5],
            )
        ]
        self.log_detector_explanation = bool(
            rospy.get_param("~log_detector_explanation", True)
        )

        self.bridge = CvBridge()
        self.image_buffer = deque(maxlen=max(5, self.max_buffer_size))
        self.point_cloud_buffer = deque(maxlen=max(3, self.point_cloud_buffer_size))
        self.last_signature = None
        self.last_process_time = 0.0
        self.yolo_worker = None
        self.track_states = {}
        self.next_local_track_id = 1
        self.selected_track_key = None
        self.semantic_memories = {}
        self.next_semantic_memory_id = 1
        self.latest_cmd_vel = {
            "linear_x_mps": 0.0,
            "linear_y_mps": 0.0,
            "angular_z_radps": 0.0,
            "arrival_wall_time": 0.0,
        }
        self.latest_odom = {
            "linear_speed_mps": 0.0,
            "angular_speed_radps": 0.0,
            "arrival_wall_time": 0.0,
            "position_x": 0.0,
            "position_y": 0.0,
            "yaw_rad": 0.0,
            "frame_id": "",
        }
        self.latest_camera_info = {
            "arrival_wall_time": 0.0,
            "frame_id": "",
            "width": 0,
            "height": 0,
            "fx": 0.0,
            "fy": 0.0,
            "cx": 0.0,
            "cy": 0.0,
        }
        self.point_cloud_transform_warned = False
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.hog = None
        if self.backend == "hog_person":
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        if self.backend == "yolo_worker":
            try:
                self.yolo_worker = YoloWorkerClient(
                    python_path=self.yolo_worker_python,
                    worker_script=self.yolo_worker_script,
                    model_name=self.yolo_model,
                    request_timeout_s=self.yolo_request_timeout_s,
                    confidence_threshold=self.yolo_confidence_threshold,
                    iou_threshold=self.yolo_iou_threshold,
                    max_det=self.yolo_max_det,
                    imgsz=self.yolo_imgsz,
                    device=self.yolo_device,
                    use_track=self.yolo_use_tracker,
                    tracker_config=self.yolo_tracker_config,
                )
                rospy.on_shutdown(self._shutdown_worker)
            except Exception as exc:
                rospy.logwarn(
                    "failed to start yolo worker (%s). detector will run without visual boxes until backend is fixed.",
                    str(exc),
                )

        self.publisher = rospy.Publisher(self.output_topic, String, queue_size=20)
        self.explanation_subscriber = rospy.Subscriber(
            self.explanation_topic, String, self._on_explanation, queue_size=20
        )
        self.image_subscriber = rospy.Subscriber(
            self.image_topic, Image, self._on_image, queue_size=5
        )
        self.camera_info_subscriber = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self._on_camera_info, queue_size=5
        )
        self.point_cloud_subscriber = rospy.Subscriber(
            self.point_cloud_topic, PointCloud2, self._on_point_cloud, queue_size=5
        )
        self.cmd_vel_subscriber = rospy.Subscriber(
            self.cmd_vel_topic, Twist, self._on_cmd_vel, queue_size=20
        )
        self.odom_subscriber = rospy.Subscriber(
            self.odom_topic, Odometry, self._on_odom, queue_size=20
        )

        rospy.loginfo(
            "driving_scene_detector started | backend=%s explanation=%s image=%s camera_info=%s point_cloud=%s output=%s cmd_vel=%s odom=%s",
            self.backend,
            self.explanation_topic,
            self.image_topic,
            self.camera_info_topic,
            self.point_cloud_topic,
            self.output_topic,
            self.cmd_vel_topic,
            self.odom_topic,
        )

    def _shutdown_worker(self):
        if self.yolo_worker is not None:
            self.yolo_worker.shutdown()
            self.yolo_worker = None

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
            rospy.logwarn("failed to parse scene detector payload: %s", str(exc))
            return {}

    def _on_image(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn("failed to decode detector image: %s", str(exc))
            return
        self.image_buffer.append(
            {
                "stamp": stamp_to_float(message.header.stamp),
                "arrival_wall_time": time.time(),
                "image": image,
            }
        )

    def _on_camera_info(self, message):
        try:
            k_values = list(message.K)
            self.latest_camera_info = {
                "arrival_wall_time": time.time(),
                "frame_id": str(message.header.frame_id or ""),
                "width": int(message.width or 0),
                "height": int(message.height or 0),
                "fx": float(k_values[0] or 0.0),
                "fy": float(k_values[4] or 0.0),
                "cx": float(k_values[2] or 0.0),
                "cy": float(k_values[5] or 0.0),
            }
        except Exception as exc:
            rospy.logwarn("failed to parse camera info: %s", str(exc))

    def _on_point_cloud(self, message):
        points = []
        try:
            step = max(1, int(message.width // 5000) or 1)
            for index, point in enumerate(
                point_cloud2.read_points(
                    message,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
            ):
                if (index % step) != 0:
                    continue
                x_value, y_value, z_value = point[:3]
                points.append(
                    (
                        float(x_value),
                        float(y_value),
                        float(z_value),
                    )
                )
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0,
                "failed to parse point cloud for detector: %s",
                str(exc),
            )
            return

        self.point_cloud_buffer.append(
            {
                "stamp": stamp_to_float(message.header.stamp),
                "arrival_wall_time": time.time(),
                "frame_id": str(message.header.frame_id or ""),
                "points_xyz": points,
            }
        )

    def _on_cmd_vel(self, message):
        self.latest_cmd_vel = {
            "linear_x_mps": float(getattr(message.linear, "x", 0.0) or 0.0),
            "linear_y_mps": float(getattr(message.linear, "y", 0.0) or 0.0),
            "angular_z_radps": float(getattr(message.angular, "z", 0.0) or 0.0),
            "arrival_wall_time": time.time(),
        }

    def _on_odom(self, message):
        position = getattr(message.pose.pose, "position", None)
        orientation = getattr(message.pose.pose, "orientation", None)
        linear = getattr(message.twist.twist, "linear", None)
        angular = getattr(message.twist.twist, "angular", None)
        linear_x = float(getattr(linear, "x", 0.0) or 0.0)
        linear_y = float(getattr(linear, "y", 0.0) or 0.0)
        linear_speed = (linear_x * linear_x + linear_y * linear_y) ** 0.5
        angular_speed = abs(float(getattr(angular, "z", 0.0) or 0.0))
        self.latest_odom = {
            "linear_speed_mps": float(linear_speed),
            "angular_speed_radps": float(angular_speed),
            "arrival_wall_time": time.time(),
            "position_x": float(getattr(position, "x", 0.0) or 0.0),
            "position_y": float(getattr(position, "y", 0.0) or 0.0),
            "yaw_rad": yaw_from_quaternion(orientation) if orientation is not None else 0.0,
            "frame_id": str(getattr(message.header, "frame_id", "") or ""),
        }

    def _nearest_image(self, target_stamp, arrival_wall_time):
        if not self.image_buffer:
            return None
        if target_stamp is None:
            latest = self.image_buffer[-1]
            if self.allow_arrival_time_fallback:
                age = arrival_wall_time - float(
                    latest.get("arrival_wall_time") or arrival_wall_time
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
        if delta <= self.max_image_dt_s:
            return best
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
                    "scene detector using oldest buffered image fallback | target=%.3f oldest=%.3f",
                    float(target_stamp),
                    float(oldest_stamp),
                )
                return oldest

        if latest_stamp is not None and target_stamp > latest_stamp:
            if (target_stamp - latest_stamp) <= self.max_future_image_fallback_s:
                rospy.loginfo_throttle(
                    5.0,
                    "scene detector using latest buffered image fallback | target=%.3f latest=%.3f",
                    float(target_stamp),
                    float(latest_stamp),
                )
                return latest

        if self.allow_arrival_time_fallback:
            latest_age = arrival_wall_time - float(
                latest.get("arrival_wall_time") or arrival_wall_time
            )
            if latest_age <= self.max_image_arrival_age_s:
                rospy.loginfo_throttle(
                    5.0,
                    "scene detector using latest buffered image by arrival-time fallback | target=%.3f latest_stamp=%.3f latest_age=%.3f",
                    float(target_stamp),
                    float(latest_stamp or 0.0),
                    float(latest_age),
                )
                return latest
        return None

    def _nearest_point_cloud(self, target_stamp, arrival_wall_time):
        if not self.point_cloud_buffer:
            return None
        if target_stamp is None:
            latest = self.point_cloud_buffer[-1]
            age = arrival_wall_time - float(
                latest.get("arrival_wall_time") or arrival_wall_time
            )
            if age <= self.max_image_arrival_age_s:
                return latest
            return None

        buffered = list(self.point_cloud_buffer)
        best = min(
            buffered,
            key=lambda item: abs((item.get("stamp") or 0.0) - target_stamp),
        )
        delta = abs((best.get("stamp") or 0.0) - target_stamp)
        if delta <= self.max_point_cloud_dt_s:
            return best
        latest = buffered[-1]
        latest_age = arrival_wall_time - float(
            latest.get("arrival_wall_time") or arrival_wall_time
        )
        if latest_age <= self.max_image_arrival_age_s:
            return latest
        return None

    def _current_camera_projection(self):
        info = self.latest_camera_info or {}
        if float(info.get("fx") or 0.0) <= 0.0 or float(info.get("fy") or 0.0) <= 0.0:
            return None
        return info

    def _lookup_point_cloud_transform(self, source_frame, target_frame):
        if not source_frame or not target_frame:
            return None
        if source_frame == target_frame:
            return {
                "translation_xyz": (0.0, 0.0, 0.0),
                "rotation_xyzw": (0.0, 0.0, 0.0, 1.0),
                "mode": "identity",
            }

        if self.point_cloud_use_tf:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rospy.Time(0),
                    rospy.Duration(0.05),
                )
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                self.point_cloud_transform_warned = False
                return {
                    "translation_xyz": (
                        float(translation.x),
                        float(translation.y),
                        float(translation.z),
                    ),
                    "rotation_xyzw": (
                        float(rotation.x),
                        float(rotation.y),
                        float(rotation.z),
                        float(rotation.w),
                    ),
                    "mode": "tf",
                }
            except Exception as exc:
                if not self.point_cloud_transform_warned:
                    rospy.logwarn(
                        "point cloud transform lookup failed (%s -> %s): %s",
                        source_frame,
                        target_frame,
                        str(exc),
                    )
                    self.point_cloud_transform_warned = True

        if (
            self.point_cloud_fallback_transform_enabled
            and source_frame == self.point_cloud_fallback_source_frame
            and target_frame == self.point_cloud_fallback_target_frame
        ):
            return {
                "translation_xyz": tuple(self.point_cloud_fallback_translation_xyz[:3]),
                "rotation_xyzw": tuple(self.point_cloud_fallback_rotation_xyzw[:4]),
                "mode": "fallback_param",
            }
        return None

    def _project_point_cloud_visual(self, bundle, image_entry):
        if not self.point_cloud_projection_enabled or image_entry is None:
            return None

        point_cloud_entry = self._nearest_point_cloud(
            bundle.get("stamp"),
            image_entry.get("arrival_wall_time") or time.time(),
        )
        if point_cloud_entry is None:
            return None

        hint = bundle.get("visual_grounding_hint", {}) or {}
        anchor = hint.get("anchor_xyz") or {}
        anchor_x = anchor.get("x")
        anchor_y = anchor.get("y")
        anchor_z = anchor.get("z")
        if anchor_x is None or anchor_y is None:
            return None

        cluster_points = []
        radius = max(0.10, float(self.point_cloud_anchor_radius_m))
        for point_xyz in point_cloud_entry.get("points_xyz") or []:
            dx = float(point_xyz[0]) - float(anchor_x)
            dy = float(point_xyz[1]) - float(anchor_y)
            dz = float(point_xyz[2]) - float(anchor_z or 0.0)
            if (dx * dx + dy * dy + dz * dz) ** 0.5 <= radius:
                cluster_points.append(point_xyz)

        if not cluster_points:
            return None

        sampled_cluster_points = cluster_points
        max_points = max(40, int(self.point_cloud_projection_max_points))
        if len(sampled_cluster_points) > max_points:
            stride = int(math.ceil(float(len(sampled_cluster_points)) / float(max_points)))
            sampled_cluster_points = sampled_cluster_points[::stride]

        visual = {
            "available": True,
            "cloud_frame_id": point_cloud_entry.get("frame_id"),
            "anchor_xyz": {
                "x": float(anchor_x),
                "y": float(anchor_y),
                "z": float(anchor_z or 0.0),
            },
            "point_count": len(cluster_points),
            "sampled_point_count": len(sampled_cluster_points),
            "cluster_points_xyz": [
                [float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])]
                for point_xyz in sampled_cluster_points
            ],
            "distance_hint_m": hint.get("distance_hint_m"),
            "projected_available": False,
        }

        camera_projection = self._current_camera_projection()
        if camera_projection is None:
            return visual

        transform = self._lookup_point_cloud_transform(
            str(point_cloud_entry.get("frame_id") or ""),
            str(camera_projection.get("frame_id") or ""),
        )
        if transform is None:
            return visual

        translation_xyz = transform.get("translation_xyz") or (0.0, 0.0, 0.0)
        rotation_xyzw = transform.get("rotation_xyzw") or (0.0, 0.0, 0.0, 1.0)
        fx = float(camera_projection.get("fx") or 0.0)
        fy = float(camera_projection.get("fy") or 0.0)
        cx = float(camera_projection.get("cx") or 0.0)
        cy = float(camera_projection.get("cy") or 0.0)
        image_width = max(
            1,
            int(camera_projection.get("width") or image_entry["image"].shape[1]),
        )
        image_height = max(
            1,
            int(camera_projection.get("height") or image_entry["image"].shape[0]),
        )
        max_range = max(0.5, float(self.point_cloud_projection_max_range_m))

        projected_points = []
        min_u = None
        min_v = None
        max_u = None
        max_v = None
        projected_depths = []

        for point_xyz in cluster_points:
            rotated = rotate_point_by_quaternion(point_xyz, rotation_xyzw)
            camera_x = float(rotated[0]) + float(translation_xyz[0])
            camera_y = float(rotated[1]) + float(translation_xyz[1])
            camera_z = float(rotated[2]) + float(translation_xyz[2])
            if camera_z <= 0.10 or camera_z >= max_range:
                continue
            pixel_u = (fx * (camera_x / camera_z)) + cx
            pixel_v = (fy * (camera_y / camera_z)) + cy
            if pixel_u < 0.0 or pixel_u >= float(image_width):
                continue
            if pixel_v < 0.0 or pixel_v >= float(image_height):
                continue
            projected_depths.append(camera_z)
            projected_points.append([int(round(pixel_u)), int(round(pixel_v))])
            min_u = pixel_u if min_u is None else min(min_u, pixel_u)
            min_v = pixel_v if min_v is None else min(min_v, pixel_v)
            max_u = pixel_u if max_u is None else max(max_u, pixel_u)
            max_v = pixel_v if max_v is None else max(max_v, pixel_v)

        if not projected_points:
            return visual

        sampled_points = projected_points
        if len(sampled_points) > max_points:
            stride = int(math.ceil(float(len(sampled_points)) / float(max_points)))
            sampled_points = sampled_points[::stride]

        bbox_xyxy_full = [
            int(max(0, round(min_u))),
            int(max(0, round(min_v))),
            int(min(image_width - 1, round(max_u))),
            int(min(image_height - 1, round(max_v))),
        ]
        visual.update({
            "projected_available": True,
            "projected_point_count": len(projected_points),
            "sampled_point_count": len(sampled_points),
            "projected_points_xy_full": sampled_points,
            "bbox_xyxy_full": bbox_xyxy_full,
            "point_radius_px": int(max(1, self.point_cloud_overlay_point_radius_px)),
            "transform_mode": transform.get("mode"),
            "camera_frame_id": camera_projection.get("frame_id"),
            "avg_depth_m": round(
                sum(projected_depths) / float(len(projected_depths)),
                3,
            ),
        })
        return visual

    def _prepare_image(self, bundle, image_bgr):
        prepared = image_bgr
        hint = bundle.get("visual_grounding_hint", {})
        crop_x0 = 0
        crop_y0 = 0
        crop_w = prepared.shape[1]
        crop_h = prepared.shape[0]
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
                crop_x0 = x0
                crop_y0 = y0
                crop_w = x1 - x0
                crop_h = y1 - y0
                prepared = prepared[y0:y1, x0:x1]

        height, width = prepared.shape[:2]
        longest = max(height, width)
        scale_x = 1.0
        scale_y = 1.0
        if self.max_image_side_px > 0 and longest > self.max_image_side_px:
            scale = float(self.max_image_side_px) / float(longest)
            resized_w = max(1, int(round(width * scale)))
            resized_h = max(1, int(round(height * scale)))
            scale_x = float(crop_w) / float(resized_w)
            scale_y = float(crop_h) / float(resized_h)
            prepared = cv2.resize(
                prepared,
                (resized_w, resized_h),
                interpolation=cv2.INTER_AREA,
            )
        else:
            scale_x = float(crop_w) / float(width)
            scale_y = float(crop_h) / float(height)
        return prepared, {
            "crop_x0": int(crop_x0),
            "crop_y0": int(crop_y0),
            "crop_w": int(crop_w),
            "crop_h": int(crop_h),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "prepared_w": int(prepared.shape[1]),
            "prepared_h": int(prepared.shape[0]),
            "full_w": int(image_bgr.shape[1]),
            "full_h": int(image_bgr.shape[0]),
            "focus_rect_xyxy_full": list(
                compute_focus_rect(
                    int(image_bgr.shape[1]),
                    int(image_bgr.shape[0]),
                    hint,
                )
            ),
        }

    def _detect_people(self, image_bgr):
        if self.hog is None:
            return []

        rects, weights = self.hog.detectMultiScale(
            image_bgr,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        detections = []
        for (x, y, w, h), weight in zip(rects, weights):
            confidence = float(weight)
            if confidence < self.hog_confidence_threshold:
                continue
            detections.append(
                {
                    "label": "person",
                    "confidence": confidence,
                    "bbox_xywh": [int(x), int(y), int(w), int(h)],
                }
            )
        return detections

    def _detect_obstacle_candidates(self, image_bgr):
        height, width = image_bgr.shape[:2]
        if height <= 0 or width <= 0:
            return []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(height * width)
        detections = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = area / image_area
            width_ratio = float(w) / float(width)
            height_ratio = float(h) / float(height)

            if area_ratio < self.obstacle_min_area_ratio:
                continue
            if width_ratio < self.obstacle_min_width_ratio:
                continue
            if height_ratio < self.obstacle_min_height_ratio:
                continue
            if width_ratio > 0.95 and height_ratio > 0.95:
                continue

            touches_left = x <= 2
            touches_right = (x + w) >= (width - 2)
            touches_top = y <= 2
            touches_bottom = (y + h) >= (height - 2)

            if (touches_left and touches_right) or (touches_top and touches_bottom):
                continue
            if touches_left and width_ratio > 0.60:
                continue
            if touches_right and width_ratio > 0.60:
                continue

            confidence = min(1.0, area_ratio * 8.0 + height_ratio * 0.8 + width_ratio * 0.4)
            detections.append(
                {
                    "label": "obstacle_candidate",
                    "confidence": round(float(confidence), 3),
                    "bbox_xywh": [int(x), int(y), int(w), int(h)],
                }
            )

        detections.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
        return detections[:5]

    def _detect_with_yolo_worker(self, image_bgr):
        if self.yolo_worker is None:
            raise RuntimeError("yolo worker is not available")
        payload = self.yolo_worker.infer(image_bgr, self.jpeg_quality)
        if payload.get("tracking_active") is False and payload.get("tracking_fallback_reason"):
            rospy.logwarn_throttle(
                5.0,
                "yolo worker tracking fallback to predict-only: %s",
                str(payload.get("tracking_fallback_reason")),
            )
        detections = payload.get("detections") or []
        normalized = []
        for item in detections:
            bbox = item.get("bbox_xyxy") or []
            if len(bbox) != 4:
                continue
            normalized.append(
                {
                    "label": str(item.get("label") or "object"),
                    "confidence": float(item.get("confidence") or 0.0),
                    "bbox_xyxy": [int(round(value)) for value in bbox],
                    "track_id": item.get("track_id"),
                }
            )
        return normalized

    def _current_motion_state(self, timestamp_now):
        if not self.use_motion_awareness:
            return {
                "motion_score": 0.0,
                "is_moving": False,
                "linear_speed_mps": 0.0,
                "angular_speed_radps": 0.0,
            }

        cmd_age = timestamp_now - float(self.latest_cmd_vel.get("arrival_wall_time") or 0.0)
        odom_age = timestamp_now - float(self.latest_odom.get("arrival_wall_time") or 0.0)

        cmd_linear = 0.0
        cmd_angular = 0.0
        if cmd_age <= 1.0:
            cmd_linear = abs(float(self.latest_cmd_vel.get("linear_x_mps") or 0.0))
            cmd_angular = abs(float(self.latest_cmd_vel.get("angular_z_radps") or 0.0))

        odom_linear = 0.0
        odom_angular = 0.0
        if odom_age <= 1.0:
            odom_linear = abs(float(self.latest_odom.get("linear_speed_mps") or 0.0))
            odom_angular = abs(float(self.latest_odom.get("angular_speed_radps") or 0.0))

        linear_speed = max(cmd_linear, odom_linear)
        angular_speed = max(cmd_angular, odom_angular)
        linear_score = clamp(
            linear_speed / max(1e-3, self.motion_linear_speed_ref_mps), 0.0, 1.0
        )
        angular_score = clamp(
            angular_speed / max(1e-3, self.motion_angular_speed_ref_radps), 0.0, 1.0
        )
        motion_score = clamp(max(linear_score, angular_score), 0.0, 1.0)
        return {
            "motion_score": motion_score,
            "is_moving": motion_score > 0.10,
            "linear_speed_mps": linear_speed,
            "angular_speed_radps": angular_speed,
        }

    def _current_odom_pose(self, timestamp_now):
        odom_age = timestamp_now - float(self.latest_odom.get("arrival_wall_time") or 0.0)
        if odom_age > 1.0:
            return None
        return {
            "x": float(self.latest_odom.get("position_x") or 0.0),
            "y": float(self.latest_odom.get("position_y") or 0.0),
            "yaw_rad": float(self.latest_odom.get("yaw_rad") or 0.0),
            "frame_id": str(self.latest_odom.get("frame_id") or ""),
        }

    def _extract_lidar_anchor_world(self, bundle, timestamp_now):
        if not self.semantic_memory_enabled:
            return None
        hint = bundle.get("visual_grounding_hint", {}) or {}
        if not hint.get("available"):
            return None
        anchor = hint.get("anchor_xyz") or {}
        if not isinstance(anchor, dict):
            return None
        anchor_x = anchor.get("x")
        anchor_y = anchor.get("y")
        anchor_z = anchor.get("z")
        if anchor_x is None or anchor_y is None:
            return None
        odom_pose = self._current_odom_pose(timestamp_now)
        if odom_pose is None:
            return None
        yaw = float(odom_pose.get("yaw_rad") or 0.0)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        world_x = float(odom_pose["x"]) + cos_yaw * float(anchor_x) - sin_yaw * float(anchor_y)
        world_y = float(odom_pose["y"]) + sin_yaw * float(anchor_x) + cos_yaw * float(anchor_y)
        return {
            "frame_id": odom_pose.get("frame_id") or "odom",
            "world_x": world_x,
            "world_y": world_y,
            "world_z": float(anchor_z or 0.0),
            "distance_hint_m": hint.get("distance_hint_m"),
            "source_kind": hint.get("source_evidence_kind"),
            "matched_visual_region_ko": hint.get("matched_visual_region_ko"),
        }

    def _map_detection_to_full_image(self, detection, prepared_meta):
        bbox = detection.get("bbox_xyxy") or []
        if len(bbox) != 4:
            return detection
        x0, y0, x1, y1 = bbox
        full_x0 = int(round(prepared_meta["crop_x0"] + x0 * prepared_meta["scale_x"]))
        full_y0 = int(round(prepared_meta["crop_y0"] + y0 * prepared_meta["scale_y"]))
        full_x1 = int(round(prepared_meta["crop_x0"] + x1 * prepared_meta["scale_x"]))
        full_y1 = int(round(prepared_meta["crop_y0"] + y1 * prepared_meta["scale_y"]))
        full_x0 = max(0, min(prepared_meta["full_w"] - 1, full_x0))
        full_y0 = max(0, min(prepared_meta["full_h"] - 1, full_y0))
        full_x1 = max(0, min(prepared_meta["full_w"] - 1, full_x1))
        full_y1 = max(0, min(prepared_meta["full_h"] - 1, full_y1))
        mapped = dict(detection)
        mapped["bbox_xyxy_full"] = [full_x0, full_y0, full_x1, full_y1]
        mapped["label_ko"] = english_label_to_korean(mapped.get("label"))
        return mapped

    def _semantic_memory_unit(self, label_ko):
        return "명" if str(label_ko or "") == "사람" else "개"

    def _make_semantic_memory_id(self):
        memory_id = "memory-{}".format(self.next_semantic_memory_id)
        self.next_semantic_memory_id += 1
        return memory_id

    def _prune_semantic_memories(self, timestamp_now):
        for memory_id, memory in list(self.semantic_memories.items()):
            stable_label_ko = memory.get("stable_label_ko") or ""
            ttl_s = self.semantic_memory_person_ttl_s if stable_label_ko == "사람" else self.semantic_memory_ttl_s
            age = timestamp_now - float(memory.get("last_seen") or timestamp_now)
            if age > ttl_s:
                del self.semantic_memories[memory_id]

    def _decay_semantic_label_scores(self, label_scores):
        decay = clamp(self.semantic_memory_label_decay, 0.0, 0.999)
        for key in list(label_scores.keys()):
            label_scores[key] *= decay
            if label_scores[key] < 0.01:
                del label_scores[key]

    def _find_semantic_memory(self, anchor_world):
        if not anchor_world:
            return None
        best_memory = None
        best_distance = None
        anchor_x = float(anchor_world.get("world_x") or 0.0)
        anchor_y = float(anchor_world.get("world_y") or 0.0)
        for memory in self.semantic_memories.values():
            dx = anchor_x - float(memory.get("world_x") or 0.0)
            dy = anchor_y - float(memory.get("world_y") or 0.0)
            distance = math.hypot(dx, dy)
            if distance > self.semantic_memory_match_radius_m:
                continue
            if best_memory is None or distance < best_distance:
                best_memory = memory
                best_distance = distance
        return best_memory

    def _update_semantic_memory(
        self,
        anchor_world,
        selected_detection,
        timestamp_now,
    ):
        if not self.semantic_memory_enabled or anchor_world is None:
            return None
        self._prune_semantic_memories(timestamp_now)
        memory = self._find_semantic_memory(anchor_world)
        if memory is None:
            if selected_detection is None:
                return None
            memory = {
                "memory_id": self._make_semantic_memory_id(),
                "world_x": float(anchor_world.get("world_x") or 0.0),
                "world_y": float(anchor_world.get("world_y") or 0.0),
                "world_z": float(anchor_world.get("world_z") or 0.0),
                "frame_id": str(anchor_world.get("frame_id") or "odom"),
                "label_scores": {},
                "hits": 0,
                "last_seen": timestamp_now,
                "last_bbox_xyxy_full": selected_detection.get("bbox_xyxy_full"),
                "last_distance_hint_m": anchor_world.get("distance_hint_m"),
            }
            self.semantic_memories[memory["memory_id"]] = memory
        else:
            alpha = clamp(self.semantic_memory_position_alpha, 0.05, 0.95)
            memory["world_x"] = (1.0 - alpha) * float(memory.get("world_x") or 0.0) + alpha * float(anchor_world.get("world_x") or 0.0)
            memory["world_y"] = (1.0 - alpha) * float(memory.get("world_y") or 0.0) + alpha * float(anchor_world.get("world_y") or 0.0)
            memory["world_z"] = (1.0 - alpha) * float(memory.get("world_z") or 0.0) + alpha * float(anchor_world.get("world_z") or 0.0)
            memory["last_distance_hint_m"] = anchor_world.get("distance_hint_m")

        self._decay_semantic_label_scores(memory["label_scores"])
        if selected_detection is not None:
            label = str(selected_detection.get("label") or "object")
            label_ko = selected_detection.get("label_ko") or english_label_to_korean(label)
            confidence = max(0.2, float(selected_detection.get("confidence") or 0.0))
            memory["label_scores"][label] = memory["label_scores"].get(label, 0.0) + confidence
            memory["last_bbox_xyxy_full"] = selected_detection.get("bbox_xyxy_full")
            memory["last_raw_label"] = label
            memory["last_raw_label_ko"] = label_ko
            memory["hits"] = int(memory.get("hits", 0)) + 1
        stable_label = self._stable_label_from_scores(
            memory.get("label_scores") or {},
            memory.get("last_raw_label") or "object",
        )
        memory["stable_label"] = stable_label
        memory["stable_label_ko"] = english_label_to_korean(stable_label)
        memory["last_seen"] = timestamp_now
        return memory

    def _apply_semantic_memory(
        self,
        detections,
        selected_detection,
        semantic_memory,
    ):
        if semantic_memory is None:
            return None
        stable_label = semantic_memory.get("stable_label")
        stable_label_ko = semantic_memory.get("stable_label_ko")
        if not stable_label or int(semantic_memory.get("hits", 0)) < self.semantic_memory_min_hits:
            return None

        if selected_detection is not None:
            selected_detection["memory_id"] = semantic_memory.get("memory_id")
            detection_confidence = float(selected_detection.get("confidence") or 0.0)
            memory_score = float(
                (semantic_memory.get("label_scores") or {}).get(stable_label) or 0.0
            )
            selected_label = str(selected_detection.get("label") or "")
            if (
                selected_label != stable_label
                and memory_score >= (detection_confidence + self.semantic_memory_override_margin)
            ):
                selected_detection["label"] = stable_label
                selected_detection["label_ko"] = stable_label_ko
                selected_detection["semantic_source"] = "lidar_memory"
            elif selected_label == stable_label:
                selected_detection["semantic_source"] = "camera_and_lidar_memory"
            return None

        return {
            "label": stable_label,
            "label_ko": stable_label_ko,
            "hits": int(semantic_memory.get("hits", 0)),
            "distance_hint_m": semantic_memory.get("last_distance_hint_m"),
            "memory_id": semantic_memory.get("memory_id"),
        }

    def _make_local_track_key(self):
        key = "local-{}".format(self.next_local_track_id)
        self.next_local_track_id += 1
        return key

    def _find_track_by_iou(self, bbox, used_keys, motion_state=None):
        motion_state = motion_state or {}
        motion_score = float(motion_state.get("motion_score") or 0.0)
        center_match_px = self.track_center_match_px + (
            motion_score * self.motion_extra_center_match_px
        )
        best_key = None
        best_score = None
        for track_key, state in self.track_states.items():
            if track_key in used_keys:
                continue
            track_bbox = state.get("bbox_xyxy_full")
            iou = bbox_iou_xyxy(bbox, track_bbox)
            if iou >= self.track_iou_match_threshold:
                score = iou
            else:
                distance = bbox_center_distance(bbox, track_bbox)
                if distance > center_match_px:
                    continue
                score = 0.10 * (1.0 - (distance / max(1.0, center_match_px)))
            if track_key == self.selected_track_key and motion_score > 0.0:
                score += motion_score * self.motion_selected_track_bonus
            if best_key is None or score > best_score:
                best_key = track_key
                best_score = score
        return best_key

    def _decay_track_labels(self, label_scores, motion_state=None, track_key=None):
        motion_state = motion_state or {}
        motion_score = float(motion_state.get("motion_score") or 0.0)
        decay = self.track_label_decay + (motion_score * self.motion_label_decay_bonus)
        if track_key is not None and track_key == self.selected_track_key:
            decay += 0.5 * motion_score * self.motion_label_decay_bonus
        decay = clamp(decay, 0.0, 0.995)
        for key in list(label_scores.keys()):
            label_scores[key] *= decay
            if label_scores[key] < 0.01:
                del label_scores[key]

    def _stable_label_from_scores(self, label_scores, fallback_label):
        if not label_scores:
            return fallback_label
        return max(label_scores.items(), key=lambda item: item[1])[0]

    def _update_single_track(self, track_key, detection, timestamp_now, motion_state=None):
        motion_state = motion_state or {}
        motion_score = float(motion_state.get("motion_score") or 0.0)
        bbox = detection.get("bbox_xyxy_full") or detection.get("bbox_xyxy")
        label = str(detection.get("label") or "object")
        confidence = float(detection.get("confidence") or 0.0)
        state = self.track_states.get(track_key)
        if state is None:
            state = {
                "track_key": track_key,
                "bbox_xyxy_full": list(bbox),
                "label_scores": {label: max(0.1, confidence)},
                "last_seen": timestamp_now,
                "hits": 1,
                "misses": 0,
                "confidence_ema": confidence,
                "last_raw_label": label,
            }
            self.track_states[track_key] = state
        else:
            previous_stable_label = state.get("stable_label") or state.get("last_raw_label")
            self._decay_track_labels(
                state["label_scores"],
                motion_state=motion_state,
                track_key=track_key,
            )
            if (
                motion_score > 0.0
                and previous_stable_label
                and int(state.get("hits", 0)) >= self.track_min_hits
            ):
                continuity_bonus = 0.20 * motion_score
                if track_key == self.selected_track_key:
                    continuity_bonus += motion_score * self.motion_selected_track_bonus
                state["label_scores"][previous_stable_label] = (
                    state["label_scores"].get(previous_stable_label, 0.0)
                    + continuity_bonus
                )
            state["label_scores"][label] = state["label_scores"].get(label, 0.0) + max(
                0.1, confidence
            )
            old_bbox = state.get("bbox_xyxy_full") or list(bbox)
            alpha = clamp(
                self.track_bbox_alpha + (0.20 * motion_score),
                0.15,
                0.80,
            )
            smoothed_bbox = []
            for old_value, new_value in zip(old_bbox, bbox):
                smoothed_bbox.append(int(round((1.0 - alpha) * old_value + alpha * new_value)))
            state["bbox_xyxy_full"] = smoothed_bbox
            state["confidence_ema"] = (1.0 - alpha) * float(
                state.get("confidence_ema", confidence)
            ) + alpha * confidence
            state["last_seen"] = timestamp_now
            state["hits"] = int(state.get("hits", 0)) + 1
            state["misses"] = 0
            state["last_raw_label"] = label

        stable_label = self._stable_label_from_scores(state["label_scores"], label)
        state["stable_label"] = stable_label
        state["stable_label_ko"] = english_label_to_korean(stable_label)
        return {
            "label": stable_label,
            "label_ko": state["stable_label_ko"],
            "confidence": round(float(state.get("confidence_ema", confidence)), 4),
            "bbox_xyxy_full": list(state["bbox_xyxy_full"]),
            "track_key": track_key,
            "hits": int(state.get("hits", 1)),
            "is_tracked_memory": False,
        }

    def _stabilize_detections(self, detections, timestamp_now, motion_state=None):
        motion_state = motion_state or {}
        motion_score = float(motion_state.get("motion_score") or 0.0)
        used_keys = set()
        stabilized = []
        for detection in detections:
            track_id = detection.get("track_id")
            track_key = None
            if track_id is not None:
                track_key = "yolo-{}".format(int(track_id))
                if track_key in used_keys:
                    track_key = None
            if track_key is None:
                bbox = detection.get("bbox_xyxy_full") or detection.get("bbox_xyxy")
                track_key = self._find_track_by_iou(
                    bbox,
                    used_keys,
                    motion_state=motion_state,
                )
            if track_key is None:
                track_key = self._make_local_track_key()
            stabilized_detection = self._update_single_track(
                track_key,
                detection,
                timestamp_now,
                motion_state=motion_state,
            )
            used_keys.add(track_key)
            stabilized.append(stabilized_detection)

        hold_ttl_s = self.track_hold_ttl_s + (
            motion_score * self.motion_extra_hold_ttl_s
        )
        for track_key, state in list(self.track_states.items()):
            if track_key in used_keys:
                continue
            state["misses"] = int(state.get("misses", 0)) + 1
            age = timestamp_now - float(state.get("last_seen") or timestamp_now)
            if age > hold_ttl_s:
                del self.track_states[track_key]
                continue
            if int(state.get("hits", 0)) < self.track_min_hits:
                continue
            stabilized.append(
                {
                    "label": state.get("stable_label") or state.get("last_raw_label") or "object",
                    "label_ko": state.get("stable_label_ko")
                    or english_label_to_korean(state.get("stable_label") or state.get("last_raw_label")),
                    "confidence": round(float(state.get("confidence_ema", 0.0)), 4),
                    "bbox_xyxy_full": list(state.get("bbox_xyxy_full") or []),
                    "track_key": track_key,
                    "hits": int(state.get("hits", 0)),
                    "is_tracked_memory": True,
                }
            )

        def sort_key(item):
            confidence = float(item.get("confidence") or 0.0)
            hits = min(0.30, 0.05 * float(item.get("hits") or 0.0))
            selected_bonus = 0.0
            if item.get("track_key") == self.selected_track_key:
                selected_bonus = motion_score * self.motion_selected_track_bonus
            memory_penalty = -0.03 if item.get("is_tracked_memory") else 0.0
            return confidence + hits + selected_bonus + memory_penalty

        stabilized.sort(key=sort_key, reverse=True)
        return stabilized

    def _select_detection(self, detections, prepared_meta, motion_state=None, pointcloud_visual=None):
        if not detections:
            return None
        motion_state = motion_state or {}
        motion_score = float(motion_state.get("motion_score") or 0.0)
        focus_rect = prepared_meta.get("focus_rect_xyxy_full") or [
            0,
            0,
            prepared_meta.get("full_w", 1) - 1,
            prepared_meta.get("full_h", 1) - 1,
        ]
        selected_track_bbox = None
        if self.selected_track_key and self.selected_track_key in self.track_states:
            selected_track_bbox = self.track_states[self.selected_track_key].get(
                "bbox_xyxy_full"
            )
        pointcloud_bbox = None
        if isinstance(pointcloud_visual, dict) and pointcloud_visual.get("available"):
            bbox = pointcloud_visual.get("bbox_xyxy_full") or []
            if len(bbox) == 4:
                pointcloud_bbox = bbox
        best = None
        best_score = None
        for item in detections:
            bbox = item.get("bbox_xyxy_full") or item.get("bbox_xyxy")
            overlap = bbox_iou_xyxy(bbox, focus_rect)
            confidence = float(item.get("confidence") or 0.0)
            hits_bonus = min(0.30, 0.05 * float(item.get("hits") or 0.0))
            memory_penalty = -0.05 if item.get("is_tracked_memory") else 0.0
            continuity_bonus = 0.0
            if item.get("track_key") == self.selected_track_key:
                continuity_bonus += motion_score * self.motion_selected_track_bonus
            elif selected_track_bbox is not None and motion_score > 0.0:
                distance = bbox_center_distance(bbox, selected_track_bbox)
                if distance <= self.motion_selected_track_distance_px:
                    continuity_bonus += (
                        0.5
                        * motion_score
                        * self.motion_selected_track_bonus
                        * (1.0 - (distance / max(1.0, self.motion_selected_track_distance_px)))
                    )
            pointcloud_bonus = 0.0
            if pointcloud_bbox is not None:
                point_iou = bbox_iou_xyxy(bbox, pointcloud_bbox)
                point_distance = bbox_center_distance(bbox, pointcloud_bbox)
                if point_iou >= self.point_cloud_association_iou_threshold:
                    pointcloud_bonus += 1.5 * point_iou
                elif point_distance <= self.point_cloud_association_center_match_px:
                    pointcloud_bonus += 0.25 * (
                        1.0
                        - (
                            point_distance
                            / max(1.0, self.point_cloud_association_center_match_px)
                        )
                    )
            score = (
                overlap * 2.0
                + confidence
                + hits_bonus
                + memory_penalty
                + continuity_bonus
                + pointcloud_bonus
            )
            if best is None or score > best_score:
                best = item
                best_score = score
        return best

    def _summarize_detected_objects(self, detections):
        if not detections:
            return []
        counts = {}
        ordered = []
        for item in detections:
            key = item.get("label_ko") or english_label_to_korean(item.get("label"))
            if key not in counts:
                ordered.append(key)
            counts[key] = counts.get(key, 0) + 1
        summary = []
        for key in ordered:
            count = counts[key]
            unit = "명" if key == "사람" else "개"
            summary.append("{} {}{}".format(key, count, unit))
        return summary

    def _scene_description(self, bundle, detections, semantic_memory_fallback=None):
        region = normalize_region_phrase(
            nested_get(
                bundle,
                "visual_grounding_hint",
                "matched_visual_region_ko",
                default="전방 통로",
            )
        )
        event_label = bundle.get("event_label")
        object_summaries = self._summarize_detected_objects(detections)
        if object_summaries:
            joined = ", ".join(object_summaries)
            return "{} 영역에서 {}{} 보인다.".format(
                region,
                joined,
                choose_subject_particle(joined),
            )
        if semantic_memory_fallback and semantic_memory_fallback.get("label_ko"):
            label_ko = semantic_memory_fallback.get("label_ko")
            return "{} 영역에서 이전에 관찰된 {} 1{}와 같은 위치의 장애물이 다시 감지된다.".format(
                region,
                label_ko,
                self._semantic_memory_unit(label_ko),
            )
        if event_label == "path_blocked":
            return "{} 영역에서 카메라로 뚜렷한 형태를 분리하진 못했지만 장애물 후보가 있는 것으로 보인다.".format(
                region
            )
        return "{} 영역에서 뚜렷한 장애물 후보는 없다.".format(region)

    def _detected_objects(self, detections, event_label, semantic_memory_fallback=None):
        object_summaries = self._summarize_detected_objects(detections)
        if object_summaries:
            return object_summaries
        if semantic_memory_fallback and semantic_memory_fallback.get("label_ko"):
            label_ko = semantic_memory_fallback.get("label_ko")
            return ["{} 1{}".format(label_ko, self._semantic_memory_unit(label_ko))]
        if event_label == "path_blocked":
            return ["미확인 장애물"]
        return []

    def _final_explanation(self, bundle, detections, selected_detection, semantic_memory_fallback=None):
        region = normalize_region_phrase(
            nested_get(
                bundle,
                "visual_grounding_hint",
                "matched_visual_region_ko",
                default="전방 통로",
            )
        )
        distance_hint_m = nested_get(
            bundle, "visual_grounding_hint", "distance_hint_m", default=None
        )
        distance_phrase = format_distance(distance_hint_m)
        event_label = bundle.get("event_label")
        selected_label = None
        if selected_detection:
            selected_label = selected_detection.get("label_ko") or english_label_to_korean(
                selected_detection.get("label")
            )
        elif semantic_memory_fallback and semantic_memory_fallback.get("label_ko"):
            selected_label = semantic_memory_fallback.get("label_ko")
        object_summaries = self._summarize_detected_objects(detections)
        object_phrase = ", ".join(object_summaries)
        if not object_phrase and semantic_memory_fallback and semantic_memory_fallback.get("label_ko"):
            label_ko = semantic_memory_fallback.get("label_ko")
            object_phrase = "{} 1{}".format(label_ko, self._semantic_memory_unit(label_ko))

        if event_label == "path_blocked":
            if selected_label:
                return "{} {}부근에 {}{} 보여 경로가 막힌 상태로 판단된다.".format(
                    region, distance_phrase, selected_label, choose_subject_particle(selected_label)
                ).replace("  ", " ")
            if object_phrase:
                return "{} {}부근에 {}{} 보여 경로가 막힌 상태로 판단된다.".format(
                    region, distance_phrase, object_phrase, choose_subject_particle(object_phrase)
                ).replace("  ", " ")
            return "{} {}부근에서 planner가 장애물 근거를 보고 있어 경로가 막힌 상태로 판단된다.".format(
                region, distance_phrase
            ).replace("  ", " ")

        if event_label == "path_update":
            if selected_label:
                return "{} {}부근에 {}{} 보여 planner가 경로를 보수적으로 조정 중이다.".format(
                    region, distance_phrase, selected_label, choose_subject_particle(selected_label)
                ).replace("  ", " ")
            if object_phrase:
                return "{} {}부근에 {}{} 보여 planner가 경로를 보수적으로 조정 중이다.".format(
                    region, distance_phrase, object_phrase, choose_subject_particle(object_phrase)
                ).replace("  ", " ")
            return "{} {}부근의 뚜렷한 장애물 후보는 없지만 planner가 경로를 조정 중이다.".format(
                region, distance_phrase
            ).replace("  ", " ")

        if selected_label:
            return "{} {}부근에 {}{} 보인다.".format(
                region, distance_phrase, selected_label, choose_subject_particle(selected_label)
            ).replace("  ", " ")
        if object_phrase:
            return "{} {}부근에 {}{} 보인다.".format(
                region, distance_phrase, object_phrase, choose_subject_particle(object_phrase)
            ).replace("  ", " ")
        return "전방 통로에서 뚜렷한 장애물 후보는 없으며 현재 경로를 따라갈 수 있는 상태로 보인다."

    def _build_payload(
        self,
        bundle,
        image_entry,
        detections,
        status,
        status_detail,
        prepared_meta=None,
        motion_state=None,
        semantic_memory=None,
        semantic_memory_fallback=None,
        pointcloud_visual=None,
    ):
        prepared_meta = prepared_meta or {}
        motion_state = motion_state or {}
        selected_detection = self._select_detection(
            detections,
            prepared_meta,
            motion_state=motion_state,
            pointcloud_visual=pointcloud_visual,
        )
        scene_description = self._scene_description(
            bundle,
            detections,
            semantic_memory_fallback=semantic_memory_fallback,
        )
        detected_objects = self._detected_objects(
            detections,
            bundle.get("event_label"),
            semantic_memory_fallback=semantic_memory_fallback,
        )
        final_explanation = self._final_explanation(
            bundle,
            detections,
            selected_detection,
            semantic_memory_fallback=semantic_memory_fallback,
        )
        return {
            "schema": "xai_driving_explainer/DrivingSceneExplanation@1",
            "stamp": bundle.get("stamp"),
            "image_stamp": image_entry.get("stamp") if image_entry else None,
            "event_label": bundle.get("event_label"),
            "event_type": bundle.get("event_type"),
            "status": status,
            "status_detail_ko": status_detail,
            "backend": self.backend,
            "matched_visual_region_ko": nested_get(
                bundle,
                "visual_grounding_hint",
                "matched_visual_region_ko",
                default="전방 통로",
            ),
            "grounding_confidence": nested_get(
                bundle,
                "visual_grounding_hint",
                "grounding_confidence",
                default="low",
            ),
            "scene_description_ko": scene_description
            if status == "ok"
            else status_detail,
            "detected_objects_ko": detected_objects if status == "ok" else [],
            "planner_reason_ko": bundle.get("planner_reason_ko"),
            "grounded_action_explanation_ko": final_explanation if status == "ok" else "",
            "visual_summary_ko": scene_description if status == "ok" else status_detail,
            "final_combined_explanation_ko": final_explanation
            if status == "ok"
            else status_detail,
            "unknowns_ko": []
            if status == "ok"
            else ["카메라 프레임과 시점을 맞추지 못했거나 detector 실행에 실패했다."],
            "faithfulness_notes_ko": "planner evidence와 lightweight camera detector를 결합한 실시간 설명",
            "confidence": "medium" if detections else "low",
            "detector_details": {
                "obstacle_count": len(detections),
                "detections": detections,
                "selected_detection": selected_detection,
                "motion_state": motion_state,
                "semantic_memory": semantic_memory,
                "semantic_memory_fallback": semantic_memory_fallback,
                "pointcloud_visual": pointcloud_visual,
            },
        }

    def _publish_payload(self, payload):
        outgoing = String()
        outgoing.data = json.dumps(payload, ensure_ascii=False)
        self.publisher.publish(outgoing)
        if self.log_detector_explanation:
            rospy.loginfo(
                "[XAI-DETECTOR] backend=%s | event=%s | status=%s | scene=%s | objects=%s | final=%s",
                payload.get("backend"),
                payload.get("event_label"),
                payload.get("status"),
                payload.get("scene_description_ko"),
                payload.get("detected_objects_ko"),
                payload.get("final_combined_explanation_ko"),
            )

    def _on_explanation(self, message):
        bundle = self._parse_json(message)
        if not bundle:
            return

        event_label = str(bundle.get("event_label") or "").strip()
        if self.allowed_event_labels and event_label not in self.allowed_event_labels:
            return

        now = time.time()
        signature = (
            bundle.get("event_label"),
            bundle.get("planner_reason_ko"),
            json.dumps(
                bundle.get("visual_grounding_hint", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        if signature == self.last_signature and (
            now - self.last_process_time
        ) < self.min_process_interval_s:
            return

        image_entry = self._nearest_image(bundle.get("stamp"), now)
        motion_state = self._current_motion_state(now)
        if image_entry is None:
            payload = self._build_payload(
                bundle,
                None,
                [],
                "no_recent_image",
                "이벤트 시점과 가까운 카메라 프레임을 찾지 못해 detector 실행을 건너뛰었다.",
                motion_state=motion_state,
            )
            self._publish_payload(payload)
            return

        prepared, prepared_meta = self._prepare_image(bundle, image_entry["image"])
        try:
            if self.backend == "hog_person":
                raw_detections = self._detect_people(prepared)
            elif self.backend == "yolo_worker":
                raw_detections = self._detect_with_yolo_worker(prepared)
            else:
                raw_detections = self._detect_obstacle_candidates(prepared)
        except Exception as exc:
            payload = self._build_payload(
                bundle,
                image_entry,
                [],
                "detector_error",
                "camera detector 실행에 실패했다: {}".format(str(exc)),
                prepared_meta=prepared_meta,
                motion_state=motion_state,
                pointcloud_visual=None,
            )
            self._publish_payload(payload)
            return

        detections = [
            self._map_detection_to_full_image(item, prepared_meta)
            for item in raw_detections
        ]
        detections = self._stabilize_detections(
            detections,
            now,
            motion_state=motion_state,
        )
        pointcloud_visual = self._project_point_cloud_visual(
            bundle,
            image_entry,
        )
        lidar_anchor_world = self._extract_lidar_anchor_world(bundle, now)
        provisional_selected = self._select_detection(
            detections,
            prepared_meta,
            motion_state=motion_state,
            pointcloud_visual=pointcloud_visual,
        )
        semantic_memory = self._update_semantic_memory(
            lidar_anchor_world,
            provisional_selected,
            now,
        )
        semantic_memory_fallback = self._apply_semantic_memory(
            detections,
            provisional_selected,
            semantic_memory,
        )
        status_detail = (
            "YOLO visual detection generated successfully."
            if self.backend == "yolo_worker"
            else "Lightweight camera detector explanation generated successfully."
        )
        payload = self._build_payload(
            bundle,
            image_entry,
            detections,
            "ok",
            status_detail,
            prepared_meta=prepared_meta,
            motion_state=motion_state,
            semantic_memory=semantic_memory,
            semantic_memory_fallback=semantic_memory_fallback,
            pointcloud_visual=pointcloud_visual,
        )
        self._publish_payload(payload)
        selected_detection = nested_get(
            payload,
            "detector_details",
            "selected_detection",
            default=None,
        )
        if isinstance(selected_detection, dict) and selected_detection.get("track_key"):
            self.selected_track_key = str(selected_detection.get("track_key"))
        elif not motion_state.get("is_moving"):
            self.selected_track_key = None
        self.last_signature = signature
        self.last_process_time = now


def main():
    rospy.init_node("driving_scene_detector")
    DrivingSceneDetectorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
