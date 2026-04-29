#!/usr/bin/python3
import argparse
import ast
import math
import os
import sys

import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as point_cloud2

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError as exc:
    raise SystemExit(
        "필수 모듈을 불러오지 못했습니다: {}.\n"
        "이 스크립트는 ROS OpenCV가 설치된 시스템 Python으로 실행해야 합니다.\n"
        "다음처럼 다시 실행해 주세요:\n"
        "  /usr/bin/python3 src/xai_driving_explainer/scripts/tune_camera_lidar_extrinsic.py ...\n"
        "현재 interpreter: {}".format(exc, sys.executable)
    )


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def quaternion_normalize(quaternion_xyzw):
    x, y, z, w = quaternion_xyzw
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def quaternion_from_euler(roll_rad, pitch_rad, yaw_rad):
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return quaternion_normalize((qx, qy, qz, qw))


def euler_from_quaternion(quaternion_xyzw):
    x, y, z, w = quaternion_normalize(quaternion_xyzw)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def rotation_matrix_from_quaternion(quaternion_xyzw):
    x, y, z, w = quaternion_normalize(quaternion_xyzw)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotate_point_by_quaternion(point_xyz, quaternion_xyzw):
    px, py, pz = point_xyz
    qx, qy, qz, qw = quaternion_xyzw
    ix = qw * px + qy * pz - qz * py
    iy = qw * py + qz * px - qx * pz
    iz = qw * pz + qx * py - qy * px
    iw = -qx * px - qy * py - qz * pz
    rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
    ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
    rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
    return (rx, ry, rz)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively tune LiDAR-to-camera extrinsic using a ROS bag."
    )
    parser.add_argument("--bag", required=True)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument(
        "--camera-info-bag",
        default="",
        help="camera_info가 없는 경우 intrinsic을 읽어올 다른 bag 경로",
    )
    parser.add_argument(
        "--point-cloud-topic",
        default="/planning/linefit_ground/non_ground_cloud",
    )
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--image-stride", type=int, default=5)
    parser.add_argument(
        "--sample-selection",
        choices=("charuco_priority", "sequential"),
        default="charuco_priority",
        help="튜너에서 샘플을 고르는 방식",
    )
    parser.add_argument(
        "--sample-start-index",
        type=int,
        default=0,
        help="sequential 모드에서 시작할 샘플 인덱스",
    )
    parser.add_argument("--max-sync-dt-s", type=float, default=0.12)
    parser.add_argument("--max-cloud-points", type=int, default=18000)
    parser.add_argument("--forward-m", type=float, default=12.0)
    parser.add_argument("--rear-m", type=float, default=2.0)
    parser.add_argument("--half-width-m", type=float, default=8.0)
    parser.add_argument("--height-abs-m", type=float, default=2.5)
    parser.add_argument("--max-range-m", type=float, default=12.0)
    parser.add_argument("--point-radius-px", type=int, default=3)
    parser.add_argument("--overview-width-px", type=int, default=560)
    parser.add_argument(
        "--camera-frustum-margin-deg",
        type=float,
        default=3.0,
        help="카메라 시야각 기반 라이다 필터에 추가할 각도 여유",
    )
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--checker-size-mm", type=float, default=25.0)
    parser.add_argument("--marker-size-mm", type=float, default=18.75)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--min-charuco-corners", type=int, default=12)
    parser.add_argument("--bbox-padding-px", type=float, default=40.0)
    parser.add_argument("--selection-radius-m", type=float, default=1.2)
    parser.add_argument("--selection-plane-slab-m", type=float, default=0.40)
    parser.add_argument("--tx", type=float, default=0.0)
    parser.add_argument("--ty", type=float, default=-0.05913)
    parser.add_argument("--tz", type=float, default=0.0)
    parser.add_argument("--qx", type=float, default=-0.5)
    parser.add_argument("--qy", type=float, default=0.5)
    parser.add_argument("--qz", type=float, default=-0.5)
    parser.add_argument("--qw", type=float, default=0.5)
    parser.add_argument("--camera-frame", default="")
    parser.add_argument("--camera-width", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=0)
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=0.0)
    parser.add_argument("--cy", type=float, default=0.0)
    parser.add_argument(
        "--intrinsic-yaml",
        default="",
        help="ChArUco intrinsic 보정 결과 YAML 경로",
    )
    parser.add_argument("--undistort-display", action="store_true", default=True)
    parser.add_argument("--no-undistort-display", action="store_true")
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/camera_lidar_extrinsic_tuned.yaml"),
    )
    return parser.parse_args()


def decode_ros_image(bridge, msg):
    return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def sample_point_cloud(msg, max_points):
    points = []
    width = int(getattr(msg, "width", 0) or 0)
    height = int(getattr(msg, "height", 0) or 0)
    estimated = max(1, width * max(1, height))
    stride = max(1, int(math.ceil(float(estimated) / float(max(1, max_points)))))
    for index, point in enumerate(
        point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
    ):
        if (index % stride) != 0:
            continue
        points.append((float(point[0]), float(point[1]), float(point[2])))
    return points


def build_charuco_board(args):
    if not hasattr(cv2, "aruco"):
        return None, None
    dictionary_id = getattr(cv2.aruco, str(args.dictionary), None)
    if dictionary_id is None:
        return None, None
    dictionary = cv2.aruco.Dictionary_get(dictionary_id)
    board = cv2.aruco.CharucoBoard_create(
        int(args.columns),
        int(args.rows),
        float(args.checker_size_mm) / 1000.0,
        float(args.marker_size_mm) / 1000.0,
        dictionary,
    )
    return dictionary, board


def camera_info_from_msg(msg):
    k_values = list(msg.K)
    return {
        "frame_id": str(msg.header.frame_id or ""),
        "width": int(msg.width or 0),
        "height": int(msg.height or 0),
        "fx": float(k_values[0] or 0.0),
        "fy": float(k_values[4] or 0.0),
        "cx": float(k_values[2] or 0.0),
        "cy": float(k_values[5] or 0.0),
        "distortion": np.array(list(msg.D), dtype=np.float32),
    }


def camera_info_from_args(args):
    required = (
        float(args.fx),
        float(args.fy),
        float(args.cx),
        float(args.cy),
        int(args.camera_width),
        int(args.camera_height),
    )
    if not all(value > 0 for value in required):
        return None
    return {
        "frame_id": str(args.camera_frame or "camera_color_optical_frame"),
        "width": int(args.camera_width),
        "height": int(args.camera_height),
        "fx": float(args.fx),
        "fy": float(args.fy),
        "cx": float(args.cx),
        "cy": float(args.cy),
        "distortion": np.zeros((5,), dtype=np.float32),
    }


def load_camera_info_from_bag(bag_path, camera_info_topic):
    bag = rosbag.Bag(bag_path)
    try:
        for topic, msg, _ in bag.read_messages(topics=[camera_info_topic]):
            if topic == camera_info_topic:
                return camera_info_from_msg(msg)
    finally:
        bag.close()
    return None


def camera_matrix_from_info(camera_info):
    return np.array(
        [
            [float(camera_info["fx"]), 0.0, float(camera_info["cx"])],
            [0.0, float(camera_info["fy"]), float(camera_info["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_camera_info_from_yaml(yaml_path):
    path = os.path.abspath(os.path.expanduser(str(yaml_path)))
    if not os.path.exists(path):
        raise RuntimeError("intrinsic yaml 파일이 존재하지 않습니다: {}".format(path))

    values = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            values[key.strip()] = raw_value.strip()

    camera_matrix = list(ast.literal_eval(values["camera_matrix"]))
    distortion = list(
        ast.literal_eval(values.get("distortion_coefficients", "[0, 0, 0, 0, 0]"))
    )
    return {
        "frame_id": str(values.get("frame_id", "camera_color_optical_frame")),
        "width": int(float(values["image_width"])),
        "height": int(float(values["image_height"])),
        "fx": float(camera_matrix[0]),
        "fy": float(camera_matrix[4]),
        "cx": float(camera_matrix[2]),
        "cy": float(camera_matrix[5]),
        "distortion": np.array(distortion, dtype=np.float32),
    }


def undistort_image_points(points_xy, camera_info):
    if not points_xy:
        return []
    camera_matrix = camera_matrix_from_info(camera_info)
    distortion = np.asarray(
        camera_info.get("distortion", np.zeros((5,), dtype=np.float32)),
        dtype=np.float64,
    ).reshape(-1, 1)
    if distortion.size == 0:
        return [tuple(float(v) for v in point_xy) for point_xy in points_xy]

    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(points, camera_matrix, distortion, P=camera_matrix)
    return [(float(point[0][0]), float(point[0][1])) for point in undistorted]


def detect_charuco_observation(image_bgr, camera_info, board, dictionary, args):
    if board is None or dictionary is None:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector_params = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)
    marker_count = 0 if ids is None else int(len(ids))
    if marker_count <= 0:
        return None

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners,
        ids,
        gray,
        board,
    )
    charuco_count = 0 if charuco_ids is None else int(len(charuco_ids))
    if charuco_count < max(4, int(args.min_charuco_corners)):
        return None

    camera_matrix = np.array(
        [
            [camera_info["fx"], 0.0, camera_info["cx"]],
            [0.0, camera_info["fy"], camera_info["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    distortion = np.asarray(camera_info.get("distortion", np.zeros((5,), dtype=np.float32)))
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        camera_matrix,
        distortion,
        None,
        None,
    )
    if not ok:
        return None

    rotation_board_to_camera, _ = cv2.Rodrigues(rvec)
    board_points = np.asarray(board.chessboardCorners, dtype=np.float32)
    board_center_board = np.mean(board_points, axis=0)
    board_center_camera = (
        rotation_board_to_camera.dot(board_center_board.reshape(3, 1)) + tvec
    ).reshape(3)
    board_normal_camera = rotation_board_to_camera[:, 2]
    board_normal_camera = board_normal_camera / max(
        1e-9, np.linalg.norm(board_normal_camera)
    )
    bbox_points = charuco_corners.reshape(-1, 2)
    bbox_min = np.min(bbox_points, axis=0) - float(args.bbox_padding_px)
    bbox_max = np.max(bbox_points, axis=0) + float(args.bbox_padding_px)
    return {
        "marker_count": marker_count,
        "charuco_count": charuco_count,
        "charuco_corners": charuco_corners.reshape(-1, 2),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "board_center_camera": board_center_camera,
        "board_normal_camera": board_normal_camera,
    }


def load_samples(args):
    bridge = CvBridge()
    samples = []
    latest_cloud = None
    camera_info = None
    if args.intrinsic_yaml:
        camera_info = load_camera_info_from_yaml(args.intrinsic_yaml)
    if camera_info is None:
        camera_info = camera_info_from_args(args)
    dictionary, board = build_charuco_board(args)
    accepted_images = 0
    bag = rosbag.Bag(args.bag)
    try:
        for topic, msg, _ in bag.read_messages(
            topics=[args.camera_info_topic, args.point_cloud_topic, args.image_topic]
        ):
            if topic == args.camera_info_topic and camera_info is None:
                camera_info = camera_info_from_msg(msg)
                continue

            if topic == args.point_cloud_topic:
                latest_cloud = {
                    "stamp": float(msg.header.stamp.to_sec()),
                    "frame_id": str(msg.header.frame_id or ""),
                    "points_xyz": sample_point_cloud(msg, args.max_cloud_points),
                }
                continue

            if topic != args.image_topic or camera_info is None or latest_cloud is None:
                continue

            image_stamp = float(msg.header.stamp.to_sec())
            if abs(image_stamp - latest_cloud["stamp"]) > float(args.max_sync_dt_s):
                continue

            accepted_images += 1
            if ((accepted_images - 1) % max(1, args.image_stride)) != 0:
                continue

            image_bgr = decode_ros_image(bridge, msg)
            sample = {
                "stamp": image_stamp,
                "image_bgr": image_bgr,
                "cloud_frame_id": latest_cloud["frame_id"],
                "points_xyz": list(latest_cloud["points_xyz"]),
            }
            sample["charuco_observation"] = detect_charuco_observation(
                image_bgr,
                camera_info,
                board,
                dictionary,
                args,
            )
            samples.append(sample)
    finally:
        bag.close()

    if camera_info is None and args.camera_info_bag:
        camera_info = load_camera_info_from_bag(
            os.path.abspath(args.camera_info_bag),
            args.camera_info_topic,
        )

    if camera_info is None:
        raise RuntimeError(
            "camera_info를 찾지 못했습니다. "
            "--camera-info-bag 또는 --fx/--fy/--cx/--cy/--camera-width/--camera-height를 사용하세요."
        )
    if not samples:
        raise RuntimeError("동기화된 image/point_cloud 샘플을 만들지 못했습니다.")
    max_samples = max(1, int(args.max_samples))
    if str(args.sample_selection) == "sequential":
        start_index = max(0, int(args.sample_start_index))
        samples = samples[start_index : start_index + max_samples]
    else:
        prioritized = [sample for sample in samples if sample.get("charuco_observation") is not None]
        if prioritized:
            prioritized.sort(
                key=lambda sample: (
                    int(sample["charuco_observation"]["charuco_count"]),
                    int(sample["charuco_observation"]["marker_count"]),
                ),
                reverse=True,
            )
            samples = prioritized[:max_samples]
        else:
            samples = samples[:max_samples]
    if not samples:
        raise RuntimeError("선택 조건에 맞는 샘플이 없습니다.")
    return camera_info, samples


def render_projection(sample, camera_info, state, args):
    image_bgr = sample["image_bgr"].copy()
    camera_matrix = camera_matrix_from_info(camera_info)
    distortion = np.asarray(
        camera_info.get("distortion", np.zeros((5,), dtype=np.float32)),
        dtype=np.float64,
    ).reshape(-1, 1)
    use_undistorted_display = bool(args.undistort_display) and distortion.size > 0
    if use_undistorted_display:
        image_bgr = cv2.undistort(image_bgr, camera_matrix, distortion, None, camera_matrix)

    fx = float(camera_info["fx"])
    fy = float(camera_info["fy"])
    cx = float(camera_info["cx"])
    cy = float(camera_info["cy"])
    image_h, image_w = image_bgr.shape[:2]
    frustum_margin_rad = math.radians(float(args.camera_frustum_margin_deg))
    left_limit_rad = math.atan2(-cx, fx) - frustum_margin_rad
    right_limit_rad = math.atan2(float(image_w - 1) - cx, fx) + frustum_margin_rad
    top_limit_rad = math.atan2(-cy, fy) - frustum_margin_rad
    bottom_limit_rad = math.atan2(float(image_h - 1) - cy, fy) + frustum_margin_rad

    roll_rad = math.radians(state["roll_deg"])
    pitch_rad = math.radians(state["pitch_deg"])
    yaw_rad = math.radians(state["yaw_deg"])
    quaternion_xyzw = quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
    translation = (state["tx"], state["ty"], state["tz"])
    rotation_lidar_to_camera = rotation_matrix_from_quaternion(quaternion_xyzw)
    translation_lidar_to_camera = np.array(translation, dtype=np.float64)

    charuco_observation = sample.get("charuco_observation")
    highlighted_board_count = 0
    board_bbox = None
    board_center_lidar = None
    board_normal_lidar = None
    if charuco_observation is not None:
        bbox_points = [
            tuple(charuco_observation["bbox_min"]),
            (
                float(charuco_observation["bbox_max"][0]),
                float(charuco_observation["bbox_min"][1]),
            ),
            tuple(charuco_observation["bbox_max"]),
            (
                float(charuco_observation["bbox_min"][0]),
                float(charuco_observation["bbox_max"][1]),
            ),
        ]
        if use_undistorted_display:
            bbox_points = undistort_image_points(bbox_points, camera_info)
        bbox_x_values = [point_xy[0] for point_xy in bbox_points]
        bbox_y_values = [point_xy[1] for point_xy in bbox_points]
        board_bbox = (
            (min(bbox_x_values), min(bbox_y_values)),
            (max(bbox_x_values), max(bbox_y_values)),
        )
        board_center_lidar = rotation_lidar_to_camera.T.dot(
            charuco_observation["board_center_camera"] - translation_lidar_to_camera
        )
        board_normal_lidar = rotation_lidar_to_camera.T.dot(
            charuco_observation["board_normal_camera"]
        )
        board_normal_lidar = board_normal_lidar / max(
            1e-9, float(np.linalg.norm(board_normal_lidar))
        )

    projected_count = 0
    context_points_lidar = []
    highlighted_points_lidar = []
    projected_points_uv = []
    highlighted_points_uv = []
    for point_xyz in sample["points_xyz"]:
        px, py, pz = point_xyz
        if px < (-args.rear_m) or px > args.forward_m:
            continue
        if abs(py) > args.half_width_m or abs(pz) > args.height_abs_m:
            continue
        context_points_lidar.append((px, py, pz))

        rotated = rotate_point_by_quaternion(point_xyz, quaternion_xyzw)
        camera_x = rotated[0] + translation[0]
        camera_y = rotated[1] + translation[1]
        camera_z = rotated[2] + translation[2]
        if camera_z <= 0.10 or camera_z >= args.max_range_m:
            continue
        horizontal_angle_rad = math.atan2(camera_x, camera_z)
        vertical_angle_rad = math.atan2(camera_y, camera_z)
        if state.get("use_camera_frustum_filter", True):
            if (
                horizontal_angle_rad < left_limit_rad
                or horizontal_angle_rad > right_limit_rad
                or vertical_angle_rad < top_limit_rad
                or vertical_angle_rad > bottom_limit_rad
            ):
                continue

        if use_undistorted_display:
            u = (fx * (camera_x / camera_z)) + cx
            v = (fy * (camera_y / camera_z)) + cy
        else:
            projected = cv2.projectPoints(
                np.asarray([[camera_x, camera_y, camera_z]], dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                camera_matrix,
                distortion,
            )[0].reshape(-1, 2)
            u = float(projected[0][0])
            v = float(projected[0][1])
        if u < 0.0 or u >= float(image_w) or v < 0.0 or v >= float(image_h):
            continue
        projected_count += 1
        projected_points_uv.append((float(u), float(v)))
        point_color = (255, 255, 0)
        if (
            board_bbox is not None
            and board_center_lidar is not None
            and board_normal_lidar is not None
        ):
            bbox_min, bbox_max = board_bbox
            image_match = (
                u >= float(bbox_min[0])
                and u <= float(bbox_max[0])
                and v >= float(bbox_min[1])
                and v <= float(bbox_max[1])
            )
            radius_match = (
                math.sqrt(
                    (px - float(board_center_lidar[0])) ** 2
                    + (py - float(board_center_lidar[1])) ** 2
                    + (pz - float(board_center_lidar[2])) ** 2
                )
                <= float(args.selection_radius_m)
            )
            plane_offset = -float(np.dot(board_normal_lidar, board_center_lidar))
            plane_match = (
                abs(
                    (px * float(board_normal_lidar[0]))
                    + (py * float(board_normal_lidar[1]))
                    + (pz * float(board_normal_lidar[2]))
                    + plane_offset
                )
                <= float(args.selection_plane_slab_m)
            )
            if image_match and radius_match and plane_match:
                point_color = (0, 0, 255)
                highlighted_board_count += 1
                highlighted_points_lidar.append((px, py, pz))
                highlighted_points_uv.append((float(u), float(v)))
        cv2.circle(
            image_bgr,
            (int(round(u)), int(round(v))),
            max(1, int(args.point_radius_px)),
            point_color,
            -1,
            lineType=cv2.LINE_AA,
        )

    if charuco_observation is not None:
        bbox_min, bbox_max = board_bbox
        corner_points = list(charuco_observation["charuco_corners"])
        if use_undistorted_display:
            corner_points = undistort_image_points(corner_points, camera_info)
        cv2.rectangle(
            image_bgr,
            (int(round(bbox_min[0])), int(round(bbox_min[1]))),
            (int(round(bbox_max[0])), int(round(bbox_max[1]))),
            (0, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        for point_uv in corner_points:
            cv2.circle(
                image_bgr,
                (int(round(point_uv[0])), int(round(point_uv[1]))),
                4,
                (0, 255, 0),
                -1,
                lineType=cv2.LINE_AA,
            )

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (8, 8), (image_w - 8, 150), (0, 0, 0), -1)
    image_bgr = cv2.addWeighted(overlay, 0.35, image_bgr, 0.65, 0.0)

    reference_projected_points_uv = state.get("reference_projected_points_uv") or []
    reference_highlighted_points_uv = state.get("reference_highlighted_points_uv") or []
    if reference_projected_points_uv:
        for u, v in reference_projected_points_uv:
            cv2.circle(
                image_bgr,
                (int(round(u)), int(round(v))),
                1,
                (0, 255, 0),
                -1,
                lineType=cv2.LINE_AA,
            )
    if reference_highlighted_points_uv:
        for u, v in reference_highlighted_points_uv:
            cv2.circle(
                image_bgr,
                (int(round(u)), int(round(v))),
                max(2, int(args.point_radius_px)),
                (0, 200, 0),
                -1,
                lineType=cv2.LINE_AA,
            )

    lines = [
        "sample {}/{} | projected {} pts".format(
            state["sample_index"] + 1,
            state["sample_count"],
            projected_count,
        ),
        "t_xyz = [{:.4f}, {:.4f}, {:.4f}] m".format(
            state["tx"], state["ty"], state["tz"]
        ),
        "rpy = [{:.2f}, {:.2f}, {:.2f}] deg".format(
            state["roll_deg"], state["pitch_deg"], state["yaw_deg"]
        ),
        "charuco = {} | board candidate pts = {}".format(
            0 if charuco_observation is None else int(charuco_observation["charuco_count"]),
            int(highlighted_board_count),
        ),
        "display = {} | frustum-filter = {} ({:.1f}deg margin)".format(
            "undistorted" if use_undistorted_display else "raw",
            "on" if state.get("use_camera_frustum_filter", True) else "off",
            float(args.camera_frustum_margin_deg),
        ),
        "step t={:.3f}m r={:.2f}deg | n/p sample | f frustum on/off | g ref capture | c ref clear | s save | q quit".format(
            state["translation_step_m"], state["rotation_step_deg"]
        ),
        "1/2:image-x  3/4:image-y  5/6:depth  u/o:roll  i/k:pitch  j/l:yaw  [-]/[=]:step",
    ]
    y = 24
    for line in lines:
        cv2.putText(
            image_bgr,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24
    overview_width = max(320, int(args.overview_width_px))
    overview_bgr = draw_lidar_overview(
        context_points_lidar,
        highlighted_points_lidar,
        state.get("reference_context_points_lidar") or [],
        state.get("reference_highlighted_points_lidar") or [],
        camera_info,
        quaternion_xyzw,
        translation,
        args,
        image_h,
        overview_width,
    )
    combined = np.hstack((image_bgr, overview_bgr))
    return combined, quaternion_xyzw, projected_points_uv, highlighted_points_uv, context_points_lidar, highlighted_points_lidar


def _normalize_vector(vector_xyz):
    norm = math.sqrt(
        float(vector_xyz[0]) * float(vector_xyz[0])
        + float(vector_xyz[1]) * float(vector_xyz[1])
        + float(vector_xyz[2]) * float(vector_xyz[2])
    )
    if norm <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.asarray(vector_xyz, dtype=np.float64) / norm


def _normalize_xy_direction(vector_xyz, fallback_xy):
    direction_xy = np.array(
        [float(vector_xyz[0]), float(vector_xyz[1])],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(direction_xy))
    if norm <= 1e-9:
        direction_xy = np.array(fallback_xy, dtype=np.float64)
        norm = float(np.linalg.norm(direction_xy))
        if norm <= 1e-9:
            return np.array([1.0, 0.0], dtype=np.float64)
    return direction_xy / norm


def draw_lidar_overview(
    context_points_lidar,
    highlighted_points_lidar,
    reference_context_points_lidar,
    reference_highlighted_points_lidar,
    camera_info,
    quaternion_xyzw,
    translation_lidar_to_camera,
    args,
    canvas_h,
    canvas_w,
):
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    top_margin = 56
    bottom_margin = 36
    side_margin = 28
    draw_h = max(40, canvas_h - top_margin - bottom_margin)
    draw_w = max(40, canvas_w - (2 * side_margin))
    robot_px = (canvas_w // 2, canvas_h - bottom_margin)

    max_forward_m = max(1.0, float(args.forward_m))
    max_rear_m = max(0.1, float(args.rear_m))
    half_width_m = max(0.5, float(args.half_width_m))
    point_radius_px = max(1, int(args.point_radius_px))

    def world_to_canvas(point_xyz):
        forward_x = float(point_xyz[0])
        lateral_y = float(point_xyz[1])
        norm_y = lateral_y / half_width_m
        depth_span = max_forward_m + max_rear_m
        norm_x = (forward_x + max_rear_m) / max(0.1, depth_span)
        px = int(round((canvas_w * 0.5) - (norm_y * (draw_w * 0.5))))
        py = int(round((canvas_h - bottom_margin) - (norm_x * draw_h)))
        return px, py

    grid_color = (40, 40, 40)
    grid_step_m = 2.0
    forward_m = -max_rear_m
    while forward_m <= (max_forward_m + 1e-6):
        _, py = world_to_canvas((forward_m, 0.0, 0.0))
        cv2.line(canvas, (side_margin, py), (canvas_w - side_margin, py), grid_color, 1, cv2.LINE_AA)
        forward_m += grid_step_m
    lateral_m = -half_width_m
    while lateral_m <= (half_width_m + 1e-6):
        px, _ = world_to_canvas((0.0, lateral_m, 0.0))
        cv2.line(canvas, (px, top_margin), (px, canvas_h - bottom_margin), grid_color, 1, cv2.LINE_AA)
        lateral_m += grid_step_m

    for point_xyz in context_points_lidar:
        px, py = world_to_canvas(point_xyz)
        if px < side_margin or px >= (canvas_w - side_margin):
            continue
        if py < top_margin or py >= (canvas_h - bottom_margin):
            continue
        cv2.circle(canvas, (px, py), max(1, point_radius_px - 1), (140, 140, 140), -1, lineType=cv2.LINE_AA)

    for point_xyz in highlighted_points_lidar:
        px, py = world_to_canvas(point_xyz)
        if px < side_margin or px >= (canvas_w - side_margin):
            continue
        if py < top_margin or py >= (canvas_h - bottom_margin):
            continue
        cv2.circle(canvas, (px, py), point_radius_px + 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    for point_xyz in reference_context_points_lidar:
        px, py = world_to_canvas(point_xyz)
        if px < side_margin or px >= (canvas_w - side_margin):
            continue
        if py < top_margin or py >= (canvas_h - bottom_margin):
            continue
        cv2.circle(canvas, (px, py), 1, (0, 120, 0), -1, lineType=cv2.LINE_AA)

    for point_xyz in reference_highlighted_points_lidar:
        px, py = world_to_canvas(point_xyz)
        if px < side_margin or px >= (canvas_w - side_margin):
            continue
        if py < top_margin or py >= (canvas_h - bottom_margin):
            continue
        cv2.circle(canvas, (px, py), point_radius_px + 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    rotation_lidar_to_camera = rotation_matrix_from_quaternion(quaternion_xyzw)
    camera_origin_lidar = -rotation_lidar_to_camera.T.dot(np.asarray(translation_lidar_to_camera, dtype=np.float64))
    camera_center_ray = rotation_lidar_to_camera.T.dot(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    left_ray_camera = _normalize_vector(
        np.array(
            [
                (0.0 - float(camera_info["cx"])) / max(1e-9, float(camera_info["fx"])),
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )
    )
    right_ray_camera = _normalize_vector(
        np.array(
            [
                ((float(camera_info["width"]) - 1.0) - float(camera_info["cx"])) / max(1e-9, float(camera_info["fx"])),
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )
    )
    camera_left_ray = rotation_lidar_to_camera.T.dot(left_ray_camera)
    camera_right_ray = rotation_lidar_to_camera.T.dot(right_ray_camera)

    camera_origin_px = world_to_canvas((camera_origin_lidar[0], camera_origin_lidar[1], camera_origin_lidar[2]))
    ray_length_m = min(max_forward_m, 4.0)
    center_xy = _normalize_xy_direction(camera_center_ray, (1.0, 0.0))
    left_xy = _normalize_xy_direction(camera_left_ray, tuple(center_xy))
    right_xy = _normalize_xy_direction(camera_right_ray, tuple(center_xy))
    center_tip = world_to_canvas(
        (
            camera_origin_lidar[0] + (center_xy[0] * ray_length_m),
            camera_origin_lidar[1] + (center_xy[1] * ray_length_m),
            0.0,
        )
    )
    left_tip = world_to_canvas(
        (
            camera_origin_lidar[0] + (left_xy[0] * ray_length_m),
            camera_origin_lidar[1] + (left_xy[1] * ray_length_m),
            0.0,
        )
    )
    right_tip = world_to_canvas(
        (
            camera_origin_lidar[0] + (right_xy[0] * ray_length_m),
            camera_origin_lidar[1] + (right_xy[1] * ray_length_m),
            0.0,
        )
    )
    fov_overlay = canvas.copy()
    fov_polygon = np.array(
        [
            [camera_origin_px[0], camera_origin_px[1]],
            [left_tip[0], left_tip[1]],
            [right_tip[0], right_tip[1]],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(fov_overlay, fov_polygon, (180, 110, 20))
    canvas = cv2.addWeighted(fov_overlay, 0.42, canvas, 0.58, 0.0)
    cv2.line(canvas, camera_origin_px, center_tip, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, camera_origin_px, left_tip, (255, 220, 80), 3, cv2.LINE_AA)
    cv2.line(canvas, camera_origin_px, right_tip, (255, 220, 80), 3, cv2.LINE_AA)
    cv2.circle(canvas, camera_origin_px, 4, (255, 255, 0), -1, lineType=cv2.LINE_AA)

    robot_triangle = np.array(
        [
            [robot_px[0], robot_px[1] - 14],
            [robot_px[0] - 10, robot_px[1] + 8],
            [robot_px[0] + 10, robot_px[1] + 8],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(canvas, robot_triangle, (120, 120, 120))
    cv2.circle(canvas, robot_px, 3, (220, 220, 220), -1, lineType=cv2.LINE_AA)

    title_lines = [
        "LiDAR overview (BEV)",
        "gray: context  red: board-candidate",
        "green: reference snapshot",
        "yellow: camera center  amber: camera FOV area",
    ]
    y = 22
    for line in title_lines:
        cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        y += 18

    return canvas


def save_yaml(output_path, camera_info, sample, state, quaternion_xyzw):
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    rotation_lidar_to_camera = rotation_matrix_from_quaternion(quaternion_xyzw)
    translation_lidar_to_camera = np.array(
        [state["tx"], state["ty"], state["tz"]],
        dtype=np.float64,
    )
    camera_origin_in_lidar = -rotation_lidar_to_camera.T.dot(translation_lidar_to_camera)
    baseline_distance_m = float(np.linalg.norm(camera_origin_in_lidar))
    content = []
    content.append("bag: {}".format(state["bag_path"]))
    content.append("source_frame: {}".format(sample["cloud_frame_id"]))
    content.append("target_frame: {}".format(camera_info["frame_id"]))
    content.append(
        "translation_xyz: [{:.6f}, {:.6f}, {:.6f}]".format(
            state["tx"], state["ty"], state["tz"]
        )
    )
    content.append(
        "rotation_xyzw: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
            quaternion_xyzw[3],
        )
    )
    content.append(
        "rotation_rpy_deg: [{:.6f}, {:.6f}, {:.6f}]".format(
            state["roll_deg"], state["pitch_deg"], state["yaw_deg"]
        )
    )
    content.append(
        "camera_origin_in_source_frame_xyz: [{:.6f}, {:.6f}, {:.6f}]".format(
            camera_origin_in_lidar[0],
            camera_origin_in_lidar[1],
            camera_origin_in_lidar[2],
        )
    )
    content.append("baseline_distance_m: {:.6f}".format(baseline_distance_m))
    content.append(
        "launch_override: >-\n  scene_detector_point_cloud_fallback_source_frame:={} scene_detector_point_cloud_fallback_target_frame:={} scene_detector_point_cloud_fallback_translation_xyz:='[{:.6f}, {:.6f}, {:.6f}]' scene_detector_point_cloud_fallback_rotation_xyzw:='[{:.6f}, {:.6f}, {:.6f}, {:.6f}]'".format(
            sample["cloud_frame_id"],
            camera_info["frame_id"],
            state["tx"],
            state["ty"],
            state["tz"],
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
            quaternion_xyzw[3],
        )
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(content) + "\n")


def main():
    args = parse_args()
    if args.no_undistort_display:
        args.undistort_display = False
    camera_info, samples = load_samples(args)

    init_quaternion = (args.qx, args.qy, args.qz, args.qw)
    init_roll, init_pitch, init_yaw = euler_from_quaternion(init_quaternion)
    state = {
        "bag_path": os.path.abspath(args.bag),
        "sample_index": 0,
        "sample_count": len(samples),
        "tx": float(args.tx),
        "ty": float(args.ty),
        "tz": float(args.tz),
        "roll_deg": math.degrees(init_roll),
        "pitch_deg": math.degrees(init_pitch),
        "yaw_deg": math.degrees(init_yaw),
        "translation_step_m": 0.01,
        "rotation_step_deg": 1.0,
        "use_camera_frustum_filter": True,
        "reference_projected_points_uv": [],
        "reference_highlighted_points_uv": [],
        "reference_context_points_lidar": [],
        "reference_highlighted_points_lidar": [],
    }

    window_name = "camera_lidar_extrinsic_tuner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1440, 900)

    while True:
        sample = samples[state["sample_index"]]
        rendered, quaternion_xyzw, projected_points_uv, highlighted_points_uv, context_points_lidar, highlighted_points_lidar = render_projection(
            sample,
            camera_info,
            state,
            args,
        )
        cv2.imshow(window_name, rendered)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            save_yaml(args.output, camera_info, sample, state, quaternion_xyzw)
            print("saved tuned extrinsic to {}".format(args.output))
            break
        if key == ord("n"):
            state["sample_index"] = (state["sample_index"] + 1) % state["sample_count"]
            continue
        if key == ord("p"):
            state["sample_index"] = (state["sample_index"] - 1) % state["sample_count"]
            continue
        if key == ord("f"):
            state["use_camera_frustum_filter"] = not state["use_camera_frustum_filter"]
            continue
        if key == ord("g"):
            state["reference_projected_points_uv"] = list(projected_points_uv)
            state["reference_highlighted_points_uv"] = list(highlighted_points_uv)
            state["reference_context_points_lidar"] = list(context_points_lidar)
            state["reference_highlighted_points_lidar"] = list(highlighted_points_lidar)
            continue
        if key == ord("c"):
            state["reference_projected_points_uv"] = []
            state["reference_highlighted_points_uv"] = []
            state["reference_context_points_lidar"] = []
            state["reference_highlighted_points_lidar"] = []
            continue

        if key == ord("1"):
            state["tx"] -= state["translation_step_m"]
        elif key == ord("2"):
            state["tx"] += state["translation_step_m"]
        elif key == ord("3"):
            state["ty"] -= state["translation_step_m"]
        elif key == ord("4"):
            state["ty"] += state["translation_step_m"]
        elif key == ord("5"):
            state["tz"] -= state["translation_step_m"]
        elif key == ord("6"):
            state["tz"] += state["translation_step_m"]
        elif key == ord("u"):
            state["roll_deg"] -= state["rotation_step_deg"]
        elif key == ord("o"):
            state["roll_deg"] += state["rotation_step_deg"]
        elif key == ord("i"):
            state["pitch_deg"] -= state["rotation_step_deg"]
        elif key == ord("k"):
            state["pitch_deg"] += state["rotation_step_deg"]
        elif key == ord("j"):
            state["yaw_deg"] -= state["rotation_step_deg"]
        elif key == ord("l"):
            state["yaw_deg"] += state["rotation_step_deg"]
        elif key in (ord("["), ord("-"), ord("_")):
            state["translation_step_m"] = max(0.001, state["translation_step_m"] * 0.5)
            state["rotation_step_deg"] = max(0.1, state["rotation_step_deg"] * 0.5)
        elif key in (ord("]"), ord("="), ord("+")):
            state["translation_step_m"] = min(0.20, state["translation_step_m"] * 2.0)
            state["rotation_step_deg"] = min(10.0, state["rotation_step_deg"] * 2.0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
