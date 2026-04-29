#!/usr/bin/python3
import argparse
import ast
import math
import os
import sys

import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as point_cloud2
from scipy.optimize import least_squares

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError as exc:
    raise SystemExit(
        "필수 모듈을 불러오지 못했습니다: {}.\n"
        "이 스크립트는 ROS OpenCV가 설치된 시스템 Python으로 실행해야 합니다.\n"
        "다음처럼 다시 실행해 주세요:\n"
        "  /usr/bin/python3 src/xai_driving_explainer/scripts/auto_calibrate_camera_lidar_charuco.py ...\n"
        "현재 interpreter: {}".format(exc, sys.executable)
    )


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


def params_to_transform(params):
    tx, ty, tz, roll_deg, pitch_deg, yaw_deg = [float(v) for v in params]
    quaternion_xyzw = quaternion_from_euler(
        math.radians(roll_deg),
        math.radians(pitch_deg),
        math.radians(yaw_deg),
    )
    rotation = rotation_matrix_from_quaternion(quaternion_xyzw)
    translation = np.array([tx, ty, tz], dtype=np.float64)
    return rotation, translation, quaternion_xyzw


def expand_params_with_args(params, args):
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    if params.size == 6:
        return params
    if params.size == 3 and bool(getattr(args, "lock_initial_rotation", False)):
        initial_rpy = getattr(args, "_initial_rpy_prior_deg", (0.0, -90.0, 0.0))
        return np.array(
            [
                float(params[0]),
                float(params[1]),
                float(params[2]),
                float(initial_rpy[0]),
                float(initial_rpy[1]),
                float(initial_rpy[2]),
            ],
            dtype=np.float64,
        )
    raise RuntimeError("지원하지 않는 파라미터 길이입니다: {}".format(params.size))


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
        "distortion": np.array(distortion, dtype=np.float64),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatically estimate LiDAR-to-camera extrinsic from a ChArUco bag."
    )
    parser.add_argument("--bag", required=True)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument(
        "--intrinsic-yaml",
        default="",
        help="ChArUco intrinsic 보정 결과 YAML 경로",
    )
    parser.add_argument("--point-cloud-topic", default="/ouster/points")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--checker-size-mm", type=float, default=25.0)
    parser.add_argument("--marker-size-mm", type=float, default=18.75)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--image-stride", type=int, default=1)
    parser.add_argument("--max-sync-dt-s", type=float, default=0.12)
    parser.add_argument("--max-cloud-points", type=int, default=20000)
    parser.add_argument("--min-charuco-corners", type=int, default=12)
    parser.add_argument("--bbox-padding-px", type=float, default=60.0)
    parser.add_argument("--selection-radius-m", type=float, default=1.20)
    parser.add_argument("--selection-plane-slab-m", type=float, default=0.40)
    parser.add_argument("--board-extent-margin-m", type=float, default=0.08)
    parser.add_argument("--plane-inlier-threshold-m", type=float, default=0.03)
    parser.add_argument("--plane-ransac-iters", type=int, default=300)
    parser.add_argument("--min-plane-points", type=int, default=8)
    parser.add_argument("--point-residual-max-per-sample", type=int, default=40)
    parser.add_argument("--board-extent-min-scale", type=float, default=0.35)
    parser.add_argument("--board-extent-max-scale", type=float, default=1.80)
    parser.add_argument("--board-aspect-tolerance", type=float, default=0.45)
    parser.add_argument("--geometry-filter-warmup-iterations", type=int, default=1)
    parser.add_argument("--outer-iterations", type=int, default=4)
    parser.add_argument("--tx", type=float, default=0.0)
    parser.add_argument("--ty", type=float, default=-0.05913)
    parser.add_argument("--tz", type=float, default=0.0)
    parser.add_argument("--qx", type=float, default=0.5)
    parser.add_argument("--qy", type=float, default=-0.5)
    parser.add_argument("--qz", type=float, default=0.5)
    parser.add_argument("--qw", type=float, default=0.5)
    parser.add_argument(
        "--lock-initial-rotation",
        action="store_true",
        help="초기 quaternion이 의미하는 회전을 고정하고 translation만 자동 보정",
    )
    parser.add_argument(
        "--prior-rotation-like-initial-sigma-deg",
        type=float,
        default=0.0,
        help="초기 quaternion이 의미하는 rpy를 얼마나 강하게 믿을지에 대한 sigma (deg). 0이면 비활성화",
    )
    parser.add_argument(
        "--prior-camera-origin-in-source-frame-xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X_M", "Y_M", "Z_M"),
        help="source frame 기준 camera origin의 대략적 위치 prior",
    )
    parser.add_argument(
        "--prior-camera-origin-sigma-m",
        type=float,
        default=0.10,
        help="camera origin prior를 얼마나 강하게 믿을지에 대한 sigma (m)",
    )
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/camera_lidar_extrinsic_auto.yaml"),
    )
    return parser.parse_args()


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
    return np.asarray(points, dtype=np.float32)


def collect_candidate_samples(args, board, dictionary):
    bridge = CvBridge()
    latest_cloud = None
    camera_info = None
    if args.intrinsic_yaml:
        camera_info = load_camera_info_from_yaml(args.intrinsic_yaml)
    candidates = []
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

            image_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            sample = {
                "stamp": image_stamp,
                "image_bgr": image_bgr,
                "cloud_frame_id": latest_cloud["frame_id"],
                "points_xyz": latest_cloud["points_xyz"],
            }
            observation = detect_charuco_observation(
                sample,
                camera_info,
                board,
                dictionary,
                args,
            )
            if observation is None:
                continue
            sample["charuco_observation"] = observation
            candidates.append(sample)
    finally:
        bag.close()

    if camera_info is None:
        raise RuntimeError("camera_info를 bag에서 찾지 못했습니다.")
    if not candidates:
        raise RuntimeError("ChArUco를 충분히 인식한 동기화 샘플을 만들지 못했습니다.")

    candidates.sort(
        key=lambda item: int(item["charuco_observation"]["charuco_count"]),
        reverse=True,
    )
    max_samples = max(1, int(args.max_samples))
    return camera_info, candidates[:max_samples]


def build_charuco_board(args):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco 모듈을 사용할 수 없습니다.")
    dictionary_id = getattr(cv2.aruco, args.dictionary, None)
    if dictionary_id is None:
        raise RuntimeError("지원하지 않는 dictionary 이름입니다: {}".format(args.dictionary))
    dictionary = cv2.aruco.Dictionary_get(dictionary_id)
    board = cv2.aruco.CharucoBoard_create(
        int(args.columns),
        int(args.rows),
        float(args.checker_size_mm) / 1000.0,
        float(args.marker_size_mm) / 1000.0,
        dictionary,
    )
    return dictionary, board


def detect_charuco_observation(sample, camera_info, board, dictionary, args):
    gray = cv2.cvtColor(sample["image_bgr"], cv2.COLOR_BGR2GRAY)
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
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        camera_matrix,
        camera_info["distortion"],
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
    board_normal_camera = board_normal_camera / max(1e-9, np.linalg.norm(board_normal_camera))
    bbox_points = charuco_corners.reshape(-1, 2)
    bbox_min = np.min(bbox_points, axis=0) - float(args.bbox_padding_px)
    bbox_max = np.max(bbox_points, axis=0) + float(args.bbox_padding_px)
    bbox_center = 0.5 * (bbox_min + bbox_max)

    return {
        "marker_count": marker_count,
        "charuco_count": charuco_count,
        "rvec": rvec.reshape(3),
        "tvec": tvec.reshape(3),
        "rotation_board_to_camera": rotation_board_to_camera,
        "board_center_camera": board_center_camera,
        "board_normal_camera": board_normal_camera,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_center": bbox_center,
    }


def transform_points_lidar_to_camera(points_xyz, rotation_lidar_to_camera, translation_lidar_to_camera):
    return points_xyz.dot(rotation_lidar_to_camera.T) + translation_lidar_to_camera.reshape(1, 3)


def project_points_camera_to_image(points_camera_xyz, camera_info):
    z = points_camera_xyz[:, 2]
    valid = z > 0.05
    u = np.zeros(len(points_camera_xyz), dtype=np.float64)
    v = np.zeros(len(points_camera_xyz), dtype=np.float64)
    camera_matrix = np.array(
        [
            [camera_info["fx"], 0.0, camera_info["cx"]],
            [0.0, camera_info["fy"], camera_info["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(
        camera_info.get("distortion", np.zeros((5,), dtype=np.float32)),
        dtype=np.float64,
    ).reshape(-1, 1)
    if np.any(valid):
        projected = cv2.projectPoints(
            np.asarray(points_camera_xyz[valid], dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            camera_matrix,
            distortion,
        )[0].reshape(-1, 2)
        u[valid] = projected[:, 0]
        v[valid] = projected[:, 1]
    return u, v, valid


def fit_plane_ransac(points_xyz, threshold_m, iterations):
    if len(points_xyz) < 3:
        return None
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    best_normal = None
    best_d = None
    best_inliers = None
    count = len(points_xyz)
    for _ in range(max(1, int(iterations))):
        ids = np.random.choice(count, 3, replace=False)
        p1, p2, p3 = points_xyz[ids]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm <= 1e-9:
            continue
        normal = normal / norm
        d_value = -float(np.dot(normal, p1))
        distances = np.abs(points_xyz.dot(normal) + d_value)
        inliers = np.where(distances <= float(threshold_m))[0]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_normal = normal
            best_d = d_value
            best_inliers = inliers
    if best_inliers is None or len(best_inliers) < 3:
        return None

    plane_points = points_xyz[best_inliers]
    centroid = np.mean(plane_points, axis=0)
    covariance = np.cov((plane_points - centroid).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    extents = (plane_points - centroid).dot(eigenvectors)
    ranges = extents.max(axis=0) - extents.min(axis=0)
    return {
        "normal": best_normal,
        "d": best_d,
        "inlier_indices": best_inliers,
        "points_xyz": plane_points,
        "centroid_xyz": centroid,
        "ranges_xyz": ranges,
        "axes_xyz": eigenvectors,
    }


def expected_board_dimensions_m(args):
    checker_size_m = float(args.checker_size_mm) / 1000.0
    return (
        max(1, int(args.columns)) * checker_size_m,
        max(1, int(args.rows)) * checker_size_m,
    )


def plane_matches_board_geometry(plane, args):
    ranges = np.asarray(plane.get("ranges_xyz", []), dtype=np.float64).reshape(-1)
    if ranges.size < 2:
        return False, {}

    plane_extents = sorted([float(abs(ranges[0])), float(abs(ranges[1]))], reverse=True)
    major_extent = plane_extents[0]
    minor_extent = plane_extents[1]
    expected_width, expected_height = expected_board_dimensions_m(args)
    expected_extents = sorted([expected_width, expected_height], reverse=True)
    expected_major = expected_extents[0]
    expected_minor = expected_extents[1]

    min_scale = max(0.05, float(args.board_extent_min_scale))
    max_scale = max(min_scale, float(args.board_extent_max_scale))
    aspect_tolerance = max(0.05, float(args.board_aspect_tolerance))

    major_ok = (expected_major * min_scale) <= major_extent <= (expected_major * max_scale)
    minor_ok = (expected_minor * min_scale) <= minor_extent <= (expected_minor * max_scale)
    observed_aspect = major_extent / max(1e-6, minor_extent)
    expected_aspect = expected_major / max(1e-6, expected_minor)
    aspect_ok = abs(math.log(max(1e-6, observed_aspect / expected_aspect))) <= aspect_tolerance

    diagnostics = {
        "major_extent_m": major_extent,
        "minor_extent_m": minor_extent,
        "expected_major_m": expected_major,
        "expected_minor_m": expected_minor,
        "observed_aspect": observed_aspect,
        "expected_aspect": expected_aspect,
        "major_ok": bool(major_ok),
        "minor_ok": bool(minor_ok),
        "aspect_ok": bool(aspect_ok),
    }
    return bool(major_ok and minor_ok and aspect_ok), diagnostics


def select_lidar_board_points(sample, observation, camera_info, args, params, enforce_board_geometry=True):
    full_params = expand_params_with_args(params, args)
    rotation_lidar_to_camera, translation_lidar_to_camera, _ = params_to_transform(full_params)
    points_lidar = np.asarray(sample["points_xyz"], dtype=np.float64)
    points_camera = transform_points_lidar_to_camera(
        points_lidar,
        rotation_lidar_to_camera,
        translation_lidar_to_camera,
    )
    u, v, valid = project_points_camera_to_image(points_camera, camera_info)
    bbox_min = observation["bbox_min"]
    bbox_max = observation["bbox_max"]
    image_mask = (
        valid
        & (u >= bbox_min[0])
        & (u <= bbox_max[0])
        & (v >= bbox_min[1])
        & (v <= bbox_max[1])
        & (u < float(camera_info["width"]))
        & (v < float(camera_info["height"]))
        & (u >= 0.0)
        & (v >= 0.0)
    )
    if not np.any(image_mask):
        return None

    board_center_camera = observation["board_center_camera"]
    board_normal_camera = observation["board_normal_camera"]
    board_center_lidar_est = rotation_lidar_to_camera.T.dot(
        board_center_camera - translation_lidar_to_camera
    )
    board_normal_lidar_est = rotation_lidar_to_camera.T.dot(board_normal_camera)
    board_normal_lidar_est = board_normal_lidar_est / max(
        1e-9, np.linalg.norm(board_normal_lidar_est)
    )
    board_x_axis_camera = observation["rotation_board_to_camera"][:, 0]
    board_y_axis_camera = observation["rotation_board_to_camera"][:, 1]
    board_x_axis_lidar_est = rotation_lidar_to_camera.T.dot(board_x_axis_camera)
    board_y_axis_lidar_est = rotation_lidar_to_camera.T.dot(board_y_axis_camera)
    board_x_axis_lidar_est = board_x_axis_lidar_est / max(
        1e-9, np.linalg.norm(board_x_axis_lidar_est)
    )
    board_y_axis_lidar_est = board_y_axis_lidar_est / max(
        1e-9, np.linalg.norm(board_y_axis_lidar_est)
    )
    expected_width_m, expected_height_m = expected_board_dimensions_m(args)
    extent_margin_m = max(0.01, float(args.board_extent_margin_m))
    half_width_m = (0.5 * expected_width_m) + extent_margin_m
    half_height_m = (0.5 * expected_height_m) + extent_margin_m
    radius_mask = np.linalg.norm(points_lidar - board_center_lidar_est.reshape(1, 3), axis=1) <= float(
        args.selection_radius_m
    )
    plane_offset = -float(np.dot(board_normal_lidar_est, board_center_lidar_est))
    plane_mask = (
        np.abs(points_lidar.dot(board_normal_lidar_est) + plane_offset)
        <= float(args.selection_plane_slab_m)
    )
    deltas = points_lidar - board_center_lidar_est.reshape(1, 3)
    local_x = np.abs(deltas.dot(board_x_axis_lidar_est))
    local_y = np.abs(deltas.dot(board_y_axis_lidar_est))
    board_rect_mask = (local_x <= half_width_m) & (local_y <= half_height_m)
    candidate_mask = image_mask & radius_mask & plane_mask & board_rect_mask
    candidate_points = points_lidar[candidate_mask]
    if len(candidate_points) < int(args.min_plane_points):
        candidate_points = points_lidar[image_mask & plane_mask & board_rect_mask]
    if len(candidate_points) < int(args.min_plane_points):
        candidate_points = points_lidar[image_mask & plane_mask]
    if len(candidate_points) < int(args.min_plane_points):
        return None

    plane = fit_plane_ransac(
        candidate_points,
        float(args.plane_inlier_threshold_m),
        int(args.plane_ransac_iters),
    )
    if plane is None or len(plane["points_xyz"]) < int(args.min_plane_points):
        return None

    if float(np.dot(plane["normal"], board_normal_lidar_est)) < 0.0:
        plane["normal"] = -plane["normal"]
        plane["d"] = -plane["d"]
    matches_board, diagnostics = plane_matches_board_geometry(plane, args)
    plane["board_geometry"] = diagnostics
    if enforce_board_geometry and not matches_board:
        return None
    return plane


def build_optimization_observations(camera_info, samples, board, dictionary, args, params, enforce_board_geometry=True):
    observations = []
    for sample in samples:
        observation = sample.get("charuco_observation")
        if observation is None:
            continue
        plane = select_lidar_board_points(
            sample,
            observation,
            camera_info,
            args,
            params,
            enforce_board_geometry=enforce_board_geometry,
        )
        if plane is None:
            continue
        points_xyz = plane["points_xyz"]
        if len(points_xyz) > int(args.point_residual_max_per_sample):
            indices = np.linspace(
                0,
                len(points_xyz) - 1,
                int(args.point_residual_max_per_sample),
                dtype=np.int32,
            )
            points_xyz = points_xyz[indices]
        observation["lidar_plane_points_xyz"] = points_xyz
        observation["lidar_plane_normal_xyz"] = plane["normal"]
        observation["lidar_plane_centroid_xyz"] = plane["centroid_xyz"]
        observation["lidar_plane_ranges_xyz"] = plane["ranges_xyz"]
        observation["stamp"] = sample["stamp"]
        observations.append(observation)
    return observations


def residual_function(params, observations, camera_info, args):
    full_params = expand_params_with_args(params, args)
    rotation_lidar_to_camera, translation_lidar_to_camera, _ = params_to_transform(full_params)
    residuals = []
    for observation in observations:
        normal_camera = observation["board_normal_camera"]
        center_camera = observation["board_center_camera"]
        lidar_points = observation["lidar_plane_points_xyz"]
        points_camera = transform_points_lidar_to_camera(
            lidar_points,
            rotation_lidar_to_camera,
            translation_lidar_to_camera,
        )
        signed_distances = (points_camera - center_camera.reshape(1, 3)).dot(normal_camera)
        residuals.extend((signed_distances * 20.0).tolist())

        lidar_normal_camera = rotation_lidar_to_camera.dot(observation["lidar_plane_normal_xyz"])
        normal_residual = np.cross(lidar_normal_camera, normal_camera) * 3.0
        residuals.extend(normal_residual.tolist())

        centroid_camera = (
            rotation_lidar_to_camera.dot(observation["lidar_plane_centroid_xyz"])
            + translation_lidar_to_camera
        )
        if centroid_camera[2] > 0.05:
            centroid_uv = cv2.projectPoints(
                np.asarray([centroid_camera], dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                np.array(
                    [
                        [camera_info["fx"], 0.0, camera_info["cx"]],
                        [0.0, camera_info["fy"], camera_info["cy"]],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    camera_info.get("distortion", np.zeros((5,), dtype=np.float32)),
                    dtype=np.float64,
                ).reshape(-1, 1),
            )[0].reshape(-1, 2)
            u = float(centroid_uv[0][0])
            v = float(centroid_uv[0][1])
        else:
            u = -1000.0
            v = -1000.0
        bbox_center = observation["bbox_center"]
        residuals.append((u - float(bbox_center[0])) / 60.0)
        residuals.append((v - float(bbox_center[1])) / 60.0)

    prior_origin = getattr(args, "prior_camera_origin_in_source_frame_xyz", None)
    if prior_origin is not None:
        camera_origin_in_lidar = -rotation_lidar_to_camera.T.dot(translation_lidar_to_camera)
        sigma = max(1e-6, float(args.prior_camera_origin_sigma_m))
        prior_residual = (camera_origin_in_lidar - np.asarray(prior_origin, dtype=np.float64)) / sigma
        residuals.extend(prior_residual.tolist())

    rotation_prior_sigma_deg = float(
        getattr(args, "prior_rotation_like_initial_sigma_deg", 0.0) or 0.0
    )
    initial_rpy_prior_deg = getattr(args, "_initial_rpy_prior_deg", None)
    if rotation_prior_sigma_deg > 0.0 and initial_rpy_prior_deg is not None:
        sigma = max(1e-6, rotation_prior_sigma_deg)
        prior_rpy = np.asarray(initial_rpy_prior_deg, dtype=np.float64)
        current_rpy = np.asarray(full_params[3:6], dtype=np.float64)
        wrapped_delta = (current_rpy - prior_rpy + 180.0) % 360.0 - 180.0
        residuals.extend((wrapped_delta / sigma).tolist())
    return np.asarray(residuals, dtype=np.float64)


def save_yaml(output_path, camera_info, source_frame, params, observations, iterations):
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    full_params = np.asarray(params, dtype=np.float64).reshape(-1)
    rotation_lidar_to_camera, translation_lidar_to_camera, quaternion_xyzw = params_to_transform(full_params)
    camera_origin_in_lidar = -rotation_lidar_to_camera.T.dot(translation_lidar_to_camera)
    baseline_distance_m = float(np.linalg.norm(camera_origin_in_lidar))
    content = []
    content.append("source_frame: {}".format(source_frame))
    content.append("target_frame: {}".format(camera_info["frame_id"]))
    content.append(
        "translation_xyz: [{:.6f}, {:.6f}, {:.6f}]".format(
            translation_lidar_to_camera[0],
            translation_lidar_to_camera[1],
            translation_lidar_to_camera[2],
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
            params[3], params[4], params[5]
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
    content.append("used_observations: {}".format(len(observations)))
    content.append("outer_iterations: {}".format(int(iterations)))
    content.append(
        "launch_override: >-\n  scene_detector_point_cloud_fallback_source_frame:={} scene_detector_point_cloud_fallback_target_frame:={} scene_detector_point_cloud_fallback_translation_xyz:='[{:.6f}, {:.6f}, {:.6f}]' scene_detector_point_cloud_fallback_rotation_xyzw:='[{:.6f}, {:.6f}, {:.6f}, {:.6f}]'".format(
            source_frame,
            camera_info["frame_id"],
            translation_lidar_to_camera[0],
            translation_lidar_to_camera[1],
            translation_lidar_to_camera[2],
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
    dictionary, board = build_charuco_board(args)
    camera_info, samples = collect_candidate_samples(args, board, dictionary)
    initial_roll, initial_pitch, initial_yaw = euler_from_quaternion(
        (args.qx, args.qy, args.qz, args.qw)
    )
    initial_full_params = np.array(
        [
            float(args.tx),
            float(args.ty),
            float(args.tz),
            math.degrees(initial_roll),
            math.degrees(initial_pitch),
            math.degrees(initial_yaw),
        ],
        dtype=np.float64,
    )
    args._initial_rpy_prior_deg = (
        float(math.degrees(initial_roll)),
        float(math.degrees(initial_pitch)),
        float(math.degrees(initial_yaw)),
    )
    if bool(args.lock_initial_rotation):
        params = initial_full_params[:3].copy()
    else:
        params = initial_full_params.copy()

    observations = []
    for iteration in range(max(1, int(args.outer_iterations))):
        use_geometry_filter = iteration >= max(0, int(args.geometry_filter_warmup_iterations))
        observations = build_optimization_observations(
            camera_info,
            samples,
            board,
            dictionary,
            args,
            params,
            enforce_board_geometry=use_geometry_filter,
        )
        if len(observations) < 3 and use_geometry_filter:
            observations = build_optimization_observations(
                camera_info,
                samples,
                board,
                dictionary,
                args,
                params,
                enforce_board_geometry=False,
            )
        print(
            "[auto-calib] iteration {} observations={} geometry_filter={}".format(
                iteration + 1,
                len(observations),
                "on" if use_geometry_filter else "off",
            )
        )
        if len(observations) < 3:
            raise RuntimeError(
                "자동 보정에 사용할 충분한 샘플을 만들지 못했습니다. "
                "체커보드가 더 크게 보이는 구간을 사용하거나 초기 extrinsic을 조정해 보세요."
            )
        result = least_squares(
            residual_function,
            params,
            args=(observations, camera_info, args),
            method="trf",
            loss="soft_l1",
            max_nfev=100,
            verbose=0,
        )
        params = result.x
        print(
            "[auto-calib] cost={:.6f} tx={:.4f} ty={:.4f} tz={:.4f} rpy=[{:.2f}, {:.2f}, {:.2f}]".format(
                float(result.cost),
                params[0],
                params[1],
                params[2],
                expand_params_with_args(params, args)[3],
                expand_params_with_args(params, args)[4],
                expand_params_with_args(params, args)[5],
            )
        )

    save_yaml(
        args.output,
        camera_info,
        samples[0]["cloud_frame_id"],
        expand_params_with_args(params, args),
        observations,
        args.outer_iterations,
    )
    print("saved automatic extrinsic to {}".format(args.output))


if __name__ == "__main__":
    main()
