#!/usr/bin/python3
import argparse
import math
import os
import sys

import numpy as np
import rosbag

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError as exc:
    raise SystemExit(
        "필수 모듈을 불러오지 못했습니다: {}.\n"
        "이 스크립트는 ROS OpenCV가 설치된 시스템 Python으로 실행해야 합니다.\n"
        "다음처럼 다시 실행해 주세요:\n"
        "  /usr/bin/python3 src/xai_driving_explainer/scripts/calibrate_camera_intrinsic_charuco.py ...\n"
        "현재 interpreter: {}".format(exc, sys.executable)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate camera intrinsic parameters from a ChArUco ROS bag."
    )
    parser.add_argument("--bag", required=True)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--checker-size-mm", type=float, default=25.0)
    parser.add_argument("--marker-size-mm", type=float, default=18.75)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument("--image-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--min-charuco-corners", type=int, default=12)
    parser.add_argument("--use-camera-info-seed", action="store_true", default=True)
    parser.add_argument("--no-camera-info-seed", action="store_true")
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/camera_intrinsic_charuco.yaml"),
    )
    return parser.parse_args()


def build_charuco_board(args):
    dictionary_id = getattr(cv2.aruco, str(args.dictionary), None)
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
        "distortion": np.array(list(msg.D), dtype=np.float64),
    }


def collect_charuco_samples(args, board, dictionary):
    bridge = CvBridge()
    detector_params = cv2.aruco.DetectorParameters_create()
    camera_info = None
    charuco_corners = []
    charuco_ids = []
    image_size = None
    kept = []
    accepted_images = 0

    bag = rosbag.Bag(args.bag)
    try:
        for topic, msg, _ in bag.read_messages(topics=[args.camera_info_topic, args.image_topic]):
            if topic == args.camera_info_topic and camera_info is None:
                camera_info = camera_info_from_msg(msg)
                continue

            if topic != args.image_topic:
                continue

            accepted_images += 1
            if ((accepted_images - 1) % max(1, int(args.image_stride))) != 0:
                continue

            image_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])

            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)
            marker_count = 0 if ids is None else int(len(ids))
            if marker_count <= 0:
                continue

            retval, interpolated_corners, interpolated_ids = cv2.aruco.interpolateCornersCharuco(
                corners,
                ids,
                gray,
                board,
            )
            detected_charuco = 0 if interpolated_ids is None else int(len(interpolated_ids))
            if detected_charuco < max(4, int(args.min_charuco_corners)):
                continue

            charuco_corners.append(interpolated_corners)
            charuco_ids.append(interpolated_ids)
            kept.append(
                {
                    "stamp": float(msg.header.stamp.to_sec()),
                    "marker_count": marker_count,
                    "charuco_count": detected_charuco,
                }
            )
            if len(charuco_corners) >= max(1, int(args.max_frames)):
                break
    finally:
        bag.close()

    if image_size is None:
        raise RuntimeError("image 토픽에서 프레임을 읽지 못했습니다.")
    if not charuco_corners:
        raise RuntimeError("ChArUco 코너를 충분히 검출한 프레임을 찾지 못했습니다.")

    return camera_info, image_size, charuco_corners, charuco_ids, kept


def calibrate_intrinsic(camera_info, image_size, charuco_corners, charuco_ids, board, use_seed):
    camera_matrix = None
    distortion = None
    flags = 0

    if camera_info is not None and use_seed:
        camera_matrix = np.array(
            [
                [camera_info["fx"], 0.0, camera_info["cx"]],
                [0.0, camera_info["fy"], camera_info["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.array(camera_info["distortion"], dtype=np.float64).reshape(-1, 1)
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS

    result = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=charuco_corners,
        charucoIds=charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=camera_matrix,
        distCoeffs=distortion,
        flags=flags,
    )

    reproj_error = float(result[0])
    calibrated_matrix = np.asarray(result[1], dtype=np.float64)
    calibrated_distortion = np.asarray(result[2], dtype=np.float64).reshape(-1)
    per_view_errors = np.asarray(result[-1], dtype=np.float64).reshape(-1)
    return {
        "reprojection_error": reproj_error,
        "camera_matrix": calibrated_matrix,
        "distortion": calibrated_distortion,
        "per_view_errors": per_view_errors,
    }


def save_yaml(output_path, camera_info, image_size, calibration, kept_samples, args):
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    frame_id = "camera_color_optical_frame"
    if camera_info is not None and camera_info.get("frame_id"):
        frame_id = str(camera_info["frame_id"])

    camera_matrix = calibration["camera_matrix"]
    distortion = calibration["distortion"]
    per_view_errors = calibration["per_view_errors"]

    lines = []
    lines.append("frame_id: {}".format(frame_id))
    lines.append("image_width: {}".format(int(image_size[0])))
    lines.append("image_height: {}".format(int(image_size[1])))
    lines.append("model: pinhole")
    lines.append("dictionary: {}".format(args.dictionary))
    lines.append("charuco_rows: {}".format(int(args.rows)))
    lines.append("charuco_columns: {}".format(int(args.columns)))
    lines.append("checker_size_mm: {:.6f}".format(float(args.checker_size_mm)))
    lines.append("marker_size_mm: {:.6f}".format(float(args.marker_size_mm)))
    lines.append("used_frames: {}".format(len(kept_samples)))
    lines.append("reprojection_error: {:.9f}".format(float(calibration["reprojection_error"])))
    lines.append(
        "camera_matrix: [{:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}]".format(
            camera_matrix[0, 0],
            camera_matrix[0, 1],
            camera_matrix[0, 2],
            camera_matrix[1, 0],
            camera_matrix[1, 1],
            camera_matrix[1, 2],
            camera_matrix[2, 0],
            camera_matrix[2, 1],
            camera_matrix[2, 2],
        )
    )
    distortion_text = ", ".join("{:.9f}".format(float(value)) for value in distortion.tolist())
    lines.append("distortion_coefficients: [{}]".format(distortion_text))
    lines.append(
        "intrinsic_override: >-\n  --camera-frame {} --camera-width {} --camera-height {} --fx {:.9f} --fy {:.9f} --cx {:.9f} --cy {:.9f}".format(
            frame_id,
            int(image_size[0]),
            int(image_size[1]),
            camera_matrix[0, 0],
            camera_matrix[1, 1],
            camera_matrix[0, 2],
            camera_matrix[1, 2],
        )
    )
    if len(per_view_errors) > 0:
        mean_view_error = float(np.mean(per_view_errors))
        max_view_error = float(np.max(per_view_errors))
        lines.append("mean_per_view_error: {:.9f}".format(mean_view_error))
        lines.append("max_per_view_error: {:.9f}".format(max_view_error))
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.no_camera_info_seed:
        args.use_camera_info_seed = False

    dictionary, board = build_charuco_board(args)
    camera_info, image_size, charuco_corners, charuco_ids, kept_samples = collect_charuco_samples(
        args,
        board,
        dictionary,
    )
    calibration = calibrate_intrinsic(
        camera_info,
        image_size,
        charuco_corners,
        charuco_ids,
        board,
        bool(args.use_camera_info_seed),
    )

    print(
        "[intrinsic] frames={} reprojection_error={:.6f}".format(
            len(kept_samples),
            calibration["reprojection_error"],
        )
    )
    print(
        "[intrinsic] fx={:.6f} fy={:.6f} cx={:.6f} cy={:.6f}".format(
            calibration["camera_matrix"][0, 0],
            calibration["camera_matrix"][1, 1],
            calibration["camera_matrix"][0, 2],
            calibration["camera_matrix"][1, 2],
        )
    )
    print(
        "[intrinsic] dist={}".format(
            [round(float(value), 9) for value in calibration["distortion"].tolist()]
        )
    )
    save_yaml(args.output, camera_info, image_size, calibration, kept_samples, args)
    print("saved intrinsic calibration to {}".format(args.output))


if __name__ == "__main__":
    main()
