#!/usr/bin/env python3
import argparse
import math
import os

import cv2
import rosbag
import sensor_msgs.point_cloud2 as point_cloud2
from cv_bridge import CvBridge


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
        "--point-cloud-topic",
        default="/planning/linefit_ground/non_ground_cloud",
    )
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--image-stride", type=int, default=80)
    parser.add_argument("--max-sync-dt-s", type=float, default=0.12)
    parser.add_argument("--max-cloud-points", type=int, default=4500)
    parser.add_argument("--forward-m", type=float, default=12.0)
    parser.add_argument("--rear-m", type=float, default=2.0)
    parser.add_argument("--half-width-m", type=float, default=8.0)
    parser.add_argument("--height-abs-m", type=float, default=2.5)
    parser.add_argument("--max-range-m", type=float, default=12.0)
    parser.add_argument("--point-radius-px", type=int, default=2)
    parser.add_argument("--tx", type=float, default=0.0)
    parser.add_argument("--ty", type=float, default=-0.05913)
    parser.add_argument("--tz", type=float, default=0.0)
    parser.add_argument("--qx", type=float, default=-0.5)
    parser.add_argument("--qy", type=float, default=0.5)
    parser.add_argument("--qz", type=float, default=-0.5)
    parser.add_argument("--qw", type=float, default=0.5)
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
    estimated = max(1, width)
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


def load_samples(args):
    bridge = CvBridge()
    samples = []
    latest_cloud = None
    camera_info = None
    accepted_images = 0
    bag = rosbag.Bag(args.bag)
    try:
        for topic, msg, _ in bag.read_messages(
            topics=[args.camera_info_topic, args.point_cloud_topic, args.image_topic]
        ):
            if topic == args.camera_info_topic and camera_info is None:
                k_values = list(msg.K)
                camera_info = {
                    "frame_id": str(msg.header.frame_id or ""),
                    "width": int(msg.width or 0),
                    "height": int(msg.height or 0),
                    "fx": float(k_values[0] or 0.0),
                    "fy": float(k_values[4] or 0.0),
                    "cx": float(k_values[2] or 0.0),
                    "cy": float(k_values[5] or 0.0),
                }
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
            samples.append(
                {
                    "stamp": image_stamp,
                    "image_bgr": image_bgr,
                    "cloud_frame_id": latest_cloud["frame_id"],
                    "points_xyz": list(latest_cloud["points_xyz"]),
                }
            )
            if len(samples) >= max(1, args.max_samples):
                break
    finally:
        bag.close()

    if camera_info is None:
        raise RuntimeError("camera_info를 bag에서 찾지 못했습니다.")
    if not samples:
        raise RuntimeError("동기화된 image/point_cloud 샘플을 만들지 못했습니다.")
    return camera_info, samples


def render_projection(sample, camera_info, state, args):
    image_bgr = sample["image_bgr"].copy()
    fx = float(camera_info["fx"])
    fy = float(camera_info["fy"])
    cx = float(camera_info["cx"])
    cy = float(camera_info["cy"])
    image_h, image_w = image_bgr.shape[:2]

    roll_rad = math.radians(state["roll_deg"])
    pitch_rad = math.radians(state["pitch_deg"])
    yaw_rad = math.radians(state["yaw_deg"])
    quaternion_xyzw = quaternion_from_euler(roll_rad, pitch_rad, yaw_rad)
    translation = (state["tx"], state["ty"], state["tz"])

    projected_count = 0
    for point_xyz in sample["points_xyz"]:
        px, py, pz = point_xyz
        if px < (-args.rear_m) or px > args.forward_m:
            continue
        if abs(py) > args.half_width_m or abs(pz) > args.height_abs_m:
            continue

        rotated = rotate_point_by_quaternion(point_xyz, quaternion_xyzw)
        camera_x = rotated[0] + translation[0]
        camera_y = rotated[1] + translation[1]
        camera_z = rotated[2] + translation[2]
        if camera_z <= 0.10 or camera_z >= args.max_range_m:
            continue

        u = (fx * (camera_x / camera_z)) + cx
        v = (fy * (camera_y / camera_z)) + cy
        if u < 0.0 or u >= float(image_w) or v < 0.0 or v >= float(image_h):
            continue
        projected_count += 1
        cv2.circle(
            image_bgr,
            (int(round(u)), int(round(v))),
            max(1, int(args.point_radius_px)),
            (255, 255, 0),
            -1,
            lineType=cv2.LINE_AA,
        )

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (8, 8), (image_w - 8, 150), (0, 0, 0), -1)
    image_bgr = cv2.addWeighted(overlay, 0.35, image_bgr, 0.65, 0.0)

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
        "step t={:.3f}m r={:.2f}deg | n/p sample | s save | q quit".format(
            state["translation_step_m"], state["rotation_step_deg"]
        ),
        "1/2:x-+ 3/4:y-+ 5/6:z-+  u/o:roll-+  i/k:pitch-+  j/l:yaw-+  [-] step down  [=] step up",
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
    return image_bgr, quaternion_xyzw


def save_yaml(output_path, camera_info, sample, state, quaternion_xyzw):
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
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
    }

    window_name = "camera_lidar_extrinsic_tuner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1440, 900)

    while True:
        sample = samples[state["sample_index"]]
        rendered, quaternion_xyzw = render_projection(
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
