#!/usr/bin/env python3
import argparse
import base64
import json
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-track", action="store_true")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    return parser.parse_args()


def resolve_device(raw_device):
    if str(raw_device).strip().lower() == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    return raw_device


def emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def decode_image(image_jpeg_b64):
    binary = base64.b64decode(image_jpeg_b64.encode("ascii"))
    array = np.frombuffer(binary, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode jpeg bytes")
    return image


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = YOLO(args.model)
    emit({"type": "ready", "model": args.model, "device": str(device)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:
            emit({"type": "error", "message": "invalid request json: {}".format(str(exc))})
            continue

        if request.get("type") == "shutdown":
            break
        if request.get("type") != "infer":
            emit({"type": "error", "message": "unsupported request type"})
            continue

        request_id = request.get("request_id")
        try:
            image = decode_image(request["image_jpeg_b64"])
            tracking_active = bool(args.use_track)
            tracking_fallback_reason = None
            if tracking_active:
                try:
                    results = model.track(
                        source=image,
                        conf=args.conf,
                        iou=args.iou,
                        max_det=args.max_det,
                        imgsz=args.imgsz,
                        device=device,
                        tracker=args.tracker,
                        persist=True,
                        verbose=False,
                    )
                except Exception as track_exc:
                    tracking_active = False
                    tracking_fallback_reason = str(track_exc)
                    results = model.predict(
                        source=image,
                        conf=args.conf,
                        iou=args.iou,
                        max_det=args.max_det,
                        imgsz=args.imgsz,
                        device=device,
                        verbose=False,
                    )
            else:
                results = model.predict(
                    source=image,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    imgsz=args.imgsz,
                    device=device,
                    verbose=False,
                )
            detections = []
            result = results[0]
            if result.boxes is not None:
                names = result.names
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                conf = result.boxes.conf.detach().cpu().numpy()
                cls = result.boxes.cls.detach().cpu().numpy()
                track_ids = None
                if getattr(result.boxes, "id", None) is not None:
                    track_ids = result.boxes.id.detach().cpu().numpy()
                for idx, (box, score, cls_idx) in enumerate(zip(xyxy, conf, cls)):
                    track_id = None
                    if track_ids is not None and idx < len(track_ids):
                        try:
                            track_id = int(track_ids[idx])
                        except Exception:
                            track_id = None
                    detections.append(
                        {
                            "label": str(names.get(int(cls_idx), int(cls_idx))),
                            "confidence": round(float(score), 4),
                            "bbox_xyxy": [float(v) for v in box.tolist()],
                            "track_id": track_id,
                        }
                    )
            emit(
                {
                    "type": "result",
                    "request_id": request_id,
                    "detections": detections,
                    "tracking_active": tracking_active,
                    "tracking_fallback_reason": tracking_fallback_reason,
                }
            )
        except Exception as exc:
            emit(
                {
                    "type": "error",
                    "request_id": request_id,
                    "message": str(exc),
                }
            )


if __name__ == "__main__":
    main()
