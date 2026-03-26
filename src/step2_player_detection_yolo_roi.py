import os
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

def create_folders(detections_dir, debug_dir):
    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "01_field_hull"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "02_bboxes"), exist_ok=True)


def compute_field_hull(frame_bgr):

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 55, 50])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)

    return hull

def detect_players_yolo(frame_bgr, conf_th=0.4, imgsz=1280, iou_th=0.5):

    results = model(frame_bgr, imgsz=imgsz, conf=conf_th, iou=iou_th)[0]

    bboxes = []
    for box in results.boxes:
        cls = int(box.cls)
        if cls == 0:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            bboxes.append([float(x1), float(y1), float(x2), float(y2), conf])

    return np.array(bboxes, dtype=np.float32)

def filter_boxes_with_hull(bboxes, hull):

    filtered = []

    for (x1, y1, x2, y2, conf) in bboxes:

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        inside = cv2.pointPolygonTest(hull, (cx, cy), False)
        if inside >= 0:
            filtered.append([x1, y1, x2, y2, conf])

    return np.array(filtered, dtype=np.float32)

def step2_player_detection(sequence_name, save_debug=True, show_live=False):

    FRAMES_DIR = f"data/frames/{sequence_name}"
    DETECTIONS_OUT_DIR = f"data/detections/yolo_roi/{sequence_name}"
    DEBUG_DIR = f"debugs/debug_step2/yolo_roi/{sequence_name}"

    create_folders(DETECTIONS_OUT_DIR, DEBUG_DIR)

    frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
    all_detections = {}

    for fname in frame_files:

        frame_bgr = cv2.imread(os.path.join(FRAMES_DIR, fname))
        debug_name = fname.replace(".jpg", ".png")

        hull = compute_field_hull(frame_bgr)
        bboxes = detect_players_yolo(frame_bgr, conf_th=0.4, imgsz=1920, iou_th=0.5)
        if hull is not None:
            bboxes = filter_boxes_with_hull(bboxes, hull)

        all_detections[fname] = bboxes
        np.save(
            os.path.join(DETECTIONS_OUT_DIR, fname.replace(".jpg", "_bboxes.npy")),
            bboxes
        )

        if save_debug and hull is not None:
            debug_field = frame_bgr.copy()
            cv2.drawContours(debug_field, [hull], -1, (0,255,255), 3)
            cv2.imwrite(
                os.path.join(DEBUG_DIR, "01_field_hull", debug_name),
                debug_field
            )

        if save_debug:
            debug_frame = frame_bgr.copy()
            for (x1, y1, x2, y2, conf) in bboxes:
                cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(debug_frame, f"{conf:.2f}", (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imwrite(
                os.path.join(DEBUG_DIR, "02_bboxes", debug_name),
                debug_frame
            )

        if show_live:
            live_frame = frame_bgr.copy()
            for (x1, y1, x2, y2, conf) in bboxes:
                cv2.rectangle(live_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(live_frame, f"{conf:.2f}", (int(x1), int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow("YOLO Player Detection", live_frame)
            k = cv2.waitKey(5) & 0xFF
            if k == 32:
                cv2.waitKey(0)
            if k == 27:
                break

    if show_live:
        cv2.destroyAllWindows()

    print("YOLO player detection completed.")

    return all_detections
