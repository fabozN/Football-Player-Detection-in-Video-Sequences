import os
import cv2
import numpy as np

CONFIG = {
    "fixed": {
        "lower_green": np.array([25, 55, 50]),
        "upper_green": np.array([95, 255, 255]),

        "morph": {
            "kernel_open": (3,3),
            "kernel_close_field": (7,7),
            "kernel_expand": (9,9),
            "kernel_close_final": (9,9),

            "apply_open_field": True,
            "apply_close_field": True,
            "apply_expand_roi": True,
            "apply_open_motion": True,
            "apply_close_motion": True,
            "apply_close_final": True,
        },

        "min_area_factor": 0.00015,
        "max_area_factor": 0.01,
        "min_height": 10,
        "aspect_ratio_min": 0.4,
        "aspect_ratio_max": 6.0,
    },

    "mobile": {
        "lower_green": np.array([35, 55, 50]),
        "upper_green": np.array([85, 255, 255]),

        "morph": {
            "kernel_open": (3,3),
            "kernel_close_field": (7,7),
            "kernel_expand": (9,9),
            "kernel_close_final": (9,9),

            "apply_open_field": False,
            "apply_close_field": False,
            "apply_expand_roi": True,
            "apply_open_motion": True,
            "apply_close_motion": True,
            "apply_close_final": True,
        },

        "min_area_factor": 0.000015,
        "max_area_factor": 0.01,
        "min_height": 10,
        "aspect_ratio_min": 0.4,
        "aspect_ratio_max": 6.0,
    }
}

def get_config(sequence_name):
    if "move" in sequence_name.lower():
        return CONFIG["mobile"]
    return CONFIG["fixed"]

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=False
)

def create_folders(detections_dir, debug_dir):
    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "01_mask_after_morph"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "02_bboxes"), exist_ok=True)

def detect_players(frame_bgr, cfg, debug_dir, frame_name=None, save_debug=False):

    hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    morph = cfg["morph"]

    mask_green = cv2.inRange(hsv_frame, cfg["lower_green"], cfg["upper_green"])
    mask_field = mask_green.copy()

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph["kernel_open"])
    kernel_close_field = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph["kernel_close_field"])

    if morph["apply_open_field"]:
        mask_field = cv2.morphologyEx(mask_field, cv2.MORPH_OPEN, kernel_open)
    if morph["apply_close_field"]:
        mask_field = cv2.morphologyEx(mask_field, cv2.MORPH_CLOSE, kernel_close_field)

    contours_field, _ = cv2.findContours(mask_field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_field) == 0:
        return []
    largest_contour = max(contours_field, key=cv2.contourArea)

    field_mask_clean = np.zeros_like(mask_field)
    cv2.drawContours(field_mask_clean, [largest_contour], -1, 255, thickness=cv2.FILLED)

    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph["kernel_expand"])
    if morph["apply_expand_roi"]:
        field_mask_expanded = cv2.morphologyEx(field_mask_clean, cv2.MORPH_DILATE, kernel_expand)
    else:
        field_mask_expanded = field_mask_clean.copy()

    mask_motion = fgbg.apply(frame_bgr)
    if morph["apply_open_motion"]:
        mask_motion = cv2.morphologyEx(mask_motion, cv2.MORPH_OPEN, kernel_open)
    if morph["apply_close_motion"]:
        mask_motion = cv2.morphologyEx(mask_motion, cv2.MORPH_CLOSE, kernel_open)

    mask_not_green = cv2.bitwise_not(mask_green)
    mask_players = cv2.bitwise_and(mask_motion, mask_not_green)
    mask_players = cv2.bitwise_and(mask_players, field_mask_expanded)

    kernel_close_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph["kernel_close_final"])
    if morph["apply_close_final"]:
        mask_players = cv2.morphologyEx(mask_players, cv2.MORPH_CLOSE, kernel_close_final)

    if save_debug:
        cv2.imwrite(os.path.join(debug_dir, "01_mask_after_morph", frame_name), mask_players)

    contours, _ = cv2.findContours(mask_players, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    H, W = hsv_frame.shape[:2]
    min_area = (H * W) * cfg["min_area_factor"]
    max_area = (H * W) * cfg["max_area_factor"]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = h / float(w)
        if aspect_ratio < cfg["aspect_ratio_min"] or aspect_ratio > cfg["aspect_ratio_max"]:
            continue

        if h < cfg["min_height"]:
            continue

        bboxes.append((x, y, w, h))

    if save_debug:
        debug_frame = frame_bgr.copy()
        for (x, y, w, h) in bboxes:
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, "02_bboxes", frame_name), debug_frame)

    return bboxes

def step2_player_detection(sequence_name, save_debug=True, show_live=False):

    cfg = get_config(sequence_name)

    FRAMES_DIR = f"data/frames/{sequence_name}"
    DETECTIONS_DIR = f"data/detections/mog2_doppio_filtro/{sequence_name}"
    DEBUG_DIR = f"debugs/debug_step2/mog2_doppio_filtro/{sequence_name}"

    create_folders(DETECTIONS_DIR, DEBUG_DIR)

    frame_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
    all_detections = {}

    for fname in frame_files:

        frame_bgr = cv2.imread(os.path.join(FRAMES_DIR, fname))
        debug_name = fname.replace(".jpg", ".png")

        bboxes = detect_players(frame_bgr, cfg, DEBUG_DIR, frame_name=debug_name, save_debug=save_debug)

        all_detections[fname] = bboxes
        np.save(os.path.join(DETECTIONS_DIR, fname.replace(".jpg", "_bboxes.npy")), bboxes)

        if show_live:
            debug_frame = frame_bgr.copy()
            for (x, y, w, h) in bboxes:
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("Player Detection", debug_frame)
            k = cv2.waitKey(5) & 0xFF
            if k == 32:
                cv2.waitKey(0)
            if k == 27:
                break

    if show_live:
        cv2.destroyAllWindows()

    print("Player detection completed.")

    return all_detections
