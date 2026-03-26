import cv2
import os
from config import FRAMES_DIR, SAVE_FRAMES, FORCE_FPS

def step1_extract_frames(video_path):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire il video: {video_path}")

    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frame dichiarati nei metadata:", declared)

    real = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        real += 1

    print("Frame realmente leggibili:", real)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire il video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if FORCE_FPS is not None:
        fps = FORCE_FPS

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sequence_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(FRAMES_DIR, sequence_name)
    os.makedirs(frames_dir, exist_ok=True)

    if SAVE_FRAMES:
        os.makedirs(FRAMES_DIR, exist_ok=True)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)

            frame_idx += 1

    cap.release()

    return fps, total_frames