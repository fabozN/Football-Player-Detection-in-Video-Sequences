from config import (
    RUN_STEP_1, RUN_STEP_2,
    SEQUENCES_DIR
)

from src.step1_video_io import step1_extract_frames
from src.step2_player_detection_yolo_roi import step2_player_detection

import os

def run_pipeline(video_name):
    video_path = os.path.join(SEQUENCES_DIR, video_name) 
    print(f"\n[INFO] Video selezionato: {video_path}\n")
    sequence_name = video_name.replace(".mp4", "")

    print("\n=== PIPELINE START ===\n")

    if RUN_STEP_1:
        print("[STEP 1] Estrazione frame + FPS")
        fps, total_frames = step1_extract_frames(video_path)
    else:
        fps, total_frames = None, None

    if RUN_STEP_2:
        print("[STEP 2] Player Detection")
        detections = step2_player_detection(sequence_name)
        
    else:
        detections = None    

    return {
        "fps": fps,
        "total_frames": total_frames,
        "detections": detections,
    }


if __name__ == "__main__":

    run_pipeline("sequence_move.mp4")
        