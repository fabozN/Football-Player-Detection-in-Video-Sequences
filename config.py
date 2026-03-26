import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SEQUENCES_DIR = os.path.join(DATA_DIR, "sequences")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")

FORCE_FPS = None
SAVE_FRAMES = True

RUN_STEP_1 = True
RUN_STEP_2 = True
