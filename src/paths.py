import os 
from pathlib import Path 

PARENT_DIR = Path(__file__).parent.resolve().parent

DATA_DIR = PARENT_DIR / "data"
MODEL_DIR = PARENT_DIR / "model_artifact"

if not DATA_DIR.exists():
    os.mkdir(DATA_DIR)
    
if not MODEL_DIR.exists():
    os.mkdir(MODEL_DIR)


