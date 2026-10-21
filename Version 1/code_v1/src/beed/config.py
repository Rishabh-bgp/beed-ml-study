"""Global configuration: paths, seeds and constants."""
from pathlib import Path
import os

ROOT = Path(os.environ.get("BEED_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data" / "BEED_Data.csv"
EXTERNAL = ROOT / "data" / "external"
BONN_CSV = EXTERNAL / "bonn_uci_epileptic_seizure_recognition.csv"
# Copy published by the creators of the UCI 'Epileptic Seizure Recognition' file
BONN_URL = ("https://raw.githubusercontent.com/QiuyiWu/Epileptic-Seizure-Recognition-Data/"
            "master/A%26B%26C%26D%26E.csv")
BONN_SHA256 = "4b3f6024ea24a864c0de51b2ba477bf8b9e4974e8cc01a00dcf45bfc59d48deb"
FIG = ROOT / "figures"
RES = ROOT / "results"

SEED = 42
N_JOBS = int(os.environ.get("BEED_N_JOBS", "8"))
CLASS_NAMES = ["Healthy (0)", "Generalised (1)", "Focal (2)", "Seizure events (3)"]
SHORT_NAMES = ["H", "GS", "FS", "SE"]
FS_BEED = 256.0      # sampling rate stated by UCI / dataset paper (Hz)
FS_BONN = 173.61     # sampling rate of the Bonn data (Hz)
BLOCK = 200          # rows per contiguous block for classes 1-3 (0.78 s at 256 Hz)
GAP = 50             # purge gap (rows) between training and test blocks (0.195 s)
N_SPLITS, N_REPEATS = 5, 2

for p in (FIG, RES, EXTERNAL):
    p.mkdir(parents=True, exist_ok=True)
