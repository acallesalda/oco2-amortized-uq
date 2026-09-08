# computes mean/std for the 20-d true profile columns on the *train* split
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN_DIR = Path("/workspace/project/scratch/simulated/files/train")
CSV = TRAIN_DIR / "aux_data.csv"

TRUE_COLS = [f"true_state_vector_{i:02d}" for i in range(1, 21)]

df = pd.read_csv(CSV)
X = df[TRUE_COLS].astype("float64").to_numpy()

mu = X.mean(axis=0).astype(np.float64)
sd = X.std(axis=0, ddof=0).astype(np.float64)
sd[sd == 0.0] = 1.0  # guard

out_dir = Path("/workspace/project/experiments/pipelines/stats")
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "profile_mu.npy", mu)
np.save(out_dir / "profile_sigma.npy", sd)

print("Saved:")
print(out_dir / "profile_mu.npy")
print(out_dir / "profile_sigma.npy")
print("μ[:3] =", np.array2string(mu[:3], precision=10))
print("σ[:3] =", np.array2string(sd[:3], precision=10))