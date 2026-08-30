"""
export_preprocessed.py — Save preprocessed MEG trials to icaed_Sai/.

For each (subject, session, poem×lis condition), calls _load_meg_trial
(resample 10×, ×1e12, z-score) and saves the result as a float32 .npy file.
Skips any (subject, session, condition) where the source .fif does not exist.

Output structure mirrors icaed/:
    icaed_Sai/{subject}/ses-{session}/meg/{subject}_sess-{session}_task-{condition}.npy
    shape: (155, T_downsampled)  float32, already z-scored

Usage (from llm_decoder/):
    python -m unified.data.export_preprocessed
    python -m unified.data.export_preprocessed --out_dir /path/to/icaed_Sai
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent))

from unified.data.base_dataset import _load_meg_trial
from unified.data.splits import SUBJECTS

POEMS     = ["poem1", "poem2"]
SESSIONS  = list(range(10))
DEFAULT_OUT = Path("/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed_Sai")


def export(out_dir: Path) -> None:
    conditions = [f"{poem}lis" for poem in POEMS]
    total = saved = skipped_missing = skipped_exists = 0

    for subject in SUBJECTS:
        for session in SESSIONS:
            for condition in conditions:
                total += 1
                out_path = (out_dir / subject / f"ses-{session}" / "meg"
                            / f"{subject}_sess-{session}_task-{condition}.npy")

                if out_path.exists():
                    skipped_exists += 1
                    continue

                data = _load_meg_trial(subject, condition, session)
                if data is None:
                    print(f"  [skip] {subject}  ses-{session}  {condition}  (no .fif)")
                    skipped_missing += 1
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(out_path), data)
                saved += 1
                print(f"  [save] {subject}  ses-{session}  {condition}  shape={data.shape}")

    print(f"\nDone.  total={total}  saved={saved}  "
          f"skipped_missing={skipped_missing}  skipped_exists={skipped_exists}")


def main():
    p = argparse.ArgumentParser(description="Export preprocessed MEG trials to .npy")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT),
                   help="Root output directory (default: icaed_Sai alongside icaed)")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    print(f"Exporting preprocessed MEG → {out_dir}\n")
    export(out_dir)


if __name__ == "__main__":
    main()
