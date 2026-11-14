"""Export slim, weights-only-loadable release checkpoints and print their SHA256.

Usage (from the repo root):
    python scripts/export_release_weights.py \
        --stage1 artifacts/r221-cleanhybrid-20261113-071351/h384-step200000.pt \
        --stage2 artifacts/r222-verifier-20261113-071450/v384-step055000.pt

Writes `models/<release filename>` for both stages. Upload exactly these files as
GitHub Release assets and keep the printed hashes in `hark.streaming.MODEL_SPECS`.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from hark.streaming import MODEL_SPECS

# Training-run arguments are not needed for inference and are not part of the release.
DROP_KEYS = {"args", "best"}


def export(src: Path, dest: Path) -> str:
    ck = torch.load(src, map_location="cpu", weights_only=True)
    slim = {k: v for k, v in ck.items() if k not in DROP_KEYS}
    slim["model"] = {k: t.detach().contiguous().clone() for k, t in ck["model"].items()}
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dest)
    # The release file must load without pickled code.
    torch.load(dest, map_location="cpu", weights_only=True)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage1", required=True, type=Path)
    p.add_argument("--stage2", required=True, type=Path)
    p.add_argument("--out-dir", default=Path("models"), type=Path)
    args = p.parse_args()
    for key, src in (("stage1", args.stage1), ("stage2", args.stage2)):
        dest = args.out_dir / MODEL_SPECS[key]["filename"]
        digest = export(src, dest)
        pinned = MODEL_SPECS[key].get("sha256")
        state = "matches pinned hash" if digest == pinned else "UPDATE MODEL_SPECS sha256"
        print(f"{key}: {dest} sha256={digest} ({state})")


if __name__ == "__main__":
    main()
