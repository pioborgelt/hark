#!/usr/bin/env bash
# Build a self-contained Home Assistant add-on repository for testing before a release.
#
#   scripts/stage_local_addon.sh OUT_DIR [MODELS_DIR]
#
# OUT_DIR becomes a Git repository (repository.json + hark_addon/) that bundles a wheel of
# the current working tree and the exported release weights (default: models/, created by
# scripts/export_release_weights.py). Serve it read-only, e.g.
#   git daemon --base-path="$(dirname OUT_DIR)" --export-all --reuseaddr
# and add git://<host>/<name> as add-on repository in Home Assistant.
set -euo pipefail

out="${1:?usage: $0 OUT_DIR [MODELS_DIR]}"
models="${2:-models}"
root="$(cd "$(dirname "$0")/.." && pwd)"

for name in hark-stage1-r221-h384.pt hark-stage2-r222-v384.pt; do
    [[ -f "$models/$name" ]] || { echo "missing $models/$name (run scripts/export_release_weights.py)" >&2; exit 1; }
done

rm -rf "$out"
mkdir -p "$out/hark_addon/local/wheels" "$out/hark_addon/local/models"
cp "$root/repository.json" "$out/"
cp -r "$root/hark_addon/." "$out/hark_addon/"
cp "$models"/hark-stage*.pt "$out/hark_addon/local/models/"
python3 -m pip wheel --quiet --no-deps --wheel-dir "$out/hark_addon/local/wheels" "$root"

git -C "$out" init --quiet --initial-branch=main
git -C "$out" add -A
git -C "$out" -c user.name=hark-dev -c user.email=hark-dev@localhost \
    commit --quiet -m "Local Hark add-on build $(date -u +%Y-%m-%dT%H:%MZ)"
echo "Staged add-on repository in $out"
