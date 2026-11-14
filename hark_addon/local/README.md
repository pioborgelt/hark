# Local build inputs (development only)

Published add-on builds leave this directory empty and install Hark from the pinned
GitHub tag, then download the pinned release weights on first start.

For testing before a release, `scripts/stage_local_addon.sh` places a locally built
`hark_ww-*.whl` in `wheels/` and the exported release checkpoints in `models/`.
The image then installs the wheel instead of the Git tag, and `run.py` seeds the model
cache from `models/`; the pinned SHA256 is still verified. Never commit these files.
