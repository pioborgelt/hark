#!/usr/bin/env python3
"""Home Assistant OS Add-on entrypoint translating /data/options.json into wyoming-hark CLI flags."""
import json
import os
from pathlib import Path

options_path = Path("/data/options.json")
opts = json.loads(options_path.read_text()) if options_path.is_file() else {}

phrases = opts.get("phrases") or ["Okay Haus:de", "Hey Jarvis:en"]
sensitivity = str(opts.get("sensitivity", "balanced"))
threads = str(opts.get("threads", 2))
log_near = bool(opts.get("log_near_misses", True))
save_clips = bool(opts.get("save_debug_clips", False))

os.environ.setdefault("HARK_CACHE_DIR", "/data/models")

argv = [
    "wyoming-hark",
    "--uri",
    "tcp://0.0.0.0:10400",
    "--sensitivity",
    sensitivity,
    "--threads",
    threads,
]
for p in phrases:
    if str(p).strip():
        argv.extend(["--phrase", str(p).strip()])
if not log_near:
    argv.append("--no-near-miss-logs")
if save_clips:
    argv.extend(["--debug-clip-dir", "/data/debug_clips"])

os.execvp("wyoming-hark", argv)
