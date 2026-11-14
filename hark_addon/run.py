#!/usr/bin/env python3
"""Home Assistant add-on entrypoint.

Translates /data/options.json into wyoming-hark flags, starts the server and, once it
accepts connections, registers it with the Supervisor so Home Assistant offers the
Wyoming integration automatically ("Discovered").
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = 10400
OPTIONS = Path("/data/options.json")
LOCAL_MODELS = Path("/opt/hark-local/models")


def log(msg: str) -> None:
    print(f"[hark-addon] {msg}", flush=True)


def build_argv(opts: dict) -> list[str]:
    phrases = opts.get("phrases") or ["Okay Haus:de", "Hey Jarvis:en"]
    argv = [
        "wyoming-hark",
        "--uri", f"tcp://0.0.0.0:{PORT}",
        "--sensitivity", str(opts.get("sensitivity", "balanced")),
        "--threads", str(opts.get("threads", 2)),
    ]
    for phrase in phrases:
        if str(phrase).strip():
            argv += ["--phrase", str(phrase).strip()]
    if not opts.get("log_near_misses", True):
        argv.append("--no-near-miss-logs")
    if opts.get("save_debug_clips", False):
        argv += ["--debug-clip-dir", "/data/debug_clips"]
    return argv


def seed_model_cache(cache_dir: Path) -> None:
    """Development builds ship the weights; the engine still verifies the pinned SHA256."""
    if not LOCAL_MODELS.is_dir():
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    for src in LOCAL_MODELS.glob("*.pt"):
        dest = cache_dir / src.name
        if not dest.exists():
            shutil.copyfile(src, dest)
            log(f"Seeded model cache with bundled {src.name}")


def wait_for_port(proc: subprocess.Popen, timeout: float = 900.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def register_discovery() -> None:
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        log("No SUPERVISOR_TOKEN; skipping discovery (add the Wyoming integration manually)")
        return
    uri = f"tcp://{socket.gethostname()}:{PORT}"
    body = json.dumps({"service": "wyoming", "config": {"uri": uri}}).encode()
    req = urllib.request.Request(
        "http://supervisor/discovery",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            log(f"Registered Wyoming discovery for {uri} (HTTP {resp.status})")
    except OSError as exc:  # discovery is a convenience; the server keeps running
        log(f"Discovery registration failed ({exc}); add Wyoming manually: host "
            f"{socket.gethostname()}, port {PORT}")


def main() -> int:
    opts = json.loads(OPTIONS.read_text()) if OPTIONS.is_file() else {}
    cache_dir = Path(os.environ.setdefault("HARK_CACHE_DIR", "/data/models"))
    seed_model_cache(cache_dir)

    proc = subprocess.Popen(build_argv(opts))
    stop_requested = []

    def forward(signum, _frame):
        stop_requested.append(signum)
        proc.send_signal(signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, forward)

    if wait_for_port(proc):
        register_discovery()
    return exit_code(proc.wait(), bool(stop_requested))


def exit_code(returncode: int, stop_requested: bool) -> int:
    """A stop requested by the Supervisor is a clean exit, not a crash shown as "Error"."""
    if stop_requested and returncode in (0, -signal.SIGTERM, -signal.SIGINT):
        return 0
    return returncode if returncode >= 0 else 128 - returncode


if __name__ == "__main__":
    sys.exit(main())
