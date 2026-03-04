# APM/LIO/deployment.py
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from multiprocessing import Process
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from src.apm_core.ssot import load_ssot, list_sensors
from src.pipeline.sensor_worker import run_sensor_process

STATE_DIR = "etc/state"
PIDFILE = "pid_supervisor.pid"
HEARTBEAT = "heartbeat_supervisor.json"
COMMANDS = "supervisor_commands.json"


def _state_dir(site_root: Path) -> Path:
    d = site_root / STATE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pidfile(site_root: Path) -> Path:
    return _state_dir(site_root) / PIDFILE


def _heartbeat(site_root: Path) -> Path:
    return _state_dir(site_root) / HEARTBEAT


def _commands(site_root: Path) -> Path:
    return _state_dir(site_root) / COMMANDS


def _utc_now_iso() -> str:
    return pd.Timestamp.utcnow().to_pydatetime().replace(microsecond=0).isoformat() + "Z"


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _best_every_seconds(ssot: dict, sensor: str, override_every: Optional[int]) -> int:
    if override_every is not None:
        return int(override_every)
    cfg = ssot.get(sensor, {}) or {}
    other = cfg.get("Other", {}) or {}
    return int(other.get("sampling_freq_seconds", 15))


def _spawn_worker(site_root: Path, ssot: dict, sensor: str, override_every: Optional[int]) -> Process:
    every = _best_every_seconds(ssot, sensor, override_every)
    print(f"SPAWN | {sensor} | every={every}s", flush=True)
    p = Process(target=run_sensor_process, args=(site_root, sensor, every), daemon=False)
    p.start()
    return p


def _kill_pid(pid: int, sig: int) -> None:
    try:
        os.kill(int(pid), sig)
    except Exception:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _terminate_worker(p: Process, timeout_s: float = 3.0) -> None:
    """
    HARD STOP:
      - SIGTERM
      - wait
      - SIGKILL if still alive
    """
    try:
        pid = getattr(p, "pid", None)
        if pid is None:
            return

        if _pid_alive(pid):
            _kill_pid(pid, signal.SIGTERM)

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if not _pid_alive(pid):
                break
            time.sleep(0.1)

        if _pid_alive(pid):
            _kill_pid(pid, signal.SIGKILL)

        try:
            p.join(timeout=1)
        except Exception:
            pass

    except Exception:
        pass


def _write_hb(site_root: Path, site: str, desired: List[str], running: List[str], worker_pids: Dict[str, int]) -> None:
    hb = {
        "site": site,
        "supervisor_pid": int(os.getpid()),
        "tick_ok_at_utc": _utc_now_iso(),
        "desired_sensors": desired,
        "running_sensors": running,
        "worker_pids": worker_pids,
    }
    _atomic_write_json(_heartbeat(site_root), hb)


def _drain_commands(site_root: Path, last_mtime: Optional[float]) -> Tuple[List[dict], Optional[float]]:
    cmd_path = _commands(site_root)
    try:
        if not cmd_path.exists():
            return [], last_mtime

        mtime = cmd_path.stat().st_mtime
        if last_mtime is not None and mtime == last_mtime:
            return [], last_mtime

        payload = _safe_read_json(cmd_path) or {"commands": []}
        cmds = payload.get("commands", []) or []

        # clear after reading
        _atomic_write_json(cmd_path, {"commands": []})
        return cmds, mtime
    except Exception:
        return [], last_mtime


def _supervisor_pids() -> List[int]:
    """
    Find ALL existing supervisors by cmdline.
    """
    pattern = r"deployment\.py.*--supervisor"
    try:
        out = subprocess.check_output(["pgrep", "-af", pattern], text=True).strip()
    except subprocess.CalledProcessError:
        return []
    pids: List[int] = []
    for line in out.splitlines():
        try:
            pids.append(int(line.split(maxsplit=1)[0]))
        except Exception:
            pass
    return sorted(set(pids))


def run_supervisor(site_root: Path, override_every: Optional[int], start_all: bool) -> None:
    # --------------------------
    # SINGLETON GUARD
    # If ANY other supervisor is already running, exit immediately.
    # This prevents the multi-supervisor madness even if Flask calls start twice.
    # --------------------------
    existing = [p for p in _supervisor_pids() if p != os.getpid()]
    if existing:
        # If we're here, this process is the "new" one.
        # Exit without doing anything.
        print(f"SUPERVISOR_ALREADY_RUNNING | existing={existing} | exiting", flush=True)
        return

    ssot = load_ssot(site_root)
    site_code = str(ssot.get("site", "UNKNOWN"))

    _pidfile(site_root).write_text(str(os.getpid()))
    print(
        f"SUPERVISOR START | site={site_code} | pid={os.getpid()} | ppid={os.getppid()} | pgid={os.getpgid(os.getpid())}",
        flush=True,
    )

    desired: Set[str] = set()
    workers: Dict[str, Process] = {}

    if start_all:
        desired.update(list_sensors(ssot))

    stop_flag = {"stop": False}
    stopped_once = {"done": False}

    def shutdown(signum, frame):
        if not stopped_once["done"]:
            print(
                f"SUPERVISOR STOP | signal={signum} | pid={os.getpid()} | ppid={os.getppid()} | pgid={os.getpgid(os.getpid())}",
                flush=True,
            )
            stopped_once["done"] = True
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    last_cmd_mtime: Optional[float] = None
    last_restart_check = 0.0
    last_hb = 0.0

    while not stop_flag["stop"]:
        now = time.time()

        # 1) Commands
        cmds, last_cmd_mtime = _drain_commands(site_root, last_cmd_mtime)
        if cmds:
            ssot = load_ssot(site_root)
            known = set(list_sensors(ssot))

            for c in cmds:
                action = str(c.get("action", "")).lower().strip()
                sensor = c.get("sensor")
                sensor = str(sensor).strip() if sensor is not None else None

                if action == "stop_supervisor":
                    stop_flag["stop"] = True
                    break

                if action == "start_all":
                    desired.update(list_sensors(ssot))
                    continue

                if action == "stop_all":
                    for s, p in list(workers.items()):
                        print(f"STOP_WORKER | {s} | pid={getattr(p,'pid',None)}", flush=True)
                        _terminate_worker(p)
                    workers.clear()
                    desired.clear()
                    continue

                if action == "start_sensor":
                    if sensor and sensor in known:
                        desired.add(sensor)
                    continue

                if action == "stop_sensor":
                    if sensor:
                        desired.discard(sensor)
                        p = workers.pop(sensor, None)
                        if p is not None:
                            print(f"STOP_WORKER | {sensor} | pid={getattr(p,'pid',None)}", flush=True)
                            _terminate_worker(p)
                    continue

        # 2) Ensure desired sensors running
        if desired:
            ssot = load_ssot(site_root)
            for s in sorted(desired):
                p = workers.get(s)
                if p is None or (p is not None and not p.is_alive()):
                    if p is not None and not p.is_alive():
                        workers.pop(s, None)
                    workers[s] = _spawn_worker(site_root, ssot, s, override_every)

        # 3) Restart dead workers
        if now - last_restart_check >= 5.0:
            last_restart_check = now
            for s, p in list(workers.items()):
                if s in desired and not p.is_alive():
                    print(f"RESTART | {s}", flush=True)
                    workers.pop(s, None)
                    ssot2 = load_ssot(site_root)
                    workers[s] = _spawn_worker(site_root, ssot2, s, override_every)

        # 4) Heartbeat
        if now - last_hb >= 2.0:
            last_hb = now
            running = [s for s, p in workers.items() if p.is_alive()]
            worker_pids = {s: int(getattr(p, "pid", -1)) for s, p in workers.items()}
            _write_hb(site_root, site_code, sorted(desired), sorted(running), worker_pids)

        time.sleep(1.0)

    # Shutdown: HARD STOP workers we spawned
    print("SUPERVISOR SHUTDOWN | stopping all workers...", flush=True)
    for s, p in list(workers.items()):
        print(f"STOP_WORKER | {s} | pid={getattr(p,'pid',None)}", flush=True)
        _terminate_worker(p)

    try:
        _pidfile(site_root).unlink()
    except Exception:
        pass

    print("SUPERVISOR EXIT", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor", action="store_true")
    parser.add_argument("--start-all", action="store_true")
    parser.add_argument("--every", type=int, default=None)
    args = parser.parse_args()

    site_root = Path(__file__).resolve().parent

    if args.supervisor:
        run_supervisor(site_root, args.every, args.start_all)
        return

    print("Please run: python3 deployment.py --supervisor", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()