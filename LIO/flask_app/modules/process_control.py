from __future__ import annotations

import datetime
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --------------------------
# Paths
# --------------------------
def _state_dir(site_root: Path) -> Path:
    d = site_root / "etc" / "state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _commands_file(site_root: Path) -> Path:
    return _state_dir(site_root) / "supervisor_commands.json"


def _pidfile(site_root: Path) -> Path:
    return _state_dir(site_root) / "pid_supervisor.pid"


def _heartbeat(site_root: Path) -> Path:
    return _state_dir(site_root) / "heartbeat_supervisor.json"


def _lockfile(site_root: Path, key: str) -> Path:
    return _state_dir(site_root) / f"lock_{key}.lock"


def _utc_now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# --------------------------
# File lock
# --------------------------
class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = open(self.path, "a+")
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                self.fd.close()
        except Exception:
            pass
        self.fd = None


# --------------------------
# Process helpers
# --------------------------
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _kill_process_group(pid: int) -> None:
    """
    Kill process group for pid, escalate to SIGKILL.
    """
    try:
        pgid = os.getpgid(int(pid))
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(0.1)
        try:
            os.kill(int(pid), 0)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
    except Exception:
        try:
            os.kill(int(pid), signal.SIGTERM)
            time.sleep(0.1)
            os.kill(int(pid), signal.SIGKILL)
        except Exception:
            pass


def _read_pidfile(site_root: Path) -> Optional[int]:
    try:
        p = _pidfile(site_root)
        if not p.exists():
            return None
        s = p.read_text().strip()
        return int(s) if s else None
    except Exception:
        return None


def _write_pidfile(site_root: Path, pid: int) -> None:
    _pidfile(site_root).write_text(str(int(pid)))


def _remove_pidfile(site_root: Path) -> None:
    try:
        _pidfile(site_root).unlink()
    except Exception:
        pass


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def _pgrep_supervisor_candidates() -> List[int]:
    """
    Returns PIDs that *look* like supervisors by cmdline.
    WARNING: forked workers may inherit this cmdline too.
    We will filter/root-select below.
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


def _pid_ppid(pid: int) -> Optional[int]:
    """
    Return PPID for pid.
    """
    try:
        out = subprocess.check_output(["ps", "-o", "ppid=", "-p", str(pid)], text=True).strip()
        return int(out)
    except Exception:
        return None


def _pick_root_supervisor(pids: List[int]) -> Optional[int]:
    """
    If pgrep returns [supervisor, worker-child, worker-child...],
    choose the ROOT-most one: the PID whose PPID is NOT in the set.
    """
    if not pids:
        return None

    s = set(pids)
    # map pid -> ppid
    ppid_map = {pid: _pid_ppid(pid) for pid in pids}

    roots = [pid for pid in pids if (ppid_map.get(pid) is None or ppid_map.get(pid) not in s)]
    if roots:
        return roots[0]

    # fallback: smallest pid
    return pids[0]


def _get_supervisor_pid(site_root: Path) -> Optional[int]:
    """
    Canonical supervisor PID:
      1) pidfile if alive
      2) root-most candidate from pgrep if alive
    """
    pid_from_file = _read_pidfile(site_root)
    if pid_from_file and _pid_alive(pid_from_file):
        return pid_from_file

    candidates = [p for p in _pgrep_supervisor_candidates() if _pid_alive(p)]
    if not candidates:
        return None

    root = _pick_root_supervisor(candidates)
    if root and _pid_alive(root):
        _write_pidfile(site_root, root)
        return root

    return None


# --------------------------
# Commands
# --------------------------
def _append_command(site_root: Path, cmd: Dict) -> None:
    cmd_path = _commands_file(site_root)
    payload = _safe_read_json(cmd_path) or {"commands": []}
    cmds = payload.get("commands", []) or []
    cmds.append(cmd)
    _atomic_write_json(cmd_path, {"commands": cmds})


# --------------------------
# Supervisor status (READ ONLY)
# --------------------------
def supervisor_status(site_root: Path, stale_seconds: int = 10) -> Tuple[str, Optional[int]]:
    sup = _get_supervisor_pid(site_root)
    if sup:
        hb = _safe_read_json(_heartbeat(site_root)) or {}
        ts = hb.get("tick_ok_at_utc")
        if ts:
            try:
                hb_dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                age = (datetime.datetime.now(datetime.timezone.utc) - hb_dt).total_seconds()
                if age <= float(stale_seconds):
                    return "Running", sup
                return "Stale", sup
            except Exception:
                return "Running", sup
        return "Running", sup

    _remove_pidfile(site_root)
    return "Stopped", None


# --------------------------
# Supervisor control
# --------------------------
def start_supervisor(site_root: Path) -> int:
    """
    Start supervisor only if not already running.
    """
    with _FileLock(_lockfile(site_root, "supervisor")):
        sup = _get_supervisor_pid(site_root)
        if sup:
            return sup

        proc = subprocess.Popen(
            ["python3", "deployment.py", "--supervisor"],
            cwd=str(site_root),
            preexec_fn=os.setsid,
        )

        # wait briefly so pidfile/ps settles
        time.sleep(0.2)

        sup2 = _get_supervisor_pid(site_root)
        if sup2:
            return sup2

        _write_pidfile(site_root, int(proc.pid))
        return int(proc.pid)


def stop_supervisor(site_root: Path) -> int:
    """
    Stop the canonical supervisor (root pid), NOT every pgrep candidate.
    Supervisor will stop workers itself.
    """
    killed = 0
    with _FileLock(_lockfile(site_root, "supervisor")):
        sup = _get_supervisor_pid(site_root)
        if sup and _pid_alive(sup):
            _kill_process_group(sup)
            killed += 1

        time.sleep(0.3)

        sup2 = _get_supervisor_pid(site_root)
        if sup2 and _pid_alive(sup2):
            _kill_process_group(sup2)
            killed += 1

        _remove_pidfile(site_root)

    return killed


# --------------------------
# Sensor start/stop (commands)
# --------------------------
def start_sensor(site_root: Path, sensor_key: str) -> int:
    with _FileLock(_lockfile(site_root, sensor_key)):
        st, _ = supervisor_status(site_root)
        if st == "Stopped":
            start_supervisor(site_root)

        _append_command(site_root, {"ts_utc": _utc_now_iso(), "action": "start_sensor", "sensor": sensor_key})
    return 0


def stop_sensor(site_root: Path, sensor_key: str, pid: Optional[int] = None) -> int:
    with _FileLock(_lockfile(site_root, sensor_key)):
        st, _ = supervisor_status(site_root)
        if st == "Stopped":
            return 0

        _append_command(site_root, {"ts_utc": _utc_now_iso(), "action": "stop_sensor", "sensor": sensor_key})
    return 0


def stop_pid(site_root: Path, sensor_key: str, pid: int) -> None:
    stop_sensor(site_root, sensor_key, pid=pid)