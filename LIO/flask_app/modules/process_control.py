from __future__ import annotations

import os
import subprocess
from pathlib import Path

import psutil


def start_sensor(site_root: Path, sensor_key: str) -> int:
    """
    Starts one sensor using the reworked runner:
      python3 deployment.py --sensors <sensor_key>

    Uses a new process group so it can be terminated cleanly.
    """
    cmd = ["python3", "deployment.py", "--sensors", sensor_key]
    proc = subprocess.Popen(cmd, cwd=str(site_root), preexec_fn=os.setsid)
    return int(proc.pid)


def stop_pid(pid: int) -> None:
    """
    Stops the full process tree for a deployment pid.
    Safer than killpg alone (handles children/grandchildren).
    """
    try:
        p = psutil.Process(int(pid))
    except psutil.NoSuchProcess:
        return

    children = p.children(recursive=True)
    for c in children:
        try:
            c.terminate()
        except Exception:
            pass

    try:
        p.terminate()
    except Exception:
        pass

    gone, alive = psutil.wait_procs(children + [p], timeout=5)

    for a in alive:
        try:
            a.kill()
        except Exception:
            pass