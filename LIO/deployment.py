# APM/LIO/deployment.py
from __future__ import annotations

import argparse
import signal
import sys
import time
from multiprocessing import Process
from pathlib import Path
from typing import Dict

from src.apm_core.ssot import load_ssot, list_sensors
from src.pipeline.sensor_worker import run_sensor_process


def main() -> None:
    parser = argparse.ArgumentParser(description="NewAPM deployment runner (legacy-like: one process per sensor)")
    parser.add_argument("--sensors", nargs="*", default=None, help="Optional sensors subset")
    parser.add_argument("--every", type=int, default=None, help="Override run interval seconds for ALL sensors")
    args = parser.parse_args()

    site_root = Path(__file__).resolve().parent
    ssot = load_ssot(site_root)

    sensors = list_sensors(ssot)
    if args.sensors:
        wanted = set(args.sensors)
        sensors = [s for s in sensors if s in wanted]

    if not sensors:
        print("No sensors selected. Exiting.")
        return

    print(f"DEPLOYMENT START | site={ssot.get('site')} | sensors={sensors}")

    procs: Dict[str, Process] = {}

    def spawn(sensor_name: str) -> Process:
        cfg = ssot.get(sensor_name, {}) or {}
        other = cfg.get("Other", {}) or {}
        every = int(args.every) if args.every is not None else int(other.get("sampling_freq_seconds", 15))

        print(f"SPAWN | {sensor_name} | every={every}s")
        p = Process(target=run_sensor_process, args=(site_root, sensor_name, every), daemon=True)
        p.start()
        return p

    for s in sensors:
        procs[s] = spawn(s)

    def shutdown(signum, frame):
        print(f"DEPLOYMENT STOP | signal={signum} | terminating child processes...")
        for p in procs.values():
            if p.is_alive():
                p.terminate()
        for p in procs.values():
            p.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Supervisor loop: restart any dead worker
    while True:
        time.sleep(5)
        for s, p in list(procs.items()):
            if not p.is_alive():
                print(f"RESTART | {s}")
                procs[s] = spawn(s)


if __name__ == "__main__":
    main()