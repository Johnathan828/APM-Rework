# APM/LIO/src/alerts/state_store.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


class StateStore:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sensor_name: str) -> Path:
        safe = sensor_name.replace("/", "_")
        return self.base_dir / f"{safe}.json"

    def load(self, sensor_name: str) -> Dict[str, Any]:
        p = self._path(sensor_name)
        if not p.exists():
            return {"episode_active": False, "episode_started_at": None}
        try:
            return json.loads(p.read_text())
        except Exception:
            return {"episode_active": False, "episode_started_at": None}

    def save(self, sensor_name: str, state: Dict[str, Any]) -> None:
        p = self._path(sensor_name)
        p.write_text(json.dumps(state, indent=2, sort_keys=True))