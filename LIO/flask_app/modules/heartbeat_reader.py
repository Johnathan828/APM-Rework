from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any


def read_heartbeat(site_root: Path, sensor_key: str) -> Optional[Dict[str, Any]]:
    p = site_root / "etc" / "state" / f"heartbeat_{sensor_key}.json"
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)


def status_from_heartbeat(hb: Optional[Dict[str, Any]], grace_seconds: int = 30) -> str:
    if not hb:
        return "Down"

    every = int(hb.get("every_seconds", 60))
    ts = hb.get("tick_ok_at_utc")
    if not ts:
        return "Stale"

    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return "Stale"

    age = (datetime.now(timezone.utc) - t).total_seconds()
    return "Running" if age <= (2 * every + grace_seconds) else "Stale"