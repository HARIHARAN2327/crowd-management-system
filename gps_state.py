from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


_lock = threading.Lock()
_latest: Dict[str, Any] = {
    'lat': None,
    'lon': None,
    'accuracy': None,
    'ts': None,
    'received_at': None,
}


def set_latest(lat: float, lon: float, accuracy: Optional[float] = None, ts: Optional[float] = None) -> None:
    with _lock:
        _latest['lat'] = float(lat)
        _latest['lon'] = float(lon)
        _latest['accuracy'] = None if accuracy is None else float(accuracy)
        _latest['ts'] = time.time() if ts is None else float(ts)
        _latest['received_at'] = time.time()


def get_latest() -> Dict[str, Any]:
    with _lock:
        return dict(_latest)
