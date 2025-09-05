from __future__ import annotations

import hashlib
from typing import List, Dict, Optional

from .config import WINDOWS_LOG_PATH


def _hash_template(line: str) -> str:
    # Fallback: hash whole line as a pseudo-template
    return hashlib.sha1(line.strip().encode("utf-8", errors="ignore")).hexdigest()[:12]


def read_windows_log(max_lines: int | None = None) -> List[str]:
    lines: List[str] = []
    with open(WINDOWS_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            if line.strip():
                lines.append(line.rstrip("\n"))
    return lines


def to_log_keys(lines: List[str]) -> List[str]:
    # Without a template miner (e.g., Drain), fall back to hashing
    return [_hash_template(l) for l in lines]


def make_sessions(keys: List[str], session_size: int = 20, stride: Optional[int] = None) -> List[List[str]]:
    """Create sessions from key sequence.

    - session_size: number of events per session
    - stride: step size between session starts; default == session_size (non-overlap)
    """
    if session_size <= 0:
        session_size = 1
    if stride is None or stride <= 0:
        stride = session_size

    sessions: List[List[str]] = []
    for i in range(0, len(keys), stride):
        window = keys[i : i + session_size]
        if not window:
            break
        sessions.append(window)
        if i + session_size >= len(keys) and stride >= session_size:
            # done
            break
    return sessions
