"""Bounded optional persistence for desktop transcript history."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from voiced.config import get_cache_dir

DEFAULT_MAX_ENTRIES = 200
MAX_FILE_BYTES = 2 * 1024 * 1024


def _timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value or "")


class TranscriptHistory:
    """Local bounded store; persistence is opt-in and writes atomically."""

    def __init__(
        self, path: Path | None = None, *, persist: bool = False, limit: int = DEFAULT_MAX_ENTRIES
    ):
        if limit < 1:
            raise ValueError("limit must be positive")
        self.path = path or (get_cache_dir() / "desktop-history.json")
        self.persist = persist
        self.limit = limit
        self._lock = threading.RLock()
        self._entries: list[dict[str, Any]] = []
        if persist:
            self._load()

    def _load(self):
        try:
            if self.path.stat().st_size > MAX_FILE_BYTES:
                return
            os.chmod(self.path, 0o600)
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, list):
                self._entries = [
                    item
                    for item in value
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                ][-self.limit :]
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError):
            self._entries = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".desktop-history-", dir=self.path.parent, text=True
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self._entries, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._entries]

    def sync(self, entries: list[dict[str, Any]]) -> None:
        """Merge daemon entries, preserving local transcripts and corrections."""
        with self._lock:
            local = [item for item in self._entries if str(item.get("id", "")).startswith("local-")]
            corrections = {
                str(item.get("id")): item["corrected_text"]
                for item in self._entries
                if item.get("corrected_text") is not None
            }
            merged = []
            for item in entries:
                if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                    continue
                copy = dict(item)
                key = str(copy.get("id", ""))
                if key in corrections:
                    copy["corrected_text"] = corrections[key]
                copy["timestamp"] = _timestamp(copy.get("timestamp"))
                merged.append(copy)
            local_ids = {str(item.get("id", "")) for item in local}
            daemon_entries = [item for item in merged if str(item.get("id", "")) not in local_ids]
            self._entries = (local + daemon_entries)[: self.limit]
            if self.persist:
                try:
                    self._save()
                except OSError:
                    pass

    def correct(self, entry_id: int | str, text: str) -> bool:
        """Set a local correction without changing the daemon's canonical result."""
        with self._lock:
            for item in self._entries:
                if str(item.get("id")) != str(entry_id):
                    continue
                item["corrected_text"] = text
                if self.persist:
                    try:
                        self._save()
                    except OSError:
                        return False
                return True
            return False

    def add_local(self, text: str) -> dict[str, Any]:
        """Add a desktop-generated transcript without changing daemon history."""
        with self._lock:
            entry = {
                "id": f"local-{time.time_ns()}",
                "text": text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._entries.insert(0, entry)
            del self._entries[self.limit :]
            if self.persist:
                try:
                    self._save()
                except OSError:
                    pass
            return dict(entry)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            if self.persist:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
