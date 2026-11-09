"""Asynchronous OpenAI-compatible conversational assistant."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from voiced.config import AssistantConfig, get_cache_dir
from voiced.tts_streamer import AudioPlayer

logger = logging.getLogger(__name__)


class Assistant:
    """Submit user text for background chat completion and speak the reply."""

    def __init__(self, config: AssistantConfig, synthesizer: Any):
        self.config = config
        self.synthesizer = synthesizer
        self._lock = threading.RLock()
        self._generation = 0
        self._closed = False
        self._state = "idle"
        self._messages: list[dict[str, str]] = []
        self._path = Path(get_cache_dir()) / "assistant" / "conversation.json"
        self._load_history()
        self._player = AudioPlayer()
        self._worker: threading.Thread | None = None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            if self._closed:
                return
            self._generation += 1
            generation = self._generation
            self._state = "thinking"
            # Snapshot prior committed turns; only the current reply is persisted.
            messages = [*self._messages, {"role": "user", "content": text}]
            worker = threading.Thread(target=self._turn, args=(generation, messages), daemon=True)
            self._worker = worker
        worker.start()

    def interrupt(self) -> None:
        try:
            with self._lock:
                self._generation += 1
                self._state = "idle"
                self._player.stop()
        except Exception:
            logger.debug("Could not stop assistant playback", exc_info=True)

    def close(self) -> None:
        try:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._generation += 1
                self._state = "closed"
                self._player.stop()
        except Exception:
            logger.debug("Could not stop assistant playback", exc_info=True)

    def _turn(self, generation: int, messages: list[dict[str, str]]) -> None:
        try:
            reply = self._request(messages)
            with self._lock:
                if self._closed or generation != self._generation:
                    return
                # Commit only complete assistant responses; user input is not durable alone.
                self._messages.extend((messages[-1], {"role": "assistant", "content": reply}))
                self._save_history()
                self._state = "speaking"
            audio = self.synthesizer.synthesize(reply)
            with self._lock:
                if self._closed or generation != self._generation:
                    return
                self._player.play(audio, block=False)
            self._player.wait()
            with self._lock:
                if not self._closed and generation == self._generation:
                    self._state = "idle"
        except Exception:
            logger.exception("Assistant turn failed")
            with self._lock:
                if not self._closed and generation == self._generation:
                    self._state = "error"

    def _request(self, messages: list[dict[str, str]]) -> str:
        base_url = self.config.base_url.rstrip("/")
        if not base_url:
            raise RuntimeError("Assistant is disabled")
        url = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
        payload = json.dumps({"model": self.config.model, "messages": messages}).encode()
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.config.api_key_env, "") if self.config.api_key_env else ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            result = json.loads(response.read())
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Invalid chat completion response") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Chat completion returned no text")
        return content.strip()

    def _load_history(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._messages = [
                    message
                    for message in data
                    if isinstance(message, dict)
                    and message.get("role") in ("user", "assistant")
                    and isinstance(message.get("content"), str)
                ]
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            logger.warning("Could not load assistant conversation history", exc_info=True)

    def _save_history(self) -> None:
        try:
            self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                dir=self._path.parent, prefix=".conversation.", suffix=".tmp"
            )
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as temporary_file:
                temporary_file.write(json.dumps(self._messages, ensure_ascii=False))
            Path(temporary_name).replace(self._path)
        except OSError:
            logger.exception("Could not save assistant conversation history")
