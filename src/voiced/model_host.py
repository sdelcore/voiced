"""Lifecycle host for lazy-loaded torch models with optional auto-unload."""

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModelHost(Generic[T]):
    """Owns a lazy-loaded model and its idle-unload timer.

    The model is loaded on the first call to ``use()`` and unloaded after
    ``idle_timeout`` seconds with zero active leases. Leases are reference-
    counted: a pending unload timer is cancelled whenever a new lease is
    acquired, so the model cannot be unloaded out from under a caller.
    """

    def __init__(
        self,
        loader: Callable[[], T],
        idle_timeout: float | None = None,
        on_unload: Callable[[T], None] | None = None,
        name: str = "model",
    ):
        self._loader = loader
        self._idle_timeout = idle_timeout
        self._on_unload = on_unload
        self._name = name

        self._model: T | None = None
        self._refcount = 0
        self._timer: threading.Timer | None = None
        self._shutdown = False
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
        with self._lock:
            return self._model is not None

    @contextmanager
    def use(self) -> Iterator[T]:
        """Acquire a lease on the model. Loads on first call.

        While at least one ``use()`` block is active, the idle timer is held
        off. The model is guaranteed to remain loaded until every lease is
        released.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError(f"ModelHost[{self._name}] has been shut down")

            self._cancel_timer()

            if self._model is None:
                logger.info(f"Loading {self._name}...")
                self._model = self._loader()
                logger.info(f"{self._name} loaded")

            self._refcount += 1
            model = self._model

        try:
            yield model
        finally:
            with self._lock:
                self._refcount -= 1
                if self._refcount == 0 and self._idle_timeout and not self._shutdown:
                    self._schedule_idle_unload()

    def unload(self) -> None:
        """Forcibly unload the model. Raises if any leases are active."""
        with self._lock:
            if self._refcount > 0:
                raise RuntimeError(f"Cannot unload {self._name}: {self._refcount} active lease(s)")
            self._cancel_timer()
            self._do_unload()

    def shutdown(self) -> None:
        """Final shutdown. Cancels the idle timer, unloads if no leases.

        After shutdown, ``use()`` raises. Active leases are allowed to finish;
        unload happens once they release.
        """
        with self._lock:
            self._shutdown = True
            self._cancel_timer()
            if self._refcount == 0:
                self._do_unload()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule_idle_unload(self) -> None:
        assert self._idle_timeout is not None
        self._cancel_timer()
        self._timer = threading.Timer(self._idle_timeout, self._on_idle_timer_fire)
        self._timer.daemon = True
        self._timer.start()

    def _on_idle_timer_fire(self) -> None:
        with self._lock:
            self._timer = None
            if self._refcount > 0:
                # A lease was acquired after the timer was scheduled but before
                # it fired (and the cancel inside use() raced with the fire).
                return
            self._do_unload()

    def _do_unload(self) -> None:
        if self._model is None:
            return
        logger.info(f"Unloading {self._name}...")
        model = self._model
        self._model = None
        if self._on_unload is not None:
            try:
                self._on_unload(model)
            except Exception:
                logger.exception(f"on_unload hook failed for {self._name}")
        del model
        logger.info(f"{self._name} unloaded")
