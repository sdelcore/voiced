"""Tests for ModelHost lifecycle, refcount, and idle-unload race fix."""

import threading
import time

import pytest

from voiced.model_host import ModelHost


class FakeModel:
    def __init__(self, value: int = 1):
        self.value = value


class TestLazyLoad:
    def test_loader_not_called_at_init(self):
        calls = []
        ModelHost(loader=lambda: calls.append(1) or FakeModel())
        assert calls == []

    def test_loader_called_on_first_use(self):
        calls = []

        def loader():
            calls.append(1)
            return FakeModel()

        host = ModelHost(loader=loader)
        with host.use() as model:
            assert isinstance(model, FakeModel)
        assert calls == [1]

    def test_loader_called_once_across_uses(self):
        calls = []

        def loader():
            calls.append(1)
            return FakeModel()

        host = ModelHost(loader=loader)
        with host.use():
            pass
        with host.use():
            pass
        assert calls == [1]

    def test_loader_error_propagates(self):
        host: ModelHost[FakeModel] = ModelHost(
            loader=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        with pytest.raises(RuntimeError, match="boom"):
            with host.use():
                pass
        # Host stays unloaded so the next use() retries
        assert not host.is_loaded


class TestRefcount:
    def test_concurrent_leases_share_one_load(self):
        calls = []

        def loader():
            calls.append(1)
            return FakeModel()

        host = ModelHost(loader=loader)
        with host.use():
            with host.use():
                pass
        assert calls == [1]

    def test_unload_blocked_by_active_lease(self):
        host = ModelHost(loader=FakeModel)
        with host.use():
            with pytest.raises(RuntimeError, match="active lease"):
                host.unload()


class TestIdleUnload:
    def test_no_idle_timeout_means_no_auto_unload(self):
        host = ModelHost(loader=FakeModel, idle_timeout=None)
        with host.use():
            pass
        time.sleep(0.05)
        assert host.is_loaded

    def test_idle_timer_unloads_after_release(self):
        host = ModelHost(loader=FakeModel, idle_timeout=0.05)
        with host.use():
            pass
        time.sleep(0.15)
        assert not host.is_loaded

    def test_new_lease_cancels_pending_unload(self):
        """The race fix: a lease acquired before the timer fires keeps the model loaded."""
        host = ModelHost(loader=FakeModel, idle_timeout=0.1)
        with host.use():
            pass
        # Sleep less than idle_timeout, then re-acquire
        time.sleep(0.03)
        with host.use():
            time.sleep(0.15)  # exceeds the original idle_timeout
            assert host.is_loaded  # still loaded — no unload during active lease
        # Now wait for the new idle timer to fire
        time.sleep(0.2)
        assert not host.is_loaded

    def test_on_unload_hook_fires(self):
        unloaded = []
        model = FakeModel(value=42)
        host = ModelHost(
            loader=lambda: model,
            idle_timeout=0.05,
            on_unload=lambda m: unloaded.append(m.value),
        )
        with host.use():
            pass
        time.sleep(0.15)
        assert unloaded == [42]

    def test_on_unload_hook_exception_does_not_break_unload(self):
        host = ModelHost(
            loader=FakeModel,
            idle_timeout=0.05,
            on_unload=lambda _m: (_ for _ in ()).throw(RuntimeError("hook fail")),
        )
        with host.use():
            pass
        time.sleep(0.15)
        # Despite the hook raising, the model is still considered unloaded
        assert not host.is_loaded


class TestShutdown:
    def test_shutdown_unloads_when_idle(self):
        host = ModelHost(loader=FakeModel)
        with host.use():
            pass
        host.shutdown()
        assert not host.is_loaded

    def test_use_after_shutdown_raises(self):
        host = ModelHost(loader=FakeModel)
        host.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            with host.use():
                pass

    def test_shutdown_with_active_lease_defers_unload(self):
        host = ModelHost(loader=FakeModel)
        with host.use():
            host.shutdown()
            assert host.is_loaded  # still loaded — lease is active
        # Note: current implementation does not unload after the lease is
        # released following shutdown; the user is responsible for not
        # holding leases across shutdown. This test documents that.


class TestRaceFix:
    """The original Synthesizer bug: idle timer firing mid-use."""

    def test_long_use_outlasts_original_idle_timeout(self):
        """A use() that runs longer than idle_timeout must not be unloaded mid-flight."""
        host = ModelHost(loader=FakeModel, idle_timeout=0.05)

        # First use to start the timer machinery
        with host.use():
            pass

        # Now grab a lease and hold it past idle_timeout
        with host.use() as model:
            time.sleep(0.15)
            # Model object must still be the same instance — not collected
            assert isinstance(model, FakeModel)
            assert host.is_loaded

    def test_concurrent_long_uses(self):
        host = ModelHost(loader=FakeModel, idle_timeout=0.05)
        results = []

        def worker():
            with host.use():
                time.sleep(0.1)
                results.append(host.is_loaded)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results == [True, True, True]
