"""Tests for device module."""

import torch

from voiced.device import DeviceConfig, resolve_device_config


class TestExplicitRequest:
    def test_cpu(self):
        cfg = resolve_device_config("cpu")
        assert cfg.device == "cpu"
        assert cfg.dtype == torch.float32
        assert cfg.attn_impl == "sdpa"

    def test_cuda(self):
        cfg = resolve_device_config("cuda")
        assert cfg.device == "cuda"
        assert cfg.dtype == torch.bfloat16
        assert cfg.attn_impl == "flash_attention_2"

    def test_mps(self):
        cfg = resolve_device_config("mps")
        assert cfg.device == "mps"
        assert cfg.dtype == torch.float32
        assert cfg.attn_impl == "sdpa"

    def test_unknown_falls_back_to_cpu(self):
        cfg = resolve_device_config("garbage")
        assert cfg.device == "cpu"


class TestAuto:
    def test_auto_returns_a_known_device(self):
        cfg = resolve_device_config("auto")
        assert cfg.device in {"cpu", "cuda", "mps"}
        assert isinstance(cfg, DeviceConfig)


class TestImmutability:
    def test_device_config_is_frozen(self):
        cfg = resolve_device_config("cpu")
        import pytest

        with pytest.raises((AttributeError, Exception)):
            cfg.device = "cuda"  # type: ignore[misc]
