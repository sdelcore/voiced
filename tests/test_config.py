"""Tests for config module — focused on the Parakeet swap."""

import tempfile
from pathlib import Path

from voiced import config as cfg_module
from voiced.config import Config, TranscriptionConfig, load_config, save_default_config


def test_default_transcription_model_is_parakeet():
    assert TranscriptionConfig().model == "nvidia/parakeet-tdt-0.6b-v3"


def test_no_compute_type_field():
    """compute_type was a faster-whisper concept; should be gone."""
    assert "compute_type" not in TranscriptionConfig.__dataclass_fields__


def test_no_vad_config_class():
    assert not hasattr(cfg_module, "VadConfig")


def test_config_has_no_vad_field():
    assert "vad" not in Config.__dataclass_fields__


def test_default_config_file_uses_parakeet(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("XDG_CONFIG_HOME", tmp)
        save_default_config()
        text = (Path(tmp) / "voiced" / "config.toml").read_text()
        assert "nvidia/parakeet-tdt-0.6b-v3" in text
        assert "compute_type" not in text
        assert "[vad]" not in text


def test_load_config_ignores_unknown_sections(monkeypatch):
    """Old configs may still have a [vad] section; loader must not crash."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("XDG_CONFIG_HOME", tmp)
        config_dir = Path(tmp) / "voiced"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[transcription]\nmodel = "nvidia/parakeet-tdt-0.6b-v2"\n'
            '[vad]\nenabled = true\n'  # Stale section — must be ignored
        )
        config = load_config()
        assert config.transcription.model == "nvidia/parakeet-tdt-0.6b-v2"


def test_load_config_falls_back_to_defaults_when_missing(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("XDG_CONFIG_HOME", tmp)
        # No file written
        config = load_config()
        assert config.transcription.model == "nvidia/parakeet-tdt-0.6b-v3"
