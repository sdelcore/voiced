"""Tests for config module."""

import tempfile
from pathlib import Path

from voiced import config as cfg_module
from voiced.config import Config, TranscriptionConfig, TTSConfig, load_config, save_default_config


def test_stt_model_is_fixed():
    """The model is hardcoded in the transcriber, not configurable."""
    from voiced.transcriber import STT_MODEL

    assert STT_MODEL == "nvidia/parakeet-tdt-0.6b-v3"
    assert "model" not in TranscriptionConfig.__dataclass_fields__


def test_tts_model_is_fixed():
    """The model is hardcoded in the synthesizer, not configurable."""
    from voiced.synthesizer import TTS_MODEL

    assert TTS_MODEL == "hexgrad/Kokoro-82M"
    assert "model" not in TTSConfig.__dataclass_fields__


def test_no_compute_type_field():
    """compute_type was a faster-whisper concept; should be gone."""
    assert "compute_type" not in TranscriptionConfig.__dataclass_fields__


def test_no_vad_config_class():
    assert not hasattr(cfg_module, "VadConfig")


def test_config_has_no_vad_field():
    assert "vad" not in Config.__dataclass_fields__


def test_default_tts_voice_is_kokoro_pack():
    assert TTSConfig().default_voice == "af_heart"


def test_default_config_file_has_no_model_keys(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("XDG_CONFIG_HOME", tmp)
        save_default_config()
        text = (Path(tmp) / "voiced" / "config.toml").read_text()
        assert "model =" not in text
        assert "compute_type" not in text
        assert "[vad]" not in text
        assert "af_heart" in text


def test_load_config_ignores_stale_model_keys(monkeypatch):
    """Old configs may still carry model keys; loader must ignore them."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("XDG_CONFIG_HOME", tmp)
        config_dir = Path(tmp) / "voiced"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[transcription]\nmodel = "nvidia/parakeet-tdt-0.6b-v2"\ndevice = "cpu"\n'
            '[tts]\nmodel = "microsoft/VibeVoice-Realtime-0.5B"\ncfg_scale = 1.5\n'
            "[vad]\nenabled = true\n"  # Stale section — must be ignored
        )
        config = load_config()
        assert not hasattr(config.transcription, "model")
        assert config.transcription.device == "cpu"
        assert not hasattr(config.tts, "model")


def test_load_config_reads_replacements(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("XDG_CONFIG_HOME", tmp)
        config_dir = Path(tmp) / "voiced"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            '[transcription.replacements]\n"cloud code" = "Claude Code"\n'
        )
        config = load_config()
        assert config.transcription.replacements == {"cloud code": "Claude Code"}


def test_load_config_falls_back_to_defaults_when_missing(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("XDG_CONFIG_HOME", tmp)
        # No file written
        config = load_config()
        assert config.transcription.device == "auto"
        assert config.tts.default_voice == "af_heart"
