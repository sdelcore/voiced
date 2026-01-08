"""Tests for transcriber module."""

import numpy as np
import pytest

from voiced.config import TranscriptionConfig
from voiced.transcriber import Transcriber


class TestTranscriberConfig:
    """Tests for transcriber configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TranscriptionConfig()
        assert config.model == "base"
        assert config.device == "auto"
        assert config.compute_type == "auto"
        assert config.language == "en"

    def test_custom_config(self):
        """Test custom configuration."""
        config = TranscriptionConfig(
            model="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
        )
        assert config.model == "tiny"
        assert config.device == "cpu"
        assert config.compute_type == "int8"


class TestTranscriber:
    """Tests for Transcriber class."""

    def test_init(self):
        """Test transcriber initialization."""
        transcriber = Transcriber()
        assert transcriber.config.model == "base"
        assert transcriber._model is None  # Model not loaded yet

    def test_init_with_config(self):
        """Test transcriber initialization with custom config."""
        config = TranscriptionConfig(model="tiny")
        transcriber = Transcriber(config)
        assert transcriber.config.model == "tiny"

    def test_get_device_explicit(self):
        """Test explicit device selection."""
        config = TranscriptionConfig(device="cpu")
        transcriber = Transcriber(config)
        assert transcriber._get_device() == "cpu"

    def test_get_compute_type_cpu(self):
        """Test compute type for CPU."""
        config = TranscriptionConfig(compute_type="auto")
        transcriber = Transcriber(config)
        assert transcriber._get_compute_type("cpu") == "int8"

    def test_get_compute_type_cuda(self):
        """Test compute type for CUDA."""
        config = TranscriptionConfig(compute_type="auto")
        transcriber = Transcriber(config)
        assert transcriber._get_compute_type("cuda") == "float16"

    def test_get_compute_type_explicit(self):
        """Test explicit compute type."""
        config = TranscriptionConfig(compute_type="float32")
        transcriber = Transcriber(config)
        assert transcriber._get_compute_type("cpu") == "float32"

    def test_transcribe_file_not_found(self):
        """Test transcribing non-existent file."""
        transcriber = Transcriber()
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe_file("/nonexistent/audio.wav")
