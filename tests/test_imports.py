"""Tests for public API imports."""


class TestPublicImports:
    """Verify all public API items are importable from sttd."""

    def test_import_version(self):
        """Test __version__ is importable."""
        from sttd import __version__

        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"

    def test_import_transcriber(self):
        """Test Transcriber class is importable."""
        from sttd import Transcriber, TranscriptionConfig

        assert Transcriber is not None
        assert TranscriptionConfig is not None

    def test_import_recorder(self):
        """Test Recorder class is importable."""
        from sttd import AudioConfig, Recorder

        assert Recorder is not None
        assert AudioConfig is not None

    def test_import_speaker_identification(self):
        """Test speaker identification classes are importable."""
        from sttd import (
            ENROLLMENT_PROMPT,
            DiarizationConfig,
            DiarizedSegment,
            IdentifiedSegment,
            SpeakerEmbedder,
            SpeakerIdentifier,
        )

        assert SpeakerEmbedder is not None
        assert SpeakerIdentifier is not None
        assert IdentifiedSegment is not None
        assert DiarizedSegment is IdentifiedSegment  # Alias
        assert DiarizationConfig is not None
        assert isinstance(ENROLLMENT_PROMPT, str)

    def test_import_profiles(self):
        """Test profile classes are importable."""
        from sttd import ProfileManager, VoiceProfile

        assert VoiceProfile is not None
        assert ProfileManager is not None

    def test_import_config(self):
        """Test config classes are importable."""
        from sttd import Config

        assert Config is not None

    def test_import_all(self):
        """Test __all__ contains expected items."""
        import sttd

        assert hasattr(sttd, "__all__")
        expected = [
            "__version__",
            "Transcriber",
            "TranscriptionConfig",
            "Recorder",
            "AudioConfig",
            "SpeakerEmbedder",
            "SpeakerIdentifier",
            "IdentifiedSegment",
            "DiarizedSegment",
            "DiarizationConfig",
            "ENROLLMENT_PROMPT",
            "VoiceProfile",
            "ProfileManager",
            "Config",
        ]
        for item in expected:
            assert item in sttd.__all__, f"{item} not in __all__"

    def test_all_items_accessible(self):
        """Test that all items in __all__ are accessible."""
        import sttd

        for name in sttd.__all__:
            assert hasattr(sttd, name), f"{name} in __all__ but not accessible"


class TestSubmoduleImports:
    """Verify submodule imports still work for advanced users."""

    def test_import_from_transcriber_module(self):
        """Test direct submodule import."""
        from sttd.transcriber import Transcriber

        assert Transcriber is not None

    def test_import_from_config_module(self):
        """Test direct config import."""
        from sttd.config import AudioConfig, TranscriptionConfig

        assert TranscriptionConfig is not None
        assert AudioConfig is not None

    def test_import_from_diarizer_module(self):
        """Test direct diarizer import."""
        from sttd.diarizer import SpeakerEmbedder, SpeakerIdentifier

        assert SpeakerIdentifier is not None
        assert SpeakerEmbedder is not None


class TestPrivateModulesNotExposed:
    """Verify daemon/CLI internals are not exposed in public API."""

    def test_daemon_not_in_all(self):
        """Test Daemon class is not in public API."""
        import sttd

        assert "Daemon" not in sttd.__all__
        assert not hasattr(sttd, "Daemon")

    def test_server_not_in_all(self):
        """Test Server class is not in public API."""
        import sttd

        assert "Server" not in sttd.__all__
        assert not hasattr(sttd, "Server")

    def test_cli_not_in_all(self):
        """Test CLI is not in public API."""
        import sttd

        assert "main" not in sttd.__all__
        assert not hasattr(sttd, "main")
