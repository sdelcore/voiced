"""Behavioral tests for the asynchronous chat assistant."""

import json
import threading
from unittest.mock import patch

from voiced.assistant import Assistant
from voiced.config import AssistantConfig


class FakeSynthesizer:
    def __init__(self):
        self.texts = []

    def synthesize(self, text):
        self.texts.append(text)
        return text


class FakePlayer:
    def __init__(self):
        self.played = []
        self.stopped = 0

    def play(self, audio, block=True):
        self.played.append(audio)

    def wait(self):
        pass

    def stop(self):
        self.stopped += 1


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": self.content}}]}).encode()


def make_assistant(tmp_path):
    synth = FakeSynthesizer()
    assistant = Assistant(
        AssistantConfig(
            base_url="https://chat.example/v1",
            model="test-model",
            api_key_env="TEST_KEY",
            timeout=3,
        ),
        synth,
    )
    assistant._path = tmp_path / "private" / "conversation.json"
    assistant._player = FakePlayer()
    return assistant, synth


def wait_state(assistant, state):
    # Avoid sleep-based races while the background thread updates state.
    import time

    deadline = time.monotonic() + 2
    while assistant.state != state and time.monotonic() < deadline:
        threading.Event().wait(0.005)
    assert assistant.state == state


def test_submit_sends_openai_request_and_persists_complete_turn(tmp_path, monkeypatch):
    assistant, synth = make_assistant(tmp_path)
    monkeypatch.setenv("TEST_KEY", "secret")
    calls = []

    def open_url(request, timeout):
        calls.append((request, timeout))
        return FakeResponse("Hello there.")

    with patch("voiced.assistant.urllib.request.urlopen", open_url):
        assistant.submit("  Hi  ")
        wait_state(assistant, "idle")

    request, timeout = calls[0]
    assert request.full_url == "https://chat.example/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer secret"
    assert json.loads(request.data) == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    assert timeout == 3
    assert synth.texts == ["Hello there."]
    assert assistant._player.played == ["Hello there."]
    assert json.loads(assistant._path.read_text()) == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there."},
    ]
    assert assistant._path.stat().st_mode & 0o077 == 0


def test_interrupt_discards_late_response_and_allows_new_request(tmp_path):
    assistant, _synth = make_assistant(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    answers = iter(["obsolete", "fresh"])

    def request(_messages):
        answer = next(answers)
        if answer == "obsolete":
            entered.set()
            release.wait(2)
        return answer

    assistant._request = request
    assistant.submit("first")
    assert entered.wait(1)
    assistant.interrupt()
    assistant.submit("second")
    wait_state(assistant, "idle")
    release.set()
    assert assistant._player.played == ["fresh"]
    assert json.loads(assistant._path.read_text()) == [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "fresh"},
    ]


def test_interrupt_before_playback_prevents_stale_generation_start(tmp_path):
    assistant, _synth = make_assistant(tmp_path)
    synthesized = threading.Event()
    release = threading.Event()
    play_calls = []

    def synthesize(text):
        synthesized.set()
        release.wait(2)
        return text

    assistant.synthesizer.synthesize = synthesize
    assistant._player.play = lambda audio, block=True: play_calls.append(audio)
    assistant._request = lambda _messages: "stale reply"
    assistant.submit("question")
    assert synthesized.wait(1)
    assistant.interrupt()
    release.set()
    wait_state(assistant, "idle")
    assert play_calls == []


def test_interrupt_invalidates_playback_without_waiting_for_it(tmp_path):
    assistant, _synth = make_assistant(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    play_calls = []

    def start_play(audio, block=True):
        assert block is False
        play_calls.append(audio)
        entered.set()

    def wait_for_playback():
        release.wait(2)

    assistant._player.play = start_play
    assistant._player.wait = wait_for_playback
    assistant._request = lambda _messages: "reply"
    assistant.submit("question")
    assert entered.wait(1)

    assistant.interrupt()
    assert assistant._player.stopped == 1
    assert assistant.state == "idle"
    release.set()
    wait_state(assistant, "idle")
    assert play_calls == ["reply"]


def test_failure_sets_error_without_persisting_user_only_turn(tmp_path):
    assistant, _synth = make_assistant(tmp_path)

    def fail(_messages):
        raise OSError("network unavailable")

    assistant._request = fail
    assistant.submit("question")
    wait_state(assistant, "error")
    assert not assistant._path.exists()
    assert assistant._player.played == []


def test_close_prevents_late_response_playback(tmp_path):
    assistant, _synth = make_assistant(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def request(_messages):
        entered.set()
        release.wait(2)
        return "late"

    assistant._request = request
    assistant.submit("question")
    assert entered.wait(1)
    assistant.close()
    release.set()
    wait_state(assistant, "closed")
    assert assistant._player.played == []
    assert assistant._player.stopped == 1
