"""Guard the daemon parent process against CUDA-initializing imports.

Process isolation only releases VRAM if the parent never imports
torch/NeMo/Kokoro — those must load exclusively inside the inference
worker process. This asserts the parent-side import graph stays clean.
"""

import subprocess
import sys

FORBIDDEN = ("torch", "torchaudio", "nemo", "kokoro", "speechbrain")

# Everything the daemon / HTTP server parent process imports. voiced.daemon
# and voiced.tray are excluded only because dasbus/gi come from the nix env,
# not the test venv; their remaining imports are covered transitively here.
PARENT_MODULES = (
    "voiced",
    "voiced.capabilities",
    "voiced.cli",
    "voiced.http_server",
    "voiced.profile_store",
    "voiced.recording_session",
    "voiced.server",
    "voiced.speaker_segments",
    "voiced.transcriber",
    "voiced.worker",
    "voiced.worker_host",
)


def test_parent_modules_do_not_import_cuda_stack():
    code = (
        "import sys\n"
        + "".join(f"import {m}\n" for m in PARENT_MODULES)
        + f"bad = [m for m in {FORBIDDEN!r} if m in sys.modules]\n"
        + "assert not bad, f'CUDA-adjacent modules imported in parent: {bad}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_building_voiced_from_config_stays_lazy():
    code = (
        "import sys\n"
        "from voiced.capabilities import Voiced\n"
        "from voiced.config import Config\n"
        "v = Voiced.from_config(Config())\n"
        "assert not v._worker_host.is_running, 'worker started eagerly'\n"
        f"bad = [m for m in {FORBIDDEN!r} if m in sys.modules]\n"
        "assert not bad, f'CUDA-adjacent modules imported: {bad}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
