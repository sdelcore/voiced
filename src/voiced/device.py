"""Device, dtype, and attention-implementation resolution for torch models.

torch is imported lazily inside ``resolve_device_config`` so that modules
which merely import this one (transcriber, capabilities, daemon) stay free
of torch in the parent process — resolution only happens in the inference
worker or in short-lived CLI processes.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceConfig:
    """Resolved torch device, dtype, and attention implementation."""

    device: str
    dtype: Any
    attn_impl: str


def resolve_device_config(requested: str = "auto") -> DeviceConfig:
    """Resolve device, dtype, and attention implementation.

    Args:
        requested: "auto", "cuda", "mps", or "cpu". When "auto", picks the best
            available device (cuda > mps > cpu).

    Returns:
        DeviceConfig with the chosen device, dtype, and attention implementation.
    """
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return DeviceConfig("cuda", torch.bfloat16, "flash_attention_2")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceConfig("mps", torch.float32, "sdpa")
        return DeviceConfig("cpu", torch.float32, "sdpa")

    if requested == "cuda":
        return DeviceConfig("cuda", torch.bfloat16, "flash_attention_2")
    if requested == "mps":
        return DeviceConfig("mps", torch.float32, "sdpa")
    return DeviceConfig("cpu", torch.float32, "sdpa")
