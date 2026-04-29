"""Device, dtype, and attention-implementation resolution for torch models."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceConfig:
    """Resolved torch device, dtype, and attention implementation."""

    device: str
    dtype: torch.dtype
    attn_impl: str


def resolve_device_config(requested: str = "auto") -> DeviceConfig:
    """Resolve device, dtype, and attention implementation.

    Args:
        requested: "auto", "cuda", "mps", or "cpu". When "auto", picks the best
            available device (cuda > mps > cpu).

    Returns:
        DeviceConfig with the chosen device, dtype, and attention implementation.
    """
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
