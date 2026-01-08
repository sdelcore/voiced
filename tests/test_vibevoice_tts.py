#!/usr/bin/env python3
"""Test script for VibeVoice Realtime 0.5B TTS.

This script demonstrates real-time TTS using the VibeVoice model from HuggingFace.
The model weights are downloaded from HuggingFace (microsoft/VibeVoice-Realtime-0.5B).
Voice presets (.pt files) contain pre-cached speaker embeddings, NOT model weights.

Usage:
    python tests/test_vibevoice_tts.py [--text "Your text here"] [--voice carter|emma|davis|...]
    python tests/test_vibevoice_tts.py --save output.wav  # Save instead of play
    python tests/test_vibevoice_tts.py --stream           # Low-latency streaming playback

Requirements:
    pip install git+https://github.com/microsoft/VibeVoice.git
    # Or: git clone https://github.com/microsoft/VibeVoice.git && cd VibeVoice && pip install -e .
"""

import argparse
import copy
import logging
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import sounddevice as sd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model and voice preset configuration
MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
SAMPLE_RATE = 24000

# Voice presets available in the VibeVoice repo
VOICE_PRESETS = {
    "carter": "en-Carter_man.pt",
    "davis": "en-Davis_man.pt",
    "emma": "en-Emma_woman.pt",
    "frank": "en-Frank_man.pt",
    "grace": "en-Grace_woman.pt",
    "mike": "en-Mike_man.pt",
}

# Cache directory for voice presets
CACHE_DIR = Path.home() / ".cache" / "voiced" / "vibevoice"


def download_voice_preset(voice_name: str) -> Path:
    """Download a voice preset from GitHub if not cached."""
    voice_file = VOICE_PRESETS.get(voice_name.lower())
    if not voice_file:
        available = ", ".join(VOICE_PRESETS.keys())
        raise ValueError(f"Unknown voice '{voice_name}'. Available: {available}")

    cache_path = CACHE_DIR / voice_file
    if cache_path.exists():
        logger.info(f"Using cached voice preset: {cache_path}")
        return cache_path

    # Download from GitHub using requests (handles SSL better)
    url = f"https://raw.githubusercontent.com/microsoft/VibeVoice/main/demo/voices/streaming_model/{voice_file}"
    logger.info(f"Downloading voice preset from {url}...")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import requests

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(response.content)
    logger.info(f"Downloaded voice preset to {cache_path}")
    return cache_path


def check_vibevoice_installed() -> bool:
    """Check if VibeVoice is installed."""
    try:
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

        return True
    except ImportError:
        return False


def get_device_and_dtype():
    """Determine the best device and dtype for the current system."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16, "flash_attention_2"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32, "sdpa"
    else:
        return "cpu", torch.float32, "sdpa"


def load_model_and_processor(device: str, dtype: torch.dtype, attn_impl: str):
    """Load the VibeVoice model and processor from HuggingFace."""
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )

    logger.info(f"Loading processor from {MODEL_PATH}...")
    processor = VibeVoiceStreamingProcessor.from_pretrained(MODEL_PATH)

    logger.info(f"Loading model from {MODEL_PATH} (device={device}, dtype={dtype})...")
    try:
        if device == "mps":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                MODEL_PATH,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                device_map=None,
            )
            model.to("mps")
        else:
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                MODEL_PATH,
                torch_dtype=dtype,
                device_map=device,
                attn_implementation=attn_impl,
            )
    except Exception as e:
        if attn_impl == "flash_attention_2":
            logger.warning(f"Flash attention failed: {e}")
            logger.info("Falling back to SDPA attention...")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                MODEL_PATH,
                torch_dtype=dtype,
                device_map=device if device != "mps" else None,
                attn_implementation="sdpa",
            )
            if device == "mps":
                model.to("mps")
        else:
            raise

    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)  # Fast inference
    return model, processor


def generate_speech(
    model,
    processor,
    text: str,
    voice_cache: dict,
    device: str,
    cfg_scale: float = 1.5,
) -> np.ndarray:
    """Generate speech audio from text."""
    logger.info(f"Generating speech for: {text[:50]}...")

    # Prepare inputs
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=voice_cache,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    # Move tensors to device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    # Generate
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        cfg_scale=cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        show_progress_bar=True,
        all_prefilled_outputs=copy.deepcopy(voice_cache),
    )
    gen_time = time.time() - start_time

    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        audio = outputs.speech_outputs[0]
        logger.info(
            f"Raw audio type: {type(audio)}, shape: {audio.shape if hasattr(audio, 'shape') else 'N/A'}"
        )

        if torch.is_tensor(audio):
            # Convert bfloat16 to float32 before numpy (numpy doesn't support bfloat16)
            audio = audio.float().cpu().numpy()

        # Ensure audio is 1D (squeeze extra dimensions)
        audio = np.squeeze(audio)
        logger.info(f"Processed audio shape: {audio.shape}")

        # Calculate duration based on total samples
        total_samples = audio.shape[-1] if len(audio.shape) > 0 else len(audio)
        duration = total_samples / SAMPLE_RATE
        rtf = gen_time / duration if duration > 0 else float("inf")
        logger.info(f"Generated {duration:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.2f}x)")
        return audio
    else:
        raise RuntimeError("No audio output generated")


def play_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Play audio using sounddevice."""
    logger.info("Playing audio...")
    sd.play(audio, sample_rate)
    sd.wait()
    logger.info("Playback complete")


def save_audio(audio: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE):
    """Save audio to a WAV file."""
    import soundfile as sf

    sf.write(path, audio, sample_rate)
    logger.info(f"Saved audio to {path}")


def play_audio_streaming(
    model,
    processor,
    text: str,
    voice_cache: dict,
    device: str,
    cfg_scale: float = 1.5,
) -> None:
    """Generate and play audio with streaming (low latency, callback-based)."""
    from vibevoice.modular.streamer import AudioStreamer

    logger.info(f"Streaming speech for: {text[:50]}...")

    # Prepare inputs
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=voice_cache,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    # Audio buffer queue for sounddevice callback
    audio_queue: Queue = Queue(maxsize=100)
    playback_finished = threading.Event()

    # Track progress
    samples_generated = 0
    samples_played = 0
    start_time = time.time()
    first_chunk_time = None
    lock = threading.Lock()

    def audio_callback(outdata, frames, time_info, status):
        """Called by sounddevice to fill output buffer."""
        nonlocal samples_played
        if status:
            logger.warning(f"Sounddevice status: {status}")

        filled = 0
        while filled < frames:
            try:
                chunk = audio_queue.get_nowait()
                if chunk is None:  # Stop signal
                    outdata[filled:, 0] = 0
                    raise sd.CallbackStop
                chunk_len = len(chunk)
                space_left = frames - filled
                if chunk_len <= space_left:
                    outdata[filled : filled + chunk_len, 0] = chunk
                    filled += chunk_len
                else:
                    outdata[filled:, 0] = chunk[:space_left]
                    # Put remainder back at front (we'll use a deque-like approach)
                    # For simplicity, just put it back and it will be picked up next
                    leftover = chunk[space_left:]
                    # Create a temporary queue to preserve order
                    temp_items = [leftover]
                    while not audio_queue.empty():
                        try:
                            temp_items.append(audio_queue.get_nowait())
                        except Empty:
                            break
                    for item in temp_items:
                        audio_queue.put(item)
                    filled = frames
                with lock:
                    samples_played += min(chunk_len, space_left)
            except Empty:
                # Buffer underrun - fill rest with silence
                outdata[filled:, 0] = 0
                filled = frames

    def on_stream_finished():
        playback_finished.set()

    # Create VibeVoice audio streamer
    vv_streamer = AudioStreamer(batch_size=1, stop_signal=None)
    generation_error = []

    def generate_thread():
        """Background thread for model generation."""
        try:
            model.generate(
                **inputs,
                cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                show_progress_bar=False,
                audio_streamer=vv_streamer,
                all_prefilled_outputs=copy.deepcopy(voice_cache),
            )
        except Exception as e:
            generation_error.append(e)
            vv_streamer.end()

    def feeder_thread():
        """Feeds chunks from VibeVoice streamer to audio queue."""
        nonlocal samples_generated, first_chunk_time

        try:
            for chunk in vv_streamer.get_stream(0):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - start_time
                    logger.info(f"First chunk latency: {latency * 1000:.0f}ms")

                # Convert to float32 numpy
                if torch.is_tensor(chunk):
                    chunk = chunk.float().cpu().numpy()
                chunk = np.squeeze(chunk).astype(np.float32)

                with lock:
                    samples_generated += len(chunk)

                audio_queue.put(chunk)
        finally:
            # Add silence padding to let final audio fully drain through speakers
            # This prevents the last word from being cut off
            silence = np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32)  # 250ms
            audio_queue.put(silence)
            audio_queue.put(None)  # Signal end

    # Start threads
    gen_thread = threading.Thread(target=generate_thread, daemon=True)
    feed_thread = threading.Thread(target=feeder_thread, daemon=True)

    gen_thread.start()
    feed_thread.start()

    # Wait for first chunk before starting playback
    logger.info("Waiting for first audio chunk...")
    while audio_queue.empty() and gen_thread.is_alive():
        time.sleep(0.01)

    if audio_queue.empty() and not gen_thread.is_alive():
        if generation_error:
            raise generation_error[0]
        raise RuntimeError("Generation finished without producing audio")

    # Start audio output stream
    logger.info("Starting audio playback...")
    try:
        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            callback=audio_callback,
            finished_callback=on_stream_finished,
            blocksize=2048,  # ~85ms blocks
        ):
            # Progress bar while playing
            with tqdm(
                desc="Streaming",
                unit="s",
                bar_format="{desc}: {n:.1f}s generated | {elapsed} elapsed | {rate_fmt}",
                dynamic_ncols=True,
            ) as pbar:
                while not playback_finished.is_set():
                    time.sleep(0.1)
                    with lock:
                        current = samples_generated / SAMPLE_RATE
                    pbar.n = current
                    pbar.refresh()

                # Final update
                with lock:
                    pbar.n = samples_generated / SAMPLE_RATE
                pbar.refresh()

    except sd.CallbackStop:
        pass  # Normal termination
    except Exception as e:
        logger.error(f"Playback error: {e}")

    # Wait for threads
    gen_thread.join(timeout=5)
    feed_thread.join(timeout=5)

    # Report results
    total_time = time.time() - start_time
    duration = samples_generated / SAMPLE_RATE
    rtf = total_time / duration if duration > 0 else float("inf")

    logger.info(f"Playback complete: {duration:.2f}s audio in {total_time:.2f}s (RTF: {rtf:.2f}x)")

    if generation_error:
        raise generation_error[0]


def main():
    parser = argparse.ArgumentParser(description="Test VibeVoice Realtime 0.5B TTS")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! This is a test of the VibeVoice real-time text to speech system. "
        "It can generate natural sounding speech with very low latency.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="carter",
        choices=list(VOICE_PRESETS.keys()),
        help="Voice preset to use",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save audio to file instead of playing",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale (default: 1.5)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream audio playback with low latency (~300ms first chunk)",
    )
    args = parser.parse_args()

    # Validate: --stream and --save are mutually exclusive
    if args.stream and args.save:
        print("Error: --stream and --save cannot be used together")
        sys.exit(1)

    # Check if VibeVoice is installed
    if not check_vibevoice_installed():
        print("VibeVoice is not installed. Please install it first:")
        print("  pip install git+https://github.com/microsoft/VibeVoice.git")
        print(
            "  # Or: git clone https://github.com/microsoft/VibeVoice.git && cd VibeVoice && pip install -e ."
        )
        sys.exit(1)

    # Get device configuration
    device, dtype, attn_impl = get_device_and_dtype()
    logger.info(f"Using device: {device}, dtype: {dtype}, attention: {attn_impl}")

    # Download voice preset
    voice_path = download_voice_preset(args.voice)

    # Load voice cache (pre-computed speaker embeddings)
    logger.info(f"Loading voice preset: {voice_path}")
    voice_cache = torch.load(voice_path, map_location=device, weights_only=False)

    # Load model and processor from HuggingFace
    model, processor = load_model_and_processor(device, dtype, attn_impl)

    # Generate/play based on mode
    if args.stream:
        # Streaming mode: low latency playback
        play_audio_streaming(
            model,
            processor,
            args.text,
            voice_cache,
            device,
            cfg_scale=args.cfg_scale,
        )
    else:
        # Batch mode: generate all then play/save
        audio = generate_speech(
            model,
            processor,
            args.text,
            voice_cache,
            device,
            cfg_scale=args.cfg_scale,
        )

        if args.save:
            save_audio(audio, args.save)
        else:
            play_audio(audio)


if __name__ == "__main__":
    main()
