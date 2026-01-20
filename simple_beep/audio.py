"""Audio synthesis and playback helpers for the beep emulation."""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import wave
from array import array
from dataclasses import dataclass
from typing import Callable, Optional

SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2  # bytes
CHANNELS = 1
AMPLITUDE = 0.6


class AudioError(RuntimeError):
    """Raised when audio output fails."""


@dataclass(frozen=True)
class AudioSpec:
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    sample_width: int = SAMPLE_WIDTH
    amplitude: float = AMPLITUDE


@dataclass
class WaveData:
    pcm: bytes
    wav: bytes
    spec: AudioSpec


def square_wave_pcm(freq_hz: int, length_ms: int, spec: AudioSpec) -> bytes:
    if length_ms <= 0:
        return b""
    num_samples = int(round(spec.sample_rate * (length_ms / 1000.0)))
    if num_samples <= 0:
        return b""
    max_amp = int(32767 * spec.amplitude)
    phase = 0.0
    step = freq_hz / spec.sample_rate
    samples = array("h")
    for _ in range(num_samples):
        samples.append(max_amp if phase < 0.5 else -max_amp)
        phase += step
        if phase >= 1.0:
            phase -= int(phase)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def silence_pcm(length_ms: int, spec: AudioSpec) -> bytes:
    if length_ms <= 0:
        return b""
    num_samples = int(round(spec.sample_rate * (length_ms / 1000.0)))
    if num_samples <= 0:
        return b""
    return b"\x00" * (num_samples * spec.sample_width)


def pcm_to_wav(pcm_bytes: bytes, spec: AudioSpec) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(spec.channels)
        wav_file.setsampwidth(spec.sample_width)
        wav_file.setframerate(spec.sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def build_sequence_pcm(
    freq_hz: int,
    length_ms: int,
    repeats: int,
    delay_ms: int,
    delay_mode: str,
    spec: AudioSpec,
) -> bytes:
    tone_pcm = square_wave_pcm(freq_hz, length_ms, spec)
    if repeats <= 1:
        if delay_mode == "after" and delay_ms > 0:
            return tone_pcm + silence_pcm(delay_ms, spec)
        return tone_pcm

    delay_pcm = silence_pcm(delay_ms, spec) if delay_ms > 0 else b""
    parts = []
    for idx in range(repeats):
        parts.append(tone_pcm)
        if delay_mode == "after" or idx < repeats - 1:
            if delay_pcm:
                parts.append(delay_pcm)
    return b"".join(parts)


@dataclass
class Backend:
    name: str
    input_format: str
    supports_device: bool
    available: Callable[[], bool]
    play: Callable[[bytes, bytes, AudioSpec, Optional[str]], None]


def _simpleaudio_available() -> bool:
    try:
        import simpleaudio  # noqa: F401
    except Exception:
        return False
    return True


def _simpleaudio_play(
    wav_bytes: bytes, pcm_bytes: bytes, spec: AudioSpec, device: Optional[str]
) -> None:
    try:
        import simpleaudio as sa
    except Exception as exc:
        raise AudioError("simpleaudio is not available") from exc
    if device:
        # simpleaudio does not support device selection.
        pass
    play_obj = sa.play_buffer(pcm_bytes, spec.channels, spec.sample_width, spec.sample_rate)
    play_obj.wait_done()


def _subprocess_play(cmd: list[str], data: bytes) -> None:
    try:
        subprocess.run(cmd, input=data, check=True)
    except subprocess.CalledProcessError as exc:
        raise AudioError(f"Audio backend failed: {' '.join(cmd)}") from exc
    except FileNotFoundError as exc:
        raise AudioError(f"Audio backend missing: {' '.join(cmd)}") from exc


def _aplay_available() -> bool:
    return shutil.which("aplay") is not None


def _aplay_play(
    wav_bytes: bytes, pcm_bytes: bytes, spec: AudioSpec, device: Optional[str]
) -> None:
    cmd = ["aplay", "-q"]
    if device:
        cmd.extend(["-D", device])
    cmd.extend(["-t", "wav", "-"])
    _subprocess_play(cmd, wav_bytes)


def _paplay_available() -> bool:
    return shutil.which("paplay") is not None


def _paplay_play(
    wav_bytes: bytes, pcm_bytes: bytes, spec: AudioSpec, device: Optional[str]
) -> None:
    cmd = [
        "paplay",
        "--raw",
        f"--rate={spec.sample_rate}",
        f"--channels={spec.channels}",
        "--format=s16le",
    ]
    if device:
        cmd.extend(["--device", device])
    _subprocess_play(cmd, pcm_bytes)


def _play_available() -> bool:
    return shutil.which("play") is not None


def _play_play(
    wav_bytes: bytes, pcm_bytes: bytes, spec: AudioSpec, device: Optional[str]
) -> None:
    cmd = ["play", "-q", "-t", "wav", "-"]
    _subprocess_play(cmd, wav_bytes)


def _ffplay_available() -> bool:
    return shutil.which("ffplay") is not None


def _ffplay_play(
    wav_bytes: bytes, pcm_bytes: bytes, spec: AudioSpec, device: Optional[str]
) -> None:
    cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "-"]
    _subprocess_play(cmd, wav_bytes)


BACKENDS = [
    Backend(
        name="aplay",
        input_format="wav",
        supports_device=True,
        available=_aplay_available,
        play=_aplay_play,
    ),
    Backend(
        name="paplay",
        input_format="pcm",
        supports_device=True,
        available=_paplay_available,
        play=_paplay_play,
    ),
    Backend(
        name="play",
        input_format="wav",
        supports_device=False,
        available=_play_available,
        play=_play_play,
    ),
    Backend(
        name="ffplay",
        input_format="wav",
        supports_device=False,
        available=_ffplay_available,
        play=_ffplay_play,
    ),
    Backend(
        name="simpleaudio",
        input_format="pcm",
        supports_device=False,
        available=_simpleaudio_available,
        play=_simpleaudio_play,
    ),
]


def select_backend(preferred: Optional[str]) -> Backend:
    if preferred:
        for backend in BACKENDS:
            if backend.name == preferred:
                if backend.available():
                    return backend
                raise AudioError(f"Requested backend '{preferred}' is not available")
        raise AudioError(f"Unknown backend '{preferred}'")

    for backend in BACKENDS:
        if backend.available():
            return backend
    raise AudioError(
        "No audio backend available. Install simpleaudio or aplay/paplay/play/ffplay."
    )


def describe_backends() -> str:
    names = [backend.name for backend in BACKENDS]
    return ", ".join(names)
