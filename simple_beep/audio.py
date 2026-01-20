"""Audio synthesis and playback helpers for the beep emulation."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
import sys
import wave
from array import array
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Protocol


SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2  # bytes
CHANNELS = 1
AMPLITUDE = 0.6

WaveType = Literal["sine", "square", "sawtooth", "triangle"]


class AudioError(RuntimeError):
    """Raised when audio output fails."""


class AudioStream(Protocol):
    def write(self, pcm_bytes: bytes) -> None:
        ...

    def __enter__(self) -> AudioStream:
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ...


class SubprocessStream(AudioStream):
    def __init__(self, cmd: list[str]) -> None:
        self.cmd = cmd
        self._proc: Optional[subprocess.Popen[bytes]] = None

    def __enter__(self) -> AudioStream:
        try:
            self._proc = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise AudioError(f"Audio backend missing: {self.cmd[0]}") from exc
        return self

    def write(self, pcm_bytes: bytes) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise AudioError("Stream is not open")
        try:
            self._proc.stdin.write(pcm_bytes)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            _, stderr = self._proc.communicate()
            raise AudioError(
                f"Audio backend failed: {self.cmd[0]}\n{stderr.decode().strip()}"
            ) from exc

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._proc:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait()
            self._proc = None


class SimpleAudioStream(AudioStream):
    def __init__(self, spec: AudioSpec) -> None:
        self.spec = spec
        self._sa: Any = None

    def __enter__(self) -> AudioStream:
        try:
            import simpleaudio as sa

            self._sa = sa
        except ImportError as exc:
            raise AudioError("simpleaudio is not installed") from exc
        return self

    def write(self, pcm_bytes: bytes) -> None:
        if self._sa is None:
            raise AudioError("simpleaudio is not initialized")
        play_obj = self._sa.play_buffer(
            pcm_bytes,
            self.spec.channels,
            self.spec.sample_width,
            self.spec.sample_rate,
        )
        play_obj.wait_done()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class SoundDeviceStream(AudioStream):
    def __init__(self, spec: AudioSpec, device: Optional[str]) -> None:
        self.spec = spec
        self.device = device
        self._sd: Any = None
        self._stream: Any = None
        self._total_samples = 0

    def __enter__(self) -> AudioStream:
        try:
            import sounddevice as sd

            self._sd = sd
            self._stream = sd.RawOutputStream(
                samplerate=self.spec.sample_rate,
                channels=self.spec.channels,
                dtype="int16",
                device=self.device,
            )
            self._stream.start()
            self._start_time = self._stream.time
            self._total_samples = 0
        except ImportError as exc:
            raise AudioError("sounddevice is not installed") from exc
        except Exception as exc:
            raise AudioError(f"sounddevice failed to open: {exc}") from exc
        return self

    def write(self, pcm_bytes: bytes) -> None:
        if self._stream is None:
            raise AudioError("sounddevice stream is not open")
        try:
            self._stream.write(pcm_bytes)
            self._total_samples += len(pcm_bytes) // (
                self.spec.sample_width * self.spec.channels
            )
        except Exception as exc:
            raise AudioError(f"sounddevice write failed: {exc}") from exc

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._stream is not None:
            try:
                # Wait for playback to finish. RawOutputStream.write() blocks until
                # data is in the buffer, but not until it's played.
                import time

                end_time = self._start_time + self._total_samples / self.spec.sample_rate
                while self._stream.time < end_time:
                    time.sleep(0.005)
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


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


def sine_wave_pcm(freq_hz: int, length_ms: int, spec: AudioSpec) -> bytes:
    if length_ms <= 0:
        return b""
    num_samples = int(round(spec.sample_rate * (length_ms / 1000.0)))
    if num_samples <= 0:
        return b""
    max_amp = int(32767 * spec.amplitude)
    samples = array("h")
    factor = 2 * math.pi * freq_hz / spec.sample_rate
    for i in range(num_samples):
        samples.append(int(round(max_amp * math.sin(factor * i))))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def sawtooth_wave_pcm(freq_hz: int, length_ms: int, spec: AudioSpec) -> bytes:
    if length_ms <= 0:
        return b""
    num_samples = int(round(spec.sample_rate * (length_ms / 1000.0)))
    if num_samples <= 0:
        return b""
    max_amp = int(32767 * spec.amplitude)
    samples = array("h")
    phase = 0.0
    step = freq_hz / spec.sample_rate
    for _ in range(num_samples):
        # Sawtooth from -max_amp to max_amp
        samples.append(int(round(max_amp * (2.0 * phase - 1.0))))
        phase += step
        if phase >= 1.0:
            phase -= int(phase)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def triangle_wave_pcm(freq_hz: int, length_ms: int, spec: AudioSpec) -> bytes:
    if length_ms <= 0:
        return b""
    num_samples = int(round(spec.sample_rate * (length_ms / 1000.0)))
    if num_samples <= 0:
        return b""
    max_amp = int(32767 * spec.amplitude)
    samples = array("h")
    phase = 0.0
    step = freq_hz / spec.sample_rate
    for _ in range(num_samples):
        if phase < 0.25:
            val = 4.0 * phase
        elif phase < 0.75:
            val = 2.0 - 4.0 * phase
        else:
            val = 4.0 * phase - 4.0
        samples.append(int(round(max_amp * val)))
        phase += step
        if phase >= 1.0:
            phase -= int(phase)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


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
    wave_type: WaveType = "square",
) -> bytes:
    if wave_type == "sine":
        tone_pcm = sine_wave_pcm(freq_hz, length_ms, spec)
    elif wave_type == "sawtooth":
        tone_pcm = sawtooth_wave_pcm(freq_hz, length_ms, spec)
    elif wave_type == "triangle":
        tone_pcm = triangle_wave_pcm(freq_hz, length_ms, spec)
    else:
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
    supports_device: bool
    available: Callable[[], bool]
    get_stream: Callable[[AudioSpec, Optional[str]], AudioStream]

    def play(
        self, wav_bytes: bytes, pcm_bytes: bytes, spec: AudioSpec, device: Optional[str]
    ) -> None:
        with self.get_stream(spec, device) as stream:
            stream.write(pcm_bytes)


def _aplay_available() -> bool:
    return shutil.which("aplay") is not None


def _paplay_available() -> bool:
    return shutil.which("paplay") is not None


def _play_available() -> bool:
    return shutil.which("play") is not None


def _ffplay_available() -> bool:
    return shutil.which("ffplay") is not None


def _simpleaudio_available() -> bool:
    try:
        import simpleaudio  # noqa: F401
    except Exception:
        return False
    return True


def _sounddevice_available() -> bool:
    try:
        import sounddevice  # noqa: F401
    except Exception:
        return False
    return True


def _simpleaudio_stream(spec: AudioSpec, device: Optional[str]) -> AudioStream:
    return SimpleAudioStream(spec)


def _sounddevice_stream(spec: AudioSpec, device: Optional[str]) -> AudioStream:
    return SoundDeviceStream(spec, device)


def _aplay_stream(spec: AudioSpec, device: Optional[str]) -> AudioStream:
    cmd = [
        "aplay",
        "-q",
        "-t",
        "raw",
        f"-f",
        "S16_LE",
        "-c",
        str(spec.channels),
        "-r",
        str(spec.sample_rate),
    ]
    if device:
        cmd.extend(["-D", device])
    cmd.append("-")
    return SubprocessStream(cmd)


def _paplay_stream(spec: AudioSpec, device: Optional[str]) -> AudioStream:
    cmd = [
        "paplay",
        "--raw",
        f"--rate={spec.sample_rate}",
        f"--channels={spec.channels}",
        "--format=s16le",
    ]
    if device:
        cmd.extend(["--device", device])
    return SubprocessStream(cmd)


def _play_stream(spec: AudioSpec, device: Optional[str]) -> AudioStream:
    cmd = [
        "play",
        "-q",
        "-t",
        "raw",
        "-e",
        "signed-integer",
        "-b",
        "16",
        "-c",
        str(spec.channels),
        "-r",
        str(spec.sample_rate),
        "-",
    ]
    return SubprocessStream(cmd)


def _ffplay_stream(spec: AudioSpec, device: Optional[str]) -> AudioStream:
    cmd = [
        "ffplay",
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "quiet",
        "-f",
        "s16le",
        "-ac",
        str(spec.channels),
        "-ar",
        str(spec.sample_rate),
        "-i",
        "-",
    ]
    return SubprocessStream(cmd)


BACKENDS = [
    Backend(
        name="aplay",
        supports_device=True,
        available=_aplay_available,
        get_stream=_aplay_stream,
    ),
    Backend(
        name="paplay",
        supports_device=True,
        available=_paplay_available,
        get_stream=_paplay_stream,
    ),
    Backend(
        name="play",
        supports_device=False,
        available=_play_available,
        get_stream=_play_stream,
    ),
    Backend(
        name="ffplay",
        supports_device=False,
        available=_ffplay_available,
        get_stream=_ffplay_stream,
    ),
    Backend(
        name="simpleaudio",
        supports_device=False,
        available=_simpleaudio_available,
        get_stream=_simpleaudio_stream,
    ),
    Backend(
        name="sounddevice",
        supports_device=True,
        available=_sounddevice_available,
        get_stream=_sounddevice_stream,
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
