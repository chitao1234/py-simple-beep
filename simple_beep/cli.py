"""CLI for beep compatibility with square-wave audio output."""

from __future__ import annotations

import logging
import os
import sys
import textwrap
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import __version__
from .audio import AudioError, AudioSpec, build_sequence_pcm, pcm_to_wav, select_backend

DEFAULT_FREQ = 440
DEFAULT_LENGTH_MS = 200
DEFAULT_REPEATS = 1
DEFAULT_DELAY_MS = 100
DEFAULT_DELAY_MODE = "between"  # "between" or "after"

HELP_TEXT = """
beep - beep with a square wave (beep-compatible)

Usage:
  beep [GLOBALS] [-f FREQ] [-l LEN] [-r REPEATS] [<-d|-D> DELAY] [-s] [-c]
  beep [GLOBALS] <TONE_OPTIONS> [-n|--new] <TONE_OPTIONS>
  beep <-h|--help>
  beep <-v|-V|--version>

Global options:
  -e DEVICE, --device=DEVICE  Select audio backend/device.
  --debug, --verbose          Increase logging verbosity.

Tone options:
  -f FREQ     Beep with a tone frequency of FREQ Hz (0 < FREQ < 20000). Default 440.
  -l LEN      Beep for a tone length of LEN milliseconds. Default 200.
  -r REPEATS  Repeat the tone including delays REPEATS times. Default 1.
  -d DELAY    Delay between repetitions (no delay after last). Default 100.
  -D DELAY    Delay after every repetition (including last).
  -n, --new   Start a new note with default values.
  -s          Beep after every line received on stdin, passing stdin to stdout.
  -c          Beep after every character received on stdin, passing stdin to stdout.

Environment:
  BEEP_LOG_LEVEL  Numeric log level (-999 to 999) used unless --debug/--verbose is set.

Notes:
  DEVICE may be specified as BACKEND or BACKEND:DEVICE. Available backends are
  auto-selected from: aplay, paplay, play, ffplay, simpleaudio.
  Use BACKEND=wav to write output to a WAV file (e.g. -e wav:beep.wav).
"""


class ParseError(ValueError):
    """Raised for CLI parsing errors."""


@dataclass
class Tone:
    frequency: int = DEFAULT_FREQ
    length_ms: int = DEFAULT_LENGTH_MS
    repeats: int = DEFAULT_REPEATS
    delay_ms: int = DEFAULT_DELAY_MS
    delay_mode: str = DEFAULT_DELAY_MODE
    input_mode: Optional[str] = None  # "line" or "char"
    _sequence_pcm: Optional[bytes] = field(default=None, init=False, repr=False)
    _sequence_wav: Optional[bytes] = field(default=None, init=False, repr=False)

    def sequence_pcm(self, spec: AudioSpec) -> bytes:
        if self._sequence_pcm is None:
            self._sequence_pcm = build_sequence_pcm(
                self.frequency,
                self.length_ms,
                self.repeats,
                self.delay_ms,
                self.delay_mode,
                spec,
            )
        return self._sequence_pcm

    def sequence_wav(self, spec: AudioSpec) -> bytes:
        if self._sequence_wav is None:
            self._sequence_wav = pcm_to_wav(self.sequence_pcm(spec), spec)
        return self._sequence_wav


@dataclass
class Config:
    tones: List[Tone]
    input_index: Optional[int]
    device: Optional[str]
    backend: Optional[str]
    log_level: int


def _log_level_from_env() -> Optional[int]:
    raw = os.environ.get("BEEP_LOG_LEVEL")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value >= 5:
        return logging.DEBUG
    if value >= 1:
        return logging.INFO
    if value <= -1:
        return logging.ERROR
    return logging.WARNING


def _configure_logger(level: int) -> logging.Logger:
    logger = logging.getLogger("simple_beep")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("beep: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def _parse_numeric(value: str, kind: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ParseError(f"Invalid {kind}: {value!r}") from exc


def _parse_frequency(value: str) -> int:
    freq = int(round(_parse_numeric(value, "frequency")))
    if freq <= 0 or freq >= 20000:
        raise ParseError("Frequency must be between 1 and 19999 Hz")
    return freq


def _parse_length(value: str) -> int:
    length = int(round(_parse_numeric(value, "length")))
    if length < 0:
        raise ParseError("Length must be >= 0")
    return length


def _parse_delay(value: str) -> int:
    delay = int(round(_parse_numeric(value, "delay")))
    if delay < 0:
        raise ParseError("Delay must be >= 0")
    return delay


def _parse_repeats(value: str) -> int:
    try:
        repeats = int(value)
    except ValueError as exc:
        raise ParseError(f"Invalid repeats: {value!r}") from exc
    if repeats < 1:
        raise ParseError("Repeats must be >= 1")
    return repeats


def _split_short_option(arg: str, opt: str) -> Optional[str]:
    if arg == opt:
        return None
    if arg.startswith(opt) and len(arg) > len(opt):
        return arg[len(opt) :]
    return None


def _consume_value(args: List[str], index: int, opt: str) -> Tuple[str, int]:
    if index + 1 >= len(args):
        raise ParseError(f"Missing value for {opt}")
    return args[index + 1], index + 1


def _parse_device_hint(device: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not device:
        return None, None
    backends = {"simpleaudio", "aplay", "paplay", "play", "ffplay", "wav"}
    if device in backends:
        return device, None
    if ":" not in device:
        return None, device
    prefix, rest = device.split(":", 1)
    if prefix in backends:
        return prefix, rest or None
    return None, device


def parse_args(argv: List[str]) -> Tuple[Optional[Config], Optional[int]]:
    tones = [Tone()]
    current = tones[-1]
    input_index: Optional[int] = None
    device: Optional[str] = None
    backend: Optional[str] = None
    log_level = _log_level_from_env() or logging.WARNING

    index = 0
    while index < len(argv):
        arg = argv[index]

        if arg in ("-h", "--help"):
            print(textwrap.dedent(HELP_TEXT).strip())
            return None, 0

        if arg in ("-v", "-V", "--version"):
            print(f"simple-beep {__version__}")
            return None, 0

        if arg in ("--debug", "--verbose"):
            log_level = logging.DEBUG
            index += 1
            continue

        if arg.startswith("--device="):
            device = arg.split("=", 1)[1]
            index += 1
            continue

        if arg in ("-e", "--device"):
            value, index = _consume_value(argv, index, arg)
            device = value
            index += 1
            continue

        if arg in ("-n", "--new"):
            current = Tone()
            tones.append(current)
            index += 1
            continue

        if arg in ("-s", "-c"):
            mode = "line" if arg == "-s" else "char"
            if current.input_mode and current.input_mode != mode:
                raise ParseError("Cannot combine -s and -c on the same note")
            if input_index is not None and input_index != len(tones) - 1:
                raise ParseError("Only one note can be marked with -s or -c")
            input_index = len(tones) - 1
            current.input_mode = mode
            index += 1
            continue

        value = _split_short_option(arg, "-f")
        if value is not None or arg == "-f":
            if value is None:
                value, index = _consume_value(argv, index, "-f")
            current.frequency = _parse_frequency(value)
            index += 1
            continue

        value = _split_short_option(arg, "-l")
        if value is not None or arg == "-l":
            if value is None:
                value, index = _consume_value(argv, index, "-l")
            current.length_ms = _parse_length(value)
            index += 1
            continue

        value = _split_short_option(arg, "-r")
        if value is not None or arg == "-r":
            if value is None:
                value, index = _consume_value(argv, index, "-r")
            current.repeats = _parse_repeats(value)
            index += 1
            continue

        value = _split_short_option(arg, "-d")
        if value is not None or arg == "-d":
            if value is None:
                value, index = _consume_value(argv, index, "-d")
            current.delay_ms = _parse_delay(value)
            current.delay_mode = "between"
            index += 1
            continue

        value = _split_short_option(arg, "-D")
        if value is not None or arg == "-D":
            if value is None:
                value, index = _consume_value(argv, index, "-D")
            current.delay_ms = _parse_delay(value)
            current.delay_mode = "after"
            index += 1
            continue

        raise ParseError(f"Unknown option: {arg}")

    backend, device = _parse_device_hint(device)

    config = Config(
        tones=tones,
        input_index=input_index,
        device=device,
        backend=backend,
        log_level=log_level,
    )
    return config, None


def _write_wav_sequence(config: Config, logger: logging.Logger) -> int:
    spec = AudioSpec()
    path = config.device or "beep.wav"
    try:
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(spec.channels)
            wav_file.setsampwidth(spec.sample_width)
            wav_file.setframerate(spec.sample_rate)

            def play_note(tone: Tone) -> None:
                wav_file.writeframes(tone.sequence_pcm(spec))

            if config.input_index is None:
                for tone in config.tones:
                    play_note(tone)
                return 0

            for idx, tone in enumerate(config.tones):
                if idx < config.input_index:
                    play_note(tone)
                    continue

                if idx == config.input_index:
                    _process_input(tone, play_note)
                    continue

                play_note(tone)
    except OSError as exc:
        logger.error("Unable to write WAV output: %s", exc)
        return 1
    return 0


def _play_sequence(config: Config, logger: logging.Logger) -> int:
    spec = AudioSpec()
    if config.backend == "wav":
        return _write_wav_sequence(config, logger)
    backend = select_backend(config.backend)

    if config.device and not backend.supports_device:
        logger.warning("Backend '%s' does not support device selection", backend.name)

    with backend.get_stream(spec, config.device) as stream:
        # `simpleaudio` doesn't provide a true gapless streaming API; each call to
        # `play_buffer()` has non-trivial startup overhead, which shows up as
        # extra delay/jitter between notes if we `.write()` multiple chunks.
        #
        # For non-interactive playback, concatenate into one contiguous buffer
        # and play it in a single call to restore timing.
        if backend.name == "simpleaudio" and config.input_index is None:
            stream.write(b"".join(tone.sequence_pcm(spec) for tone in config.tones))
            return 0

        def play_note(tone: Tone) -> None:
            stream.write(tone.sequence_pcm(spec))

        if config.input_index is None:
            for tone in config.tones:
                play_note(tone)
            return 0

        for idx, tone in enumerate(config.tones):
            if idx < config.input_index:
                play_note(tone)
                continue

            if idx == config.input_index:
                _process_input(tone, play_note)
                continue

            play_note(tone)

    return 0


def _process_input(tone: Tone, play_note) -> None:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    if tone.input_mode == "char":
        while True:
            chunk = stdin.read(4096)
            if not chunk:
                break
            for byte in chunk:
                try:
                    stdout.write(bytes((byte,)))
                    stdout.flush()
                except BrokenPipeError:
                    return
                play_note(tone)
    else:
        while True:
            chunk = stdin.readline()
            if not chunk:
                break
            try:
                stdout.write(chunk)
                stdout.flush()
            except BrokenPipeError:
                return
            play_note(tone)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        config, exit_code = parse_args(argv)
    except ParseError as exc:
        print(f"beep: {exc}", file=sys.stderr)
        print("Try 'beep --help' for more information.", file=sys.stderr)
        return 1

    if exit_code is not None:
        return exit_code

    logger = _configure_logger(config.log_level)

    try:
        return _play_sequence(config, logger)
    except AudioError as exc:
        logger.error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
