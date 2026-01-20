# simple-beep

A Python replacement for the Linux `beep` command that emits square-wave audio
through standard audio outputs instead of the PC speaker. It follows the
`beep(1)` option semantics (frequency, length, repeats, delays, and `--new`
chaining) and supports `-s`/`-c` stdin monitoring.

## Quick start

```bash
python -m simple_beep
python -m simple_beep -f 880 -l 100
```

To expose a `beep` command:

```bash
python -m pip install -e .
beep -f 750 -r 3 -d 120
```

## Audio backends

The command auto-selects the first available backend in this order:

1. `aplay` (ALSA)
2. `paplay` (PulseAudio)
3. `play` (SoX)
4. `ffplay` (FFmpeg)
5. `simpleaudio` (Python package)

Install any one of these if you do not already have one.

### Selecting a backend or device

Use `-e`/`--device` to control output. The value can be either:

- `BACKEND` (force a backend, e.g. `aplay`)
- `BACKEND:DEVICE` (backend plus device string)
- `wav:PATH` (write output to a WAV file)

Examples:

```bash
beep -e aplay
beep -e aplay:default
beep -e paplay:alsa_output.pci-0000_00_1b.0.analog-stereo
beep -e wav:beep.wav
```

If you pass a bare device string (e.g. `-e hw:0,0`) the backend is auto-chosen
and the string is forwarded to backends that support device selection.

## Notes on compatibility

- Options, defaults, and `--new` chaining mirror `beep(1)`.
- Frequencies are rounded to integer Hz like the kernel APIs used by `beep`.
- `-s` and `-c` pass stdin to stdout and beep on lines or characters.
- The PC speaker is not used; square-wave audio is rendered to standard audio
  backends instead.

## License

This project is provided as-is. Add a license file if you plan to distribute it.
