# Contributing

Small, tested changes are welcome. The ground rules, which the test suite and CI enforce where they can:

- **No hardware in tests.** `python -m pytest tests -q` must pass on a laptop with nothing plugged in; it does today
  (about 3 minutes). Cameras, buses, `v4l2-ctl`, `ssh` and the Jetson are faked in the tests, never contacted.
- **A bug fix ships with the test that fails without it.**
- **Measured numbers only** in docs and commit messages. If you quote a residual, a frame rate or a timing, say where it
  was measured.
- **Keep secrets and addresses out.** No LAN addresses, tokens, passphrases or home paths in committed files; CI's
  push protection will refuse known secret formats, but the rule covers everything.
- **Style:** `ruff check .` with the repo's `pyproject.toml`. The code is deliberately dense; keep a module docstring
  that says how to run the file, and a short `# why:` comment on any decision that is not obvious from the code.
- **Do not rename** existing `so101-*` commands, HTTP routes or JSON keys; the Jetson copy of `jetson/`, the
  dashboard and the docs depend on them.

Open an issue first for anything that changes a calibration convention (mat frame, joint offsets, marker layout) so
the stored calibration files and the docs move together.
