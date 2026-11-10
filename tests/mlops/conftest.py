"""pytest entry point for tests/mlops: the fixtures and helpers live in mlhelpers.py (a uniquely named module the test files
import from directly); everything is re-exported here so pytest sees the fixtures."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import mlhelpers as _h  # noqa: E402
globals().update({_k: _v for _k, _v in vars(_h).items() if not _k.startswith("__")})
