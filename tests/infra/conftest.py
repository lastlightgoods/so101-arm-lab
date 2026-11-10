"""pytest entry point for tests/infra: the helpers live in infrahelpers.py, a uniquely named module that test files import from
directly (two folders each importing a module called `conftest` collided depending on how pytest was invoked).
Everything is re-exported here so fixtures defined there are visible to pytest."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import infrahelpers as _h  # noqa: E402
globals().update({_k: _v for _k, _v in vars(_h).items() if not _k.startswith("__")})
