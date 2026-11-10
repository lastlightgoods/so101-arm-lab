"""pytest entry point for tests/tools: the fixtures live in toolhelpers.py, a uniquely named module the test files import from
directly (several folders each importing a module called `conftest` collided depending on how pytest was invoked)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import toolhelpers as _h  # noqa: E402
globals().update({_k: _v for _k, _v in vars(_h).items() if not _k.startswith("__")})
