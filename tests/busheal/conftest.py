"""pytest entry point for tests/busheal: the fixtures and helpers live in bhelpers.py, a uniquely named module that test files
import from directly (two folders each importing a module called `conftest` collided depending on how pytest was invoked).
Everything is re-exported here, including the underscore-prefixed autouse fixtures a star-import would skip."""
import os as _os, sys as _sys, importlib.util as _ilu
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# why: these 74 tests exercise the real lerobot motor-bus classes, and CI installs only requirements-dev.txt; without lerobot the folder is not collected (pip install "lerobot[feetech]==0.6.1" to run it)
if _ilu.find_spec("lerobot") is None: collect_ignore_glob = ["test_*.py"]
else:
    import bhelpers as _h
    globals().update({_k: _v for _k, _v in vars(_h).items() if not _k.startswith("__")})
