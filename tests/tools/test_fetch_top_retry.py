"""tools/pick_loop.py fetch_top retries the dashboard's 404 (its newest frame older than 3 s): 11 Nov 2026 18:2x a 404 right after the place
crashed the pick after the arm had finished, before the record and the state were written."""
import importlib.util, io, os, urllib.error
import pytest
from pickhelpers import TOOLS


@pytest.fixture
def pl():   # the real fetch_top: the shared fixture stubs it for hermetic tests
    spec = importlib.util.spec_from_file_location("pick_loop_fetch", os.path.join(TOOLS, "pick_loop.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _err(code): return urllib.error.HTTPError("http://x/cam/top.jpg", code, "Not Found", {}, io.BytesIO(b""))


def test_a_short_404_hiccup_is_retried(pl, monkeypatch):
    calls = []
    def urlopen(url, timeout=5):
        calls.append(url)
        if len(calls) < 3: raise _err(404)
        return io.BytesIO(b"jpeg")
    monkeypatch.setattr(pl.urllib.request, "urlopen", urlopen); slept = []
    assert pl.fetch_top(sleep=slept.append) == b"jpeg" and len(calls) == 3 and slept == [0.5, 0.5]


def test_a_feed_that_stays_down_still_raises(pl, monkeypatch):
    def urlopen(url, timeout=5): raise _err(404)
    monkeypatch.setattr(pl.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError): pl.fetch_top(tries=3, sleep=lambda s: None)


def test_other_http_errors_are_not_retried(pl, monkeypatch):
    calls = []
    def urlopen(url, timeout=5): calls.append(url); raise _err(500)
    monkeypatch.setattr(pl.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError): pl.fetch_top(sleep=lambda s: None)
    assert len(calls) == 1
