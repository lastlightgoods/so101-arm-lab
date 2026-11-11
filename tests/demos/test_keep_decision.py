"""tools/demo_session.keep_decision: the keep/discard rule as a pure function (plan section 5). Table driven; no files, no lerobot."""
import pytest

OK = {"ok": True, "reasons": []}


def cycle(**kw):
    c = {"found": True, "picked": True, "landed": True, "landed_err_mm": 5.0, "retries": 0}; c.update(kw); return c


def test_a_clean_landed_pick_is_kept_as_tier_A(dses):
    d = dses.keep_decision(cycle(), 15.0, None, True, OK); assert d.keep and d.tier == "A" and d.reasons == [] and d.reason is None
    assert d.as_json() == {"keep": True, "tier": "A", "reasons": [], "discard_reason": None}


def test_success_after_the_retry_is_tier_R(dses):
    d = dses.keep_decision(cycle(retries=1), 15.0, None, True, OK); assert d.keep and d.tier == "R"


@pytest.mark.parametrize("err,keep", [(14.9, True), (15.0, True), (15.1, False), (0.0, True), (80.0, False)])
def test_the_radius_edge(dses, err, keep):
    d = dses.keep_decision(cycle(landed_err_mm=err), 15.0, None, True, OK); assert d.keep is keep and (keep or d.reason == "outside_radius")


def test_the_radius_is_rechecked_not_trusted_from_the_landed_flag(dses):
    d = dses.keep_decision(cycle(landed=True, landed_err_mm=20.0), 15.0, None, True, OK); assert not d.keep and d.reason == "outside_radius"
    assert dses.keep_decision(cycle(landed_err_mm=20.0), 25.0, None, True, OK).keep, "--radius-mm is honoured"


@pytest.mark.parametrize("kw,reason", [
    (dict(trip="workspace"), "guardrail:workspace"), (dict(trip="consecutive_failures"), "guardrail:consecutive_failures"),
    (dict(jaw_held=False), "jaw_check_failed"), (dict(jaw_held=None), "jaw_check_missing"),
    (dict(health=None), "health_missing"), (dict(health={"ok": False, "reasons": ["cadence:9pct_over_budget"]}), "recording_health:cadence:9pct_over_budget"), (dict(health={"ok": False}), "recording_health:not_ok"),
])
def test_each_condition_alone_discards(dses, kw, reason):
    args = dict(trip=None, jaw_held=True, health=OK); args.update(kw); d = dses.keep_decision(cycle(), 15.0, **args)
    assert not d.keep and d.tier is None and d.reason == reason


@pytest.mark.parametrize("c,reason", [
    (cycle(picked=False), "not_picked"), (cycle(found=False, picked=False, landed_err_mm=None), "nut_not_found"), (cycle(landed_err_mm=None), "no_landing_measured"),
    (cycle(error="nut on the target and the reset did not move it away"), "pre_reset_artefact"), (cycle(error="ik failed"), "cycle_error")])
def test_cycle_level_conditions_discard(dses, c, reason):
    d = dses.keep_decision(c, 15.0, None, True, OK); assert not d.keep and reason in d.reasons


def test_every_reason_is_listed_in_order_and_the_first_is_the_discard_reason(dses):
    d = dses.keep_decision(cycle(picked=False, landed_err_mm=40.0), 15.0, "markers", False, {"ok": False, "reasons": ["camera_frozen:top"]})
    assert d.reasons == ["not_picked", "outside_radius", "guardrail:markers", "jaw_check_failed", "recording_health:camera_frozen:top"] and d.reason == "not_picked"


@pytest.mark.parametrize("bad", [None, {}, {"picked": True}, {"picked": True, "landed_err_mm": "far"}, {"picked": True, "landed_err_mm": 3.0, "retries": "many"}, "not a dict"])
def test_it_never_keeps_on_malformed_input_and_never_raises(dses, bad):
    d = dses.keep_decision(bad, 15.0, None, True, OK); assert d.keep is False and d.reasons


def test_it_never_keeps_when_the_radius_itself_is_garbage(dses):
    d = dses.keep_decision(cycle(), "wide", None, True, OK); assert d.keep is False and d.reason.startswith("decision_error")
    d = dses.keep_decision(cycle(), float("nan"), None, True, OK); assert d.keep is False


def test_jaw_state_folds_the_pick_record_and_the_scene_object(dses):
    js = dses.jaw_state
    assert js({"in_jaws": True}, {"in_jaws": True}) is True and js({"in_jaws": True}, None) is True and js(None, {"in_jaws": True}) is True
    assert js({"in_jaws": True}, {"in_jaws": False}) is False and js({"in_jaws": False}, None) is False
    assert js({"in_jaws": None}, {}) is None and js(None, None) is None
