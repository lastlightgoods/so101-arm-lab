"""ACT + async inference engine shim.

lerobot-rollout --inference.type=rtc runs policy inference on a background thread and calls
``policy.predict_action_chunk(batch, inference_delay=..., prev_chunk_left_over=...)``.
ACT's method takes only ``batch`` and declares no RTC support, so the engine refuses it.
This shim makes ACT accept (and ignore) the extra arguments and report RTC support, which turns
the engine into plain asynchronous chunking: the control loop never waits for the 30 ms forward
pass; a fresh chunk computed from a recent observation replaces the queue when it runs low.
No chunk blending is done (that needs a flow-matching policy such as pi0/SmolVLA).
Disable with LEROBOT_ACTASYNC=0.
"""
import logging, os
logger = logging.getLogger("lerobot_policy_actasync")
try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    _orig = ACTPolicy.predict_action_chunk
    if os.environ.get("LEROBOT_ACTASYNC", "1") != "0" and not getattr(_orig, "_actasync", False):
        def predict_action_chunk(self, batch, inference_delay=None, prev_chunk_left_over=None, **_):
            return _orig(self, batch)
        predict_action_chunk._actasync = True
        ACTPolicy.predict_action_chunk = predict_action_chunk
        ACTPolicy.supports_rtc = lambda self: True
        logger.info("[actasync] ACTPolicy accepts the async/rtc call shape (plain background chunking)")
except Exception as e:  # pragma: no cover
    logger.warning(f"[actasync] not installed: {e!r}")
