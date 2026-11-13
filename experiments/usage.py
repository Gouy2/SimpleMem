"""Aggregate recorded request metadata without estimating missing prices."""

from collections import Counter


def summarize_usage(events):
    calls = [event for event in events if event["kind"] in ("chat", "embedding")]
    received = [event for event in calls if event.get("status") in ("ok", "invalid")]
    return {
        "api_attempts": dict(Counter(event["kind"] for event in calls)),
        "errors": sum(event.get("status") == "error" for event in calls),
        "invalid_responses": sum(event.get("status") == "invalid" for event in calls),
        "chat_input_tokens": sum((event.get("usage") or {}).get("prompt_tokens", 0)
                                 for event in received if event["kind"] == "chat"),
        "chat_output_tokens": sum((event.get("usage") or {}).get("completion_tokens", 0)
                                  for event in received if event["kind"] == "chat"),
        "embedding_input_tokens": sum((event.get("usage") or {}).get("prompt_tokens", 0)
                                      for event in received if event["kind"] == "embedding"),
        "reported_embedding_cost_usd": sum((event.get("usage") or {}).get("cost", 0) or 0
                                           for event in received if event["kind"] == "embedding"),
        "embedding_cache_hits": sum(event.get("cache_hits", 0) for event in events),
        "summed_request_seconds": sum(event.get("elapsed_seconds", 0) for event in calls),
    }
