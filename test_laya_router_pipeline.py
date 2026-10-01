"""Runnable check for laya_router_pipeline.py: `python test_laya_router_pipeline.py`.

Uses LAYA_MODE=server with `requests.post` stubbed, so neither laya/torch nor the
network is needed.
"""
import json
from unittest import mock

from laya_router_pipeline import Pipeline


class Resp:
    def __init__(self, data=None, lines=()):
        self._data, self._lines = data, lines

    def raise_for_status(self):
        pass

    def json(self):
        return self._data

    def iter_lines(self):
        return iter(self._lines)


def laya(tier, kind):
    return Resp({"answers": {"tier": {"choice": tier}, "kind": {"choice": kind}}})


def sse(delta):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]})


def make():
    p = Pipeline()
    p.valves.LAYA_MODE = "server"
    return p


def test_classify():
    p = make()
    with mock.patch("requests.post", return_value=laya("expensive", "code")):
        assert p.classify([], "refactor this") == ("expensive", "code")
    # An answer outside the declared criteria falls back instead of routing blindly.
    with mock.patch("requests.post", return_value=laya("huge", "code")):
        assert p.classify([], "```x```") == ("cheap", "code")


def test_stream_drops_comments_and_skips_footer_on_tool_round():
    tool = sse({"tool_calls": [{"index": 0, "function": {"name": "search"}}]})
    out = list(Pipeline._stream_with_footer(
        [b": OPENROUTER PROCESSING", b"", tool.encode(), b"data: [DONE]"], "FOOTER"))
    assert out == [tool.encode(), b"data: [DONE]"]

    out = list(Pipeline._stream_with_footer([sse({"content": "hi"}), "data: [DONE]"], "FOOTER"))
    assert "FOOTER" in out[1] and out[2] == "data: [DONE]"


def test_tool_round_keeps_route():
    p = make()
    body = {"stream": False, "messages": [{"role": "user", "content": "q"}]}
    backend = Resp({"choices": [{"message": {"content": "a"}}]})
    with mock.patch("requests.post", side_effect=[laya("expensive", "text"), backend]) as post:
        p.pipe("q", "laya", body["messages"], body)
        assert post.call_args.kwargs["json"]["model"] == "z-ai/glm-5.3"

    body["messages"] = body["messages"] + [{"role": "tool", "content": "results"}]
    with mock.patch("requests.post", return_value=backend) as post:
        p.pipe("q", "laya", body["messages"], body)
        assert post.call_count == 1  # no second classification
        assert post.call_args.kwargs["json"]["model"] == "z-ai/glm-5.3"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
