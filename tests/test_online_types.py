# tests/test_online_types.py
import numpy as np

from nano_omni.online.types import OmniChunk


def test_omnichunk_done_minimal():
    c = OmniChunk(request_id="r1", type="done")
    assert c.request_id == "r1"
    assert c.type == "done"
    assert c.text is None
    assert c.audio is None
    assert c.error is None


def test_omnichunk_text():
    c = OmniChunk(request_id="r1", type="text", text="hi")
    assert c.text == "hi"


def test_omnichunk_audio():
    buf = np.zeros(100, dtype=np.float32)
    c = OmniChunk(request_id="r1", type="audio", audio=buf)
    assert c.audio is buf


def test_omnichunk_error():
    c = OmniChunk(request_id="r1", type="error", error="boom")
    assert c.error == "boom"
