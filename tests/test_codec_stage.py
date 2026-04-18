import numpy as np
import torch
import pytest
from unittest.mock import MagicMock

from nano_omni.types import StageConfig, StageInput, StageOutput
from nano_omni.stage.codec_stage import CodecStageEngine


def make_config() -> StageConfig:
    return StageConfig(name="code2wav", stage_type="codec", max_batch_size=4)


def make_mock_codec():
    """mock codec model: forward(codec_codes) → np.ndarray audio [batch, samples]"""
    model = MagicMock()
    def fake_forward(codec_codes):
        batch = codec_codes.shape[0]
        return np.zeros((batch, 24000), dtype=np.float32)
    model.side_effect = fake_forward
    return model


def test_initially_no_unfinished():
    engine = CodecStageEngine(model=make_mock_codec(), config=make_config())
    assert engine.has_unfinished() is False


def test_add_request_marks_unfinished():
    engine = CodecStageEngine(model=make_mock_codec(), config=make_config())
    engine.add_request(StageInput(request_id="r1", token_ids=[1, 2, 3, 4, 5, 6, 7, 8]))
    assert engine.has_unfinished() is True


def test_step_returns_audio_output():
    engine = CodecStageEngine(model=make_mock_codec(), config=make_config())
    engine.add_request(StageInput(
        request_id="r1",
        token_ids=list(range(8)),
        extra={"num_codebooks": 1},
    ))
    outputs = engine.step()
    assert len(outputs) == 1
    out = outputs[0]
    assert out.request_id == "r1"
    assert out.is_finished is True
    assert out.audio is not None
    assert isinstance(out.audio, np.ndarray)


def test_step_processes_all_waiting():
    engine = CodecStageEngine(model=make_mock_codec(), config=make_config())
    engine.add_request(StageInput(request_id="r1", token_ids=list(range(8))))
    engine.add_request(StageInput(request_id="r2", token_ids=list(range(8))))
    outputs = engine.step()
    assert len(outputs) == 2
    assert engine.has_unfinished() is False


def test_step_empty_queue_returns_empty():
    engine = CodecStageEngine(model=make_mock_codec(), config=make_config())
    outputs = engine.step()
    assert outputs == []
