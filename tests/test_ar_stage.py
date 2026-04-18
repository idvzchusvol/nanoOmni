# tests/test_ar_stage.py
import torch
import pytest
from unittest.mock import MagicMock
from transformers import DynamicCache

from nano_omni.types import StageConfig, StageInput, StageOutput, SamplingParams
from nano_omni.stage.ar_stage import ARStageEngine


def make_config(max_tokens=5) -> StageConfig:
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=[99])
    return StageConfig(
        name="test_ar",
        stage_type="ar",
        max_batch_size=4,
        chunk_size=4,
        max_tokens_per_step=16,
        kv_cache_max_requests=4,
        sampling_params=sp,
    )


def make_mock_model(vocab_size=100, hidden_size=64):
    """Mock model: forward() returns (logits, cache, hidden_states). Always predicts token 7."""
    model = MagicMock()
    def fake_forward(input_ids, past_key_values, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, vocab_size)
        logits[:, :, 7] = 100.0   # always predict token 7
        out = MagicMock()
        out.logits = logits
        out.past_key_values = past_key_values if past_key_values else DynamicCache()
        out.hidden_states = (torch.zeros(batch, seq, hidden_size),) * 3
        return out
    model.side_effect = fake_forward
    return model


def test_add_request_registers_as_unfinished():
    engine = ARStageEngine(model=make_mock_model(), config=make_config())
    assert engine.has_unfinished() is False
    engine.add_request(StageInput(request_id="r1", token_ids=[1, 2, 3]))
    assert engine.has_unfinished() is True


def test_step_prefills_then_decodes():
    engine = ARStageEngine(model=make_mock_model(), config=make_config(max_tokens=3))
    engine.add_request(StageInput(request_id="r1", token_ids=[1, 2]))
    # First step: prefill (2 tokens < chunk_size 4), then first decode
    out1 = engine.step()
    # Still not finished (max_tokens=3, only 1 token generated so far)
    assert engine.has_unfinished() is True
    assert out1 == []


def test_step_returns_output_when_max_tokens():
    """Model always predicts 7 (not stop token 99), finishes at max_tokens=2."""
    engine = ARStageEngine(model=make_mock_model(), config=make_config(max_tokens=2))
    engine.add_request(StageInput(request_id="r1", token_ids=[1]))
    outputs = []
    for _ in range(10):
        outputs.extend(engine.step())
        if not engine.has_unfinished():
            break
    assert len(outputs) == 1
    out = outputs[0]
    assert out.request_id == "r1"
    assert len(out.token_ids) == 2
    assert out.is_finished is True
    assert out.finish_reason == "max_tokens"


def test_step_stops_on_stop_token():
    """Model predicts stop token 99, should stop immediately after first token."""
    model = make_mock_model()
    def fake_forward_stop(input_ids, past_key_values, **kwargs):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, 100)
        logits[:, :, 99] = 100.0  # predict stop token
        out = MagicMock()
        out.logits = logits
        out.past_key_values = past_key_values if past_key_values else DynamicCache()
        out.hidden_states = (torch.zeros(batch, seq, 64),) * 3
        return out
    model.side_effect = fake_forward_stop

    engine = ARStageEngine(model=model, config=make_config(max_tokens=10))
    engine.add_request(StageInput(request_id="r1", token_ids=[1, 2]))

    outputs = []
    for _ in range(5):
        outputs.extend(engine.step())
        if not engine.has_unfinished():
            break
    assert len(outputs) == 1
    assert outputs[0].finish_reason == "stop_token"
    assert outputs[0].token_ids == [99]


def test_output_has_embeddings():
    engine = ARStageEngine(model=make_mock_model(), config=make_config(max_tokens=1))
    engine.add_request(StageInput(request_id="r1", token_ids=[1]))
    outputs = []
    for _ in range(5):
        outputs.extend(engine.step())
        if not engine.has_unfinished():
            break
    assert len(outputs) == 1
    assert outputs[0].embeddings is not None
    assert outputs[0].embeddings.shape == (1, 64)


def test_multiple_requests_both_complete():
    engine = ARStageEngine(model=make_mock_model(), config=make_config(max_tokens=2))
    engine.add_request(StageInput(request_id="r1", token_ids=[1]))
    engine.add_request(StageInput(request_id="r2", token_ids=[2]))
    outputs = []
    for _ in range(10):
        outputs.extend(engine.step())
        if not engine.has_unfinished():
            break
    rids = {o.request_id for o in outputs}
    assert rids == {"r1", "r2"}
