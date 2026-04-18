import numpy as np
import torch
import pytest
from nano_omni.types import (
    SamplingParams, OmniRequest, StageInput, StageOutput, OmniOutput, StageConfig, PipelineConfig
)

def test_sampling_params_defaults():
    p = SamplingParams()
    assert p.temperature == 1.0
    assert p.top_p == 1.0
    assert p.top_k == -1
    assert p.max_tokens == 2048
    assert p.stop_token_ids == []
    assert p.repetition_penalty == 1.0

def test_omni_request_text_only():
    req = OmniRequest(request_id="r1", text="hello")
    assert req.request_id == "r1"
    assert req.text == "hello"
    assert req.audio is None
    assert req.images is None

def test_stage_input_defaults():
    inp = StageInput(request_id="r1", token_ids=[1, 2, 3])
    assert inp.embeddings is None
    assert inp.extra == {}
    assert inp.sampling_params is None  # None means use stage default

def test_stage_output_defaults():
    out = StageOutput(request_id="r1", token_ids=[42])
    assert out.embeddings is None
    assert out.audio is None
    assert out.is_finished is False
    assert out.finish_reason is None

def test_stage_output_with_embeddings():
    emb = torch.randn(1, 4096)
    out = StageOutput(request_id="r1", token_ids=[42], embeddings=emb, is_finished=True, finish_reason="stop_token")
    assert out.embeddings.shape == (1, 4096)

def test_omni_output():
    audio = np.zeros(24000, dtype=np.float32)
    out = OmniOutput(request_id="r1", text="hi", audio=audio)
    assert out.text == "hi"
    assert len(out.audio) == 24000

def test_stage_config_defaults():
    cfg = StageConfig(name="thinker", stage_type="ar")
    assert cfg.max_batch_size == 32
    assert cfg.chunk_size == 512
    assert cfg.max_tokens_per_step == 2048
    assert cfg.kv_cache_max_requests == 32

def test_pipeline_config():
    cfg = PipelineConfig(model_path="/fake/path")
    assert cfg.model_path == "/fake/path"
    assert cfg.stages == []
