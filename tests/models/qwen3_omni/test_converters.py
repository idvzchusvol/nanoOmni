import torch
import pytest
from nano_omni.types import StageOutput, StageInput, SamplingParams
from nano_omni.models.qwen_omni.qwen3_converters import qwen3_thinker2talker, qwen3_talker2code2wav

TALKER_SAMPLING = SamplingParams(temperature=0.9, top_k=50, max_tokens=4096,
                                  stop_token_ids=[2150], repetition_penalty=1.05)

def test_thinker2talker_basic():
    out = StageOutput(
        request_id="r1",
        token_ids=[10, 11, 12],
        embeddings=torch.zeros(1, 64),
        is_finished=True,
    )
    inp = qwen3_thinker2talker(out, talker_sampling=TALKER_SAMPLING)
    assert isinstance(inp, StageInput)
    assert inp.request_id == "r1"
    assert inp.embeddings is not None
    assert inp.sampling_params.stop_token_ids == [2150]

def test_thinker2talker_no_embeddings_raises():
    out = StageOutput(request_id="r1", token_ids=[1, 2], embeddings=None, is_finished=True)
    with pytest.raises(ValueError, match="embeddings"):
        qwen3_thinker2talker(out)

def test_talker2code2wav_reshapes_tokens():
    out = StageOutput(
        request_id="r1",
        token_ids=list(range(16)),
        is_finished=True,
    )
    inp = qwen3_talker2code2wav(out, num_codebooks=8)
    assert inp.request_id == "r1"
    assert inp.extra["num_codebooks"] == 8
    assert inp.token_ids == list(range(16))

def test_talker2code2wav_default_codebooks():
    out = StageOutput(request_id="r1", token_ids=list(range(8)), is_finished=True)
    inp = qwen3_talker2code2wav(out)
    assert inp.extra["num_codebooks"] == 8   # default
