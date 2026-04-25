# tests/test_pipeline.py
import numpy as np
import pytest
from unittest.mock import MagicMock

from nano_omni.types import OmniRequest, StageInput, StageOutput, OmniOutput, SamplingParams
from nano_omni.pipeline import Pipeline
from nano_omni.stage.base import StageEngine


def _make_mock_model():
    """Mock model with prepare_inputs and decode for stage 0."""
    model = MagicMock()
    model.prepare_inputs = lambda req: StageInput(
        request_id=req.request_id, token_ids=[1, 2, 3],
    )
    model.decode = lambda token_ids: " ".join(str(t) for t in token_ids)
    return model


def make_mock_stage(outputs_per_request: dict, model=None) -> StageEngine:
    """
    Mock StageEngine: add_request records requests, step() returns all outputs on first call.
    """
    stage = MagicMock(spec=StageEngine)
    if model is not None:
        stage.model = model
    pending = {}
    returned = set()

    def add(inp: StageInput):
        pending[inp.request_id] = inp

    def step():
        results = []
        for rid, inp in list(pending.items()):
            if rid not in returned:
                returned.add(rid)
                out = outputs_per_request.get(rid)
                if out:
                    results.append(out)
        return results

    def has_unfinished():
        return any(rid not in returned for rid in pending)

    stage.add_request.side_effect = add
    stage.step.side_effect = step
    stage.has_unfinished.side_effect = has_unfinished
    return stage


def make_thinker_out(rid: str):
    import torch
    return StageOutput(
        request_id=rid,
        token_ids=[10, 11, 12],
        embeddings=torch.zeros(1, 64),
        is_finished=True,
    )


def make_talker_out(rid: str):
    return StageOutput(
        request_id=rid,
        token_ids=list(range(16)),
        is_finished=True,
    )


def make_codec_out(rid: str):
    return StageOutput(
        request_id=rid,
        token_ids=list(range(16)),
        audio=np.zeros(24000, dtype=np.float32),
        is_finished=True,
    )


def test_pipeline_single_request():
    thinker = make_mock_stage({"r1": make_thinker_out("r1")}, model=_make_mock_model())
    talker = make_mock_stage({"r1": make_talker_out("r1")})
    codec = make_mock_stage({"r1": make_codec_out("r1")})

    def thinker2talker(out: StageOutput) -> StageInput:
        return StageInput(request_id=out.request_id, token_ids=out.token_ids)

    def talker2codec(out: StageOutput) -> StageInput:
        return StageInput(request_id=out.request_id, token_ids=out.token_ids,
                          extra={"num_codebooks": 1})

    pipeline = Pipeline(
        stages=[thinker, talker, codec],
        converters=[thinker2talker, talker2codec],
    )

    req = OmniRequest(request_id="r1", text="hello")
    results, metrics = pipeline.run(requests=[req])

    assert len(results) == 1
    out = results[0]
    assert out.request_id == "r1"
    assert out.audio is not None
    assert metrics.total_s >= 0
    assert len(metrics.stages) == 3


def test_pipeline_converter_count_mismatch():
    thinker = make_mock_stage({}, model=_make_mock_model())
    with pytest.raises(AssertionError):
        Pipeline(stages=[thinker], converters=[lambda x: x])  # 1 stage, 1 converter → error


def test_pipeline_multiple_requests():
    thinker_outs = {rid: make_thinker_out(rid) for rid in ["r1", "r2"]}
    talker_outs = {rid: make_talker_out(rid) for rid in ["r1", "r2"]}
    codec_outs = {rid: make_codec_out(rid) for rid in ["r1", "r2"]}

    pipeline = Pipeline(
        stages=[
            make_mock_stage(thinker_outs, model=_make_mock_model()),
            make_mock_stage(talker_outs),
            make_mock_stage(codec_outs),
        ],
        converters=[
            lambda out: StageInput(request_id=out.request_id, token_ids=out.token_ids),
            lambda out: StageInput(request_id=out.request_id, token_ids=out.token_ids,
                                   extra={"num_codebooks": 1}),
        ],
    )
    reqs = [OmniRequest(request_id=rid, text="hi") for rid in ["r1", "r2"]]
    results, metrics = pipeline.run(requests=reqs)
    assert len(results) == 2
    assert {r.request_id for r in results} == {"r1", "r2"}
    assert metrics.total_s >= 0


def test_pipeline_two_stages_no_audio():
    thinker = make_mock_stage({"r1": make_thinker_out("r1")}, model=_make_mock_model())
    talker = make_mock_stage({"r1": make_talker_out("r1")})

    pipeline = Pipeline(
        stages=[thinker, talker],
        converters=[lambda out: StageInput(request_id=out.request_id, token_ids=out.token_ids)],
    )
    req = OmniRequest(request_id="r1", text="hello")
    results, metrics = pipeline.run(requests=[req])
    assert results[0].audio is None
    assert len(metrics.stages) == 2


def test_pipeline_stage0_without_model_raises():
    stage = MagicMock(spec=StageEngine)
    with pytest.raises(TypeError, match="prepare_inputs"):
        Pipeline(stages=[stage], converters=[])
