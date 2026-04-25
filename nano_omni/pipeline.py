from __future__ import annotations

import time
from typing import Callable

from nano_omni.stage.base import StageEngine
from nano_omni.types import (
    OmniOutput, OmniRequest, PipelineMetrics, StageInput, StageMetrics, StageOutput,
)


class Pipeline:
    """
    Coordinates multiple StageEngines in sequence.

    run() drives each StageEngine's step() loop until all requests complete,
    then uses converters[i] to transform stage i output into stage i+1 input.
    """

    def __init__(
        self,
        stages: list[StageEngine],
        converters: list[Callable[[StageOutput], StageInput]],
    ):
        assert len(converters) == len(stages) - 1, (
            f"Need {len(stages)-1} converters, got {len(converters)}"
        )
        self.stages = stages
        self.converters = converters
        stage0_model = getattr(stages[0], "model", None)
        if stage0_model is None or not hasattr(stage0_model, "prepare_inputs"):
            raise TypeError(
                "stages[0].model must implement prepare_inputs()."
            )
        self._preprocess = stage0_model.prepare_inputs
        self._text_decoder = (
            stage0_model.decode
            if hasattr(stage0_model, "decode")
            else None
        )

    def run(
        self,
        requests: list[OmniRequest],
    ) -> tuple[list[OmniOutput], PipelineMetrics]:

        pipeline_start = time.perf_counter()
        stage_metrics: list[StageMetrics] = []

        # Feed all requests into stage 0
        for req in requests:
            self.stages[0].add_request(self._preprocess(req))

        # Drive each stage to completion, pass outputs to next stage.
        stage0_outputs: dict[str, StageOutput] = {}
        final_outputs: dict[str, StageOutput] = {}
        for i, stage in enumerate(self.stages):
            stage_results: dict[str, StageOutput] = {}
            stage_name = getattr(stage, "config", None)
            stage_name = stage_name.name if stage_name else f"stage_{i}"
            ttft: float | None = None
            first_token_seen = False

            stage_start = time.perf_counter()
            while stage.has_unfinished():
                for out in stage.step():
                    if not first_token_seen and i == 0:
                        ttft = time.perf_counter() - stage_start
                        first_token_seen = True
                    stage_results[out.request_id] = out
            stage_elapsed = time.perf_counter() - stage_start

            total_tokens = sum(len(o.token_ids) for o in stage_results.values())
            stage_metrics.append(StageMetrics(
                name=stage_name,
                elapsed_s=stage_elapsed,
                num_tokens=total_tokens,
                ttft_s=ttft if i == 0 else None,
            ))

            if i == 0:
                stage0_outputs = stage_results
            if i < len(self.stages) - 1:
                for out in stage_results.values():
                    self.stages[i + 1].add_request(self.converters[i](out))
            else:
                final_outputs = stage_results

        pipeline_elapsed = time.perf_counter() - pipeline_start

        # Compute audio duration from final outputs
        audio_duration_s = None
        for out in final_outputs.values():
            if out.audio is not None:
                audio_duration_s = (audio_duration_s or 0.0) + len(out.audio) / 24000.0

        metrics = PipelineMetrics(
            stages=stage_metrics,
            total_s=pipeline_elapsed,
            audio_duration_s=audio_duration_s,
        )

        # Assemble OmniOutputs in original request order
        results = []
        for req in requests:
            final = final_outputs.get(req.request_id)
            if final is None:
                continue
            text_out = stage0_outputs.get(req.request_id)
            if text_out is not None and self._text_decoder is not None:
                text = self._text_decoder(text_out.token_ids)
            elif text_out is not None:
                text = " ".join(str(t) for t in text_out.token_ids)
            else:
                text = ""
            results.append(OmniOutput(
                request_id=req.request_id,
                text=text,
                audio=final.audio,
            ))
        return results, metrics
