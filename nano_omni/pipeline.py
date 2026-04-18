from __future__ import annotations

from typing import Callable

from nano_omni.stage.base import StageEngine
from nano_omni.types import OmniOutput, OmniRequest, StageInput, StageOutput


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
    ) -> list[OmniOutput]:

        # Feed all requests into stage 0
        for req in requests:
            self.stages[0].add_request(self._preprocess(req))

        # Drive each stage to completion, pass outputs to next stage
        stage0_outputs: dict[str, StageOutput] = {}
        final_outputs: dict[str, StageOutput] = {}
        for i, stage in enumerate(self.stages):
            stage_results: dict[str, StageOutput] = {}
            while stage.has_unfinished():
                for out in stage.step():
                    stage_results[out.request_id] = out
            if i == 0:
                stage0_outputs = stage_results
            if i < len(self.stages) - 1:
                for out in stage_results.values():
                    self.stages[i + 1].add_request(self.converters[i](out))
            else:
                final_outputs = stage_results

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
        return results
