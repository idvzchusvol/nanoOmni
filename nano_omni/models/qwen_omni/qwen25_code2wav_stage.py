from __future__ import annotations

from collections import deque

import numpy as np
import torch

from nano_omni.stage.base import StageEngine
from nano_omni.types import StageConfig, StageInput, StageOutput

# Qwen2.5-Omni specialized Code2Wav (token2wav) stage without AR
class Qwen25OmniCode2WavStage(StageEngine):
    def __init__(self, model, config: StageConfig):
        super().__init__(config)
        self.model = model
        self._waiting: deque[StageInput] = deque()
        self._device = next(model.model.parameters()).device

    def add_request(self, inp: StageInput) -> None:
        self._waiting.append(inp)

    def has_unfinished(self) -> bool:
        return bool(self._waiting)

    def step(self) -> list[StageOutput]:
        if not self._waiting:
            return []

        inputs = list(self._waiting)
        self._waiting.clear()

        outputs: list[StageOutput] = []
        for inp in inputs:
            conditioning = inp.extra.get("conditioning")
            reference_mel = inp.extra.get("reference_mel")
            if conditioning is None or reference_mel is None:
                raise ValueError(
                    "Qwen25OmniCode2WavStage: extra must include "
                    "'conditioning' and 'reference_mel' tensors."
                )

            # Talker codec stream: drop the initial [mask*prompt, pad, bos]
            # seed and the trailing eos. The converter stores prompt_len in
            # extra so we can slice consistently; if absent, take raw ids.
            codec_ids = list(inp.token_ids)
            if not codec_ids:
                continue

            code = torch.tensor(
                [codec_ids], dtype=torch.long, device=self._device
            )

            prev_cudnn = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False
            try:
                with torch.no_grad():
                    waveform = self.model.model(
                        code,
                        conditioning=conditioning.to(self._device).float(),
                        reference_mel=reference_mel.to(self._device).float(),
                    )
            finally:
                torch.backends.cudnn.enabled = prev_cudnn

            if isinstance(waveform, torch.Tensor):
                audio = waveform.detach().cpu().float().numpy()
            else:
                audio = waveform
            if isinstance(audio, np.ndarray) and audio.ndim == 2:
                audio = audio[0]

            outputs.append(StageOutput(
                request_id=inp.request_id,
                token_ids=inp.token_ids,
                audio=audio,
                is_finished=True,
                finish_reason="completed",
            ))

        return outputs
