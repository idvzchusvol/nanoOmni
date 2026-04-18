from __future__ import annotations

from collections import deque

import numpy as np
import torch

from nano_omni.stage.base import StageEngine
from nano_omni.types import StageConfig, StageInput, StageOutput


class CodecStageEngine(StageEngine):
    """
    Non-autoregressive CodecStageEngine for Code2Wav stage.

    No KV Cache or Scheduler needed: each step() takes all waiting requests,
    runs one non-AR forward pass, and returns all results immediately.
    """

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

        outputs = []
        for inp in inputs:
            num_codebooks = inp.extra.get("num_codebooks", 8)
            T = len(inp.token_ids) // num_codebooks
            codec_codes = torch.tensor(
                inp.token_ids, dtype=torch.long, device=self._device,
            ).reshape(1, num_codebooks, T)

            with torch.no_grad():
                audio = self.model(codec_codes)  # np.ndarray [1, num_samples] or [num_samples]

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
