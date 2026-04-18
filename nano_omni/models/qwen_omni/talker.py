from __future__ import annotations

from typing import Optional

import torch
from transformers import DynamicCache


class Talker:
    """
    Wraps Talker model (audio AR, generates RVQ codec codes).

    First call (prefill): pass inputs_embeds = Thinker hidden states
    Subsequent calls (decode): pass input_ids = last sampled codec token
    """

    CODEC_BOS_TOKEN_ID = 4197
    CODEC_EOS_TOKEN_ID = 4198

    def __init__(self, model):
        self.model = model

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        past_key_values: Optional[DynamicCache],
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        """
        Forward pass.
        - Prefill: pass inputs_embeds (Thinker hidden states as prefix)
        - Decode: pass input_ids (last sampled codec token)
        """
        return self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
