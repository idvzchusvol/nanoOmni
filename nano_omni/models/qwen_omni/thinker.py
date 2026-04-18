from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from transformers import DynamicCache


if TYPE_CHECKING:
    from nano_omni.types import OmniRequest, StageInput


class Thinker:
    """
    Wraps Omni Thinker model.

    Provides:
      - prepare_inputs(request) → StageInput  (multimodal tokenization)
      - forward(input_ids, past_key_values, ...) → HF model output
      - __call__ delegates to forward
    """

    AUDIO_START_TOKEN_ID = 151669
    AUDIO_END_TOKEN_ID = 151670

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def prepare_inputs(self, request: "OmniRequest") -> "StageInput":
        """Convert OmniRequest to StageInput with multimodal token_ids."""
        from nano_omni.types import StageInput

        # Qwen2.5-Omni requires a specific system prompt for audio output,
        # and its processor expects content as a list of typed parts.
        messages = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual "
                    "inputs, as well as generating text and speech."
                ),
            }],
        }]
        if request.text:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": request.text}],
            })

        inputs_dict: dict = {
            "text": self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        }
        if request.audio is not None:
            inputs_dict["audio"] = request.audio
            inputs_dict["sampling_rate"] = 16000
        if request.images:
            inputs_dict["images"] = request.images

        inputs = self.processor(**inputs_dict, return_tensors="pt")
        token_ids = inputs["input_ids"][0].tolist()

        attention_mask = None
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]

        return StageInput(
            request_id=request.request_id,
            token_ids=token_ids,
            extra={
                "processor_inputs": inputs,
                "prompt_ids": token_ids,
                "attention_mask": attention_mask,
            },
            sampling_params=request.sampling_params,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[DynamicCache],
        output_hidden_states: bool = True,
        **kwargs,
    ):
        # Forward pass. Returns HF CausalLM output with .logits, .past_key_values, .hidden_states
        return self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def decode(self, token_ids: list[int]) -> str:
        # Decode Thinker-generated token ids back to natural-language text.
        return self.processor.batch_decode(
            [token_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
