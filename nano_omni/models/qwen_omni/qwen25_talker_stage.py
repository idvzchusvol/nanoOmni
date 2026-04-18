from __future__ import annotations

import torch

from nano_omni.kv_cache.manager import KVCacheManager
from nano_omni.scheduler.scheduler import SequenceState
from nano_omni.stage.ar_stage import ARStageEngine
from nano_omni.types import StageConfig, StageInput

# Qwen2.5-Omni specialized Talker stage with thinker_reply_part sliding state
class Qwen25OmniTalkerStage(ARStageEngine):

    def __init__(self, model, config: StageConfig):
        super().__init__(model=model, config=config)

    def _run_prefill(self, seq: SequenceState, chunk: list[int]) -> None:
        """
        Talker prefill uses `inputs_embeds` (pre-built by converter), not input_ids.
        We assume the converter writes inputs_embeds + other Talker kwargs into
        `seq.inp.prefill_kwargs`, and `chunk` is ignored for the embedding path.
        """
        cache = self.kv_manager.get_or_create(seq.inp.request_id)

        pk = dict(seq.inp.prefill_kwargs)
        inputs_embeds = pk.pop("inputs_embeds", None)
        if inputs_embeds is None:
            # Fallback: call base class behavior
            super()._run_prefill(seq, chunk)
            return

        # Initialize sliding thinker_reply_part state from prefill_kwargs
        thinker_reply_part = pk.pop("thinker_reply_part", None)
        if thinker_reply_part is not None:
            seq.model_state["thinker_reply_part"] = thinker_reply_part

        # Remember attention_mask so we can extend it each decode step.
        attention_mask = pk.get("attention_mask", None)
        if attention_mask is not None:
            seq.model_state["attention_mask"] = attention_mask

        with torch.no_grad():
            out = self.model(
                input_ids=None,
                past_key_values=cache,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                **pk,
            )
        # HF Talker returns the shifted thinker_reply_part inside its output
        if hasattr(out, "thinker_reply_part") and out.thinker_reply_part is not None:
            seq.model_state["thinker_reply_part"] = out.thinker_reply_part

    def _prepare_decode_kwargs(self, seq: SequenceState) -> dict:
        """Feed the current sliding thinker_reply_part into each decode step."""
        kwargs = dict(seq.inp.decode_kwargs)
        trp = seq.model_state.get("thinker_reply_part")
        if trp is not None:
            kwargs["thinker_reply_part"] = trp
        am = seq.model_state.get("attention_mask")
        if am is not None:
            # Extend attention_mask by one column (HF does this in
            # _update_model_kwargs_for_generation).
            am = torch.cat([am, am.new_ones((am.shape[0], 1))], dim=1)
            seq.model_state["attention_mask"] = am
            kwargs["attention_mask"] = am
        return kwargs

    def _post_decode_hook(self, seq: SequenceState, out) -> None:
        """HF Talker returns sliced thinker_reply_part; save for next step."""
        if hasattr(out, "thinker_reply_part") and out.thinker_reply_part is not None:
            seq.model_state["thinker_reply_part"] = out.thinker_reply_part

        # Suppress forbidden codec tokens (e.g. codec_bos) before sampling.
        suppress = seq.inp.extra.get("suppress_tokens")
        if suppress:
            out.logits[..., suppress] = float("-inf")
