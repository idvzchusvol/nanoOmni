"""
Qwen2.5-Omni specific Thinker→Talker converter.

Replicates the preparation logic from HF's
`Qwen2_5OmniForConditionalGeneration.generate()` (modeling_qwen2_5_omni.py
around L3858-L3944): masking multimodal tokens, building `thinker_reply_part`,
`talker_inputs_embeds`, `talker_input_ids`, `talker_input_text_ids`.
"""
from __future__ import annotations

import torch

from nano_omni.types import SamplingParams, StageInput, StageOutput


def qwen25_thinker2talker(
    full_model,
    speaker: str = "Chelsie",
    talker_sampling: SamplingParams | None = None,
):
    """
    Build a converter that closes over `full_model`.

    The returned converter takes a Thinker StageOutput (with per_step_*),
    reads the original prompt token_ids from `thinker_out.extra["prompt_ids"]`,
    and constructs a StageInput for Qwen2.5-Omni Talker.
    """
    thinker = full_model.thinker
    talker = full_model.talker
    thinker_embed = thinker.get_input_embeddings()
    speaker_params = full_model.speaker_map[speaker]

    cfg_thinker = full_model.config.thinker_config
    audio_token_id = getattr(cfg_thinker, "audio_token_index", None)
    image_token_id = getattr(cfg_thinker, "image_token_index", None)
    video_token_id = getattr(cfg_thinker, "video_token_index", None)

    def _convert(thinker_out: StageOutput) -> StageInput:
        if thinker_out.per_step_hidden_states is None or thinker_out.per_step_token_embeds is None:
            raise ValueError(
                "Qwen25 thinker2talker: per_step_hidden_states / per_step_token_embeds "
                "must be populated by Thinker stage."
            )

        prompt_ids = thinker_out.extra.get("prompt_ids")
        if prompt_ids is None:
            raise ValueError("Qwen25 thinker2talker: extra['prompt_ids'] missing.")

        device = thinker_out.per_step_hidden_states[0].device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Attention mask for the full prompt (defaults to all-ones).
        prompt_attention_mask = thinker_out.extra.get("attention_mask")
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(input_ids)
        else:
            prompt_attention_mask = prompt_attention_mask.to(device)

        # Mask multimodal tokens in step-0 token embeddings
        embeds_to_talker = thinker_out.per_step_token_embeds[0].clone()
        for tok_id in (audio_token_id, image_token_id, video_token_id):
            if tok_id is None:
                continue
            mask = (input_ids == tok_id).unsqueeze(-1).expand_as(embeds_to_talker)
            zeros = torch.zeros(
                [mask.sum() // embeds_to_talker.shape[-1], embeds_to_talker.shape[-1]],
                dtype=embeds_to_talker.dtype,
                device=device,
            )
            embeds_to_talker = embeds_to_talker.masked_scatter(mask, zeros)

        token_embeds = [embeds_to_talker] + [
            e.to(device) for e in thinker_out.per_step_token_embeds[1:]
        ]
        hidden_states = [h.to(device) for h in thinker_out.per_step_hidden_states]

        thinker_generate_ids = torch.tensor(
            [thinker_out.token_ids], dtype=torch.long, device=device
        )

        # Build talker_input_text_ids: [prompt, talker_text_bos, first_generated]
        talker_text_bos = speaker_params["bos_token"]
        talker_input_text_ids = torch.cat(
            [
                input_ids,
                torch.tensor([[talker_text_bos]], dtype=torch.long, device=device),
                thinker_generate_ids[:, :1],
            ],
            dim=-1,
        )

        # Build talker_input_ids (codec space): [mask*prompt_len, pad, bos]
        talker_input_ids = torch.cat(
            [
                torch.full_like(input_ids, fill_value=talker.codec_mask_token),
                torch.tensor([[talker.codec_pad_token]], dtype=torch.long, device=device),
                torch.tensor([[talker.codec_bos_token]], dtype=torch.long, device=device),
            ],
            dim=1,
        )

        # Build thinker_reply_part (steps 1+ only)
        if len(hidden_states) > 1:
            thinker_reply_part = torch.cat(hidden_states[1:], dim=1) + torch.cat(
                token_embeds[1:], dim=1
            )
        else:
            # Edge case: thinker generated 0 tokens (shouldn't happen in practice)
            hidden_size = hidden_states[0].shape[-1]
            thinker_reply_part = torch.zeros(1, 0, hidden_size, device=device, dtype=hidden_states[0].dtype)

        # Build talker_inputs_embeds: step0_prompt + text_bos + first_reply
        talker_inputs_embeds = hidden_states[0] + token_embeds[0]
        talker_text_bos_t = torch.tensor([[talker_text_bos]], dtype=torch.long, device=device)
        talker_text_bos_embed = thinker_embed(talker_text_bos_t).to(device)
        talker_inputs_embeds = torch.cat(
            [talker_inputs_embeds, talker_text_bos_embed, thinker_reply_part[:, :1, :]],
            dim=1,
        )

        # Append eos + pad to thinker_reply_part for decode steps
        eos_t = torch.tensor([[talker.text_eos_token]], dtype=torch.long, device=device)
        pad_t = torch.tensor([[talker.text_pad_token]], dtype=torch.long, device=device)
        eos_embed = thinker_embed(eos_t).to(device)
        pad_embed = thinker_embed(pad_t).to(device)
        thinker_reply_part = torch.cat(
            [thinker_reply_part[:, 1:, :], eos_embed, pad_embed], dim=1
        )

        # Packaged into StageInput for the Talker stage.
        # token_ids here holds the talker_input_ids sequence (codec space) so the
        # Scheduler's prefill_offset bookkeeping still works end-to-end.
        codec_token_ids = talker_input_ids[0].tolist()

        # Talker attention mask during prefill covers [prompt, pad, bos] (prompt_len + 2).
        talker_attention_mask = torch.cat(
            [prompt_attention_mask, prompt_attention_mask.new_ones((1, 2))],
            dim=1,
        )

        return StageInput(
            request_id=thinker_out.request_id,
            token_ids=codec_token_ids,
            sampling_params=talker_sampling,
            prefill_kwargs={
                "inputs_embeds": talker_inputs_embeds,
                "input_text_ids": talker_input_text_ids,
                "thinker_reply_part": thinker_reply_part,
                "attention_mask": talker_attention_mask,
            },
            decode_kwargs={
                "input_text_ids": talker_input_text_ids,
            },
            extra={
                "speaker": speaker,
                "prompt_ids": prompt_ids,
                # Suppress non-audio codec tokens so only audio codes (<8192)
                # or codec_eos (8294, used as the stop signal) get sampled.
                "suppress_tokens": [
                    talker.codec_bos_token,
                    talker.codec_pad_token,
                    talker.codec_mask_token,
                ],
            },
        )

    return _convert


def qwen25_talker2token2wav(full_model, speaker: str = "Chelsie"):
    # Build a converter that packages Talker codec output for token2wav
    speaker_params = full_model.speaker_map[speaker]
    codec_eos = full_model.talker.codec_eos_token

    def _convert(talker_out: StageOutput) -> StageInput:
        # Strip trailing codec_eos (HF does: talker_result[:, seed_len : -1])
        codes = list(talker_out.token_ids)
        if codes and codes[-1] == codec_eos:
            codes = codes[:-1]
        return StageInput(
            request_id=talker_out.request_id,
            token_ids=codes,
            extra={
                "conditioning": speaker_params["cond"],
                "reference_mel": speaker_params["ref_mel"],
            },
        )

    return _convert
