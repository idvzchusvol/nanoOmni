from __future__ import annotations

import torch

from nano_omni.kv_cache.manager import KVCacheManager
from nano_omni.scheduler.scheduler import Scheduler, SequenceState
from nano_omni.stage.base import StageEngine
from nano_omni.types import StageConfig, StageInput, StageOutput


class ARStageEngine(StageEngine):
    """
    Autoregressive StageEngine for Thinker and Talker stages.

    Each step():
      1. Scheduler produces ScheduleBatch (prefill_seqs + decode_seqs)
      2. For each prefill seq: run forward (fills KV cache), no sampling
      3. When prefill completes: immediately run first decode step, sample token
      4. For each decode seq: run single forward, sample token
      5. Check stop conditions, return completed StageOutputs
    """

    def __init__(self, model, config: StageConfig):
        super().__init__(config)
        self.model = model
        self.scheduler = Scheduler(
            max_batch_size=config.max_batch_size,
            chunk_size=config.chunk_size,
            max_tokens_per_step=config.max_tokens_per_step,
        )
        self.kv_manager = KVCacheManager(max_requests=config.kv_cache_max_requests)
        self._default_sp = config.sampling_params
        # Resolve device from the underlying HF module so we place input tensors correctly.
        self._device = next(model.model.parameters()).device

    def add_request(self, inp: StageInput) -> None:
        self.scheduler.add(inp)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def step(self) -> list[StageOutput]:
        batch = self.scheduler.schedule()
        completed: list[StageOutput] = []

        # --- Prefill ---
        for seq, chunk in batch.prefill_seqs:
            self._run_prefill(seq, chunk)
            if not seq.is_prefilling:
                # Full prefill done: run first decode step immediately
                out = self._run_decode_step(seq)
                if out is not None:
                    completed.append(out)
                    self.scheduler.finish(seq.inp.request_id)
                    self.kv_manager.free(seq.inp.request_id)

        # --- Decode ---
        for seq in batch.decode_seqs:
            out = self._run_decode_step(seq)
            if out is not None:
                completed.append(out)
                self.scheduler.finish(seq.inp.request_id)
                self.kv_manager.free(seq.inp.request_id)

        return completed

    def _run_prefill(self, seq: SequenceState, chunk: list[int]) -> None:
        """Run one prefill chunk, update KV cache. No sampling."""
        cache = self.kv_manager.get_or_create(seq.inp.request_id)
        input_ids = torch.tensor([chunk], dtype=torch.long, device=self._device)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                output_hidden_states=True,
                **seq.inp.prefill_kwargs,
            )
        # For downstream stages that need per-step hidden/embedding sequences,
        # record prefill span as the first "step" (only on the final prefill chunk
        # so the prompt is captured intact).
        if out.hidden_states is not None and not seq.is_prefilling:
            # hidden_states is tuple of (L+1) tensors: [embedding, layer_1, ..., layer_L].
            seq.per_step_token_embeds.append(out.hidden_states[0].detach())
            seq.per_step_hidden_states.append(out.hidden_states[-1].detach())

    def _run_decode_step(self, seq: SequenceState) -> StageOutput | None:
        """
        Run one decode step, sample next token.
        Returns StageOutput if done (stop condition met), else None.
        """
        sp = seq.inp.sampling_params or self._default_sp
        cache = self.kv_manager.get_or_create(seq.inp.request_id)

        # Input token: on first decode use last prompt token; afterwards use last generated token
        if not seq.generated_token_ids:
            last_token = seq.inp.token_ids[-1]
        else:
            last_token = seq.generated_token_ids[-1]

        input_ids = torch.tensor([[last_token]], dtype=torch.long, device=self._device)
        decode_kwargs = self._prepare_decode_kwargs(seq)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                output_hidden_states=True,
                **decode_kwargs,
            )
        self._post_decode_hook(seq, out)

        # Accumulate per-step hidden states (layer 0 = embedding, layer -1 = final).
        if out.hidden_states is not None:
            seq.per_step_token_embeds.append(out.hidden_states[0].detach())
            seq.per_step_hidden_states.append(out.hidden_states[-1].detach())

        logits = out.logits[:, -1, :]  # [1, vocab_size]
        token_history = seq.inp.token_ids + seq.generated_token_ids
        next_token = self._sample(logits, sp, token_history)
        seq.generated_token_ids.append(next_token)

        # Check stop conditions
        finish_reason = None
        if next_token in sp.stop_token_ids:
            finish_reason = "stop_token"
        elif len(seq.generated_token_ids) >= sp.max_tokens:
            finish_reason = "max_tokens"

        if finish_reason is not None:
            last_hidden = out.hidden_states[-1][:, -1, :] if out.hidden_states is not None else None
            return StageOutput(
                request_id=seq.inp.request_id,
                token_ids=list(seq.generated_token_ids),
                embeddings=last_hidden,
                is_finished=True,
                finish_reason=finish_reason,
                per_step_hidden_states=list(seq.per_step_hidden_states) or None,
                per_step_token_embeds=list(seq.per_step_token_embeds) or None,
                extra=dict(seq.inp.extra),
            )
        return None

    # --- Hooks for model-specific subclasses ---
    def _prepare_decode_kwargs(self, seq: SequenceState) -> dict:
        """Return extra forward kwargs for this decode step. Subclasses override."""
        return dict(seq.inp.decode_kwargs)

    def _post_decode_hook(self, seq: SequenceState, out) -> None:
        """Hook called after each decode forward. Subclasses can update model_state."""
        pass

    @staticmethod
    def _sample(logits: torch.Tensor, sp, token_history: list[int] | None = None) -> int:
        """Sample next token. Supports greedy, temperature, top-k, top-p, repetition_penalty.

        token_history: 已出现过的 token id 序列（prompt + 已生成），用于 repetition_penalty。
        """
        if sp.repetition_penalty != 1.0 and token_history:
            unique_ids = torch.tensor(
                list(set(token_history)), dtype=torch.long, device=logits.device
            )
            logits = logits.clone()
            seen = logits[:, unique_ids]
            seen = torch.where(
                seen > 0, seen / sp.repetition_penalty, seen * sp.repetition_penalty
            )
            logits[:, unique_ids] = seen

        if sp.temperature == 0.0:
            return int(logits.argmax(dim=-1).item())

        logits = logits / sp.temperature

        if sp.top_k > 0:
            top_k = min(sp.top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1:]] = float('-inf')

        if sp.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cum_probs - torch.softmax(sorted_logits, dim=-1) > sp.top_p
            sorted_logits[remove_mask] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
