from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from nano_omni.types import StageInput


@dataclass
class SequenceState:
    """Tracks a single request's progress within a stage."""
    inp: StageInput
    generated_token_ids: list[int] = field(default_factory=list)
    prefill_offset: int = 0     # how many tokens of inp.token_ids have been prefilled
    is_prefilling: bool = True  # False = prefill done, now in decode phase

    # Per-step hidden state accumulation (populated during prefill & decode when needed).
    # List indexed by generation step: idx 0 covers the prefill span (length = prompt_len),
    # idx 1.. cover each decoded token (length = 1 each).
    per_step_hidden_states: list[torch.Tensor] = field(default_factory=list)
    per_step_token_embeds: list[torch.Tensor] = field(default_factory=list)

    # Model-specific mutable state carried across decode steps (e.g. Qwen2.5-Omni Talker's
    # sliding thinker_reply_part). Stage subclasses can read/write freely.
    model_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleBatch:
    prefill_seqs: list[tuple[SequenceState, list[int]]]  # (state, token_chunk)
    decode_seqs: list[SequenceState]


class Scheduler:
    """
    Continuous Batching + Chunked Prefill scheduler.

    Each schedule() returns a ScheduleBatch:
    - prefill_seqs: requests to prefill this step (with their token chunk)
    - decode_seqs: requests to decode this step (already completed prefill)

    Token budget: decode_seqs consume len(decode_seqs) tokens (1 each).
    Remaining budget goes to prefill chunks (up to chunk_size per request).
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        chunk_size: int = 512,
        max_tokens_per_step: int = 2048,
    ):
        self.max_batch_size = max_batch_size
        self.chunk_size = chunk_size
        self.max_tokens_per_step = max_tokens_per_step
        self._waiting: deque[SequenceState] = deque()
        self._running: list[SequenceState] = []

    def add(self, inp: StageInput) -> None:
        """Add new request to waiting queue."""
        self._waiting.append(SequenceState(inp=inp))

    def schedule(self) -> ScheduleBatch:
        budget = self.max_tokens_per_step

        # All running requests do one decode step (1 token each)
        decode_seqs = list(self._running)
        budget -= len(decode_seqs)

        # Fill remaining budget with prefill chunks from waiting queue
        prefill_seqs: list[tuple[SequenceState, list[int]]] = []
        while (
            self._waiting
            and budget > 0
            and len(self._running) + len(prefill_seqs) < self.max_batch_size
        ):
            seq = self._waiting[0]
            remaining = seq.inp.token_ids[seq.prefill_offset:]
            chunk_len = min(len(remaining), self.chunk_size, budget)
            chunk = remaining[:chunk_len]
            seq.prefill_offset += chunk_len
            budget -= chunk_len
            prefill_seqs.append((seq, chunk))

            if seq.prefill_offset >= len(seq.inp.token_ids):
                # Full prefill done: move to running
                self._waiting.popleft()
                seq.is_prefilling = False
                self._running.append(seq)
            else:
                # Still needs more prefill: leave at front of waiting, stop for this step
                break

        return ScheduleBatch(prefill_seqs=prefill_seqs, decode_seqs=decode_seqs)

    def finish(self, request_id: str) -> SequenceState | None:
        """Remove finished request from running queue."""
        for i, seq in enumerate(self._running):
            if seq.inp.request_id == request_id:
                return self._running.pop(i)
        return None

    def has_unfinished(self) -> bool:
        return bool(self._waiting or self._running)
