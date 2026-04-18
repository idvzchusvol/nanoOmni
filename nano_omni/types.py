from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 2048
    stop_token_ids: list[int] = field(default_factory=list)
    repetition_penalty: float = 1.0


@dataclass
class OmniRequest:
    """进入 Pipeline 的原始请求。"""
    request_id: str
    text: Optional[str] = None
    audio: Optional[np.ndarray] = None      # float32, 16kHz
    images: Optional[list] = None           # list[PIL.Image]
    # None = 使用 StageConfig 里的 sampling_params（与 StageInput 约定一致）
    sampling_params: Optional[SamplingParams] = None


@dataclass
class StageInput:
    """Pipeline 传递给各 StageEngine 的输入。"""
    request_id: str
    token_ids: list[int]
    embeddings: Optional[torch.Tensor] = None   # Thinker→Talker 时携带 hidden states
    extra: dict = field(default_factory=dict)
    sampling_params: Optional[SamplingParams] = None  # None = use stage default
    # 模型特定的 forward kwargs（如 thinker_reply_part, input_text_ids 等）。
    # 由 StageEngine 子类识别并透传给 model.forward。
    prefill_kwargs: dict = field(default_factory=dict)
    decode_kwargs: dict = field(default_factory=dict)


@dataclass
class StageOutput:
    """StageEngine 返回的单次完成输出。"""
    request_id: str
    token_ids: list[int]                        # 本阶段生成的所有 token
    embeddings: Optional[torch.Tensor] = None   # 最后一步的 last hidden state
    audio: Optional[np.ndarray] = None          # Code2Wav 阶段填充
    is_finished: bool = False
    finish_reason: Optional[str] = None         # "stop_token" | "max_tokens" | None
    # 生成过程中每一步的 last hidden state / token embedding。
    # 供下游阶段（如 Qwen2.5-Omni Talker）构造 thinker_reply_part 使用。
    per_step_hidden_states: Optional[list[torch.Tensor]] = None
    per_step_token_embeds: Optional[list[torch.Tensor]] = None

    # 透传元数据（如 prompt token_ids）给下一阶段的 converter 使用。
    extra: dict = field(default_factory=dict)


@dataclass
class OmniOutput:
    """Pipeline 返回给用户的最终结果。"""
    request_id: str
    text: str
    audio: Optional[np.ndarray] = None          # float32 waveform, 24kHz


@dataclass
class StageConfig:
    """单个 Stage 的运行时配置。"""
    name: str
    stage_type: str                             # "ar" | "codec"
    max_batch_size: int = 32
    chunk_size: int = 512
    max_tokens_per_step: int = 2048
    kv_cache_max_requests: int = 32
    sampling_params: SamplingParams = field(default_factory=SamplingParams)


@dataclass
class PipelineConfig:
    """整个 Pipeline 的配置，包含所有 Stage 配置。"""
    model_path: str
    model_family: str = "qwen3_omni"
    stages: list[StageConfig] = field(default_factory=list)
