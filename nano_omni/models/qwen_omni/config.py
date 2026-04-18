from __future__ import annotations

import yaml

from nano_omni.types import PipelineConfig, SamplingParams, StageConfig


def load_pipeline_config(yaml_path: str) -> PipelineConfig:
    """Load PipelineConfig from a YAML file."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    stages = []
    for s in raw.get("stages", []):
        sampling_raw = s.get("sampling", {})
        sp = SamplingParams(
            temperature=sampling_raw.get("temperature", 1.0),
            top_p=sampling_raw.get("top_p", 1.0),
            top_k=sampling_raw.get("top_k", -1),
            max_tokens=sampling_raw.get("max_tokens", 2048),
            stop_token_ids=sampling_raw.get("stop_token_ids", []),
            repetition_penalty=sampling_raw.get("repetition_penalty", 1.0),
        )
        stages.append(StageConfig(
            name=s["name"],
            stage_type=s["type"],
            max_batch_size=s.get("max_batch_size", 32),
            chunk_size=s.get("chunk_size", 512),
            max_tokens_per_step=s.get("max_tokens_per_step", 2048),
            kv_cache_max_requests=s.get("kv_cache_max_requests", 32),
            sampling_params=sp,
        ))

    return PipelineConfig(
        model_path=raw["model_path"],
        model_family=raw.get("model_family", "qwen3_omni"),
        stages=stages,
    )
