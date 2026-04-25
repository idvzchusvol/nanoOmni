"""
Pipeline builders for Qwen-Omni family (e.g., qwen3_omni & qwen25_omni).
Workflow: build_omni_pipeline -> (construct stages with converters)
"""
from __future__ import annotations

from nano_omni.models.qwen_omni.qwen3_converters import qwen3_thinker2talker, qwen3_talker2code2wav
from nano_omni.models.qwen_omni.loader import load_qwen25_omni, load_qwen3_omni
from nano_omni.models.qwen_omni.qwen25_code2wav_stage import Qwen25OmniCode2WavStage
from nano_omni.models.qwen_omni.qwen25_converters import qwen25_thinker2talker, qwen25_talker2token2wav
from nano_omni.models.qwen_omni.qwen25_talker_stage import Qwen25OmniTalkerStage
from nano_omni.models.registry import register_pipeline_family, register_online_family



from nano_omni.pipeline import Pipeline
from nano_omni.online import OnlineEngine
from nano_omni.stage.ar_stage import ARStageEngine
from nano_omni.stage.codec_stage import CodecStageEngine
from nano_omni.types import ModelConfig

def _build_qwen3_omni_parts(cfg: ModelConfig, device: str = "cuda"):
    thinker, talker, code2wav, _full = load_qwen3_omni(cfg.model_path, device=device)
    thinker_cfg, talker_cfg, codec_cfg = cfg.stages
    stages = [
        ARStageEngine(model=thinker, config=thinker_cfg),
        ARStageEngine(model=talker, config=talker_cfg),
        CodecStageEngine(model=code2wav, config=codec_cfg),
    ]
    talker_sp = talker_cfg.sampling_params
    converters = [
        lambda out: qwen3_thinker2talker(out, talker_sampling=talker_sp),
        lambda out: qwen3_talker2code2wav(out),
    ]
    return stages, converters

@register_pipeline_family("qwen3_omni")
def build_qwen3_omni_pipeline(cfg: ModelConfig, device: str = "cuda") -> Pipeline:
    stages, converters = _build_qwen3_omni_parts(cfg, device)
    return Pipeline(stages=stages, converters=converters)

@register_online_family("qwen3_omni")
def build_qwen3_omni_online(cfg: ModelConfig, device: str = "cuda") -> OnlineEngine:
    stages, converters = _build_qwen3_omni_parts(cfg, device)
    return OnlineEngine(stages=stages, converters=converters)


def _build_qwen25_omni_parts(cfg: ModelConfig, device: str = "cuda"):
    thinker, talker, code2wav, full_model = load_qwen25_omni(cfg.model_path, device=device)
    thinker_cfg, talker_cfg, codec_cfg = cfg.stages
    speaker = "Chelsie" # _QWEN25_DEFAULT_SPEAKER
    stages = [
        ARStageEngine(model=thinker, config=thinker_cfg),
        Qwen25OmniTalkerStage(model=talker, config=talker_cfg),
        Qwen25OmniCode2WavStage(model=code2wav, config=codec_cfg),
    ]
    converters = [
        qwen25_thinker2talker(
            full_model,
            speaker=speaker,
            talker_sampling=talker_cfg.sampling_params,
        ),
        qwen25_talker2token2wav(full_model, speaker=speaker),
    ]
    return stages, converters

@register_pipeline_family("qwen25_omni")
def build_qwen25_omni_pipeline(cfg: ModelConfig, device: str = "cuda") -> Pipeline:
    stages, converters = _build_qwen25_omni_parts(cfg, device)
    return Pipeline(stages=stages, converters=converters)

@register_online_family("qwen25_omni")
def build_qwen25_omni_online(cfg: ModelConfig, device: str = "cuda") -> OnlineEngine:
    stages, converters = _build_qwen25_omni_parts(cfg, device)
    return OnlineEngine(stages=stages, converters=converters)
