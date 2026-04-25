# Omni model-family registry
from __future__ import annotations

from typing import Callable

from nano_omni.pipeline import Pipeline
from nano_omni.online import OnlineEngine
from nano_omni.types import ModelConfig

PipelineBuilder = Callable[[ModelConfig, str], Pipeline]
OnlineBuilder = Callable[[ModelConfig, str], OnlineEngine]

_PIPELINE_BUILDERS: dict[str, PipelineBuilder] = {}
_ONLINE_BUILDERS: dict[str, OnlineBuilder] = {}


def register_pipeline_family(name: str) -> Callable[[PipelineBuilder], PipelineBuilder]:
    # Decorator: register a pipeline builder
    def _decorator(builder: PipelineBuilder) -> PipelineBuilder:
        if name in _PIPELINE_BUILDERS:
            raise ValueError(f"model_family '{name}' already registered")
        _PIPELINE_BUILDERS[name] = builder
        return builder
    return _decorator

def register_online_family(name: str) -> Callable[[OnlineBuilder], OnlineBuilder]:
    def _decorator(builder: OnlineBuilder) -> OnlineBuilder:
        if name in _ONLINE_BUILDERS:
            raise ValueError(f"online_family '{name} already registered")
        _ONLINE_BUILDERS[name] = builder
        return builder
    return _decorator




def build_omni_pipeline(cfg: ModelConfig, device: str = "cuda") -> Pipeline:
    # Dispatch to the builder registered for `cfg.model_family`.
    builder = _PIPELINE_BUILDERS.get(cfg.model_family)
    if builder is None:
        raise ValueError(
            f"Unknown model_family for Pipeline '{cfg.model_family}'. "
            f"Registered: {sorted(_PIPELINE_BUILDERS)}"
        )
    return builder(cfg, device)

def build_omni_online_engine(cfg: ModelConfig, device: str = "cuda") -> OnlineEngine:
    builder = _ONLINE_BUILDERS.get(cfg.model_family)
    if builder is None:
        raise ValueError(
            f"Unknown model_family for OnlineEngine '{cfg.model_family}'. "
            f"Registered: {sorted(_ONLINE_BUILDERS)}"
        )
    return builder(cfg, device)



