"""
Omni model-family registry.

Each model family (qwen3_omni, qwen25_omni, ...) registers a pipeline builder
via `@register_family(name)`. `build_omni_pipeline(cfg, device)` looks up the
builder by `cfg.model_family` and returns a ready-to-run Pipeline.

A builder is any callable with signature:
    (cfg: PipelineConfig, device: str) -> Pipeline
"""
from __future__ import annotations

from typing import Callable

from nano_omni.pipeline import Pipeline
from nano_omni.types import PipelineConfig

PipelineBuilder = Callable[[PipelineConfig, str], Pipeline]

_BUILDERS: dict[str, PipelineBuilder] = {}


def register_family(name: str) -> Callable[[PipelineBuilder], PipelineBuilder]:
    # Decorator: register a pipeline builder
    def _decorator(builder: PipelineBuilder) -> PipelineBuilder:
        if name in _BUILDERS:
            raise ValueError(f"model_family '{name}' already registered")
        _BUILDERS[name] = builder
        return builder
    return _decorator


def build_omni_pipeline(cfg: PipelineConfig, device: str = "cuda") -> Pipeline:
    # Dispatch to the builder registered for `cfg.model_family`.
    builder = _BUILDERS.get(cfg.model_family)
    if builder is None:
        raise ValueError(
            f"Unknown model_family '{cfg.model_family}'. "
            f"Registered: {sorted(_BUILDERS)}"
        )
    return builder(cfg, device)


def registered_families() -> list[str]:
    return sorted(_BUILDERS)
