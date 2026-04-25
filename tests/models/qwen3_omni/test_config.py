import pytest
from nano_omni.models.qwen_omni.config import load_model_config

YAML_CONTENT = """
model_path: /fake/path
stages:
  - id: 0
    name: thinker
    type: ar
    kv_cache_max_requests: 16
    max_tokens_per_step: 1024
    max_batch_size: 8
    chunk_size: 256
    sampling:
      temperature: 0.5
      top_p: 0.95
      top_k: -1
      max_tokens: 512
      stop_token_ids: []
      repetition_penalty: 1.0
  - id: 1
    name: code2wav
    type: codec
    max_batch_size: 4
"""

def test_load_pipeline_config(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(YAML_CONTENT)
    cfg = load_model_config(str(cfg_file))
    assert cfg.model_path == "/fake/path"
    assert len(cfg.stages) == 2

def test_stage_config_fields(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(YAML_CONTENT)
    cfg = load_model_config(str(cfg_file))
    thinker_cfg = cfg.stages[0]
    assert thinker_cfg.name == "thinker"
    assert thinker_cfg.stage_type == "ar"
    assert thinker_cfg.max_batch_size == 8
    assert thinker_cfg.chunk_size == 256
    assert thinker_cfg.kv_cache_max_requests == 16
    assert thinker_cfg.sampling_params.temperature == 0.5
    assert thinker_cfg.sampling_params.max_tokens == 512

def test_codec_stage_defaults(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(YAML_CONTENT)
    cfg = load_model_config(str(cfg_file))
    codec_cfg = cfg.stages[1]
    assert codec_cfg.stage_type == "codec"
    assert codec_cfg.max_batch_size == 4
    assert codec_cfg.sampling_params.temperature == 1.0  # default
