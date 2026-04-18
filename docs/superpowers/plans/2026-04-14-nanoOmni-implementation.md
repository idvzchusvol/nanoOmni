# nanoOmni Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现面向 Qwen3-Omni 的轻量多阶段推理框架，支持 KV Cache、Continuous Batching 和 Chunked Prefill。

**Architecture:** Pipeline 协调三个 StageEngine（Thinker/Talker/Code2Wav），每个 StageEngine 封装 Scheduler + KVCacheManager + 模型前向。KV Cache 使用 HF `DynamicCache`（比 PagedAttention 简单，适合 nano），KVCacheManager 负责容量跟踪。Continuous Batching 在 Scheduler 层实现：每步混合 prefill chunk 和 decode 步骤。

**Tech Stack:** Python 3.10+，PyTorch 2.0+，HuggingFace Transformers 4.50+（含 `DynamicCache`、`Qwen3OmniMoe*`），pytest，PyYAML

---

## File Map

| 文件 | 职责 |
|------|------|
| `nano_omni/types.py` | 核心数据类型：`OmniRequest`, `StageInput`, `StageOutput`, `OmniOutput`, `SamplingParams`, `StageConfig` |
| `nano_omni/kv_cache/manager.py` | `KVCacheManager`：基于 HF `DynamicCache` 的 per-request KV Cache 存储与容量管理 |
| `nano_omni/scheduler/scheduler.py` | `Scheduler`：Continuous Batching + Chunked Prefill 调度，输出 `ScheduleBatch` |
| `nano_omni/stage/base.py` | `StageEngine` 抽象基类 |
| `nano_omni/stage/ar_stage.py` | `ARStageEngine`：自回归 stage（Thinker & Talker 复用） |
| `nano_omni/stage/codec_stage.py` | `CodecStageEngine`：非自回归 codec stage（Code2Wav） |
| `nano_omni/pipeline.py` | `Pipeline`：驱动各 StageEngine，传递阶段间数据 |
| `nano_omni/models/qwen3_omni/config.py` | 从 YAML 加载 `PipelineConfig`，从 HF 加载模型配置 |
| `nano_omni/models/qwen3_omni/thinker.py` | `Thinker`：包装 HF Thinker 模型，提供 `prepare_inputs` / `forward` |
| `nano_omni/models/qwen3_omni/talker.py` | `Talker`：包装 HF Talker 模型，提供 `forward` |
| `nano_omni/models/qwen3_omni/code2wav.py` | `Code2Wav`：包装 HF Codec 解码器，提供 `forward` |
| `nano_omni/models/qwen3_omni/converters.py` | `thinker2talker`, `talker2code2wav`：阶段间数据转换 |
| `configs/qwen3_omni.yaml` | Pipeline 配置文件 |
| `examples/run.py` | 端到端运行示例 |
| `tests/test_types.py` | types.py 单元测试 |
| `tests/test_kv_cache.py` | KVCacheManager 单元测试 |
| `tests/test_scheduler.py` | Scheduler 单元测试 |
| `tests/test_ar_stage.py` | ARStageEngine 单元测试（mock 模型） |
| `tests/test_codec_stage.py` | CodecStageEngine 单元测试（mock 模型） |
| `tests/test_pipeline.py` | Pipeline 集成测试（mock stages） |
| `tests/models/qwen3_omni/test_converters.py` | converters 单元测试 |

---

## Task 1: 项目结构与环境

**Files:**
- Create: `pyproject.toml`
- Create: `nano_omni/__init__.py`
- Create: `nano_omni/kv_cache/__init__.py`
- Create: `nano_omni/scheduler/__init__.py`
- Create: `nano_omni/stage/__init__.py`
- Create: `nano_omni/models/__init__.py`
- Create: `nano_omni/models/qwen3_omni/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/models/__init__.py`
- Create: `tests/models/qwen3_omni/__init__.py`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "nano_omni"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "transformers>=4.50",
    "numpy",
    "pillow",
    "pyyaml",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-mock"]

[tool.setuptools.packages.find]
where = ["."]
include = ["nano_omni*"]
```

- [ ] **Step 2: 创建所有 `__init__.py` 空文件**

```bash
touch nano_omni/__init__.py \
      nano_omni/kv_cache/__init__.py \
      nano_omni/scheduler/__init__.py \
      nano_omni/stage/__init__.py \
      nano_omni/models/__init__.py \
      nano_omni/models/qwen3_omni/__init__.py \
      tests/__init__.py \
      tests/models/__init__.py \
      tests/models/qwen3_omni/__init__.py
```

- [ ] **Step 3: 安装开发依赖**

```bash
pip install -e ".[dev]"
```

预期输出：`Successfully installed nano_omni-0.1.0`

- [ ] **Step 4: 验证测试框架可用**

```bash
pytest --collect-only 2>&1 | head -5
```

预期：`no tests ran` 或 `0 items`（无报错）

- [ ] **Step 5: Commit**

```bash
git init && git add pyproject.toml nano_omni/ tests/
git commit -m "feat: project scaffold for nanoOmni"
```

---

## Task 2: 核心数据类型（`types.py`）

**Files:**
- Create: `nano_omni/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_types.py
import numpy as np
import torch
import pytest
from nano_omni.types import (
    SamplingParams, OmniRequest, StageInput, StageOutput, OmniOutput, StageConfig
)

def test_sampling_params_defaults():
    p = SamplingParams()
    assert p.temperature == 1.0
    assert p.top_p == 1.0
    assert p.top_k == -1
    assert p.max_tokens == 2048
    assert p.stop_token_ids == []
    assert p.repetition_penalty == 1.0

def test_omni_request_text_only():
    req = OmniRequest(request_id="r1", text="hello")
    assert req.request_id == "r1"
    assert req.text == "hello"
    assert req.audio is None
    assert req.images is None

def test_stage_input_defaults():
    inp = StageInput(request_id="r1", token_ids=[1, 2, 3])
    assert inp.embeddings is None
    assert inp.extra == {}
    assert inp.sampling_params.temperature == 1.0

def test_stage_output_defaults():
    out = StageOutput(request_id="r1", token_ids=[42])
    assert out.embeddings is None
    assert out.audio is None
    assert out.is_finished is False
    assert out.finish_reason is None

def test_stage_output_with_embeddings():
    emb = torch.randn(1, 4096)
    out = StageOutput(request_id="r1", token_ids=[42], embeddings=emb, is_finished=True, finish_reason="stop_token")
    assert out.embeddings.shape == (1, 4096)

def test_omni_output():
    audio = np.zeros(24000, dtype=np.float32)
    out = OmniOutput(request_id="r1", text="hi", audio=audio)
    assert out.text == "hi"
    assert len(out.audio) == 24000

def test_stage_config_defaults():
    cfg = StageConfig(name="thinker", stage_type="ar")
    assert cfg.max_batch_size == 32
    assert cfg.chunk_size == 512
    assert cfg.max_tokens_per_step == 2048
    assert cfg.kv_cache_max_requests == 32
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_types.py -v
```

预期：`ImportError: cannot import name 'SamplingParams' from 'nano_omni.types'`

- [ ] **Step 3: 实现 `nano_omni/types.py`**

```python
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
    sampling_params: SamplingParams = field(default_factory=SamplingParams)


@dataclass
class StageInput:
    """Pipeline 传递给各 StageEngine 的输入。"""
    request_id: str
    token_ids: list[int]
    embeddings: Optional[torch.Tensor] = None   # Thinker→Talker 时携带 hidden states
    extra: dict = field(default_factory=dict)
    sampling_params: SamplingParams = field(default_factory=SamplingParams)


@dataclass
class StageOutput:
    """StageEngine 返回的单次完成输出。"""
    request_id: str
    token_ids: list[int]                        # 本阶段生成的所有 token
    embeddings: Optional[torch.Tensor] = None   # 最后一步的 last hidden state
    audio: Optional[np.ndarray] = None          # Code2Wav 阶段填充
    is_finished: bool = False
    finish_reason: Optional[str] = None         # "stop_token" | "max_tokens" | None


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
    stages: list[StageConfig] = field(default_factory=list)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_types.py -v
```

预期：全部 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add nano_omni/types.py tests/test_types.py
git commit -m "feat: core data types (OmniRequest, StageInput, StageOutput, StageConfig)"
```

---

## Task 3: KV Cache Manager（`kv_cache/manager.py`）

**Files:**
- Create: `nano_omni/kv_cache/manager.py`
- Create: `tests/test_kv_cache.py`

> **设计说明**：nanoOmni 使用 HF `DynamicCache` 作为底层 KV 存储（per-request），`KVCacheManager` 负责容量跟踪与生命周期管理。这比 PagedAttention block table 简单，适合 nano 场景；PagedAttention 需要自定义 CUDA kernel。

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_kv_cache.py
import pytest
from transformers import DynamicCache
from nano_omni.kv_cache.manager import KVCacheManager


def test_manager_initial_state():
    mgr = KVCacheManager(max_requests=4)
    assert mgr.num_active == 0
    assert mgr.has_capacity() is True


def test_get_or_create_returns_cache():
    mgr = KVCacheManager(max_requests=4)
    cache = mgr.get_or_create("req_1")
    assert isinstance(cache, DynamicCache)


def test_get_or_create_same_object():
    mgr = KVCacheManager(max_requests=4)
    c1 = mgr.get_or_create("req_1")
    c2 = mgr.get_or_create("req_1")
    assert c1 is c2


def test_capacity_tracking():
    mgr = KVCacheManager(max_requests=2)
    mgr.get_or_create("req_1")
    assert mgr.num_active == 1
    assert mgr.has_capacity() is True
    mgr.get_or_create("req_2")
    assert mgr.num_active == 2
    assert mgr.has_capacity() is False


def test_free_releases_slot():
    mgr = KVCacheManager(max_requests=2)
    mgr.get_or_create("req_1")
    mgr.get_or_create("req_2")
    assert mgr.has_capacity() is False
    mgr.free("req_1")
    assert mgr.num_active == 1
    assert mgr.has_capacity() is True


def test_free_nonexistent_is_noop():
    mgr = KVCacheManager(max_requests=4)
    mgr.free("does_not_exist")   # should not raise


def test_get_cache_raises_when_full():
    mgr = KVCacheManager(max_requests=1)
    mgr.get_or_create("req_1")
    with pytest.raises(RuntimeError, match="KV cache capacity"):
        mgr.get_or_create("req_2")
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_kv_cache.py -v
```

预期：`ImportError` 或 `ModuleNotFoundError`

- [ ] **Step 3: 实现 `nano_omni/kv_cache/manager.py`**

```python
from __future__ import annotations

from transformers import DynamicCache


class KVCacheManager:
    """
    Per-request KV Cache 管理器。
    使用 HF DynamicCache 作为底层存储，跟踪容量上限。
    """

    def __init__(self, max_requests: int = 32):
        self.max_requests = max_requests
        self._caches: dict[str, DynamicCache] = {}

    def get_or_create(self, request_id: str) -> DynamicCache:
        """返回请求的 DynamicCache，不存在时新建。容量已满时抛出 RuntimeError。"""
        if request_id in self._caches:
            return self._caches[request_id]
        if len(self._caches) >= self.max_requests:
            raise RuntimeError(
                f"KV cache capacity exceeded: max_requests={self.max_requests}"
            )
        cache = DynamicCache()
        self._caches[request_id] = cache
        return cache

    def free(self, request_id: str) -> None:
        """释放请求的 KV Cache，不存在时静默忽略。"""
        self._caches.pop(request_id, None)

    def has_capacity(self) -> bool:
        return len(self._caches) < self.max_requests

    @property
    def num_active(self) -> int:
        return len(self._caches)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_kv_cache.py -v
```

预期：全部 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add nano_omni/kv_cache/manager.py tests/test_kv_cache.py
git commit -m "feat: KVCacheManager using HF DynamicCache with capacity tracking"
```

---

## Task 4: Scheduler（`scheduler/scheduler.py`）

**Files:**
- Create: `nano_omni/scheduler/scheduler.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_scheduler.py
import pytest
from nano_omni.types import StageInput, SamplingParams
from nano_omni.scheduler.scheduler import Scheduler, SequenceState, ScheduleBatch


def make_input(rid: str, length: int) -> StageInput:
    return StageInput(request_id=rid, token_ids=list(range(length)),
                      sampling_params=SamplingParams(max_tokens=10))


def test_empty_scheduler():
    sched = Scheduler(max_batch_size=4, chunk_size=8, max_tokens_per_step=32)
    assert sched.has_unfinished() is False


def test_add_request():
    sched = Scheduler(max_batch_size=4, chunk_size=8, max_tokens_per_step=32)
    sched.add(make_input("r1", 5))
    assert sched.has_unfinished() is True


def test_short_prompt_fully_prefilled_in_one_step():
    sched = Scheduler(max_batch_size=4, chunk_size=8, max_tokens_per_step=32)
    sched.add(make_input("r1", 4))  # prompt length 4 < chunk_size 8
    batch = sched.schedule()
    assert len(batch.prefill_seqs) == 1
    seq, chunk = batch.prefill_seqs[0]
    assert seq.inp.request_id == "r1"
    assert chunk == [0, 1, 2, 3]
    assert len(batch.decode_seqs) == 0
    # After full prefill, seq moves to running
    assert seq.prefill_offset == 4
    assert seq.is_prefilling is False


def test_long_prompt_chunked_across_steps():
    sched = Scheduler(max_batch_size=4, chunk_size=4, max_tokens_per_step=32)
    sched.add(make_input("r1", 10))   # needs 3 chunks: 4+4+2
    batch1 = sched.schedule()
    assert len(batch1.prefill_seqs) == 1
    _, chunk1 = batch1.prefill_seqs[0]
    assert chunk1 == [0, 1, 2, 3]   # first 4 tokens
    assert sched.has_unfinished() is True  # still prefilling
    batch2 = sched.schedule()
    _, chunk2 = batch2.prefill_seqs[0]
    assert chunk2 == [4, 5, 6, 7]
    batch3 = sched.schedule()
    _, chunk3 = batch3.prefill_seqs[0]
    assert chunk3 == [8, 9]


def test_running_requests_go_to_decode():
    sched = Scheduler(max_batch_size=4, chunk_size=8, max_tokens_per_step=32)
    sched.add(make_input("r1", 3))  # short prompt, fully prefilled
    batch1 = sched.schedule()
    seq = batch1.prefill_seqs[0][0]
    # Simulate decode started: seq is now in running
    assert not seq.is_prefilling
    batch2 = sched.schedule()
    assert len(batch2.decode_seqs) == 1
    assert batch2.decode_seqs[0].inp.request_id == "r1"


def test_finish_removes_from_running():
    sched = Scheduler(max_batch_size=4, chunk_size=8, max_tokens_per_step=32)
    sched.add(make_input("r1", 3))
    sched.schedule()   # moves r1 to running
    sched.finish("r1")
    assert sched.has_unfinished() is False


def test_token_budget_limits_concurrent_prefill():
    # budget=6: decode takes 0 (no running), 2 prefills of 3 tokens each
    sched = Scheduler(max_batch_size=4, chunk_size=3, max_tokens_per_step=6)
    sched.add(make_input("r1", 3))
    sched.add(make_input("r2", 3))
    batch = sched.schedule()
    assert len(batch.prefill_seqs) == 2


def test_decode_budget_limits_new_prefill():
    # max_tokens_per_step=4, 3 running decode seqs consume 3 tokens → 1 left for prefill
    sched = Scheduler(max_batch_size=8, chunk_size=4, max_tokens_per_step=4)
    # put 3 seqs in running
    for rid in ["r1", "r2", "r3"]:
        sched.add(make_input(rid, 2))
        sched.schedule()   # each short, fully prefilled, now running
    sched.add(make_input("r4", 4))
    batch = sched.schedule()
    assert len(batch.decode_seqs) == 3
    # prefill chunk limited to 1 token (budget=4-3=1)
    if batch.prefill_seqs:
        _, chunk = batch.prefill_seqs[0]
        assert len(chunk) == 1
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_scheduler.py -v
```

预期：`ImportError`

- [ ] **Step 3: 实现 `nano_omni/scheduler/scheduler.py`**

```python
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from nano_omni.types import StageInput


@dataclass
class SequenceState:
    """追踪单个请求在某 Stage 内的处理进度。"""
    inp: StageInput
    generated_token_ids: list[int] = field(default_factory=list)
    prefill_offset: int = 0     # inp.token_ids 中已 prefill 的 token 数
    is_prefilling: bool = True  # False 表示已完成 prefill，进入 decode 阶段


@dataclass
class ScheduleBatch:
    prefill_seqs: list[tuple[SequenceState, list[int]]]  # (state, token_chunk)
    decode_seqs: list[SequenceState]


class Scheduler:
    """
    Continuous Batching + Chunked Prefill 调度器。

    每次 schedule() 返回一个 ScheduleBatch：
    - prefill_seqs：本步需要做 prefill 的请求及其 token chunk
    - decode_seqs：本步需要做 decode 的请求（已完成 prefill，自回归中）

    Token 预算规则：
      decode_seqs 占用 len(decode_seqs) 个 token 预算（每请求 1 个）
      剩余预算分配给 prefill（每次最多 chunk_size 个 token/请求）
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
        """将新请求加入等待队列。"""
        self._waiting.append(SequenceState(inp=inp))

    def schedule(self) -> ScheduleBatch:
        budget = self.max_tokens_per_step

        # Step 1: 所有 running 请求各用 1 个 token（decode step）
        decode_seqs = list(self._running)
        budget -= len(decode_seqs)

        # Step 2: 从 waiting 中取请求做 prefill chunk
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
                # 全部 prefill 完成，移入 running
                self._waiting.popleft()
                seq.is_prefilling = False
                self._running.append(seq)
            else:
                # 还有剩余，留在 waiting 队首，等下次 schedule
                break

        return ScheduleBatch(prefill_seqs=prefill_seqs, decode_seqs=decode_seqs)

    def finish(self, request_id: str) -> SequenceState | None:
        """将请求从 running 中移除（生成结束）。"""
        for i, seq in enumerate(self._running):
            if seq.inp.request_id == request_id:
                return self._running.pop(i)
        return None

    def has_unfinished(self) -> bool:
        return bool(self._waiting or self._running)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_scheduler.py -v
```

预期：全部 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add nano_omni/scheduler/scheduler.py tests/test_scheduler.py
git commit -m "feat: Scheduler with continuous batching and chunked prefill"
```

---

## Task 5: StageEngine 基类与 ARStageEngine

**Files:**
- Create: `nano_omni/stage/base.py`
- Create: `nano_omni/stage/ar_stage.py`
- Create: `tests/test_ar_stage.py`

- [ ] **Step 1: 实现 `nano_omni/stage/base.py`**

（此文件是 ABC，通过 ARStageEngine 的测试间接验证）

```python
from __future__ import annotations

from abc import ABC, abstractmethod

from nano_omni.types import StageConfig, StageInput, StageOutput


class StageEngine(ABC):
    """
    每个推理阶段的基类。
    对外只暴露 add_request / step / has_unfinished 三个接口。
    """

    def __init__(self, config: StageConfig):
        self.config = config

    @abstractmethod
    def add_request(self, inp: StageInput) -> None:
        """将新请求加入处理队列。"""

    @abstractmethod
    def step(self) -> list[StageOutput]:
        """
        执行一个调度步，返回本步已完成的请求列表。
        未完成的请求保留在内部队列，下次 step() 继续处理。
        """

    @abstractmethod
    def has_unfinished(self) -> bool:
        """是否还有未完成的请求。"""
```

- [ ] **Step 2: 编写 ARStageEngine 失败测试**

```python
# tests/test_ar_stage.py
import torch
import pytest
from unittest.mock import MagicMock, patch
from transformers import DynamicCache

from nano_omni.types import StageConfig, StageInput, StageOutput, SamplingParams
from nano_omni.stage.ar_stage import ARStageEngine


def make_config(max_tokens=5) -> StageConfig:
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=[99])
    return StageConfig(
        name="test_ar",
        stage_type="ar",
        max_batch_size=4,
        chunk_size=4,
        max_tokens_per_step=16,
        kv_cache_max_requests=4,
        sampling_params=sp,
    )


def make_mock_model(vocab_size=100, hidden_size=64):
    """
    返回一个 mock model，forward() 返回 (logits, DynamicCache, hidden_state)。
    logits shape: [batch, seq_len, vocab_size]
    """
    model = MagicMock()
    def fake_forward(input_ids, past_key_values, output_hidden_states=False, **kwargs):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, vocab_size)
        logits[:, :, 7] = 100.0   # always predict token 7
        out = MagicMock()
        out.logits = logits
        out.past_key_values = past_key_values if past_key_values else DynamicCache()
        hidden = torch.zeros(batch, seq, hidden_size)
        out.hidden_states = (hidden,) * 3  # 3 layers
        return out
    model.side_effect = fake_forward
    return model


def test_add_request_registers_as_unfinished():
    engine = ARStageEngine(model=make_mock_model(), config=make_config())
    assert engine.has_unfinished() is False
    engine.add_request(StageInput(request_id="r1", token_ids=[1, 2, 3]))
    assert engine.has_unfinished() is True


def test_step_prefills_then_decodes():
    engine = ARStageEngine(model=make_mock_model(), config=make_config(max_tokens=3))
    engine.add_request(StageInput(request_id="r1", token_ids=[1, 2]))
    # First step: prefill
    out1 = engine.step()
    # r1 is short (2 tokens < chunk_size 4), fully prefilled + first decode token sampled
    # Still not finished (max_tokens=3)
    assert engine.has_unfinished() is True
    assert out1 == []  # not finished yet


def test_step_returns_output_when_stop_token():
    """stop_token_id=99, model always predicts 7, so runs until max_tokens."""
    engine = ARStageEngine(model=make_mock_model(), config=make_config(max_tokens=2))
    engine.add_request(StageInput(request_id="r1", token_ids=[1]))
    # Run until finished
    outputs = []
    for _ in range(10):
        outputs.extend(engine.step())
        if not engine.has_unfinished():
            break
    assert len(outputs) == 1
    out = outputs[0]
    assert out.request_id == "r1"
    assert len(out.token_ids) == 2   # max_tokens=2
    assert out.is_finished is True
    assert out.finish_reason == "max_tokens"


def test_step_stops_on_stop_token():
    model = make_mock_model()
    # Override to predict token 99 (stop token) immediately
    def fake_forward_stop(input_ids, past_key_values, **kwargs):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, 100)
        logits[:, :, 99] = 100.0  # predict stop token
        out = MagicMock()
        out.logits = logits
        out.past_key_values = past_key_values if past_key_values else DynamicCache()
        out.hidden_states = (torch.zeros(batch, seq, 64),) * 3
        return out
    model.side_effect = fake_forward_stop

    engine = ARStageEngine(model=model, config=make_config(max_tokens=10))
    engine.add_request(StageInput(request_id="r1", token_ids=[1, 2]))

    outputs = []
    for _ in range(5):
        outputs.extend(engine.step())
        if not engine.has_unfinished():
            break
    assert len(outputs) == 1
    assert outputs[0].finish_reason == "stop_token"


def test_multiple_requests_batched():
    engine = ARStageEngine(model=make_mock_model(), config=make_config(max_tokens=2))
    engine.add_request(StageInput(request_id="r1", token_ids=[1]))
    engine.add_request(StageInput(request_id="r2", token_ids=[2]))
    outputs = []
    for _ in range(10):
        outputs.extend(engine.step())
        if not engine.has_unfinished():
            break
    rids = {o.request_id for o in outputs}
    assert rids == {"r1", "r2"}
```

- [ ] **Step 3: 运行测试确认失败**

```bash
pytest tests/test_ar_stage.py -v
```

预期：`ImportError: cannot import name 'ARStageEngine'`

- [ ] **Step 4: 实现 `nano_omni/stage/ar_stage.py`**

```python
from __future__ import annotations

import torch

from nano_omni.kv_cache.manager import KVCacheManager
from nano_omni.scheduler.scheduler import Scheduler, SequenceState
from nano_omni.stage.base import StageEngine
from nano_omni.types import StageConfig, StageInput, StageOutput


class ARStageEngine(StageEngine):
    """
    自回归 StageEngine，用于 Thinker 和 Talker 阶段。

    每次 step() 执行：
      1. Scheduler 决定本步 prefill seqs 和 decode seqs
      2. 对每个 prefill seq 运行前向（填充 KV Cache）
      3. 对每个 decode seq 运行单步前向（生成下一 token）
      4. 采样、检测终止条件
      5. 返回本步完成的请求
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
                # 全部 prefill 完成后，立即生成第一个 decode token
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
        """运行一个 prefill chunk，更新 KV Cache。不采样。"""
        cache = self.kv_manager.get_or_create(seq.inp.request_id)
        input_ids = torch.tensor([chunk], dtype=torch.long)
        with torch.no_grad():
            self.model(
                input_ids=input_ids,
                past_key_values=cache,
                output_hidden_states=True,
            )
        # DynamicCache 已 in-place 更新，无需额外操作

    def _run_decode_step(self, seq: SequenceState) -> StageOutput | None:
        """
        运行单步 decode，采样下一 token。
        若触发终止条件，返回 StageOutput；否则返回 None（继续 decode）。
        """
        sp = seq.inp.sampling_params or self._default_sp
        cache = self.kv_manager.get_or_create(seq.inp.request_id)

        # 确定输入：首次 decode 用 prefill 最后一个 token；后续用上次采样 token
        if not seq.generated_token_ids:
            # 首次 decode：用 prompt 最后一个 token 触发预测
            last_token = seq.inp.token_ids[-1]
        else:
            last_token = seq.generated_token_ids[-1]

        input_ids = torch.tensor([[last_token]], dtype=torch.long)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                output_hidden_states=True,
            )

        logits = out.logits[:, -1, :]  # [1, vocab_size]
        next_token = self._sample(logits, sp)
        seq.generated_token_ids.append(next_token)

        # 检查终止条件
        finish_reason = None
        if next_token in sp.stop_token_ids:
            finish_reason = "stop_token"
        elif len(seq.generated_token_ids) >= sp.max_tokens:
            finish_reason = "max_tokens"

        if finish_reason is not None:
            last_hidden = out.hidden_states[-1][:, -1, :]   # [1, hidden_size]
            return StageOutput(
                request_id=seq.inp.request_id,
                token_ids=list(seq.generated_token_ids),
                embeddings=last_hidden,
                is_finished=True,
                finish_reason=finish_reason,
            )
        return None

    @staticmethod
    def _sample(logits: torch.Tensor, sp) -> int:
        """从 logits 采样下一 token，支持 temperature / top-k / top-p / greedy。"""
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
            mask = cum_probs - torch.softmax(sorted_logits, dim=-1) > sp.top_p
            sorted_logits[mask] = float('-inf')
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_ar_stage.py -v
```

预期：全部 `PASSED`

- [ ] **Step 6: Commit**

```bash
git add nano_omni/stage/base.py nano_omni/stage/ar_stage.py tests/test_ar_stage.py
git commit -m "feat: StageEngine base class and ARStageEngine with continuous batching"
```

---

## Task 6: CodecStageEngine（`stage/codec_stage.py`）

**Files:**
- Create: `nano_omni/stage/codec_stage.py`
- Create: `tests/test_codec_stage.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_codec_stage.py
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock

from nano_omni.types import StageConfig, StageInput, StageOutput
from nano_omni.stage.codec_stage import CodecStageEngine


def make_config() -> StageConfig:
    return StageConfig(name="code2wav", stage_type="codec", max_batch_size=4)


def make_mock_codec():
    """mock codec model: forward(codec_codes) → np.ndarray audio"""
    model = MagicMock()
    def fake_forward(codec_codes):
        # codec_codes: [batch, num_codebooks, T]
        batch = codec_codes.shape[0]
        return np.zeros((batch, 24000), dtype=np.float32)
    model.side_effect = fake_forward
    return model


def test_initially_no_unfinished():
    engine = CodecStageEngine(model=make_mock_codec(), config=make_config())
    assert engine.has_unfinished() is False


def test_add_request_marks_unfinished():
    engine = CodecStageEngine(model=make_mock_codec(), config=make_config())
    engine.add_request(StageInput(request_id="r1", token_ids=[1, 2, 3, 4, 5, 6, 7, 8]))
    assert engine.has_unfinished() is True


def test_step_returns_audio_output():
    engine = CodecStageEngine(model=make_mock_codec(), config=make_config())
    # token_ids represent 8 RVQ codes (num_codebooks=1, T=8 for simplicity)
    engine.add_request(StageInput(
        request_id="r1",
        token_ids=list(range(8)),
        extra={"num_codebooks": 1},
    ))
    outputs = engine.step()
    assert len(outputs) == 1
    out = outputs[0]
    assert out.request_id == "r1"
    assert out.is_finished is True
    assert out.audio is not None
    assert isinstance(out.audio, np.ndarray)


def test_step_processes_all_waiting():
    engine = CodecStageEngine(model=make_mock_codec(), config=make_config())
    engine.add_request(StageInput(request_id="r1", token_ids=list(range(8))))
    engine.add_request(StageInput(request_id="r2", token_ids=list(range(8))))
    outputs = engine.step()
    assert len(outputs) == 2
    assert engine.has_unfinished() is False


def test_step_empty_queue_returns_empty():
    engine = CodecStageEngine(model=make_mock_codec(), config=make_config())
    outputs = engine.step()
    assert outputs == []
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_codec_stage.py -v
```

预期：`ImportError: cannot import name 'CodecStageEngine'`

- [ ] **Step 3: 实现 `nano_omni/stage/codec_stage.py`**

```python
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from nano_omni.stage.base import StageEngine
from nano_omni.types import StageConfig, StageInput, StageOutput


class CodecStageEngine(StageEngine):
    """
    非自回归 CodecStageEngine，用于 Code2Wav 阶段。

    不需要 KV Cache 和 Scheduler：每次 step() 处理所有等待请求，
    一次性非自回归前向生成音频波形。
    """

    def __init__(self, model, config: StageConfig):
        super().__init__(config)
        self.model = model
        self._waiting: deque[StageInput] = deque()

    def add_request(self, inp: StageInput) -> None:
        self._waiting.append(inp)

    def has_unfinished(self) -> bool:
        return bool(self._waiting)

    def step(self) -> list[StageOutput]:
        if not self._waiting:
            return []

        inputs = list(self._waiting)
        self._waiting.clear()

        outputs = []
        for inp in inputs:
            num_codebooks = inp.extra.get("num_codebooks", 8)
            T = len(inp.token_ids) // num_codebooks
            codec_codes = torch.tensor(
                inp.token_ids, dtype=torch.long
            ).reshape(1, num_codebooks, T)

            with torch.no_grad():
                audio = self.model(codec_codes)  # np.ndarray [1, num_samples] or [num_samples]

            if isinstance(audio, np.ndarray) and audio.ndim == 2:
                audio = audio[0]

            outputs.append(StageOutput(
                request_id=inp.request_id,
                token_ids=inp.token_ids,
                audio=audio,
                is_finished=True,
                finish_reason="completed",
            ))

        return outputs
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_codec_stage.py -v
```

预期：全部 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add nano_omni/stage/codec_stage.py tests/test_codec_stage.py
git commit -m "feat: CodecStageEngine for non-autoregressive Code2Wav stage"
```

---

## Task 7: Pipeline 协调器（`pipeline.py`）

**Files:**
- Create: `nano_omni/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/test_pipeline.py
import numpy as np
import pytest
from unittest.mock import MagicMock

from nano_omni.types import OmniRequest, StageInput, StageOutput, OmniOutput, SamplingParams
from nano_omni.pipeline import Pipeline
from nano_omni.stage.base import StageEngine


def make_mock_stage(outputs_per_request: dict[str, StageOutput]) -> StageEngine:
    """
    Mock StageEngine：add_request 记录请求，step() 在第一次调用时返回所有输出。
    """
    stage = MagicMock(spec=StageEngine)
    pending = {}
    returned = set()

    def add(inp: StageInput):
        pending[inp.request_id] = inp

    def step():
        results = []
        for rid, inp in list(pending.items()):
            if rid not in returned:
                returned.add(rid)
                out = outputs_per_request.get(rid)
                if out:
                    results.append(out)
        return results

    def has_unfinished():
        return any(rid not in returned for rid in pending)

    stage.add_request.side_effect = add
    stage.step.side_effect = step
    stage.has_unfinished.side_effect = has_unfinished
    return stage


def make_thinker_out(rid: str) -> StageOutput:
    import torch
    return StageOutput(
        request_id=rid,
        token_ids=[10, 11, 12],
        embeddings=torch.zeros(1, 64),
        is_finished=True,
    )


def make_talker_out(rid: str) -> StageOutput:
    return StageOutput(
        request_id=rid,
        token_ids=list(range(16)),   # 16 codec codes
        is_finished=True,
    )


def make_codec_out(rid: str) -> StageOutput:
    return StageOutput(
        request_id=rid,
        token_ids=list(range(16)),
        audio=np.zeros(24000, dtype=np.float32),
        is_finished=True,
    )


def test_pipeline_single_request():
    thinker = make_mock_stage({"r1": make_thinker_out("r1")})
    talker = make_mock_stage({"r1": make_talker_out("r1")})
    codec = make_mock_stage({"r1": make_codec_out("r1")})

    def thinker2talker(out: StageOutput) -> StageInput:
        return StageInput(request_id=out.request_id, token_ids=out.token_ids)

    def talker2codec(out: StageOutput) -> StageInput:
        return StageInput(request_id=out.request_id, token_ids=out.token_ids,
                          extra={"num_codebooks": 1})

    pipeline = Pipeline(
        stages=[thinker, talker, codec],
        converters=[thinker2talker, talker2codec],
    )

    req = OmniRequest(request_id="r1", text="hello")
    results = pipeline.run(
        requests=[req],
        preprocess=lambda r: StageInput(request_id=r.request_id, token_ids=[1, 2, 3]),
    )

    assert len(results) == 1
    out = results[0]
    assert out.request_id == "r1"
    assert out.text == "10 11 12"    # token_ids joined as text placeholder
    assert out.audio is not None


def test_pipeline_two_stages_no_audio():
    """两阶段 Pipeline（无 codec），结果无 audio。"""
    thinker = make_mock_stage({"r1": make_thinker_out("r1")})
    talker = make_mock_stage({"r1": make_talker_out("r1")})

    pipeline = Pipeline(
        stages=[thinker, talker],
        converters=[lambda out: StageInput(request_id=out.request_id, token_ids=out.token_ids)],
    )
    req = OmniRequest(request_id="r1", text="hello")
    results = pipeline.run(
        requests=[req],
        preprocess=lambda r: StageInput(request_id=r.request_id, token_ids=[1]),
    )
    assert results[0].audio is None


def test_pipeline_converter_count_mismatch():
    thinker = make_mock_stage({})
    with pytest.raises(AssertionError):
        Pipeline(stages=[thinker], converters=[lambda x: x])   # 1 stage, 1 converter → error


def test_pipeline_multiple_requests():
    thinker_outs = {rid: make_thinker_out(rid) for rid in ["r1", "r2"]}
    talker_outs = {rid: make_talker_out(rid) for rid in ["r1", "r2"]}
    codec_outs = {rid: make_codec_out(rid) for rid in ["r1", "r2"]}

    pipeline = Pipeline(
        stages=[
            make_mock_stage(thinker_outs),
            make_mock_stage(talker_outs),
            make_mock_stage(codec_outs),
        ],
        converters=[
            lambda out: StageInput(request_id=out.request_id, token_ids=out.token_ids),
            lambda out: StageInput(request_id=out.request_id, token_ids=out.token_ids,
                                   extra={"num_codebooks": 1}),
        ],
    )
    reqs = [OmniRequest(request_id=rid, text="hi") for rid in ["r1", "r2"]]
    results = pipeline.run(
        requests=reqs,
        preprocess=lambda r: StageInput(request_id=r.request_id, token_ids=[1]),
    )
    assert len(results) == 2
    assert {r.request_id for r in results} == {"r1", "r2"}
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_pipeline.py -v
```

预期：`ImportError: cannot import name 'Pipeline'`

- [ ] **Step 3: 实现 `nano_omni/pipeline.py`**

```python
from __future__ import annotations

from typing import Callable

from nano_omni.stage.base import StageEngine
from nano_omni.types import OmniOutput, OmniRequest, StageInput, StageOutput


class Pipeline:
    """
    协调多个 StageEngine 的推理流水线。

    run() 驱动每个 StageEngine 的 step() 循环直到所有请求完成，
    用 converters[i] 将第 i 阶段的输出转换为第 i+1 阶段的输入。
    """

    def __init__(
        self,
        stages: list[StageEngine],
        converters: list[Callable[[StageOutput], StageInput]],
    ):
        assert len(converters) == len(stages) - 1, (
            f"需要 {len(stages)-1} 个 converter，got {len(converters)}"
        )
        self.stages = stages
        self.converters = converters

    def run(
        self,
        requests: list[OmniRequest],
        preprocess: Callable[[OmniRequest], StageInput],
    ) -> list[OmniOutput]:
        """
        Args:
            requests: 原始请求列表
            preprocess: 将 OmniRequest 转换为 Stage 0 的 StageInput
        Returns:
            每个请求对应的 OmniOutput，顺序与 requests 一致
        """
        # 将所有请求送入第一个 stage
        for req in requests:
            stage_inp = preprocess(req)
            self.stages[0].add_request(stage_inp)

        # 逐阶段驱动
        final_outputs: dict[str, StageOutput] = {}
        for i, stage in enumerate(self.stages):
            stage_results: dict[str, StageOutput] = {}
            while stage.has_unfinished():
                for out in stage.step():
                    stage_results[out.request_id] = out
            # 若不是最后一个 stage，将输出转换并送入下一 stage
            if i < len(self.stages) - 1:
                for out in stage_results.values():
                    next_inp = self.converters[i](out)
                    self.stages[i + 1].add_request(next_inp)
            else:
                final_outputs = stage_results

        # 组装 OmniOutput
        results = []
        for req in requests:
            final = final_outputs.get(req.request_id)
            if final is None:
                continue
            text = " ".join(str(t) for t in final.token_ids) if final.token_ids else ""
            results.append(OmniOutput(
                request_id=req.request_id,
                text=text,
                audio=final.audio,
            ))
        return results
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_pipeline.py -v
```

预期：全部 `PASSED`

- [ ] **Step 5: 全量测试**

```bash
pytest tests/ -v --ignore=tests/models
```

预期：全部 `PASSED`

- [ ] **Step 6: Commit**

```bash
git add nano_omni/pipeline.py tests/test_pipeline.py
git commit -m "feat: Pipeline orchestrator with stage-to-stage data flow"
```

---

## Task 8: Qwen3-Omni 配置加载（`models/qwen3_omni/config.py`）

**Files:**
- Create: `nano_omni/models/qwen3_omni/config.py`
- Create: `configs/qwen3_omni.yaml`
- Create: `tests/models/qwen3_omni/test_config.py`

- [ ] **Step 1: 创建 `configs/qwen3_omni.yaml`**

```yaml
model_path: /path/to/Qwen3-Omni   # 替换为实际路径

stages:
  - id: 0
    name: thinker
    type: ar
    kv_cache_max_requests: 32
    max_tokens_per_step: 2048
    max_batch_size: 32
    chunk_size: 512
    sampling:
      temperature: 0.4
      top_p: 0.9
      top_k: -1
      max_tokens: 2048
      stop_token_ids: []
      repetition_penalty: 1.05

  - id: 1
    name: talker
    type: ar
    kv_cache_max_requests: 32
    max_tokens_per_step: 2048
    max_batch_size: 32
    chunk_size: 512
    sampling:
      temperature: 0.9
      top_p: 1.0
      top_k: 50
      max_tokens: 4096
      stop_token_ids: [2150]
      repetition_penalty: 1.05

  - id: 2
    name: code2wav
    type: codec
    max_batch_size: 32
```

- [ ] **Step 2: 编写失败测试**

```python
# tests/models/qwen3_omni/test_config.py
import os
import pytest
from nano_omni.models.qwen3_omni.config import load_pipeline_config


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
    cfg = load_pipeline_config(str(cfg_file))
    assert cfg.model_path == "/fake/path"
    assert len(cfg.stages) == 2


def test_stage_config_fields(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(YAML_CONTENT)
    cfg = load_pipeline_config(str(cfg_file))
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
    cfg = load_pipeline_config(str(cfg_file))
    codec_cfg = cfg.stages[1]
    assert codec_cfg.stage_type == "codec"
    assert codec_cfg.max_batch_size == 4
    # codec stage has no sampling params, uses defaults
    assert codec_cfg.sampling_params.temperature == 1.0
```

- [ ] **Step 3: 运行测试确认失败**

```bash
pytest tests/models/qwen3_omni/test_config.py -v
```

预期：`ImportError`

- [ ] **Step 4: 实现 `nano_omni/models/qwen3_omni/config.py`**

```python
from __future__ import annotations

import yaml

from nano_omni.types import PipelineConfig, SamplingParams, StageConfig


def load_pipeline_config(yaml_path: str) -> PipelineConfig:
    """从 YAML 文件加载 PipelineConfig。"""
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

    return PipelineConfig(model_path=raw["model_path"], stages=stages)
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/models/qwen3_omni/test_config.py -v
```

预期：全部 `PASSED`

- [ ] **Step 6: Commit**

```bash
git add nano_omni/models/qwen3_omni/config.py configs/qwen3_omni.yaml \
        tests/models/qwen3_omni/test_config.py
git commit -m "feat: YAML config loader for Qwen3-Omni pipeline"
```

---

## Task 9: Thinker 模型包装（`models/qwen3_omni/thinker.py`）

**Files:**
- Create: `nano_omni/models/qwen3_omni/thinker.py`

> **说明**：Thinker 包装 HF 的 `Qwen3OmniMoeThinkerForConditionalGeneration`（通过 `trust_remote_code=True` 加载）。
> 提供两个接口：`prepare_inputs`（OmniRequest → token_ids + embeddings）和 `forward`（供 ARStageEngine 调用）。
> 实际模型权重路径需在 `configs/qwen3_omni.yaml` 的 `model_path` 中配置。

- [ ] **Step 1: 实现 `nano_omni/models/qwen3_omni/thinker.py`**

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoProcessor, DynamicCache

if TYPE_CHECKING:
    from nano_omni.types import OmniRequest, StageInput


class Thinker:
    """
    包装 HF Qwen3-Omni Thinker 模型。

    用法：
        thinker = Thinker.from_pretrained("/path/to/Qwen3-Omni")
        # 预处理原始请求
        token_ids = thinker.prepare_inputs(request)
        # ARStageEngine 内部调用
        logits, cache, hidden = thinker.forward(input_ids, cache)
    """

    # 特殊 token id（来自 Qwen3-Omni tokenizer）
    AUDIO_START_TOKEN_ID = 151669
    AUDIO_END_TOKEN_ID = 151670

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        """从 HF 模型路径加载 Thinker。"""
        from transformers import AutoModelForCausalLM
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, subfolder="thinker"
        )
        # Qwen3-omni 通过 model_stage 参数区分 thinker/talker/code2wav
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device,
        )
        # 若模型包含 thinker 子模块，提取之
        if hasattr(model, "thinker"):
            model = model.thinker
        model.eval()
        return cls(model=model, processor=processor)

    def prepare_inputs(self, request: "OmniRequest") -> "StageInput":
        """
        将 OmniRequest 转换为 Thinker 的 StageInput。
        使用 processor 处理文本/音频/图像，生成多模态 token_ids。
        """
        from nano_omni.types import StageInput

        messages = []
        if request.text:
            messages.append({"role": "user", "content": request.text})

        # 构造 processor 输入
        inputs_dict: dict = {"text": self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )}
        if request.audio is not None:
            inputs_dict["audio"] = request.audio
            inputs_dict["sampling_rate"] = 16000
        if request.images:
            inputs_dict["images"] = request.images

        inputs = self.processor(**inputs_dict, return_tensors="pt")
        token_ids = inputs["input_ids"][0].tolist()

        return StageInput(
            request_id=request.request_id,
            token_ids=token_ids,
            extra={"processor_inputs": inputs},
            sampling_params=request.sampling_params,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache | None,
        output_hidden_states: bool = True,
        **kwargs,
    ):
        """
        前向传播接口（供 ARStageEngine 调用）。
        返回值与 HF CausalLM 一致：output.logits, output.past_key_values, output.hidden_states
        """
        return self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
```

- [ ] **Step 2: 验证模块可导入（无权重测试）**

```bash
python -c "from nano_omni.models.qwen3_omni.thinker import Thinker; print('OK')"
```

预期：`OK`

- [ ] **Step 3: Commit**

```bash
git add nano_omni/models/qwen3_omni/thinker.py
git commit -m "feat: Thinker wrapper for Qwen3-Omni HF model"
```

---

## Task 10: Talker 模型包装（`models/qwen3_omni/talker.py`）

**Files:**
- Create: `nano_omni/models/qwen3_omni/talker.py`

- [ ] **Step 1: 实现 `nano_omni/models/qwen3_omni/talker.py`**

```python
from __future__ import annotations

import torch
from transformers import DynamicCache


class Talker:
    """
    包装 HF Qwen3-Omni Talker 模型（小型音频 AR，生成 RVQ codec codes）。

    Talker 的输入是 Thinker 的 hidden states（作为 prefix embedding）
    + Talker 专用起始 token，自回归生成 codec codes。
    """

    # Talker 词表特殊 token（codec token ids）
    CODEC_BOS_TOKEN_ID = 4197
    CODEC_EOS_TOKEN_ID = 4198  # = stop_token_ids 中的 2150（实际值依模型版本）

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        """从 HF 模型路径加载 Talker 子模块。"""
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device,
        )
        if hasattr(model, "talker"):
            model = model.talker
        model.eval()
        return cls(model=model)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache | None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        """
        前向传播。
        - 首次调用（prefill）：传入 inputs_embeds（Thinker hidden states）
        - 后续 decode：传入 input_ids（上一步采样的 codec token）
        """
        return self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
```

- [ ] **Step 2: 验证可导入**

```bash
python -c "from nano_omni.models.qwen3_omni.talker import Talker; print('OK')"
```

预期：`OK`

- [ ] **Step 3: Commit**

```bash
git add nano_omni/models/qwen3_omni/talker.py
git commit -m "feat: Talker wrapper for Qwen3-Omni audio AR model"
```

---

## Task 11: Code2Wav 模型包装（`models/qwen3_omni/code2wav.py`）

**Files:**
- Create: `nano_omni/models/qwen3_omni/code2wav.py`

- [ ] **Step 1: 实现 `nano_omni/models/qwen3_omni/code2wav.py`**

```python
from __future__ import annotations

import numpy as np
import torch


class Code2Wav:
    """
    包装 Qwen3-Omni Code2Wav 模块（RVQ Codec 解码器）。
    输入：codec codes tensor [batch, num_codebooks, T]
    输出：audio waveform np.ndarray [batch, num_samples]，float32，24kHz
    """

    NUM_CODEBOOKS = 8   # Qwen3-Omni 使用 8 层 RVQ

    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cuda", dtype=torch.float32):
        """从 HF 模型路径加载 Code2Wav 子模块。"""
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device,
        )
        if hasattr(model, "code2wav"):
            model = model.code2wav
        elif hasattr(model, "vocoder"):
            model = model.vocoder
        model.eval()
        return cls(model=model)

    def forward(self, codec_codes: torch.Tensor) -> np.ndarray:
        """
        Args:
            codec_codes: [batch, num_codebooks, T]（torch.long）
        Returns:
            audio: [batch, num_samples]（np.float32，24kHz）
        """
        with torch.no_grad():
            waveform = self.model(codec_codes)
        if isinstance(waveform, torch.Tensor):
            return waveform.cpu().float().numpy()
        return waveform

    def __call__(self, codec_codes: torch.Tensor) -> np.ndarray:
        return self.forward(codec_codes)
```

- [ ] **Step 2: 验证可导入**

```bash
python -c "from nano_omni.models.qwen3_omni.code2wav import Code2Wav; print('OK')"
```

预期：`OK`

- [ ] **Step 3: Commit**

```bash
git add nano_omni/models/qwen3_omni/code2wav.py
git commit -m "feat: Code2Wav wrapper for Qwen3-Omni codec decoder"
```

---

## Task 12: 阶段间数据转换（`models/qwen3_omni/converters.py`）

**Files:**
- Create: `nano_omni/models/qwen3_omni/converters.py`
- Create: `tests/models/qwen3_omni/test_converters.py`

- [ ] **Step 1: 编写失败测试**

```python
# tests/models/qwen3_omni/test_converters.py
import torch
import pytest
from nano_omni.types import StageOutput, StageInput, SamplingParams
from nano_omni.models.qwen3_omni.converters import thinker2talker, talker2code2wav


TALKER_SAMPLING = SamplingParams(temperature=0.9, top_k=50, max_tokens=4096,
                                  stop_token_ids=[2150], repetition_penalty=1.05)


def test_thinker2talker_basic():
    out = StageOutput(
        request_id="r1",
        token_ids=[10, 11, 12],
        embeddings=torch.zeros(1, 64),
        is_finished=True,
    )
    inp = thinker2talker(out, talker_sampling=TALKER_SAMPLING)
    assert isinstance(inp, StageInput)
    assert inp.request_id == "r1"
    assert inp.embeddings is not None
    assert inp.sampling_params.stop_token_ids == [2150]


def test_thinker2talker_no_embeddings_raises():
    out = StageOutput(request_id="r1", token_ids=[1, 2], embeddings=None, is_finished=True)
    with pytest.raises(ValueError, match="embeddings"):
        thinker2talker(out)


def test_talker2code2wav_reshapes_tokens():
    # 16 tokens, 8 codebooks → T=2
    out = StageOutput(
        request_id="r1",
        token_ids=list(range(16)),
        is_finished=True,
    )
    inp = talker2code2wav(out, num_codebooks=8)
    assert inp.request_id == "r1"
    assert inp.extra["num_codebooks"] == 8
    # token_ids preserved for reshaping in CodecStageEngine
    assert inp.token_ids == list(range(16))


def test_talker2code2wav_default_codebooks():
    out = StageOutput(request_id="r1", token_ids=list(range(8)), is_finished=True)
    inp = talker2code2wav(out)
    assert inp.extra["num_codebooks"] == 8   # default
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/models/qwen3_omni/test_converters.py -v
```

预期：`ImportError`

- [ ] **Step 3: 实现 `nano_omni/models/qwen3_omni/converters.py`**

```python
from __future__ import annotations

from nano_omni.types import SamplingParams, StageInput, StageOutput

# 默认 Talker 采样参数（与 configs/qwen3_omni.yaml 中的 talker sampling 对应）
_DEFAULT_TALKER_SAMPLING = SamplingParams(
    temperature=0.9,
    top_k=50,
    max_tokens=4096,
    stop_token_ids=[2150],
    repetition_penalty=1.05,
)

_DEFAULT_NUM_CODEBOOKS = 8


def thinker2talker(
    thinker_out: StageOutput,
    talker_sampling: SamplingParams | None = None,
) -> StageInput:
    """
    将 Thinker 的 StageOutput 转换为 Talker 的 StageInput。

    - embeddings：Thinker 最后一步的 last hidden state，作为 Talker 的 prefix embedding
    - token_ids：Thinker 生成的文本 token_ids（供参考/调试，Talker 实际用 embeddings）
    - sampling_params：Talker 专用采样参数（codec 生成）
    """
    if thinker_out.embeddings is None:
        raise ValueError(
            f"thinker2talker: request {thinker_out.request_id} 缺少 embeddings，"
            "请确保 Thinker 的 ARStageEngine 开启了 output_hidden_states=True"
        )
    return StageInput(
        request_id=thinker_out.request_id,
        token_ids=thinker_out.token_ids,
        embeddings=thinker_out.embeddings,
        sampling_params=talker_sampling or _DEFAULT_TALKER_SAMPLING,
        extra={"thinker_token_ids": thinker_out.token_ids},
    )


def talker2code2wav(
    talker_out: StageOutput,
    num_codebooks: int = _DEFAULT_NUM_CODEBOOKS,
) -> StageInput:
    """
    将 Talker 的 StageOutput 转换为 Code2Wav 的 StageInput。

    - token_ids：Talker 生成的 codec code token_ids，共 num_codebooks * T 个
    - extra["num_codebooks"]：供 CodecStageEngine.step() 中 reshape 使用
    """
    return StageInput(
        request_id=talker_out.request_id,
        token_ids=talker_out.token_ids,
        extra={"num_codebooks": num_codebooks},
    )
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/models/qwen3_omni/test_converters.py -v
```

预期：全部 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add nano_omni/models/qwen3_omni/converters.py \
        tests/models/qwen3_omni/test_converters.py
git commit -m "feat: stage converters thinker2talker and talker2code2wav"
```

---

## Task 13: 端到端装配与 Examples（`examples/run.py`）

**Files:**
- Create: `examples/run.py`
- Modify: `nano_omni/models/qwen3_omni/__init__.py`

- [ ] **Step 1: 实现 `nano_omni/models/qwen3_omni/__init__.py`**（便利导出）

```python
from nano_omni.models.qwen3_omni.config import load_pipeline_config
from nano_omni.models.qwen3_omni.code2wav import Code2Wav
from nano_omni.models.qwen3_omni.converters import thinker2talker, talker2code2wav
from nano_omni.models.qwen3_omni.talker import Talker
from nano_omni.models.qwen3_omni.thinker import Thinker

__all__ = [
    "load_pipeline_config",
    "Thinker", "Talker", "Code2Wav",
    "thinker2talker", "talker2code2wav",
]
```

- [ ] **Step 2: 实现 `examples/run.py`**

```python
#!/usr/bin/env python3
"""
nanoOmni 端到端推理示例：Qwen3-Omni 文本→音频对话

用法：
    python examples/run.py --config configs/qwen3_omni.yaml \
        --text "你好，请用中文介绍一下你自己。"

依赖：
    - model_path 在 qwen3_omni.yaml 中已正确配置
    - 安装了 nano_omni（pip install -e .）
"""
import argparse
import soundfile as sf
import torch

from nano_omni.models.qwen3_omni import (
    Code2Wav, Talker, Thinker,
    load_pipeline_config, thinker2talker, talker2code2wav,
)
from nano_omni.pipeline import Pipeline
from nano_omni.stage.ar_stage import ARStageEngine
from nano_omni.stage.codec_stage import CodecStageEngine
from nano_omni.types import OmniRequest


def build_pipeline(config_path: str, device: str = "cuda") -> tuple[Pipeline, Thinker]:
    cfg = load_pipeline_config(config_path)
    model_path = cfg.model_path

    thinker_model = Thinker.from_pretrained(model_path, device=device)
    talker_model = Talker.from_pretrained(model_path, device=device)
    code2wav_model = Code2Wav.from_pretrained(model_path, device=device)

    thinker_cfg = cfg.stages[0]
    talker_cfg = cfg.stages[1]
    codec_cfg = cfg.stages[2]

    pipeline = Pipeline(
        stages=[
            ARStageEngine(model=thinker_model, config=thinker_cfg),
            ARStageEngine(model=talker_model, config=talker_cfg),
            CodecStageEngine(model=code2wav_model, config=codec_cfg),
        ],
        converters=[
            lambda out: thinker2talker(out, talker_sampling=talker_cfg.sampling_params),
            lambda out: talker2code2wav(out),
        ],
    )
    return pipeline, thinker_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_omni.yaml")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[nanoOmni] 加载模型配置：{args.config}")
    pipeline, thinker_model = build_pipeline(args.config, device=args.device)

    request = OmniRequest(request_id="demo", text=args.text)

    print(f"[nanoOmni] 开始推理：{args.text}")
    results = pipeline.run(
        requests=[request],
        preprocess=lambda req: thinker_model.prepare_inputs(req),
    )

    if not results:
        print("[nanoOmni] 推理失败，无输出")
        return

    out = results[0]
    print(f"[nanoOmni] 生成文本：{out.text}")
    if out.audio is not None:
        sf.write(args.output, out.audio, samplerate=24000)
        print(f"[nanoOmni] 音频已保存：{args.output}")
    else:
        print("[nanoOmni] 无音频输出")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 全量测试**

```bash
pytest tests/ -v
```

预期：全部 `PASSED`，无报错

- [ ] **Step 4: Commit**

```bash
git add nano_omni/models/qwen3_omni/__init__.py examples/run.py
git commit -m "feat: end-to-end assembly and run.py example for Qwen3-Omni"
```

---

## 自审 Checklist

### Spec 覆盖检查

| 设计节 | 对应 Task |
|--------|-----------|
| types.py | Task 2 |
| KVCacheManager（HF DynamicCache） | Task 3 |
| Scheduler（Continuous Batching + Chunked Prefill） | Task 4 |
| StageEngine ABC | Task 5 |
| ARStageEngine | Task 5 |
| CodecStageEngine | Task 6 |
| Pipeline | Task 7 |
| 配置加载 | Task 8 |
| Thinker | Task 9 |
| Talker | Task 10 |
| Code2Wav | Task 11 |
| converters | Task 12 |
| 端到端示例 | Task 13 |

### 类型一致性检查

- `StageInput.sampling_params`：在 types.py（Task 2）定义，scheduler.py（Task 4）和 ar_stage.py（Task 5）使用 ✓
- `StageOutput.embeddings`：在 types.py 定义，ar_stage.py 填充，converters.py 读取 ✓
- `StageOutput.audio`：在 types.py 定义，codec_stage.py 填充，pipeline.py 读取 ✓
- `KVCacheManager.get_or_create(request_id)` / `.free(request_id)`：Task 3 定义，Task 5 使用 ✓
- `Scheduler.add(inp: StageInput)` / `.schedule()` / `.finish(request_id)`：Task 4 定义，Task 5 使用 ✓
- `Pipeline.run(requests, preprocess)` 签名：Task 7 定义，Task 13 调用 ✓

### 已知简化点（相对设计文档）

1. **KV Cache**：使用 HF `DynamicCache`（动态增长）而非 block-based PagedAttention。真正的 block 管理需要自定义 CUDA attention kernel。
2. **Decode batching**：当前实现逐请求串行 decode（无真正 GPU 批次并行）。批量 decode 需要 KV Cache padding + attention mask，是下一步优化点。
3. **`Pipeline.run()` 的 `text` 字段**：当前用 `" ".join(token_ids)` 作为占位符；实际使用时需在 Pipeline 或 preprocess 中传入 tokenizer 用于 detokenize。
