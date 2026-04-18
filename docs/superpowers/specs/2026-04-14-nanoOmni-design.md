# nanoOmni 架构设计文档

**日期**：2026-04-14  
**状态**：待实现  
**目标模型**：Qwen3-Omni（首要），可扩展至 Qwen2.5-Omni 等同类 Omni 模型

---

## 1. 项目定位

nanoOmni 是一个面向 Omni 多模态大模型的**轻量推理框架**，以 Qwen3-Omni 为首要适配目标。

核心设计原则：

- **Nano**：仅依赖 PyTorch + HuggingFace Transformers，无 vLLM 等重型依赖
- **阶段化**：将 Omni 推理过程显式拆分为多个 Stage，每个 Stage 独立封装
- **高效**：支持 KV Cache、Continuous Batching、Chunked Prefill
- **单 GPU**：所有阶段运行在同一张卡上，共享显存

---

## 2. 整体架构

### 2.1 层次结构

```
┌─────────────────────────────────────────────────────┐
│                    Pipeline                         │
│         （用户入口，协调阶段间数据流）                │
├───────────────┬─────────────────┬───────────────────┤
│  StageEngine  │   StageEngine   │   StageEngine     │
│  [Thinker]    │   [Talker]      │   [Code2Wav]      │
│  ARStage      │   ARStage       │   CodecStage      │
├───────────────┴─────────────────┴───────────────────┤
│       Scheduler      │      KVCacheManager          │
│   （每 StageEngine 各持有一份）                      │
├─────────────────────────────────────────────────────┤
│            ModelRunner（HF Transformers）            │
│              （仅负责 forward pass）                 │
└─────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
nanoOmni/
├── nano_omni/
│   ├── types.py                  # 核心数据类型
│   ├── pipeline.py               # Pipeline 协调器
│   ├── stage/
│   │   ├── base.py               # StageEngine 抽象基类
│   │   ├── ar_stage.py           # ARStageEngine（自回归）
│   │   └── codec_stage.py        # CodecStageEngine（非自回归）
│   ├── scheduler/
│   │   └── scheduler.py          # Continuous Batching + Chunked Prefill
│   ├── kv_cache/
│   │   └── manager.py            # Block 式 KV Cache 管理
│   └── models/
│       └── qwen3_omni/
│           ├── config.py         # 模型配置加载
│           ├── thinker.py        # Thinker 前向（视觉/音频编码 + LLM）
│           ├── talker.py         # Talker 前向（音频 AR）
│           ├── code2wav.py       # Code2Wav 前向（Codec 解码器）
│           └── converters.py     # 阶段间数据转换函数
├── configs/
│   └── qwen3_omni.yaml           # 阶段配置
└── examples/
    └── run.py
```

---

## 3. 核心数据类型（`types.py`）

所有跨层传递的数据结构集中定义，避免循环依赖。

```python
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
    text: str | None
    audio: np.ndarray | None        # float32, 16kHz
    images: list[PIL.Image] | None
    sampling_params: SamplingParams

@dataclass
class StageInput:
    """Pipeline 传递给各 StageEngine 的输入。"""
    request_id: str
    token_ids: list[int]
    embeddings: torch.Tensor | None  # Thinker→Talker 时携带 hidden states
    extra: dict                      # 阶段特定附加信息（如 speaker_id）

@dataclass
class StageOutput:
    """StageEngine 返回的输出。"""
    request_id: str
    token_ids: list[int]
    embeddings: torch.Tensor | None
    audio: np.ndarray | None        # Code2Wav 阶段填充
    is_finished: bool
    finish_reason: str | None       # "stop_token" | "max_tokens" | None

@dataclass
class OmniOutput:
    """Pipeline 返回给用户的最终结果。"""
    request_id: str
    text: str
    audio: np.ndarray | None        # float32 waveform, 24kHz
```

---

## 4. StageEngine 抽象（`stage/base.py`）

```python
class StageEngine(ABC):
    """
    每个推理阶段的基类。
    封装该阶段的 Scheduler、KVCacheManager 和 ModelRunner，
    对外只暴露 add_request / step / has_unfinished 三个接口。
    """

    def __init__(self, model: nn.Module, config: StageConfig):
        self.scheduler = Scheduler(config)
        self.kv_manager = KVCacheManager(config)
        self.model = model

    def add_request(self, inp: StageInput) -> None:
        """将新请求加入等待队列。"""
        self.scheduler.add(inp)

    @abstractmethod
    def step(self) -> list[StageOutput]:
        """
        执行一个调度步，返回本步已完成的请求输出。
        未完成的请求保留在内部队列中，下次 step() 继续处理。
        """

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()
```

### 4.1 ARStageEngine（`stage/ar_stage.py`）

用于 Thinker 和 Talker 两个自回归阶段，`step()` 内部流程：

```
step():
  1. scheduler.schedule()
       → ScheduleBatch(prefill_seqs, decode_seqs)
  2. kv_manager.allocate() 为新 token 扩展 Block
       → 若显存不足，抢占优先级最低的请求
  3. model.forward(prefill_seqs, decode_seqs, block_tables)
       → prefill_seqs 做 Chunked Prefill（填充 KV Cache）
       → decode_seqs 做单步 decode（读取 KV Cache）
  4. 采样（temperature / top-p / top-k）
  5. 检测 stop_token，标记完成请求
  6. kv_manager.free(completed_requests)
  7. 返回 list[StageOutput]
```

### 4.2 CodecStageEngine（`stage/codec_stage.py`）

用于 Code2Wav 阶段，非自回归，`step()` 内部流程：

```
step():
  1. 取出所有等待请求（无需调度，直接批处理）
  2. model.forward(codec_codes_batch)
       → 一次性生成 audio waveform
  3. 返回 list[StageOutput]（audio 字段填充）
```

Code2Wav 无需 KV Cache 和 Scheduler，`CodecStageEngine` 继承 `StageEngine` 并覆盖 `__init__`，跳过 `Scheduler` 和 `KVCacheManager` 的初始化；`has_unfinished()` 直接检查内部等待队列长度。

---

## 5. Pipeline 协调器（`pipeline.py`）

```python
class Pipeline:
    def __init__(
        self,
        stages: list[StageEngine],
        converters: list[Callable[[StageOutput], StageInput]],
    ):
        # converters[i]：将第 i 阶段的输出转换为第 i+1 阶段的输入
        assert len(converters) == len(stages) - 1

    def run(self, requests: list[OmniRequest]) -> list[OmniOutput]:
        # 1. 将 OmniRequest 预处理为 StageInput 并加入 Stage 0
        # 2. 逐阶段驱动：
        #    while stage_i.has_unfinished():
        #        outputs = stage_i.step()
        #        for out in outputs:
        #            stage_i+1.add_request(converters[i](out))
        # 3. 收集 Stage 最后一阶段的输出，组装 OmniOutput
```

Pipeline 本身不含任何推理逻辑，只负责：
- 调用各 StageEngine 的 `step()` 直到完成
- 用 `converters` 做阶段间数据转换
- 组装最终结果

---

## 6. Scheduler（`scheduler/scheduler.py`）

### 6.1 调度策略

采用 **Continuous Batching + Chunked Prefill**，每步执行一次以下决策：

```
每步调度逻辑：

输入：waiting_queue（新请求）、running_queue（decode 中）
预算：max_tokens_per_step（本步最多处理多少 token）

Step 1：为 running_queue 中的每个请求分配 decode 所需 KV slot（1 个 token/请求）
        → 若 KV 不足，按 FCFS 逆序抢占（swap out 到 waiting_queue 末尾）
        → 消耗预算：len(running_queue) 个 token

Step 2：从 waiting_queue 头部取请求，计算可用于 prefill 的剩余预算
        → 剩余预算 = max_tokens_per_step - decode_token_count
        → 若请求的剩余 prompt 长度 > 剩余预算，截断为一个 chunk
        → 分配 KV slot，加入本步 prefill_seqs

输出：ScheduleBatch
  prefill_seqs: list[(request, token_chunk)]
  decode_seqs:  list[request]
```

### 6.2 配置参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `max_tokens_per_step` | 每步最大 token 预算 | 2048 |
| `max_batch_size` | 最大并发请求数 | 32 |
| `chunk_size` | 单次 prefill 的 token 上限 | 512 |

---

## 7. KV Cache Manager（`kv_cache/manager.py`）

### 7.1 Block 式管理

```
物理 Block 池（每个 StageEngine 独立分配）：

 ┌──────┬──────┬──────┬──────┬──────┬──────┐
 │ B[0] │ B[1] │ B[2] │ B[3] │ B[4] │ B[5] │  ← 物理 Block
 └──────┴──────┴──────┴──────┴──────┴──────┘
    空闲   已用   已用   空闲   空闲   已用

 请求 A（已生成 10 token，block_size=8）：
   逻辑 Block 0 → 物理 B[1]  （满）
   逻辑 Block 1 → 物理 B[5]  （使用 2/8）
```

### 7.2 接口

```python
class KVCacheManager:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 num_blocks: int, block_size: int, dtype: torch.dtype): ...

    def allocate(self, request_id: str, num_new_tokens: int) -> bool:
        """为请求分配存放 num_new_tokens 的 block，返回是否成功。"""

    def free(self, request_id: str) -> None:
        """释放请求占用的所有 block。"""

    def get_block_table(self, request_id: str) -> torch.Tensor:
        """返回逻辑→物理 block 映射表（[num_logical_blocks]），传入 attention。"""

    @property
    def num_free_blocks(self) -> int: ...
```

### 7.3 显存规划（单 GPU）

三个 StageEngine 共享同一张卡，需在初始化时按比例预分配：

| 阶段 | 模型大小估算 | KV Cache 比例 |
|------|------------|--------------|
| Thinker（LLM + 视觉/音频编码器） | ~14GB | 60% |
| Talker（小型音频 AR） | ~2GB | 30% |
| Code2Wav（Codec 解码器，无 KV） | ~1GB | 0% |

实际比例通过 `configs/qwen3_omni.yaml` 中的 `kv_cache_fraction` 字段配置。

---

## 8. Qwen3-Omni 适配（`models/qwen3_omni/`）

### 8.1 三阶段对应

| Stage | StageEngine 类型 | 模型组件 | 输入 | 输出 |
|-------|----------------|---------|------|------|
| 0: Thinker | ARStageEngine | 视觉编码器 + 音频编码器 + LLM | 文本/图像/音频 token | 文本 token + hidden states |
| 1: Talker | ARStageEngine | 音频 AR 模型（8 层 RVQ） | Thinker hidden states | Codec codes |
| 2: Code2Wav | CodecStageEngine | Codec 解码器 | Codec codes | 音频波形 |

### 8.2 阶段间转换（`converters.py`）

**thinker2talker**：
- 输入：Thinker 的 `StageOutput`（token_ids + embeddings）
- 操作：构造 Talker 的 prompt（将 Thinker hidden states 作为 prefix embedding，拼接 Talker 专用起始 token）
- 输出：Talker 的 `StageInput`

**talker2code2wav**：
- 输入：Talker 的 `StageOutput`（codec code token_ids）
- 操作：将 codec codes reshape 为 Code2Wav 期望的形状（`[batch, num_codebooks, T]`）
- 输出：Code2Wav 的 `StageInput`

### 8.3 Thinker 模型结构

```
OmniRequest（文本 + 图像 + 音频）
       ↓
  [AudioEncoder] → audio_embeds
  [VisualEncoder] → visual_embeds
  [Tokenizer] → text_token_ids
       ↓
  多模态 token 合并（text + audio_embeds + visual_embeds 按位置插入）
       ↓
  [LLM Backbone]（带 KV Cache + Chunked Prefill）
       ↓
  text_token_ids + last_hidden_states
```

### 8.4 配置文件（`configs/qwen3_omni.yaml`）

```yaml
model_path: /path/to/Qwen3-Omni
stages:
  - id: 0
    name: thinker
    type: ar          # ARStageEngine
    kv_cache_fraction: 0.60
    max_tokens_per_step: 2048
    max_batch_size: 32
    chunk_size: 512
    sampling:
      temperature: 0.4
      top_p: 0.9
      max_tokens: 2048
      stop_token_ids: []
      repetition_penalty: 1.05

  - id: 1
    name: talker
    type: ar          # ARStageEngine
    kv_cache_fraction: 0.30
    max_tokens_per_step: 2048
    max_batch_size: 32
    chunk_size: 512
    sampling:
      temperature: 0.9
      top_k: 50
      max_tokens: 4096
      stop_token_ids: [2150]
      repetition_penalty: 1.05

  - id: 2
    name: code2wav
    type: codec       # CodecStageEngine
    max_batch_size: 32
```

---

## 9. 数据流全景

```
用户调用：pipeline.run([OmniRequest])

┌─ 预处理 ─────────────────────────────────────────────┐
│ 文本 → tokenize → token_ids                          │
│ 音频 → AudioEncoder → audio_embeds                   │
│ 图像 → VisualEncoder → visual_embeds                 │
│ 合并为多模态 token 序列 → StageInput[0]               │
└──────────────────────────────────────────────────────┘
           ↓ add_request(StageInput)
┌─ Stage 0: Thinker ───────────────────────────────────┐
│ Scheduler → ScheduleBatch                            │
│ KVCacheManager.allocate()                            │
│ LLM.forward(prefill_chunk / decode_step)             │
│ 采样 → 生成 text_token_ids + hidden_states           │
│ 完成后 → StageOutput[0]                              │
└──────────────────────────────────────────────────────┘
           ↓ thinker2talker(StageOutput[0])
           → StageInput[1]
┌─ Stage 1: Talker ────────────────────────────────────┐
│ Scheduler → ScheduleBatch                            │
│ KVCacheManager.allocate()                            │
│ AudioAR.forward(hidden_states prefix + decode_step)  │
│ 生成 codec_codes（8 层 RVQ token_ids）               │
│ 完成后 → StageOutput[1]                              │
└──────────────────────────────────────────────────────┘
           ↓ talker2code2wav(StageOutput[1])
           → StageInput[2]
┌─ Stage 2: Code2Wav ──────────────────────────────────┐
│ Codec.forward(codec_codes) → audio waveform          │
│ 完成后 → StageOutput[2]（audio 字段填充）             │
└──────────────────────────────────────────────────────┘
           ↓ 组装
    OmniOutput(text, audio)
```

---

## 10. 扩展性设计

当需要适配其他 Omni 模型（如 Qwen2.5-Omni）时，只需：

1. 在 `models/` 下新增对应目录，实现 `thinker.py / talker.py / code2wav.py`
2. 实现 `converters.py` 中的阶段间转换函数
3. 新增 `configs/<model_name>.yaml`

Pipeline、Scheduler、KVCacheManager、StageEngine 基类无需修改。

新阶段类型（如纯文本 AR 无 Talker）通过配置 `type: ar` 并将 `converters` 列表对应位置留空即可跳过。

---

## 11. 不在范围内（当前版本）

- 流式输出（streaming token/audio）
- 跨 GPU 流水线（Thinker/Talker 分卡）
- 异步 Thinker-Talker 并行（async chunk）
- 量化（int8/fp8）
- LoRA 适配
- 前缀缓存（Prefix Caching）
