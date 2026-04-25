# nanoOmni 在线调度引擎(OnlineEngine)设计

**日期**:2026-04-22
**状态**:设计已确认,等待实现
**定位**:作者的 nanoOmni 探索练习——骨架由 Claude 提供,调度主体逻辑由作者补全

---

## 1. 背景与目标

当前 `nano_omni/pipeline.py` 的 `Pipeline.run()` 是**离线批处理**模式:一次性接收所有请求,把 stage 0(Thinker)全部跑完,再把输出整体推给 stage 1(Talker),以此类推。这种模式对 Omni 流式语音场景不友好——请求无法中途加入,用户必须等全流程结束才能拿到音频。

本次新增 **`OnlineEngine`**,提供**在线调度**能力:请求异步到达、输出异步流式产出、生命周期可管理(start / shutdown)。

### 调度实时性分级(在本设计中的取舍)

- **A 级:请求级异步入队**。请求任意时刻到达并行 batch,单请求内部仍按 stage 串行(Thinker 说完整句才开始 Talker)。
- **B 级:跨 stage 的 token 级流水**。Thinker 每吐几个 token,Talker 立刻开始消费;Talker 的 codec token 流向 Code2Wav 合成首包音频。Qwen-Omni 论文意义上的"streaming talker"就在这一档。
- **C 级:流式输入**。音频请求边录边送,Thinker 增量 prefill。

**本设计覆盖 A 级**,在骨架的注释里明确标出 B / C 级的扩展点,作者可后期按需演进。

### 不改动现有代码

- `pipeline.py`、`stage/*.py`、`scheduler/*.py`、`types.py` 全部保持不变。
- `Pipeline` 继续作为离线入口存在,`OnlineEngine` 与它并列,共享同一套 `StageEngine` / `Scheduler` / `KVCacheManager`。

---

## 2. 层次概念澄清:两层"调度"不冲突

| 层级 | 类 | 粒度 | 关心的事 |
|---|---|---|---|
| 外层(本次新增) | `OnlineEngine` | 跨 stage、跨请求 | 请求何时到达?stage 0 输出何时流到 stage 1?何时把 chunk 推给用户? |
| 内层(已存在) | `Scheduler`(在 `nano_omni/scheduler/scheduler.py`) | 单 stage 内、单次 forward | 这次 `step()` 该把哪些请求塞进一个 batch?prefill / decode 如何在 token 预算下混排? |

`OnlineEngine._engine_loop` 调用 `stage.step()`,`stage.step()` 内部调用 `self.scheduler.schedule()` 组装 GPU forward 的 batch。**两者名字都叫"调度"但不是同一件事**。现有 `Scheduler` 完全复用,不动。

---

## 3. 架构总览

```
        ┌───────── 用户(asyncio 协程) ─────────┐
        │   async for chunk in engine.submit(req):  │
        └───────────────────────────────────────────┘
                        │ (1) 放入 pending queue
                        ▼
        ┌────────── OnlineEngine (facade) ──────────┐
        │  pending_q : queue.Queue                  │
        │  handles   : dict[req_id → _RequestHandle]│◄── (4) engine 线程发布 chunk / 完成
        │  engine_thr: threading.Thread             │
        └───────────────────────────────────────────┘
                        │ (2) engine 线程取出
                        ▼
        ┌───────── engine_loop(一个线程)─────────┐
        │  while running:                           │
        │    drain pending_q → stages[0].add(...)   │
        │    for i, stage in enumerate(stages):     │
        │        for out in stage.step():           │
        │            if i == last: publish(chunk)   │
        │            else: stages[i+1].add(conv(out))│
        │    idle sleep if no work                  │
        └───────────────────────────────────────────┘
                        │ (3) publish 用
                        │     loop.call_soon_threadsafe
                        ▼
        ┌──── _RequestHandle(每请求一个)────┐
        │  out_q : asyncio.Queue[OmniChunk]  │
        │  loop  : asyncio event loop 引用   │
        └────────────────────────────────────┘
```

三个关键边界:

1. **asyncio 世界 → engine 线程**:`submit()` 把请求塞进 thread-safe `queue.Queue`,立即返回一个 async generator。
2. **engine 线程内部**:沿用 A 级语义——每个请求在 stage 里按现有 `add_request/step` 跑完再流向下一个 stage,只在**最后一个 stage**才逐个产出 `OmniChunk`。
3. **engine 线程 → asyncio 世界**:engine 线程拿到完成的 `StageOutput` 后,用 `loop.call_soon_threadsafe(out_q.put_nowait, chunk)` 把 chunk 推回对应请求的异步队列。

> **重要**:以上三个边界的**具体实现全部留为 TODO**,骨架只提供类型签名 + 详尽注释。这是作者探索 nanoOmni 调度机制的练习主体。

---

## 4. 对外 API 与数据类型

### 4.1 新增数据类型

```python
# nano_omni/online/types.py
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

@dataclass
class OmniChunk:
    """
    流式输出的单个数据块。

    A 级语义(本骨架):
        - Thinker 全部跑完才会出 type="text"(整段文本,只发一次)
        - Code2Wav 每完成一个音频片段发一次 type="audio"
        - 完成时发一次 type="done"
    B 级扩展点:
        - text 可改成 per-token delta,Thinker 每 sample 一个 token 就推一次
        - audio 的 chunk 粒度由 Code2Wav 的 chunked decode 决定
    """
    request_id: str
    type: Literal["text", "audio", "done", "error"]
    text: Optional[str] = None
    audio: Optional[np.ndarray] = None   # float32, 24kHz
    error: Optional[str] = None
```

`OmniRequest` 复用 `nano_omni/types.py` 中已有的同名类型。

### 4.2 公共 API

```python
# nano_omni/online/engine.py
class OnlineEngine:
    """
    在线调度引擎。包装已有 StageEngine 列表,提供异步入队 + 流式输出接口。

    用法:
        engine = OnlineEngine(stages=[...], converters=[...])
        engine.start()                        # 启动后台 engine 线程
        async for chunk in engine.submit(req):
            ...
        await engine.shutdown()

    与 Pipeline 的关系:
        Pipeline      —— 离线批处理,一次性跑完所有请求
        OnlineEngine  —— 在线调度,请求异步到达,输出异步产出
        两者共享 StageEngine / Scheduler / KVCacheManager。
    """

    def __init__(
        self,
        stages: list[StageEngine],
        converters: list[Callable[[StageOutput], StageInput]],
        *,
        idle_sleep_s: float = 0.001,   # engine 线程空闲时的睡眠时长
    ) -> None: ...

    def start(self) -> None:
        """启动后台 engine 线程。幂等。"""

    async def shutdown(self, *, drain: bool = True) -> None:
        """
        停止 engine 线程。
            drain=True:  等 pending + in-flight 请求全部完成后再退出
            drain=False: 立即停,未完成请求收到 OmniChunk(type="error")
        """

    def submit(self, req: OmniRequest) -> AsyncIterator[OmniChunk]:
        """
        提交请求,**立即返回** async generator。
        调用者 `async for chunk in ...` 逐块消费。

        注意:这个函数本身是 sync 的,返回值才是 async iterator。
        这样可以避免调用者必须 `await engine.submit(...)`。
        """
```

### 4.3 两个接口取舍

- `submit` 返回 async iterator 而不是 "await 出一个 iterator":因为"入队"本身只是一次 thread-safe queue put,不需要 async,让 iterator 承担所有 await 更直观。
- `shutdown` 分 `drain=True/False`:方便作者在练习中观察"优雅 vs 强制"两种收尾路径。

---

## 5. engine 线程内部(核心 TODO 区)

### 5.1 请求句柄

```python
@dataclass
class _RequestHandle:
    """
    engine 线程用来追踪单个请求状态。
    由 submit() 在 asyncio 侧创建,通过 pending_q 传给 engine 线程。
    """
    request_id: str
    request: OmniRequest
    out_q: "asyncio.Queue[OmniChunk]"      # engine 线程 → 调用方
    loop: "asyncio.AbstractEventLoop"      # 绑定创建 submit 时的 event loop
    # A 级状态:记录 stage 0(Thinker)产出的文本,完成后随 audio 一起推
    text_so_far: str = ""
    # B 级扩展点:per-token text delta 发布时间戳、已发布位置等可加在这
    # C 级扩展点:若支持流式输入,这里还要放增量 prefill 的游标
```

### 5.2 engine_loop(作者要填的主体)

期望行为(A 级)由 docstring 写明:

```
while self._running:
    1) 把 pending_q 里所有新请求 drain 出来,
       preprocess 后 add_request 到 stages[0],
       并把 _RequestHandle 登记到 self._handles[req_id]。
    2) 依次驱动每个 stage 一次 step():
         outs = stages[i].step()
         对 outs 里每个 StageOutput:
           - 若 i < last: 用 converters[i] 转成 StageInput,
             add_request 到 stages[i+1]。
             特别:i == 0(Thinker)时,用 model.decode()
             把 token_ids 还原成文本,写到 handle.text_so_far。
           - 若 i == last: 调 self._publish(handle, OmniChunk(
                type="audio", audio=out.audio))
             再调 self._publish(handle, OmniChunk(
                type="text", text=handle.text_so_far))
             最后 OmniChunk(type="done"),并清理 handle。
    3) 若本轮 pending_q 为空 且 所有 stage has_unfinished()==False:
         sleep(self._idle_sleep_s),避免空转占 CPU。
    4) 捕获异常:对受影响的 handle 推 type="error" chunk,
       不要让整个 engine 线程挂掉。
```

B / C 级升级路径在 docstring 中说明:

- **B 级**:把 stage 0 的 step() 输出改成 per-token 流式——最小改动方案是在 `ARStageEngine` 增加一个 `on_token` 回调,`_engine_loop` 订阅该回调即可。跨 stage token 级流水则需要 engine_loop 不再严格按 stage 顺序驱动,改为轮询所有 stage(谁有活谁干),并引入背压与优先级。
- **C 级**:`OmniRequest` 改成可多次 `append_audio`,`stages[0]` 的 `Scheduler` 需支持"prefill 分段持续追加",`engine_loop` 每轮需 drain 新增音频分片、调用 `model.prepare_inputs_incremental()` 再 `add_request`。

### 5.3 _publish(边界 3)

```python
def _publish(self, handle: "_RequestHandle", chunk: OmniChunk) -> None:
    """
    把 chunk 从 engine 线程安全地推给对应请求的 out_q。

    要点:
        - engine 线程不能直接 out_q.put_nowait(...),因为 asyncio.Queue
          不是 thread-safe。必须用
              handle.loop.call_soon_threadsafe(handle.out_q.put_nowait, chunk)
          这样 put 发生在 out_q 所属的 event loop 线程里。
        - 若 handle.loop 已关闭(用户提前取消/退出):静默丢弃,不要抛。
        - type="done" / type="error" 之后建议再 call_soon_threadsafe 一次
          put_nowait(None) 作为迭代结束哨兵(见 submit 的 generator)。
    """
    raise NotImplementedError("TODO")
```

### 5.4 submit / start / shutdown(边界 1 + 生命周期)

`submit` 要点:

- `asyncio.get_running_loop()` 拿当前 event loop(因此必须在 coroutine 内调用)。
- 创建 `_RequestHandle`,内含新建的 `asyncio.Queue` 和 loop 引用。
- `self._pending_q.put(handle)`(thread-safe `queue.Queue`)。
- 返回一个 async generator:循环 `await out_q.get()`,碰到 `None` 哨兵就 return,否则 yield chunk。
- 不要在 submit 里 await 任何东西;它本身是同步函数,只返回 generator。

`start` 要点:

- `self._running = True`
- `threading.Thread(target=self._engine_loop, name="nanoOmni-engine", daemon=True).start()`
- 已启动时 idempotent return。

`shutdown` 要点:

- `drain=True`:设 `self._accepting = False`,submit 之后调用抛 `RuntimeError`;在 event loop 里以 `asyncio.sleep` 轮询等待 `self._handles` 与 `pending_q` 清空,不要 busy-wait。
- `drain=False`:`self._running = False`,engine 线程退出前对所有仍在 `self._handles` 里的请求 `_publish` 一条 `type="error"` 再放哨兵。
- 共通:`self._engine_thread.join(timeout=...)`。

### 5.5 刻意取舍

- **A 级下 Thinker 的 text 与 Code2Wav audio 一起发,不单独流式**:现有 `ARStageEngine.step()` 是"整段完成才 return StageOutput",per-token 文本流需要动 stage 层代码——这是 B 级入口,注释里已点明。
- **在 `_publish` 的注释中明确钉死 `asyncio.Queue` 非 thread-safe + `call_soon_threadsafe` 用法**:跨线程最容易踩的坑,用文字固化。

---

## 6. 文件布局

```
nano_omni/online/
├── __init__.py          # 导出 OnlineEngine, OmniChunk
├── types.py             # OmniChunk
└── engine.py            # _RequestHandle, OnlineEngine(全是 TODO + 注释)
```

现有文件**不改动**。

---

## 7. 测试

新增 `tests/test_online_engine.py`——只做**骨架自检**,不依赖真实模型:

- 用一个假 `StageEngine` 替身(每次 `step()` 返回固定 `StageOutput`),使得在 `_engine_loop` 等方法填充的任何阶段都能运行 `pytest -k online` 看到接口通没通,不需要加载模型,秒过。

覆盖:

- `submit` 返回 async iterator,`async for` 能拿到至少一个 `type="done"` chunk
- `start` 幂等
- `shutdown(drain=True)` 能优雅退出
- `shutdown(drain=False)` 对 in-flight 请求发 `type="error"`

测试里用 `# TODO:` 标出"这条断言现在会失败,等你实现 `_engine_loop` 后才过",作为实现进度的标尺。

### 示例脚本

`examples/run_online.py` **暂不提供**——等作者跑通 `_engine_loop` 后再写,否则是空壳。

---

## 8. 后续实现计划

作者确认本 spec 后,通过 `writing-plans` 技能把 spec 拆成可执行的实现计划。计划会分步产出:

1. `nano_omni/online/types.py`(`OmniChunk` 完整实现,非 TODO)
2. `nano_omni/online/engine.py`(类型签名 + docstring + `raise NotImplementedError`)
3. `nano_omni/online/__init__.py`
4. `tests/test_online_engine.py`(骨架自检 + 标注 TODO 断言)

之后作者按自己的节奏把 `_engine_loop` / `_publish` / `submit` / `start` / `shutdown` 逐个填充,测试作为进度标尺。
