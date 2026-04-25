# OnlineEngine 骨架实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 nanoOmni 落地 `OnlineEngine` 的**骨架**(A 级在线调度),提供类型、类签名、docstring、NotImplementedError stub 与骨架自检测试,供作者后期自行填充调度逻辑。

**Architecture:** 新增 `nano_omni/online/` 子包,`OnlineEngine` 与现有 `Pipeline` 并列(均包装同一套 `StageEngine`),对外提供 `submit() → AsyncIterator[OmniChunk]` 流式接口 + 后台 engine 线程。具体 `_engine_loop` / `_publish` / `submit` / `start` / `shutdown` 全部 `raise NotImplementedError`,在 docstring 中写明期望行为与 B/C 级扩展点。

**Tech Stack:** Python 3.9+ / asyncio / threading.Thread / queue.Queue / pytest(同步 `asyncio.run()`,不引入 pytest-asyncio)。

**Source spec:** `docs/superpowers/specs/2026-04-22-online-engine-design.md`

---

## 文件结构

**新增:**
- `nano_omni/online/__init__.py` — 导出 `OnlineEngine`, `OmniChunk`
- `nano_omni/online/types.py` — `OmniChunk`(完整实现)
- `nano_omni/online/engine.py` — `_RequestHandle`(完整)+ `OnlineEngine`(skeleton)
- `tests/test_online_types.py` — `OmniChunk` 单测
- `tests/test_online_engine.py` — 假 Stage 驱动的骨架自检 + skip-gated 进度测试

**不改动:** `nano_omni/pipeline.py`、`nano_omni/stage/*.py`、`nano_omni/scheduler/*.py`、`nano_omni/types.py`、`pyproject.toml`。

---

## Task 1:OmniChunk 类型

**Files:**
- Create: `nano_omni/online/types.py`
- Create: `tests/test_online_types.py`

- [ ] **Step 1: 写失败测试**

Write `tests/test_online_types.py`:

```python
# tests/test_online_types.py
import numpy as np

from nano_omni.online.types import OmniChunk


def test_omnichunk_done_minimal():
    c = OmniChunk(request_id="r1", type="done")
    assert c.request_id == "r1"
    assert c.type == "done"
    assert c.text is None
    assert c.audio is None
    assert c.error is None


def test_omnichunk_text():
    c = OmniChunk(request_id="r1", type="text", text="hi")
    assert c.text == "hi"


def test_omnichunk_audio():
    buf = np.zeros(100, dtype=np.float32)
    c = OmniChunk(request_id="r1", type="audio", audio=buf)
    assert c.audio is buf


def test_omnichunk_error():
    c = OmniChunk(request_id="r1", type="error", error="boom")
    assert c.error == "boom"
```

- [ ] **Step 2: 跑一次确认失败**

Run: `uv run pytest tests/test_online_types.py -v`
Expected: 4 tests FAIL with `ModuleNotFoundError: No module named 'nano_omni.online'`.

- [ ] **Step 3: 实现 OmniChunk**

First create the package dir marker so the import works. Since `__init__.py` is built properly in Task 3, write a **temporary empty** one here that Task 3 will overwrite.

Create `nano_omni/online/__init__.py` with just a single blank line (to mark it as a package). Task 3 will populate it.

Create `nano_omni/online/types.py`:

```python
# nano_omni/online/types.py
from __future__ import annotations

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
        - 出错时发一次 type="error"

    B 级扩展点:
        - text 可改成 per-token delta,Thinker 每 sample 一个 token 推一次
        - audio 的 chunk 粒度由 Code2Wav 的 chunked decode 决定
    """
    request_id: str
    type: Literal["text", "audio", "done", "error"]
    text: Optional[str] = None
    audio: Optional[np.ndarray] = None   # float32, 24kHz
    error: Optional[str] = None
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_online_types.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add nano_omni/online/__init__.py nano_omni/online/types.py tests/test_online_types.py
git commit -m "$(cat <<'EOF'
feat(online): add OmniChunk streaming output type

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2:OnlineEngine 骨架

**Files:**
- Create: `nano_omni/online/engine.py`
- Create: `tests/test_online_engine.py`(这一步只放最小 smoke 测试,Task 3 扩展)

- [ ] **Step 1: 写 smoke 测试**

Create `tests/test_online_engine.py`:

```python
# tests/test_online_engine.py
"""
OnlineEngine 骨架自检测试。

不依赖真实模型:用 _FakeStage 驱动,验证接口形状与异步语义。
当前只有 smoke 测试通过;随着你实现 _engine_loop / _publish / submit /
shutdown,解除对应 test 的 @pytest.mark.skip 即可一步步看到进度。
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from nano_omni.online import OmniChunk, OnlineEngine


def test_imports():
    assert OmniChunk is not None
    assert OnlineEngine is not None
```

- [ ] **Step 2: 跑一次确认失败**

Run: `uv run pytest tests/test_online_engine.py -v`
Expected: FAIL with `ImportError: cannot import name 'OnlineEngine' from 'nano_omni.online'`(Task 1 留下的 `__init__.py` 是空的)。

- [ ] **Step 3: 实现 engine.py 骨架**

Create `nano_omni/online/engine.py`:

```python
# nano_omni/online/engine.py
from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

from nano_omni.online.types import OmniChunk
from nano_omni.stage.base import StageEngine
from nano_omni.types import OmniRequest, StageInput, StageOutput


@dataclass
class _RequestHandle:
    """
    engine 线程用来追踪单个请求状态。
    由 submit() 在 asyncio 侧创建,通过 pending_q 传给 engine 线程。
    """
    request_id: str
    request: OmniRequest
    out_q: "asyncio.Queue[Optional[OmniChunk]]"   # engine 线程 → 调用方
    loop: "asyncio.AbstractEventLoop"             # 绑定 submit 所在 event loop
    # A 级状态:记录 stage 0(Thinker)产出的文本,完成后随 audio 一起推。
    text_so_far: str = ""
    # B 级扩展点:per-token text delta 发布时间戳、已发布位置等可加在这里。
    # C 级扩展点:若支持流式输入,这里还要放增量 prefill 的游标。


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
        idle_sleep_s: float = 0.001,
    ) -> None:
        assert len(converters) == len(stages) - 1, (
            f"Need {len(stages)-1} converters, got {len(converters)}"
        )
        self.stages = stages
        self.converters = converters
        self._idle_sleep_s = idle_sleep_s

        stage0_model = getattr(stages[0], "model", None)
        if stage0_model is None or not hasattr(stage0_model, "prepare_inputs"):
            raise TypeError(
                "stages[0].model must implement prepare_inputs()."
            )
        self._preprocess = stage0_model.prepare_inputs
        self._text_decoder = (
            stage0_model.decode if hasattr(stage0_model, "decode") else None
        )

        # ---- 状态,供你在 TODO 方法中读写 ----
        self._pending_q: "queue.Queue[_RequestHandle]" = queue.Queue()
        self._handles: dict[str, _RequestHandle] = {}
        self._running: bool = False
        self._accepting: bool = True
        self._engine_thread: Optional[threading.Thread] = None

    # ====================================================================
    # 公共 API:start / submit / shutdown
    # 三者都是 TODO —— 这是边界 1(asyncio 世界 → engine 线程)+ 生命周期。
    # ====================================================================

    def start(self) -> None:
        """启动后台 engine 线程。幂等。

        要点(留给你实现):
            - 若 self._engine_thread 已存在且 is_alive():直接 return(幂等)
            - self._running = True
            - self._engine_thread = threading.Thread(
                  target=self._engine_loop,
                  name="nanoOmni-engine",
                  daemon=True,
              )
            - self._engine_thread.start()
        """
        raise NotImplementedError("TODO: 启动 engine 线程")

    async def shutdown(self, *, drain: bool = True) -> None:
        """停止 engine 线程。

            drain=True:  等 pending + in-flight 请求全部完成后再退出
            drain=False: 立即停,未完成请求收到 OmniChunk(type="error")

        要点(留给你实现):
            drain=True:
                - self._accepting = False,submit 后续调用应抛 RuntimeError
                - 在 event loop 里用 `await asyncio.sleep(...)` 轮询等待
                  self._handles 与 self._pending_q 清空,不要在 loop 里 busy-wait
                - 最后 self._running = False
            drain=False:
                - self._running = False
                - engine 线程退出前,对所有仍在 self._handles 的请求
                  _publish 一条 type="error" 再放哨兵(None)
            共通:
                - self._engine_thread.join(timeout=...) 回收线程
        """
        raise NotImplementedError("TODO: 收尾")

    def submit(self, req: OmniRequest) -> AsyncIterator[OmniChunk]:
        """提交请求,**立即返回** async generator。

        调用者:
            async for chunk in engine.submit(req):
                ...

        注意:这个函数本身是 sync 的,返回值才是 async iterator。
        这样可以避免调用者必须 `await engine.submit(...)`。

        要点(留给你实现):
            - asyncio.get_running_loop() 拿当前 event loop
              (因此必须在 coroutine 内调用)
            - 若 not self._accepting: raise RuntimeError("engine shutting down")
            - 创建 _RequestHandle,含新建的 asyncio.Queue 和 loop 引用
            - self._pending_q.put(handle)(thread-safe queue.Queue)
            - 返回一个 async generator:
                  async def _gen():
                      while True:
                          item = await handle.out_q.get()
                          if item is None:      # _publish 放的哨兵
                              return
                          yield item
                  return _gen()
            - 不要在 submit 里 await 任何东西;它本身是同步函数,只返回 generator
        """
        raise NotImplementedError("TODO: 入口 + 异步生成器")

    # ====================================================================
    # engine 线程主体:_engine_loop —— 这就是"调度"本身,练习主战场。
    # ====================================================================

    def _engine_loop(self) -> None:
        """后台线程入口。整个"在线调度"语义都在这里。

        期望行为(A 级):
            while self._running:
                # 1) drain pending_q 里所有新请求:
                #      handle = self._pending_q.get_nowait() (用 queue.Empty 兜底)
                #      self._handles[handle.request_id] = handle
                #      stage_input = self._preprocess(handle.request)
                #      self.stages[0].add_request(stage_input)
                #
                # 2) 依次驱动每个 stage 一次 step():
                #      outs = self.stages[i].step()
                #      对 outs 里每个 StageOutput:
                #        - 若 i < last(非终点 stage):
                #            next_inp = self.converters[i](out)
                #            self.stages[i + 1].add_request(next_inp)
                #            特别:i == 0(Thinker)时,
                #              handle.text_so_far = self._text_decoder(out.token_ids)
                #              用 handle = self._handles[out.request_id] 找句柄。
                #        - 若 i == last(终点 stage):
                #            handle = self._handles[out.request_id]
                #            if out.audio is not None:
                #                self._publish(handle, OmniChunk(
                #                    request_id=handle.request_id,
                #                    type="audio", audio=out.audio))
                #            self._publish(handle, OmniChunk(
                #                request_id=handle.request_id,
                #                type="text", text=handle.text_so_far))
                #            self._publish(handle, OmniChunk(
                #                request_id=handle.request_id, type="done"))
                #            del self._handles[handle.request_id]
                #
                # 3) 若本轮 pending_q 空 且 所有 stage has_unfinished() 都是 False:
                #      time.sleep(self._idle_sleep_s)  (engine 线程 ≠ event loop)
                #
                # 4) 用 try/except 包住 step()/convert(),捕获到异常时:
                #      拿到受影响的 handle,_publish(handle, OmniChunk(
                #          request_id=..., type="error", error=repr(exc)))
                #      清理 self._handles 里的那一项,不要让整个线程挂掉。

        B 级升级路径(保留为注释,方便后期改造):
            - 把 stage 0 的 step() 输出改成 per-token 流式:
              最小改动是在 ARStageEngine 加一个 on_token 回调,engine_loop 订阅它。
            - 跨 stage token 级流水:engine_loop 不再严格按 stage 顺序驱动,
              改为轮询所有 stage "谁有活谁干"。会引入:
                - 背压(Code2Wav 消费不过来时 Talker 要不要降速)
                - 优先级(首包 TTFB 敏感 vs 吞吐敏感的请求混跑)

        C 级升级路径:
            - OmniRequest 改成可多次 append_audio
            - stages[0] 的 Scheduler 需支持 "prefill 分段持续追加",
              而不是一次性把 token_ids 灌完就转到 decode。
            - engine_loop 多一个步骤:drain 请求上的新增音频分片,调用
              model.prepare_inputs_incremental() 再 add_request。
        """
        raise NotImplementedError("TODO: 练习主体")

    # ====================================================================
    # 边界 3:engine 线程 → asyncio 世界
    # ====================================================================

    def _publish(self, handle: _RequestHandle, chunk: OmniChunk) -> None:
        """把 chunk 从 engine 线程安全地推给对应请求的 out_q。

        要点(留给你实现):
            - engine 线程不能直接 out_q.put_nowait(...),因为 asyncio.Queue
              不是 thread-safe。必须用
                  handle.loop.call_soon_threadsafe(
                      handle.out_q.put_nowait, chunk
                  )
              这样 put 发生在 out_q 所属的 event loop 线程里。
            - 若 handle.loop 已关闭(用户提前取消 / 退出):静默吞掉
              RuntimeError,不要抛。
            - type="done" / type="error" 之后**再** call_soon_threadsafe 一次
                  handle.out_q.put_nowait(None)
              作为 async generator 的结束哨兵(见 submit 里的 _gen)。
        """
        raise NotImplementedError("TODO: 跨线程发布 chunk")
```

- [ ] **Step 4: 更新 `__init__.py` 导出**

Overwrite `nano_omni/online/__init__.py`:

```python
# nano_omni/online/__init__.py
from nano_omni.online.engine import OnlineEngine
from nano_omni.online.types import OmniChunk

__all__ = ["OnlineEngine", "OmniChunk"]
```

- [ ] **Step 5: 跑 smoke 测试确认通过**

Run: `uv run pytest tests/test_online_engine.py::test_imports -v`
Expected: 1 passed.

- [ ] **Step 6: 跑全部测试,确认没破坏现有行为**

Run: `uv run pytest -v`
Expected: 新增 5 个 passed(`test_online_types.py` 4 个 + `test_online_engine.py::test_imports` 1 个),旧的 `test_pipeline.py` 仍然全绿,无 import error。

- [ ] **Step 7: Commit**

```bash
git add nano_omni/online/__init__.py nano_omni/online/engine.py tests/test_online_engine.py
git commit -m "$(cat <<'EOF'
feat(online): add OnlineEngine skeleton with TODO docstrings

All scheduler-critical methods (_engine_loop, _publish, submit,
start, shutdown) raise NotImplementedError with detailed docstrings
describing expected A-level behavior plus B/C-level extension notes.
Intended as a nanoOmni scheduling-exercise scaffold.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3:骨架自检测试(假 Stage + skip-gated 进度标尺)

**Files:**
- Modify: `tests/test_online_engine.py`(从 smoke test 扩到完整骨架自检)

- [ ] **Step 1: 扩写测试文件**

Overwrite `tests/test_online_engine.py` with:

```python
# tests/test_online_engine.py
"""
OnlineEngine 骨架自检测试。

不依赖真实模型:用 _FakeStage 驱动,验证接口形状与异步语义。
当前只有 smoke 测试通过;随着你实现 _engine_loop / _publish / submit /
shutdown,解除对应 test 的 @pytest.mark.skip 即可一步步看到进度。

同步驱动 async 测试:用 asyncio.run(...) 而不是 pytest-asyncio,避免
引入新依赖(dev deps 里现在只有 pytest + pytest-mock)。
"""
from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import pytest

from nano_omni.online import OmniChunk, OnlineEngine
from nano_omni.stage.base import StageEngine
from nano_omni.types import (
    OmniRequest, StageConfig, StageInput, StageOutput,
)


# ---------- 假 Stage 替身 ----------

class _FakeModel:
    """stages[0].model 必须实现 prepare_inputs / decode(OnlineEngine 对 stage0 的契约)。"""
    def prepare_inputs(self, req: OmniRequest) -> StageInput:
        return StageInput(request_id=req.request_id, token_ids=[1, 2, 3])

    def decode(self, token_ids: list[int]) -> str:
        return "fake-text:" + ",".join(str(t) for t in token_ids)


class _FakeStage(StageEngine):
    """
    每个 add_request 后,下一次 step() 即把该请求标记已完成返回。
    has_audio=True 时填充 audio(模拟 Code2Wav 终点 stage)。
    """
    def __init__(self, has_audio: bool = False):
        super().__init__(StageConfig(name="fake", stage_type="ar"))
        self.model = _FakeModel()
        self._pending: list[StageInput] = []
        self._has_audio = has_audio

    def add_request(self, inp: StageInput) -> None:
        self._pending.append(inp)

    def step(self) -> list[StageOutput]:
        outs: list[StageOutput] = []
        while self._pending:
            inp = self._pending.pop(0)
            outs.append(StageOutput(
                request_id=inp.request_id,
                token_ids=[4, 5, 6],
                audio=(np.zeros(100, dtype=np.float32) if self._has_audio else None),
                is_finished=True,
                finish_reason="stop_token",
            ))
        return outs

    def has_unfinished(self) -> bool:
        return bool(self._pending)


def _passthrough_converter(out: StageOutput) -> StageInput:
    return StageInput(request_id=out.request_id, token_ids=out.token_ids)


@pytest.fixture
def engine() -> OnlineEngine:
    return OnlineEngine(
        stages=[_FakeStage(has_audio=False), _FakeStage(has_audio=True)],
        converters=[_passthrough_converter],
    )


# ==========================================================================
# 基础:可导入、类型可构造(不依赖任何 TODO 方法的实现)
# ==========================================================================

def test_imports():
    assert OmniChunk is not None
    assert OnlineEngine is not None


def test_engine_init(engine: OnlineEngine):
    assert len(engine.stages) == 2
    assert len(engine.converters) == 1
    assert engine._running is False
    assert engine._engine_thread is None


def test_engine_rejects_stage0_without_model():
    from unittest.mock import MagicMock
    stage = MagicMock(spec=StageEngine)
    with pytest.raises(TypeError, match="prepare_inputs"):
        OnlineEngine(stages=[stage], converters=[])


def test_engine_converter_count_mismatch():
    with pytest.raises(AssertionError):
        OnlineEngine(
            stages=[_FakeStage(), _FakeStage(has_audio=True)],
            converters=[],   # should be 1 converter
        )


# ==========================================================================
# 进度标尺:随实现进度 **逐条删除 @pytest.mark.skip** 即可看到测试转绿。
# 顺序大致对应推荐的实现顺序:start → submit+engine_loop+_publish → shutdown。
# ==========================================================================

@pytest.mark.skip(reason="TODO: 等 start() 实现后删除本装饰器")
def test_start_is_idempotent(engine: OnlineEngine):
    engine.start()
    t1 = engine._engine_thread
    engine.start()   # 第二次调用不应抛
    t2 = engine._engine_thread
    assert t1 is t2                  # 同一个线程,未重新创建
    assert t1 is not None
    assert t1.is_alive()
    # 清理:停掉,不走 shutdown(shutdown 还没实现)
    engine._running = False
    t1.join(timeout=1.0)


@pytest.mark.skip(reason="TODO: 等 submit + _engine_loop + _publish 全部实现后删除")
def test_submit_yields_done_chunk(engine: OnlineEngine):
    """端到端最基础:提交一个请求,能拿到 done chunk。"""
    async def _run():
        engine.start()
        try:
            req = OmniRequest(request_id="r1", text="hi")
            chunks: list[OmniChunk] = []
            async for chunk in engine.submit(req):
                chunks.append(chunk)
                if chunk.type == "done":
                    break
            return chunks
        finally:
            # drain=False 是因为 shutdown 可能还没实现完备;
            # 若你已实现 drain=True,改成 True 更优雅。
            await engine.shutdown(drain=False)

    chunks = asyncio.run(_run())
    types = [c.type for c in chunks]
    assert "done" in types
    # A 级:文本与音频都会被推出来
    assert "text" in types
    assert "audio" in types


@pytest.mark.skip(reason="TODO: 等 shutdown(drain=True) 实现后删除")
def test_shutdown_drain_true_graceful(engine: OnlineEngine):
    async def _run():
        engine.start()
        req = OmniRequest(request_id="r1", text="hi")
        async for chunk in engine.submit(req):
            if chunk.type == "done":
                break
        await engine.shutdown(drain=True)
        assert engine._running is False
        assert engine._engine_thread is not None
        assert not engine._engine_thread.is_alive()

    asyncio.run(_run())


@pytest.mark.skip(reason="TODO: 等 shutdown(drain=False) 实现后删除")
def test_shutdown_drain_false_emits_error(engine: OnlineEngine):
    """强制关闭时,未消费完的请求应收到 type='error' chunk。"""
    async def _run():
        engine.start()
        req = OmniRequest(request_id="r1", text="hi")
        agen = engine.submit(req)
        # 故意不消费;立即强关
        await engine.shutdown(drain=False)
        chunks: list[OmniChunk] = []
        async for chunk in agen:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_run())
    assert any(c.type == "error" for c in chunks)


@pytest.mark.skip(reason="TODO: 等 shutdown(drain=True) 实现后删除")
def test_submit_after_shutdown_raises(engine: OnlineEngine):
    async def _run():
        engine.start()
        await engine.shutdown(drain=True)
        with pytest.raises(RuntimeError):
            engine.submit(OmniRequest(request_id="r2", text="hi"))

    asyncio.run(_run())
```

- [ ] **Step 2: 跑测试,确认 basic 通过 + skip-gated 被标 skip**

Run: `uv run pytest tests/test_online_engine.py -v`
Expected:
- `test_imports` PASS
- `test_engine_init` PASS
- `test_engine_rejects_stage0_without_model` PASS
- `test_engine_converter_count_mismatch` PASS
- `test_start_is_idempotent` SKIPPED
- `test_submit_yields_done_chunk` SKIPPED
- `test_shutdown_drain_true_graceful` SKIPPED
- `test_shutdown_drain_false_emits_error` SKIPPED
- `test_submit_after_shutdown_raises` SKIPPED

汇总:**4 passed, 5 skipped**。

- [ ] **Step 3: 全量跑测试,确认不破坏现有测试**

Run: `uv run pytest -v`
Expected: 原有 `tests/test_pipeline.py` 全部 PASS;新增 `test_online_types.py` 4 PASS;`test_online_engine.py` 4 PASS + 5 SKIPPED。

- [ ] **Step 4: Commit**

```bash
git add tests/test_online_engine.py
git commit -m "$(cat <<'EOF'
test(online): add skeleton self-check with fake stage harness

4 basic tests pass immediately (imports, __init__ contract, guards);
5 skip-gated tests mark the progress ladder — unskip one at a time
as you fill in start/submit/_engine_loop/_publish/shutdown.

Uses asyncio.run() instead of pytest-asyncio to avoid adding a
dev dependency.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## 自检清单(plan 完成后人工验收)

按顺序逐条核对:

1. **`uv run pytest -v` 全绿 + 5 skipped**:原有测试未受影响,新增骨架通过自检。
2. **`nano_omni/online/` 目录存在**,含 `__init__.py` / `types.py` / `engine.py` 三个文件。
3. **`from nano_omni.online import OnlineEngine, OmniChunk` 可用**。
4. **`OnlineEngine(stages=[...], converters=[...])` 构造不抛**(当 stage0.model 有 prepare_inputs 时)。
5. **调用 `engine.start()` / `engine.submit(...)` / `await engine.shutdown()` 任一项都应抛 `NotImplementedError`**,消息里含"TODO"。
6. **`docs/superpowers/specs/2026-04-22-online-engine-design.md` 与实际代码一致**(任何设计调整要回写 spec)。
7. **Git 历史有 3 个新 commit**:`feat(online): add OmniChunk ...` / `feat(online): add OnlineEngine skeleton ...` / `test(online): add skeleton self-check ...`。

完成后,作者可以开始**按 skip 顺序逐个实现 TODO**,每填一个就删一条 `@pytest.mark.skip`,直到所有测试转绿,在线调度 A 级就跑通了。
