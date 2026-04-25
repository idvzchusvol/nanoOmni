# tests/test_online_engine.py
from __future__ import annotations

import asyncio

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



def test_submit_after_shutdown_raises(engine: OnlineEngine):
    async def _run():
        engine.start()
        await engine.shutdown(drain=True)
        with pytest.raises(RuntimeError):
            engine.submit(OmniRequest(request_id="r2", text="hi"))

    asyncio.run(_run())
