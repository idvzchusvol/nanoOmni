# nano_omni/online/engine.py
from __future__ import annotations

import asyncio
import queue, time
import threading
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

from nano_omni.online.types import OmniChunk
from nano_omni.stage.base import StageEngine
from nano_omni.types import OmniRequest, StageInput, StageOutput


@dataclass
class _RequestHandle:
    # Track request status for engine thread
    request_id: str
    request: OmniRequest
    out_q: asyncio.Queue[Optional[OmniChunk]] 
    loop: asyncio.AbstractEventLoop             # Bind event loop of submiter
    # A 级状态:记录 stage 0(Thinker)产出的文本,完成后随 audio 一起推。
    text_so_far: str = ""
    # B 级扩展点:per-token text delta 发布时间戳、已发布位置等可加在这里。
    # C 级扩展点:若支持流式输入,这里还要放增量 prefill 的游标。


class OnlineEngine:
    """
    Online Engine for Scheduling Requests
    Usage:
        engine = OnlineEngine(stages=[...], converters=[...])
        engine.start()                  
        async for chunk in engine.submit(req):
            ...
        await engine.shutdown()
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

        self._pending_q: "queue.Queue[_RequestHandle]" = queue.Queue()
        self._handles: dict[str, _RequestHandle] = {}
        self._running: bool = False
        self._accepting: bool = True
        self._engine_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running is True:
            assert(self._engine_thread is not None and self._engine_thread.is_alive())
            return 
        self._engine_thread = threading.Thread(
            target=self._engine_loop,
            name="nanoOmni-onlineEngine",
            daemon=True,
        )
        self._running = True
        self._engine_thread.start()


    async def shutdown(self, *, drain: bool = True) -> None:
        if drain is True:
            self._accepting = False
            while self._handles or not self._pending_q.empty():
                await asyncio.sleep(self._idle_sleep_s)
           
        self._running = False
        
        wait_t = 5
        while self._engine_thread and self._engine_thread.is_alive():
            self._engine_thread.join(timeout = wait_t)
            if self._engine_thread.is_alive():
                print(f"engine thread cannot exits with {wait_t}s")
        print(f"engine shutdown successfully")

    def submit(self, req: OmniRequest) -> AsyncIterator[OmniChunk]:
        cur_loop = asyncio.get_running_loop()
        if not self._accepting:
            raise RuntimeError("engine shutting down")
        out_q = asyncio.Queue()
        handle = _RequestHandle(
            request_id=req.request_id,
            request=req,
            out_q=out_q,
            loop=cur_loop
        )
        self._pending_q.put(handle)
        
        async def _gen():
            while True:
                item = await handle.out_q.get()
                if item is None:
                    return
                yield item
        
        return _gen()
    
    def _consume_pending_req(self) -> bool:
        Processed: bool = False
        # Consume all req and push into stage[0]
        while True:    
            try:
                handle = self._pending_q.get_nowait()
                Processed = True
                self._handles[handle.request_id] = handle
                stage_input = self._preprocess(handle.request)
                self.stages[0].add_request(stage_input)
            except queue.Empty:
                break
        return Processed
    
    
    def _publish_exit(self) -> None:
        while True:    
            try:
                handle = self._pending_q.get_nowait()
                self._handles[handle.request_id] = handle
            except queue.Empty:
                break
        
        for req_id, handle in list(self._handles.items()):
            self._publish(handle, OmniChunk(req_id, type="error", error=f"engine exits"))
            self._publish(handle, OmniChunk(req_id, type="done"))
            del self._handles[req_id]

    def _engine_loop(self) -> None:
        while self._running:
            Processed: bool = self._consume_pending_req()
            
            for i, stage in enumerate(self.stages):
                try:
                    outs = stage.step()
                except Exception as e:
                    for req_id, handle in list(self._handles.items()):
                        self._publish(handle, OmniChunk(request_id=req_id, type='error', error=f"stage[{i}] step failed: {e!r}"))
                    continue
                
                for out in outs:
                    Processed = True
                    try:
                        handle = self._handles[out.request_id]
                        if i < len(self.stages) - 1:
                            converted_out = self.converters[i](out)
                            self.stages[i + 1].add_request(converted_out)
                            if i == 0 and self._text_decoder is not None:
                                handle.text_so_far = self._text_decoder(out.token_ids)
                        elif i == len(self.stages) - 1:
                            if out.audio is not None:
                                self._publish(handle, OmniChunk(handle.request_id, type="audio", audio=out.audio))
                            self._publish(handle, OmniChunk(handle.request_id, type="text", text=handle.text_so_far))
                            self._publish(handle, OmniChunk(handle.request_id, type="done"))
                            del self._handles[out.request_id]
                    except Exception as e:
                        self._publish(handle, OmniChunk(handle.request_id, type="error", error=f"stage[{i}] processed failed: {e!r}"))
                        del self._handles[out.request_id]
            
            if not Processed:
                time.sleep(self._idle_sleep_s)
        
        self._publish_exit()
            
        """
        Future expansion:
            - 把 stage 0 的 step() 输出改成 per-token 流式:
              最小改动是在 ARStageEngine 加一个 on_token 回调,engine_loop 订阅它。
            - 跨 stage token 级流水:engine_loop 不再严格按 stage 顺序驱动,
              改为轮询所有 stage "谁有活谁干"。会引入:
                - 背压(Code2Wav 消费不过来时 Talker 要不要降速)
                - 优先级(首包 TTFB 敏感 vs 吞吐敏感的请求混跑)

        Future expansion:
            - OmniRequest 改成可多次 append_audio
            - stages[0] 的 Scheduler 需支持 "prefill 分段持续追加",
              而不是一次性把 token_ids 灌完就转到 decode。
            - engine_loop 多一个步骤:drain 请求上的新增音频分片,调用
              model.prepare_inputs_incremental() 再 add_request。
        """

    def _publish(self, handle: _RequestHandle, chunk: OmniChunk) -> None:
        try:
            handle.loop.call_soon_threadsafe(handle.out_q.put_nowait, chunk)
            if chunk.type in ["done", "error"]:
                handle.loop.call_soon_threadsafe(handle.out_q.put_nowait, None)
        except:
            pass
