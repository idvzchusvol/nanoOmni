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
