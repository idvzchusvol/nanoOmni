from __future__ import annotations

import numpy as np
import torch


class Code2Wav:
    """
    Wraps Qwen3-Omni Code2Wav module (RVQ Codec decoder).

    Input:  codec codes tensor [batch, num_codebooks, T] (torch.long)
    Output: audio waveform np.ndarray [batch, num_samples] float32 at 24kHz
    """

    NUM_CODEBOOKS = 8

    def __init__(self, model):
        self.model = model

    def forward(self, codec_codes: torch.Tensor) -> np.ndarray:
        """
        Args:
            codec_codes: [batch, num_codebooks, T] torch.long
        Returns:
            audio: [batch, num_samples] np.float32 at 24kHz
        """
        with torch.no_grad():
            waveform = self.model(codec_codes)
        if isinstance(waveform, torch.Tensor):
            return waveform.cpu().float().numpy()
        return waveform

    def __call__(self, codec_codes: torch.Tensor) -> np.ndarray:
        return self.forward(codec_codes)
