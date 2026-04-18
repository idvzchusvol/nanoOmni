from __future__ import annotations

import torch

from nano_omni.models.qwen_omni.code2wav import Code2Wav
from nano_omni.models.qwen_omni.talker import Talker
from nano_omni.models.qwen_omni.thinker import Thinker


def load_qwen3_omni(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Thinker, Talker, Code2Wav, object]:
    """
    Load Qwen3-Omni once and split into Thinker / Talker / Code2Wav wrappers.

    Loads the full Qwen3OmniMoeForConditionalGeneration model a single time,
    then extracts the three sub-modules. This avoids loading the 30B parameters
    three times when constructing each wrapper via from_pretrained().
    """
    from transformers import AutoProcessor
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration

    print(f"[nanoOmni] Loading processor from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"[nanoOmni] Loading full Qwen3-Omni model (dtype={dtype}, device={device})")
    full_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device,
    )
    full_model.eval()

    thinker = Thinker(model=full_model.thinker, processor=processor)
    talker = Talker(model=full_model.talker)
    code2wav = Code2Wav(model=full_model.code2wav)

    return thinker, talker, code2wav, full_model


def load_qwen25_omni(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Thinker, Talker, Code2Wav, object]:
    # Load Qwen2.5-Omni model and split into Thinker / Talker / Code2Wav wrappers.
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

    print(f"[nanoOmni] Loading processor from {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"[nanoOmni] Loading full Qwen2.5-Omni model (dtype={dtype}, device={device})")
    full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device,
    )
    full_model.eval()

    thinker = Thinker(model=full_model.thinker, processor=processor)
    talker = Talker(model=full_model.talker)
    # Qwen2.5-Omni names the codec-to-wav module `token2wav`
    # Qwen2.5 token2wav requires fp32 inference.
    full_model.token2wav.float()
    code2wav = Code2Wav(model=full_model.token2wav)

    return thinker, talker, code2wav, full_model


_LOADERS = {
    "qwen3_omni": load_qwen3_omni,
    "qwen25_omni": load_qwen25_omni,
}

def load_omni_model(
    model_family: str,
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Thinker, Talker, Code2Wav, object]:
    """
    Dispatch to specifical loader based on model_family.

    Args:
        model_family: (e.g., "qwen3_omni" or "qwen25_omni")
        model_path:   local path or HF repo id
        device:       target device
        dtype:        model weight dtype
    """
    loader = _LOADERS.get(model_family)
    if loader is None:
        raise ValueError(
            f"Unknown model_family '{model_family}'. "
            f"Supported: {list(_LOADERS)}"
        )
    return loader(model_path, device=device, dtype=dtype)
