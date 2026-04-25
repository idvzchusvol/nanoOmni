#!/usr/bin/env python3
"""
nanoOmni end-to-end inference example.

Usage:
    python examples/run.py --config configs/qwen3_omni.yaml \
        --text "你好，请介绍一下你自己。"
    python examples/run.py --config configs/qwen25_omni.yaml \
        --text "你好，请介绍一下你自己。"
"""
import argparse

import soundfile as sf

from nano_omni.models import build_omni_pipeline
from nano_omni.models.qwen_omni import load_model_config
from nano_omni.types import OmniRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_omni.yaml")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[nanoOmni] Loading config: {args.config}")
    cfg = load_model_config(args.config)
    pipeline = build_omni_pipeline(cfg, device=args.device)

    request = OmniRequest(request_id="demo", text=args.text)
    print(f"[nanoOmni] Running inference: {args.text}")
    results, _metrics = pipeline.run(requests=[request])

    if not results:
        print("[nanoOmni] No output.")
        return

    out = results[0]
    print(f"[nanoOmni] Text: {out.text}")
    if out.audio is not None:
        sf.write(args.output, out.audio, samplerate=24000)
        print(f"[nanoOmni] Audio saved: {args.output}")
    else:
        print("[nanoOmni] No audio output.")


if __name__ == "__main__":
    main()
