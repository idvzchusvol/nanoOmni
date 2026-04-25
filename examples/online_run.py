import argparse

import soundfile as sf

from nano_omni.models import build_omni_online_engine
from nano_omni.models.qwen_omni import load_model_config
from nano_omni.online import OmniChunk, OnlineEngine
import asyncio
from nano_omni.types import OmniRequest


async def process_req(engine: OnlineEngine, req: OmniRequest):
    text_final: str = ""
    audio = None
    
    async for chunk in engine.submit(req):
      if chunk.type == "text":
        if chunk.text is not None:
          text_final = text_final + chunk.text
          print(f"{chunk.text}")
      elif chunk.type == "audio":
        audio = chunk.audio
      elif chunk.type == "error":
        print(f"[error] {chunk.error}")
      elif chunk.type == "done":
        break
      
    return text_final, audio

async def _amain(args):
    cfg = load_model_config(args.config)     
    engine = build_omni_online_engine(cfg, device=args.device)
    engine.start()
                                                                           
    try:
        reqs = [
            OmniRequest(request_id=f"demo-{i}", text=args.text)
            for i in range(args.n)]       
        results = await asyncio.gather(
            *(process_req(engine, req) for req in reqs),
            return_exceptions=True,)
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                print(f"[demo-{i}] failed: {res!r}")    
                continue
            text, audio = res             
            print(f"[demo-{i}] final text: {text}")
            if audio is not None:         
                out_path = args.output.replace(".wav", f"_{i}.wav")    
                sf.write(out_path, audio, samplerate=24000)
                print(f"[demo-{i}] audio saved: {out_path}")             
    finally:    
          await engine.shutdown(drain=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_omni.yaml")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default="1", help="concurrent requests")
    args = parser.parse_args()
    
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()