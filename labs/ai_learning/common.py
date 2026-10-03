from __future__ import annotations
import json
import random
import time
from pathlib import Path
import numpy as np
import torch


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(cfg):
    requested = cfg.get("device", "cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose device=cpu or install the matching PyTorch build")
    return torch.device(requested)


def outdir(cfg):
    p = Path(cfg["output"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}; run the documented prepare/train stage first")
    return json.loads(p.read_text(encoding="utf-8-sig"))


def record(cfg, stage, metrics, validation_type="offline_algorithm"):
    result = {"project": cfg["project"], "stage": stage, "validation_type": validation_type,
              "status": "completed", "seed": cfg.get("seed", 42), "config": cfg,
              "metrics": metrics, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(outdir(cfg) / f"{stage}_{cfg.get('method', 'default')}_{cfg.get('profile', 'offline-core')}.json", result)
    return result


def require_local(path, label="model"):
    if not path or not Path(path).is_dir():
        raise FileNotFoundError(f"Prepare an existing local {label} directory and pass its path; automatic downloads are disabled")
    return str(Path(path).resolve())


def local_causal(cfg):
    path = require_local(cfg.get("model_path"), "Qwen2.5-0.5B-Instruct")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = device_for(cfg)
    dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                               torch_dtype=dtype).to(dev)
    model.config.use_cache = False
    return tokenizer, model


@torch.no_grad()
def local_generate(tokenizer, model, prompt, max_new_tokens=80, max_time=None):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=768).to(next(model.parameters()).device)
    model.eval()
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, "pad_token_id": tokenizer.pad_token_id}
    if max_time is not None:
        kwargs["max_time"] = max(0.01, max_time)
    output = model.generate(**inputs, **kwargs)
    return tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
