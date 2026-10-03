from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Offline-first AI teaching laboratories")
    p.add_argument("project", choices=["minigpt", "posttrain", "rag", "agent", "multimodal"])
    p.add_argument("stage", choices=["prepare", "train", "index", "eval", "demo", "generate"])
    p.add_argument("--config")
    p.add_argument("--profile", choices=["offline-core", "local-model"])
    p.add_argument("--output")
    p.add_argument("--model-path", help="Existing local Qwen/CLIP directory; never downloaded")
    p.add_argument("--embedding-path", help="Existing local MiniLM directory")
    p.add_argument("--checkpoint")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--method", choices=["sft", "lora", "dpo"])
    p.add_argument("--prompt")
    p.add_argument("--steps", type=int)
    args = p.parse_args()
    config_path = Path(args.config or f"labs/{args.project}/config.json")
    cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
    for name in ("profile", "output", "model_path", "embedding_path", "checkpoint", "method", "prompt", "steps"):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value
    if cfg.get("steps", 1) < 1:
        p.error("steps must be a positive integer")
    cfg["resume"] = args.resume
    try:
        result = importlib.import_module(f"labs.ai_learning.{args.project}").run(args.stage, cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        p.exit(2, f"NOT VALIDATED: {exc}\n")


if __name__ == "__main__":
    main()
