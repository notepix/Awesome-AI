"""Run actual CLI stages and record results; never downloads datasets or models."""
from __future__ import annotations
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent.parent
    commands = [
        ["minigpt", "prepare"], ["minigpt", "train"], ["minigpt", "eval"],
        ["minigpt", "generate", "--prompt", "The cat"], ["minigpt", "train", "--resume", "--steps", "20"], ["minigpt", "eval"],
        ["posttrain", "prepare"]]
    for method in ("sft", "lora", "dpo"):
        commands += [["posttrain", stage, "--method", method] for stage in ("train", "eval", "demo")]
    commands += [["rag", stage] for stage in ("prepare", "index", "eval", "demo")]
    commands += [["rag", "demo", "--prompt", "Who discovered planet Neptune?"]]
    commands += [["agent", stage] for stage in ("prepare", "index", "eval", "demo")]
    commands += [["multimodal", stage] for stage in ("prepare", "train", "index", "eval", "demo")]
    packages = {}
    for name in ("numpy", "torch", "scikit-learn", "Pillow", "scipy", "pytest", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    report = {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": platform.python_version(),
              "platform": platform.platform(), "packages": packages, "validation_type": "offline_algorithm", "runs": []}
    output = root / "labs/outputs/validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    for i, command in enumerate(commands, 1):
        start = time.perf_counter()
        result = subprocess.run([sys.executable, "-m", "labs.run", *command], cwd=root, capture_output=True, text=True,
                                encoding="utf-8", timeout=240)
        entry = {"command": "python -m labs.run " + " ".join(command), "exit_code": result.returncode,
                 "seconds": round(time.perf_counter() - start, 3)}
        if result.returncode == 0:
            entry["result"] = json.loads(result.stdout)
        else:
            entry["error"] = result.stderr
        report["runs"].append(entry)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{i}/{len(commands)}] exit={result.returncode}, {entry['seconds']}s: {' '.join(command)}", flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
            raise SystemExit(result.returncode)
    negative = subprocess.run([sys.executable, "-m", "labs.run", "posttrain", "demo", "--profile", "local-model", "--model-path", str(root / "labs/outputs/model-not-provided")],
                              cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=60)
    report["missing_model_handling"] = {"exit_code": negative.returncode, "message": negative.stderr,
        "behaved_as_expected": negative.returncode == 2 and "NOT VALIDATED" in negative.stderr,
        "does_not_validate_local_model": True}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recorded {len(commands)} actual CLI stages in {output}", flush=True)
    if not report["missing_model_handling"]["behaved_as_expected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
