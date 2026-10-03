from __future__ import annotations
from pathlib import Path
import torch
from torch.nn import functional as F
from .common import seed_all, device_for, outdir, read_json, write_json, record, local_causal, local_generate
from .minigpt import load_checkpoint, save_checkpoint, add_lora, generate


def examples():
    rows = []
    facts = [("cat", "red"), ("dog", "blue"), ("fox", "green"), ("bird", "yellow")]
    for animal, color in facts:
        for n in range(1, 13):
            rows.append({"id": f"{animal}-{n}", "prompt": f"Context: animal {animal} has {n} {color} balls. What color are its balls?",
                         "chosen": color + ".", "rejected": "purple.", "split": "validation" if n >= 10 else "train"})
    return rows


def encode_responses(tokenizer, row, responses, block_size, local=False):
    if local:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suffixes = {key: tokenizer.encode(row[key], add_special_tokens=False) + [tokenizer.eos_token_id] for key in responses}
    else:
        prefix = [tokenizer.bos] + tokenizer.encode("Question: " + row["prompt"] + "\nAnswer: ")
        suffixes = {key: tokenizer.encode(row[key]) + [tokenizer.eos] for key in responses}
    room = block_size - max(len(suffix) for suffix in suffixes.values())
    if room < 1:
        raise ValueError("Response does not fit block_size; increase context length")
    # All answers in a preference pair condition on exactly the same truncated x.
    # Reserve space for the longest answer, retaining each complete answer and EOS.
    prefix = prefix[-room:]
    if not prefix:
        raise ValueError("At least one prompt token is required")
    return {key: (prefix + suffix, [-100] * len(prefix) + suffix) for key, suffix in suffixes.items()}


def encode_example(tokenizer, row, response, block_size, local=False):
    return encode_responses(tokenizer, row, (response,), block_size, local)[response]


def encode_preference_pair(tokenizer, row, block_size, local=False):
    return encode_responses(tokenizer, row, ("chosen", "rejected"), block_size, local)


def pack(encoded, pad, device):
    length = max(len(x[0]) for x in encoded)
    ids = torch.tensor([x + [pad] * (length - len(x)) for x, _ in encoded], device=device)
    labels = torch.tensor([y + [-100] * (length - len(y)) for _, y in encoded], device=device)
    mask = torch.tensor([[1] * len(x) + [0] * (length - len(x)) for x, _ in encoded], device=device)
    return ids, labels, mask


def sequence_scores(model, batch, local=False):
    ids, labels, mask = batch
    logits = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1]).logits if local else model(ids[:, :-1])
    target = labels[:, 1:]
    valid = target != -100
    safe_target = target.masked_fill(~valid, 0)
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    return (logp * valid).sum(-1), valid.sum(-1)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=.1):
    return -F.logsigmoid(beta * ((chosen - rejected) - (ref_chosen - ref_rejected))).mean()


def adapter_state(model):
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name.endswith((".a", ".b"))}


def base_signature(model):
    fields = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size")
    return {key: getattr(model.config, key, None) for key in fields}


def load_adapter(model, path, expected_lora, expected_base_path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing SFT adapter {path}; run local-model train --method sft first")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("lora") != expected_lora:
        raise ValueError("Adapter rank/alpha differ from the current configuration")
    if Path(data.get("base_model_path", "")).resolve() != Path(expected_base_path).resolve():
        raise ValueError("Adapter base model directory differs; use the original explicitly prepared model")
    if data.get("base_config") != base_signature(model):
        raise ValueError("Adapter base model configuration differs")
    expected = set(adapter_state(model))
    if set(data["adapter"]) != expected:
        raise ValueError("Adapter layout is incompatible with this local model")
    model.load_state_dict(data["adapter"], strict=False)
    return data


def run(stage, cfg):
    seed_all(cfg.get("seed", 42))
    root, dev = outdir(cfg), device_for(cfg)
    method, local = cfg.get("method", "sft"), cfg.get("profile") == "local-model"
    kind = "local_model" if local else "offline_algorithm"
    if stage == "prepare":
        rows = examples()
        write_json(root / "data.json", {"rows": rows, "source": "Repository-authored factual prompts and intentionally incorrect preference responses"})
        return record(cfg, stage, {"train_pairs": sum(r["split"] == "train" for r in rows), "validation_pairs": sum(r["split"] == "validation" for r in rows)})
    rows = read_json(root / "data.json")["rows"]
    lora_config = {"rank": cfg.get("rank", 4), "alpha": cfg.get("alpha", 8)}
    output_file = root / (("local_" if local else "") + method + ".pt")
    if local:
        tok, model = local_causal(cfg)
        add_lora(model, local=True, **lora_config)
        if stage != "train":
            load_adapter(model, cfg.get("checkpoint") or output_file, lora_config, cfg["model_path"])
        elif cfg.get("checkpoint") or method == "dpo":
            load_adapter(model, cfg.get("checkpoint") or root / "local_sft.pt", lora_config, cfg["model_path"])
        model.config.use_cache = False
        block_size, pad = cfg.get("local_block_size", 128), tok.pad_token_id
    else:
        if stage == "train":
            source = cfg.get("checkpoint") or (str(root / "sft.pt") if method == "dpo" else cfg["pretrain_checkpoint"])
        else:
            source = cfg.get("checkpoint") or str(output_file)
        model, tok, saved = load_checkpoint(source, dev)
        if stage == "train" and method == "lora" and not saved.get("lora"):
            add_lora(model, **lora_config)
        if saved.get("lora"):
            lora_config = saved["lora"]
        block_size, pad = model.spec["block_size"], tok.pad
    model.to(dev)
    batches = {}
    for split in ("train", "validation"):
        selected = [r for r in rows if r["split"] == split]
        pairs = [encode_preference_pair(tok, row, block_size, local) for row in selected]
        batches[split] = {key: [pair[key] for pair in pairs] for key in ("chosen", "rejected")}
    batch_size = cfg.get("local_batch_size", 1) if local else cfg.get("batch_size", 8)

    def scores(data):
        values, lengths = [], []
        for i in range(0, len(data), batch_size):
            score, length = sequence_scores(model, pack(data[i:i + batch_size], pad, dev), local)
            values.append(score)
            lengths.append(length)
        return torch.cat(values), torch.cat(lengths)

    if stage == "train":
        train = batches["train"]
        reference = None
        if method == "dpo":
            # Cache the initial SFT policy once, in eval mode, then never recompute it.
            # This is a frozen reference without holding a second 0.5B model in VRAM.
            model.eval()
            with torch.no_grad():
                reference = [scores(train[key])[0].detach() for key in ("chosen", "rejected")]
        parameters = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(parameters, lr=cfg.get("local_lr", .0002) if local else cfg.get("lr", .001))
        model.train(method != "dpo")  # DPO policy/reference both disable dropout; gradients remain enabled.
        losses = []
        for step in range(cfg.get("steps", 60)):
            indices = torch.randint(len(train["chosen"]), (batch_size,)).tolist()
            chosen, counts = sequence_scores(model, pack([train["chosen"][i] for i in indices], pad, dev), local)
            if method == "dpo":
                rejected, _ = sequence_scores(model, pack([train["rejected"][i] for i in indices], pad, dev), local)
                loss = dpo_loss(chosen, rejected, reference[0][indices], reference[1][indices], cfg.get("beta", .1))
            else:
                loss = -chosen.sum() / counts.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            opt.step()
            losses.append(loss.item())
        if local:
            torch.save({"adapter": adapter_state(model), "lora": lora_config, "base_model_path": cfg["model_path"], "base_config": base_signature(model), "method": method}, output_file)
        else:
            has_lora = any(name.endswith(".a") for name, _ in model.named_parameters())
            save_checkpoint(output_file, model, tok, opt, len(losses), lora_config if has_lora else None)
        return record(cfg, stage, {"objective": "dpo" if method == "dpo" else "response_only_sft",
            "update": "lora" if local or method == "lora" else "full_tiny_model", "reference_frozen": method == "dpo",
            "initial_loss": losses[0], "final_loss": losses[-1], "trainable_parameters": sum(p.numel() for p in parameters), "checkpoint": str(output_file)}, kind)
    if stage == "eval":
        model.eval()
        with torch.no_grad():
            c, n = scores(batches["validation"]["chosen"])
            r, _ = scores(batches["validation"]["rejected"])
        return record(cfg, stage, {"response_token_nll": (-c.sum() / n.sum()).item(), "synthetic_preference_accuracy": (c > r).float().mean().item(), "mean_sequence_logprob_margin": (c - r).mean().item(), "interpretation": "Held-out synthetic examples only; not evidence of general alignment"}, kind)
    if stage in ("demo", "generate"):
        prompt = cfg.get("prompt", "Context: animal cat has 2 red balls. What color are its balls?")
        text = local_generate(tok, model, prompt) if local else generate(model, tok, "Question: " + prompt + "\nAnswer: ", 30)
        return record(cfg, stage, {"prompt": prompt, "response": text}, kind)
    raise ValueError(f"Unsupported posttrain stage: {stage}")
