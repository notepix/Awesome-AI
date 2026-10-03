from __future__ import annotations
from collections import Counter
import math
import torch
from torch import nn
from torch.nn import functional as F
from .common import seed_all, device_for, outdir, write_json, read_json, record


class ByteBPE:
    """Reversible byte BPE; merge training is intentionally simple and deterministic."""
    pad, bos, eos = 0, 1, 2

    def __init__(self, merges=None):
        self.merges = [tuple(x) for x in (merges or [])]
        self.pieces = {i + 3: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.pieces[259 + i] = self.pieces[a] + self.pieces[b]

    @property
    def vocab_size(self):
        return 259 + len(self.merges)

    @staticmethod
    def merge(seq, pair, token):
        result, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                result.append(token)
                i += 2
            else:
                result.append(seq[i])
                i += 1
        return result

    @classmethod
    def fit(cls, texts, num_merges=32):
        sequences = [[x + 3 for x in s.encode("utf-8")] for s in texts]
        merges = []
        for _ in range(num_merges):
            counts = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
            if not counts or counts.most_common(1)[0][1] < 2:
                break
            pair = counts.most_common(1)[0][0]
            sequences = [cls.merge(s, pair, 259 + len(merges)) for s in sequences]
            merges.append(pair)
        return cls(merges)

    def encode(self, text, special=False):
        ids = [b + 3 for b in text.encode("utf-8")]
        for i, pair in enumerate(self.merges):
            ids = self.merge(ids, pair, 259 + i)
        return [self.bos] + ids + [self.eos] if special else ids

    def decode(self, ids):
        return b"".join(self.pieces.get(int(i), b"") for i in ids).decode("utf-8", errors="replace")


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d // self.heads)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, d)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spec = dict(config)
        d = config["dim"]
        self.token = nn.Embedding(config["vocab_size"], d)
        self.position = nn.Embedding(config["block_size"], d)
        self.blocks = nn.Sequential(*[Block(d, config["heads"]) for _ in range(config["layers"])])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, config["vocab_size"], bias=False)

    def forward(self, ids):
        if ids.shape[1] > self.spec["block_size"]:
            raise ValueError("Sequence exceeds checkpoint block_size")
        x = self.token(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        return self.head(self.norm(self.blocks(x)))


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        for p in base.parameters():
            p.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype) * .02)
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def add_lora(model, rank=4, alpha=8, local=False):
    for p in model.parameters():
        p.requires_grad_(False)
    found = 0
    for name, child in list(model.named_modules()):
        if isinstance(child, nn.Linear) and (name.endswith(("q_proj", "v_proj")) if local else ".attn." in name):
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, LoRALinear(child, rank, alpha))
            found += 1
    if not found:
        raise ValueError("No supported attention linear layers found for LoRA")


def save_checkpoint(path, model, tokenizer, optimizer=None, step=0, lora=None):
    torch.save({"format_version": 1, "config": model.spec, "merges": tokenizer.merges,
                "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                "step": step, "lora": lora, "rng_state": torch.get_rng_state()}, path)


def load_checkpoint(path, device="cpu"):
    from pathlib import Path
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing checkpoint {path}; run MiniGPT train first")
    data = torch.load(path, map_location=device, weights_only=True)
    model = TinyGPT(data["config"]).to(device)
    if data.get("lora"):
        add_lora(model, **data["lora"])
    model.load_state_dict(data["model"])
    return model, ByteBPE(data["merges"]), data


def corpus():
    # Entirely authored templates, split by document before fitting the tokenizer.
    train = [f"The {animal} visits the {place}. It sees a {color} ball. The ball is {color}.\n"
             for animal in ["cat", "dog", "fox", "bird"]
             for place in ["garden", "school", "park"] for color in ["red", "blue", "green"]]
    valid = [f"The {animal} visits the library. It sees a yellow ball. The ball is yellow.\n"
             for animal in ["cat", "dog", "fox", "bird"]]
    return {"train": train, "validation": valid, "source": "Repository-authored synthetic stories; document-level split"}


def windows(texts, tokenizer, block_size):
    stream = [i for text in texts for i in tokenizer.encode(text, special=True)]
    if len(stream) <= block_size:
        raise ValueError("Corpus is too short for block_size")
    return torch.tensor([stream[i:i + block_size + 1]
                         for i in range(0, len(stream) - block_size, max(1, block_size // 2))])


def validation_windows(texts, tokenizer, block_size):
    # Score each validation target exactly once, including short documents and tails.
    batches = []
    for text in texts:
        tokens = tokenizer.encode(text, special=True)
        for start in range(0, len(tokens) - 1, block_size):
            batches.append(torch.tensor([tokens[start:start + block_size + 1]]))
    return batches


def perplexity_metrics(nll):
    if not math.isfinite(nll):
        return {"validation_token_nll": None, "perplexity": None, "perplexity_status": "non_finite_nll"}
    try:
        perplexity = math.exp(nll)
    except OverflowError:
        return {"validation_token_nll": nll, "perplexity": None, "perplexity_status": "overflow"}
    return {"validation_token_nll": nll, "perplexity": perplexity, "perplexity_status": "finite"}


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    total, count = 0., 0
    for batch in (data.split(16) if isinstance(data, torch.Tensor) else data):
        batch = batch.to(device)
        loss = F.cross_entropy(model(batch[:, :-1]).reshape(-1, model.spec["vocab_size"]), batch[:, 1:].reshape(-1), reduction="sum")
        total += loss.item()
        count += batch[:, 1:].numel()
    nll = total / count
    return dict(perplexity_metrics(nll), tokens=count)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=60):
    model.eval()
    ids = [tokenizer.bos] + tokenizer.encode(prompt)
    dev = next(model.parameters()).device
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.spec["block_size"]:]], device=dev)
        scores = model(x)[0, -1].clone()
        scores[tokenizer.pad] = scores[tokenizer.bos] = float("-inf")
        token = scores.argmax().item()
        if token == tokenizer.eos:
            break
        ids.append(token)
    return tokenizer.decode(ids)


def run(stage, cfg):
    seed_all(cfg.get("seed", 42))
    root, dev = outdir(cfg), device_for(cfg)
    if cfg.get("profile") == "local-model":
        raise ValueError("MiniGPT is a from-scratch project; use offline-core. Local Qwen belongs to posttrain/RAG/Agent.")
    if stage == "prepare":
        data = corpus()
        write_json(root / "data.json", data)
        tok = ByteBPE.fit(data["train"], cfg.get("merges", 32))
        write_json(root / "tokenizer.json", {"merges": tok.merges})
        return record(cfg, stage, {"train_documents": len(data["train"]), "validation_documents": len(data["validation"]), "vocab_size": tok.vocab_size})
    data = read_json(root / "data.json")
    checkpoint = cfg.get("checkpoint") or str(root / "model.pt")
    if stage == "train":
        if cfg.get("resume"):
            model, tok, saved = load_checkpoint(checkpoint, dev)
            start = saved["step"]
            torch.set_rng_state(saved["rng_state"].cpu())
        else:
            tok = ByteBPE(read_json(root / "tokenizer.json")["merges"])
            spec = {k: cfg[k] for k in ("dim", "heads", "layers", "block_size")}
            model, saved, start = TinyGPT(dict(spec, vocab_size=tok.vocab_size)).to(dev), {}, 0
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", .002))
        if saved.get("optimizer"):
            opt.load_state_dict(saved["optimizer"])
        train = windows(data["train"], tok, model.spec["block_size"])
        losses = []
        model.train()
        for step in range(start, start + cfg.get("steps", 80)):
            batch = train[torch.randint(len(train), (cfg.get("batch_size", 8),))].to(dev)
            loss = F.cross_entropy(model(batch[:, :-1]).reshape(-1, tok.vocab_size), batch[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(loss.item())
        save_checkpoint(root / "model.pt", model, tok, opt, start + len(losses))
        return record(cfg, stage, {"initial_loss": losses[0], "final_loss": losses[-1], "completed_steps": start + len(losses), "checkpoint": str(root / "model.pt")})
    model, tok, _ = load_checkpoint(checkpoint, dev)
    if stage == "eval":
        return record(cfg, stage, evaluate(model, validation_windows(data["validation"], tok, model.spec["block_size"]), dev))
    if stage in ("demo", "generate"):
        return record(cfg, stage, {"text": generate(model, tok, cfg.get("prompt", "The cat"), cfg.get("max_new_tokens", 60))})
    raise ValueError(f"Unsupported MiniGPT stage: {stage}")
