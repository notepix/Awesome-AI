from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from .common import seed_all, device_for, outdir, read_json, write_json, record, require_local

COLORS = {"red": (230, 45, 45), "green": (35, 180, 65), "blue": (40, 85, 230), "yellow": (235, 200, 30)}
SHAPES = ["circle", "square", "triangle"]
HELD_OUT = {("red", "triangle"), ("green", "square"), ("blue", "circle")}


def prepare_data(root, per_class=12, seed=42):
    if per_class < 4:
        raise ValueError("images_per_class must be at least 4")
    rng = np.random.default_rng(seed)
    folder = root / "images"
    folder.mkdir(exist_ok=True)
    rows = []
    for color, rgb in COLORS.items():
        for shape in SHAPES:
            for i in range(per_class):
                image = Image.new("RGB", (32, 32), (245, 245, 245))
                draw = ImageDraw.Draw(image)
                x, y = rng.integers(3, 9, size=2).tolist()
                size = int(rng.integers(16, 23))
                box = [x, y, min(x + size, 30), min(y + size, 30)]
                if shape == "circle":
                    draw.ellipse(box, fill=rgb)
                elif shape == "square":
                    draw.rectangle(box, fill=rgb)
                else:
                    draw.polygon([(x + size // 2, y), (x, min(y + size, 30)), (min(x + size, 30), min(y + size, 30))], fill=rgb)
                name = f"{color}_{shape}_{i:02d}.png"
                image.save(folder / name)
                split = "test" if (color, shape) in HELD_OUT else ("validation" if i >= per_class - 3 else "train")
                rows.append({"id": name[:-4], "image": str(folder / name), "caption": f"a {color} {shape}", "split": split})
    write_json(root / "data.json", {"rows": rows, "held_out_combinations": sorted([list(x) for x in HELD_OUT]),
        "source": "Repository-generated geometric images and authored English captions; all primitives seen in training"})
    return rows


def image_tensor(rows):
    images = []
    for row in rows:
        with Image.open(row["image"]) as image:
            images.append(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.)
    return torch.tensor(np.stack(images)).permute(0, 3, 1, 2)


def text_tensor(captions, vocab):
    try:
        return torch.tensor([[vocab[w] for w in caption.split()] for caption in captions])
    except KeyError as exc:
        raise ValueError(f"Unknown primitive word {exc}; this toy model supports only generated color/shape captions") from exc


class TwoTower(nn.Module):
    def __init__(self, vocab_size, dim=32):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(512, dim))
        self.words = nn.Embedding(vocab_size, dim)
        self.text = nn.Linear(dim, dim)
        self.log_scale = nn.Parameter(torch.tensor(np.log(10.), dtype=torch.float32))

    def encode_image(self, images):
        return F.normalize(self.image(images), dim=-1)

    def encode_text(self, tokens):
        return F.normalize(self.text(self.words(tokens).mean(1)), dim=-1)

    def forward(self, images, tokens):
        return self.log_scale.exp().clamp(max=100) * self.encode_image(images) @ self.encode_text(tokens).T


def contrastive_loss(logits, labels):
    # Every matching caption is a positive; duplicate-class images are not false negatives.
    positive = labels[:, None] == labels[None, :]
    def direction(scores, mask):
        log_positive = torch.logsumexp(scores.masked_fill(~mask, float("-inf")), dim=1)
        return (torch.logsumexp(scores, dim=1) - log_positive).mean()
    return (direction(logits, positive) + direction(logits.T, positive.T)) / 2


def recalls(image_vectors, text_vectors, image_labels, text_labels):
    similarity = image_vectors @ text_vectors.T
    values = {}
    for k in (1, 3, 5):
        image_ranks = np.argsort(-similarity, axis=1)[:, :min(k, len(text_labels))]
        text_ranks = np.argsort(-similarity.T, axis=1)[:, :min(k, len(image_labels))]
        values[f"image_to_text_recall_at_{k}"] = float(np.mean([any(text_labels[j] == label for j in rank) for label, rank in zip(image_labels, image_ranks)]))
        # Only query captions that have at least one relevant image in this split.
        valid = [i for i, label in enumerate(text_labels) if label in set(image_labels)]
        values[f"text_to_image_recall_at_{k}"] = float(np.mean([any(image_labels[j] == text_labels[i] for j in text_ranks[i]) for i in valid]))
    return values


@torch.no_grad()
def encode_tiny(model, rows, captions, vocab, dev):
    model.eval()
    iv = torch.cat([model.encode_image(batch.to(dev)).cpu() for batch in image_tensor(rows).split(32)]).numpy()
    tv = model.encode_text(text_tensor(captions, vocab).to(dev)).cpu().numpy()
    return iv, tv


@torch.no_grad()
def encode_clip(cfg, rows, captions):
    path = require_local(cfg.get("model_path"), "openai/clip-vit-base-patch32")
    from transformers import AutoProcessor, CLIPModel
    dev = device_for(cfg)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = CLIPModel.from_pretrained(path, local_files_only=True).to(dev).eval()
    vectors = []
    for i in range(0, len(rows), 8):
        images = []
        for row in rows[i:i + 8]:
            with Image.open(row["image"]) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(dev)
        vectors.append(F.normalize(model.get_image_features(**inputs), dim=-1).cpu().numpy())
    inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(dev)
    texts = F.normalize(model.get_text_features(**inputs), dim=-1).cpu().numpy()
    return np.concatenate(vectors), texts


def run(stage, cfg):
    seed_all(cfg.get("seed", 42))
    root, dev = outdir(cfg), device_for(cfg)
    local = cfg.get("profile") == "local-model"
    kind = "local_model" if local else "offline_algorithm"
    if stage == "prepare":
        rows = prepare_data(root, cfg.get("images_per_class", 12), cfg.get("seed", 42))
        return record(cfg, stage, {split: sum(r["split"] == split for r in rows) for split in ("train", "validation", "test")})
    rows = read_json(root / "data.json")["rows"]
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    captions = sorted({r["caption"] for r in rows})
    vocab = {w: i for i, w in enumerate(sorted({w for r in train for w in r["caption"].split()}))}
    if local:
        if stage == "train":
            raise ValueError("Local CLIP is an inference extension; run index, eval and demo. Train the two-tower with offline-core.")
        if stage == "index":
            iv, tv = encode_clip(cfg, test, captions)
            np.savez(root / "clip_index.npz", images=iv, texts=tv)
            write_json(root / "clip_index.json", {"model_path": cfg["model_path"], "image_ids": [r["id"] for r in test], "captions": captions})
            return record(cfg, stage, {"images": len(test), "captions": len(captions), "caption_language": "English"}, kind)
        require_local(cfg.get("model_path"), "CLIP")
        metadata = read_json(root / "clip_index.json")
        if metadata["model_path"] != cfg.get("model_path") or metadata["image_ids"] != [r["id"] for r in test] or metadata["captions"] != captions:
            raise ValueError("CLIP index metadata changed; run index again")
        saved_vectors = np.load(root / "clip_index.npz")
        iv, tv = saved_vectors["images"], saved_vectors["texts"]
    else:
        model = TwoTower(len(vocab), cfg.get("dim", 32)).to(dev)
        if stage == "train":
            images = image_tensor(train).to(dev)
            tokens = text_tensor([r["caption"] for r in train], vocab).to(dev)
            class_ids = {caption: i for i, caption in enumerate(captions)}
            labels = torch.tensor([class_ids[r["caption"]] for r in train], device=dev)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", .003))
            losses = []
            before_iv, before_tv = encode_tiny(model, test, captions, vocab, dev)
            model.train()
            for step in range(cfg.get("steps", 100)):
                indices = torch.randint(len(train), (cfg.get("batch_size", 32),), device=dev)
                loss = contrastive_loss(model(images[indices], tokens[indices]), labels[indices])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            torch.save({"model": model.state_dict(), "vocab": vocab, "dim": cfg.get("dim", 32)}, root / "model.pt")
            return record(cfg, stage, {"initial_loss": losses[0], "final_loss": losses[-1],
                "untrained_baseline": recalls(before_iv, before_tv, [r["caption"] for r in test], captions)}, kind)
        path = Path(cfg.get("checkpoint") or root / "model.pt")
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run train first")
        saved = torch.load(path, map_location=dev, weights_only=True)
        model = TwoTower(len(saved["vocab"]), saved["dim"]).to(dev)
        model.load_state_dict(saved["model"])
        iv, tv = encode_tiny(model, test, captions, saved["vocab"], dev)
        if stage == "index":
            np.savez(root / "tiny_index.npz", images=iv, texts=tv)
            return record(cfg, stage, {"images": len(test), "captions": len(captions)}, kind)
    if stage == "eval":
        metrics = recalls(iv, tv, [r["caption"] for r in test], captions)
        return record(cfg, stage, dict(metrics, split="held_out_attribute_combinations", relevance="same semantic caption, not arbitrary image pairing", candidate_captions=len(captions), test_images=len(test)), kind)
    if stage == "demo":
        query = cfg.get("prompt", "a red triangle")
        if query not in captions:
            raise ValueError("Choose an existing caption: " + ", ".join(captions))
        scores = iv @ tv[captions.index(query)]
        ranking = np.argsort(-scores)[:5]
        return record(cfg, stage, {"query": query, "results": [dict(test[i], similarity=float(scores[i])) for i in ranking]}, kind)
    raise ValueError(f"Unsupported multimodal stage: {stage}")
