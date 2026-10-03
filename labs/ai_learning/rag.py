from __future__ import annotations
from collections import Counter
import json
import pickle
import re
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from .common import seed_all, device_for, outdir, read_json, write_json, record, require_local, local_causal, local_generate

STOP = set("a an the is are was to of for in at and what when where how does do can i it its".split())


def terms(text):
    return [x for x in re.findall(r"[a-z0-9]+", text.lower()) if x not in STOP]


def dataset():
    facts = [
        ("library", "The Lumen campus library opens at 08:00 and closes at 20:00. Books may be borrowed for 21 days."),
        ("robotics", "The robotics laboratory is in building Cedar, room 204. Its coordinator is Mira."),
        ("wifi", "The guest wifi network is LumenGuest. A visitor obtains a daily access code from the reception desk."),
        ("cafe", "The campus cafe serves lunch from 11:30 to 14:00. Vegetarian meals are available every weekday."),
        ("exam", "The introduction to AI examination takes place on November 12 in Hall Birch. Calculators are allowed."),
        ("printing", "The printing room is beside the library. Monochrome printing costs 0.10 credits per page."),
        ("bus", "The campus shuttle bus departs every 20 minutes. The first shuttle departs at 07:00."),
        ("support", "The computing help desk opens from 09:00 to 17:00. Its email is support@example.invalid."),
        ("sports", "The sports center closes at 22:00. Indoor court reservations last 60 minutes."),
        ("garden", "The campus garden contains a blue sculpture named Orbit. Visitors enter through the east gate.")]
    validation = [
        ("When does the library open?", "library", "08:00"), ("Where is the robotics laboratory?", "robotics", "204"),
        ("Which wifi network serves guests?", "wifi", "LumenGuest"), ("Who won the world chess championship?", None, None),
        ("What is the population of Tokyo?", None, None), ("What is the garden gate?", "garden", "east")]
    test = [("How many days can books be borrowed?", "library", "21"),
        ("Who coordinates the robotics laboratory?", "robotics", "Mira"), ("When is lunch served at the cafe?", "cafe", "11:30"),
        ("Where is the AI examination?", "exam", "Birch"), ("How much does monochrome printing cost?", "printing", "0.10"),
        ("When does the first shuttle depart?", "bus", "07:00"), ("What is the computing help desk email?", "support", "support@example.invalid"),
        ("What time does the sports center close?", "sports", "22:00"), ("What is the garden sculpture called?", "garden", "Orbit"),
        ("What is the ocean depth?", None, None), ("Who discovered planet Neptune?", None, None)]
    convert = lambda rows: [{"query": q, "gold_id": g, "answer": a} for q, g, a in rows]
    return {"documents": [{"id": i, "text": t} for i, t in facts], "validation": convert(validation), "test": convert(test),
            "source": "Entirely fictional, repository-authored Lumen campus; no real personal data"}


class BM25:
    def __init__(self, documents):
        self.rows = [Counter(terms(x)) for x in documents]
        self.lengths = np.array([sum(x.values()) for x in self.rows], dtype=float)
        self.avg = max(self.lengths.mean(), 1.)
        self.df = Counter(t for row in self.rows for t in row)

    def score(self, query):
        result = np.zeros(len(self.rows))
        for t in terms(query):
            df = self.df[t]
            idf = np.log(1 + (len(self.rows) - df + .5) / (df + .5))
            f = np.array([row[t] for row in self.rows], dtype=float)
            result += idf * (f * 2.5) / (f + 1.5 * (.25 + .75 * self.lengths / self.avg))
        return result


class LocalEmbedder:
    def __init__(self, cfg):
        path = require_local(cfg.get("embedding_path"), "paraphrase-multilingual-MiniLM-L12-v2")
        from transformers import AutoTokenizer, AutoModel
        self.device = device_for(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False).to(self.device).eval()

    @torch.no_grad()
    def encode(self, texts):
        result = []
        for i in range(0, len(texts), 8):
            batch = self.tokenizer(texts[i:i + 8], padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            hidden = self.model(**batch).last_hidden_state
            mask = batch.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            result.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
        return np.concatenate(result)


class Retriever:
    def __init__(self, docs, vectorizer, svd, vectors, embedder=None):
        self.docs, self.vectorizer, self.svd, self.vectors, self.embedder = docs, vectorizer, svd, vectors, embedder
        self.bm25 = BM25([d["text"] for d in docs])

    def search(self, query, k=3):
        q = self.embedder.encode([query]) if self.embedder else normalize(self.svd.transform(self.vectorizer.transform([query])))
        semantic = np.maximum(0., (self.vectors @ q[0]))
        lexical = self.bm25.score(query)
        hybrid = .55 * lexical / (lexical + 2.) + .45 * semantic
        candidates = np.argsort(-hybrid)[:min(6, len(self.docs))]
        query_terms = set(terms(query))
        overlap = np.array([len(query_terms & set(terms(d["text"]))) / max(len(query_terms), 1) for d in self.docs])
        reranked = .7 * hybrid + .3 * overlap
        chosen = sorted(candidates, key=lambda i: (-reranked[i], self.docs[i]["id"]))[:k]
        return [{**self.docs[i], "score": float(reranked[i]), "bm25": float(lexical[i]), "semantic": float(semantic[i])} for i in chosen]


def choose_threshold(retriever, rows):
    observations = [(retriever.search(r["query"])[0]["score"], r["gold_id"] is not None) for r in rows]
    candidates = sorted({0., 1.} | {s + delta for s, _ in observations for delta in (-1e-6, 1e-6)})
    return max(candidates, key=lambda t: (sum((s >= t) == answerable for s, answerable in observations), t))


def answer(retriever, query, threshold, local_pair=None):
    hits = retriever.search(query)
    if not hits or hits[0]["score"] < threshold:
        return {"answer": "NO_ANSWER", "citations": [], "no_answer": True, "retrieval": hits}
    if local_pair is None:
        return {"answer": hits[0]["text"], "citations": [hits[0]["id"]], "no_answer": False, "retrieval": hits}
    tok, model = local_pair
    context = "\n".join(f"[{r['id']}] {r['text']}" for r in hits)
    prompt = ('Use only the supplied evidence. Return one JSON object with keys answer (string), citations (list of evidence IDs), no_answer (boolean). '
              'If unsupported, return {"answer":"NO_ANSWER","citations":[],"no_answer":true}.\nEvidence:\n' + context + '\nQuestion: ' + query)
    raw = local_generate(tok, model, prompt, 100)
    try:
        result = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        if set(result) != {"answer", "citations", "no_answer"} or not isinstance(result["answer"], str) or not isinstance(result["no_answer"], bool) or not isinstance(result["citations"], list):
            raise ValueError("Invalid answer schema")
        if any(not isinstance(x, str) or x not in {h["id"] for h in hits} for x in result["citations"]):
            raise ValueError("Citation ID is absent from retrieved evidence")
        if not result["no_answer"] and not result["citations"]:
            raise ValueError("An answer must cite evidence")
        return dict(result, retrieval=hits, raw=raw)
    except (ValueError, TypeError, AttributeError):
        return {"answer": "NO_ANSWER", "citations": [], "no_answer": True, "retrieval": hits, "generation_error": "invalid_schema_or_citation", "raw": raw}


def run(stage, cfg):
    seed_all(cfg.get("seed", 42))
    root = outdir(cfg)
    local = cfg.get("profile") == "local-model"
    kind = "local_model" if local else "offline_algorithm"
    if stage == "prepare":
        data = dataset()
        write_json(root / "data.json", data)
        return record(cfg, stage, {"documents": len(data["documents"]), "test_queries": len(data["test"])})
    data = read_json(root / "data.json")
    docs, texts = data["documents"], [d["text"] for d in data["documents"]]
    index_path = root / ("index_local.pkl" if local else "index_offline.pkl")
    embedder = LocalEmbedder(cfg) if local else None
    if stage in ("index", "train"):
        vec = TfidfVectorizer(ngram_range=(1, 2))
        matrix = vec.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(cfg.get("svd_components", 6), matrix.shape[0] - 1, matrix.shape[1] - 1), random_state=cfg.get("seed", 42))
        reduced = normalize(svd.fit_transform(matrix))
        vectors = embedder.encode(texts) if local else reduced
        retriever = Retriever(docs, vec, svd, vectors, embedder)
        threshold = choose_threshold(retriever, data["validation"])
        with index_path.open("wb") as f:
            pickle.dump({"documents": docs, "vectorizer": vec, "svd": svd, "vectors": vectors, "threshold": threshold,
                         "embedding_path": cfg.get("embedding_path") if local else None}, f)
        return record(cfg, "index", {"index": str(index_path), "no_answer_threshold": threshold,
            "semantic_method": "local_MiniLM_mean_pooling" if local else "TFIDF_TruncatedSVD", "reranker": "lexical_overlap_rule"}, kind)
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}; run index for this profile first")
    # Only load indexes generated locally by this project, never untrusted pickle files.
    with index_path.open("rb") as f:
        index = pickle.load(f)
    if index["documents"] != docs:
        raise ValueError("Source documents changed; rebuild the index")
    if local and index["embedding_path"] != cfg.get("embedding_path"):
        raise ValueError("Embedding directory changed; rebuild the index")
    retriever = Retriever(index["documents"], index["vectorizer"], index["svd"], index["vectors"], embedder)
    # Precompute query vectors, then release the encoder before loading the generator.
    queries = [r["query"] for r in data["test"]] if stage == "eval" else [cfg.get("prompt", "Where is the robotics laboratory?")]
    if local:
        cached = dict(zip(queries, embedder.encode(queries)))
        class Cached:
            def encode(self, texts):
                return np.array([cached[t] for t in texts])
        retriever.embedder = Cached()
        del embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pair = local_causal(cfg) if local else None
    if stage == "eval":
        outputs = [answer(retriever, q, index["threshold"], pair) for q in queries]
        supported = [(r, o) for r, o in zip(data["test"], outputs) if r["gold_id"]]
        n = len(supported)
        metrics = {"retrieval_recall_at_3": sum(r["gold_id"] in [h["id"] for h in o["retrieval"]] for r, o in supported) / n,
            "answer_keyword_accuracy": sum(r["answer"].lower() in o["answer"].lower() and not o["no_answer"] for r, o in supported) / n,
            "citation_accuracy_supported": sum(r["gold_id"] in o["citations"] for r, o in supported) / n,
            "no_answer_accuracy": sum("generation_error" not in o and o["no_answer"] == (r["gold_id"] is None) for r, o in zip(data["test"], outputs)) / len(outputs),
            "generation_failures": sum("generation_error" in o for o in outputs), "examples": [dict(query=q, **o) for q, o in zip(queries, outputs)]}
        return record(cfg, stage, metrics, kind)
    if stage == "demo":
        return record(cfg, stage, answer(retriever, queries[0], index["threshold"], pair), kind)
    raise ValueError(f"Unsupported RAG stage: {stage}")
