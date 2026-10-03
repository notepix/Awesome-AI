"""Meaningful algorithm/contract tests; no network, downloaded assets, or digest checks."""
import json
import math
import tempfile
import threading
import time
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from labs.ai_learning.minigpt import ByteBPE, TinyGPT, add_lora, save_checkpoint, load_checkpoint, windows, validation_windows, evaluate, run as run_minigpt
from labs.ai_learning.posttrain import encode_example, encode_preference_pair, pack, sequence_scores, dpo_loss
from labs.ai_learning.rag import BM25, dataset, run as run_rag
from labs.ai_learning.agent import calculator, dispatch, controller, FixtureProvider, LocalProvider, fixtures, answer_matches, evaluate_cases
from labs.ai_learning.multimodal import prepare_data, contrastive_loss, recalls


class LanguageModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.tokenizer = ByteBPE.fit(["red blue red blue", "中文学习 AI red"], 8)
        self.config = {"vocab_size": self.tokenizer.vocab_size, "dim": 16, "heads": 2, "layers": 1, "block_size": 64}

    def test_reversible_bpe_unseen_unicode(self):
        text = "中文🙂\nUnseen café 123"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text, special=True)), text)

    def test_causal_attention_cannot_see_future(self):
        model = TinyGPT(self.config).eval()
        a = torch.tensor([[1, 6, 7, 8, 9]])
        b = a.clone()
        b[0, 3:] = torch.tensor([10, 11])
        torch.testing.assert_close(model(a)[:, :3], model(b)[:, :3])

    def test_window_shift(self):
        data = windows(["abcde" * 30], self.tokenizer, 8)
        self.assertEqual(data.shape[1], 9)
        self.assertTrue(torch.equal(data[0, 4:], data[1, :5]))

    def test_validation_counts_short_documents_once(self):
        texts = ["short", "a" * 150]
        batches = validation_windows(texts, self.tokenizer, 16)
        expected = sum(len(self.tokenizer.encode(text, special=True)) - 1 for text in texts)
        self.assertEqual(sum(batch.shape[1] - 1 for batch in batches), expected)
        self.assertTrue(all(1 < batch.shape[1] <= 17 for batch in batches))

    def test_cpu_resume_matches_uninterrupted_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"project": "minigpt", "profile": "offline-core", "seed": 7, "device": "cpu",
                "merges": 4, "block_size": 32, "dim": 16, "heads": 2, "layers": 1, "batch_size": 2,
                "steps": 2, "lr": .002, "output": str(Path(tmp) / "a")}
            run_minigpt("prepare", cfg)
            run_minigpt("train", cfg)
            other = dict(cfg, output=str(Path(tmp) / "b"), steps=1)
            run_minigpt("prepare", other)
            run_minigpt("train", other)
            run_minigpt("train", dict(other, resume=True))
            a, _, first = load_checkpoint(Path(cfg["output"]) / "model.pt")
            b, _, second = load_checkpoint(Path(other["output"]) / "model.pt")
            self.assertEqual(first["step"], second["step"])
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)

    def test_response_mask_and_padding(self):
        row = {"prompt": "What color?", "chosen": "red."}
        item = encode_example(self.tokenizer, row, "chosen", 64)
        tokens, labels = item
        self.assertTrue(all(x == -100 for x in labels[:-len(self.tokenizer.encode("red.")) - 1]))
        self.assertEqual(labels[-1], self.tokenizer.eos)
        ids, targets, mask = pack([item, ([1, 2], [-100, 2])], 0, "cpu")
        self.assertTrue((targets[1, 2:] == -100).all())
        score, counts = sequence_scores(TinyGPT(self.config), (ids, targets, mask))
        self.assertEqual(counts.tolist(), [len(self.tokenizer.encode("red.")) + 1, 1])
        self.assertTrue(torch.isfinite(score).all())

    def test_preference_pair_truncates_one_shared_prompt(self):
        row = {"prompt": "long context " * 60, "chosen": "yes", "rejected": "a longer wrong answer"}
        pair = encode_preference_pair(self.tokenizer, row, 40)
        prefixes = []
        for key in ("chosen", "rejected"):
            ids, labels = pair[key]
            prefix_len = labels.index(next(token for token in labels if token != -100))
            prefixes.append(ids[:prefix_len])
            self.assertEqual(ids[prefix_len:], self.tokenizer.encode(row[key]) + [self.tokenizer.eos])
            self.assertEqual(labels[:prefix_len], [-100] * prefix_len)
            self.assertLessEqual(len(ids), 40)
        self.assertEqual(prefixes[0], prefixes[1])
        self.assertEqual(max(len(ids) for ids, _ in pair.values()), 40)
        with self.assertRaises(ValueError):
            encode_preference_pair(self.tokenizer, dict(row, rejected="x" * 100), 10)

    def test_local_preference_pair_uses_identical_chat_prefix(self):
        class ChatTokenizer:
            eos_token_id = 2
            def apply_chat_template(self, messages, **kwargs):
                return "<user>" + messages[0]["content"] + "<assistant>"
            def encode(self, text, **kwargs):
                return list(text.encode("utf-8"))
        tok = ChatTokenizer()
        row = {"prompt": "context " * 80, "chosen": "yes", "rejected": "definitely no"}
        pair = encode_preference_pair(tok, row, 32, local=True)
        prefixes = [[token for token, label in zip(ids, labels) if label == -100] for ids, labels in pair.values()]
        self.assertEqual(prefixes[0], prefixes[1])
        for key, (ids, labels) in pair.items():
            self.assertEqual(ids[len(prefixes[0]):], tok.encode(row[key]) + [tok.eos_token_id])
            self.assertLessEqual(len(ids), 32)

    def test_perplexity_above_exp80_is_not_clipped(self):
        class FixedLogits(torch.nn.Module):
            spec = {"vocab_size": 2}
            def forward(self, ids):
                return torch.tensor([0., -90.], dtype=torch.float64).expand(*ids.shape, 2)
        result = evaluate(FixedLogits(), [torch.tensor([[0, 1]])], "cpu")
        self.assertAlmostEqual(result["validation_token_nll"], 90.)
        self.assertEqual(result["perplexity"], math.exp(90))
        self.assertEqual(result["perplexity_status"], "finite")
        json.dumps(result, allow_nan=False)

    def test_perplexity_overflow_and_nonfinite_are_json_safe(self):
        for nll, status in ((1000., "overflow"), (float("inf"), "non_finite_nll"), (float("nan"), "non_finite_nll")):
            with self.subTest(nll=nll):
                class FixedLogits(torch.nn.Module):
                    spec = {"vocab_size": 2}
                    def forward(self, ids):
                        return torch.tensor([0., -nll], dtype=torch.float64).expand(*ids.shape, 2)
                result = evaluate(FixedLogits(), [torch.tensor([[0, 1]])], "cpu")
                self.assertIsNone(result["perplexity"])
                self.assertEqual(result["perplexity_status"], status)
                self.assertEqual(result["validation_token_nll"], 1000. if status == "overflow" else None)
                json.dumps(result, allow_nan=False)

    def test_dpo_matches_reference_and_gradient_direction(self):
        chosen = torch.tensor([-2.], requires_grad=True)
        rejected = torch.tensor([-3.], requires_grad=True)
        loss = dpo_loss(chosen, rejected, chosen.detach(), rejected.detach())
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertLess(chosen.grad.item(), 0)
        self.assertGreater(rejected.grad.item(), 0)

    def test_lora_initial_equivalence_and_checkpoint(self):
        model = TinyGPT(self.config)
        ids = torch.tensor([[1, 4, 6]])
        before = model(ids).detach()
        add_lora(model, rank=2, alpha=4)
        torch.testing.assert_close(model(ids), before)
        model(ids).sum().backward()
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if not n.endswith((".a", ".b"))))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint(path, model, self.tokenizer, lora={"rank": 2, "alpha": 4})
            restored, tok, _ = load_checkpoint(path)
            torch.testing.assert_close(restored(ids), model(ids))
            self.assertEqual(tok.merges, self.tokenizer.merges)


class RetrievalAgentTests(unittest.TestCase):
    def test_bm25_unknown_query(self):
        bm = BM25(["red cat", "blue dog"])
        self.assertEqual(int(bm.score("cat").argmax()), 0)
        np.testing.assert_array_equal(bm.score("spaceship"), [0, 0])

    def test_rag_end_to_end_with_separate_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"project": "rag", "output": tmp, "profile": "offline-core", "device": "cpu", "seed": 42}
            run_rag("prepare", cfg)
            run_rag("index", cfg)
            result = run_rag("eval", cfg)
            for metric in ("retrieval_recall_at_3", "answer_keyword_accuracy", "citation_accuracy_supported", "no_answer_accuracy"):
                self.assertTrue(0 <= result["metrics"][metric] <= 1)
            demo = run_rag("demo", dict(cfg, prompt="quasar spaceship nebula"))
            self.assertTrue(demo["metrics"]["no_answer"])
            self.assertEqual(demo["metrics"]["citations"], [])

    def test_calculator_rejects_execution_and_oversized_arithmetic(self):
        self.assertEqual(calculator("(6 + 1) * 6")["value"], 42)
        for expression in ("__import__('os').getcwd()", "2**99999", "True", "1e100", "[1][0]"):
            with self.assertRaises((ValueError, SyntaxError)):
                calculator(expression)

    def test_tool_schema_whitelist(self):
        for action in ({"tool": "shell", "arguments": {}}, {"tool": "search", "arguments": {"query": "a", "path": "../secret"}},
                       {"tool": "search", "arguments": {"query": "a", "k": True}}):
            with self.assertRaises(ValueError):
                dispatch(action, dataset()["documents"])

    def test_controller_recovers_and_obeys_budget(self):
        for case in fixtures():
            result = controller(FixtureProvider(case["actions"]), case["request"], dataset()["documents"], {})
            self.assertEqual(result["status"], case["status"])
            self.assertIn(case["expected"], result["final"])

    def test_final_requires_actual_tool(self):
        result = controller(FixtureProvider([{"final": "42"}]), "calculate", [], {})
        self.assertEqual(result["status"], "error_budget")
        self.assertTrue(all("error" in row for row in result["trace"]))

    def test_local_provider_receives_the_actual_call_budget(self):
        for total, per_call in ((20., .5), (.5, 20.)):
            with self.subTest(total=total, per_call=per_call):
                with patch("labs.ai_learning.agent.local_causal", return_value=(object(), object())):
                    provider = LocalProvider({})
                with patch("labs.ai_learning.agent.local_generate", return_value='{"final":"42"}') as generate_call:
                    result = controller(provider, "calculate", [], {"max_seconds": total, "call_timeout": per_call, "require_tool_before_final": False})
                self.assertEqual(result["status"], "completed")
                budget = generate_call.call_args.kwargs["max_time"]
                self.assertGreater(budget, 0)
                self.assertLessEqual(budget, min(total, per_call))
                if per_call < total:
                    self.assertEqual(budget, per_call)

    def test_local_evaluation_never_reuses_a_timed_out_provider(self):
        release, finished = threading.Event(), threading.Event()
        calls = []
        def blocked_provider(request, observations, budget):
            calls.append(request)
            try:
                release.wait(2)
                return '{"final":"42"}'
            finally:
                finished.set()
        try:
            outputs = evaluate_cases(blocked_provider, fixtures()[:2], dataset()["documents"],
                                     {"call_timeout": .02, "max_seconds": 5}, local=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(outputs[0]["status"], "timeout")
            self.assertFalse(outputs[0]["provider_reusable"])
            self.assertEqual(outputs[1]["status"], "not_run_after_timeout")
            self.assertFalse(any(row["success"] for row in outputs))
            self.assertFalse(finished.is_set())
        finally:
            release.set()
            self.assertTrue(finished.wait(1))

    def test_numeric_answer_is_exact_not_substring(self):
        case = fixtures()[0]
        for value in ("42", "42.0", "4.2e1", "The answer is 42."):
            self.assertTrue(answer_matches(case, value), value)
        for value in ("142", "420", "-42", "142 and 42", "6 * 7 = 42"):
            self.assertFalse(answer_matches(case, value), value)
        bad = dict(case, actions=[case["actions"][0], {"final": "142"}])
        self.assertFalse(evaluate_cases(None, [bad], [], {})[0]["success"])

    def test_time_answer_is_exact_and_unambiguous(self):
        case = fixtures()[1]
        for value in ("08:00 [library]", "The library opens at 8:00."):
            self.assertTrue(answer_matches(case, value), value)
        for value in ("108:00", "18:00", "08:000", "08:00 and 18:00", "08:00:00"):
            self.assertFalse(answer_matches(case, value), value)

    def test_controller_timeout(self):
        def slow(*args):
            time.sleep(.02)
            return '{"final":"late"}'
        self.assertEqual(controller(slow, "test", [], {"call_timeout": .001})["status"], "timeout")


class MultiModalTests(unittest.TestCase):
    def test_combinations_are_disjoint_but_primitives_seen(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = prepare_data(Path(tmp), per_class=4)
            train = {r["caption"] for r in rows if r["split"] == "train"}
            test = {r["caption"] for r in rows if r["split"] == "test"}
            self.assertFalse(train & test)
            self.assertLessEqual({w for c in test for w in c.split()}, {w for c in train for w in c.split()})

    def test_duplicate_captions_are_positive(self):
        logits = torch.randn(3, 3, requires_grad=True)
        loss = contrastive_loss(logits, torch.zeros(3, dtype=torch.long))
        self.assertAlmostEqual(loss.item(), 0, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bidirectional_semantic_recall(self):
        iv = np.array([[1., 0.], [1., 0.], [0., 1.]])
        tv = np.eye(2)
        metrics = recalls(iv, tv, ["red", "red", "blue"], ["red", "blue"])
        self.assertEqual(metrics["image_to_text_recall_at_1"], 1)
        self.assertEqual(metrics["text_to_image_recall_at_1"], 1)


if __name__ == "__main__":
    unittest.main()
