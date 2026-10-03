"""Regression tests for content navigation and generation, never checksums."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_full
from content_model import anchors, dependency_cycles, heading_slug, prerequisite_ids, validate_manifest


class ContentToolsTests(unittest.TestCase):
    def test_legacy_full_book_intro_links_remain_valid(self):
        legacy = {"awesome-ai-dictionary从线性回归到现代基础模型", "0-如何使用本讲义",
                  "01-标记约定", "02-代码环境", "03-宏观依赖图", "04-推荐学习路径", "05-章节导航"}
        self.assertLessEqual(legacy, anchors(build_full.build()))

    def test_id_ranges(self):
        self.assertEqual(prerequisite_ids("A09–A11，B02–04 与 C01"), {"A09", "A10", "A11", "B02", "B03", "B04", "C01"})

    def test_cycle_and_dag(self):
        self.assertEqual(dependency_cycles({"A01": set(), "B01": {"A01"}}), [])
        self.assertEqual(dependency_cycles({"A01": {"B01"}, "B01": {"A01"}}), [["A01", "B01", "A01"]])

    def test_anchors_ignore_fences_and_support_duplicate_titles(self):
        text = '# A. 数学\n<a id="a01"></a>\n## 示例\n## 示例\n```python\n# fake\n```\n'
        self.assertEqual(anchors(text), {"a-数学", "a01", "示例", "示例-1"})

    def test_slug_inline_link(self):
        self.assertEqual(heading_slug("S01 [检查](a.md) 与 `代码`"), "s01-检查-与-代码")

    def test_explicit_anchor_inside_code_is_not_a_target(self):
        self.assertEqual(anchors('```html\n<a id="example"></a>\n```\n<a id="real"></a>'), {"real"})

    def test_manifest_rejects_ambiguous_unit_and_escaping_path(self):
        data = {"version": 1, "reading_index": "readings/README.md", "chapters": [
            {"path": "docs/a.md", "title": "A", "kind": "knowledge", "units": ["A01", "A01"]}]}
        with self.assertRaisesRegex(ValueError, "重复归属"):
            validate_manifest(data)
        data["chapters"][0]["units"] = ["A01"]
        data["chapters"][0]["path"] = "../outside.md"
        with self.assertRaisesRegex(ValueError, "仓库内"):
            validate_manifest(data)

    def test_full_rewrites_cross_chapter_fragments_and_preserves_code(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "docs").mkdir()
            (root / "full").mkdir()
            a, b = root / "docs/a.md", root / "docs/b.md"
            a.write_text('# A\n[跨章](b.md#b01)\n[资产](image.png)\n```text\n[示例](b.md#b01)\n```\n', encoding="utf-8")
            b.write_text('# B\n<a id="b01"></a>\n## 同名\n## 同名\n', encoding="utf-8")
            with patch.object(build_full, "ROOT", root), patch.object(build_full, "OUTPUT", root / "full/book.md"):
                output = build_full.render_page(a, {a.resolve(), b.resolve()})
                self.assertIn('[跨章](#page-docs-b-md-b01)', output)
                self.assertIn('[资产](../docs/image.png)', output)
                self.assertIn('[示例](b.md#b01)', output)
                rendered_b = build_full.render_page(b, {a.resolve(), b.resolve()})
                self.assertIn('id="page-docs-b-md-b01"', rendered_b)
                self.assertIn('id="page-docs-b-md-同名-1"', rendered_b)

    def test_same_page_fragment_and_external_links(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            page = root / "a.md"
            page.write_text('# Page\n[本页](#part)\n<a id="part"></a>\n[外部](https://example.org/p#x)\n', encoding="utf-8")
            with patch.object(build_full, "ROOT", root), patch.object(build_full, "OUTPUT", root / "full.md"):
                output = build_full.render_page(page, {page.resolve()})
                self.assertIn('[本页](#page-a-md-part)', output)
                self.assertIn('[外部](https://example.org/p#x)', output)


if __name__ == "__main__":
    unittest.main()
