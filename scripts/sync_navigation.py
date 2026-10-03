"""Generate knowledge index and unit-to-reading links; compare plain text in --check."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from content_model import ROOT, load_manifest

START = "<!-- readings:start -->"
END = "<!-- readings:end -->"
INDEX = ROOT / "docs/07-resources/knowledge-index.md"


def relative_link(source: Path, target: str) -> str:
    return Path(os.path.relpath(ROOT / target, source.parent)).as_posix()


def expected_files() -> dict[Path, str]:
    manifest = load_manifest()
    catalog = json.loads((ROOT / "readings/catalog.json").read_text(encoding="utf-8"))
    mapping = {}
    for kind in ("papers", "projects"):
        for item in catalog[kind]:
            for unit in item["units"]:
                mapping.setdefault(unit, []).append(item)
    files = {}
    index = [
        "# 知识点与术语索引",
        "[学习目录](../README.md) · [系统推导](../08-walkthroughs/README.md) · [论文与源码](../../readings/README.md) · [实践](../../labs/README.md)",
        "> 由章节标题和精读清单自动生成：`python scripts/sync_navigation.py`。标题是检索入口，编号是稳定跳转接口；精读不是该单元的必修前置。",
    ]
    for chapter in manifest["chapters"]:
        page = ROOT / chapter["path"]
        body = page.read_text(encoding="utf-8")
        body = re.sub(r"\n?<!-- readings:start -->.*?<!-- readings:end -->\n?", "", body, flags=re.S)
        index.extend([f"## {chapter['title']}", "| 编号 | 知识点 / 术语 | 延伸精读 |", "|---|---|---|"])
        def insert(match):
            unit, title = match[1], match[2]
            items = mapping.get(unit, [])
            name = re.sub(r"^`?\[[^]]+\]`?\s*", "", title).replace("`", "").replace("|", "／")
            reading_links = "、".join(f"[{x['title'].replace('|', '／')}]({relative_link(INDEX, x['path'])})" for x in items)
            index.append(f"| {unit} | [{name}]({relative_link(INDEX, chapter['path'])}#{unit.lower()}) | {reading_links or '—'} |")
            if not items:
                return match[0]
            links = " · ".join(f"[{x['title']}]({relative_link(page, x['path'])})" for x in items)
            return match[0] + f"\n{START}\n**进一步精读：** {links}\n{END}\n"
        body = re.sub(r"(?m)^### ([A-S]\d{2,})\s+(.+)\n", insert, body)
        files[page] = body
    index_text = "\n\n".join(index) + "\n"
    files[INDEX] = re.sub(r"(?m)(\|[^\n]*\|)\n\n(?=\|)", r"\1\n", index_text)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    dirty = []
    for path, expected in expected_files().items():
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != expected:
            dirty.append(path.relative_to(ROOT).as_posix())
            if not args.check:
                path.write_text(expected, encoding="utf-8", newline="\n")
    if args.check and dirty:
        print("[FAIL] 导航未同步：" + ", ".join(dirty))
        return 1
    print(f"[OK] 导航{'逐字一致' if args.check else '已生成'}；{len(dirty)} 个文件需更新")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
