"""Shared, standard-library-only content manifest and Markdown utilities."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs" / "curriculum.json"


def load_manifest():
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(data)
    return data


def validate_manifest(data):
    """Fail early on ambiguous ownership or unsafe/duplicate source paths."""
    if data.get("version") != 1 or not data.get("chapters"):
        raise ValueError("章节清单需要 version=1 和非空 chapters")
    owners, paths = {}, set()
    source_paths = list(data.get("front_matter", [])) + list(data.get("back_matter", []))
    source_paths += list(data.get("walkthroughs", [])) + [data["reading_index"]]
    for chapter in data["chapters"]:
        if chapter.get("kind") not in {"knowledge", "project"}:
            raise ValueError(f"未知章节类型：{chapter.get('kind')}")
        if not chapter.get("title") or not chapter.get("units"):
            raise ValueError("每章需要标题和知识单元")
        source_paths.append(chapter["path"])
        for unit in chapter["units"]:
            if not re.fullmatch(r"[A-Z]\d{2,}", unit):
                raise ValueError(f"无效知识点编号：{unit}")
            if unit in owners:
                raise ValueError(f"知识点重复归属：{unit}")
            owners[unit] = chapter["path"]
    for name in source_paths:
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or p.suffix != ".md":
            raise ValueError(f"源文档路径须为仓库内 Markdown：{name}")
        if name in paths:
            raise ValueError(f"源文档重复收录：{name}")
        paths.add(name)


def heading_slug(title: str) -> str:
    title = re.sub(r"!?\[([^]]*)\]\([^)]*\)", r"\1", title)
    title = re.sub(r"<[^>]*>", "", title).lower().strip()
    title = "".join(c for c in title if not unicodedata.category(c).startswith(("P", "S")) or c in "-_")
    return re.sub(r"\s", "-", title)


def anchors(text: str) -> set[str]:
    result = set()
    counts = {}
    marker = None
    for line in text.splitlines():
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            if marker is None:
                marker = fence.group(1)
            elif fence.group(1)[0] == marker[0] and len(fence.group(1)) >= len(marker):
                marker = None
            continue
        if marker:
            continue
        result.update(re.findall(r'<a\s+(?:id|name)=[\"\']([^\"\']+)', line))
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            base = heading_slug(match.group(1))
            count = counts.get(base, 0)
            counts[base] = count + 1
            result.add(base if count == 0 else f"{base}-{count}")
    return result


def prerequisite_ids(text: str) -> set[str]:
    """Expand explicit IDs and ranges, e.g. A09–A13 or A09–13."""
    result = set(re.findall(r"\b([A-Z]\d{2,})\b", text))
    for m in re.finditer(r"\b([A-Z])(\d{2,})\s*[–—−-]\s*([A-Z])?(\d{2,})\b", text):
        prefix, first, end_prefix, last = m.groups()
        if end_prefix in (None, prefix):
            result.update(f"{prefix}{i:02d}" for i in range(int(first), int(last) + 1))
    return result


def dependency_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    done, active, stack, cycles = set(), set(), [], []
    def visit(node):
        if node in active:
            cycles.append(stack[stack.index(node):] + [node])
            return
        if node in done:
            return
        active.add(node)
        stack.append(node)
        for parent in sorted(graph.get(node, ())):
            if parent in graph:
                visit(parent)
        stack.pop()
        active.remove(node)
        done.add(node)
    for node in sorted(graph):
        visit(node)
    return cycles
