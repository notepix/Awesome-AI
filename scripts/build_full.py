"""Build the full Markdown book from the declared source pages (no checksums)."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from content_model import ROOT, heading_slug, load_manifest

OUTPUT = ROOT / "full/AI_Encyclopedia.md"
LINK = re.compile(r"(!?\[[^\]\n]*\]\()([^\s)]+)(\))")


def prefix(path: Path) -> str:
    return "page-" + re.sub(r"[^a-z0-9]+", "-", path.relative_to(ROOT).as_posix().lower()).strip("-")


def source_paths(manifest):
    paths = manifest["front_matter"] + [c["path"] for c in manifest["chapters"]] + manifest.get("walkthroughs", []) + manifest["back_matter"]
    paths.append(manifest["reading_index"])
    return [ROOT / p for p in paths]


def render_page(path: Path, included: set[Path]) -> str:
    body = path.read_text(encoding="utf-8")
    out, counts, marker = [], {}, None
    def replace_link(match):
        target = match[2].strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            return match[0]
        local = unquote(parsed.path)
        resolved = ((ROOT / local.lstrip("/")) if local.startswith("/") else (path.parent / local)).resolve() if local else path.resolve()
        if resolved in included:
            dest = "#" + prefix(resolved)
            if parsed.fragment:
                dest += "-" + unquote(parsed.fragment)
        else:
            dest = Path(os.path.relpath(resolved, OUTPUT.parent)).as_posix()
            if parsed.query:
                dest += "?" + parsed.query
            if parsed.fragment:
                dest += "#" + parsed.fragment
        return match[1] + dest.replace(" ", "%20") + match[3]
    for line in body.splitlines():
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            if marker is None:
                marker = fence[1]
            elif fence[1][0] == marker[0] and len(fence[1]) >= len(marker):
                marker = None
            out.append(line)
            continue
        if marker is not None:
            out.append(line)
            continue
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if heading:
            slug = heading_slug(heading[1])
            n = counts.get(slug, 0)
            counts[slug] = n + 1
            anchor = slug if n == 0 else f"{slug}-{n}"
            out.extend([f'<a id="{prefix(path)}-{anchor}"></a>', ""])
        for explicit in re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', line):
            out.extend([f'<a id="{prefix(path)}-{explicit}"></a>', ""])
        out.append(LINK.sub(replace_link, line))
    return "\n".join(out).strip()


def build() -> str:
    manifest = load_manifest()
    paths = source_paths(manifest)
    included = {p.resolve() for p in paths}
    units = sum(len(c["units"]) for c in manifest["chapters"] if c["kind"] == "knowledge")
    projects = sum(len(c["units"]) for c in manifest["chapters"] if c["kind"] == "project")
    parts = [
        "# Awesome AI：系统讲义与精读导航",
        '<a id="awesome-ai-dictionary从线性回归到现代基础模型"></a>',
        "> 自动生成文件，请修改分章正文后运行 `python scripts/build_full.py`。",
        "> [仓库首页](../README.md) · [分章目录](../docs/README.md) · [论文与源码精读](../readings/README.md) · [完整实践项目](../labs/README.md)",
        f"> {units} 个知识单元，{projects} 个项目与验收单元。资料日期按各条来源记录；不将旧结论统一改为新日期。",
        '<a id="05-章节导航"></a>',
        "## 全文目录\n\n" + "\n".join(f'- [{p.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")}](#{prefix(p)})' for p in paths),
    ]
    for p in paths:
        parts.extend(["---", f'<a id="{prefix(p)}"></a>', render_page(p, included)])
    return "\n\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare text without writing")
    args = parser.parse_args()
    try:
        expected = build()
    except (OSError, ValueError, KeyError) as exc:
        print(f"[FAIL] 无法构建全文：{exc}")
        return 1
    if args.check:
        actual = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if actual != expected:
            print("[FAIL] 全文与分章不同步，请运行 python scripts/build_full.py")
            return 1
        print("[OK] 全文与源文件逐字一致（直接文本比较）")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
    print(f"[OK] 已生成 {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
