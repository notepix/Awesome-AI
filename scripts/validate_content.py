#!/usr/bin/env python3
"""Validate the structure and Markdown content of the Awesome-AI encyclopedia.

The validator deliberately uses only Python's standard library so the same
command works locally and in a minimal CI runner:

    python scripts/validate_content.py
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def ids(prefix: str, first: int, last: int) -> tuple[str, ...]:
    return tuple(f"{prefix}{number:02d}" for number in range(first, last + 1))


# These paths form the public content contract. Splitting B and C keeps the
# classical-AI and probabilistic/black-box additions discoverable without
# changing the stable knowledge-point identifiers.
CHAPTER_FILES: dict[Path, tuple[str, ...]] = {
    Path("docs/01-foundations/a-math-statistics-optimization.md"): ids("A", 1, 18),
    Path("docs/01-foundations/b-programming-data-experiments.md"): ids("B", 1, 6),
    Path("docs/01-foundations/b-plus-classical-ai.md"): ids("B", 7, 12),
    Path("docs/01-foundations/c-machine-learning.md"): ids("C", 1, 13),
    Path("docs/01-foundations/c-plus-probabilistic-black-box.md"): ids("C", 14, 16),
    Path("docs/01-foundations/d-deep-learning.md"): ids("D", 1, 16),
    Path("docs/02-perception-language/e-computer-vision.md"): ids("E", 1, 9),
    Path("docs/02-perception-language/f-nlp-transformers-llms.md"): ids("F", 1, 14),
    Path("docs/02-perception-language/g-generative-models.md"): ids("G", 1, 8),
    Path("docs/02-perception-language/h-multimodal-vlm.md"): ids("H", 1, 7),
    Path("docs/03-decision-specialties/i-reinforcement-learning.md"): ids("I", 1, 14),
    Path("docs/03-decision-specialties/j-graph-neural-networks.md"): ids("J", 1, 5),
    Path("docs/03-decision-specialties/k-speech-audio.md"): ids("K", 1, 5),
    Path("docs/03-decision-specialties/l-time-series.md"): ids("L", 1, 5),
    Path("docs/03-decision-specialties/m-causal-inference.md"): ids("M", 1, 5),
    Path("docs/03-decision-specialties/n-recommendation-search-retrieval.md"): ids("N", 1, 6),
    Path("docs/04-systems-agents-robotics/o-llm-posttraining-rag-agents.md"): ids("O", 1, 8),
    Path("docs/04-systems-agents-robotics/p-mlops-safety-evaluation.md"): ids("P", 1, 9),
    Path("docs/04-systems-agents-robotics/q-embodied-ai-robotics.md"): ids("Q", 1, 8),
    Path("docs/05-frontier/r-frontier-2024-2026.md"): ids("R", 1, 12),
    Path("docs/06-projects/s-projects-assessment.md"): ids("S", 1, 11),
}

SUPPORT_FILES = (
    Path("docs/README.md"),
    Path("docs/00-start/README.md"),
    Path("docs/07-resources/t-source-index.md"),
    Path("full/AI_Encyclopedia.md"),
)

EXPECTED_KNOWLEDGE = tuple(
    unit_id
    for chapter_path, chapter_ids in CHAPTER_FILES.items()
    if not chapter_path.name.startswith("s-")
    for unit_id in chapter_ids
)
EXPECTED_PROJECTS = ids("S", 1, 11)
EXPECTED_ALL = EXPECTED_KNOWLEDGE + EXPECTED_PROJECTS
EXPECTED_OWNER = {
    unit_id: path for path, chapter_ids in CHAPTER_FILES.items() for unit_id in chapter_ids
}

UNIT_HEADING_RE = re.compile(r"^###\s+([A-S]\d{2})\b(?:\s+.*)?$")
FENCE_OPEN_RE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
MARKDOWN_LINK_RE = re.compile(
    r"!?\[[^\]\n]*\]\(\s*(?P<target><[^>]+>|[^\s)]+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
REFERENCE_LINK_RE = re.compile(
    r"^[ \t]{0,3}\[[^\]\n]+\]:\s*(?P<target><[^>]+>|\S+)", re.MULTILINE
)
MALFORMED_HTTP_RE = re.compile(r"(?i)\bhttps?:/(?!/)[^\s<>)\]]*")
BARE_LATEX_RE = re.compile(
    r"\\(?:"
    r"begin|end|frac|dfrac|tfrac|sqrt|sum|prod|int|lim|operatorname|"
    r"mathrm|mathbf|mathbb|mathcal|mathit|text|left|right|cdot|times|"
    r"otimes|odot|oplus|leq|geq|neq|approx|infty|partial|nabla|"
    r"forall|exists|notin|subseteq|supseteq|subset|supset|cup|cap|"
    r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|"
    r"iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|varphi|"
    r"chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega"
    r")\b"
)

FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    "先修": re.compile(r"(?m)^[ \t]*(?:[-*]\s+)?\*\*\s*先修\s*[：:]?\s*\*\*"),
    "定义与解析": re.compile(
        r"(?m)^[ \t]*(?:[-*]\s+)?\*\*\s*定义与解析\s*[：:]?\s*\*\*"
    ),
    "公式/机制": re.compile(
        r"(?m)^[ \t]*(?:[-*]\s+)?\*\*\s*(?:公式\s*/\s*机制|机制)\s*[：:]?\s*\*\*"
    ),
    "资料": re.compile(
        r"(?m)^[ \t]*(?:[-*]\s+)?\*\*\s*(?:资料定位|精确资料|资料)\s*[：:]?\s*\*\*"
    ),
    "检测": re.compile(
        r"(?m)^[ \t]*(?:[-*]\s+)?\*\*\s*(?:检测题\s*/\s*小实验|检测\s*/\s*实验)\s*[：:]?\s*\*\*"
    ),
    "常见坑": re.compile(
        r"(?m)^[ \t]*(?:[-*]\s+)?\*\*\s*(?:常见坑|常见误区)\s*[：:]?\s*\*\*"
    ),
}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str


@dataclass(frozen=True)
class CodeBlock:
    language: str
    code: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Unit:
    unit_id: str
    path: Path
    start_line: int
    end_line: int
    body: str
    blocks: tuple[CodeBlock, ...]


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_fences(path: Path, lines: list[str]) -> tuple[list[CodeBlock], set[int], list[Finding]]:
    blocks: list[CodeBlock] = []
    fenced_lines: set[int] = set()
    findings: list[Finding] = []
    opening: tuple[str, int, str] | None = None
    code_lines: list[str] = []

    for line_number, line in enumerate(lines, start=1):
        match = FENCE_OPEN_RE.match(line)
        if opening is None:
            if not match:
                continue
            marker = match.group("fence")
            info = match.group("info").strip()
            opening = (marker, line_number, info)
            code_lines = []
            fenced_lines.add(line_number)
            continue

        marker, start_line, info = opening
        fenced_lines.add(line_number)
        stripped = line.strip()
        is_close = (
            bool(stripped)
            and set(stripped) == {marker[0]}
            and len(stripped) >= len(marker)
        )
        if is_close:
            language = info.split(maxsplit=1)[0].casefold() if info else ""
            blocks.append(
                CodeBlock(language, "\n".join(code_lines) + "\n", start_line, line_number)
            )
            opening = None
            code_lines = []
        else:
            code_lines.append(line)

    if opening is not None:
        marker, start_line, _ = opening
        findings.append(
            Finding(path, start_line, f"代码围栏 {marker!r} 没有对应的结束围栏")
        )
    return blocks, fenced_lines, findings


def parse_units(
    path: Path, lines: list[str], blocks: list[CodeBlock], fenced_lines: set[int]
) -> list[Unit]:
    headings: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines, start=1):
        if line_number in fenced_lines:
            continue
        match = UNIT_HEADING_RE.match(line)
        if match:
            headings.append((line_number, match.group(1)))

    units: list[Unit] = []
    for index, (start_line, unit_id) in enumerate(headings):
        end_line = headings[index + 1][0] - 1 if index + 1 < len(headings) else len(lines)
        unit_blocks = tuple(
            block for block in blocks if start_line < block.start_line <= end_line
        )
        units.append(
            Unit(
                unit_id=unit_id,
                path=path,
                start_line=start_line,
                end_line=end_line,
                body="\n".join(lines[start_line:end_line]),
                blocks=unit_blocks,
            )
        )
    return units


def strip_inline_code(line: str) -> str:
    """Mask inline-code spans while retaining columns for useful diagnostics."""
    output = list(line)
    index = 0
    while index < len(line):
        if line[index] != "`":
            index += 1
            continue
        run_end = index
        while run_end < len(line) and line[run_end] == "`":
            run_end += 1
        marker = line[index:run_end]
        close = line.find(marker, run_end)
        if close < 0:
            index = run_end
            continue
        for position in range(index, close + len(marker)):
            output[position] = " "
        index = close + len(marker)
    return "".join(output)


def check_math_and_latex(
    path: Path, lines: list[str], fenced_lines: set[int]
) -> list[Finding]:
    findings: list[Finding] = []
    display_math = False
    display_start = 0

    for line_number, raw_line in enumerate(lines, start=1):
        if line_number in fenced_lines:
            continue
        line = strip_inline_code(raw_line)
        visible = list(line)
        inline_math = False
        inline_start = 0
        index = 0

        while index < len(line):
            if line[index] == "\\" and index + 1 < len(line):
                if display_math or inline_math:
                    visible[index] = visible[index + 1] = " "
                index += 2
                continue
            if line.startswith("$$", index):
                visible[index] = visible[index + 1] = " "
                if not inline_math:
                    display_math = not display_math
                    if display_math:
                        display_start = line_number
                index += 2
                continue
            if line[index] == "$" and not display_math:
                visible[index] = " "
                inline_math = not inline_math
                if inline_math:
                    inline_start = index + 1
                index += 1
                continue
            if display_math or inline_math:
                visible[index] = " "
            index += 1

        if inline_math:
            findings.append(
                Finding(path, line_number, f"存在未配对的行内 $（约第 {inline_start} 列）")
            )

        outside_math = "".join(visible)
        bare = BARE_LATEX_RE.search(outside_math)
        if bare:
            findings.append(
                Finding(
                    path,
                    line_number,
                    f"LaTeX 命令 {bare.group(0)!r} 位于数学定界符之外",
                )
            )

    if display_math:
        findings.append(Finding(path, display_start, "存在未配对的块级 $$"))
    return findings


def markdown_targets(text: str) -> list[tuple[int, str]]:
    matches = list(MARKDOWN_LINK_RE.finditer(text)) + list(REFERENCE_LINK_RE.finditer(text))
    matches.sort(key=lambda item: item.start())
    targets: list[tuple[int, str]] = []
    for match in matches:
        target = match.group("target").strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1].strip()
        line = text.count("\n", 0, match.start()) + 1
        targets.append((line, target))
    return targets


def check_links(path: Path, text: str, fenced_lines: set[int]) -> list[Finding]:
    findings: list[Finding] = []
    lines = text.splitlines()

    for line_number, line in enumerate(lines, start=1):
        if line_number in fenced_lines:
            continue
        malformed = MALFORMED_HTTP_RE.search(strip_inline_code(line))
        if malformed:
            findings.append(
                Finding(path, line_number, f"HTTP(S) 地址格式错误：{malformed.group(0)!r}")
            )

    for line_number, target in markdown_targets(text):
        if line_number in fenced_lines or not target:
            continue
        if any(character.isspace() for character in target):
            findings.append(Finding(path, line_number, f"链接目标含空白字符：{target!r}"))
            continue

        parsed = urlsplit(target)
        scheme = parsed.scheme.casefold()
        if scheme in {"http", "https"}:
            if not target.startswith(("http://", "https://")) or not parsed.netloc:
                findings.append(Finding(path, line_number, f"HTTP(S) 链接格式错误：{target!r}"))
            continue
        if target.casefold().startswith(("http:", "https:")):
            findings.append(Finding(path, line_number, f"HTTP(S) 链接格式错误：{target!r}"))
            continue
        if scheme in {"mailto", "tel", "data"} or target.startswith("#"):
            continue
        if scheme:
            # Other explicit URI schemes are not local file references.
            continue

        relative_text = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if not relative_text:
            continue
        if relative_text.startswith("/"):
            resolved = ROOT / relative_text.lstrip("/")
        else:
            resolved = path.parent / relative_text
        if not resolved.exists():
            findings.append(
                Finding(path, line_number, f"相对链接指向不存在：{target!r}")
            )
    return findings


def validate() -> tuple[list[Finding], int, int, int]:
    findings: list[Finding] = []

    required = tuple(CHAPTER_FILES) + SUPPORT_FILES
    for relative_path in required:
        absolute_path = ROOT / relative_path
        if not absolute_path.is_file():
            findings.append(Finding(absolute_path, 1, "必需的章节或支持文件不存在"))

    if not DOCS.is_dir():
        findings.append(Finding(DOCS, 1, "docs 目录不存在；请先生成百科章节"))
        return findings, 0, 0, 0

    markdown_files = sorted(DOCS.rglob("*.md"))
    if not markdown_files:
        findings.append(Finding(DOCS, 1, "docs/**/*.md 未找到任何 Markdown 文件"))
        return findings, 0, 0, 0

    units: list[Unit] = []
    all_blocks = 0
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        blocks, fenced_lines, fence_findings = parse_fences(path, lines)
        all_blocks += len(blocks)
        findings.extend(fence_findings)
        findings.extend(check_math_and_latex(path, lines, fenced_lines))
        findings.extend(check_links(path, text, fenced_lines))
        units.extend(parse_units(path, lines, blocks, fenced_lines))

    occurrences: dict[str, list[Unit]] = {}
    for unit in units:
        occurrences.setdefault(unit.unit_id, []).append(unit)

    actual_ids = set(occurrences)
    expected_ids = set(EXPECTED_ALL)
    for unit_id in sorted(expected_ids - actual_ids):
        owner = ROOT / EXPECTED_OWNER[unit_id]
        findings.append(Finding(owner, 1, f"缺少编号 {unit_id}"))
    for unit_id in sorted(actual_ids - expected_ids):
        for unit in occurrences[unit_id]:
            findings.append(Finding(unit.path, unit.start_line, f"出现未规划编号 {unit_id}"))
    for unit_id, copies in sorted(occurrences.items()):
        if len(copies) > 1:
            locations = ", ".join(
                f"{display_path(copy.path)}:{copy.start_line}" for copy in copies
            )
            findings.append(
                Finding(copies[0].path, copies[0].start_line, f"编号 {unit_id} 重复：{locations}")
            )

    for relative_path, expected_ids_for_file in CHAPTER_FILES.items():
        path = ROOT / relative_path
        if not path.is_file():
            continue
        actual_order = [unit.unit_id for unit in units if unit.path.resolve() == path.resolve()]
        if actual_order != list(expected_ids_for_file):
            findings.append(
                Finding(
                    path,
                    1,
                    "章节编号不连续或顺序/归属错误；"
                    f"期望 {', '.join(expected_ids_for_file)}，实际 {', '.join(actual_order) or '无'}",
                )
            )

    knowledge_units = [unit for unit in units if unit.unit_id in EXPECTED_KNOWLEDGE]
    if len(knowledge_units) != 177:
        findings.append(
            Finding(DOCS, 1, f"知识单元总数应为 177，实际为 {len(knowledge_units)}")
        )

    for unit in knowledge_units:
        for field_name, pattern in FIELD_PATTERNS.items():
            if not pattern.search(unit.body):
                findings.append(
                    Finding(unit.path, unit.start_line, f"{unit.unit_id} 缺少字段：{field_name}")
                )

        python_blocks = [
            block for block in unit.blocks if block.language in {"python", "py"}
        ]
        if len(python_blocks) != 1:
            findings.append(
                Finding(
                    unit.path,
                    unit.start_line,
                    f"{unit.unit_id} 必须恰有一个 Python 代码块，实际为 {len(python_blocks)}",
                )
            )
            continue
        block = python_blocks[0]
        try:
            ast.parse(block.code, filename=f"{display_path(unit.path)}:{block.start_line + 1}")
        except SyntaxError as error:
            code_line = block.start_line + (error.lineno or 1)
            findings.append(
                Finding(
                    unit.path,
                    code_line,
                    f"{unit.unit_id} Python 代码语法错误：{error.msg}",
                )
            )

    return findings, len(markdown_files), len(knowledge_units), all_blocks


def main() -> int:
    findings, markdown_count, knowledge_count, block_count = validate()
    findings.sort(key=lambda item: (display_path(item.path), item.line, item.message))
    if findings:
        print(f"[FAIL] 内容校验发现 {len(findings)} 个问题：")
        for finding in findings:
            print(f"  {display_path(finding.path)}:{finding.line}: {finding.message}")
        return 1

    print(
        "[OK] 内容校验通过："
        f"{markdown_count} 个 docs Markdown 文件，"
        f"{knowledge_count} 个知识单元，11 个综合项目，{block_count} 个代码块。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
