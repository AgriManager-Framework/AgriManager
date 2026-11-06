#!/usr/bin/env python3
"""List identity and internal-infrastructure hazards before a public release.

    python tools/release_identity_scan.py                       # tracked files, text report
    python tools/release_identity_scan.py --format md --output report.md --history
    RELEASE_FORBID_TERMS="name1,name2" python tools/release_identity_scan.py

The scan is a reviewer's aid: it lists every hit with file and line so a person
can decide what to remove, parametrize or keep. It changes nothing.

What it looks for:

* structural patterns that need no configuration: e-mail addresses,
  experiment-tracker (W&B) entities, home and cluster filesystem paths, batch
  scheduler accounts, institutional host names, GitHub and Hugging Face owners
  that are not on the upstream allow-list;
* forbidden terms supplied at run time through ``RELEASE_FORBID_TERMS``
  (comma-separated) or ``--terms-file`` (one per line, ``#`` comments).

Names of people are never written into this file: hardcoding a name in order
to detect it would publish that name with the scanner. Keep the terms file
outside the repository.

``--history`` adds the author and committer identities found in git history,
which a release that keeps the full history would publish too.

Exit status is 1 when anything is found (use ``--no-fail`` to always exit 0),
so the script can gate a release job.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Public upstream projects the code legitimately links to, and the release
# organization itself. These are organizations, not people.
UPSTREAM_GITHUB_OWNERS = {
    "agrimanager-framework", "verl-project", "volcengine", "intelligent-reliable-autonomous-systems",
    "kora-labs", "psumodeling", "ajwdewit", "spack", "pdidev", "dssat",
    "dao-ailab", "huggingface", "vllm-project", "sgl-project", "openai",
    "google", "pytorch", "nvidia", "dlr-rm", "stable-baselines-team",
    "hill-a", "astral-sh", "conda-forge", "anthropics", "farama-foundation",
    "openai-gym", "ray-project", "hydra-core", "facebookresearch",
    "squidfunk", "mkdocs", "humancompatibleai",
}
UPSTREAM_HF_OWNERS = {
    "api", "qwen", "meta-llama", "agrimanager", "agrimanager-framework", "datasets", "spaces", "docs",
    "openai", "google", "nvidia", "deepseek-ai", "mistralai",
}
EXAMPLE_EMAIL = re.compile(r"@(example\.(com|org)|users\.noreply\.github\.com|anthropic\.com)$", re.I)

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("wandb_entity", re.compile(r"wandb\.ai/([A-Za-z0-9_.-]+)")),
    ("wandb_entity", re.compile(r"(?:WANDB_ENTITY\s*=\s*|--wandb[-_]entity[= ]\s*|\bentity:\s*)([\"']?[A-Za-z0-9_.-]{2,}[\"']?)")),
    ("cluster_path", re.compile(r"(?<![\w.~-])/(?:projects|scratch|storage|lustre|gpfs|anvil|ocean|jet)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?")),
    ("cluster_path", re.compile(r"(?<![\w.~-])/work/(?:hdd|nvme|ssd)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?")),
    ("home_path", re.compile(r"(?<![\w.~-])(?:/home|/u|/Users|/data)/(?!AgriManager|agrimanager\b)[a-z][A-Za-z0-9_.-]{1,30}/")),
    ("home_path", re.compile(r"(?<![\w.~-])/tmp/(?!claude)[a-z][a-z0-9_]{1,20}/")),
    ("scheduler_account", re.compile(r"(?:--account[= ]|#SBATCH\s+-A\s+|\bsrun\s+-A\s+|\bsbatch\s+-A\s+)([A-Za-z0-9_-]+)")),
    ("scheduler_account", re.compile(r"\bdelta_b[a-z]{3}\b")),
    ("host_name", re.compile(r"\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:edu|ac\.[a-z]{2})\b")),
    ("host_name", re.compile(r"\b(?:[a-z]{1,4}-login\d*|login\d+\.[a-z]+)\b")),
    ("github_owner", re.compile(r"github\.com[/:]([A-Za-z0-9_.-]+)", re.I)),
    ("hf_owner", re.compile(r"huggingface\.co/(?:datasets/|spaces/|collections/)?([A-Za-z0-9_.-]+)", re.I)),
    ("hf_owner", re.compile(r"(?<![/\w.$-])([A-Za-z0-9][A-Za-z0-9_.-]*)/(?:schema-|t32-diffmodels|agrimanager-)[A-Za-z0-9_.-]+")),
]

TEXT_SUFFIX_SKIP = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".tar", ".pt", ".bin",
                    ".safetensors", ".parquet", ".pkl", ".npy", ".npz", ".whl", ".ico", ".svgz"}
MAX_BYTES = 5_000_000


@dataclass
class Hit:
    category: str
    value: str
    path: str
    line: int
    text: str


def repo_root(start: Path) -> Path:
    out = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    return Path(out.stdout.strip()) if out.returncode == 0 else start


def list_files(root: Path, all_files: bool) -> list[Path]:
    if not all_files:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
                              "--exclude-standard"], capture_output=True)
        if out.returncode == 0:
            return [root / p for p in out.stdout.decode().split("\0") if p]
    skip_dirs = {".git", ".cache", "node_modules", "__pycache__", "data", "results", "logs", "verl"}
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        files.extend(Path(dirpath) / f for f in filenames)
    return files


def load_terms(terms_file: str | None) -> list[str]:
    terms = [t.strip() for t in os.environ.get("RELEASE_FORBID_TERMS", "").split(",") if t.strip()]
    if terms_file:
        for raw in Path(terms_file).read_text(encoding="utf-8").splitlines():
            raw = raw.split("#", 1)[0].strip()
            if raw:
                terms.append(raw)
    return list(dict.fromkeys(terms))


def term_pattern(term: str) -> re.Pattern[str]:
    # Word-ish boundaries so a short term does not fire inside unrelated words.
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.I)


def keep(category: str, value: str) -> bool:
    v = value.strip("\"'").lower()
    if category == "email":
        return not EXAMPLE_EMAIL.search(v) and not v.startswith(("git@", "user@", "you@", "name@"))
    if category == "github_owner":
        return v not in UPSTREAM_GITHUB_OWNERS and v not in {"<org>", "todo"}
    if category == "hf_owner":
        return v not in UPSTREAM_HF_OWNERS and not v.startswith(("<", "{", "$", "todo")) and v not in {
            "your-hf-user", "owner", "org", "user", "local"}
    if category == "wandb_entity":
        # "agrimanager" is the release team on W&B (an organization, not a person).
        return v not in {"null", "none", "~", "''", '""', "${wandb_entity}", "agrimanager"}
    if category == "host_name":
        # Upstream simulator homepages cited in the docs.
        return v not in {"example.edu", "plantscience.psu.edu"}
    if category == "home_path":
        # Generic scratch folders used as defaults, not anyone's home.
        return v not in {"/tmp/dssat_configs/", "/tmp/wofost_configs/", "/home/someone/"}
    return True


def scan(root: Path, files: list[Path], terms: list[str]) -> list[Hit]:
    term_pats = [(t, term_pattern(t)) for t in terms]
    hits: list[Hit] = []
    for path in files:
        if not path.is_file() or path.is_symlink() or path.suffix.lower() in TEXT_SUFFIX_SKIP:
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        text = data.decode("utf-8", errors="replace")
        rel = str(path.relative_to(root))
        for lineno, line in enumerate(text.splitlines(), 1):
            for category, pat in PATTERNS:
                for m in pat.finditer(line):
                    value = m.group(1) if m.groups() else m.group(0)
                    if keep(category, value):
                        hits.append(Hit(category, value.strip("\"'"), rel, lineno, line.strip()))
            for term, pat in term_pats:
                if pat.search(line):
                    hits.append(Hit("forbidden_term", term, rel, lineno, line.strip()))
    return hits


def history_identities(root: Path) -> collections.Counter:
    out = subprocess.run(["git", "-C", str(root), "log", "--all", "--format=%an <%ae>%n%cn <%ce>"],
                         capture_output=True, text=True)
    return collections.Counter(l for l in out.stdout.splitlines() if l.strip())


def snippet(text: str, value: str, width: int = 150) -> str:
    i = text.lower().find(value.lower())
    if i < 0 or len(text) <= width:
        return text[:width]
    start = max(0, i - width // 3)
    return ("…" if start else "") + text[start:start + width] + ("…" if start + width < len(text) else "")


def render(hits: list[Hit], history: collections.Counter | None, fmt: str, samples: int, root: Path) -> str:
    md = fmt == "md"
    out: list[str] = []
    h = (lambda lvl, s: out.append(("#" * lvl + " " + s) if md else s.upper()))
    h(1, "Release identity scan")
    out.append("")
    out.append(f"Root: `{root.name}` · files with hits: {len({x.path for x in hits})} · hits: {len(hits)}")
    out.append("")

    by_cat = collections.Counter(x.category for x in hits)
    h(2, "Hits by category")
    out.append("")
    if md:
        out += ["| Category | Hits | Files | Distinct values (most frequent first) |", "|---|---:|---:|---|"]
    for cat, n in by_cat.most_common():
        files = len({x.path for x in hits if x.category == cat})
        vals = collections.Counter(x.value for x in hits if x.category == cat)
        top = ", ".join(f"`{v}` ({c})" for v, c in vals.most_common(12))
        more = f" … +{len(vals) - 12}" if len(vals) > 12 else ""
        out.append(f"| {cat} | {n} | {files} | {top}{more} |" if md else f"{cat}: {n} hits in {files} files; {top}{more}")
    out.append("")

    h(2, "Hits by top-level directory")
    out.append("")
    by_dir = collections.Counter(x.path.split("/", 1)[0] if "/" in x.path else "(root)" for x in hits)
    if md:
        out += ["| Directory | Hits | Files |", "|---|---:|---:|"]
    for d, n in by_dir.most_common():
        files = len({x.path for x in hits if (x.path.split("/", 1)[0] if "/" in x.path else "(root)") == d})
        out.append(f"| `{d}/` | {n} | {files} |" if md else f"{d}/: {n} hits, {files} files")
    out.append("")

    h(2, "Files with the most hits")
    out.append("")
    by_file = collections.Counter(x.path for x in hits)
    if md:
        out += ["| File | Hits | Categories |", "|---|---:|---|"]
    for p, n in by_file.most_common(40):
        cats = ", ".join(sorted({x.category for x in hits if x.path == p}))
        out.append(f"| `{p}` | {n} | {cats} |" if md else f"{p}: {n} ({cats})")
    out.append("")

    h(2, f"Samples (up to {samples} per category)")
    for cat, _ in by_cat.most_common():
        out.append("")
        h(3, cat)
        out.append("")
        seen = 0
        for x in hits:
            if x.category != cat:
                continue
            line = f"`{x.path}:{x.line}` [{x.value}] {snippet(x.text, x.value)}"
            out.append(("- " + line.replace("|", "\\|")) if md else "  " + line)
            seen += 1
            if seen >= samples:
                break
    if history is not None:
        out.append("")
        h(2, "Identities in git history (all branches)")
        out.append("")
        if md:
            out += ["| Identity | Commits (author + committer lines) |", "|---|---:|"]
        for ident, n in history.most_common():
            out.append(f"| `{ident}` | {n} |" if md else f"{ident}: {n}")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--all-files", action="store_true", help="walk the tree instead of git-tracked and untracked, non-ignored files")
    ap.add_argument("--terms-file", help="file of forbidden terms, one per line (keep it outside the repo)")
    ap.add_argument("--history", action="store_true", help="also list identities in git history")
    ap.add_argument("--format", choices=["text", "md"], default="text")
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--no-fail", action="store_true")
    args = ap.parse_args(argv)

    root = repo_root(args.root)
    hits = scan(root, list_files(root, args.all_files), load_terms(args.terms_file))
    report = render(hits, history_identities(root) if args.history else None, args.format, args.samples, root)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"{len(hits)} hits in {len({x.path for x in hits})} files; report written to {args.output}")
    else:
        sys.stdout.write(report)
    return 0 if (args.no_fail or not hits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
