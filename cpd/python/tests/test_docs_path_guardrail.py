from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[3]
GUARDED_DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "cpd" / "python" / "README.md",
    REPO_ROOT / "cpd" / "python" / "QUICKSTART.md",
    REPO_ROOT / "cpd" / "python" / "examples" / "notebooks" / "README.md",
    REPO_ROOT / "docs" / "getting-started" / "installation.md",
    REPO_ROOT / "docs" / "getting-started" / "quickstart.md",
]
LOCAL_PATH_PATTERNS = [
    re.compile(r"/Users/[^/\s]+/"),
    re.compile(r"/home/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\s]+\\\\"),
]


def test_docs_guardrail_no_machine_local_absolute_paths() -> None:
    violations: list[str] = []

    for doc_path in GUARDED_DOCS:
        assert doc_path.is_file(), f"guarded doc is missing: {doc_path}"
        for line_no, line in enumerate(doc_path.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern in LOCAL_PATH_PATTERNS:
                match = pattern.search(line)
                if match:
                    violations.append(
                        f"{doc_path.relative_to(REPO_ROOT)}:{line_no} -> {match.group(0)!r}"
                    )

    assert not violations, "Machine-local absolute path leaks found:\n" + "\n".join(violations)
