import re
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]

PROHIBITED_SUFFIXES = {
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".arrow",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".duckdb",
    ".dcm",
    ".nii",
    ".pkl",
    ".pickle",
    ".joblib",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".safetensors",
}

SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def tracked_candidate_files() -> list[Path]:
    ignored_parts = {".git", ".venv", "__pycache__", ".pytest_cache"}
    return [
        path
        for path in REPOSITORY.rglob("*")
        if path.is_file() and not ignored_parts.intersection(path.parts)
    ]


def test_no_sensitive_file_types_are_present() -> None:
    violations = [
        str(path.relative_to(REPOSITORY))
        for path in tracked_candidate_files()
        if path.suffix.lower() in PROHIBITED_SUFFIXES
    ]
    assert not violations, f"Sensitive file types present: {violations}"


def test_no_common_secret_patterns_are_present() -> None:
    violations: list[str] = []
    for path in tracked_candidate_files():
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            violations.append(str(path.relative_to(REPOSITORY)))
    assert not violations, f"Potential credentials present: {violations}"
