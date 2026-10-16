"""Discover and extract candidate resume files, caching by content hash.

Every scoring invocation re-checks each resume's SHA-256. Text extraction is reused
only when the file bytes are unchanged, so edited resumes automatically refresh.
Extracted text and cache metadata stay local under data/resume_cache.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .storage import PROJECT_ROOT, DATA_DIR, load_profile, read_json, write_json

CACHE_PATH = DATA_DIR / "resume_cache.json"
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


class ResumeReadError(ValueError):
    """A resume could not be read safely or completely."""


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _infer_variant(path: Path) -> str:
    name = path.stem.casefold()
    if "robot" in name or "fleet" in name or "autonomy" in name:
        return "robotics"
    if "backend" in name or "distributed" in name or "platform" in name:
        return "backend"
    if "hybrid" in name or "general" in name:
        return "hybrid"
    return "unspecified"


def _extract_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            document_xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(document_xml)
    except (OSError, KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise ResumeReadError(f"Could not read DOCX resume {path}: {exc}") from exc

    # Preserve paragraph boundaries without depending on python-docx.
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        text = "".join((node.text or "") for node in paragraph.findall(".//w:t", ns))
        if text.strip():
            paragraphs.append(text.strip())
    return "\n".join(paragraphs)


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # optional dependency, needed only for PDFs
    except ImportError as exc:
        raise ResumeReadError(
            "PDF resume support requires pypdf. Run: python3 -m pip install -r requirements.txt"
        ) from exc
    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:  # pypdf raises several format-specific exceptions
        raise ResumeReadError(f"Could not extract text from PDF resume {path}: {exc}") from exc
    text = "\n\n".join(page for page in pages if page)
    if not text.strip():
        raise ResumeReadError(
            f"No selectable text was found in {path}. It may be scanned; OCR is not enabled."
        )
    return text


def extract_resume_text(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix in {".txt", ".md"}:
        try:
            return path.read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise ResumeReadError(f"Resume text file is not UTF-8: {path}") from exc
    if suffix == ".docx":
        return _extract_docx(path).strip()
    if suffix == ".pdf":
        return _extract_pdf(path).strip()
    raise ResumeReadError(f"Unsupported resume format {suffix!r}: {path}")


def discover_resume_paths(
    extra_paths: list[str | Path] | None = None,
    profile: dict[str, Any] | None = None,
) -> list[Path]:
    """Use configured resume_files, extra CLI paths, or auto-discover under resumes/."""
    profile = profile or load_profile()
    candidate = profile.get("candidate_profile", {})
    configured = candidate.get("resume_files", [])
    path_strings: list[str | Path] = []
    if isinstance(configured, list):
        path_strings.extend(x for x in configured if isinstance(x, str) and x.strip())
    if extra_paths:
        path_strings.extend(extra_paths)

    paths: list[Path] = []
    if path_strings:
        for item in path_strings:
            candidate_path = Path(item).expanduser()
            if not candidate_path.is_absolute():
                candidate_path = PROJECT_ROOT / candidate_path
            if not candidate_path.exists():
                raise ResumeReadError(f"Configured resume file does not exist: {candidate_path}")
            paths.append(candidate_path.resolve())
    else:
        resume_dir = PROJECT_ROOT / "resumes"
        if resume_dir.exists():
            paths.extend(
                p.resolve() for p in sorted(resume_dir.iterdir())
                if p.is_file()
                and p.suffix.casefold() in SUPPORTED_SUFFIXES
                and p.name.casefold() not in {"readme.md", "readme.txt"}
                and not p.name.startswith(".")
            )

    deduplicated: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise ResumeReadError(f"Resume path is not a file: {path}")
        if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
            raise ResumeReadError(
                f"Unsupported resume file {path.name}. Supported: PDF, DOCX, TXT, MD."
            )
        deduplicated[str(path)] = path
    return list(deduplicated.values())


def load_resume_documents(
    extra_paths: list[str | Path] | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read current resume files on every run, reusing extracted text if hashes match.

    Returns {documents: [...], status: {...}}. Full extracted text is held locally only.
    """
    paths = discover_resume_paths(extra_paths=extra_paths, profile=profile)
    cache = read_json(CACHE_PATH, {"schema_version": 1, "files": {}})
    cache_files = cache.get("files", {}) if isinstance(cache, dict) else {}
    refreshed_cache: dict[str, Any] = {}
    documents: list[dict[str, Any]] = []
    refreshed_count = 0
    reused_count = 0

    for path in paths:
        path_key = str(path)
        digest = _hash_file(path)
        cached = cache_files.get(path_key, {}) if isinstance(cache_files, dict) else {}
        if cached.get("sha256") == digest and isinstance(cached.get("text"), str) and cached.get("text"):
            text = cached["text"]
            reused_count += 1
            last_indexed = cached.get("indexed_at", "")
        else:
            text = extract_resume_text(path)
            if not text:
                raise ResumeReadError(f"Resume has no extractable text: {path}")
            refreshed_count += 1
            last_indexed = datetime.now(timezone.utc).isoformat(timespec="seconds")

        variant = _infer_variant(path)
        cache_record = {
            "path": path_key,
            "filename": path.name,
            "sha256": digest,
            "variant": variant,
            "indexed_at": last_indexed,
            "text": text,
        }
        refreshed_cache[path_key] = cache_record
        documents.append({
            "filename": path.name,
            "path": path_key,
            "sha256": digest,
            "variant": variant,
            "text": text,
        })

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(CACHE_PATH, {"schema_version": 1, "files": refreshed_cache})
    return {
        "documents": documents,
        "status": {
            "resume_count": len(documents),
            "filenames": [doc["filename"] for doc in documents],
            "refreshed_count": refreshed_count,
            "cached_count": reused_count,
            "cache_path": str(CACHE_PATH),
            "message": (
                "Resumes loaded; cached text was reused for unchanged files."
                if documents else
                "No resume files found. Scoring will use the manually configured candidate profile only."
            ),
        },
    }
