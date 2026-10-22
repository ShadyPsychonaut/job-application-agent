"""Conservative local extraction of candidate facts from resume text.

This module never calls a remote model. Extracted facts are persisted with source
filenames/hashes and a review status so the user can approve them before they are
promoted into the manually maintained profile.
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .storage import DATA_DIR, load_profile, read_json, write_json

FACTS_PATH = DATA_DIR / "candidate_facts.json"

_DEGREE_PATTERNS = [
    (re.compile(r"\bB\s*\.\s*Tech\b|\bBTech\b", re.I), "B.Tech", True),
    (
        re.compile(r"\bBachelor(?:'s)?\s+of\s+Technology\b", re.I),
        "Bachelor of Technology",
        True,
    ),
    (
        re.compile(r"\bB\s*\.\s*E\s*\.?\b|\bBEng\b|\bB\s*\.\s*Eng\b", re.I),
        "B.E./B.Eng.",
        True,
    ),
    (
        re.compile(r"\bBachelor(?:'s)?\s+of\s+Engineering\b", re.I),
        "Bachelor of Engineering",
        True,
    ),
    (re.compile(r"\bB\s*\.\s*Sc\s*\.?\b|\bBSc\b", re.I), "B.Sc.", True),
    (
        re.compile(r"\bBachelor(?:'s)?\s+of\s+Science\b", re.I),
        "Bachelor of Science",
        True,
    ),
    (re.compile(r"\bM\s*\.\s*Tech\s*\.?\b|\bMTech\b", re.I), "M.Tech", False),
    (
        re.compile(r"\bMaster(?:'s)?\s+of\s+Technology\b", re.I),
        "Master of Technology",
        False,
    ),
]


def _clean_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip(" \t|•")
        for line in text.splitlines()
        if line.strip()
    ]


def _education_section(lines: list[str]) -> list[str]:
    start = next(
        (i for i, line in enumerate(lines) if re.search(r"\beducation\b", line, re.I)),
        None,
    )
    if start is None:
        # Without an education heading, do not guess a section from arbitrary mentions.
        return []
    end = len(lines)
    section_headers = re.compile(
        r"^(?:professional experience|work experience|experience|certifications|projects|technical skills|core skills)\b",
        re.I,
    )
    for i in range(start + 1, len(lines)):
        if section_headers.search(lines[i]):
            end = i
            break
    return lines[start + 1 : end]


def _extract_education(text: str, filename: str, digest: str) -> list[dict[str, Any]]:
    lines = _clean_lines(text)
    section = _education_section(lines)
    if not section:
        return []

    degree_idx = None
    degree = None
    bachelor_level = False
    for i, line in enumerate(section):
        for pattern, label, is_bachelor in _DEGREE_PATTERNS:
            match = pattern.search(line)
            if match:
                degree_idx, degree, bachelor_level = i, label, is_bachelor
                break
        if degree_idx is not None:
            break
    if degree_idx is None or degree is None:
        return []

    degree_line = section[degree_idx]
    field = None
    field_match = re.search(
        r"(?:B\s*\.\s*Tech|BTech|Bachelor(?:'s)?\s+of\s+Technology|B\s*\.\s*E\.?|BEng|B\s*\.\s*Eng|Bachelor(?:'s)?\s+of\s+Engineering|B\s*\.\s*Sc\.?|BSc|Bachelor(?:'s)?\s+of\s+Science)\s+in\s+([^|(]+)",
        degree_line,
        re.I,
    )
    if field_match:
        field = field_match.group(1).strip(" -")

    minor = None
    minor_match = re.search(r"\bminor\s+in\s+([^)|]+)", degree_line, re.I)
    if minor_match:
        minor = minor_match.group(1).strip(" -")
        if minor.casefold() in {"it", "information technology"}:
            minor = "Information Technology"

    gpa = None
    gpa_match = re.search(
        r"\bGPA\s*[:=-]?\s*([0-9]+(?:\.[0-9]+)?\s*/\s*[0-9]+(?:\.[0-9]+)?)",
        degree_line,
        re.I,
    )
    if gpa_match:
        gpa = re.sub(r"\s+", "", gpa_match.group(1))

    institution = None
    for line in reversed(section[:degree_idx]):
        cleaned_institution = re.sub(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2}\s*[–—-]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+)?(?:19|20)\d{2}\b",
            "",
            line,
            flags=re.I,
        ).strip(" -–—|")
        if cleaned_institution and not re.search(
            r"\b(?:education|gpa|graduated|bachelor|master)\b",
            cleaned_institution,
            re.I,
        ):
            institution = cleaned_institution
            break

    # Look only around this education entry, not the experience date range elsewhere.
    nearby = section[max(0, degree_idx - 1) : min(len(section), degree_idx + 3)]
    graduation_year = None
    for line in nearby:
        range_match = re.search(r"\b(?:19|20)\d{2}\s*[–—-]\s*((?:19|20)\d{2})\b", line)
        if range_match:
            graduation_year = int(range_match.group(1))
            break
        years = re.findall(r"\b(?:19|20)\d{2}\b", line)
        if years:
            graduation_year = int(years[-1])
            break

    facts: dict[str, Any] = {
        "degree": degree,
        "field": field,
        "minor": minor,
        "institution": institution,
        "graduation_year": graduation_year,
        "gpa": gpa,
        "bachelor_level": bachelor_level,
        "source_filename": filename,
        "source_sha256": digest,
        "source_section": "Education",
        "evidence_excerpt": degree_line[:240],
        "verification_status": "extracted_needs_review",
    }
    return [facts]


def _extract_stated_years(
    text: str, filename: str, digest: str
) -> list[dict[str, Any]]:
    # Search summary/profile statements only; avoid inferring duration from every date range.
    top = "\n".join(_clean_lines(text)[:18])
    patterns = (
        r"\bwith\s+(\d+(?:\.\d+)?)\+?\s+years?\s+of\s+(?:professional\s+)?experience\b",
        r"\b(\d+(?:\.\d+)?)\+?\s+years?\s+of\s+(?:professional\s+)?experience\b",
    )
    for pattern in patterns:
        match = re.search(pattern, top, re.I)
        if match:
            return [
                {
                    "years_experience": float(match.group(1)),
                    "source_filename": filename,
                    "source_sha256": digest,
                    "source_section": "Summary",
                    "evidence_excerpt": match.group(0),
                    "verification_status": "extracted_needs_review",
                }
            ]
    return []


def extract_candidate_facts(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract supported education/experience facts and track where each came from."""
    education: list[dict[str, Any]] = []
    years: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for doc in documents:
        text = str(doc.get("text", ""))
        filename = str(doc.get("filename", "resume"))
        digest = str(doc.get("sha256", ""))
        sources.append(
            {
                "filename": filename,
                "sha256": digest,
                "variant": str(doc.get("variant", "unspecified")),
            }
        )
        education.extend(_extract_education(text, filename, digest))
        years.extend(_extract_stated_years(text, filename, digest))

    # Keep one record per distinct education profile / stated experience value.
    dedup_education: list[dict[str, Any]] = []
    seen_education: set[tuple[Any, ...]] = set()
    for item in education:
        key = (
            item.get("degree"),
            item.get("field"),
            item.get("institution"),
            item.get("graduation_year"),
        )
        if key not in seen_education:
            seen_education.add(key)
            dedup_education.append(item)
    dedup_years: list[dict[str, Any]] = []
    seen_years: set[tuple[Any, ...]] = set()
    for item in years:
        key = (item.get("years_experience"), item.get("source_filename"))
        if key not in seen_years:
            seen_years.add(key)
            dedup_years.append(item)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "review_status": "needs_review",
        "sources": sources,
        "extracted_facts": {
            "education": dedup_education,
            "years_experience": dedup_years,
        },
        "limitations": [
            "Facts are extracted locally using patterns, not independently verified credentials.",
            "Review source excerpts before approving facts into profile.json.",
            "Absence of an extracted fact does not mean the candidate lacks it.",
        ],
    }


def _norm_fact_value(value: Any) -> str:
    """Normalize a factual value while ignoring harmless formatting differences."""
    text = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _same_reviewed_claim(group: str, old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Match substantive facts across PDF/DOCX sources, ignoring provenance fields.

    A source filename/hash change alone must not invalidate a user's review. At the
    same time, changed substantive facts must be reviewed again.
    """
    if group == "years_experience":
        try:
            return float(old.get("years_experience")) == float(
                new.get("years_experience")
            )
        except (TypeError, ValueError):
            return False

    if group == "education":
        # Degree and field are the minimum identity. Where both source versions
        # provide institution/year, require those details to agree too. Optional
        # fields absent from one parser output do not alone invalidate review.
        required = ("degree", "field")
        if any(
            not _norm_fact_value(old.get(key))
            or not _norm_fact_value(new.get(key))
            or _norm_fact_value(old.get(key)) != _norm_fact_value(new.get(key))
            for key in required
        ):
            return False
        for key in ("institution", "graduation_year", "minor", "gpa", "bachelor_level"):
            old_value, new_value = old.get(key), new.get(key)
            if old_value not in (None, "") and new_value not in (None, ""):
                if _norm_fact_value(old_value) != _norm_fact_value(new_value):
                    return False
        return True
    return False


def persist_candidate_facts(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Refresh extracted facts while preserving verified, unchanged claims."""

    candidate_facts = extract_candidate_facts(documents)
    existing = read_json(FACTS_PATH, {})

    old_extracted = (
        existing.get("extracted_facts", {}) if isinstance(existing, dict) else {}
    )

    old_global_approved = (
        isinstance(existing, dict) and existing.get("review_status") == "approved"
    )

    # Recover prior approvals from profile.json if candidate_facts.json
    # was regenerated and lost its review metadata.
    profile = load_profile()
    candidate = profile.get("candidate_profile", {})

    profile_facts_approved = (
        candidate.get("profile_facts_review_status")
        == "approved_from_resume_extraction"
    )

    profile_reviewed_facts = {
        "education": (candidate.get("education", []) if profile_facts_approved else []),
        "years_experience": (
            candidate.get("resume_stated_years_experience", [])
            if profile_facts_approved
            else []
        ),
    }

    all_current_facts = []

    for group in ("education", "years_experience"):
        current_facts = candidate_facts["extracted_facts"].get(group, [])
        all_current_facts.extend(current_facts)

        previous_file_facts = (
            old_extracted.get(group, []) if isinstance(old_extracted, dict) else []
        )

        previously_approved_profile_facts = profile_reviewed_facts.get(group, [])

        for fact in current_facts:
            # First, try to preserve approval from candidate_facts.json.
            matching_previous = [
                old
                for old in previous_file_facts
                if isinstance(old, dict) and _same_reviewed_claim(group, old, fact)
            ]

            already_reviewed = any(
                old_global_approved
                or old.get("verification_status")
                in {"user_reviewed", "approved", "verified"}
                for old in matching_previous
            )

            # Next, recover approval from the profile if the old facts file
            # lost its approval metadata.
            if not already_reviewed and profile_facts_approved:
                already_reviewed = any(
                    isinstance(old, dict) and _same_reviewed_claim(group, old, fact)
                    for old in previously_approved_profile_facts
                )

            if already_reviewed:
                fact["verification_status"] = "user_reviewed"

    all_reviewed = bool(all_current_facts) and all(
        fact.get("verification_status") == "user_reviewed" for fact in all_current_facts
    )

    candidate_facts["review_status"] = "approved" if all_reviewed else "needs_review"

    if all_reviewed:
        reviewed_at = (
            existing.get("reviewed_at") if isinstance(existing, dict) else None
        )

        reviewed_at = reviewed_at or candidate.get("profile_facts_reviewed_at")

        if reviewed_at:
            candidate_facts["reviewed_at"] = reviewed_at

    write_json(FACTS_PATH, candidate_facts)
    return candidate_facts


def apply_extracted_facts(
    profile: dict[str, Any], facts: dict[str, Any]
) -> dict[str, Any]:
    """Add the extracted facts to a scoring-time copy of the profile without promoting them."""
    result = deepcopy(profile)
    candidate = result.setdefault("candidate_profile", {})
    extracted = facts.get("extracted_facts", {}) if isinstance(facts, dict) else {}
    candidate["resume_extracted_education"] = extracted.get("education", [])
    candidate["resume_extracted_years_experience"] = extracted.get(
        "years_experience", []
    )
    candidate["resume_facts_review_status"] = facts.get("review_status", "needs_review")
    return result


def approve_candidate_facts() -> dict[str, Any]:
    """Promote explicitly reviewed facts into profile.json; caller must confirm first."""
    facts = read_json(FACTS_PATH, {})
    if not isinstance(facts, dict) or not facts.get("sources"):
        raise ValueError(
            "No candidate facts have been extracted yet. Run `python3 main.py profile-facts` first."
        )
    extracted = facts.get("extracted_facts", {})
    # Explicit approval applies to the extracted statements themselves, not just
    # the top-level file. Keep their source excerpts and hashes intact.
    for group_name in ("education", "years_experience"):
        for item in extracted.get(group_name, []):
            item["verification_status"] = "user_reviewed"
    profile = load_profile()
    candidate = profile.setdefault("candidate_profile", {})
    education = extracted.get("education", [])
    if education:
        # The extracted evidence remains source-traceable in profile.json.
        candidate["education"] = education
        candidate["education_status"] = (
            "bachelor_degree_documented"
            if any(x.get("bachelor_level") for x in education)
            else "degree_documented"
        )
    stated_years = extracted.get("years_experience", [])
    if stated_years:
        # Do not replace profile values with a different extraction without explicit agreement.
        candidate["resume_stated_years_experience"] = stated_years
    candidate["profile_facts_review_status"] = "approved_from_resume_extraction"
    candidate["profile_facts_reviewed_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    write_json(DATA_DIR / "profile.json", profile)
    facts["review_status"] = "approved"
    facts["reviewed_at"] = candidate["profile_facts_reviewed_at"]
    write_json(FACTS_PATH, facts)
    return profile


def candidate_facts_summary(facts: dict[str, Any]) -> str:
    """Return human-readable output suitable for CLI review."""
    lines = [
        f"Extracted facts: {FACTS_PATH}",
        f"Review status: {facts.get('review_status', 'unknown')}",
    ]
    sources = facts.get("sources", [])
    lines.append(
        "Sources: "
        + (
            ", ".join(x.get("filename", "resume") for x in sources)
            if sources
            else "none"
        )
    )
    extracted = facts.get("extracted_facts", {})
    education = extracted.get("education", [])
    if education:
        lines.append("Education facts (needs manual review unless approved):")
        for item in education:
            detail = " ".join(x for x in [item.get("degree"), item.get("field")] if x)
            suffixes = []
            if item.get("minor"):
                suffixes.append(f"minor: {item['minor']}")
            if item.get("institution"):
                suffixes.append(f"institution: {item['institution']}")
            if item.get("graduation_year"):
                suffixes.append(f"year: {item['graduation_year']}")
            if item.get("gpa"):
                suffixes.append(f"GPA: {item['gpa']}")
            lines.append(
                "  - " + detail + (" | " + "; ".join(suffixes) if suffixes else "")
            )
            if item.get("evidence_excerpt"):
                lines.append("    Evidence: " + item["evidence_excerpt"])
    else:
        lines.append("Education facts: none found by the local parser.")
    years = extracted.get("years_experience", [])
    if years:
        lines.append("Years-of-experience statements:")
        for item in years:
            lines.append(
                f"  - {item.get('years_experience')} years stated in {item.get('source_filename')}: {item.get('evidence_excerpt')}"
            )
    for limitation in facts.get("limitations", []):
        lines.append("Note: " + limitation)
    lines.append(
        "To approve these facts into data/profile.json, run `python3 main.py approve-facts` and confirm interactively."
    )
    return "\n".join(lines)
