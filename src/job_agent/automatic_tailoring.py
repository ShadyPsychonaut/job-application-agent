"""Automatic, gated resume tailoring triggered by the existing `process` workflow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .resume_ingestion import ResumeReadError, load_resume_documents
from .resume_tailoring import ResumeTailoringError, tailor_resume_document
from .storage import PROJECT_ROOT, load_applications, now_iso, save_applications, upsert_log

TERMINAL_APPLICATION_STATUSES = {
    "applied", "submitted", "application_submitted", "rejected",
    "withdrawn", "interviewing", "offer",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_docx_resume(
    scorecard: dict[str, Any], resume_paths: list[str] | None
) -> dict[str, Any]:
    """Choose the recommended native DOCX source; fail rather than guess."""
    resume_data = load_resume_documents(extra_paths=resume_paths)
    documents = [
        doc for doc in resume_data["documents"]
        if Path(doc["path"]).suffix.casefold() == ".docx"
    ]
    if not documents:
        raise ResumeTailoringError(
            "No DOCX master was found. Put backend_resume.docx and/or robotics_resume.docx "
            "under resumes/, add them to candidate_profile.resume_files in data/profile.json, "
            "or pass a Word file with --resume."
        )

    recommended = str(scorecard.get("recommended_resume_variant", "unspecified"))
    matches = [doc for doc in documents if doc.get("variant") == recommended]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(doc["filename"] for doc in matches)
        raise ResumeTailoringError(
            f"Multiple DOCX resumes match variant {recommended!r}: {names}. "
            "Pass only the intended Word resume with --resume."
        )
    if len(documents) == 1:
        return documents[0]

    available = ", ".join(
        f"{doc['filename']} [{doc.get('variant', 'unspecified')}]"
        for doc in documents
    )
    raise ResumeTailoringError(
        f"Could not select a DOCX for recommended variant {recommended!r}. Available: {available}. "
        "Pass the intended Word resume with --resume."
    )


def _persist_result(
    application: dict[str, Any],
    source_resume: dict[str, Any],
    tailoring_result: dict[str, Any],
) -> None:
    apps_data = load_applications()
    application_id = application["application_id"]
    stored = next(
        (item for item in apps_data.get("applications", [])
         if item.get("application_id") == application_id),
        None,
    )
    if stored is None:
        stored = application
        apps_data.setdefault("applications", []).append(stored)

    stored["tailored_resume_path"] = tailoring_result["output_path"]
    stored["tailoring_report_path"] = tailoring_result["report_path"]
    stored["tailoring_source_resume"] = source_resume["filename"]
    stored["tailoring_generated_at"] = now_iso()

    status = str(stored.get("status", "")).lower()
    if status not in TERMINAL_APPLICATION_STATUSES:
        pending_count = int(stored.get("pending_required_count", 0) or 0)
        if pending_count:
            stored["status"] = "awaiting_user_input"
            stored["next_action"] = (
                f"Answer the {pending_count} pending required field(s), then review the tailored DOCX "
                "and audit JSON before applying manually."
            )
        else:
            stored["status"] = "resume_tailoring_review_pending"
            stored["next_action"] = (
                "Review the tailored DOCX and audit JSON. Confirm all claims and formatting before applying manually."
            )

    stored["updated_at"] = now_iso()
    save_applications(apps_data)
    upsert_log(stored)
    # Update the object returned by process_job, including duplicate-URL results.
    application.update(stored)


def auto_tailor_after_process(
    *,
    job: dict[str, Any],
    scorecard: dict[str, Any],
    application: dict[str, Any],
    resume_paths: list[str] | None = None,
    consent_to_share: bool = False,
) -> dict[str, Any] | None:
    """Tailor from process only if score > threshold, eligibility is clear, and sharing is approved.

    Reuses a prior draft for the same JD URL and unchanged source resume to avoid
    paying for repeated LLM calls when a duplicate URL is processed.
    """
    score = float(scorecard.get("overall_score", 0.0))
    threshold = float(scorecard.get("threshold", 7.0))
    passes_threshold = bool(
        scorecard.get("passes_threshold", score > threshold)
    ) and score > threshold

    if scorecard.get("hard_eligibility_block"):
        print("\nResume tailoring skipped: a hard eligibility block remains.")
        return None
    if not passes_threshold:
        print(
            f"\nResume tailoring skipped: score {score:.2f} must be strictly greater than {threshold:.2f}."
        )
        return None

    if str(application.get("status", "")).casefold() in TERMINAL_APPLICATION_STATUSES:
        print("\nResume tailoring skipped: this application is already in a terminal status.")
        return None

    if not consent_to_share:
        print(
            "\nThe job passed the fit and eligibility gates, but tailoring was not run because sharing was not approved. "
            "Run `process` with --share-resume-with-llm to allow Groq to receive only the Professional Summary, "
            "existing Skills inventory, and experience-bullet text."
        )
        return None

    source_resume = _select_docx_resume(scorecard, resume_paths)
    source_path = Path(source_resume["path"]).expanduser().resolve()
    source_hash = _sha256_file(source_path)
    existing_report_path = Path(str(application.get("tailoring_report_path", "")))
    existing_output_path = Path(str(application.get("tailored_resume_path", "")))

    if existing_report_path.is_file() and existing_output_path.is_file():
        try:
            existing_report = json.loads(existing_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_report = {}
        same_url = (
            str(existing_report.get("jd_url", "")).strip().rstrip("/")
            == str(job.get("jd_url", "")).strip().rstrip("/")
        )
        if existing_report.get("source_sha256") == source_hash and same_url:
            reused = {
                "output_path": str(existing_output_path),
                "report_path": str(existing_report_path),
                "model": existing_report.get("model", "previously generated"),
                "report": existing_report,
            }
            _persist_result(application, source_resume, reused)
            print(f"\nReusing existing tailored resume: {existing_output_path}")
            print(f"Audit report: {existing_report_path}")
            return reused

    result = tailor_resume_document(
        job=job,
        scorecard=scorecard,
        resume_document=source_resume,
        output_dir=PROJECT_ROOT / "tailored_resumes",
        allow_resume_text_to_llm=True,
    )
    _persist_result(application, source_resume, result)
    print(f"\nTailored resume draft: {result['output_path']}")
    print(f"Tailoring audit report: {result['report_path']}")
    print("Review both files before using the resume; nothing has been submitted.")
    return result
