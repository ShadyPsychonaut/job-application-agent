"""Core job intake and missing-answer queue logic."""
from __future__ import annotations

import re
import uuid
from datetime import date, timedelta
from typing import Any

from .resolver import classify_intent, resolve_answer
from .scoring import score_job
from .storage import (
    load_applications,
    load_pending,
    now_iso,
    save_applications,
    save_pending,
    upsert_log,
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "job"


def _context_for_field(job: dict[str, Any], field: dict[str, Any]) -> dict[str, Any]:
    context = {
        "country": job.get("country"),
        "employment_type": job.get("employment_type"),
        "location": job.get("location"),
        "company": job.get("company"),
        "job_title": job.get("title"),
    }
    context.update(field.get("context", {}))
    return {key: value for key, value in context.items() if value is not None and value != ""}


def _pending_count(application_id: str, pending_data: dict[str, Any]) -> int:
    return sum(
        1 for question in pending_data.get("questions", [])
        if question.get("application_id") == application_id
        and question.get("required")
        and question.get("status") == "awaiting_user"
    )


def _update_application_status(application: dict[str, Any], pending_data: dict[str, Any]) -> None:
    outstanding = _pending_count(application["application_id"], pending_data)
    application["pending_required_count"] = outstanding
    # Answering reusable profile questions must never turn a below-threshold job
    # into an application ready for submission.
    if application.get("score_decision") == "NOT_SHORTLISTED_BELOW_THRESHOLD":
        application["status"] = "not_shortlisted_profile_questions_pending" if outstanding else "not_shortlisted"
        application["next_action"] = (
            "Answer reusable profile questions; job remains below the tailoring threshold"
            if outstanding else "Job remains below the tailoring threshold; no application tailoring queued"
        )
    elif application.get("score_decision") == "SKIP_HARD_ELIGIBILITY_BLOCK":
        application["status"] = "blocked_eligibility_profile_questions_pending" if outstanding else "blocked_eligibility"
        application["next_action"] = (
            "Resolve the pending eligibility questions before reconsidering this role"
            if outstanding else "Role remains blocked by the eligibility gate"
        )
    else:
        application["status"] = "awaiting_user_input" if outstanding else "resume_tailoring_pending"
        application["next_action"] = (
            "Answer the pending required fields"
            if outstanding else "Required application fields are resolved; tailor and review the selected resume and optional cover letter before submitting manually"
        )
    application["updated_at"] = now_iso()


# These fields help build a reusable, user-confirmed answer bank even when a
# role is below the resume-tailoring threshold. Optional cover letters and other
# application-specific narrative fields are deliberately excluded.
PROFILE_INTAKE_INTENTS = {
    "work_authorization", "sponsorship", "relocation", "notice_period",
    "expected_compensation", "current_compensation",
}


def _queue_profile_intake_questions(job: dict[str, Any], application: dict[str, Any], pending_data: dict[str, Any]) -> list[str]:
    """Queue unanswered reusable intake fields without advancing a low-fit job."""
    app_id = application["application_id"]
    existing_for_app = [q for q in pending_data.get("questions", []) if q.get("application_id") == app_id]
    known_labels = {str(q.get("field_label", "")).strip().casefold() for q in existing_for_app}
    known_intents = {q.get("intent") for q in existing_for_app}
    # Include questions already answered for this application.
    for answer in application.get("answers", []):
        if not isinstance(answer, dict):
            continue

        label = str(answer.get("field_label", "")).strip().casefold()
        intent = answer.get("intent")

        if label:
            known_labels.add(label)

        if intent:
            known_intents.add(intent)
    new_ids: list[str] = []
    intake_fields = list(job.get("fields", []))
    # Form fields are often unavailable from job-board pages. Still collect the
    # two core eligibility facts when the job's country is known explicitly.
    country = str(job.get("country", "")).strip()
    if country:
        default_eligibility_fields = [
            {"label": "Are you legally authorized to work in the country of this job?", "intent": "work_authorization", "required": True, "type": "single_choice", "options": ["Yes", "No"], "context": {"country": country}},
            {"label": "Do you require visa sponsorship now or in the future?", "intent": "sponsorship", "required": True, "type": "single_choice", "options": ["Yes", "No"], "context": {"country": country}},
        ]
        existing_intents = {classify_intent(str(f.get("label", "")), f.get("intent")) for f in intake_fields if isinstance(f, dict)}
        intake_fields.extend(f for f in default_eligibility_fields if f["intent"] not in existing_intents)
    for field in intake_fields:
        if not isinstance(field, dict) or not field.get("label"):
            continue
        intent = classify_intent(str(field["label"]), field.get("intent"))
        if intent not in PROFILE_INTAKE_INTENTS or not bool(field.get("required", True)):
            continue
        label_key = str(field["label"]).strip().casefold()
        if label_key in known_labels or intent in known_intents:
            continue
        context = _context_for_field(job, field)
        result = resolve_answer(intent, context)
        options = field.get("options", []) or []
        if result.get("status") == "resolved" and options and result.get("answer") not in options:
            result = {"status": "needs_user_input", "reason": "The stored answer does not match this form's allowed choices.", "intent": intent}
        if result.get("status") == "resolved":
            application.setdefault("answers", []).append({
                "field_label": field["label"], "intent": intent, "answer": result["answer"],
                "source": result["source"], "answer_id": result.get("answer_id"), "required": True,
            })
            continue
        question_id = f"q_{uuid.uuid4().hex[:10]}"
        pending_data.setdefault("questions", []).append({
            "question_id": question_id, "application_id": app_id,
            "company": job["company"], "job_title": job["title"], "jd_url": str(job.get("jd_url", "")),
            "field_label": field["label"], "intent": intent, "required": True,
            "field_type": field.get("type", "text"), "options": options, "context": context,
            "status": "awaiting_user", "reason": result.get("reason", "No approved answer available."),
            "created_at": now_iso(), "answered_at": None, "answer": None,
            "reuse_approved": False, "profile_intake": True,
        })
        new_ids.append(question_id)
        known_labels.add(label_key)
        known_intents.add(intent)
    return new_ids


def _sync_scorecard_to_application(
    application: dict[str, Any],
    scorecard: dict[str, Any],
) -> None:
    """Refresh scoring fields without overwriting user-entered application data."""

    components = scorecard.get("component_scores", {}) or {}

    fields = {
        "overall_score": scorecard.get("overall_score"),
        "score_decision": scorecard.get("decision"),
        "technical_match": components.get("technical_skills"),
        "responsibility_match": components.get("responsibility_match"),
        "experience_evidence": components.get("experience_evidence"),
        "seniority_match": components.get("seniority_match"),
        "domain_match": components.get("domain_match"),
        "eligibility_match": components.get("eligibility_match"),
        "eligibility_status": scorecard.get("eligibility_status"),
        "resume_variant": scorecard.get("recommended_resume_variant"),
        "scorecard": scorecard,
    }

    for key, value in fields.items():
        if value is not None:
            application[key] = value

    application["updated_at"] = now_iso()


def _prepare_application_fields_for_existing(job: dict[str, Any], application: dict[str, Any], pending_data: dict[str, Any]) -> list[str]:
    """Prepare fields for an existing shortlisted record without duplicating answers/questions."""
    app_id = application["application_id"]
    answers = application.setdefault("answers", [])
    field_results: list[dict[str, Any]] = []
    pending_ids: list[str] = []

    app_pending = [
        q for q in pending_data.get("questions", [])
        if q.get("application_id") == app_id and q.get("status") == "awaiting_user"
    ]

    def answer_for_field(label: str, intent: str) -> dict[str, Any] | None:
        label_key = label.strip().casefold()
        # Prefer an exact field-label match. For reusable profile intents, matching
        # the intent also protects an answer collected by the earlier intake phase.
        for answer in answers:
            if isinstance(answer, dict) and str(answer.get("field_label", "")).strip().casefold() == label_key:
                return answer
        if intent in PROFILE_INTAKE_INTENTS:
            for answer in answers:
                if isinstance(answer, dict) and answer.get("intent") == intent:
                    return answer
        return None

    def question_for_field(label: str, intent: str) -> dict[str, Any] | None:
        label_key = label.strip().casefold()
        for question in app_pending:
            if str(question.get("field_label", "")).strip().casefold() == label_key:
                return question
        if intent in PROFILE_INTAKE_INTENTS:
            for question in app_pending:
                if question.get("intent") == intent:
                    return question
        return None

    for index, field in enumerate(job.get("fields", []), start=1):
        if not isinstance(field, dict) or not field.get("label"):
            raise ValueError(f"Field #{index} must be an object with a non-empty 'label'.")

        label = str(field["label"]).strip()
        required = bool(field.get("required", True))
        intent = classify_intent(label, field.get("intent"))
        existing_answer = answer_for_field(label, intent)
        if existing_answer is not None:
            field_results.append({
                "field_label": label,
                "status": "skipped_optional" if existing_answer.get("source") == "skipped_optional" else "resolved",
                "source": existing_answer.get("source", "existing_application_answer"),
            })
            continue

        existing_question = question_for_field(label, intent)
        if existing_question is not None:
            pending_ids.append(existing_question["question_id"])
            field_results.append({
                "field_label": label,
                "status": "awaiting_user_input",
                "question_id": existing_question["question_id"],
                "reason": existing_question.get("reason", "Awaiting your answer."),
            })
            continue

        context = _context_for_field(job, field)
        result = resolve_answer(intent, context)
        options = field.get("options", []) or []
        if result.get("status") == "resolved" and options and result.get("answer") not in options:
            result = {
                "status": "needs_user_input",
                "reason": "The stored answer does not match this form's allowed choices.",
                "intent": intent,
            }

        if result.get("status") == "resolved":
            answer_entry = {
                "field_label": label,
                "intent": intent,
                "answer": result.get("answer"),
                "source": result.get("source", "answer_bank"),
                "answer_id": result.get("answer_id"),
                "required": required,
            }
            answers.append(answer_entry)
            field_results.append({
                "field_label": label,
                "status": "resolved",
                "source": answer_entry["source"],
            })
            continue

        if not required:
            answers.append({
                "field_label": label,
                "intent": intent,
                "answer": None,
                "source": "skipped_optional",
                "required": False,
            })
            field_results.append({"field_label": label, "status": "skipped_optional"})
            continue

        question_id = f"q_{uuid.uuid4().hex[:10]}"
        question = {
            "question_id": question_id,
            "application_id": app_id,
            "company": job["company"],
            "job_title": job["title"],
            "jd_url": str(job.get("jd_url", "")),
            "field_label": label,
            "intent": intent,
            "required": True,
            "field_type": field.get("type", "text"),
            "options": options,
            "context": context,
            "status": "awaiting_user",
            "reason": result.get("reason", "No approved answer available."),
            "created_at": now_iso(),
            "answered_at": None,
            "answer": None,
            "reuse_approved": False,
            "profile_intake": False,
        }
        pending_data.setdefault("questions", []).append(question)
        app_pending.append(question)
        pending_ids.append(question_id)
        field_results.append({
            "field_label": label,
            "status": "awaiting_user_input",
            "question_id": question_id,
            "reason": question["reason"],
        })

    application["field_results"] = field_results
    _update_application_status(application, pending_data)
    application["next_action"] = (
        "Answer pending required application fields, then continue to resume tailoring."
        if application.get("pending_required_count", 0)
        else "Application fields are resolved. Tailor and review the backend resume and optional cover letter, then submit manually."
    )
    application["updated_at"] = now_iso()
    return pending_ids


def process_job(job: dict[str, Any], force_fields: bool = False, scorecard: dict[str, Any] | None = None) -> dict[str, Any]:
    required_keys = ["company", "title", "jd_url", "fields"]
    missing = [key for key in required_keys if key not in job]
    if missing:
        raise ValueError(f"Job JSON is missing required keys: {', '.join(missing)}")
    if not isinstance(job["fields"], list):
        raise ValueError("The job's 'fields' property must be a list.")

    jd_url = str(job["jd_url"]).strip()
    scorecard = scorecard or score_job(job)
    apps_data = load_applications()
    for existing in apps_data.get("applications", []):
        if (
            jd_url
            and existing.get("jd_url", "").strip().rstrip("/")
            == jd_url.rstrip("/")
        ):
            pending_data = load_pending()
            new_ids = []

            existing_status = str(existing.get("status", "")).lower()

            # Always refresh the stored scorecard, even if no questions are added.
            _sync_scorecard_to_application(existing, scorecard)

            new_decision = str(
                scorecard.get("decision")
                or existing.get("score_decision", "")
            ).upper()

            if (
                scorecard.get("hard_eligibility_block")
                or new_decision == "SKIP_HARD_ELIGIBILITY_BLOCK"
            ):
                new_ids = _queue_profile_intake_questions(
                    job, existing, pending_data
                )
                _update_application_status(existing, pending_data)

            elif (
                not scorecard.get("passes_threshold", False)
                or new_decision == "NOT_SHORTLISTED_BELOW_THRESHOLD"
            ):
                new_ids = _queue_profile_intake_questions(
                    job, existing, pending_data
                )
                _update_application_status(existing, pending_data)

            elif (
                existing_status == "score_eligible_needs_application_refresh"
                or existing_status.startswith("not_shortlisted")
                or existing_status.startswith("blocked_eligibility")
            ):
                new_ids = _prepare_application_fields_for_existing(
                    job, existing, pending_data
                )

            elif existing_status in {
                "processing",
                "awaiting_user_input",
                "ready_for_review",
                "resume_tailoring_pending",
            }:
                new_ids = _prepare_application_fields_for_existing(
                    job, existing, pending_data
                )

            elif not existing.get("field_results") and existing_status not in {
                "applied",
                "submitted",
                "application_submitted",
                "rejected",
                "withdrawn",
                "interviewing",
                "offer",
            }:
                new_ids = _prepare_application_fields_for_existing(
                    job, existing, pending_data
                )

            # Persist even when there are no new questions.
            save_pending(pending_data)
            save_applications(apps_data)
            upsert_log(existing)

            return {
                "status": "duplicate",
                "application": existing,
                "pending_question_ids": new_ids,
                "scorecard": scorecard,
            }

    stamp = now_iso()
    app_id = f"app_{date.today().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
    application: dict[str, Any] = {
        "application_id": app_id,
        "discovered_at": stamp,
        "created_at": stamp,
        "updated_at": stamp,
        "company": job["company"],
        "designation": job["title"],
        "title": job["title"],
        "location": job.get("location", ""),
        "country": job.get("country", ""),
        "employment_type": job.get("employment_type", ""),
        "jd_url": jd_url,
        "application_url": job.get("application_url", ""),
        "overall_score": scorecard["overall_score"],
        "score_decision": scorecard["decision"],
        "technical_match": scorecard["component_scores"]["technical_skills"],
        "responsibility_match": scorecard["component_scores"]["responsibility_match"],
        "experience_evidence": scorecard["component_scores"]["experience_evidence"],
        "seniority_match": scorecard["component_scores"]["seniority_match"],
        "domain_match": scorecard["component_scores"]["domain_match"],
        "eligibility_match": scorecard["component_scores"]["eligibility_match"],
        "eligibility_status": scorecard["eligibility_status"],
        "resume_variant": scorecard["recommended_resume_variant"],
        "scorecard": scorecard,
        "resume_path": job.get("resume_path", ""),
        "status": "processing",
        "applied_at": "",
        "pending_required_count": 0,
        "next_action": "",
        "notes": job.get("notes", ""),
        "answers": [],
        "field_results": [],
    }

    pending_data = load_pending()
    pending_ids = []
    field_results = []

    # A role below the strict threshold is logged but not advanced to application prompts.
    # --force-fields exists only to test the answer workflow, not to imply a job is suitable.
    if scorecard["hard_eligibility_block"]:
        pending_ids = _queue_profile_intake_questions(job, application, pending_data)
        application["status"] = "blocked_eligibility_profile_questions_pending" if pending_ids else "blocked_eligibility"
        application["next_action"] = "Resolve pending eligibility questions; do not proceed while a hard eligibility conflict remains" if pending_ids else "Skip unless eligibility details change"
        application["pending_required_count"] = len(pending_ids)
        application["updated_at"] = now_iso()
        apps_data.setdefault("applications", []).append(application)
        save_applications(apps_data)
        save_pending(pending_data)
        upsert_log(application)
        return {"status": application["status"], "application": application, "pending_question_ids": pending_ids, "scorecard": scorecard}
    if not scorecard["passes_threshold"] and not force_fields:
        pending_ids = _queue_profile_intake_questions(job, application, pending_data)
        application["status"] = "not_shortlisted_profile_questions_pending" if pending_ids else "not_shortlisted"
        application["next_action"] = (f"Answer reusable profile questions; job score {scorecard['overall_score']}/10 does not exceed {scorecard['threshold']}" if pending_ids else f"Score {scorecard['overall_score']}/10 does not exceed {scorecard['threshold']}; no tailoring queued")
        application["pending_required_count"] = len(pending_ids)
        application["updated_at"] = now_iso()
        apps_data.setdefault("applications", []).append(application)
        save_applications(apps_data)
        save_pending(pending_data)
        upsert_log(application)
        return {"status": application["status"], "application": application, "pending_question_ids": pending_ids, "scorecard": scorecard}

    for index, field in enumerate(job["fields"], start=1):
        if not isinstance(field, dict) or not field.get("label"):
            raise ValueError(f"Field #{index} must be an object with a non-empty 'label'.")

        required = bool(field.get("required", True))
        intent = classify_intent(str(field["label"]), field.get("intent"))
        context = _context_for_field(job, field)
        result = resolve_answer(intent, context)
        allowed_options = field.get("options", []) or []
        if (
            result["status"] == "resolved"
            and allowed_options
            and result.get("answer") not in allowed_options
        ):
            result = {
                "status": "needs_user_input",
                "reason": "The stored answer does not match this form's allowed choices.",
                "intent": intent,
            }

        if result["status"] == "resolved":
            answer_entry = {
                "field_label": field["label"],
                "intent": intent,
                "answer": result["answer"],
                "source": result["source"],
                "answer_id": result["answer_id"],
                "required": required,
            }
            application["answers"].append(answer_entry)
            field_results.append({"field_label": field["label"], "status": "resolved", "source": result["source"]})
            continue

        # Unknown optional fields can be left blank, but are recorded as such.
        if not required:
            application["answers"].append({
                "field_label": field["label"], "intent": intent, "answer": None,
                "source": "skipped_optional", "required": False,
            })
            field_results.append({"field_label": field["label"], "status": "skipped_optional"})
            continue

        question_id = f"q_{uuid.uuid4().hex[:10]}"
        question = {
            "question_id": question_id,
            "application_id": app_id,
            "company": job["company"],
            "job_title": job["title"],
            "jd_url": jd_url,
            "field_label": field["label"],
            "intent": intent,
            "required": required,
            "field_type": field.get("type", "text"),
            "options": field.get("options", []),
            "context": context,
            "status": "awaiting_user",
            "reason": result.get("reason", "No approved answer available."),
            "created_at": stamp,
            "answered_at": None,
            "answer": None,
            "reuse_approved": False,
        }
        pending_data.setdefault("questions", []).append(question)
        pending_ids.append(question_id)
        field_results.append({
            "field_label": field["label"],
            "status": "awaiting_user_input",
            "question_id": question_id,
            "reason": question["reason"],
        })

    application["field_results"] = field_results
    _update_application_status(application, pending_data)
    apps_data.setdefault("applications", []).append(application)
    save_applications(apps_data)
    save_pending(pending_data)
    upsert_log(application)

    return {"status": application["status"], "application": application, "pending_question_ids": pending_ids, "scorecard": scorecard}


def answer_pending_question(question_id: str, answer: str, save_for_reuse: bool = False) -> dict[str, Any]:
    pending_data = load_pending()
    question = next((q for q in pending_data.get("questions", []) if q.get("question_id") == question_id), None)
    if question is None:
        raise ValueError(f"Pending question '{question_id}' was not found.")
    if question.get("status") != "awaiting_user":
        raise ValueError(f"Question '{question_id}' is not waiting for an answer.")

    answer = answer.strip()
    if not answer:
        raise ValueError("Answer cannot be empty.")

    options = question.get("options") or []
    if options and answer not in options:
        raise ValueError(f"Answer must exactly match one of the allowed options: {', '.join(options)}")

    # No user response is ever treated as reusable unless they explicitly opt in.
    if save_for_reuse:
        if question["intent"] == "unknown":
            raise ValueError("Unknown question types cannot be added to the reusable answer bank yet.")
        if question["intent"] == "work_authorization" and not question.get("context", {}).get("country"):
            raise ValueError("Cannot save a work-authorization answer for reuse without a known job country.")

    answered_at = now_iso()
    question["answer"] = answer
    question["status"] = "answered"
    question["answered_at"] = answered_at
    question["reuse_approved"] = bool(save_for_reuse)

    apps_data = load_applications()
    application = next((a for a in apps_data.get("applications", []) if a.get("application_id") == question["application_id"]), None)
    if application is None:
        raise ValueError("The application associated with this question was not found.")

    application["answers"].append({
        "field_label": question["field_label"],
        "intent": question["intent"],
        "answer": answer,
        "source": "user_entered",
        "required": question["required"],
        "question_id": question_id,
    })

    if save_for_reuse:
        answer_bank = __import__("job_agent.storage", fromlist=["load_answer_bank", "save_answer_bank"])
        bank = answer_bank.load_answer_bank()
        context = question.get("context", {})
        applicability = {}
        # Keep a reusable answer scoped to the exact known country/employment type.
        for key in ("country", "employment_type"):
            if context.get(key):
                applicability[key] = [context[key]]
        if question["intent"] == "work_authorization" and "country" not in applicability:
            raise ValueError("Work-authorization reuse requires an applicability country.")
        if not applicability:
            raise ValueError("Cannot reuse this answer without at least one known applicability dimension.")
        today = date.today()
        profile = __import__("job_agent.storage", fromlist=["load_profile"]).load_profile()
        freshness_days = int(profile.get("answer_freshness_days", 90))
        record = {
            "id": f"ans_{uuid.uuid4().hex[:10]}",
            "intent": question["intent"],
            "answer": answer,
            "status": "approved",
            "applicability": applicability,
            "verified_at": today.isoformat(),
            "expires_at": (today + timedelta(days=freshness_days)).isoformat(),
            "reuse_policy": "only_when_applicable",
            "source": "user_confirmed",
            "created_at": answered_at,
        }
        bank.setdefault("answers", []).append(record)
        answer_bank.save_answer_bank(bank)

    _update_application_status(application, pending_data)
    save_pending(pending_data)
    save_applications(apps_data)
    upsert_log(application)
    return {"question": question, "application": application}


def list_pending() -> list[dict[str, Any]]:
    return [q for q in load_pending().get("questions", []) if q.get("status") == "awaiting_user"]


def list_applications() -> list[dict[str, Any]]:
    return load_applications().get("applications", [])
