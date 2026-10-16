"""Conservative resolution of application fields from user-approved answers."""
from __future__ import annotations

from datetime import date
from typing import Any

from .storage import load_answer_bank


# Conservative starter classifier. Unknown questions always go to the user.
KEYWORD_INTENTS = {
    "expected_compensation": ["expected compensation", "expected salary", "salary expectation", "desired salary", "expected ctc", "expected pay"],
    "current_compensation": ["current compensation", "current salary", "current ctc", "current pay"],
    "work_authorization": ["authorized to work", "authorised to work", "work authorization", "work authorisation", "legally eligible to work"],
    "sponsorship": ["visa sponsorship", "require sponsorship", "sponsorship now", "future sponsorship"],
    "notice_period": ["notice period", "earliest joining", "earliest start", "available to join"],
    "relocation": ["relocate", "relocation"],
    "years_experience": ["years of experience", "how many years of experience"],
}


def classify_intent(label: str, explicit_intent: str | None = None) -> str:
    if explicit_intent and explicit_intent.strip():
        return explicit_intent.strip().lower()
    normalized = " ".join(label.lower().strip().split())
    for intent, keywords in KEYWORD_INTENTS.items():
        if any(keyword in normalized for keyword in keywords):
            return intent
    return "unknown"


def _context_matches(applicability: dict[str, Any], context: dict[str, Any]) -> bool:
    """Require every dimension recorded on an answer to be known and match."""
    if not applicability:
        return False
    for key, allowed_values in applicability.items():
        actual = context.get(key)
        if actual is None or actual == "":
            return False
        if not isinstance(allowed_values, list):
            allowed_values = [allowed_values]
        if actual not in allowed_values:
            return False
    return True


def _is_fresh(record: dict[str, Any], today: date | None = None) -> bool:
    today = today or date.today()
    verified_at = record.get("verified_at")
    expires_at = record.get("expires_at")
    if not verified_at or not expires_at:
        return False
    try:
        verified_date = date.fromisoformat(verified_at[:10])
        expiry_date = date.fromisoformat(expires_at[:10])
    except (TypeError, ValueError):
        return False
    if verified_date > today or expiry_date < today:
        return False
    return True


def resolve_answer(
    intent: str,
    context: dict[str, Any],
    bank: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a resolved answer only if exactly one approved, applicable answer exists."""
    bank = bank if bank is not None else load_answer_bank()

    # Work authorization is never reused when the job country is missing.
    if intent == "work_authorization" and not context.get("country"):
        return {
            "status": "needs_user_input",
            "reason": "Job country is missing, so a work-authorization answer cannot be safely reused.",
            "intent": intent,
        }

    matches = []
    for record in bank.get("answers", []):
        if record.get("intent") != intent or record.get("status") != "approved":
            continue
        if record.get("reuse_policy") != "only_when_applicable":
            continue
        if not record.get("answer"):
            continue
        if not _context_matches(record.get("applicability", {}), context):
            continue
        if not _is_fresh(record):
            continue
        matches.append(record)

    if len(matches) == 0:
        return {
            "status": "needs_user_input",
            "reason": "No fresh, approved answer matches this question and job context.",
            "intent": intent,
        }
    if len(matches) > 1:
        return {
            "status": "needs_user_input",
            "reason": "More than one approved answer matches; resolve the ambiguity manually.",
            "intent": intent,
        }

    record = matches[0]
    return {
        "status": "resolved",
        "answer": record["answer"],
        "answer_id": record["id"],
        "source": "approved_answer_bank",
    }
