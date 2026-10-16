"""Small, dependency-free persistence helpers for the job application agent."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROFILE_PATH = DATA_DIR / "profile.json"
ANSWER_BANK_PATH = DATA_DIR / "answer_bank.json"
PENDING_PATH = DATA_DIR / "pending_questions.json"
APPLICATIONS_PATH = DATA_DIR / "applications.json"
LOG_PATH = DATA_DIR / "application_log.csv"

LOG_FIELDS = [
    "application_id",
    "discovered_at",
    "company",
    "designation",
    "location",
    "jd_url",
    "application_url",
    "overall_score",
    "score_decision",
    "technical_match",
    "responsibility_match",
    "experience_evidence",
    "seniority_match",
    "domain_match",
    "eligibility_match",
    "eligibility_status",
    "resume_variant",
    "resume_path",
    "status",
    "applied_at",
    "pending_required_count",
    "next_action",
    "notes",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    """Atomically replace a JSON file to reduce the chance of partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_profile() -> dict[str, Any]:
    return read_json(PROFILE_PATH, {})


def load_answer_bank() -> dict[str, Any]:
    return read_json(ANSWER_BANK_PATH, {"schema_version": 1, "answers": []})


def save_answer_bank(data: dict[str, Any]) -> None:
    write_json(ANSWER_BANK_PATH, data)


def load_pending() -> dict[str, Any]:
    return read_json(PENDING_PATH, {"schema_version": 1, "questions": []})


def save_pending(data: dict[str, Any]) -> None:
    write_json(PENDING_PATH, data)


def load_applications() -> dict[str, Any]:
    return read_json(APPLICATIONS_PATH, {"schema_version": 1, "applications": []})


def save_applications(data: dict[str, Any]) -> None:
    write_json(APPLICATIONS_PATH, data)


def ensure_data_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    defaults = {
        PROFILE_PATH: {
            "schema_version": 1,
            "expected_compensation": None,
            "work_authorization_policy": "reuse_previously_verified_answer_only_when_applicable",
            "require_user_approval_before_submission": True,
            "answer_freshness_days": 90,
            "notes": "No expected compensation or work-authorization answer is prefilled. Add only user-confirmed facts.",
        },
        ANSWER_BANK_PATH: {"schema_version": 1, "answers": []},
        PENDING_PATH: {"schema_version": 1, "questions": []},
        APPLICATIONS_PATH: {"schema_version": 1, "applications": []},
    }
    for path, default in defaults.items():
        if not path.exists():
            write_json(path, default)
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=LOG_FIELDS).writeheader()


def upsert_log(application: dict[str, Any]) -> None:
    """Upsert one CSV row per application_id."""
    rows: list[dict[str, str]] = []
    if LOG_PATH.exists():
        with LOG_PATH.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    row = {field: application.get(field, "") for field in LOG_FIELDS}
    replaced = False
    for index, existing in enumerate(rows):
        if existing.get("application_id") == application.get("application_id"):
            rows[index] = {**existing, **row}
            replaced = True
            break
    if not replaced:
        rows.append(row)

    with LOG_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
