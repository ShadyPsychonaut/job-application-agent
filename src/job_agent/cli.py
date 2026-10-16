"""Command-line interface for the local job application assistant."""
from __future__ import annotations

import argparse
import json
import sys
import os
import tempfile
from pathlib import Path
from typing import Any

from .agent import answer_pending_question, list_applications, list_pending, process_job
from .scoring import score_job
from .storage import PROJECT_ROOT, ensure_data_files
from .resume_ingestion import ResumeReadError, load_resume_documents
from .profile_facts import FACTS_PATH, approve_candidate_facts, candidate_facts_summary, persist_candidate_facts
from .llm_scoring import LLMScoringError, load_local_env


def _load_job(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Job file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("Job JSON root must be an object.")
    return result


def _print_pending() -> None:
    pending = list_pending()
    if not pending:
        print("No unanswered required fields.")
        return
    print(f"\n{len(pending)} unanswered required field(s):\n")
    for question in pending:
        print(f"  {question['question_id']} | {question['company']} | {question['job_title']}")
        print(f"    Field: {question['field_label']}")
        print(f"    Why asking: {question['reason']}")
        if question.get("options"):
            print(f"    Allowed options: {', '.join(question['options'])}")
        print()


def _answer_interactively(question_id: str) -> None:
    pending = list_pending()
    question = next((q for q in pending if q["question_id"] == question_id), None)
    if question is None:
        print(f"No unanswered question found with ID {question_id}.", file=sys.stderr)
        raise SystemExit(2)

    print(f"\nCompany: {question['company']}")
    print(f"Role: {question['job_title']}")
    print(f"Question: {question['field_label']}")
    print(f"Intent: {question['intent']}")
    if question.get("options"):
        print(f"Allowed options: {', '.join(question['options'])}")
    print("\nEnter the answer exactly as you want it recorded. No answer is prefilled.")
    answer = input("Answer: ").strip()
    if not answer:
        print("No answer entered. The question remains pending.")
        return

    reuse = False
    if question["intent"] != "unknown":
        print("\nThis answer can remain specific to this application.")
        reuse_input = input(
            "Save as a reusable, user-confirmed answer for jobs matching this context? [y/N]: "
        ).strip().lower()
        reuse = reuse_input in {"y", "yes"}
        if reuse:
            context = question.get("context", {})
            print("Reuse scope will be restricted to these known context fields:")
            shown = {k: context[k] for k in ("country", "employment_type") if context.get(k)}
            print(json.dumps(shown, indent=2) if shown else "No safe scope available")
            confirm = input("Confirm saving this reusable answer? [y/N]: ").strip().lower()
            reuse = confirm in {"y", "yes"}

    result = answer_pending_question(question_id, answer, save_for_reuse=reuse)
    print("\nAnswer saved to this application.")
    print(f"Reusable answer bank updated: {'yes' if reuse else 'no'}")
    print(f"Application status: {result['application']['status']}")
    print(f"CSV log: {PROJECT_ROOT / 'data' / 'application_log.csv'}")



def _print_scorecard(job: dict[str, Any], scorecard: dict[str, Any]) -> None:
    print("\nJOB FIT SCORECARD")
    print("=" * 64)
    print(f"{job.get('company', 'Unknown company')} | {job.get('title', 'Untitled role')}")
    print(f"Overall match: {scorecard['overall_score']:.2f}/10")
    print(f"Tailoring threshold: strictly greater than {scorecard['threshold']:.2f}/10")
    print(f"Decision: {scorecard['decision']}")
    print(f"Eligibility: {scorecard['eligibility_status']} (separate from the fit score)")
    print(f"Suggested resume: {scorecard['recommended_resume_variant']}")
    print("\nComponent scores:")
    labels = {
        "technical_skills": "Technical skills",
        "responsibility_match": "Responsibilities",
        "experience_evidence": "Experience evidence",
        "seniority_match": "Seniority",
        "domain_match": "Domain relevance",
        "eligibility_match": "Eligibility",
    }
    for key, label in labels.items():
        value = scorecard["component_scores"][key]
        weight = scorecard["weights"].get(key)
        if weight is None:
            print(f"  {label:24s} {value:>4.1f}/10  (separate eligibility gate; excluded from fit score)")
        else:
            print(f"  {label:24s} {value:>4.1f}/10  (weight {weight:g}%)")
    comp = scorecard["components"]
    skill_topics = [x["topic"] for x in comp["technical_skills"].get("matched_topics", []) if x.get("profile_strength", 0) >= 0.5]
    responsibility_topics = [x["topic"] for x in comp["responsibility_match"].get("matched_topics", []) if x.get("profile_strength", 0) >= 0.5]
    if skill_topics:
        print("\nRelevant technical evidence detected: " + ", ".join(skill_topics))
    if responsibility_topics:
        print("Relevant responsibility evidence detected: " + ", ".join(responsibility_topics))
    reasons = list(dict.fromkeys(comp["experience_evidence"].get("reasons", []) + comp["seniority_match"].get("reasons", [])))
    if reasons:
        print("\nAssessment notes:")
        for reason in reasons:
            print(f"  - {reason}")
    flags = scorecard.get("gaps_and_flags", [])
    if flags:
        print("\nGaps / review flags:")
        for flag in flags:
            print(f"  - {flag}")
    print(f"\nMethod: {scorecard.get('score_method', 'unknown')}")
    print(f"LLM status: {scorecard.get('llm_status', 'unknown')}")
    resume_status = scorecard.get("resume_sources", {})
    if resume_status:
        print(f"Resume files: {', '.join(resume_status.get('filenames', [])) or 'none found'}")
        if resume_status.get("message"):
            print(f"Resume note: {resume_status['message']}")
    if scorecard.get("resume_text_shared_with_llm"):
        print("Privacy: resume text was sent to the configured LLM API for this score.")
    elif scorecard.get("llm_status") == "used":
        print("Privacy: full resume text was not sent; the model received configured profile facts only.")
    if scorecard.get("llm_error"):
        print(f"LLM fallback reason: {scorecard['llm_error']}")
    if scorecard.get("llm_summary"):
        print("\nLLM assessment: " + scorecard["llm_summary"])
    quality = scorecard.get("evidence_quality", {})
    if quality:
        print(f"Evidence confidence: {quality.get('confidence', 'unknown')} ({quality.get('confidence_score', 0):.2f})")
        print(f"Evidence review: {quality.get('note', 'No quality note available.')}")
    evidence_map = scorecard.get("evidence_map", [])
    if evidence_map:
        print("\nEvidence map:")
        for item in evidence_map[:10]:
            requirement = item.get("requirement", "Unspecified requirement")
            assessment = item.get("assessment", "unclear")
            variant = item.get("source_variant", "unspecified")
            evidence = item.get("resume_evidence", "").strip()
            print(f"  - [{assessment}] {requirement} | source: {variant}" + (f" | {evidence}" if evidence else ""))
    facts = scorecard.get("candidate_facts", {})
    if facts:
        print(f"Candidate facts: {facts.get('education_facts_detected', 0)} education fact(s), "
              f"{facts.get('years_statements_detected', 0)} experience statement(s); review status: {facts.get('review_status', 'unknown')}")
    print("This is a fit estimate, not a hiring probability.")


def _write_scored_job(path: Path, job: dict[str, Any], scorecard: dict[str, Any]) -> None:
    job["scorecard"] = scorecard
    job["overall_score"] = scorecard["overall_score"]
    job["eligibility_status"] = scorecard["eligibility_status"]
    job["resume_variant"] = scorecard["recommended_resume_variant"]
    job["scoring_decision"] = scorecard["decision"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(job, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-agent",
        description="Local job application assistant: answer reuse, missing-field prompts, and CSV tracking.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create local data files if they do not exist.")
    process = sub.add_parser("process", help="Score a job, then process fields only if it passes the threshold.")
    process.add_argument("job_json", help="Path to a job JSON file.")
    process.add_argument("--force-fields", action="store_true", help="For workflow testing only: process fields even if the job misses the score threshold.")
    process.add_argument("--mode", choices=["auto", "rules", "llm"], default="auto", help="auto uses the LLM when configured; rules disables it; llm requires an API key.")
    process.add_argument("--resume", action="append", help="Additional resume file to use (repeat for multiple files).")
    score = sub.add_parser("score", help="Score a job against the configured candidate profile.")
    score.add_argument("job_json", help="Path to a job JSON file.")
    score.add_argument("--mode", choices=["auto", "rules", "llm"], default="auto", help="auto uses the LLM when configured; rules disables it; llm requires an API key.")
    score.add_argument("--resume", action="append", help="Additional resume file to use (repeat for multiple files).")
    score.add_argument("--write", action="store_true", help="Write the scorecard and summary fields back into the input JSON.")
    sub.add_parser("resume-status", help="Discover resume files and report local cache status.")
    sub.add_parser("profile-facts", help="Extract and review local education/experience facts from the resumes.")
    sub.add_parser("approve-facts", help="Promote displayed resume facts into the local candidate profile after confirmation.")
    sub.add_parser("pending", help="List unanswered required application fields.")
    answer = sub.add_parser("answer", help="Answer one pending question interactively.")
    answer.add_argument("question_id", help="Question ID shown by the pending command.")
    sub.add_parser("applications", help="List tracked applications and their statuses.")
    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_data_files()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            (PROJECT_ROOT / "resumes").mkdir(parents=True, exist_ok=True)
            print(f"Data files ready in: {PROJECT_ROOT / 'data'}")
            print(f"Place DOCX/PDF/TXT/MD resumes in: {PROJECT_ROOT / 'resumes'}")
            print("Copy .env.example to .env and add a Groq API key if you want LLM scoring.")
            return 0
        if args.command == "resume-status":
            load_local_env()
            resume_data = load_resume_documents()
            status = resume_data["status"]
            print(status["message"])
            print(f"Resume count: {status['resume_count']}")
            print(f"Newly extracted: {status['refreshed_count']}; unchanged cache reused: {status['cached_count']}")
            print(f"Cache: {status['cache_path']}")
            for doc in resume_data["documents"]:
                print(f"  - {doc['filename']} [{doc['variant']}] sha256={doc['sha256'][:12]}…")
            return 0
        if args.command == "profile-facts":
            resume_data = load_resume_documents()
            facts = persist_candidate_facts(resume_data["documents"])
            print(candidate_facts_summary(facts))
            return 0
        if args.command == "approve-facts":
            if not FACTS_PATH.exists():
                resume_data = load_resume_documents()
                persist_candidate_facts(resume_data["documents"])
            # Show the exact extracted data before asking for explicit confirmation.
            from .storage import read_json
            facts = read_json(FACTS_PATH, {})
            print(candidate_facts_summary(facts))
            if facts.get("review_status") == "approved":
                print("These candidate facts are already marked approved.")
                return 0
            confirm = input("Approve these extracted facts into data/profile.json? [y/N]: ").strip().lower()
            if confirm not in {"y", "yes"}:
                print("No changes made to data/profile.json.")
                return 0
            approve_candidate_facts()
            print("Approved. Reviewed extracted facts were saved to data/profile.json. Work authorization was not changed.")
            return 0
        if args.command == "score":
            job_path = Path(args.job_json)
            if not job_path.is_absolute():
                job_path = Path.cwd() / job_path
            job = _load_job(job_path)
            scorecard = score_job(job, mode=args.mode, resume_paths=args.resume)
            _print_scorecard(job, scorecard)
            if args.write:
                _write_scored_job(job_path, job, scorecard)
                print(f"\nScorecard written to: {job_path}")
            return 0
        if args.command == "process":
            job_path = Path(args.job_json)
            if not job_path.is_absolute():
                job_path = Path.cwd() / job_path
            job = _load_job(job_path)
            scorecard = score_job(job, mode=args.mode, resume_paths=args.resume)
            _print_scorecard(job, scorecard)
            result = process_job(job, force_fields=args.force_fields, scorecard=scorecard)
            if result["status"] == "duplicate":
                app = result["application"]
                print(f"\nDuplicate JD URL detected. Existing record: {app['application_id']} ({app['status']})")
                if result.get("pending_question_ids"):
                    print("\nQueued missing reusable profile/eligibility questions for this existing record:")
                    for qid in result["pending_question_ids"]:
                        print(f"  python main.py answer {qid}")
                return 0
            app = result["application"]
            print(f"\nRecorded: {app['company']} | {app['designation']}")
            print(f"Application ID: {app['application_id']}")
            print(f"Status: {app['status']}")
            print(f"Pending required fields: {app['pending_required_count']}")
            if app["status"].startswith("not_shortlisted"):
                print("The role remains below the tailoring threshold. Only reusable profile/eligibility questions are queued; application-specific tailoring is not started.")
            if app["status"].startswith("blocked_eligibility"):
                print("The role has a potential eligibility conflict. Any queued questions are for clarification only, not permission to proceed.")
            if result.get("pending_question_ids"):
                print("\nAnswer these fields using the command below:")
                for qid in result["pending_question_ids"]:
                    print(f"  python main.py answer {qid}")
            print(f"\nCSV log: {PROJECT_ROOT / 'data' / 'application_log.csv'}")
            return 0
        if args.command == "pending":
            _print_pending()
            return 0
        if args.command == "answer":
            _answer_interactively(args.question_id)
            return 0
        if args.command == "applications":
            applications = list_applications()
            if not applications:
                print("No applications recorded yet.")
                return 0
            for app in applications:
                print(
                    f"{app['application_id']} | {app['company']} | {app['designation']} | "
                    f"{app['status']} | pending required: {app.get('pending_required_count', 0)}"
                )
            return 0
    except (ValueError, OSError, json.JSONDecodeError, ResumeReadError, LLMScoringError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
