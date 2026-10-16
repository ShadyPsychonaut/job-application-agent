"""Evidence-grounded fit scoring for the local Job Application Agent.

Role fit is scored across five weighted categories. Work authorization and sponsorship
are assessed separately as eligibility gates. The scorer supports deterministic local
rules or an optional evidence-constrained Groq assessment with a local confidence report.
"""
from __future__ import annotations

import re
import copy
import os
from typing import Any

from .storage import load_profile, now_iso
from .resume_ingestion import load_resume_documents
from .profile_facts import apply_extracted_facts, persist_candidate_facts
from .llm_scoring import LLMScoringError, load_local_env, score_with_groq
from .resolver import resolve_answer

FIT_CATEGORIES = (
    "technical_skills",
    "responsibility_match",
    "experience_evidence",
    "seniority_match",
    "domain_match",
)

# Eligibility is a separate gate. These percentages describe technical/role fit only.
DEFAULT_WEIGHTS = {
    "technical_skills": 28,
    "responsibility_match": 28,
    "experience_evidence": 22,
    "seniority_match": 11,
    "domain_match": 11,
}

# Each topic has trigger phrases in the job description and a candidate evidence key.
# The strength is deliberately evidence-based and configurable in data/profile.json.
SKILL_TOPICS: dict[str, dict[str, Any]] = {
    "cpp": {"label": "C++", "terms": ["c++", "modern c++", "c plus plus"], "weight": 1.25},
    "linux": {"label": "Linux / Unix", "terms": ["linux", "unix", "posix"], "weight": 0.85},
    "concurrency": {"label": "Concurrency / multithreading", "terms": ["multithread", "multi-thread", "concurrency", "thread-safe", "parallel processing"], "weight": 1.0},
    "distributed_systems": {"label": "Distributed systems / services", "terms": ["distributed", "microservices", "multi-tier", "high availability", "highly available", "scalable services"], "weight": 1.1},
    "object_oriented_design": {"label": "Object-oriented design / patterns", "terms": ["object-oriented", "object oriented", "design patterns", "oop"], "weight": 0.75},
    "system_design": {"label": "System design / architecture", "terms": ["system design", "software architecture", "architecture", "design trade-offs", "design tradeoffs", "technical design"], "weight": 0.8},
    "database": {"label": "Databases", "terms": ["database", "mongodb", "postgresql", "postgres", "sql", "elasticsearch", "data store"], "weight": 0.65},
    "messaging": {"label": "Messaging / event-driven systems", "terms": ["rabbitmq", "kafka", "message queue", "messaging", "event-driven", "event driven", "pub/sub"], "weight": 0.7},
    "containers": {"label": "Docker / Kubernetes", "terms": ["docker", "kubernetes", "containerized", "containerised", "containers"], "weight": 0.6},
    "observability": {"label": "Monitoring / observability", "terms": ["monitoring", "observability", "metrics", "dashboards", "alarms", "logging", "on-call", "on call"], "weight": 0.6},
    "apis": {"label": "APIs / service interfaces", "terms": ["rest api", "restful", "api", "service interface", "http endpoint"], "weight": 0.55},
    "testing_ci": {"label": "Testing / build / CI", "terms": ["unit test", "testing", "test automation", "continuous integration", "ci/cd", "build process", "code review", "source control"], "weight": 0.7},
    "performance": {"label": "Performance / latency", "terms": ["low-latency", "low latency", "high-throughput", "high throughput", "performance-critical", "performance critical", "profiling", "latency"], "weight": 0.8},
    "robotics": {"label": "Robotics / mobile robots", "terms": ["robotics", "robotic", "autonomous robot", "mobile robot", "amr", "agv", "automated guided vehicle"], "weight": 1.0},
    "ros2": {"label": "ROS 2 / Nav2", "terms": ["ros 2", "ros2", "nav2", "robot operating system"], "weight": 0.65},
    "fleet_management": {"label": "Robot fleet / task orchestration", "terms": ["fleet management", "fleet manager", "task allocation", "task scheduling", "mission dispatch", "multi-robot", "multi robot", "traffic management"], "weight": 1.0},
    "simulation": {"label": "Simulation / visualization tooling", "terms": ["simulation", "simulator", "gazebo", "isaac sim", "unity", "visualization", "visualisation"], "weight": 0.75},
    "autonomy_algorithms": {"label": "Autonomy / planning algorithms", "terms": ["slam", "path planning", "motion planning", "trajectory planning", "localization", "localisation", "perception", "sensor fusion"], "weight": 0.85},
    "cloud": {"label": "Cloud platforms", "terms": ["aws", "amazon web services", "gcp", "google cloud", "azure", "cloud infrastructure"], "weight": 0.45},
    "java": {"label": "Java", "terms": ["java", "spring boot"], "weight": 0.5},
}

RESPONSIBILITY_TOPICS: dict[str, dict[str, Any]] = {
    "backend_delivery": {"label": "Backend/service development", "terms": ["backend", "services", "service development", "distributed applications", "software development", "implement", "build and operate"], "weight": 1.0},
    "ownership_lifecycle": {"label": "End-to-end ownership", "terms": ["end-to-end", "end to end", "own features", "feature ownership", "requirements, design, implementation", "full lifecycle"], "weight": 1.0},
    "architecture_scaling": {"label": "Architecture / scale / reliability", "terms": ["architecture", "scaling", "scalable", "reliability", "high availability", "design discussions", "technical roadmap"], "weight": 0.9},
    "production_operations": {"label": "Production operations / reliability", "terms": ["production", "on-call", "on call", "operational excellence", "incident", "alarms", "defect reduction", "rollout", "canary"], "weight": 0.9},
    "cross_team": {"label": "Cross-functional collaboration", "terms": ["partner teams", "cross-functional", "cross functional", "collaborate", "stakeholders", "communicate progress", "align with"], "weight": 0.7},
    "mentoring_leadership": {"label": "Technical leadership / mentoring", "terms": ["mentor", "mentoring", "lead design", "technical lead", "unblock junior", "coach", "lead engineers"], "weight": 0.8},
    "testing_release": {"label": "Testing / safe delivery", "terms": ["testing", "test coverage", "safe rollout", "canary", "deployment", "release", "code review", "defect"], "weight": 0.7},
    "fleet_task_orchestration": {"label": "Fleet / task orchestration", "terms": ["fleet management", "task allocation", "task scheduling", "mission dispatch", "robot traffic", "multi-robot"], "weight": 1.0},
    "simulation_tooling": {"label": "Simulation / developer tooling", "terms": ["simulation", "simulator", "developer tooling", "debugging tools", "visualization", "replayer"], "weight": 0.8},
    "robotics_integration": {"label": "Robotics integration", "terms": ["robotics", "robot", "amr", "agv", "ros 2", "ros2", "navigation stack", "sensor"], "weight": 0.9},
    "performance_tuning": {"label": "Performance optimization", "terms": ["low-latency", "low latency", "performance", "optimize", "optimise", "profiling", "throughput"], "weight": 0.8},
}

DOMAIN_TOPICS: dict[str, dict[str, Any]] = {
    "robotics": {"label": "Robotics / autonomy", "terms": ["robotics", "robot", "amr", "agv", "autonomy", "ros2", "ros 2", "navigation", "fleet management"]},
    "backend_platform": {"label": "Backend / platform / distributed services", "terms": ["backend", "distributed", "services", "platform", "api", "microservices", "high availability", "scalable"]},
    "logistics": {"label": "Logistics / supply chain", "terms": ["logistics", "delivery", "transportation", "supply chain", "warehouse", "fulfillment", "fulfilment", "order promise"]},
    "ecommerce": {"label": "E-commerce / consumer products", "terms": ["e-commerce", "ecommerce", "retail", "orders", "marketplace", "customer experience"]},
    "embedded": {"label": "Embedded / device systems", "terms": ["embedded", "firmware", "device driver", "real-time", "real time"]},
    "trading": {"label": "Trading / financial infrastructure", "terms": ["trading", "exchange", "market data", "order book", "low-latency trading"]},
    "healthcare": {"label": "Healthcare / medical devices", "terms": ["healthcare", "medical device", "surgical", "clinical"]},
}


def _norm(text: str) -> str:
    text = text.casefold().replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text)


def _contains(text: str, phrase: str) -> bool:
    """Case-insensitive phrase check with word boundaries where practical."""
    text_n = _norm(text)
    phrase_n = _norm(phrase)
    if phrase_n in {"c++", "c#", "oop", "sql", "amr", "agv", "api", "ros2", "ros 2", "aws", "gcp", "java"}:
        return re.search(r"(?<![a-z0-9])" + re.escape(phrase_n) + r"(?![a-z0-9])", text_n) is not None
    return phrase_n in text_n


def _job_text(job: dict[str, Any]) -> str:
    parts: list[str] = [str(job.get("title", "")), str(job.get("location", ""))]
    jd = job.get("job_description", {})
    if isinstance(jd, dict):
        for key, value in jd.items():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(item) for item in value)
            elif isinstance(value, dict):
                parts.extend(str(v) for v in value.values())
    elif isinstance(jd, str):
        parts.append(jd)
    for key in ("description", "responsibilities", "qualifications", "requirements", "seniority_level"):
        value = job.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return "\n".join(parts)


def _matched_topics(text: str, catalog: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    found = []
    for key, item in catalog.items():
        hits = [term for term in item["terms"] if _contains(text, term)]
        if hits:
            found.append((key, {**item, "hits": hits}))
    return found


def _weighted_strength_score(
    matches: list[tuple[str, dict[str, Any]]],
    evidence_strength: dict[str, Any],
    *,
    fallback: float = 5.5,
) -> tuple[float, list[dict[str, Any]], list[str]]:
    if not matches:
        return fallback, [], ["The JD did not expose enough recognizable signals for this category."]
    numerator = 0.0
    denominator = 0.0
    details: list[dict[str, Any]] = []
    gaps: list[str] = []
    for key, topic in matches:
        weight = float(topic.get("weight", 1.0))
        strength_raw = evidence_strength.get(key)
        strength = float(strength_raw) if strength_raw is not None else 0.25
        strength = min(1.0, max(0.0, strength))
        numerator += weight * strength
        denominator += weight
        details.append({
            "topic": topic["label"],
            "profile_strength": round(strength, 2),
            "weighted_contribution": round(weight * strength, 3),
            "job_signals": topic["hits"][:5],
        })
        if strength < 0.5:
            gaps.append(f"Limited documented evidence for {topic['label']}")
    return round(10 * numerator / denominator, 2), details, gaps


def _extract_required_years(job: dict[str, Any], text: str) -> float | None:
    # Prefer the explicit basic qualifications block if the input provides it.
    jd = job.get("job_description", {})
    qual = ""
    if isinstance(jd, dict):
        for key in ("basic_qualifications", "requirements", "qualifications", "preferred_qualifications"):
            val = jd.get(key)
            if isinstance(val, list):
                qual += " " + " ".join(str(x) for x in val)
            elif isinstance(val, str):
                qual += " " + val
    scan = qual or text
    patterns = [
        r"at least\s+(\d+(?:\.\d+)?)\s*\+?\s+years?",
        r"minimum of\s+(\d+(?:\.\d+)?)\s*\+?\s+years?",
        r"(\d+(?:\.\d+)?)\s*\+\s+years?",
        r"(\d+(?:\.\d+)?)\s+years? of (?:non-internship |professional |hands-on )?(?:professional )?(?:software|industry|work|development|design|engineering|relevant|experience)",
    ]
    values: list[float] = []
    for pattern in patterns:
        values.extend(float(m.group(1)) for m in re.finditer(pattern, scan, re.I))
    return max(values) if values else None


def _seniority_score(job: dict[str, Any], text: str, candidate_years: float) -> tuple[float, list[str]]:
    title = _norm(str(job.get("title", "")))
    source = job.get("source", {}) if isinstance(job.get("source"), dict) else {}
    level = _norm(str(source.get("seniority_level", "")))
    combined = title + " " + level
    reasons: list[str] = []
    if any(x in combined for x in ("staff", "principal", "director", "distinguished", "head of")):
        score = 4.0
        reasons.append("Role appears staff/principal/director level, above the configured individual-contributor profile.")
    elif any(x in combined for x in ("senior", "sr.")) and not any(x in combined for x in ("sde ii", "software development engineer ii", "level 2")):
        score = 6.0
        reasons.append("Senior title may expect deeper ownership than the documented 3-year profile.")
    elif any(x in combined for x in ("sde ii", "software development engineer ii", "mid-senior", "mid senior", "engineer ii", "level 2")):
        score = 7.2
        reasons.append("Role appears mid-level; broadly plausible, with some ownership/mentoring expectations to verify.")
    elif any(x in combined for x in ("junior", "entry level", "entry-level", "associate", "sde i", "engineer i", "graduate")):
        score = 8.7
        reasons.append("Role appears junior/entry level, within or below the configured experience band.")
    else:
        score = 6.5
        reasons.append("Seniority signal is ambiguous; a neutral-to-moderate score is used.")
    required_years = _extract_required_years(job, text)
    if required_years is not None:
        if candidate_years < required_years:
            penalty = min(4.0, (required_years - candidate_years) * 1.2)
            score -= penalty
            reasons.append(f"JD states up to {required_years:g}+ years; profile records {candidate_years:g} years.")
    return round(min(10.0, max(0.0, score)), 2), reasons


def _experience_score(job: dict[str, Any], text: str, profile: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    candidate = profile.get("candidate_profile", {})
    years = float(candidate.get("years_experience", 3.0))
    required = _extract_required_years(job, text)
    reasons: list[str] = []
    flags: list[str] = []
    if required is None:
        years_score = 6.5
        reasons.append("The JD does not state a clearly extractable minimum years requirement.")
    elif required <= 0:
        years_score = 6.5
    else:
        years_score = min(10.0, years / required * 10.0)
        reasons.append(f"Years-of-experience fit: {years:g} recorded vs {required:g} extracted minimum.")
        if years < required:
            flags.append(f"Potential experience gap: JD minimum is {required:g} years; profile records {years:g}.")

    # Architecture, ownership and mentoring requirements are checked against declared strengths.
    strengths = candidate.get("experience_strength", candidate.get("evidence_strength", {}))
    req_strengths: list[tuple[float, float, str]] = []
    expectations = [
        ("architecture", ["architecture", "design patterns", "system design", "design discussions"], "system_design", 1.0),
        ("ownership", ["own features", "end-to-end", "end to end", "own the", "feature ownership"], "ownership", 1.0),
        ("mentoring", ["mentor", "mentoring", "unblock junior", "lead engineers"], "mentoring", 0.7),
        ("production operations", ["on-call", "on call", "operate", "operational excellence", "production operations"], "production_operations", 0.7),
    ]
    for label, terms, strength_key, weight in expectations:
        hits = [term for term in terms if _contains(text, term)]
        if hits:
            strength = float(strengths.get(strength_key, 0.25))
            req_strengths.append((strength, weight, label))
            if strength < 0.5:
                flags.append(f"Review experience evidence for {label}; the JD emphasizes it.")
    if req_strengths:
        role_score = 10 * sum(s * w for s, w, _ in req_strengths) / sum(w for _, w, _ in req_strengths)
    else:
        role_score = 7.0
        reasons.append("No strong ownership/leadership-specific requirement was detected.")
    score = 0.7 * years_score + 0.3 * role_score

    # Degree facts are extracted locally. Their own review status determines
    # whether they are pending review, not the profile's education_status field.
    degree_required = any(_contains(text, phrase)
        for phrase in (
            "bachelor's degree",
            "bachelor’s degree",
            "bachelor degree",
            "bachelor's in",
            "bachelor of",
        )
    )

    if degree_required:
        education = candidate.get("education_status", "unknown")
        extracted = (
            candidate.get("resume_extracted_education", [])
            or candidate.get("education", [])
        )

        bachelor_facts = [
            fact
            for fact in extracted
            if isinstance(fact, dict) and fact.get("bachelor_level")
        ]

        if bachelor_facts:
            fact = bachelor_facts[0]
            degree_label = " ".join(
                str(value)
                for value in (fact.get("degree"), fact.get("field"))
                if value
            )

            fact_status = str(
                fact.get("verification_status", "extracted_needs_review")
            ).lower()
            review_status = str(
                candidate.get("resume_facts_review_status", "needs_review")
            ).lower()

            reviewed = (
                fact_status in {"user_reviewed", "verified", "approved"}
                or review_status == "approved"
            )

            if reviewed:
                flags.append(
                    f"Resume text lists {degree_label or 'a bachelor-level degree'}; "
                    "the extracted education evidence has been reviewed."
                )
            else:
                flags.append(
                    f"Resume text lists {degree_label or 'a bachelor-level degree'}; "
                    "the fact was locally extracted and is pending review."
                )

        elif education in {"unknown", "extracted_needs_review"}:
            flags.append(
                "JD includes a bachelor's degree requirement, but no degree fact "
                "was detected in the available resume text; verify manually."
            )
    return round(min(10.0, max(0.0, score)), 2), reasons, flags


def _domain_score(text: str, profile: dict[str, Any]) -> tuple[float, list[dict[str, Any]], str]:
    matches = _matched_topics(text, DOMAIN_TOPICS)
    strengths = profile.get("candidate_profile", {}).get("domain_strength", {})
    score, details, _ = _weighted_strength_score(matches, strengths, fallback=6.0)
    # Pick a resume family from the strongest recognized domain signals, not a title guess alone.
    domains = {key: len(item["hits"]) for key, item in matches}
    robotics_count = domains.get("robotics", 0)
    backend_count = domains.get("backend_platform", 0)
    if robotics_count > backend_count:
        resume = "robotics"
    elif backend_count > robotics_count:
        resume = "backend"
    elif robotics_count and backend_count:
        resume = "hybrid"
    else:
        resume = str(profile.get("candidate_profile", {}).get("default_resume_variant", "backend"))
    return score, details, resume


def _yes_no_answer(value: Any) -> bool | None:
    """Normalize an explicit yes/no answer without guessing."""

    normalized = re.sub(
        r"\s+",
        " ",
        str(value or "").strip().casefold(),
    )

    if normalized in {
        "yes", "y", "true", "authorized", "authorised",
        "i am authorized", "i am authorised",
    }:
        return True

    if normalized in {
        "no", "n", "false", "not authorized", "not authorised",
        "i am not authorized", "i am not authorised",
    }:
        return False

    return None


def _eligibility_assessment(job: dict[str, Any], text: str, profile: dict[str, Any]) -> dict[str, Any]:
    candidate = profile.get("candidate_profile", {})
    eligibility = candidate.get("eligibility", {})

    # Use the raw country value for answer-bank matching.
    raw_country = str(job.get("country", "")).strip()
    country = raw_country.upper()

    profile_work_auth = eligibility.get(
        "work_authorization_by_country", {}
    ).get(country, "unknown")

    work_auth = profile_work_auth

    answer_context = {"country": raw_country}

    if job.get("employment_type"):
        answer_context["employment_type"] = job["employment_type"]

    # Each answer is resolved independently using the existing validation rules.
    auth_result = resolve_answer(
        "work_authorization", answer_context
    )
    sponsorship_result = resolve_answer(
        "sponsorship", answer_context
    )

    auth_value = (
        _yes_no_answer(auth_result.get("answer"))
        if auth_result.get("status") == "resolved"
        else None
    )

    sponsorship_value = (
        _yes_no_answer(sponsorship_result.get("answer"))
        if sponsorship_result.get("status") == "resolved"
        else None
    )

    work_auth_source = "profile"

    # A confirmed sponsorship requirement must not be erased by a
    # separate answer saying the candidate is currently authorized.
    if sponsorship_value is True:
        work_auth = "requires_sponsorship"
        work_auth_source = "approved_answer_bank:sponsorship"

    elif sponsorship_value is False and auth_value is not None:
        work_auth = "authorized" if auth_value else "not_authorized"
        work_auth_source = "approved_answer_bank"

    elif (
        sponsorship_value is False
        and profile_work_auth in {
            "requires_sponsorship",
            "needs_sponsorship",
        }
    ):
        # The answer bank and profile conflict. Require verification.
        work_auth = "unknown"
        work_auth_source = "conflicting_profile_and_answer_bank"

    elif auth_value is not None:
        if profile_work_auth in {
            "requires_sponsorship",
            "needs_sponsorship",
        }:
            # Being authorized now does not establish that sponsorship
            # will not be required in the future.
            work_auth = profile_work_auth
        else:
            work_auth = "authorized" if auth_value else "not_authorized"
            work_auth_source = "approved_answer_bank"

    preferred = {
        str(x).upper()
        for x in eligibility.get("preferred_countries", ["IN"])
    }
    explicit_no_sponsor = any(_contains(text, phrase) for phrase in (
        "unable to sponsor", "will not sponsor", "cannot sponsor", "cannot provide sponsorship",
        "do not provide visa sponsorship", "do not provide sponsorship", "will not provide visa sponsorship",
        "no visa sponsorship", "does not provide sponsorship", "without sponsorship",
        "must not require sponsorship", "not able to sponsor",
    ))
    asks_for_authorization = any(_contains(text, phrase) for phrase in (
        "authorized to work", "authorised to work", "work authorization", "work authorisation",
        "legally eligible to work", "require sponsorship", "visa sponsorship",
    ))

    if country == "US" and work_auth == "requires_sponsorship" and explicit_no_sponsor:
        return {
            "score": 0.0,
            "status": "blocked",
            "hard_blocked": True,
            "country": country,
            "reason": "Profile indicates US sponsorship is needed, and the JD appears to prohibit sponsorship.",
            "flags": ["Do not apply unless the role's sponsorship wording is verified to allow your situation."],
        }

    if work_auth in {"authorized", "authorised"}:
        score, status, reason = 10.0, "eligible_known", "Previously verified work authorization matches this country."
    elif work_auth in {"requires_sponsorship", "needs_sponsorship"}:
        score, status, reason = 4.0, "needs_sponsorship_review", "Previously recorded profile says sponsorship is needed for this country."
        if explicit_no_sponsor:
            score, status = 0.0, "blocked"
            reason = "The profile indicates sponsorship is needed and the JD appears to prohibit it."
            return {"score": score, "status": status, "hard_blocked": True, "country": country, "reason": reason, "flags": [reason]}
    elif work_auth in {"not_authorized", "not_authorised"}:
        score, status, reason = 2.0, "needs_sponsorship_review", "Profile says the candidate is not currently authorized for this country."
    elif country and country in preferred:
        score, status, reason = 6.5, "needs_review", "Location is within the configured target countries, but legal work authorization is not confirmed."
    elif country:
        score, status, reason = 5.0, "needs_review", "Work authorization for this country is not confirmed in the profile."
    else:
        score, status, reason = 4.5, "needs_review", "The job country is missing, so work-authorization applicability cannot be assessed."

    flags: list[str] = []
    if asks_for_authorization and work_auth == "unknown":
        flags.append("JD mentions work authorization/sponsorship; answer manually or configure a verified country-specific answer.")
    if status == "needs_review":
        flags.append("Eligibility is uncertain. Do not infer legal work authorization from the job location or residence alone.")
    return {"score": score, "status": status, "hard_blocked": False, "country": country, "work_authorization_source" : work_auth_source, "reason": reason, "flags": flags}


def _score_job_rules(job: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Deterministic first pass; the eligibility diagnostic is deliberately excluded from fit."""
    cfg = profile.get("job_scoring", {})
    threshold = float(cfg.get("tailoring_threshold", 8.5))
    configured_weights = cfg.get("weights", DEFAULT_WEIGHTS)
    weights = {key: float(configured_weights[key]) for key in FIT_CATEGORIES if key in configured_weights}
    if set(weights) != set(FIT_CATEGORIES):
        weights = dict(DEFAULT_WEIGHTS)
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("Job scoring fit weights must sum to a positive number.")
    # Migrate an older profile's five fit weights (that used to total 90 alongside eligibility).
    if abs(weight_sum - 100.0) > 0.001:
        weights = {key: value * 100.0 / weight_sum for key, value in weights.items()}

    candidate = profile.get("candidate_profile", {})
    evidence = candidate.get("evidence_strength", {})
    text = _job_text(job)

    tech_matches = _matched_topics(text, SKILL_TOPICS)
    tech_score, tech_details, tech_gaps = _weighted_strength_score(tech_matches, evidence, fallback=5.5)

    resp_matches = _matched_topics(text, RESPONSIBILITY_TOPICS)
    responsibility_strength = candidate.get("responsibility_strength", {})
    responsibility_score, resp_details, resp_gaps = _weighted_strength_score(resp_matches, responsibility_strength, fallback=5.5)

    experience_score, experience_reasons, experience_flags = _experience_score(job, text, profile)
    years = float(candidate.get("years_experience", 3.0))
    seniority_score, seniority_reasons = _seniority_score(job, text, years)
    domain_score, domain_details, resume_variant = _domain_score(text, profile)
    eligibility = _eligibility_assessment(job, text, profile)

    fit_scores = {
        "technical_skills": tech_score,
        "responsibility_match": responsibility_score,
        "experience_evidence": experience_score,
        "seniority_match": seniority_score,
        "domain_match": domain_score,
    }
    # Re-normalize by fit weights only. Eligibility must never dilute or inflate fit.
    overall = round(sum(fit_scores[key] * weights[key] for key in FIT_CATEGORIES) / 100.0, 2)

    if eligibility["hard_blocked"]:
        decision = "SKIP_HARD_ELIGIBILITY_BLOCK"
    elif overall > threshold:
        decision = "TAILOR_RESUME_ELIGIBILITY_REVIEW" if eligibility["status"] != "eligible_known" else "TAILOR_RESUME"
    else:
        decision = "NOT_SHORTLISTED_BELOW_THRESHOLD"

    all_gaps = list(dict.fromkeys(tech_gaps + resp_gaps + experience_flags + eligibility.get("flags", [])))
    matched = [x["topic"] for x in tech_details if x["profile_strength"] >= 0.5]
    responsibility_matches = [x["topic"] for x in resp_details if x["profile_strength"] >= 0.5]
    return {
        "scoring_version": "fit_v2_eligibility_separate",
        "scored_at": now_iso(),
        "score_method": "deterministic keyword/evidence rules; inspect rationale before relying on the score",
        "overall_score": overall,
        "technical_fit_score": overall,
        "threshold": threshold,
        "passes_threshold": overall > threshold,
        "decision": decision,
        "eligibility_status": eligibility["status"],
        "eligibility_score": float(eligibility["score"]),
        "hard_eligibility_block": eligibility["hard_blocked"],
        "recommended_resume_variant": resume_variant,
        "weights": weights,
        "weighted_fit_categories": list(FIT_CATEGORIES),
        # Kept as a separate diagnostic for compatibility; not part of technical_fit_score.
        "component_scores": {**fit_scores, "eligibility_match": float(eligibility["score"])},
        "components": {
            "technical_skills": {"score": tech_score, "weight": weights["technical_skills"], "matched_topics": tech_details, "gaps": tech_gaps},
            "responsibility_match": {"score": responsibility_score, "weight": weights["responsibility_match"], "matched_topics": resp_details, "gaps": resp_gaps},
            "experience_evidence": {"score": experience_score, "weight": weights["experience_evidence"], "reasons": experience_reasons, "flags": experience_flags},
            "seniority_match": {"score": seniority_score, "weight": weights["seniority_match"], "reasons": seniority_reasons},
            "domain_match": {"score": domain_score, "weight": weights["domain_match"], "matched_domains": domain_details, "recommended_resume_variant": resume_variant},
            "eligibility_match": {"score": eligibility["score"], "included_in_overall_score": False, "status": eligibility["status"], "reason": eligibility["reason"], "flags": eligibility.get("flags", [])},
        },
        "strong_matches": matched + responsibility_matches,
        "gaps_and_flags": all_gaps,
        "evidence_map": [],
        "evidence_quality": {
            "confidence": "low",
            "confidence_score": 0.3,
            "evidence_items": 0,
            "supported_items": 0,
            "support_rate": None,
            "resume_text_provided_to_scorer": False,
            "note": "Rule mode uses configured strengths and keyword matches; it does not semantically ground each score in resume excerpts.",
        },
        "required_years_detected": _extract_required_years(job, text),
        "profile_years_experience": years,
        "work_authorization_source": eligibility.get("work_authorization_source", "profile"),
        "note": "This is a role-fit estimate, not a hiring probability. Legal/work authorization is reported separately and never enters the fit score.",
    }


def _confidence_from_evidence(evidence_map: list[dict[str, Any]], *, resume_text_included: bool) -> dict[str, Any]:
    """Compute confidence locally from evidence-map quantity and support, not model self-rating."""
    count = len(evidence_map)
    support_values = {"strong": 1.0, "partial": 0.65, "unclear": 0.25, "missing": 0.0}
    scored_values = [support_values.get(str(item.get("assessment", "unclear")), 0.25) for item in evidence_map]
    support_rate = round(sum(scored_values) / len(scored_values), 3) if scored_values else None
    source_values = [item.get("source_variant") for item in evidence_map]
    resume_grounded = any(value in {"robotics", "backend", "hybrid", "unspecified"} for value in source_values)
    quantity = min(1.0, count / 8.0)
    score = 0.2 + 0.3 * quantity + 0.35 * (support_rate or 0.0)
    if resume_text_included and resume_grounded:
        score += 0.15
    elif not resume_text_included:
        score = min(score, 0.64)
    score = round(min(0.95, max(0.1, score)), 2)
    level = "high" if score >= 0.75 else "moderate" if score >= 0.5 else "low"
    note_parts = [f"{count} evidence item(s) were returned"]
    if support_rate is not None:
        note_parts.append(f"weighted support rate {support_rate:.0%}")
    note_parts.append("resume text was included in the API request" if resume_text_included else "resume text was not sent to the LLM")
    return {
        "confidence": level,
        "confidence_score": score,
        "evidence_items": count,
        "supported_items": sum(1 for item in evidence_map if item.get("assessment") in {"strong", "partial"}),
        "support_rate": support_rate,
        "resume_text_provided_to_scorer": resume_text_included,
        "note": "; ".join(note_parts) + ". This is evidence coverage confidence, not the probability of getting hired.",
    }


def _merge_unique(*groups: Any) -> list[str]:
    result: list[str] = []
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if isinstance(item, str) and item.strip() and item not in result:
                result.append(item)
    return result


def _jd_accepts_cpp_as_an_explicit_language_alternative(job: dict[str, Any]) -> bool:
    text = _job_text(job).lower()
    # Keep this deliberately conservative: only suppress alternative-language
    # gaps when the JD explicitly lists a set of acceptable languages.
    patterns = (
        r"\bc#\s*,\s*c\+\+\s*,\s*java\s*,?\s*or\s+perl\b",
        r"\bc\+\+\s*,\s*java\s*,?\s*or\s+c#\b",
        r"\b(?:c\+\+|java|c#|perl)\b.{0,35}\bor\b.{0,35}\b(?:c\+\+|java|c#|perl)\b",
    )
    return "c++" in text and any(re.search(pattern, text, flags=re.I | re.S) for pattern in patterns)


def _remove_false_alternative_language_gaps(items: list[str], job: dict[str, Any]) -> list[str]:
    if not _jd_accepts_cpp_as_an_explicit_language_alternative(job):
        return items
    filtered = []
    for item in items:
        lower = item.lower()
        language_gap = ("java" in lower or "c#" in lower or "c sharp" in lower) and any(
            marker in lower for marker in ("limited", "no proven", "lack", "absence", "does not demonstrate", "missing evidence")
        )
        if not language_gap:
            filtered.append(item)
    return filtered


def score_job(
    job: dict[str, Any],
    profile: dict[str, Any] | None = None,
    *,
    mode: str | None = None,
    resume_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Score role fit, extract local reviewable facts, then optionally enrich with Groq."""
    load_local_env()
    selected_mode = (mode or os.getenv("JOB_AGENT_SCORING_MODE", "auto")).strip().lower()
    if selected_mode not in {"auto", "rules", "llm"}:
        raise ValueError("Scoring mode must be 'auto', 'rules', or 'llm'.")

    base_profile = copy.deepcopy(profile if profile is not None else load_profile())
    resume_data = load_resume_documents(extra_paths=resume_paths, profile=base_profile)
    documents = resume_data.get("documents", [])
    facts = persist_candidate_facts(documents)
    scoring_profile = apply_extracted_facts(base_profile, facts)
    scorecard = _score_job_rules(job, scoring_profile)
    scorecard["resume_sources"] = resume_data.get("status", {})
    scorecard["resume_text_shared_with_llm"] = False
    scorecard["llm_status"] = "disabled" if selected_mode == "rules" else "not_configured"
    scorecard["candidate_facts"] = {
        "path": "data/candidate_facts.json",
        "review_status": facts.get("review_status", "needs_review"),
        "education_facts_detected": len(facts.get("extracted_facts", {}).get("education", [])),
        "years_statements_detected": len(facts.get("extracted_facts", {}).get("years_experience", [])),
    }

    available_variants = sorted({doc.get("variant") for doc in documents if doc.get("variant") in {"robotics", "backend", "hybrid"}})
    suggested_local = scorecard["recommended_resume_variant"]
    if not available_variants:
        scorecard["recommended_resume_variant"] = "unspecified"
        if suggested_local != "unspecified":
            scorecard["gaps_and_flags"] = _merge_unique(
                scorecard.get("gaps_and_flags", []),
                ["No resume file with a recognized variant is available; add a named resume before selecting a resume variant."],
            )
    elif suggested_local not in available_variants:
        default_variant = scoring_profile.get("candidate_profile", {}).get("default_resume_variant", "unspecified")
        fallback_variant = default_variant if default_variant in available_variants else available_variants[0]
        scorecard["recommended_resume_variant"] = fallback_variant
        scorecard["gaps_and_flags"] = _merge_unique(
            scorecard.get("gaps_and_flags", []),
            [f"The suggested {suggested_local} resume variant has no matching resume file available; using {fallback_variant}, the best configured available option."],
        )

    # Rules mode and automatic fallback make no network request.
    api_key_present = bool(os.getenv("GROQ_API_KEY", "").strip())
    if selected_mode == "rules":
        return scorecard
    if selected_mode == "auto" and not api_key_present:
        return scorecard

    include_resume_text = os.getenv("JOB_AGENT_SHARE_RESUME_WITH_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}
    try:
        assessment = score_with_groq(job, scoring_profile, documents, include_resume_text=include_resume_text)
    except LLMScoringError as exc:
        if selected_mode == "llm":
            raise
        scorecard["llm_status"] = "fallback_to_rules"
        scorecard["llm_error"] = str(exc)
        scorecard["score_method"] += "; Groq was unavailable, so the local rules score was retained"
        return scorecard

    suggested = assessment.get("recommended_resume_variant", "unspecified")
    if available_variants and suggested not in available_variants:
        suggested = scorecard["recommended_resume_variant"]
        assessment["gaps_and_flags"] = _merge_unique(
            assessment.get("gaps_and_flags", []),
            ["The LLM suggested a resume variant for which no matching resume file is available; the suggestion was replaced with an available variant."],
        )
    elif not available_variants:
        suggested = "unspecified"
        if assessment.get("recommended_resume_variant") != "unspecified":
            assessment["gaps_and_flags"] = _merge_unique(
                assessment.get("gaps_and_flags", []),
                ["The LLM suggested a resume variant, but no recognized resume-variant file is available."],
            )

    fit_scores = assessment["component_scores"]
    for key in FIT_CATEGORIES:
        scorecard["component_scores"][key] = round(float(fit_scores[key]), 2)
        scorecard["components"][key]["score"] = round(float(fit_scores[key]), 2)
        scorecard["components"][key]["scoring_source"] = "Groq semantic assessment"
    scorecard["overall_score"] = round(sum(scorecard["component_scores"][key] * scorecard["weights"][key] for key in FIT_CATEGORIES) / 100.0, 2)
    scorecard["technical_fit_score"] = scorecard["overall_score"]
    scorecard["passes_threshold"] = scorecard["overall_score"] > scorecard["threshold"]
    scorecard["recommended_resume_variant"] = suggested
    scorecard["llm_status"] = "used"
    scorecard["resume_text_shared_with_llm"] = include_resume_text
    scorecard["llm_model"] = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"
    scorecard["llm_summary"] = assessment.get("summary", "")
    scorecard["evidence_map"] = assessment.get("evidence_map", [])
    if not include_resume_text:
        # The model only received local profile facts; do not label this evidence as resume-quoted.
        for evidence_item in scorecard["evidence_map"]:
            if isinstance(evidence_item, dict):
                evidence_item["source_variant"] = "profile_only"
    scorecard["uncertainties"] = _merge_unique(assessment.get("uncertainties", []))
    scorecard["uncertainties"] = _remove_false_alternative_language_gaps(scorecard["uncertainties"], job)
    llm_gaps = _remove_false_alternative_language_gaps(assessment.get("gaps_and_flags", []), job)
    scorecard["gaps_and_flags"] = _merge_unique(
        _remove_false_alternative_language_gaps(scorecard.get("gaps_and_flags", []), job),
        llm_gaps,
        scorecard.get("uncertainties", []),
    )
    if _jd_accepts_cpp_as_an_explicit_language_alternative(job):
        summary = scorecard.get("llm_summary", "")
        summary = re.sub(r"(?:lack|lacks|lack of) demonstrated Java/C# proficiency(?:,?\s*)", "", summary, flags=re.I)
        summary = re.sub(r"(?:lack|lacks|lack of) demonstrated Java or C# proficiency(?:,?\s*)", "", summary, flags=re.I)
        if summary != scorecard.get("llm_summary", ""):
            scorecard["llm_summary"] = summary.strip()
            scorecard["gaps_and_flags"] = _merge_unique(scorecard["gaps_and_flags"], [
                "The JD explicitly accepts C++ as one of several language alternatives; missing Java/C# evidence was not counted as a separate language gap."
            ])
    scorecard["evidence_quality"] = _confidence_from_evidence(
        scorecard["evidence_map"], resume_text_included=include_resume_text
    )
    scorecard["score_method"] = f"Groq semantic assessment ({scorecard['llm_model']}) for role fit; deterministic local eligibility gate"
    if scorecard["hard_eligibility_block"]:
        scorecard["decision"] = "SKIP_HARD_ELIGIBILITY_BLOCK"
    elif scorecard["passes_threshold"]:
        scorecard["decision"] = "TAILOR_RESUME" if scorecard["eligibility_status"] == "eligible_known" else "TAILOR_RESUME_ELIGIBILITY_REVIEW"
    else:
        scorecard["decision"] = "NOT_SHORTLISTED_BELOW_THRESHOLD"
    return scorecard
