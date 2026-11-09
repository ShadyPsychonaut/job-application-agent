import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from job_agent.scoring import score_job, _remove_false_alternative_language_gaps, _remove_llm_eligibility_claims, _score_job_rules  # noqa: E402
from job_agent.agent import answer_pending_question, process_job  # noqa: E402


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads((ROOT / "data" / "profile.json").read_text(encoding="utf-8"))

    def test_weights_total_100_and_score_is_bounded(self):
        job = {
            "company": "Example",
            "title": "C++ Software Engineer",
            "country": "IN",
            "job_description": {"summary": "Build C++ distributed multithreaded backend services.", "responsibilities": ["Test and operate production services"]},
        }
        result = score_job(job, self.profile)
        self.assertAlmostEqual(sum(result["weights"].values()), 100.0)
        self.assertGreaterEqual(result["overall_score"], 0.0)
        self.assertLessEqual(result["overall_score"], 10.0)
        self.assertIn("technical_skills", result["component_scores"])

    def test_random_amazon_example_is_scored_and_explained(self):
        job = json.loads((ROOT / "examples" / "sample_job.json").read_text(encoding="utf-8"))
        result = score_job(job, self.profile)
        self.assertIsInstance(result["overall_score"], float)
        self.assertIn(result["decision"], {"TAILOR_RESUME", "TAILOR_RESUME_ELIGIBILITY_REVIEW", "NOT_SHORTLISTED_BELOW_THRESHOLD", "SKIP_HARD_ELIGIBILITY_BLOCK"})
        self.assertTrue(result["components"]["technical_skills"]["matched_topics"])
        reasons = result["components"]["experience_evidence"]["reasons"] + result["components"]["seniority_match"]["reasons"]
        self.assertEqual(len(reasons), len(set(reasons)))
        self.assertEqual(sum(result["weights"].values()), 100.0)

    def test_strong_robotics_role_gets_robotics_resume_when_variant_exists(self):
        profile = json.loads(json.dumps(self.profile))
        # This test uses a hypothetical confirmed authorization only to test the high-fit branch.
        profile["candidate_profile"]["eligibility"]["work_authorization_by_country"]["IN"] = "authorized"
        job = {
            "company": "Example Robotics",
            "title": "Robotics Software Development Engineer II C++",
            "country": "IN",
            "employment_type": "full_time",
            "job_description": {
                "summary": "Build C++ software for AMR fleet management with ROS 2 and multi-robot navigation.",
                "responsibilities": [
                    "Implement task scheduling, mission dispatch and robot traffic management.",
                    "Build simulator tooling and monitoring for production fleets.",
                    "Develop multithreaded distributed services using Linux, RabbitMQ and MongoDB.",
                    "Test and operate high-availability services with metrics and dashboards."
                ],
                "basic_qualifications": ["At least 3 years of C++ software development experience on robotics or distributed systems."]
            }
        }
        resume_docs = [{"filename": "robotics_resume.docx", "variant": "robotics", "text": "Production C++ robotics fleet systems", "sha256": "robotics-test"}]
        facts = {"review_status": "needs_review", "extracted_facts": {"education": [], "years_experience": []}}
        with patch("job_agent.scoring.load_resume_documents", return_value={"documents": resume_docs, "status": {"resume_count": 1, "filenames": ["robotics_resume.docx"]}}), \
             patch("job_agent.scoring.persist_candidate_facts", return_value=facts):
            result = score_job(job, profile, mode="rules")
        self.assertEqual(result["recommended_resume_variant"], "robotics")
        self.assertGreater(result["component_scores"]["technical_skills"], 8.0)
        self.assertTrue(result["passes_threshold"])

    def test_rules_mode_never_recommends_a_missing_resume_variant(self):
        job = {"company": "Example Robotics", "title": "Robotics Software Engineer C++", "country": "IN",
               "job_description": {"summary": "Build C++ software for robot fleet management and autonomy."}}
        resume_docs = [{"filename": "backend_resume.pdf", "variant": "backend", "text": "C++ backend systems", "sha256": "backend-test"}]
        facts = {"review_status": "needs_review", "extracted_facts": {"education": [], "years_experience": []}}
        with patch("job_agent.scoring.load_resume_documents", return_value={"documents": resume_docs, "status": {"resume_count": 1, "filenames": ["backend_resume.pdf"]}}), \
             patch("job_agent.scoring.persist_candidate_facts", return_value=facts):
            result = score_job(job, self.profile, mode="rules")
        self.assertEqual(result["recommended_resume_variant"], "backend")
        self.assertTrue(any("no matching resume file available" in flag for flag in result["gaps_and_flags"]))

    def test_no_resume_variant_is_recommended_when_no_resume_is_available(self):
        job = {"company": "Example", "title": "Software Engineer", "country": "IN"}
        facts = {"review_status": "needs_review", "extracted_facts": {"education": [], "years_experience": []}}
        with patch("job_agent.scoring.load_resume_documents", return_value={"documents": [], "status": {"resume_count": 0, "filenames": []}}), \
             patch("job_agent.scoring.persist_candidate_facts", return_value=facts):
            result = score_job(job, self.profile, mode="rules")
        self.assertEqual(result["recommended_resume_variant"], "unspecified")

    def test_eligibility_is_separate_from_technical_fit_score(self):
        job = {
            "company": "Example", "title": "C++ Software Engineer", "country": "IN",
            "job_description": {"summary": "Build C++ distributed multithreaded backend services."},
        }
        unknown = json.loads(json.dumps(self.profile))
        authorized = json.loads(json.dumps(self.profile))
        authorized["candidate_profile"]["eligibility"]["work_authorization_by_country"]["IN"] = "authorized"
        unknown_result = score_job(job, unknown, mode="rules")
        authorized_result = score_job(job, authorized, mode="rules")
        self.assertEqual(unknown_result["overall_score"], authorized_result["overall_score"])
        self.assertNotEqual(unknown_result["eligibility_status"], authorized_result["eligibility_status"])
        self.assertNotIn("eligibility_match", unknown_result["weights"])
        self.assertFalse(unknown_result["components"]["eligibility_match"]["included_in_overall_score"])

    def test_extracted_degree_is_evidence_not_an_unknown_education_gap(self):
        profile = json.loads(json.dumps(self.profile))
        docs = [{"filename": "backend_resume.pdf", "variant": "backend", "text": "B.Tech in Mechanical Engineering", "sha256": "abc"}]
        facts = {"review_status": "needs_review", "extracted_facts": {"education": [{
            "degree": "B.Tech", "field": "Mechanical Engineering", "bachelor_level": True,
        }], "years_experience": []}}
        job = {"company": "Example", "title": "Software Engineer", "country": "IN",
               "job_description": {"basic_qualifications": ["Bachelor's degree in Engineering or related field"]}}
        with patch("job_agent.scoring.load_resume_documents", return_value={"documents": docs, "status": {"resume_count": 1, "filenames": ["backend_resume.pdf"]}}), \
             patch("job_agent.scoring.persist_candidate_facts", return_value=facts):
            result = score_job(job, profile, mode="rules")
        self.assertTrue(any("Resume text lists B.Tech Mechanical Engineering" in flag for flag in result["gaps_and_flags"]))
        self.assertFalse(any("education has not been configured" in flag for flag in result["gaps_and_flags"]))

    def test_unknown_country_auth_is_not_marked_eligible(self):
        job = {"company": "Example", "title": "Engineer", "country": "IN", "job_description": {"summary": "Software services"}}
        result = score_job(job, self.profile)
        self.assertEqual(result["eligibility_status"], "needs_review")

    def test_us_no_sponsorship_role_is_blocked(self):
        job = {
            "company": "Example US", "title": "Senior C++ Engineer", "country": "US",
            "job_description": {"summary": "C++ backend services. We do not provide visa sponsorship and applicants must already be authorized to work in the US."}
        }
        result = score_job(job, self.profile)
        self.assertTrue(result["hard_eligibility_block"])
        self.assertEqual(result["decision"], "SKIP_HARD_ELIGIBILITY_BLOCK")

    def test_low_score_still_queues_reusable_profile_questions(self):
        job = json.loads((ROOT / "examples" / "sample_job.json").read_text(encoding="utf-8"))
        saved_pending = {"questions": []}
        with patch("job_agent.agent.load_applications", return_value={"applications": []}), \
             patch("job_agent.agent.load_pending", side_effect=lambda: saved_pending), \
             patch("job_agent.agent.save_applications"), \
             patch("job_agent.agent.save_pending", side_effect=lambda data: saved_pending.update(data)), \
             patch("job_agent.agent.upsert_log"), \
             patch("job_agent.resolver.load_answer_bank", return_value={"answers": []}):
            result = process_job(job, scorecard={
                "overall_score": 6.0, "threshold": 8.5, "passes_threshold": False,
                "decision": "NOT_SHORTLISTED_BELOW_THRESHOLD", "hard_eligibility_block": False,
                "component_scores": {"technical_skills": 6, "responsibility_match": 6, "experience_evidence": 6, "seniority_match": 6, "domain_match": 6, "eligibility_match": 6},
                "eligibility_status": "needs_review", "recommended_resume_variant": "backend",
                "weights": {},
            })
        self.assertEqual(result["status"], "not_shortlisted_profile_questions_pending")
        self.assertGreater(len(result["pending_question_ids"]), 0)
        intents = {q["intent"] for q in saved_pending["questions"]}
        self.assertIn("work_authorization", intents)
        self.assertIn("sponsorship", intents)
        self.assertIn("expected_compensation", intents)
        self.assertNotIn("cover_letter", intents)
        self.assertGreater(result["scorecard"]["threshold"], 0)

    def test_low_score_generates_default_eligibility_questions_when_form_fields_missing(self):
        job = {"company": "Example", "title": "Engineer", "jd_url": "https://example.test/role", "country": "IN", "fields": []}
        saved_pending = {"questions": []}
        scorecard = {
            "overall_score": 5.0, "threshold": 8.5, "passes_threshold": False,
            "decision": "NOT_SHORTLISTED_BELOW_THRESHOLD", "hard_eligibility_block": False,
            "component_scores": {"technical_skills": 5, "responsibility_match": 5, "experience_evidence": 5, "seniority_match": 5, "domain_match": 5, "eligibility_match": 6},
            "eligibility_status": "needs_review", "recommended_resume_variant": "backend", "weights": {},
        }
        with patch("job_agent.agent.load_applications", return_value={"applications": []}), \
             patch("job_agent.agent.load_pending", side_effect=lambda: saved_pending), \
             patch("job_agent.agent.save_applications"), \
             patch("job_agent.agent.save_pending", side_effect=lambda data: saved_pending.update(data)), \
             patch("job_agent.agent.upsert_log"):
            result = process_job(job, scorecard=scorecard)
        intents = {q["intent"] for q in saved_pending["questions"]}
        self.assertEqual(intents, {"work_authorization", "sponsorship"})
        self.assertEqual(len(result["pending_question_ids"]), 2)

    def test_duplicate_low_score_backfills_missing_profile_questions(self):
        job = json.loads((ROOT / "examples" / "sample_job.json").read_text(encoding="utf-8"))
        existing = {"application_id": "app_existing", "jd_url": job["jd_url"], "score_decision": "NOT_SHORTLISTED_BELOW_THRESHOLD", "status": "not_shortlisted", "answers": []}
        apps = {"applications": [existing]}
        pending = {"questions": []}
        with patch("job_agent.agent.load_applications", return_value=apps), \
             patch("job_agent.agent.load_pending", return_value=pending), \
             patch("job_agent.agent.save_applications"), \
             patch("job_agent.agent.save_pending", side_effect=lambda data: pending.update(data)), \
             patch("job_agent.agent.upsert_log"), \
             patch("job_agent.resolver.load_answer_bank", return_value={"answers": []}):
            result = process_job(job, scorecard={"overall_score": 6.0, "threshold": 8.5})
        self.assertEqual(result["status"], "duplicate")
        self.assertTrue(result["pending_question_ids"])
        self.assertTrue(any(q.get("application_id") == "app_existing" for q in pending["questions"]))



    def test_llm_eligibility_warnings_do_not_conflict_with_local_gate(self):
        profile = json.loads(json.dumps(self.profile))
        profile["candidate_profile"].setdefault("eligibility", {}).setdefault(
            "work_authorization_by_country", {}
        )["IN"] = "authorized"
        job = {
            "company": "Example",
            "title": "C++ Backend Engineer",
            "country": "IN",
            "job_description": {"summary": "Build C++ distributed backend services."},
        }
        documents = [{
            "filename": "backend_resume.docx", "variant": "backend",
            "sha256": "test-hash", "text": "C++ distributed backend services",
        }]
        facts = {"review_status": "approved", "extracted_facts": {"education": [], "years_experience": []}}
        assessment = {
            "component_scores": {
                "technical_skills": 8.0,
                "responsibility_match": 7.0,
                "experience_evidence": 7.0,
                "seniority_match": 6.5,
                "domain_match": 6.0,
            },
            "recommended_resume_variant": "backend",
            "summary": "Strong fit for backend service work.",
            "gaps_and_flags": [
                "Work-authorization status for India remains unknown",
                "Limited documented evidence for mentoring",
            ],
            "uncertainties": ["Legal work authorization is not confirmed"],
            "evidence_map": [],
        }
        with patch("job_agent.scoring.load_resume_documents", return_value={"documents": documents, "status": {"resume_count": 1, "filenames": ["backend_resume.docx"]}}), \
             patch("job_agent.scoring.persist_candidate_facts", return_value=facts), \
             patch("job_agent.scoring.score_with_groq", return_value=assessment):
            result = score_job(job, profile, mode="llm")

        self.assertEqual(result["eligibility_status"], "eligible_known")
        self.assertFalse(any("work authorization" in flag.casefold() for flag in result["gaps_and_flags"]))
        self.assertFalse(any("work authorization" in item.casefold() for item in result["uncertainties"]))
        self.assertIn("Limited documented evidence for mentoring", result["gaps_and_flags"])

    def test_llm_work_authorization_warnings_are_suppressed_from_fit_flags(self):
        flags = [
            "Work-authorization status for India remains unknown",
            "Eligibility is uncertain. Do not infer legal work authorization from location.",
            "Limited documented evidence for performance / latency",
            "Candidate may need visa sponsorship for the role",
        ]
        result = _remove_llm_eligibility_claims(flags)
        self.assertEqual(result, ["Limited documented evidence for performance / latency"])

    def test_alternative_language_is_not_reported_as_a_gap_when_cpp_is_accepted(self):
        job = {"title": "Software Development Engineer II", "job_description": {
            "basic_qualifications": ["Develop distributed applications using C#, C++, Java, or Perl."]
        }}
        gaps = ["Limited documented evidence for Java", "No proven Java or C# experience; these are acceptable alternatives", "Limited documented evidence for mentoring"]
        result = _remove_false_alternative_language_gaps(gaps, job)
        self.assertEqual(result, ["Limited documented evidence for mentoring"])

    def test_cpp_satisfies_c_cpp_or_java_or_python_or_golang_alternatives(self):
        job = {
            "job_description": {
                "basic_qualifications": [
                    "Proficiency in C/C++ or Java or Python or Golang is required."
                ]
            }
        }
        gaps = [
            "Limited documented evidence for Java",
            "No proven Python experience",
            "Missing evidence for Golang",
        ]
        self.assertEqual(_remove_false_alternative_language_gaps(gaps, job), [])

    def test_rules_score_language_or_group_as_one_requirement(self):
        profile = json.loads(json.dumps(self.profile))
        job = {
            "company": "Example",
            "title": "Software Engineer",
            "country": "IN",
            "job_description": {
                "summary": "Proficiency in C/C++ or Java or Python or Golang is required.",
            },
        }
        result = _score_job_rules(job, profile)
        topics = [
            item["topic"]
            for item in result["components"]["technical_skills"]["matched_topics"]
        ]
        self.assertIn("C++", topics)
        self.assertNotIn("Java", topics)
        self.assertFalse(
            any("Java" in gap for gap in result["components"]["technical_skills"]["gaps"])
        )

    def test_independently_required_java_is_not_suppressed_by_language_or_group(self):
        job = {
            "job_description": {
                "basic_qualifications": [
                    "Proficiency in C/C++ or Java or Python or Golang.",
                    "Professional Java experience is required for this position.",
                ]
            }
        }
        gaps = [
            "Limited documented evidence for Java",
            "No proven Python experience",
            "Missing evidence for Golang",
        ]
        self.assertEqual(
            _remove_false_alternative_language_gaps(gaps, job),
            ["Limited documented evidence for Java"],
        )

    def test_mentoring_as_needed_does_not_create_a_mentoring_gap(self):
        profile = json.loads(json.dumps(self.profile))
        job = {
            "company": "Example",
            "title": "Software Engineer",
            "country": "IN",
            "job_description": {
                "summary": "Build C++ distributed software services.",
                "responsibilities": [
                    "Collaborate with teammates and coach others, as needed."
                ],
            },
        }
        result = _score_job_rules(job, profile)
        self.assertNotIn(
            "Technical leadership / mentoring",
            [item["topic"] for item in result["components"]["responsibility_match"]["matched_topics"]],
        )
        self.assertFalse(
            any("mentoring" in flag.lower() for flag in result["gaps_and_flags"])
        )

    def test_mandatory_team_mentoring_remains_a_meaningful_gap(self):
        profile = json.loads(json.dumps(self.profile))
        job = {
            "company": "Example",
            "title": "Software Engineer",
            "country": "IN",
            "job_description": {
                "summary": "Build C++ distributed software services.",
                "basic_qualifications": [
                    "Must mentor junior engineers and coach team members."
                ],
            },
        }
        result = _score_job_rules(job, profile)
        self.assertTrue(
            any("mentoring" in flag.lower() for flag in result["gaps_and_flags"])
        )
        self.assertTrue(
            any("mentoring" in flag.lower() for flag in result["components"]["experience_evidence"]["flags"])
        )

    def test_soft_mentoring_does_not_hide_a_separate_mandatory_mentoring_requirement(self):
        profile = json.loads(json.dumps(self.profile))
        job = {
            "company": "Example",
            "title": "Software Engineer",
            "country": "IN",
            "job_description": {
                "summary": "Build C++ distributed software services.",
                "responsibilities": ["Coach peers, as needed."],
                "basic_qualifications": ["Must mentor junior engineers."],
            },
        }
        result = _score_job_rules(job, profile)
        self.assertTrue(
            any("mentoring" in flag.lower() for flag in result["gaps_and_flags"])
        )

    def test_preferred_ai_coding_assistant_has_less_impact_than_mandatory(self):
        profile = json.loads(json.dumps(self.profile))
        profile["candidate_profile"]["evidence_strength"]["ai_assisted_development"] = 0.65
        preferred = {
            "company": "Example",
            "title": "Software Engineer",
            "country": "IN",
            "job_description": {
                "summary": "Build C++ distributed software services.",
                "preferred_qualifications": [
                    "Experience using AI coding assistants such as Claude Code."
                ],
            },
        }
        mandatory = {
            "company": "Example",
            "title": "Software Engineer",
            "country": "IN",
            "job_description": {
                "summary": "Build C++ distributed software services.",
                "basic_qualifications": [
                    "Required: experience using AI coding assistants such as Claude Code."
                ],
            },
        }
        preferred_result = _score_job_rules(preferred, profile)
        mandatory_result = _score_job_rules(mandatory, profile)
        preferred_topic = next(
            item for item in preferred_result["components"]["technical_skills"]["matched_topics"]
            if item["topic"] == "AI-assisted development tools"
        )
        mandatory_topic = next(
            item for item in mandatory_result["components"]["technical_skills"]["matched_topics"]
            if item["topic"] == "AI-assisted development tools"
        )
        self.assertLess(
            preferred_topic["weighted_contribution"],
            mandatory_topic["weighted_contribution"],
        )

    def test_professional_claude_code_answer_bank_fact_enriches_local_scoring(self):
        job = {
            "company": "Example",
            "title": "Software Engineer",
            "country": "IN",
            "job_description": {
                "summary": "Build C++ software and use AI coding assistants such as Claude Code."
            },
        }
        facts = {"review_status": "needs_review", "extracted_facts": {"education": [], "years_experience": []}}
        answer_bank = {
            "answers": [{
                "id": "ans_ai_tools",
                "intent": "ai_assisted_development",
                "question": "Have you used AI-assisted development tools professionally?",
                "answer": "Yes. I have used Claude Code in my professional software development work.",
                "status": "approved",
                "applicability": {},
                "verified_at": "2026-11-03",
                "expires_at": "2027-02-01",
                "reuse_policy": "only_when_applicable",
            }]
        }
        with (
            patch("job_agent.scoring.load_resume_documents", return_value={"documents": [], "status": {"resume_count": 0, "filenames": []}}),
            patch("job_agent.scoring.persist_candidate_facts", return_value=facts),
            patch("job_agent.scoring.load_answer_bank", return_value=answer_bank),
        ):
            result = score_job(job, self.profile, mode="rules")
        topic = next(
            item for item in result["components"]["technical_skills"]["matched_topics"]
            if item["topic"] == "AI-assisted development tools"
        )
        self.assertEqual(topic["profile_strength"], 0.65)

    def test_ai_assisted_development_answer_can_be_saved_as_universal_reusable_fact(self):
        pending = {
            "questions": [{
                "question_id": "q_ai_tools",
                "application_id": "app_test",
                "field_label": "Have you used AI coding assistants professionally?",
                "intent": "ai_assisted_development",
                "required": True,
                "status": "awaiting_user",
                "context": {"country": "IN", "employment_type": "full_time"},
                "options": [],
            }]
        }
        applications = {
            "applications": [{
                "application_id": "app_test",
                "answers": [],
                "score_decision": "TAILOR_RESUME",
            }]
        }
        bank = {"schema_version": 1, "answers": []}
        with (
            patch("job_agent.agent.load_pending", return_value=pending),
            patch("job_agent.agent.load_applications", return_value=applications),
            patch("job_agent.agent.save_pending"),
            patch("job_agent.agent.save_applications"),
            patch("job_agent.agent.upsert_log"),
            patch("job_agent.storage.load_answer_bank", return_value=bank),
            patch("job_agent.storage.save_answer_bank") as save_bank,
        ):
            answer_pending_question(
                "q_ai_tools",
                "Yes. I have used Claude Code in my professional software development work.",
                save_for_reuse=True,
            )
        saved = save_bank.call_args.args[0]["answers"][0]
        self.assertEqual(saved["intent"], "ai_assisted_development")
        self.assertEqual(saved["applicability"], {})
        self.assertEqual(saved["status"], "approved")
        self.assertIn("Claude Code", saved["answer"])

    def test_threshold_is_strict(self):
        profile = json.loads(json.dumps(self.profile))
        profile["job_scoring"]["tailoring_threshold"] = 0
        result = score_job({"company": "Example", "title": "Software Engineer", "country": "IN"}, profile)
        self.assertTrue(result["passes_threshold"])  # score is > 0, not >= 0


if __name__ == "__main__":
    unittest.main()
