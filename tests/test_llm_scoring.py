import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from job_agent.scoring import score_job  # noqa: E402
from job_agent.llm_scoring import _redact_contact_details, _json_output_from_response, score_with_groq  # noqa: E402


class LLMScoringTests(unittest.TestCase):
    def test_resume_contact_details_are_redacted_before_api_use(self):
        text = "Email sameer@example.com | +91 77018 73414 | https://linkedin.com/in/example | C++ fleet systems"
        cleaned = _redact_contact_details(text)
        self.assertNotIn("sameer@example.com", cleaned)
        self.assertNotIn("77018 73414", cleaned)
        self.assertNotIn("linkedin.com/in/example", cleaned)
        self.assertIn("C++ fleet systems", cleaned)

    def setUp(self):
        self.profile = json.loads((ROOT / "data" / "profile.json").read_text(encoding="utf-8"))
        self.job = {
            "company": "Example Robotics",
            "title": "Robotics Software Engineer C++",
            "country": "IN",
            "employment_type": "full_time",
            "job_description": {
                "summary": "Build C++ software for multi-robot fleet management and simulation.",
                "responsibilities": ["Implement task scheduling and monitoring pipelines."],
                "basic_qualifications": ["3 years of software development experience."],
            },
        }

    def test_llm_scores_semantic_categories_but_not_eligibility(self):
        assessment = {
            "component_scores": {
                "technical_skills": 9.2,
                "responsibility_match": 9.4,
                "experience_evidence": 8.8,
                "seniority_match": 8.0,
                "domain_match": 9.8,
            },
            "evidence_map": [{
                "category": "technical_skills",
                "requirement": "Modern C++",
                "resume_evidence": "Production C++ distributed systems",
                "assessment": "strong",
                "source_variant": "backend",
            }],
            "recommended_resume_variant": "robotics",
            "summary": "Strong semantic match.",
            "gaps_and_flags": [],
            "uncertainties": ["Degree status is not configured."],
        }
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key", "JOB_AGENT_SHARE_RESUME_WITH_LLM": "false", "GROQ_MODEL": "llama-3.3-70b-versatile"}, clear=False), \
             patch("job_agent.scoring.load_resume_documents", return_value={"documents": [{"filename": "robotics_resume.docx", "variant": "robotics", "text": "Production fleet C++"}], "status": {"resume_count": 1, "filenames": ["robotics_resume.docx"], "message": "loaded"}}), \
             patch("job_agent.scoring.persist_candidate_facts", return_value={"review_status": "needs_review", "extracted_facts": {"education": [], "years_experience": []}}), \
             patch("job_agent.scoring.score_with_groq", return_value=assessment):
            result = score_job(self.job, self.profile, mode="llm")
        self.assertEqual(result["llm_status"], "used")
        self.assertEqual(result["component_scores"]["technical_skills"], 9.2)
        self.assertEqual(result["component_scores"]["eligibility_match"], 10.0)
        self.assertEqual(result["recommended_resume_variant"], "robotics")
        self.assertIn("Degree status is not configured.", result["gaps_and_flags"])
        self.assertEqual(result["decision"], "TAILOR_RESUME")
        self.assertNotIn("eligibility_match", result["weights"])
        self.assertEqual(result["evidence_map"][0]["source_variant"], "profile_only")
        self.assertIn(result["evidence_quality"]["confidence"], {"low", "moderate", "high"})
        self.assertFalse(result["evidence_quality"]["resume_text_provided_to_scorer"])


    def test_llm_cannot_recommend_resume_variant_that_is_not_available(self):
        assessment = {
            "component_scores": {
                "technical_skills": 9.0,
                "responsibility_match": 9.0,
                "experience_evidence": 8.5,
                "seniority_match": 8.0,
                "domain_match": 9.5,
            },
            "evidence_map": [],
            "recommended_resume_variant": "robotics",
            "summary": "Strong robotics-domain match.",
            "gaps_and_flags": [],
            "uncertainties": [],
        }
        resume_status = {"resume_count": 1, "filenames": ["backend_resume.pdf"], "message": "loaded"}
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key", "JOB_AGENT_SHARE_RESUME_WITH_LLM": "false", "GROQ_MODEL": "llama-3.3-70b-versatile"}, clear=False), \
             patch("job_agent.scoring.load_resume_documents", return_value={"documents": [{"filename": "backend_resume.pdf", "variant": "backend", "text": "C++ distributed systems"}], "status": resume_status}), \
             patch("job_agent.scoring.persist_candidate_facts", return_value={"review_status": "needs_review", "extracted_facts": {"education": [], "years_experience": []}}), \
             patch("job_agent.scoring.score_with_groq", return_value=assessment):
            result = score_job(self.job, self.profile, mode="llm")
        self.assertEqual(result["recommended_resume_variant"], "backend")
        self.assertTrue(any("no matching resume file is available" in x for x in result["gaps_and_flags"]))

    def test_groq_chat_response_parser_reads_json_object(self):
        parsed = _json_output_from_response({
            "choices": [{"message": {"content": '{"summary":"ok"}'}}]
        })
        self.assertEqual(parsed, {"summary": "ok"})

    def test_groq_request_uses_chat_completions_and_json_object_mode(self):
        assessment = {
            "component_scores": {
                "technical_skills": 9.0,
                "responsibility_match": 9.0,
                "experience_evidence": 8.0,
                "seniority_match": 8.0,
                "domain_match": 9.0,
            },
            "evidence_map": [],
            "recommended_resume_variant": "backend",
            "summary": "A test assessment.",
            "gaps_and_flags": [],
            "uncertainties": [],
        }

        class FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": json.dumps(assessment)}}]}).encode("utf-8")

        import urllib.request
        with patch.dict(os.environ, {
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "llama-3.3-70b-versatile",
            "GROQ_API_BASE": "https://api.groq.com/openai/v1",
        }, clear=False), patch("job_agent.llm_scoring.urllib.request.urlopen", return_value=FakeResponse()) as mocked:
            result = score_with_groq(self.job, self.profile, [], include_resume_text=False)

        self.assertEqual(result["summary"], "A test assessment.")
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.groq.com/openai/v1/chat/completions")
        request_payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request_payload["model"], "llama-3.3-70b-versatile")
        self.assertEqual(request_payload["response_format"], {"type": "json_object"})

    def test_auto_mode_uses_rules_without_api_key(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "", "JOB_AGENT_SCORING_MODE": "auto"}, clear=False), \
             patch("job_agent.scoring.load_resume_documents", return_value={"documents": [], "status": {"resume_count": 0, "filenames": [], "message": "No resume files found."}}), \
             patch("job_agent.scoring.persist_candidate_facts", return_value={"review_status": "needs_review", "extracted_facts": {"education": [], "years_experience": []}}):
            result = score_job(self.job, self.profile, mode="auto")
        self.assertEqual(result["llm_status"], "not_configured")
        self.assertEqual(result["scoring_version"], "fit_v2_eligibility_separate")


if __name__ == "__main__":
    unittest.main()
