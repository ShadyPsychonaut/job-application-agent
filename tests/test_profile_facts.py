import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from job_agent.profile_facts import extract_candidate_facts, persist_candidate_facts  # noqa: E402
from job_agent.resume_ingestion import load_resume_documents  # noqa: E402


class ProfileFactsTests(unittest.TestCase):
    def test_extracts_education_and_experience_from_saved_resume(self):
        docs = load_resume_documents()
        facts = extract_candidate_facts(docs["documents"])
        education = facts["extracted_facts"]["education"]
        self.assertTrue(education)
        self.assertEqual(education[0]["degree"], "B.Tech")
        self.assertEqual(education[0]["field"], "Mechanical Engineering")
        self.assertEqual(education[0]["minor"], "Information Technology")
        self.assertEqual(education[0]["institution"], "Delhi Technological University (formerly DCE)")
        self.assertEqual(education[0]["graduation_year"], 2023)
        self.assertEqual(education[0]["gpa"], "8.04/10")
        self.assertEqual(education[0]["verification_status"], "extracted_needs_review")
        self.assertEqual(facts["extracted_facts"]["years_experience"][0]["years_experience"], 3.0)


    def test_reviewed_facts_survive_pdf_to_docx_source_change(self):
        base_text = (
            "Summary\nSoftware engineer with 3 years of experience\n\nEducation\n"
            "Example University\nB.Tech in Computer Science\n2020 - 2024"
        )
        pdf_docs = [{
            "filename": "backend_resume.pdf", "sha256": "pdf-hash", "variant": "backend",
            "text": base_text,
        }]
        docx_docs = [{
            "filename": "backend_resume.docx", "sha256": "docx-hash", "variant": "backend",
            "text": base_text,
        }]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate_facts.json"
            with patch("job_agent.profile_facts.FACTS_PATH", path):
                first = persist_candidate_facts(pdf_docs)
                first["review_status"] = "approved"
                first["reviewed_at"] = "2026-10-01T00:00:00+00:00"
                for group in ("education", "years_experience"):
                    for fact in first["extracted_facts"][group]:
                        fact["verification_status"] = "user_reviewed"
                path.write_text(json.dumps(first), encoding="utf-8")
                second = persist_candidate_facts(docx_docs)

        self.assertEqual(second["review_status"], "approved")
        self.assertEqual(second["sources"][0]["filename"], "backend_resume.docx")
        self.assertEqual(second["reviewed_at"], "2026-10-01T00:00:00+00:00")
        self.assertTrue(all(
            fact["verification_status"] == "user_reviewed"
            for group in ("education", "years_experience")
            for fact in second["extracted_facts"][group]
        ))

    def test_changed_experience_claim_is_not_automatically_approved(self):
        def doc(filename, digest, years):
            return [{
                "filename": filename, "sha256": digest, "variant": "backend",
                "text": f"Summary\nSoftware engineer with {years} years of experience\n\nEducation\nExample University\nB.Tech in Computer Science\n2020 - 2024",
            }]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate_facts.json"
            with patch("job_agent.profile_facts.FACTS_PATH", path):
                first = persist_candidate_facts(doc("backend_resume.pdf", "pdf-hash", 3))
                first["review_status"] = "approved"
                for group in ("education", "years_experience"):
                    for fact in first["extracted_facts"][group]:
                        fact["verification_status"] = "user_reviewed"
                path.write_text(json.dumps(first), encoding="utf-8")
                second = persist_candidate_facts(doc("backend_resume.docx", "docx-hash", 4))

        years_facts = second["extracted_facts"]["years_experience"]
        self.assertEqual(years_facts[0]["verification_status"], "extracted_needs_review")
        self.assertEqual(second["review_status"], "needs_review")

    def test_persisted_facts_keep_review_state_when_sources_are_unchanged(self):
        docs = [{"filename": "resume.txt", "sha256": "abc", "variant": "backend", "text": (
            "Summary\nSoftware engineer with 3 years of experience\n\nEducation\n"
            "Example University\nB.Tech in Computer Science\n2020 - 2024"
        )}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate_facts.json"
            with patch("job_agent.profile_facts.FACTS_PATH", path):
                first = persist_candidate_facts(docs)
                first["review_status"] = "approved"
                path.write_text(json.dumps(first), encoding="utf-8")
                second = persist_candidate_facts(docs)
        self.assertEqual(second["review_status"], "approved")
        self.assertEqual(second["extracted_facts"]["education"][0]["graduation_year"], 2024)
        self.assertEqual(second["extracted_facts"]["education"][0]["verification_status"], "user_reviewed")


if __name__ == "__main__":
    unittest.main()
