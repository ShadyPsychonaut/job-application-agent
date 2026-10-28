import tempfile
import unittest
from pathlib import Path

from docx import Document

from job_agent.resume_tailoring import (
    ResumeTailoringError,
    _docx_to_source_text,
    _experience_groups,
    _skill_items,
    _validate_written_docx,
    _write_docx,
    parse_resume_text,
)


class ResumeTailoringIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.source_path = self.root / "source.docx"
        self.output_path = self.root / "tailored.docx"

        doc = Document()
        doc.add_paragraph("PROFESSIONAL SUMMARY")
        doc.add_paragraph(
            "Robotics software engineer with C++ experience and distributed systems exposure."
        )
        doc.add_paragraph("────────────────────────")
        doc.add_paragraph("CORE TECHNICAL SKILLS")
        doc.add_paragraph("Languages: C++, Python, SQL")
        doc.add_paragraph("Tools: Git, Docker, Linux")
        doc.add_paragraph("────────────────────────")
        doc.add_paragraph("PROFESSIONAL EXPERIENCE")
        doc.add_paragraph("Robotics Engineer | Example Co. | 2023-2026")
        doc.add_paragraph(
            "Built C++ fleet management components for live robots.",
            style="List Bullet",
        )
        doc.add_paragraph(
            "Developed an ImGui simulation editor with undo/redo.",
            style="List Bullet",
        )
        doc.add_paragraph("SELECTED ROBOTICS PROJECT")
        doc.add_paragraph(
            "Built a personal ROS 2 multi-robot simulation.",
            style="List Bullet",
        )
        doc.add_paragraph("EDUCATION")
        doc.add_paragraph("B.Tech Mechanical Engineering, minor IT, 2023")
        doc.save(self.source_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def _build_result(self):
        source = Document(self.source_path)
        parsed = parse_resume_text(_docx_to_source_text(source))
        skill_ids = [item["id"] for item in _skill_items(parsed)]
        groups = _experience_groups(parsed)
        result = {
            "professional_summary": (
                "C++ software engineer with three years of production experience "
                "building distributed robotics components and simulation tools."
            ),
            "skills": [
                {"label": "Core Languages", "skill_ids": skill_ids[:3]},
                {"label": "Platforms & Tools", "skill_ids": skill_ids[3:]},
            ],
            "experience_bullet_order": {
                group["group_id"]: [
                    bullet["id"] for bullet in reversed(group["bullets"])
                ]
                for group in groups
            },
        }
        return source, parsed, result

    def test_selected_robotics_project_is_not_treated_as_experience(self):
        source, parsed, _ = self._build_result()
        kinds = [section["kind"] for section in parsed["sections"]]
        self.assertEqual(kinds, ["summary", "skills", "experience", "other", "other"])
        bullets = [
            bullet
            for group in _experience_groups(parsed)
            for bullet in group["bullets"]
        ]
        self.assertEqual(len(bullets), 2)
        self.assertNotIn(
            "Built a personal ROS 2 multi-robot simulation.",
            [bullet["text"] for bullet in bullets],
        )

    def test_written_docx_passes_integrity_check(self):
        source, parsed, result = self._build_result()
        _write_docx(source, parsed, result, self.output_path)
        _validate_written_docx(self.source_path, self.output_path, parsed, result)
        self.assertTrue(self.output_path.exists())

    def test_integrity_check_rejects_changes_to_education(self):
        source, parsed, result = self._build_result()
        _write_docx(source, parsed, result, self.output_path)
        output = Document(self.output_path)
        education = False
        for paragraph in output.paragraphs:
            if paragraph.text.strip() == "EDUCATION":
                education = True
                continue
            if education and paragraph.text.strip():
                paragraph.text = "Changed education text"
                break
        output.save(self.output_path)
        with self.assertRaisesRegex(ResumeTailoringError, "content outside the three permitted sections"):
            _validate_written_docx(self.source_path, self.output_path, parsed, result)


if __name__ == "__main__":
    unittest.main()
