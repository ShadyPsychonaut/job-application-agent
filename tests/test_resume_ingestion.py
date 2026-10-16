import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from job_agent.resume_ingestion import load_resume_documents  # noqa: E402


class ResumeIngestionTests(unittest.TestCase):
    def test_resume_cache_refreshes_only_when_file_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            resume = temp_path / "robotics_resume.txt"
            cache = temp_path / "resume_cache.json"
            resume.write_text("C++ fleet management and task scheduling", encoding="utf-8")
            with patch("job_agent.resume_ingestion.CACHE_PATH", cache):
                first = load_resume_documents([str(resume)], profile={"candidate_profile": {}})
                second = load_resume_documents([str(resume)], profile={"candidate_profile": {}})
                resume.write_text("C++ fleet management, ROS 2, and simulation tooling", encoding="utf-8")
                third = load_resume_documents([str(resume)], profile={"candidate_profile": {}})
            self.assertEqual(first["status"]["refreshed_count"], 1)
            self.assertEqual(second["status"]["cached_count"], 1)
            self.assertEqual(third["status"]["refreshed_count"], 1)
            self.assertIn("ROS 2", third["documents"][0]["text"])
            self.assertEqual(third["documents"][0]["variant"], "robotics")

    def test_auto_discovery_skips_readme(self):
        with tempfile.TemporaryDirectory() as temp:
            # Auto discovery is project-root bound; this verifies no unsupported README is a candidate.
            status = load_resume_documents(profile={"candidate_profile": {"resume_files": []}})["status"]
            self.assertNotIn("README.md", status["filenames"])


if __name__ == "__main__":
    unittest.main()
