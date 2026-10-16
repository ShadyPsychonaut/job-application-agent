import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from job_agent.resolver import classify_intent, resolve_answer  # noqa: E402


class ResolverTests(unittest.TestCase):
    def setUp(self):
        today = date.today()
        self.bank = {
            "answers": [
                {
                    "id": "ans_us",
                    "intent": "work_authorization",
                    "answer": "No",
                    "status": "approved",
                    "applicability": {"country": ["US"], "employment_type": ["full_time"]},
                    "verified_at": today.isoformat(),
                    "expires_at": (today + timedelta(days=30)).isoformat(),
                    "reuse_policy": "only_when_applicable",
                    "source": "user_confirmed",
                }
            ]
        }

    def test_exact_applicability_resolves(self):
        result = resolve_answer("work_authorization", {"country": "US", "employment_type": "full_time"}, self.bank)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["answer"], "No")

    def test_different_country_does_not_resolve(self):
        result = resolve_answer("work_authorization", {"country": "IN", "employment_type": "full_time"}, self.bank)
        self.assertEqual(result["status"], "needs_user_input")

    def test_missing_country_does_not_resolve(self):
        result = resolve_answer("work_authorization", {"employment_type": "full_time"}, self.bank)
        self.assertEqual(result["status"], "needs_user_input")

    def test_expired_answer_does_not_resolve(self):
        self.bank["answers"][0]["expires_at"] = (date.today() - timedelta(days=1)).isoformat()
        result = resolve_answer("work_authorization", {"country": "US", "employment_type": "full_time"}, self.bank)
        self.assertEqual(result["status"], "needs_user_input")

    def test_unknown_intent_is_not_guessed(self):
        self.assertEqual(classify_intent("What is your favorite debugging tool?"), "unknown")

    def test_compensation_classifier(self):
        self.assertEqual(classify_intent("What is your expected CTC?"), "expected_compensation")


if __name__ == "__main__":
    unittest.main()
