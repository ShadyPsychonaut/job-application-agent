import json
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from job_agent.jd_ingestion import (
    JobIngestionError,
    _extract_visible_job_description,
    _likely_official_or_company_ats,
    _PageParser,
    _parse_jsonld,
    _parse_linkedin_title,
    _trim_non_job_boilerplate,
    ingest_job_url,
    is_job_description_processable,
    main,
    normalize_job_url,
    test_brave_search_connection,
)

JOB_HTML = r"""<!doctype html><html><head>
<title>Software Engineer - Example Systems</title>
<meta property="og:title" content="Software Engineer - Example Systems" />
<script type="application/ld+json">
{
 "@context":"https://schema.org",
 "@type":"JobPosting",
 "title":"Software Engineer",
 "identifier":{"@type":"PropertyValue","name":"Example Systems","value":"JOB-1234"},
 "datePosted":"2026-10-20",
 "employmentType":"FULL_TIME",
 "hiringOrganization":{"@type":"Organization","name":"Example Systems"},
 "jobLocation":{"@type":"Place","address":{"@type":"PostalAddress","addressLocality":"Bengaluru","addressRegion":"Karnataka","addressCountry":"IN"}},
 "url":"https://careers.example.com/jobs/JOB-1234",
 "description":"<p>Build reliable C++ distributed services for a production platform. Work with modern design, integration, quality, and reliability practices.</p><h2>Responsibilities</h2><ul><li>Design and implement C++ services.</li><li>Write unit tests and maintain observability.</li><li>Collaborate with distributed engineering teams to deliver high quality software.</li></ul><h2>Basic Qualifications</h2><ul><li>Three years of software engineering experience.</li><li>Experience with C++ and Linux.</li></ul><h2>Preferred Qualifications</h2><ul><li>Experience with Kubernetes.</li></ul>"
}
</script>
</head><body><main><h1>Software Engineer</h1><p>Build reliable systems.</p></main></body></html>"""

LINKEDIN_METADATA_ONLY = r"""<!doctype html><html><head>
<title>Software Engineer at Cisco — Bengaluru, Karnataka, India | LinkedIn Jobs</title>
<meta property="og:title" content="Software Engineer at Cisco — Bengaluru, Karnataka, India | LinkedIn Jobs" />
<meta property="og:description" content="Apply for Software Engineer at Cisco in Bengaluru, Karnataka, India. Full-time Entry level role. See responsibilities, qualifications, and similar jobs on LinkedIn." />
</head><body><main><h1>Software Engineer</h1><p>Cisco Bengaluru, Karnataka, India</p></main></body></html>"""

LINKEDIN_FULL_VISIBLE = r"""<!doctype html><html><head>
<title>Software Engineer at Cisco — Bengaluru, Karnataka, India | LinkedIn Jobs</title>
<meta property="og:title" content="Software Engineer at Cisco — Bengaluru, Karnataka, India | LinkedIn Jobs" />
</head><body><main>
<h2>Meet the Team</h2><p>At Cisco, we build software for global infrastructure and large-scale cloud applications. Join engineers who solve complex technical challenges in a collaborative environment.</p>
<h2>Your Impact</h2><ul>
<li>Improve products and services using software engineering best practices.</li>
<li>Work independently with guidance and determine an appropriate approach to assigned work.</li>
<li>Lead small-to-medium features from technical design through completion.</li>
<li>Perform coding, debugging, testing, and troubleshooting throughout the development process.</li>
</ul>
<h2>Minimum Qualifications</h2><ul>
<li>Bachelor's degree in Computer Science or a related field and relevant experience.</li>
<li>Strong grasp of data structures and algorithms.</li>
<li>Proficiency in C/C++, Java, Python, or Golang.</li>
<li>Ability to design distributed systems and reason about application scalability and cloud environments.</li>
</ul>
<h2>Preferred Qualifications</h2><ul><li>Excellent collaboration and communication skills.</li></ul>
<h2>Disclaimer</h2><p>Legal and other company information that is not part of the job description.</p>
<h2>Similar jobs</h2><p>Other roles</p>
</main></body></html>"""

OFFICIAL_CISCO_HTML = r"""<!doctype html><html><head>
<title>Software Engineer | Cisco Careers</title>
<script type="application/ld+json">
{
 "@context":"https://schema.org", "@type":"JobPosting",
 "title":"Software Engineer",
 "identifier":{"@type":"PropertyValue","name":"Cisco","value":"2028048"},
 "employmentType":"FULL_TIME",
 "hiringOrganization":{"@type":"Organization","name":"Cisco"},
 "jobLocation":{"@type":"Place","address":{"@type":"PostalAddress","addressLocality":"Bangalore","addressRegion":"Karnataka","addressCountry":"India"}},
 "url":"https://cisco.wd5.myworkdayjobs.com/Cisco_Careers/job/Bangalore-India/Software-Engineer_2028048-1",
 "description":"<p>Cisco is hiring a Software Engineer in Bengaluru, Karnataka, India. The role focuses on scalable cloud applications, distributed systems, and product development. Work with a global software team and deliver quality software through its lifecycle.</p><h2>Responsibilities</h2><ul><li>Improve products and services using software engineering best practices.</li><li>Work independently with guidance and determine an appropriate approach.</li><li>Lead small-to-medium features from technical design through completion.</li><li>Perform coding, debugging, testing, and troubleshooting throughout development.</li></ul><h2>Minimum Qualifications</h2><ul><li>Bachelor's degree in Computer Science or related field.</li><li>Strong grasp of data structures and algorithms.</li><li>Proficiency in C/C++, Java, Python, or Golang.</li><li>Ability to design distributed systems and reason about scalability and cloud environments.</li></ul><h2>Preferred Qualifications</h2><ul><li>Excellent collaboration and communication skills.</li></ul>"
}
</script></head><body><h1>Software Engineer</h1></body></html>"""


class FakeResponse:
    def __init__(
        self, payload: bytes, url: str, content_type: str = "text/html; charset=utf-8"
    ):
        self.payload = payload
        self.url = url
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self):
        return self.url

    def read(self, n=-1):
        return self.payload if n < 0 else self.payload[:n]


class JDIngestionTests(unittest.TestCase):
    def test_linkedin_search_url_normalizes_to_direct_posting(self):
        url, job_id = normalize_job_url(
            "https://www.linkedin.com/jobs/search-results/?currentJobId=4477656621&trackingId=abc&keywords=C%2B%2B"
        )
        self.assertEqual(url, "https://www.linkedin.com/jobs/view/4477656621/")
        self.assertEqual(job_id, "4477656621")

    def test_extracts_schema_jobposting_into_existing_schema(self):
        with patch(
            "job_agent.jd_ingestion.urlopen",
            return_value=FakeResponse(
                JOB_HTML.encode(), "https://careers.example.com/jobs/JOB-1234"
            ),
        ):
            job = ingest_job_url(
                "https://careers.example.com/jobs/JOB-1234", search_fallback=False
            )
        self.assertEqual(job["company"], "Example Systems")
        self.assertEqual(job["title"], "Software Engineer")
        self.assertEqual(job["country"], "IN")
        self.assertEqual(job["employment_type"], "full_time")
        self.assertEqual(job["source"]["job_id"], "JOB-1234")
        self.assertEqual(job["source"]["company_job_code"], "JOB-1234")
        self.assertEqual(job["source"]["jd_extraction_status"], "structured")
        self.assertTrue(job["processing_ready"])
        self.assertTrue(
            any(
                "Design and implement C++ services." in x
                for x in job["job_description"]["responsibilities"]
            )
        )
        self.assertTrue(
            any(
                "C++ and Linux" in x
                for x in job["job_description"]["basic_qualifications"]
            )
        )
        self.assertTrue(
            any(
                "Kubernetes" in x
                for x in job["job_description"]["preferred_qualifications"]
            )
        )
        self.assertEqual(job["fields"], [])
        self.assertEqual(job["eligibility_status"], "needs_review")
        self.assertIsNone(job["overall_score"])

    def test_linkedin_metadata_extracts_company_title_location_but_not_fake_requisition(
        self,
    ):
        with (
            patch(
                "job_agent.jd_ingestion.urlopen",
                return_value=FakeResponse(
                    LINKEDIN_METADATA_ONLY.encode(),
                    "https://www.linkedin.com/jobs/view/4477656621/",
                ),
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            job = ingest_job_url(
                "https://www.linkedin.com/jobs/view/4477656621/", search_fallback=False
            )
        self.assertEqual(job["company"], "Cisco")
        self.assertEqual(job["title"], "Software Engineer")
        self.assertEqual(job["location"], "Bengaluru, Karnataka, India")
        self.assertEqual(job["country"], "IN")
        self.assertEqual(job["source"]["job_id"], "4477656621")
        self.assertIsNone(job["source"]["company_job_code"])
        self.assertEqual(job["source"]["jd_extraction_status"], "needs_review")
        self.assertFalse(job["processing_ready"])

    def test_linkedin_visible_full_description_is_usable_without_search(self):
        with patch(
            "job_agent.jd_ingestion.urlopen",
            return_value=FakeResponse(
                LINKEDIN_FULL_VISIBLE.encode(),
                "https://www.linkedin.com/jobs/view/4477656621/",
            ),
        ):
            job = ingest_job_url(
                "https://www.linkedin.com/jobs/view/4477656621/", search_fallback=False
            )
        self.assertEqual(job["company"], "Cisco")
        self.assertEqual(job["title"], "Software Engineer")
        self.assertEqual(job["source"]["jd_extraction_method"], "visible_page_text")
        self.assertEqual(job["source"]["jd_extraction_status"], "visible_page")
        self.assertTrue(job["processing_ready"])
        self.assertTrue(job["job_description"]["responsibilities"])
        self.assertTrue(job["job_description"]["basic_qualifications"])
        self.assertNotIn("Similar jobs", job["job_description"]["description_text"])

    def test_cli_can_smoke_test_brave_without_a_job_url(self):
        result = {
            "ok": True,
            "provider": "brave_search_api",
            "query": "Cisco Software Engineer Bengaluru Karnataka India",
            "result_count": 1,
            "results": [
                {
                    "title": "Cisco Software Engineer",
                    "url": "https://careers.cisco.example/job/123",
                    "description": "C++ software role in Bengaluru.",
                }
            ],
        }
        output = StringIO()
        with (
            patch(
                "job_agent.jd_ingestion.test_brave_search_connection",
                return_value=result,
            ),
            redirect_stdout(output),
        ):
            code = main(["--test-brave"])
        self.assertEqual(code, 0)
        self.assertIn("Brave Search API: CONNECTED", output.getvalue())
        self.assertIn("Results returned: 1", output.getvalue())

    def test_company_marketing_after_preferred_qualifications_is_removed(self):
        description = (
            "Meet the Team\nBuild C++ distributed software for products.\n"
            "Responsibilities\nImplement software and test components.\n"
            "Preferred Qualifications\nExperience with Linux.\n"
            "Why Cisco?\nCisco has been innovating for 40 years.\n"
            "We are Cisco, and our power starts with you."
        )
        cleaned = _trim_non_job_boilerplate(description)
        self.assertIn("Preferred Qualifications", cleaned)
        self.assertIn("Experience with Linux.", cleaned)
        self.assertNotIn("Why Cisco?", cleaned)
        self.assertNotIn("innovating for 40 years", cleaned)

    def test_brave_search_connection_uses_configured_key_without_exposing_it(self):
        sample_results = [
            {
                "title": "Cisco Software Engineer - Bengaluru",
                "url": "https://jobs.cisco.example/software-engineer",
                "description": "C++ software engineer role in Bengaluru.",
            }
        ]
        with (
            patch(
                "job_agent.jd_ingestion._load_brave_api_key",
                return_value="secret-test-key",
            ),
            patch(
                "job_agent.jd_ingestion._brave_search", return_value=sample_results
            ) as search,
        ):
            result = test_brave_search_connection("Cisco Software Engineer Bengaluru")
        self.assertTrue(result["ok"])
        self.assertEqual(result["result_count"], 1)
        self.assertNotIn("secret-test-key", json.dumps(result))
        self.assertEqual(search.call_args.args[1], "secret-test-key")

    def test_metadata_only_uses_verified_company_careers_fallback(self):
        search_results = [
            {
                "title": "Software Engineer | Cisco Careers",
                "url": "https://cisco.wd5.myworkdayjobs.com/Cisco_Careers/job/Bangalore-India/Software-Engineer_2028048-1",
                "description": "Cisco Software Engineer Bangalore Bengaluru Karnataka India careers",
            }
        ]
        response_queue = [
            (LINKEDIN_METADATA_ONLY, "https://www.linkedin.com/jobs/view/4477656621/"),
            (OFFICIAL_CISCO_HTML, search_results[0]["url"]),
        ]

        def fetch(url, timeout=15.0):
            html, final_url = response_queue.pop(0)
            return html, final_url, "text/html; charset=utf-8"

        with (
            patch(
                "job_agent.jd_ingestion._load_brave_api_key", return_value="test-key"
            ),
            patch("job_agent.jd_ingestion._brave_search", return_value=search_results),
            patch("job_agent.jd_ingestion._fetch_html", side_effect=fetch),
        ):
            job = ingest_job_url("https://www.linkedin.com/jobs/view/4477656621/")
        self.assertEqual(job["company"], "Cisco")
        self.assertEqual(job["title"], "Software Engineer")
        self.assertEqual(job["source"]["job_id"], "4477656621")
        self.assertEqual(job["source"]["company_job_code"], "2028048")
        self.assertEqual(job["source"]["jd_extraction_status"], "recovered_official")
        self.assertEqual(
            job["source"]["jd_extraction_method"], "official_careers_fallback"
        )
        self.assertTrue(job["processing_ready"])
        self.assertEqual(
            job["jd_url"], "https://www.linkedin.com/jobs/view/4477656621/"
        )
        self.assertEqual(
            job["source"]["verified_job_page_url"], search_results[0]["url"]
        )
        self.assertEqual(
            job["source"]["jd_recovery"]["verification_status"],
            "verified_company_posting",
        )

    def test_no_search_key_keeps_metadata_only_job_blocked(self):
        with (
            patch(
                "job_agent.jd_ingestion.urlopen",
                return_value=FakeResponse(
                    LINKEDIN_METADATA_ONLY.encode(),
                    "https://www.linkedin.com/jobs/view/4477656621/",
                ),
            ),
            patch("job_agent.jd_ingestion._load_brave_api_key", return_value=""),
        ):
            job = ingest_job_url("https://www.linkedin.com/jobs/view/4477656621/")
        self.assertEqual(
            job["source"]["jd_recovery"]["verification_status"],
            "search_api_key_missing",
        )
        self.assertFalse(job["processing_ready"])

    def test_ambiguous_official_matches_are_not_auto_selected(self):
        result_a = {
            "title": "Software Engineer | Cisco Careers",
            "url": "https://jobs.cisco.com/job/a",
            "description": "Cisco Software Engineer Bengaluru Karnataka India",
        }
        result_b = {
            "title": "Software Engineer | Cisco Careers",
            "url": "https://cisco.wd5.myworkdayjobs.com/job/b",
            "description": "Cisco Software Engineer Bengaluru Karnataka India",
        }
        page_a = OFFICIAL_CISCO_HTML.replace("2028048", "1111111").replace(
            "2028048-1", "1111111-1"
        )
        page_b = OFFICIAL_CISCO_HTML.replace("2028048", "2222222").replace(
            "2028048-1", "2222222-1"
        )
        queue = [
            (LINKEDIN_METADATA_ONLY, "https://www.linkedin.com/jobs/view/4477656621/"),
            (page_a, result_a["url"]),
            (page_b, result_b["url"]),
        ]

        def fetch(url, timeout=15.0):
            html, final_url = queue.pop(0)
            return html, final_url, "text/html"

        with (
            patch(
                "job_agent.jd_ingestion._load_brave_api_key", return_value="test-key"
            ),
            patch(
                "job_agent.jd_ingestion._brave_search",
                return_value=[result_a, result_b],
            ),
            patch("job_agent.jd_ingestion._fetch_html", side_effect=fetch),
        ):
            job = ingest_job_url("https://www.linkedin.com/jobs/view/4477656621/")
        self.assertEqual(job["source"]["jd_extraction_status"], "ambiguous_match")
        self.assertEqual(
            job["source"]["jd_recovery"]["verification_status"], "ambiguous_match"
        )
        self.assertFalse(job["processing_ready"])

    def test_blocked_linkedin_page_can_recover_identity_from_index_and_jd_from_company_site(
        self,
    ):
        identity_result = {
            "title": "Software Engineer at Cisco — Bengaluru, Karnataka, India | LinkedIn Jobs",
            "url": "https://www.linkedin.com/jobs/view/4477656621/",
            "description": "Software Engineer at Cisco in Bengaluru, Karnataka, India",
        }
        official_result = {
            "title": "Software Engineer | Cisco Careers",
            "url": "https://cisco.wd5.myworkdayjobs.com/Cisco_Careers/job/Bangalore-India/Software-Engineer_2028048-1",
            "description": "Cisco Software Engineer Bangalore Bengaluru Karnataka India careers",
        }
        response_queue = [(OFFICIAL_CISCO_HTML, official_result["url"])]

        def fetch(url, timeout=15.0):
            html, final_url = response_queue.pop(0)
            return html, final_url, "text/html; charset=utf-8"

        fetch_calls = {"count": 0}

        def fetch_or_block(url, timeout=15.0):
            if fetch_calls["count"] == 0:
                fetch_calls["count"] += 1
                raise JobIngestionError("SOURCE_BLOCKED:403:blocked")
            return fetch(url, timeout)

        with (
            patch(
                "job_agent.jd_ingestion._load_brave_api_key", return_value="test-key"
            ),
            patch(
                "job_agent.jd_ingestion._brave_search",
                side_effect=[
                    [identity_result],
                    [official_result],
                    [official_result],
                    [official_result],
                ],
            ),
            patch("job_agent.jd_ingestion._fetch_html", side_effect=fetch_or_block),
        ):
            job = ingest_job_url("https://www.linkedin.com/jobs/view/4477656621/")
        self.assertEqual(job["company"], "Cisco")
        self.assertEqual(job["title"], "Software Engineer")
        self.assertEqual(job["source"]["jd_extraction_status"], "recovered_official")
        self.assertTrue(job["processing_ready"])
        self.assertEqual(
            job["source"]["jd_recovery"]["verification_status"],
            "verified_company_posting",
        )

    def test_official_fallback_with_wrong_city_is_rejected(self):
        wrong_city_html = OFFICIAL_CISCO_HTML.replace("Bangalore", "Hyderabad").replace(
            "Bengaluru", "Hyderabad"
        )
        search_result = {
            "title": "Software Engineer | Cisco Careers",
            "url": "https://cisco.wd5.myworkdayjobs.com/Cisco_Careers/job/Hyderabad/Software-Engineer_2028048-1",
            "description": "Cisco Software Engineer Hyderabad India careers",
        }
        response_queue = [
            (LINKEDIN_METADATA_ONLY, "https://www.linkedin.com/jobs/view/4477656621/"),
            (wrong_city_html, search_result["url"]),
        ]

        def fetch(url, timeout=15.0):
            html, final_url = response_queue.pop(0)
            return html, final_url, "text/html; charset=utf-8"

        with (
            patch(
                "job_agent.jd_ingestion._load_brave_api_key", return_value="test-key"
            ),
            patch("job_agent.jd_ingestion._brave_search", return_value=[search_result]),
            patch("job_agent.jd_ingestion._fetch_html", side_effect=fetch),
        ):
            job = ingest_job_url("https://www.linkedin.com/jobs/view/4477656621/")
        self.assertEqual(job["source"]["jd_extraction_status"], "needs_review")
        self.assertFalse(job["processing_ready"])
        self.assertEqual(
            job["source"]["jd_recovery"]["verification_status"], "no_verified_match"
        )

    def test_unverified_json_is_blocked_from_scoring(self):
        job = {
            "source": {"jd_extraction_status": "needs_review"},
            "job_description": {
                "summary": "Software Engineer at Cisco. Full-time role. Apply now."
            },
        }
        ready, reason = is_job_description_processable(job)
        self.assertFalse(ready)
        self.assertIn("needs_review", reason)

    def test_manually_verified_complete_job_is_processable(self):
        job = {
            "company": "Example Systems",
            "title": "Software Engineer",
            "location": "Bengaluru, Karnataka, India",
            "source": {"jd_extraction_status": "manual_verified"},
            "job_description": {
                "description_text": (
                    "Build reliable C++ distributed services for a production platform. "
                    "Design and implement services, write unit tests, troubleshoot incidents, "
                    "collaborate with engineering teams, and maintain documentation and observability. "
                    "The engineer will contribute to system design, integration, quality, and reliability. "
                    "Responsibilities include implementing maintainable software, reviewing technical tradeoffs, "
                    "supporting production releases, and improving service performance through measured changes."
                )
            },
        }
        ready, reason = is_job_description_processable(job)
        self.assertTrue(ready, reason)

    def test_rejects_non_public_or_non_http_url(self):
        with self.assertRaises(JobIngestionError):
            normalize_job_url("file:///etc/passwd")
        with self.assertRaises(JobIngestionError):
            normalize_job_url("http://127.0.0.1/private")

    def test_jsonld_graph_is_supported(self):
        parser = _PageParser()
        parser.feed(
            '<script type="application/ld+json">{"@graph":[{"@type":"Organization","name":"X"},{"@type":"JobPosting","title":"Engineer"}]}</script>'
        )
        posting = _parse_jsonld(parser.jsonld)
        self.assertIsNotNone(posting)
        self.assertEqual(posting["title"], "Engineer")

    def test_linkedin_title_parser_extracts_company_and_location(self):
        self.assertEqual(
            _parse_linkedin_title(
                "Software Engineer at Cisco — Bengaluru, Karnataka, India | LinkedIn Jobs"
            ),
            ("Software Engineer", "Cisco", "Bengaluru, Karnataka, India"),
        )

    def test_generic_job_aggregator_is_not_accepted_as_official_source(self):
        self.assertFalse(
            _likely_official_or_company_ats(
                {
                    "title": "Software Engineer at Cisco",
                    "url": "https://www.indeed.com/viewjob?jk=123",
                    "description": "Cisco Software Engineer Bengaluru",
                },
                "Cisco",
            )
        )


if __name__ == "__main__":
    unittest.main()
