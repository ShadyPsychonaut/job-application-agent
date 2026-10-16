# Job Application Agent

A local Python assistant for evidence-grounded job-fit scoring, persistent application answers, missing-field prompts, and CSV application tracking. This milestone adds optional LLM-assisted scoring and resume-file ingestion. It still does **not** scrape LinkedIn or submit applications.

## Important: where candidate information currently comes from

The current `data/profile.json` contains manually configured career context and evidence-strength estimates. These numbers are rubric inputs, not facts inferred automatically from a resume. An available copy of the backend resume is included as `resumes/backend_resume.pdf`; replace it if you have a newer version. A separate robotics resume has not been added.

The agent now extracts a small set of explicit facts locally (education credentials and years-of-experience statements) into `data/candidate_facts.json`. Each extracted fact records its source file/hash and a short evidence excerpt. It remains `needs_review` until you explicitly approve it; the scorer can use it as flagged evidence without silently treating it as verified. It does not infer skills or ownership from degree titles, and it does not infer work authorization from resume location.

Place any additional or updated resume files under `resumes/` (recommended filenames below). The agent checks the folder every time `score` or `process` runs. It calculates SHA-256 hashes; text extraction is repeated when a file changes and cached text is reused when the file has not changed. The text cache stays local at `data/resume_cache.json`.

Supported formats:

- PDF with selectable text (`pypdf` is required)
- DOCX
- TXT
- Markdown

Scanned image-only PDFs are not OCR-processed. Resume text and extracted facts stay local unless you explicitly enable resume-text sharing with Groq. The facts file does not contain the full resume text. To review detected credentials and experience statements, run `python3 main.py profile-facts`. To approve the currently displayed facts into `data/profile.json`, run `python3 main.py approve-facts` and confirm interactively. Approval changes only documented education/experience fields; it never changes work authorization.

## Add your resumes

Place files in `resumes/`, for example:

```text
resumes/
├── robotics_resume.docx
└── backend_resume.docx
```

Filename hints map `robotics`, `backend`/`platform`, or `hybrid`/`general` to a resume variant. With one resume, the filename can be anything supported. You can also configure `candidate_profile.resume_files` in `data/profile.json`, or pass files explicitly with repeatable `--resume` arguments.

Check discovery and cache behavior:

```bash
python3 main.py resume-status
```

Run once after adding a resume. A first run should show it as newly extracted; unchanged subsequent runs should report a cached copy. If you edit the resume, the changed hash causes a fresh extraction automatically.

Review locally extracted facts:

```bash
python3 main.py profile-facts
python3 main.py approve-facts
```

`profile-facts` refreshes and displays the candidate facts file with evidence excerpts. `approve-facts` shows the same facts and asks for confirmation before promoting them to the persistent profile. If a resume changes, its hash changes and its newly extracted facts return to `needs_review`.

## Set up Groq LLM scoring

The LLM integration uses Groq's OpenAI-compatible Chat Completions API at `https://api.groq.com/openai/v1/chat/completions`. It uses JSON Object Mode plus local schema validation, so it works with models that do not support Groq's constrained JSON-schema mode. No Groq SDK is needed; the HTTP call uses Python's standard library. See the [official Groq OpenAI compatibility guide](https://console.groq.com/docs/openai) and [Structured Outputs docs](https://console.groq.com/docs/structured-outputs).

1. Copy `.env.example` to `.env` (`cp .env.example .env`).
2. Add your key from the [Groq Console](https://console.groq.com/keys) to `GROQ_API_KEY`.
3. `GROQ_MODEL=llama-3.3-70b-versatile` is configured because that is the Llama 3.3 70B model you requested.
4. To explicitly permit sending actual resume text to Groq, change `JOB_AGENT_SHARE_RESUME_WITH_LLM=false` to `true`.
5. Run `python3 main.py score examples/sample_job.json --mode llm` to require an LLM assessment.

**Important model-availability note (checked 13 October 2026):** Groq's model page currently labels `llama-3.3-70b-versatile` Enterprise, and Groq's deprecation notes say it was deprecated for free/developer tiers effective 16 August 2026. If your Groq project/key is not entitled to it, the request will fail with an access/model error. The current documented replacement for Llama 3.3 70B is `openai/gpt-oss-120b`, also available on Groq and supporting strict JSON-schema outputs. To use that instead, change `GROQ_MODEL=openai/gpt-oss-120b` in `.env`. The agent does not silently switch models; this avoids misrepresenting which model scored a job. Sources: [Groq model card](https://console.groq.com/docs/model/llama-3.3-70b-versatile), [Groq model deprecations](https://console.groq.com/docs/deprecations), [supported models](https://console.groq.com/docs/models).

A Groq API key and any applicable API usage limits/billing are separate from a ChatGPT subscription. Do not put `.env` or your API key in source control or share it in chat.

**Privacy behavior:** without a `GROQ_API_KEY`, `auto` mode uses the local rules scorer and makes no API call. With a key, `auto` uses Groq for semantic fit, sending the JD and configured candidate-profile facts. Actual resume text is kept out of the API request unless `JOB_AGENT_SHARE_RESUME_WITH_LLM=true`. If enabled, actual resume text is sent to Groq after email addresses, phone numbers, and LinkedIn/GitHub profile URLs are redacted. Other resume content, including names and employment details, may remain. Requests do not write prompts/resume text to the local application log, and the API integration does not request stateful conversation storage. The local extracted-text cache is private and should not be uploaded or committed.

## Quick start

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py init
python main.py resume-status
```

The `requirements.txt` entry is only used to extract PDFs. DOCX/TXT/MD extraction uses the Python standard library.

### Score one job

Use automatic mode (LLM when configured, local rules otherwise):

```bash
python3 main.py score examples/sample_job.json
```

Force the local rules scorer, without a network call:

```bash
python3 main.py score examples/sample_job.json --mode rules
```

Require the LLM and show errors rather than silently falling back:

```bash
python3 main.py score examples/sample_job.json --mode llm
```

Pass a resume explicitly (repeat `--resume` to use more than one):

```bash
python3 main.py score examples/sample_job.json --resume resumes/robotics_resume.docx --resume resumes/backend_resume.docx
```

Write the scorecard back into the job JSON:

```bash
python3 main.py score examples/sample_job.json --write
```

### Process a job

```bash
python3 main.py process examples/sample_job.json
```

Processing calculates the score first. Jobs must score **strictly greater than 8.5/10** to advance to application-field prompts. A hard eligibility block cannot be overridden by a high semantic score. Work authorization and sponsorship remain deterministic checks; the LLM is not allowed to decide or infer legal eligibility.

Use `--mode`, `--resume`, and the testing-only `--force-fields` options as in the score command.

## How the scoring works

Five fit-category weights are configured in `data/profile.json` and total 100%:

- Technical skills: 28%
- Responsibility match: 28%
- Experience evidence: 22%
- Seniority match: 11%
- Domain relevance: 11%

**Eligibility is a separate gate, not a sixth weighted category.** It cannot lower or inflate technical fit. A likely hard blocker can still stop processing regardless of the fit score, and uncertain authorization routes the job to manual review. The local rules calculate the score after normalizing older profiles' five fit weights. Jobs advance only when role-fit score is strictly greater than 8.5/10.

When LLM scoring is active, the model scores the five fit categories and returns an evidence map showing each requirement, the evidence cited, its strength (`strong`, `partial`, `missing`, or `unclear`) and its source. If actual resume text was not sent, the application labels evidence as `profile_only`; it does not claim the model read the resume. The application calculates a separate, local evidence-confidence estimate based on the number and quality of evidence-map items and whether resume text was included. This is coverage confidence, not a hiring probability.

The prompt treats actual resume text as the primary source for career claims and the profile as supplemental context. It directs the model not to equate team contribution with primary ownership, and not to treat personal ROS 2/Nav2 project work as production autonomy algorithm ownership. The scorecard includes gaps and uncertainties for manual review. LLM outputs are still estimates, not hiring probabilities; inspect them before tailoring a resume or applying.

If Groq is unavailable in `auto` mode, the agent falls back to the deterministic keyword/evidence scorer and records that fallback. `--mode llm` surfaces errors instead.

## Persistent application answers and log

- `data/answer_bank.json`: approved, reusable answers and applicability rules.
- `data/pending_questions.json`: unresolved application fields that need your input.
- `data/applications.json`: detailed local application records and answers.
- `data/application_log.csv`: spreadsheet-friendly tracker.

Expected compensation remains unset. Work authorization is reused only from an approved answer whose country/context matches. Sponsorship is a separate question. Even when a role falls below the tailoring threshold, processing queues reusable profile questions (work authorization, sponsorship, relocation, notice period, and compensation when present) so you can build the answer bank. It does not queue application-specific narrative fields or start resume tailoring for below-threshold roles. Reprocessing a duplicate JD can backfill missing intake questions. The agent never marks a job as applied; submission approval remains manual.

Example commands:

```bash
python3 main.py pending
python3 main.py answer q_0123456789
python3 main.py applications
```

Use the actual question ID printed by the agent.

## Sample job

`examples/sample_job.json` is the Amazon SDE II listing resolved from LinkedIn job ID `4477845127`. The URL's search keywords mention “senior robotics software engineer”, but the selected job itself is an Amazon software-development role. It is a sample to exercise processing; it is not being presented as a recommended vacancy. Application fields are explicitly marked as unverified because the publicly visible listing did not expose the real form fields.

## Tests

Run the unit test suite:

```bash
python3 -m unittest discover -s tests -v
```

Tests cover scoring, threshold gates, eligibility protection, answer reuse, resume extraction cache invalidation, and mocked LLM-score integration. Tests do not make a live API call. To verify a real LLM call, configure `.env` and run `python3 main.py score examples/sample_job.json --mode llm`.

## Safety and current limitations

- No LinkedIn scraping/search harvesting, resume rewriting, live browser form filling, or automatic submission yet.
- Evidence-strength estimates and career-scope notes remain manually configured; the local parser only extracts explicit education/experience statements and does not infer technical ownership.
- An LLM can misread or overrate evidence. Review category scores, evidence-map items, confidence, gaps, and the actual JD.
- Extracted education is available for review, but remains pending until you approve it. India work authorization remains unknown unless you separately verify it in `data/profile.json` or the approved answer bank.
- Do not store passwords, API keys, or authentication tokens in application records.
- Protect `data/` and `resumes/` as personal information. `data/resume_cache.json` contains extracted resume text and is excluded from version control.
