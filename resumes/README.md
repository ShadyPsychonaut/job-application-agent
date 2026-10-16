# Add your resumes here

Place the current, factual resume files in this folder. Supported formats:

- `robotics_resume.docx` or `robotics_resume.pdf`
- `backend_resume.docx` or `backend_resume.pdf`
- `.txt` and `.md` are also supported

The agent checks files on each scoring/process run using SHA-256. If a file changes, it re-extracts the text; if unchanged, the locally cached text is reused. The cache is stored at `data/resume_cache.json` and should remain private. Files are never uploaded unless LLM scoring is enabled and `JOB_AGENT_SHARE_RESUME_WITH_LLM=true` is set in `.env` or the shell environment.

Filename hints are used only to label resume variants (`robotics`, `backend`, `hybrid`); use clear filenames. If you have a single resume, any supported filename works.
