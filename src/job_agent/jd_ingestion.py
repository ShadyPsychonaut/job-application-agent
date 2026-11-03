"""Public job-posting ingestion into the Job Application Agent's JSON schema.

This module uses the Python standard library and optionally the Brave Search
API. It prefers Schema.org JobPosting JSON-LD, then visible page text, and can
recover from a verified company-careers/ATS page. It never invents application
form questions or qualifications and never bypasses access controls.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

MAX_HTML_BYTES = 5 * 1024 * 1024
USER_AGENT = (
    "JobApplicationAgent/0.1 (+public-job-posting-ingestion; respect site access rules)"
)


class JobIngestionError(ValueError):
    """Raised when a public job page cannot be ingested reliably."""


def _validate_public_http_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"}:
        raise JobIngestionError("Job URL must start with http:// or https://.")
    if not parsed.hostname:
        raise JobIngestionError("Job URL has no hostname.")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local")):
        raise JobIngestionError("Local/private hostnames are not supported.")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (not ip.is_global):
        raise JobIngestionError(
            "Private, loopback, or reserved IP URLs are not supported."
        )
    return url


def normalize_job_url(url: str) -> tuple[str, str | None]:
    """Normalize common LinkedIn search-result URLs to a single job URL.

    Returns (normalized_url, detected_job_id). This does not bypass access
    controls; it merely rewrites LinkedIn's public currentJobId parameter to
    the conventional direct-posting URL before making a normal public GET.
    """
    url = _validate_public_http_url(url)
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    query = parse_qs(parsed.query)
    job_id: str | None = None

    if host == "linkedin.com" or host.endswith(".linkedin.com"):
        match = re.search(r"/jobs/view/(\d+)", parsed.path)
        if match:
            job_id = match.group(1)
        if not job_id:
            values = query.get("currentJobId") or query.get("jobId") or []
            if values and re.fullmatch(r"\d+", values[0]):
                job_id = values[0]
        if job_id:
            return f"https://www.linkedin.com/jobs/view/{job_id}/", job_id

    # Remove common click/tracking parameters, while preserving unknown query
    # fields that may be part of a site's actual job identifier.
    tracking_prefixes = ("utm_", "trk", "tracking", "refId", "lipi", "eBP")
    kept = []
    for key, values in query.items():
        if key.casefold().startswith(tuple(x.casefold() for x in tracking_prefixes)):
            continue
        for value in values:
            kept.append((key, value))
    new_query = urlencode(kept, doseq=True)
    normalized = urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, "", new_query, "")
    )
    path_match = re.search(
        r"(?:jobs?/|position/|requisition/)([A-Za-z0-9_-]{4,})", parsed.path, re.I
    )
    if path_match:
        job_id = path_match.group(1)
    return normalized, job_id


class _PageParser(HTMLParser):
    BLOCKS = {
        "address",
        "article",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "section",
        "tr",
        "ul",
    }
    SKIP = {"script", "style", "noscript", "svg", "iframe"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0
        self._skip_tags: list[str] = []
        self.visible: list[str] = []
        self.jsonld: list[str] = []
        self._active_jsonld = False
        self._jsonld_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k.casefold(): (v or "") for k, v in attrs}
        tag = tag.casefold()
        if tag == "meta":
            key = attr.get("property") or attr.get("name") or attr.get("itemprop")
            content = attr.get("content", "").strip()
            if key and content:
                self.meta[key.casefold()] = content
        if tag == "title":
            self._in_title = True
        if tag == "script" and "ld+json" in attr.get("type", "").casefold():
            self._active_jsonld = True
            self._jsonld_depth = 1
            self.jsonld.append("")
            return
        if self._active_jsonld and tag == "script":
            self._jsonld_depth += 1
        if not self._active_jsonld and tag in self.SKIP:
            self._skip_depth += 1
            self._skip_tags.append(tag)
        if not self._active_jsonld and not self._skip_depth and tag in self.BLOCKS:
            self.visible.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "title":
            self._in_title = False
        if self._active_jsonld and tag == "script":
            self._jsonld_depth -= 1
            if self._jsonld_depth <= 0:
                self._active_jsonld = False
                self._jsonld_depth = 0
            return
        if not self._active_jsonld and self._skip_depth and tag in self._skip_tags:
            self._skip_depth -= 1
            try:
                self._skip_tags.remove(tag)
            except ValueError:
                pass
        if not self._active_jsonld and not self._skip_depth and tag in self.BLOCKS:
            self.visible.append("\n")

    def handle_data(self, data: str) -> None:
        if self._active_jsonld and self.jsonld:
            self.jsonld[-1] += data
            return
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        if data.strip():
            self.visible.append(data.strip())


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(part for item in value if (part := _clean_text(item)))
    text = str(value)
    parser = _PageParser()
    # For HTML-like content, parse visible text. Plain text remains plain text.
    if "<" in text and ">" in text:
        try:
            parser.feed(text)
            text = "\n".join(parser.visible)
        except Exception:
            text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text).replace("\xa0", " ").replace("\u200b", "")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[\t\r ]+", " ", line).strip()
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    return "\n".join(lines).strip()


def _walk_job_postings(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, list):
        for item in obj:
            yield from _walk_job_postings(item)
    elif isinstance(obj, dict):
        kind = obj.get("@type", [])
        if isinstance(kind, str):
            kind = [kind]
        kinds = {str(x).rstrip("/").split("/")[-1].casefold() for x in kind}
        if "jobposting" in kinds:
            yield obj
        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from _walk_job_postings(value)


def _parse_jsonld(blobs: list[str]) -> dict[str, Any] | None:
    for blob in blobs:
        if not blob.strip():
            continue
        try:
            obj = json.loads(blob.strip())
        except json.JSONDecodeError:
            # Some publishers wrap JSON-LD in comments; remove common wrappers.
            candidate = re.sub(r"^\s*<!--|-->\s*$", "", blob.strip())
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
        postings = list(_walk_job_postings(obj))
        if postings:
            # Prefer records with a title, employer and substantial description.
            postings.sort(
                key=lambda p: (
                    bool(p.get("title")),
                    bool(p.get("hiringOrganization")),
                    len(_clean_text(p.get("description", ""))),
                ),
                reverse=True,
            )
            return postings[0]
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [x for item in value if (x := _clean_text(item))]
    if isinstance(value, dict):
        # Common JobPosting fields sometimes use arrays/objects rather than strings.
        for key in ("name", "description", "value", "text"):
            if value.get(key):
                return _as_list(value[key])
        return [_clean_text(value)] if _clean_text(value) else []
    text = _clean_text(value)
    if not text:
        return []
    # Split list-like fields, but preserve a paragraph as one item.
    if "\n" in text:
        return [
            line.strip(" •\t-–") for line in text.splitlines() if line.strip(" •\t-–")
        ]
    return [text]


def _field_heading(line: str) -> str | None:
    s = re.sub(r"[\s:—–-]+$", "", line.strip().casefold())
    s = re.sub(r"^#+\s*", "", s)
    if len(s) > 100:
        return None
    # Company-culture and legal boilerplate is not a job qualification.
    if re.match(
        r"^(?:why\s+.{1,70}\??|about\s+(?:the\s+)?(?:company|us|[a-z0-9&.-]{2,40})|"
        r"equal opportunity(?: employer)?|eeo statement|diversity and inclusion)$",
        s,
    ):
        return "ignore"
    if re.match(
        r"^(preferred qualifications|preferred skills|preferred experience|nice to have|desired qualifications|bonus points|desirable qualifications|good to have)\b",
        s,
    ):
        return "preferred"
    if re.match(
        r"^(basic qualifications|minimum qualifications|minimum requirements|required qualifications|required skills|requirements|what you bring|what we.re looking for|what you.ll bring|what you will bring|qualifications|education requirements|experience requirements|must have|minimum qualifications)\b",
        s,
    ):
        return "basic"
    if re.match(
        r"^(responsibilities|what you.ll do|what you will do|key responsibilities|key duties|your responsibilities|duties|role and responsibilities|the role|how you.ll contribute|your impact|what you will work on)\b",
        s,
    ):
        return "responsibilities"
    if re.match(
        r"^(about the role|job summary|overview|position summary|about this role)\b", s
    ):
        return "summary"
    return None


def _trim_non_job_boilerplate(description: str) -> str:
    """Remove trailing company-culture/legal sections after job qualifications."""
    kept: list[str] = []
    for line in description.splitlines():
        if _field_heading(line) == "ignore":
            break
        kept.append(line)
    return "\n".join(kept).strip()


def _extract_sections(description: str) -> tuple[str, list[str], list[str], list[str]]:
    description = _trim_non_job_boilerplate(description)
    lines = [line.strip() for line in description.splitlines() if line.strip()]
    summary_lines: list[str] = []
    responsibilities: list[str] = []
    basic: list[str] = []
    preferred: list[str] = []
    current = "summary"
    for line in lines:
        heading = _field_heading(line)
        if heading:
            current = heading
            continue
        # Drop common navigation/boilerplate, not substantive content.
        if line.casefold() in {
            "apply now",
            "apply",
            "share this job",
            "back to search results",
        }:
            continue
        if current == "summary":
            summary_lines.append(line)
        elif current == "responsibilities":
            responsibilities.append(line.lstrip("•*-– "))
        elif current == "basic":
            basic.append(line.lstrip("•*-– "))
        elif current == "preferred":
            preferred.append(line.lstrip("•*-– "))
        # current == "ignore": intentionally discard employer boilerplate.
    summary = " ".join(summary_lines)
    if not summary and lines:
        summary = " ".join(lines[:3])
    # If there are no recognizable sections, preserve the text as context in
    # summary and do not pretend to know which lines are requirements.
    return summary[:1800], _dedupe(responsibilities), _dedupe(basic), _dedupe(preferred)


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = re.sub(r"\s+", " ", item).strip()
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def _country_code(address: Any, location_text: str) -> str:
    country_name = ""
    if isinstance(address, dict):
        country = address.get("addressCountry", "")
        if isinstance(country, dict):
            country_name = str(country.get("name") or country.get("identifier") or "")
        else:
            country_name = str(country or "")
    normalized = re.sub(r"[^a-z]", "", country_name.casefold())
    mapping = {
        "india": "IN",
        "in": "IN",
        "unitedstates": "US",
        "usa": "US",
        "us": "US",
        "unitedkingdom": "GB",
        "uk": "GB",
        "greatbritain": "GB",
        "germany": "DE",
        "de": "DE",
        "canada": "CA",
        "japan": "JP",
        "singapore": "SG",
        "australia": "AU",
        "france": "FR",
        "ireland": "IE",
        "netherlands": "NL",
        "czechia": "CZ",
        "czechrepublic": "CZ",
        "poland": "PL",
        "sweden": "SE",
        "switzerland": "CH",
        "spain": "ES",
        "italy": "IT",
        "brazil": "BR",
    }
    if normalized in mapping:
        return mapping[normalized]
    if re.fullmatch(r"[A-Za-z]{2}", country_name.strip()):
        return country_name.strip().upper()
    loc = location_text.casefold()
    for phrase, code in (
        ("india", "IN"),
        ("united states", "US"),
        (" usa", "US"),
        ("united kingdom", "GB"),
        ("germany", "DE"),
        ("canada", "CA"),
        ("singapore", "SG"),
        ("japan", "JP"),
        ("australia", "AU"),
    ):
        if phrase in loc:
            return code
    return ""


def _job_locations(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [
            x for item in value if isinstance(item, dict) for x in _job_locations(item)
        ]
    if isinstance(value, dict):
        typ = value.get("@type")
        if typ == "Place" or "address" in value:
            return [value]
    return []


def _location_text(posting: dict[str, Any], page_text: str = "") -> tuple[str, str]:
    places = _job_locations(posting.get("jobLocation"))
    address = places[0].get("address", {}) if places else {}
    if isinstance(address, dict):
        bits = [
            address.get(k) for k in ("addressLocality", "addressRegion", "postalCode")
        ]
        country = address.get("addressCountry", "")
        if isinstance(country, dict):
            country = country.get("name") or country.get("identifier") or ""
        if country:
            bits.append(country)
        location = ", ".join(str(b).strip() for b in bits if b)
        return location, _country_code(address, location)
    value = posting.get("jobLocationType")
    if value and str(value).upper() == "TELECOMMUTE":
        return "Remote", _country_code({}, page_text)
    return "", _country_code({}, page_text)


def _employment_type(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    raw = str(value or "").strip().upper().replace(" ", "_")
    mapping = {
        "FULL_TIME": "full_time",
        "PART_TIME": "part_time",
        "CONTRACTOR": "contract",
        "TEMPORARY": "temporary",
        "INTERN": "internship",
        "INTERNATIONAL": "full_time",
        "PER_DIEM": "contract",
        "OTHER": "other",
    }
    return mapping.get(raw, "other" if raw else "")


def _company_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "")
    return str(value or "")


def _job_identifier(posting: dict[str, Any], url_job_id: str | None) -> str | None:
    value = posting.get("identifier")
    if isinstance(value, dict):
        value = value.get("value") or value.get("name")
    if isinstance(value, list) and value:
        item = value[0]
        value = (
            item.get("value") or item.get("name") if isinstance(item, dict) else item
        )
    return str(value or url_job_id or "") or None


def _fetch_html(url: str, timeout: float = 15.0) -> tuple[str, str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/ld+json;q=0.9,*/*;q=0.7",
            "Accept-Language": "en-US,en;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            final_url = response.geturl()
            _validate_public_http_url(final_url)
            content_type = response.headers.get("Content-Type", "")
            raw = response.read(MAX_HTML_BYTES + 1)
            if len(raw) > MAX_HTML_BYTES:
                raise JobIngestionError(
                    f"The job page exceeded the {MAX_HTML_BYTES // (1024 * 1024)} MB size limit."
                )
            charset_match = re.search(r"charset=([\w.-]+)", content_type, re.I)
            charset = charset_match.group(1) if charset_match else "utf-8"
            try:
                html = raw.decode(charset, errors="replace")
            except LookupError:
                html = raw.decode("utf-8", errors="replace")
            return html, final_url, content_type
    except HTTPError as exc:
        if exc.code in {401, 403, 407, 429}:
            raise JobIngestionError(
                f"SOURCE_BLOCKED:{exc.code}:The source returned HTTP {exc.code}. The page may require login "
                "or may block automated access. This tool will not bypass access controls."
            ) from exc
        raise JobIngestionError(
            f"Could not fetch job page: HTTP {exc.code} {exc.reason}."
        ) from exc
    except URLError as exc:
        raise JobIngestionError(f"Could not retrieve job page: {exc.reason}.") from exc
    except TimeoutError as exc:
        raise JobIngestionError("Timed out while retrieving the job page.") from exc


def _metadata_description(parser: _PageParser) -> str:
    for key in ("og:description", "description", "twitter:description"):
        if parser.meta.get(key):
            return _clean_text(parser.meta[key])
    return ""


def _linkedin_metadata(parser: _PageParser) -> tuple[str, str, str, str]:
    """Return (title, company, location, seniority/employment hint) from metadata."""
    raw_title = (
        parser.meta.get("og:title")
        or parser.meta.get("twitter:title")
        or "".join(parser.title_parts).strip()
    )
    title, company, location = _parse_linkedin_title(raw_title)
    if not location:
        visible = "\n".join(parser.visible)
        location_match = re.search(
            r"(?im)^([^\n]{2,80},\s*(?:India|United States|United Kingdom|Germany|Canada|Singapore|Japan|Australia))$",
            visible,
        )
        if location_match:
            location = location_match.group(1).strip()
    hint = "\n".join(parser.visible[:60])
    return title, company, location, hint


def _extract_visible_job_description(visible_lines: list[str]) -> str:
    """Trim navigation and recommendations from page-visible text where possible."""
    lines = [re.sub(r"\s+", " ", x).strip() for x in visible_lines if x.strip()]
    if not lines:
        return ""

    start_markers = {
        "about the job",
        "job description",
        "about this role",
        "meet the team",
        "your impact",
        "the role",
        "role overview",
        "overview",
    }
    stop_markers = {
        "similar jobs",
        "people also viewed",
        "show more show less",
        "show less",
        "disclaimer",
        "about linkedin",
        "about the company",
        "get notified about new",
    }
    start = 0
    for idx, line in enumerate(lines):
        if line.casefold().rstrip(":") in start_markers:
            start = idx
            break
    selected: list[str] = []
    for line in lines[start:]:
        folded = line.casefold().rstrip(":")
        if selected and folded in stop_markers:
            break
        if folded in {
            "apply",
            "apply now",
            "save",
            "share",
            "join now",
            "sign in",
            "agree & join linkedin",
            "see who you know",
            "back to search results",
        }:
            continue
        # Skip obvious LinkedIn navigation boilerplate.
        if folded.startswith(
            ("by clicking continue to join", "new to linkedin?", "join or sign in")
        ):
            continue
        selected.append(line)
    # Remove repeated adjacent text and page title/company/location chrome.
    cleaned = _dedupe(selected)
    return "\n".join(cleaned).strip()


def _load_brave_api_key() -> str:
    """Read the key from environment, with a minimal project .env fallback."""
    key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if key:
        return key
    # Prefer the current working directory, then the repository root inferred
    # from src/job_agent/jd_ingestion.py. This also works when invoked elsewhere.
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    seen: set[Path] = set()
    for env_path in candidates:
        try:
            resolved = env_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if not resolved.is_file():
                continue
            for line in resolved.read_text(encoding="utf-8").splitlines():
                match = re.match(
                    r"^\s*(?:export\s+)?BRAVE_SEARCH_API_KEY\s*=\s*(.*?)\s*$", line
                )
                if match:
                    value = match.group(1).strip().strip("'\"")
                    if value and not value.startswith("#"):
                        return value
        except OSError:
            continue
    return ""


def _brave_search(
    query: str,
    api_key: str,
    *,
    country: str = "IN",
    count: int = 8,
    timeout: float = 15.0,
) -> list[dict[str, str]]:
    """Search public web results using Brave's documented Web Search API."""
    params = urlencode(
        {
            "q": query[:600],
            "count": max(1, min(count, 20)),
            "country": country if re.fullmatch(r"[A-Z]{2}", country or "") else "IN",
            "search_lang": "en",
            "safesearch": "moderate",
        }
    )
    request = Request(
        f"https://api.search.brave.com/res/v1/web/search?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise JobIngestionError("Search provider response exceeded 2 MB.")
            data = json.loads(raw.decode("utf-8", errors="replace"))
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise JobIngestionError(
                "Brave Search API authentication failed; check BRAVE_SEARCH_API_KEY and its plan."
            ) from exc
        if exc.code == 429:
            raise JobIngestionError(
                "Brave Search API rate limit reached; retry later."
            ) from exc
        raise JobIngestionError(f"Brave Search API returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError) as exc:
        raise JobIngestionError(f"Brave Search API request failed: {exc}.") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise JobIngestionError("Brave Search API returned invalid JSON.") from exc

    results = data.get("web", {}).get("results", []) if isinstance(data, dict) else []
    normalized: list[dict[str, str]] = []
    for item in results if isinstance(results, list) else []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        normalized.append(
            {
                "title": _clean_text(item.get("title", "")),
                "url": str(item.get("url", "")),
                "description": _clean_text(item.get("description", "")),
                "source": str(
                    item.get("profile", {}).get("long_name", "")
                    if isinstance(item.get("profile"), dict)
                    else ""
                ),
            }
        )
    return normalized


def test_brave_search_connection(
    query: str = "Cisco Software Engineer Bengaluru Karnataka India",
    *,
    country: str = "IN",
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Run a minimal live Brave API request without printing or exposing the key."""
    api_key = _load_brave_api_key()
    if not api_key:
        raise JobIngestionError(
            "BRAVE_SEARCH_API_KEY was not found in the environment or project .env."
        )
    results = _brave_search(query, api_key, country=country, count=5, timeout=timeout)
    return {
        "ok": True,
        "provider": "brave_search_api",
        "query": query,
        "result_count": len(results),
        "results": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", "")[:240],
            }
            for item in results[:5]
        ],
    }


def _normal_words(text: str) -> set[str]:
    text = text.casefold().replace("bengaluru", "bangalore")
    return {
        x
        for x in re.findall(r"[a-z0-9]+", text)
        if x
        not in {
            "the",
            "and",
            "for",
            "at",
            "in",
            "of",
            "a",
            "an",
            "to",
            "job",
            "jobs",
            "careers",
            "career",
        }
    }


def _similarity(left: str, right: str) -> float:
    a, b = _normal_words(left), _normal_words(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _company_tokens(company: str) -> set[str]:
    return _normal_words(company) - {
        "inc",
        "ltd",
        "limited",
        "corp",
        "corporation",
        "technologies",
        "technology",
    }


def _is_aggregator_host(host: str) -> bool:
    blocked = (
        "linkedin.com",
        "indeed.com",
        "glassdoor.com",
        "ziprecruiter.com",
        "monster.com",
        "naukri.com",
        "foundit.in",
        "simplyhired.com",
        "jooble.org",
        "talent.com",
        "careerbuilder.com",
    )
    host = host.casefold()
    return any(host == domain or host.endswith("." + domain) for domain in blocked)


def _likely_official_or_company_ats(result: dict[str, str], company: str) -> bool:
    """Accept company domains or company-branded ATS paths, not generic aggregators."""
    url = result.get("url", "")
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if not host or _is_aggregator_host(host):
        return False
    tokens = _company_tokens(company)
    compact_host = re.sub(r"[^a-z0-9]", "", host)
    compact_path = re.sub(r"[^a-z0-9]", "", parsed.path.casefold())
    company_in_host = any(token in compact_host for token in tokens if len(token) >= 3)
    company_in_path = any(token in compact_path for token in tokens if len(token) >= 3)
    text = (result.get("title", "") + " " + result.get("description", "")).casefold()
    company_in_result = bool(tokens) and all(
        token in _normal_words(text) for token in tokens
    )
    ats_domains = (
        "myworkdayjobs.com",
        "greenhouse.io",
        "lever.co",
        "ashbyhq.com",
        "smartrecruiters.com",
        "jobvite.com",
        "icims.com",
        "successfactors.com",
        "oraclecloud.com",
        "bamboohr.com",
    )
    is_ats = any(host == d or host.endswith("." + d) for d in ats_domains)
    has_career_context = any(
        x in (host + parsed.path).casefold()
        for x in ("career", "job", "workday", "greenhouse", "lever", "ashby", "recruit")
    )
    # Company-owned domains are accepted directly. Generic ATS hosts must have
    # an employer-specific slug or employer identity in the search result.
    return company_in_host or (
        is_ats and has_career_context and (company_in_path or company_in_result)
    )


def _candidate_search_score(
    result: dict[str, str], *, company: str, title: str, location: str
) -> float:
    text = result.get("title", "") + " " + result.get("description", "")
    company_tokens = _company_tokens(company)
    company_score = (
        1.0
        if company_tokens
        and all(
            t in _normal_words(text + " " + result.get("url", ""))
            for t in company_tokens
        )
        else 0.0
    )
    title_score = _similarity(
        title, result.get("title", "") + " " + result.get("description", "")
    )
    location_score = _similarity(location, text) if location else 0.5
    return round(0.45 * company_score + 0.4 * title_score + 0.15 * location_score, 4)


def _search_queries(company: str, title: str, location: str) -> list[str]:
    identity = f'"{company}" "{title}"'
    phrases = [
        f'{identity} "{location}" official careers'
        if location
        else f"{identity} official careers",
        f'{identity} "{location}" jobs' if location else f"{identity} jobs",
        f'"{company}" careers "{title}"',
    ]
    return list(dict.fromkeys(q for q in phrases if company.strip() and title.strip()))


def _parse_page(
    html: str, final_url: str, content_type: str, url_job_id: str | None = None
) -> dict[str, Any]:
    parser = _PageParser()
    parser.feed(html)
    posting = _parse_jsonld(parser.jsonld)
    if posting:
        title = str(posting.get("title") or "").strip()
        company = _company_name(posting.get("hiringOrganization"))
        location, country = _location_text(
            posting, _clean_text(posting.get("description", ""))
        )
        employment_type = _employment_type(posting.get("employmentType"))
        post_date = str(posting.get("datePosted") or "")
        job_id = _job_identifier(posting, url_job_id)
        application_url = str(posting.get("url") or final_url)
        description_text = _trim_non_job_boilerplate(
            _clean_text(posting.get("description", ""))
        )
        method = "schema_org_jobposting_jsonld"
        if len(description_text) < 250:
            visible_description = _extract_visible_job_description(parser.visible)
            if (
                len(visible_description) >= 450
                and len(_normal_words(visible_description)) >= 35
            ):
                description_text = visible_description
                method = "visible_page_text_with_structured_identity"
        summary, responsibilities, basic, preferred = _extract_sections(
            description_text
        )
        if posting.get("responsibilities"):
            responsibilities = _as_list(posting.get("responsibilities"))
        qualifications: list[str] = []
        for key in (
            "qualifications",
            "educationRequirements",
            "experienceRequirements",
            "skills",
        ):
            if posting.get(key):
                qualifications.extend(_as_list(posting.get(key)))
        if qualifications:
            basic = _dedupe(basic + qualifications)
        if posting.get("experienceInPlaceOfEducation"):
            preferred = _dedupe(
                preferred + _as_list(posting.get("experienceInPlaceOfEducation"))
            )
        return {
            "title": title,
            "company": company,
            "location": location,
            "country": country,
            "employment_type": employment_type,
            "posted_date": post_date,
            "job_id": job_id,
            "company_job_code": _job_identifier(posting, None),
            "seniority_level": "",
            "application_url": application_url,
            "description_text": description_text,
            "summary": summary,
            "responsibilities": responsibilities,
            "basic_qualifications": basic,
            "preferred_qualifications": preferred,
            "method": method,
            "content_type": content_type,
            "final_url": final_url,
            "structured": method == "schema_org_jobposting_jsonld",
        }

    title, company, location, _hint = _linkedin_metadata(parser)
    if not title:
        title = (
            parser.meta.get("og:title")
            or parser.meta.get("twitter:title")
            or "".join(parser.title_parts).strip()
        )
    if not company:
        site_name = (
            parser.meta.get("og:site_name") or parser.meta.get("application-name") or ""
        )
        if site_name and "linkedin" not in site_name.casefold():
            company = site_name.strip()
        else:
            page_title = (
                parser.meta.get("og:title") or "".join(parser.title_parts).strip()
            )
            suffix_match = re.match(
                r"^(?P<title>.+?)\s+[|–—-]\s+(?P<company>.+?)(?:\s+(?:careers|jobs?))?$",
                page_title,
                re.I,
            )
            if suffix_match:
                title = suffix_match.group("title").strip()
                company = suffix_match.group("company").strip()
    visible_description = _extract_visible_job_description(parser.visible)
    metadata_description = _metadata_description(parser)
    # Visible page body is preferred over social metadata. Metadata snippets are not JDs.
    description_text = (
        _trim_non_job_boilerplate(visible_description)
        if len(visible_description) >= 250
        else ""
    )
    summary, responsibilities, basic, preferred = _extract_sections(description_text)
    visible_lower = (" ".join(parser.visible) + " " + metadata_description).casefold()
    employment_type = ""
    if re.search(r"\bfull[- ]time\b", visible_lower):
        employment_type = "full_time"
    elif re.search(r"\bpart[- ]time\b", visible_lower):
        employment_type = "part_time"
    elif re.search(r"\bcontract(?:or)?\b", visible_lower):
        employment_type = "contract"
    country = _country_code({}, location + " " + visible_description)
    seniority = ""
    for pattern, label in (
        (r"\bentry[- ]level\b", "Entry level"),
        (r"\bmid[- ]senior level\b", "Mid-Senior level"),
        (r"\bassociate\b", "Associate"),
        (r"\bdirector\b", "Director"),
        (r"\binternship\b", "Internship"),
    ):
        if re.search(pattern, visible_lower):
            seniority = label
            break
    return {
        "title": title,
        "company": company,
        "location": location,
        "country": country,
        "employment_type": employment_type,
        "posted_date": "",
        "job_id": url_job_id,
        "seniority_level": seniority,
        "company_job_code": None,
        "application_url": final_url,
        "description_text": description_text,
        "summary": summary or metadata_description,
        "responsibilities": responsibilities,
        "basic_qualifications": basic,
        "preferred_qualifications": preferred,
        "method": "visible_page_text" if description_text else "page_metadata_fallback",
        "content_type": content_type,
        "final_url": final_url,
        "structured": False,
    }


def _description_is_substantive(parsed: dict[str, Any]) -> bool:
    text = str(parsed.get("description_text") or "").strip()
    if len(text) < 250 or len(_normal_words(text)) < 35:
        return False
    boilerplate = re.sub(r"\s+", " ", text.casefold()).strip()
    if any(
        x in boilerplate
        for x in (
            "apply for software engineer at",
            "see responsibilities, qualifications, and similar jobs",
            "join or sign in to find your next job",
        )
    ):
        return False
    # Structured descriptions may legitimately be free-form; unstructured pages
    # should expose some recognizable content beyond a short preview/snippet.
    if not parsed.get("structured") and len(text) < 450:
        return False
    return True


def _location_matches(
    expected: str, actual: str, expected_country: str = "", actual_country: str = ""
) -> bool:
    if not expected:
        return bool(actual)
    if not actual:
        return False
    aliases = {
        "bengaluru": "bangalore",
        "gurugram": "gurgaon",
        "noida": "noida",
        "new delhi": "delhi",
        "delhi ncr": "delhi",
        "pune": "pune",
        "hyderabad": "hyderabad",
        "mumbai": "mumbai",
        "chennai": "chennai",
        "kolkata": "kolkata",
        "san francisco": "san francisco",
    }
    expected_city = expected.split(",", 1)[0].strip().casefold()
    actual_city = actual.split(",", 1)[0].strip().casefold()
    expected_city = aliases.get(expected_city, expected_city)
    actual_city = aliases.get(actual_city, actual_city)
    city_ok = (
        expected_city == actual_city
        or expected_city in actual_city
        or actual_city in expected_city
    )
    if not city_ok:
        return False
    expected_country = expected_country or _country_code({}, expected)
    actual_country = actual_country or _country_code({}, actual)
    if (
        expected_country
        and actual_country
        and expected_country.casefold() != actual_country.casefold()
    ):
        return False
    return True


def _validate_candidate(
    parsed: dict[str, Any], *, company: str, title: str, location: str
) -> tuple[bool, float, list[str]]:
    reasons: list[str] = []
    expected_company = _company_tokens(company)
    actual_company = _company_tokens(str(parsed.get("company", "")))
    company_ok = bool(
        expected_company
        and actual_company
        and (expected_company <= actual_company or actual_company <= expected_company)
    )
    title_score = _similarity(title, str(parsed.get("title", "")))
    location_ok = _location_matches(
        location,
        str(parsed.get("location", "")),
    )
    substantive = _description_is_substantive(parsed)
    if company_ok:
        reasons.append("Employer matches the source listing.")
    else:
        reasons.append(
            f"Employer mismatch or not independently identified: {parsed.get('company') or 'unknown'}."
        )
    if title_score >= 0.5:
        reasons.append(f"Job title similarity {title_score:.2f}.")
    else:
        reasons.append(f"Job title mismatch (similarity {title_score:.2f}).")
    if location_ok:
        reasons.append("Location matches or no conflicting location was found.")
    else:
        reasons.append(f"Location mismatch: {parsed.get('location') or 'unknown'}.")
    if substantive:
        reasons.append(
            "A substantive job description was extracted from the candidate page."
        )
    else:
        reasons.append("Candidate page did not expose a substantive job description.")
    confidence = (
        0.38 * float(company_ok)
        + 0.27 * title_score
        + 0.15 * float(location_ok)
        + 0.20 * float(substantive)
    )
    return (
        bool(company_ok and title_score >= 0.5 and location_ok and substantive),
        round(confidence, 4),
        reasons,
    )


def _parse_linkedin_title(raw_title: str) -> tuple[str, str, str]:
    raw = re.sub(r"\s*\|\s*LinkedIn Jobs?\s*$", "", raw_title, flags=re.I).strip()
    match = re.match(
        r"^(?P<title>.+?)\s+at\s+(?P<company>.+?)(?:\s+[—–-]\s+(?P<location>.+))?$",
        raw,
        flags=re.I,
    )
    if not match:
        return raw, "", ""
    return (
        match.group("title").strip(),
        match.group("company").strip(" -—–|"),
        (match.group("location") or "").strip(),
    )


def _recover_linkedin_identity(
    job_id: str, api_key: str, *, country: str = "IN", timeout: float = 15.0
) -> tuple[str, str, str, dict[str, Any]]:
    """Use indexed LinkedIn result metadata only to identify the job, never as its JD."""
    query = f'"{job_id}" LinkedIn job'
    results = _brave_search(query, api_key, country=country, count=8, timeout=timeout)
    candidates: list[tuple[str, str, str, dict[str, Any]]] = []
    for item in results:
        parsed_url = urlparse(item.get("url", ""))
        if "linkedin.com" not in (parsed_url.hostname or "").casefold():
            continue
        if job_id not in parsed_url.path and job_id not in item.get("url", ""):
            continue
        title, company, location = _parse_linkedin_title(item.get("title", ""))
        if title and company:
            candidates.append((title, company, location, item))
    unique = {(x[0].casefold(), x[1].casefold(), x[2].casefold()) for x in candidates}
    if len(unique) == 1 and candidates:
        title, company, location, item = candidates[0]
        return (
            title,
            company,
            location,
            {
                "identity_source_url": item.get("url"),
                "identity_search_query": query,
                "identity_verified_from_indexed_linkedin_result": True,
            },
        )
    return (
        "",
        "",
        "",
        {
            "identity_search_query": query,
            "identity_verified_from_indexed_linkedin_result": False,
            "identity_candidate_count": len(unique),
        },
    )


def _recover_from_official_page(
    *,
    company: str,
    title: str,
    location: str,
    country: str,
    api_key: str,
    timeout: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    query_records: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for query in _search_queries(company, title, location):
        try:
            results = _brave_search(
                query, api_key, country=country or "IN", count=8, timeout=timeout
            )
            query_records.append({"query": query, "result_count": len(results)})
            for result in results:
                if not _likely_official_or_company_ats(result, company):
                    continue
                normalized, _ = normalize_job_url(result["url"])
                result["url"] = normalized
                result["search_score"] = _candidate_search_score(
                    result, company=company, title=title, location=location
                )
                candidates.setdefault(normalized, result)
        except JobIngestionError as exc:
            errors.append(str(exc))
            break

    ranked = sorted(
        candidates.values(), key=lambda x: x.get("search_score", 0), reverse=True
    )[:8]
    checked: list[dict[str, Any]] = []
    valid: list[tuple[float, dict[str, Any], dict[str, Any], list[str]]] = []
    for result in ranked[:5]:
        try:
            html, final_url, content_type = _fetch_html(result["url"], timeout=timeout)
            parsed = _parse_page(html, final_url, content_type)
            if not parsed.get("company") and _likely_official_or_company_ats(
                result, company
            ):
                # The result has already passed the company-owned-domain or
                # company-branded ATS check. Record this provenance explicitly.
                parsed["company"] = company
                parsed["company_identity_source"] = (
                    "company_owned_domain_or_branded_ats"
                )
            accepted, confidence, reasons = _validate_candidate(
                parsed, company=company, title=title, location=location
            )
            if parsed.get("company_identity_source"):
                reasons.append(
                    "Employer identity was matched from the company-owned domain or company-branded ATS hostname/path."
                )
            checked.append(
                {
                    "title": result.get("title", ""),
                    "url": final_url,
                    "search_score": result.get("search_score"),
                    "match_confidence": confidence,
                    "accepted": accepted,
                    "reasons": reasons,
                    "company_identity_source": parsed.get(
                        "company_identity_source", "page_content"
                    ),
                }
            )
            if accepted:
                valid.append((confidence, parsed, result, reasons))
        except (JobIngestionError, OSError, ValueError) as exc:
            checked.append(
                {
                    "title": result.get("title", ""),
                    "url": result["url"],
                    "accepted": False,
                    "error": str(exc),
                }
            )

    valid.sort(key=lambda x: x[0], reverse=True)
    recovery = {
        "attempted": True,
        "provider": "brave_search_api",
        "queries": query_records,
        "candidate_count": len(candidates),
        "checked_candidates": checked,
        "search_errors": errors,
    }
    if not valid:
        recovery["verification_status"] = "no_verified_match"
        return None, recovery
    if len(valid) > 1 and (valid[0][0] - valid[1][0]) < 0.10:
        recovery["verification_status"] = "ambiguous_match"
        recovery["ambiguity_note"] = (
            "Multiple official/company ATS pages matched with similar confidence; no page was selected automatically."
        )
        return None, recovery
    confidence, parsed, result, reasons = valid[0]
    recovery.update(
        {
            "verification_status": "verified_company_posting",
            "recovered_from_url": parsed["final_url"],
            "recovered_page_title": result.get("title", ""),
            "recovered_job_id": parsed.get("job_id"),
            "match_confidence": confidence,
            "match_reasons": reasons,
        }
    )
    return parsed, recovery


def is_job_description_processable(job: dict[str, Any]) -> tuple[bool, str]:
    """Gate scoring/tailoring on a substantive, reviewed JD (not metadata snippets)."""
    source = job.get("source", {}) if isinstance(job, dict) else {}
    status = str(source.get("jd_extraction_status", "")).casefold()
    if status in {"ambiguous_match", "needs_review", "blocked"}:
        return (
            False,
            f"Job description extraction status is {status}; verify the source before scoring.",
        )
    if not str(job.get("company", "")).strip() or str(
        job.get("company", "")
    ).casefold().startswith("unknown"):
        return False, "Employer identity is missing or unverified."
    if not str(job.get("title", "")).strip():
        return False, "Job title is missing."
    if not str(job.get("location", "")).strip() or str(
        job.get("location", "")
    ).casefold().startswith("unknown"):
        return False, "Job location is missing or unverified."
    description = job.get("job_description", {}) if isinstance(job, dict) else {}
    text = str(
        description.get("description_text") or description.get("summary") or ""
    ).strip()
    if status not in {
        "structured",
        "visible_page",
        "recovered_official",
        "manual_verified",
    }:
        return False, "Job description has not been verified as substantive."
    if len(text) < 250 or len(_normal_words(text)) < 35:
        return (
            False,
            "Job description is too short to support a reliable fit assessment.",
        )
    if (
        status == "recovered_official"
        and source.get("jd_recovery", {}).get("verification_status")
        != "verified_company_posting"
    ):
        return (
            False,
            "Fallback source has not passed company, title, and location verification.",
        )
    return True, "Job description is sufficiently complete for scoring."


def ingest_job_url(
    url: str,
    *,
    timeout: float = 15.0,
    search_fallback: bool = True,
    search_api_key: str | None = None,
) -> dict[str, Any]:
    """Fetch a job page, then recover from a verified company/ATS page if needed.

    Search fallback uses Brave Search API when BRAVE_SEARCH_API_KEY is configured.
    It never bypasses login walls, captchas, or other access controls.
    """
    normalized_url, url_job_id = normalize_job_url(url)
    original_error = ""
    parsed: dict[str, Any] | None = None
    try:
        html, final_url, content_type = _fetch_html(normalized_url, timeout=timeout)
        parsed = _parse_page(html, final_url, content_type, url_job_id)
    except JobIngestionError as exc:
        original_error = str(exc)
        if not original_error.startswith("SOURCE_BLOCKED:"):
            # Network/HTTP failures can still be recovered through public search,
            # but preserve the original error if recovery is unavailable.
            pass

    if parsed is None:
        parsed = {
            "title": "",
            "company": "",
            "location": "",
            "country": "",
            "employment_type": "",
            "posted_date": "",
            "job_id": url_job_id,
            "application_url": normalized_url,
            "description_text": "",
            "summary": "",
            "responsibilities": [],
            "basic_qualifications": [],
            "preferred_qualifications": [],
            "method": "fetch_failed",
            "content_type": "",
            "final_url": normalized_url,
            "structured": False,
        }

    warnings: list[str] = []
    recovery: dict[str, Any] = {"attempted": False, "verification_status": "not_needed"}
    initial_complete = _description_is_substantive(parsed)
    identity_complete = bool(
        parsed.get("title") and parsed.get("company") and parsed.get("location")
    )
    if not initial_complete or not identity_complete:
        if not initial_complete:
            warnings.append(
                "The original posting did not expose a substantive job description; metadata snippets are not treated as the full JD."
            )
        if not identity_complete:
            warnings.append(
                "The original posting did not expose all identity fields (title, employer, location); attempting public-source verification."
            )
        if original_error:
            warnings.append(f"Original source retrieval: {original_error}")
        if search_fallback:
            key = (search_api_key or _load_brave_api_key()).strip()
            if not key:
                recovery = {
                    "attempted": False,
                    "provider": "brave_search_api",
                    "verification_status": "search_api_key_missing",
                    "message": "Set BRAVE_SEARCH_API_KEY to enable automatic official-careers/ATS fallback.",
                }
                warnings.append(
                    "Automatic official-careers fallback was not run because BRAVE_SEARCH_API_KEY is not configured."
                )
            else:
                identity_recovery: dict[str, Any] = {}
                if (
                    not parsed.get("company") or not parsed.get("title")
                ) and url_job_id:
                    try:
                        id_title, id_company, id_location, identity_recovery = (
                            _recover_linkedin_identity(
                                url_job_id,
                                key,
                                country=parsed.get("country") or "IN",
                                timeout=timeout,
                            )
                        )
                        if id_title and id_company:
                            parsed["title"] = id_title
                            parsed["company"] = id_company
                            parsed["location"] = parsed.get("location") or id_location
                            parsed["country"] = parsed.get("country") or _country_code(
                                {}, id_location
                            )
                    except JobIngestionError as exc:
                        identity_recovery = {"identity_search_error": str(exc)}
                if parsed.get("company") and parsed.get("title"):
                    recovered, recovery = _recover_from_official_page(
                        company=parsed.get("company", ""),
                        title=parsed.get("title", ""),
                        location=parsed.get("location", ""),
                        country=parsed.get("country", ""),
                        api_key=key,
                        timeout=timeout,
                    )
                    if identity_recovery:
                        recovery["identity_recovery"] = identity_recovery
                else:
                    recovered = None
                    recovery = {
                        "attempted": False,
                        "provider": "brave_search_api",
                        "verification_status": "insufficient_source_identity",
                        "message": "Company and title could not be identified from the original page or indexed LinkedIn result.",
                        "identity_recovery": identity_recovery,
                    }
                if recovered:
                    parsed = recovered
                    recovery["attempted"] = True
                else:
                    warnings.append(
                        "No unambiguous official company/ATS posting with a matching employer, title, and location was verified."
                    )
        else:
            warnings.append("Official-careers fallback was disabled by configuration.")

    title = str(parsed.get("title") or "").strip()
    company = str(parsed.get("company") or "").strip()
    location = str(parsed.get("location") or "").strip()
    country = str(parsed.get("country") or _country_code({}, location)).strip()
    employment_type = str(parsed.get("employment_type") or "").strip()
    description_text = str(parsed.get("description_text") or "").strip()
    summary = str(parsed.get("summary") or "").strip()
    responsibilities = parsed.get("responsibilities", []) or []
    basic = parsed.get("basic_qualifications", []) or []
    preferred = parsed.get("preferred_qualifications", []) or []

    if not title:
        raise JobIngestionError(
            "No job title could be extracted. The posting may be inaccessible; use a direct public careers URL or provide company/title/location hints."
        )
    if not company:
        warnings.append(
            "Employer name could not be verified from the original or fallback page."
        )
    if not location:
        warnings.append(
            "Job location could not be verified from the original or fallback page."
        )
    if not _description_is_substantive(parsed):
        warnings.append(
            "Description remains incomplete/metadata-only; do not process this record for fit scoring or tailoring."
        )

    if recovery.get("verification_status") == "verified_company_posting":
        extraction_method = "official_careers_fallback"
        extraction_status = "recovered_official"
    elif recovery.get("verification_status") == "ambiguous_match":
        extraction_method = "official_careers_fallback_ambiguous"
        extraction_status = "ambiguous_match"
    elif parsed.get("method") == "schema_org_jobposting_jsonld":
        extraction_method = parsed["method"]
        extraction_status = (
            "structured"
            if _description_is_substantive(parsed) and company and title and location
            else "needs_review"
        )
    elif parsed.get("method") == "visible_page_text_with_structured_identity":
        extraction_method = parsed["method"]
        extraction_status = (
            "visible_page"
            if _description_is_substantive(parsed) and company and title and location
            else "needs_review"
        )
    elif _description_is_substantive(parsed):
        extraction_method = parsed.get("method", "visible_page_text")
        extraction_status = (
            "visible_page" if company and title and location else "needs_review"
        )
    else:
        extraction_method = parsed.get("method", "page_metadata_fallback")
        extraction_status = "needs_review"

    source_host = (urlparse(url).hostname or "unknown source").lower()
    source_platform = "LinkedIn Jobs" if "linkedin.com" in source_host else source_host
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    job: dict[str, Any] = {
        "company": company or "Unknown company - verify",
        "title": title,
        "location": location or "Unknown - verify",
        "country": country,
        "employment_type": employment_type or "other",
        "jd_url": normalized_url,
        "application_url": normalized_url,
        "overall_score": None,
        "eligibility_status": "needs_review",
        "resume_variant": "unspecified",
        "resume_path": "",
        "processing_ready": False,
        "notes": (
            "Job information was ingested from a public posting. Verify the source and extracted requirements before processing. "
            "Application-form fields, compensation, work authorization, and sponsorship were not inferred."
        ),
        "source": {
            "platform": source_platform,
            "source_search_url": str(url),
            "job_id": url_job_id or parsed.get("job_id"),
            # Never copy the LinkedIn job ID into the employer requisition field.
            "company_job_code": parsed.get("company_job_code"),
            "job_page_title": title,
            "posted_text_as_seen": parsed.get("posted_date") or None,
            "applicant_count_as_seen": None,
            "application_destination_verified": False,
            "jd_extraction_method": extraction_method,
            "jd_extraction_status": extraction_status,
            "jd_extracted_at": now,
            "jd_extraction_warnings": _dedupe(warnings),
            "final_page_url": parsed.get("final_url") or normalized_url,
            "content_type": parsed.get("content_type", ""),
            "jd_content_source_url": parsed.get("final_url") or normalized_url,
            "verified_job_page_url": parsed.get("final_url")
            if extraction_status == "recovered_official"
            else None,
            "seniority_level": parsed.get("seniority_level"),
            "jd_recovery": recovery,
        },
        "job_description": {
            "summary": summary[:1800],
            "responsibilities": responsibilities,
            "basic_qualifications": basic,
            "preferred_qualifications": preferred,
            "description_text": description_text,
        },
        "application_fields_source": "Not extracted. Inspect the actual application form before treating any fields as required.",
        "fields": [],
    }
    ready, reason = is_job_description_processable(job)
    job["processing_ready"] = ready
    job["source"]["processing_gate_reason"] = reason
    return job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="job-ingest-url",
        description="Extract a publicly accessible job posting into the Job Application Agent JSON schema.",
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="Direct public job-posting URL or a LinkedIn search URL with currentJobId",
    )
    parser.add_argument("--output", help="Path to write the normalized job JSON")
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds (default: 15)",
    )
    parser.add_argument(
        "--no-search-fallback",
        action="store_true",
        help="Disable automatic official-careers/ATS search fallback",
    )
    parser.add_argument(
        "--test-brave",
        action="store_true",
        help="Send a small live query to verify BRAVE_SEARCH_API_KEY and API access",
    )
    parser.add_argument(
        "--query",
        default="Cisco Software Engineer Bengaluru Karnataka India",
        help="Search query used with --test-brave",
    )
    args = parser.parse_args(argv)

    try:
        if args.test_brave:
            if args.url or args.output:
                parser.error(
                    "--test-brave cannot be combined with a job URL or --output."
                )
            result = test_brave_search_connection(
                args.query, country="IN", timeout=args.timeout
            )
            print("Brave Search API: CONNECTED")
            print(f"Query: {result['query']}")
            print(f"Results returned: {result['result_count']}")
            for index, item in enumerate(result["results"], start=1):
                print(f"{index}. {item['title'] or '(untitled)'}")
                print(f"   {item['url']}")
                if item.get("description"):
                    print(f"   {item['description']}")
            print("API key was not displayed.")
            return 0

        if not args.url:
            parser.error("A job URL is required unless --test-brave is used.")
        if not args.output:
            parser.error("--output is required when ingesting a job URL.")

        job = ingest_job_url(
            args.url, timeout=args.timeout, search_fallback=not args.no_search_fallback
        )
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Extracted: {job['company']} | {job['title']}")
        print(
            f"Location: {job['location']} | Country: {job['country'] or 'UNKNOWN - verify'}"
        )
        print(
            f"Extraction status: {job['source']['jd_extraction_status']} ({job['source']['jd_extraction_method']})"
        )
        print(
            f"Processing ready: {job['processing_ready']} ({job['source']['processing_gate_reason']})"
        )
        recovery = job["source"].get("jd_recovery", {})
        if recovery.get("recovered_from_url"):
            print(f"Verified JD source: {recovery['recovered_from_url']}")
            print(f"Match confidence: {recovery.get('match_confidence')}")
        print(
            f"Responsibilities extracted: {len(job['job_description']['responsibilities'])}"
        )
        print(
            f"Basic qualification items extracted: {len(job['job_description']['basic_qualifications'])}"
        )
        print(f"Review warnings: {len(job['source']['jd_extraction_warnings'])}")
        for warning in job["source"]["jd_extraction_warnings"]:
            print(f"  - {warning}")
        print(f"JSON written to: {output_path}")
        return 0
    except (JobIngestionError, OSError, ValueError) as exc:
        print(f"Job ingestion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
