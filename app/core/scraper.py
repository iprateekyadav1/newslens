"""
NewsLens — Article Scraper

Uses trafilatura for main-content extraction (it handles boilerplate removal
better than raw BeautifulSoup) and httpx for async fetching so FastAPI's
event loop isn't blocked.

Returns a structured ArticleData dict regardless of whether extraction
succeeds fully — partial results (title-only, or text-only) are still useful.
"""

import re
import logging
from urllib.parse import urlparse
from typing import Optional

import httpx
import trafilatura

from config import SCRAPER_TIMEOUT, SCRAPER_USER_AGENT

logger = logging.getLogger(__name__)

# Known outlet display names by domain fragment
_OUTLET_MAP: dict[str, str] = {
    "nytimes":      "New York Times",
    "washingtonpost": "Washington Post",
    "theguardian":  "The Guardian",
    "bbc":          "BBC",
    "foxnews":      "Fox News",
    "cnn":          "CNN",
    "msnbc":        "MSNBC",
    "breitbart":    "Breitbart",
    "huffpost":     "HuffPost",
    "reuters":      "Reuters",
    "apnews":       "AP News",
    "bloomberg":    "Bloomberg",
    "wsj":          "Wall Street Journal",
    "politico":     "Politico",
    "thehill":      "The Hill",
    "npr":          "NPR",
    "vox":          "Vox",
    "axios":        "Axios",
    "nbcnews":      "NBC News",
    "abcnews":      "ABC News",
    "cbsnews":      "CBS News",
}


def extract_outlet_name(url: str) -> tuple[str, str]:
    """
    Returns (display_name, domain) from a URL.
    E.g. "https://www.nytimes.com/..." → ("New York Times", "nytimes.com")
    """
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower().removeprefix("www.")
        # Match against known outlets
        for fragment, name in _OUTLET_MAP.items():
            if fragment in host:
                return name, host
        # Fallback: capitalise the root domain
        root = host.split(".")[0].title()
        return root, host
    except Exception:
        return "Unknown", ""


async def scrape_article(url: str) -> dict:
    """
    Fetch and extract article text from a URL.

    Returns:
        {
            "text":         str | None,
            "title":        str | None,
            "author":       str | None,
            "date":         str | None,
            "outlet_name":  str,
            "domain":       str,
            "url":          str,
            "error":        str | None,
        }
    """
    outlet_name, domain = extract_outlet_name(url)
    result: dict = {
        "text":        None,
        "title":       None,
        "author":      None,
        "date":        None,
        "outlet_name": outlet_name,
        "domain":      domain,
        "url":         url,
        "error":       None,
    }

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
        }
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=SCRAPER_TIMEOUT,
            headers=headers,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            raw_html = response.text

        # trafilatura: extract main body text
        text = trafilatura.extract(
            raw_html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )

        # trafilatura metadata
        meta = trafilatura.extract_metadata(raw_html)
        if meta:
            result["title"]  = meta.title
            result["author"] = meta.author
            result["date"]   = meta.date

        result["text"] = text

        if not text:
            result["error"] = "Could not extract article body. The page may require JavaScript or block scrapers."

    except httpx.TimeoutException:
        result["error"] = "Request timed out. The site may be too slow or blocking scrapers."
    except httpx.ConnectError:
        result["error"] = "Could not reach that URL. Check the address or try again later."
    except httpx.HTTPStatusError as e:
        result["error"] = f"The site returned HTTP {e.response.status_code}. The article may require login or is unavailable."
    except Exception as e:
        logger.warning("Scrape failed for %s: %s", url, e)
        result["error"] = "Unable to fetch the article. The site may be blocking automated access."

    return result
