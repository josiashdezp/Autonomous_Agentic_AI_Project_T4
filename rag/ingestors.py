# ---------------------------------------------------------------------
# Source-specific ingestors
# Assumes BaseHTMLIngestor already exists in this file.
# ---------------------------------------------------------------------
from __future__ import annotations
from bs4 import BeautifulSoup
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse, urldefrag
import re
import requests
from rag.structures import TravelDocument
from rag.rag_config import DEFAULT_HEADERS
import hashlib
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class BaseHTMLIngestor:
    def __init__(
        self,
            source_name: str,
        timeout: int = 20,
            min_content_chars: int = 200,
        session: Optional[requests.Session] = None,
            source_type: str = "web_html",
    ) -> None:
        self.source_name = source_name
        self.timeout = timeout
        self.min_content_chars = min_content_chars
        self.source_type = source_type
        self.session = session or requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[403, 429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def fetch_html(self, url: str) -> str:
        time.sleep(1.5)
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def fetch_json(self, url: str, params: Optional[dict] = None) -> dict:
        time.sleep(1.5)
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def parse_html(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")

    def extract_title(self, soup: BeautifulSoup) -> str:
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        if soup.title and soup.title.text:
            return soup.title.text.strip()

        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)

        return f"Untitled {self.source_name} page"

    def remove_noise(self, soup: BeautifulSoup) -> None:
        for selector in [
            "script", "style", "noscript", "svg", "footer",
            "header", "nav", "form", "iframe", "aside"
        ]:
            for tag in soup.select(selector):
                tag.decompose()

        for selector in [
            ".cookie", ".modal", ".popup", ".advertisement", ".ad",
            ".newsletter", ".subscribe", ".breadcrumbs", ".breadcrumb",
            ".social-share", ".share", ".recommended", ".related-content",
        ]:
            for tag in soup.select(selector):
                tag.decompose()

    def clean_text(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_text(self, html: str) -> Dict[str, Any]:
        soup = self.parse_html(html)
        self.remove_noise(soup)
        title = self.extract_title(soup)

        text = soup.get_text(separator="\n", strip=True)
        text = self.clean_text(text)

        return {
            "title": title,
            "content": text[:15000],
        }

    def make_doc_id(self, url: str) -> str:
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()
        return f"{self.source_name}::{digest}"

    def to_document(
        self,
        url: str,
        destination: str,
            state: str,
        category: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> TravelDocument:
        html = self.fetch_html(url)
        extracted = self.extract_text(html)

        content = extracted["content"]
        if len(content) < self.min_content_chars:
            raise ValueError(f"Extracted content is too short ({len(content)} chars) for URL: {url}")

        metadata: Dict[str, Any] = {
            "city": destination,
            "state": state,
            "source_type": self.source_type,
            "category": category,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return TravelDocument(
            doc_id=self.make_doc_id(url),
            title=extracted["title"],
            url=url,
            source=self.source_name,
            destination=destination,
            state=state,
            category=category,
            content=content,
            metadata=metadata,
        )


class GenericArticleIngestor(BaseHTMLIngestor):
    """
    Generic article/list-page ingestor for standard HTML sources.

    This is the default for sources where:
    - we fetch one HTML page
    - strip common noise
    - collect headings/paragraphs/list items
    """

    def __init__(
        self,
            source_name: str = "generic_article",
            timeout: int = 20,
            min_content_chars: int = 200,
            session: Optional[requests.Session] = None,
            source_type: str = "web_article",
    ) -> None:
        super().__init__(
            source_name=source_name,
            timeout=timeout,
            min_content_chars=min_content_chars,
            session=session,
            source_type=source_type,
        )

    def extract_text_from_soup(self, soup: BeautifulSoup) -> Dict[str, Any]:
        self.remove_noise(soup)
        title = self.extract_title(soup)

        parts: List[str] = []
        for element in soup.find_all(["h1", "h2", "h3", "p", "li"]):
            text = element.get_text(" ", strip=True)
            if text and len(text) > 3:
                parts.append(text)

        content = self.clean_text("\n".join(parts))

        return {
            "title": title,
            "content": content[:15000],
        }

    def extract_text(self, html: str) -> Dict[str, Any]:
        soup = self.parse_html(html)
        return self.extract_text_from_soup(soup)


class DepthOneArticleIngestor(GenericArticleIngestor):
    """
    Supports exactly one additional scrape layer for sources with scrape-level = 1.

    Behavior:
    - scrape the source page
    - collect same-site child content links from that page
    - scrape those linked pages
    - stop there
    - no recursive crawling beyond that second layer
    """

    def __init__(
            self,
            source_name: str,
            timeout: int = 20,
            min_content_chars: int = 200,
            session: Optional[requests.Session] = None,
            source_type: str = "web_article",
            max_child_pages: int = 5,
    ) -> None:
        super().__init__(
            source_name=source_name,
            timeout=timeout,
            min_content_chars=min_content_chars,
            session=session,
            source_type=source_type,
        )
        self.max_child_pages = max_child_pages

    def _is_candidate_child_link(self, href: str, base_url: str) -> bool:
        if not href:
            return False
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:") or href.startswith(
                "javascript:"):
            return False

        absolute = urljoin(base_url, href)
        absolute, _ = urldefrag(absolute)

        base_netloc = urlparse(base_url).netloc
        child_netloc = urlparse(absolute).netloc

        # same host only
        if child_netloc != base_netloc:
            return False

        # skip obvious non-content assets
        lower = absolute.lower()
        bad_suffixes = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".pdf", ".zip", ".webp")
        if lower.endswith(bad_suffixes):
            return False

        # avoid auth/share/noise pages
        bad_fragments = [
            "/login", "/signup", "/register", "/account",
            "/share", "/search", "/map", "/maps", "/book",
        ]
        if any(fragment in lower for fragment in bad_fragments):
            return False

        return True

    def _collect_child_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        seen = set()
        child_links: List[str] = []

        for a in soup.find_all("a", href=True):
            href = a.get("href", "").strip()
            if not self._is_candidate_child_link(href, base_url):
                continue

            absolute = urljoin(base_url, href)
            absolute, _ = urldefrag(absolute)

            if absolute == base_url:
                continue
            if absolute in seen:
                continue

            seen.add(absolute)
            child_links.append(absolute)

            if len(child_links) >= self.max_child_pages:
                break

        return child_links

    def to_document(
            self,
            url: str,
            destination: str,
            state: str,
            category: str,
            extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> TravelDocument:
        primary_html = self.fetch_html(url)
        primary_soup = self.parse_html(primary_html)
        primary_extracted = self.extract_text_from_soup(primary_soup)

        all_text_parts = [primary_extracted["content"]]
        child_links = self._collect_child_links(primary_soup, url)

        for child_url in child_links:
            try:
                child_html = self.fetch_html(child_url)
                child_extracted = self.extract_text(child_html)
                child_text = child_extracted["content"].strip()
                if child_text:
                    all_text_parts.append(child_text)
            except Exception:
                # Best effort only: skip broken child pages
                continue

        combined_content = self.clean_text("\n\n".join(all_text_parts))

        if len(combined_content) < self.min_content_chars:
            raise ValueError(
                f"Extracted content is too short ({len(combined_content)} chars) for URL: {url}"
            )

        metadata: Dict[str, Any] = {
            "city": destination,
            "state": state,
            "source_type": self.source_type,
            "category": category,
            "scrape_level": 1,
            "child_pages_scraped": len(child_links),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return TravelDocument(
            doc_id=self.make_doc_id(url),
            title=primary_extracted["title"],
            url=url,
            source=self.source_name,
            destination=destination,
            state=state,
            category=category,
            content=combined_content,
            metadata=metadata,
        )


# ---------------------------------------------------------------------
# Registry-covered sources
# ---------------------------------------------------------------------

# 1) wikivoyage -> scrape-level 1 -> DepthOneArticleIngestor
class WikivoyageIngestor(DepthOneArticleIngestor):
    def __init__(self, timeout: int = 20, session: Optional[requests.Session] = None) -> None:
        super().__init__(
            source_name="wikivoyage",
            timeout=timeout,
            session=session,
            source_type="travel_guide",
            max_child_pages=5,
        )


# 2) visittheusa -> scrape-level 0 -> custom content-root handling is justified
class VisitTheUSAIngestor(BaseHTMLIngestor):
    def __init__(
            self,
            timeout: int = 20,
            min_content_chars: int = 400,
            session: Optional[requests.Session] = None,
    ) -> None:
        super().__init__(
            source_name="visittheusa",
            timeout=timeout,
            min_content_chars=min_content_chars,
            session=session,
            source_type="official_tourism",
        )

    def _find_best_content_root(self, soup: BeautifulSoup) -> BeautifulSoup:
        preferred_selectors = [
            "main",
            "article",
            '[role="main"]',
            ".article",
            ".article-body",
            ".node__content",
            ".page-content",
            ".content",
            ".layout-content",
            ".field--name-body",
        ]
        for selector in preferred_selectors:
            node = soup.select_one(selector)
            if node:
                return node
        return soup

    def extract_text(self, html: str) -> Dict[str, Any]:
        soup = self.parse_html(html)
        self.remove_noise(soup)
        title = self.extract_title(soup)
        content_root = self._find_best_content_root(soup)

        parts: List[str] = []
        for element in content_root.find_all(["h1", "h2", "h3", "p", "li"]):
            text = element.get_text(" ", strip=True)
            if text and len(text) > 3:
                parts.append(text)

        content = self.clean_text("\n".join(parts))

        return {
            "title": title,
            "content": content[:15000],
        }


# 3) timeout -> scrape-level 1 -> DepthOneArticleIngestor
class TimeOutIngestor(DepthOneArticleIngestor):
    def __init__(self, timeout: int = 20, session: Optional[requests.Session] = None) -> None:
        super().__init__(
            source_name="timeout",
            timeout=timeout,
            session=session,
            source_type="city_guide",
            max_child_pages=5,
        )


# 4) recreation_gov -> scrape-level 0 -> generic article/listing page
class RecreationGovIngestor(GenericArticleIngestor):
    def __init__(self, timeout: int = 20, session: Optional[requests.Session] = None) -> None:
        super().__init__(
            source_name="recreation_gov",
            timeout=timeout,
            session=session,
            source_type="outdoor_listing",
        )


# 5) alltrails -> scrape-level 0 -> generic article/listing page
class AllTrailsIngestor(GenericArticleIngestor):
    def __init__(self, timeout: int = 20, session: Optional[requests.Session] = None) -> None:
        super().__init__(
            source_name="alltrails",
            timeout=timeout,
            session=session,
            source_type="trail_listing",
        )


# 6) the_dyrt -> scrape-level 0 -> generic article/listing page
class TheDyrtIngestor(GenericArticleIngestor):
    def __init__(self, timeout: int = 20, session: Optional[requests.Session] = None) -> None:
        super().__init__(
            source_name="the_dyrt",
            timeout=timeout,
            session=session,
            source_type="campground_listing",
        )


# 7) numbeo -> scrape-level 0 -> custom table extraction is justified
class NumbeoIngestor(BaseHTMLIngestor):
    def __init__(self, timeout: int = 20, session: Optional[requests.Session] = None) -> None:
        super().__init__(
            source_name="numbeo",
            timeout=timeout,
            session=session,
            source_type="cost_table",
        )

    def extract_text(self, html: str) -> Dict[str, Any]:
        soup = self.parse_html(html)
        self.remove_noise(soup)
        title = self.extract_title(soup)

        tables = []
        for table in soup.find_all("table"):
            text = table.get_text(" ", strip=True)
            if text:
                tables.append(text)

        content = self.clean_text("\n\n".join(tables))

        return {
            "title": title,
            "content": content[:15000],
        }


# 8) expatistan -> scrape-level 0 -> custom table extraction is justified
class ExpatistanIngestor(GenericArticleIngestor):
    def __init__(self, timeout: int = 20, session: Optional[requests.Session] = None) -> None:
        super().__init__(
            source_name="expatistan",
            timeout=timeout,
            session=session,
            source_type="cost_snapshot",
        )


# 9) gasbuddy -> scrape-level 0 -> generic article/listing page is sufficient
class GasBuddyIngestor(GenericArticleIngestor):
    def __init__(self, timeout: int = 20, session: Optional[requests.Session] = None) -> None:
        super().__init__(
            source_name="gasbuddy",
            timeout=timeout,
            session=session,
            source_type="fuel_snapshot",
        )


# ---------------------------------------------------------------------
# Explicit source_registry-to-ingestor mapping
# ---------------------------------------------------------------------
SOURCE_INGESTOR_CLASS_MAP = {
    "wikivoyage": WikivoyageIngestor,
    "visittheusa": VisitTheUSAIngestor,
    "timeout": TimeOutIngestor,
    "recreation_gov": RecreationGovIngestor,
    "alltrails": AllTrailsIngestor,
    "the_dyrt": TheDyrtIngestor,
    "numbeo": NumbeoIngestor,
    "expatistan": ExpatistanIngestor,
    "gasbuddy": GasBuddyIngestor
}
