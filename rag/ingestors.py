#---------------------------------------------------------------------
# These are the Ingestors for each one of the sources we use in the project
# Wikipedia, National Park Services and Visit the USA
#---------------------------------------------------------------------
from bs4 import BeautifulSoup
import requests
from typing import Any, Dict, Iterable, List, Optional
from rag.structures import TravelDocument
import hashlib
import re
from typing import Optional


class WikipediaIngestor:
    DEFAULT_HEADERS = {
        "User-Agent": (
            "TravelAgentRAGBot/1.0 "
            "(student project; contact: your_email@example.com) "
            "python-requests/2.x"
        )
    }

    def __init__(
        self,
        timeout: int = 20,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)

    def fetch_page(self, title: str) -> dict:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "redirects": 1,
            "formatversion": 2,
        }

        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def to_document(self, title: str, destination: str, state: str, category: str) -> TravelDocument:
        raw = self.fetch_page(title)

        pages = raw.get("query", {}).get("pages", [])
        if not pages:
            raise ValueError(f"No Wikipedia page returned for title: {title}")

        page = pages[0]
        if "missing" in page:
            raise ValueError(f"Wikipedia page not found for title: {title}")

        text = page.get("extract", "").strip()
        if not text:
            raise ValueError(f"No extract text found for title: {title}")

        resolved_title = page.get("title", title)

        return TravelDocument(
            doc_id=f"wiki::{resolved_title}",
            title=resolved_title,
            url=f"https://en.wikipedia.org/wiki/{resolved_title.replace(' ', '_')}",
            source="wikipedia",
            destination=destination,
            state=state,
            category=category,
            content=text,
            metadata={
                "city": destination,
                "state": state,
                "source_type": "encyclopedic",
            },
        )

class NPSIngestor:
    def fetch_html(self, url: str) -> str:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.text

    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, features="html.parser")

        # Remove obvious junk
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    def to_document(self, url: str, title: str, destination: str, state: str, category: str):
        html = self.fetch_html(url)
        text = self.extract_text(html)

        return TravelDocument(
            doc_id=f"nps::{title}",
            title=title,
            url=url,
            source="nps",
            destination=destination,
            state=state,
            category=category,
            content=text,
            metadata={
                "city": destination,
                "state": state,
                "source_type": "official_government",
                "section_type": "plan_your_visit",
            },
        )

class VisitUSAIngestor:
    """
    Ingestor for curated Visit The USA pages.

    Responsibilities:
    - fetch HTML from selected URLs
    - extract meaningful travel content
    - remove obvious junk
    - convert results into TravelDocument objects

    Notes:
    - This is intended for a curated allowlist of pages, not full-site crawling.
    - For a class project, keep the URL list small and handpicked.
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    def __init__(
        self,
        timeout: int = 20,
        min_content_chars: int = 400,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout = timeout
        self.min_content_chars = min_content_chars
        self.session = session or requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)

    def fetch_html(self, url: str) -> str:
        """
        Download raw HTML from a page.
        """
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def parse_html(self, html: str) -> BeautifulSoup:
        """
        Parse HTML into a BeautifulSoup object.
        """
        return BeautifulSoup(html, "html.parser")

    def extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extract the best available page title.
        """
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        if soup.title and soup.title.text:
            return soup.title.text.strip()

        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)

        return "Untitled Visit The USA Page"

    def remove_noise(self, soup: BeautifulSoup) -> None:
        """
        Remove tags that usually add clutter rather than useful content.
        """
        noise_selectors = [
            "script",
            "style",
            "noscript",
            "svg",
            "footer",
            "header",
            "nav",
            "form",
            "iframe",
            "aside",
        ]

        for selector in noise_selectors:
            for tag in soup.select(selector):
                tag.decompose()

        # Common utility or promo blocks
        for selector in [
            ".breadcrumb",
            ".breadcrumbs",
            ".share",
            ".social-share",
            ".newsletter",
            ".subscribe",
            ".cookie",
            ".modal",
            ".popup",
            ".advertisement",
            ".ad",
            ".related-content",
            ".recommended",
        ]:
            for tag in soup.select(selector):
                tag.decompose()

    def _find_best_content_root(self, soup: BeautifulSoup) -> BeautifulSoup:
        """
        Try to identify the main article/content region.
        Falls back to the full soup if nothing obvious is found.
        """
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
        """
        Extract title and cleaned body text from a Visit The USA page.
        """
        soup = self.parse_html(html)
        title = self.extract_title(soup)
        self.remove_noise(soup)

        content_root = self._find_best_content_root(soup)

        # Preserve some structure by collecting headings + paragraphs + list items
        parts: List[str] = []
        for element in content_root.find_all(["h1", "h2", "h3", "p", "li"]):
            text = element.get_text(" ", strip=True)
            if not text:
                continue

            # skip very short list fragments or junky labels
            if len(text) < 3:
                continue

            parts.append(text)

        raw_text = "\n".join(parts)
        cleaned_text = self.clean_text(raw_text)

        return {
            "title": title,
            "content": cleaned_text,
        }

    def clean_text(self, text: str) -> str:
        """
        Normalize whitespace and remove obvious low-value lines.
        """
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        drop_patterns = [
            r"^Skip to .*",
            r"^Back to top$",
            r"^Share$",
            r"^Follow us$",
            r"^Sign up$",
            r"^Subscribe$",
            r"^Read more$",
        ]

        cleaned_lines: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            if any(re.match(pattern, line, flags=re.IGNORECASE) for pattern in drop_patterns):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def make_doc_id(self, url: str) -> str:
        """
        Create a stable document ID from the URL.
        """
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()
        return f"visitusa::{digest}"

    def to_document(
        self,
        url: str,
        destination: str,
        category: str,
        state: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> TravelDocument:
        """
        Fetch one URL and convert it into a TravelDocument.
        """
        html = self.fetch_html(url)
        extracted = self.extract_text(html)

        content = extracted["content"]
        if len(content) < self.min_content_chars:
            raise ValueError(
                f"Extracted content is too short ({len(content)} chars) for URL: {url}"
            )

        metadata: Dict[str, Any] = {
            "source_type": "official_tourism",
            "city": destination,
            "state": state,
            "category": category,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return TravelDocument(
            doc_id=self.make_doc_id(url),
            title=extracted["title"],
            url=url,
            source="visit_the_usa",
            destination=destination,
            state=state,
            category=category,
            content=content,
            metadata=metadata,
        )

    def ingest_many(
        self,
        pages: Iterable[Dict[str, Any]],
        skip_errors: bool = True,
    ) -> List[TravelDocument]:
        """
        Ingest a batch of curated pages.

        Expected page structure:
        {
            "url": "...",
            "destination": "New York City",
            "state": "NY",
            "category": "city_guide",
            "extra_metadata": {...}
        }
        """
        documents: List[TravelDocument] = []

        for page in pages:
            try:
                doc = self.to_document(
                    url=page["url"],
                    destination=page["destination"],
                    state=page.get("state"),
                    category=page["category"],
                    extra_metadata=page.get("extra_metadata"),
                )
                documents.append(doc)
            except Exception:
                if not skip_errors:
                    raise

        return documents