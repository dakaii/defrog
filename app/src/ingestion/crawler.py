"""
DeFi Protocol Documentation Crawler
Fetches whitepapers and technical docs from major DeFi protocols.
Supports both synchronous (requests) and asynchronous (aiohttp) crawling.
Sources can be loaded from the database or fall back to built-in defaults.
"""
import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeFiDocument:
    """Represents a fetched DeFi document"""
    protocol: str
    title: str
    url: str
    content: str
    doc_type: str  # whitepaper, docs, litepaper
    metadata: dict
    content_hash: str


# Default sources used when the database table is empty or unavailable
_DEFAULT_SOURCES = {
    "aave": {
        "whitepaper": "https://github.com/aave/protocol-v2/blob/master/aave-v2-whitepaper.pdf",
        "docs": "https://docs.aave.com",
        "protocol": "Aave V2",
    },
    "uniswap_v2": {
        "whitepaper": "https://uniswap.org/whitepaper.pdf",
        "docs": "https://docs.uniswap.org",
        "protocol": "Uniswap V2",
    },
    "uniswap_v3": {
        "whitepaper": "https://uniswap.org/whitepaper-v3.pdf",
        "docs": "https://docs.uniswap.org/protocol/V3/",
        "protocol": "Uniswap V3",
    },
    "compound": {
        "whitepaper": "https://compound.finance/documents/Compound.Whitepaper.pdf",
        "docs": "https://docs.compound.finance",
        "protocol": "Compound",
    },
    "makerdao": {
        "whitepaper": "https://makerdao.com/en/whitepaper",
        "docs": "https://docs.makerdao.com",
        "protocol": "MakerDAO",
    },
    "curve": {
        "whitepaper": "https://curve.fi/files/stableswap-paper.pdf",
        "docs": "https://resources.curve.fi",
        "protocol": "Curve Finance",
    },
    "balancer": {
        "whitepaper": "https://balancer.fi/whitepaper.pdf",
        "docs": "https://docs.balancer.fi",
        "protocol": "Balancer",
    },
    "synthetix": {
        "litepaper": "https://docs.synthetix.io/litepaper",
        "docs": "https://docs.synthetix.io",
        "protocol": "Synthetix",
    },
}


class DeFiCrawler:
    """Crawler for DeFi protocol documentation"""

    # Kept for backward compatibility; code uses _DEFAULT_SOURCES internally
    DEFI_SOURCES = _DEFAULT_SOURCES

    def __init__(self, storage_path: str = "./data/raw_docs", vector_store=None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.vector_store = vector_store
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; DeFi-RAG-Crawler/1.0)"
        })

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def _load_sources(self) -> list[dict]:
        """
        Return a flat list of {protocol, doc_type, url} dicts.
        Reads from document_sources DB table when a vector_store is provided,
        falling back to the built-in defaults on any error or empty table.
        """
        if self.vector_store is None:
            return self._default_source_list()

        conn = None
        cur = None
        try:
            conn = self.vector_store.get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT protocol_name, doc_type, url FROM document_sources WHERE enabled = TRUE ORDER BY id"
            )
            rows = cur.fetchall()
            if rows:
                return [{"protocol": r[0], "doc_type": r[1], "url": r[2]} for r in rows]
        except Exception as e:
            logger.warning(f"Could not read document_sources from DB, using defaults: {e}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

        return self._default_source_list()

    @staticmethod
    def _default_source_list() -> list[dict]:
        sources = []
        for source in _DEFAULT_SOURCES.values():
            protocol = source["protocol"]
            for doc_type in ("whitepaper", "litepaper", "docs"):
                if doc_type in source:
                    sources.append({"protocol": protocol, "doc_type": doc_type, "url": source[doc_type]})
        return sources

    # ------------------------------------------------------------------
    # Synchronous fetch helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    def fetch_pdf(self, url: str, protocol: str, title: str) -> DeFiDocument | None:
        """Fetch and extract text from a PDF whitepaper"""
        try:
            logger.info(f"Fetching PDF: {title} from {url}")
            if "github.com" in url and "/blob/" in url:
                url = url.replace("/blob/", "/raw/")

            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            pdf_file = BytesIO(response.content)
            reader = PdfReader(pdf_file)

            text_content = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_content.append(f"[Page {page_num + 1}]\n{text}")

            full_text = "\n\n".join(text_content)
            content_hash = hashlib.sha256(full_text.encode()).hexdigest()

            pdf_path = self.storage_path / f"{protocol}_{title.replace(' ', '_')}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            return DeFiDocument(
                protocol=protocol,
                title=title,
                url=url,
                content=full_text,
                doc_type="whitepaper",
                metadata={"pages": len(reader.pages), "file_size": len(response.content), "local_path": str(pdf_path)},
                content_hash=content_hash,
            )
        except Exception as e:
            logger.error(f"Error fetching PDF {url}: {e}")
            return None

    def fetch_web_docs(self, url: str, protocol: str, title: str) -> DeFiDocument | None:
        """Fetch documentation from a web page"""
        try:
            logger.info(f"Fetching web docs: {title} from {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            text, content_hash = self._extract_html_text(response.text)

            return DeFiDocument(
                protocol=protocol,
                title=title,
                url=url,
                content=text,
                doc_type="docs",
                metadata={"source": "web", "char_count": len(text)},
                content_hash=content_hash,
            )
        except Exception as e:
            logger.error(f"Error fetching web docs {url}: {e}")
            return None

    def crawl_all(self) -> list[DeFiDocument]:
        """Crawl all configured DeFi protocols (synchronous)"""
        documents: list[DeFiDocument] = []
        sources = self._load_sources()

        for entry in sources:
            protocol = entry["protocol"]
            doc_type = entry["doc_type"]
            url = entry["url"]
            title = f"{protocol} {doc_type.capitalize()}"

            if doc_type in ("whitepaper",):
                doc = self.fetch_pdf(url, protocol, title)
            else:
                doc = self.fetch_web_docs(url, protocol, title)

            if doc:
                doc.doc_type = doc_type
                documents.append(doc)

        logger.info(f"Crawled {len(documents)} documents from {len({s['protocol'] for s in sources})} protocols")
        return documents

    # ------------------------------------------------------------------
    # Async fetch helpers
    # ------------------------------------------------------------------

    async def _fetch_pdf_async(self, session, url: str, protocol: str, title: str) -> DeFiDocument | None:
        """Async version of fetch_pdf using aiohttp"""
        import aiohttp

        try:
            logger.info(f"[async] Fetching PDF: {title} from {url}")
            if "github.com" in url and "/blob/" in url:
                url = url.replace("/blob/", "/raw/")

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                content = await response.read()

            pdf_file = BytesIO(content)
            reader = PdfReader(pdf_file)

            text_content = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_content.append(f"[Page {page_num + 1}]\n{text}")

            full_text = "\n\n".join(text_content)
            content_hash = hashlib.sha256(full_text.encode()).hexdigest()

            pdf_path = self.storage_path / f"{protocol}_{title.replace(' ', '_')}.pdf"
            pdf_path.write_bytes(content)

            return DeFiDocument(
                protocol=protocol,
                title=title,
                url=url,
                content=full_text,
                doc_type="whitepaper",
                metadata={"pages": len(reader.pages), "file_size": len(content), "local_path": str(pdf_path)},
                content_hash=content_hash,
            )
        except Exception as e:
            logger.error(f"[async] Error fetching PDF {url}: {e}")
            return None

    async def _fetch_web_docs_async(self, session, url: str, protocol: str, title: str) -> DeFiDocument | None:
        """Async version of fetch_web_docs using aiohttp"""
        import aiohttp

        try:
            logger.info(f"[async] Fetching web docs: {title} from {url}")
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                html = await response.text()

            text, content_hash = self._extract_html_text(html)

            return DeFiDocument(
                protocol=protocol,
                title=title,
                url=url,
                content=text,
                doc_type="docs",
                metadata={"source": "web", "char_count": len(text)},
                content_hash=content_hash,
            )
        except Exception as e:
            logger.error(f"[async] Error fetching web docs {url}: {e}")
            return None

    async def crawl_all_async(self) -> list[DeFiDocument]:
        """
        Crawl all configured sources concurrently using aiohttp.
        Significantly faster than the synchronous version for large source lists.
        """
        import aiohttp

        sources = self._load_sources()

        async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0 (compatible; DeFi-RAG-Crawler/1.0)"}) as session:
            tasks = []
            for entry in sources:
                protocol = entry["protocol"]
                doc_type = entry["doc_type"]
                url = entry["url"]
                title = f"{protocol} {doc_type.capitalize()}"

                if doc_type == "whitepaper":
                    tasks.append(self._fetch_pdf_async(session, url, protocol, title))
                else:
                    tasks.append(self._fetch_web_docs_async(session, url, protocol, title))

            results = await asyncio.gather(*tasks, return_exceptions=False)

        documents = []
        for doc, entry in zip(results, sources, strict=True):
            if doc is not None:
                doc.doc_type = entry["doc_type"]
                documents.append(doc)

        logger.info(f"[async] Crawled {len(documents)} documents from {len({s['protocol'] for s in sources})} protocols")
        return documents

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_html_text(html: str):
        """Extract readable text from HTML and return (text, sha256_hash)."""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        main_content = None
        for selector in ["main", "article", ".content", ".documentation-content", '[role="main"]', "#content"]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body

        text = main_content.get_text(separator="\n", strip=True) if main_content else ""
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        return text, content_hash

    def save_metadata(self, documents: list[DeFiDocument]):
        """Save document metadata for tracking"""
        metadata_file = self.storage_path / "metadata.json"
        metadata = [
            {"protocol": d.protocol, "title": d.title, "url": d.url,
             "doc_type": d.doc_type, "content_hash": d.content_hash, "metadata": d.metadata}
            for d in documents
        ]
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata for {len(documents)} documents")


if __name__ == "__main__":
    crawler = DeFiCrawler()
    docs = crawler.crawl_all()
    crawler.save_metadata(docs)
    print(f"\nCrawled {len(docs)} documents:")
    for doc in docs:
        print(f"  - {doc.title}: {len(doc.content)} chars")
