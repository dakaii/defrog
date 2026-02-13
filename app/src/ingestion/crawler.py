"""
DeFi Protocol Documentation Crawler
Fetches whitepapers and technical docs from major DeFi protocols
"""
import os
import json
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from bs4 import BeautifulSoup
from pypdf import PdfReader
from io import BytesIO
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeFiDocument:
    """Represents a fetched DeFi document"""
    protocol: str
    title: str
    url: str
    content: str
    doc_type: str  # whitepaper, docs, github
    metadata: Dict
    content_hash: str


class DeFiCrawler:
    """Crawler for DeFi protocol documentation"""
    
    # Core DeFi protocols and their documentation sources
    DEFI_SOURCES = {
        "aave": {
            "whitepaper": "https://github.com/aave/protocol-v2/blob/master/aave-v2-whitepaper.pdf",
            "docs": "https://docs.aave.com",
            "protocol": "Aave V2"
        },
        "uniswap_v2": {
            "whitepaper": "https://uniswap.org/whitepaper.pdf",
            "docs": "https://docs.uniswap.org",
            "protocol": "Uniswap V2"
        },
        "uniswap_v3": {
            "whitepaper": "https://uniswap.org/whitepaper-v3.pdf",
            "docs": "https://docs.uniswap.org/protocol/V3/",
            "protocol": "Uniswap V3"
        },
        "compound": {
            "whitepaper": "https://compound.finance/documents/Compound.Whitepaper.pdf",
            "docs": "https://docs.compound.finance",
            "protocol": "Compound"
        },
        "makerdao": {
            "whitepaper": "https://makerdao.com/en/whitepaper",
            "docs": "https://docs.makerdao.com",
            "protocol": "MakerDAO"
        },
        "curve": {
            "whitepaper": "https://curve.fi/files/stableswap-paper.pdf",
            "docs": "https://resources.curve.fi",
            "protocol": "Curve Finance"
        },
        "balancer": {
            "whitepaper": "https://balancer.fi/whitepaper.pdf",
            "docs": "https://docs.balancer.fi",
            "protocol": "Balancer"
        },
        "synthetix": {
            "litepaper": "https://docs.synthetix.io/litepaper",
            "docs": "https://docs.synthetix.io",
            "protocol": "Synthetix"
        }
    }
    
    def __init__(self, storage_path: str = "./data/raw_docs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DeFi-RAG-Crawler/1.0)'
        })
    
    def fetch_pdf(self, url: str, protocol: str, title: str) -> Optional[DeFiDocument]:
        """Fetch and extract text from PDF whitepaper"""
        try:
            logger.info(f"Fetching PDF: {title} from {url}")
            
            # Handle GitHub raw URLs
            if "github.com" in url and "/blob/" in url:
                url = url.replace("/blob/", "/raw/")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract text from PDF
            pdf_file = BytesIO(response.content)
            reader = PdfReader(pdf_file)
            
            text_content = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_content.append(f"[Page {page_num + 1}]\n{text}")
            
            full_text = "\n\n".join(text_content)
            content_hash = hashlib.sha256(full_text.encode()).hexdigest()
            
            # Save raw PDF
            pdf_path = self.storage_path / f"{protocol}_{title.replace(' ', '_')}.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            return DeFiDocument(
                protocol=protocol,
                title=title,
                url=url,
                content=full_text,
                doc_type="whitepaper",
                metadata={
                    "pages": len(reader.pages),
                    "file_size": len(response.content),
                    "local_path": str(pdf_path)
                },
                content_hash=content_hash
            )
            
        except Exception as e:
            logger.error(f"Error fetching PDF {url}: {e}")
            return None
    
    def fetch_web_docs(self, url: str, protocol: str, title: str) -> Optional[DeFiDocument]:
        """Fetch documentation from web pages"""
        try:
            logger.info(f"Fetching web docs: {title} from {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract main content (adaptive to different doc platforms)
            main_content = None
            
            # Try different content selectors
            selectors = [
                'main',
                'article',
                '.content',
                '.documentation-content',
                '[role="main"]',
                '#content'
            ]
            
            for selector in selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.body
            
            text = main_content.get_text(separator='\n', strip=True) if main_content else ""
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            
            return DeFiDocument(
                protocol=protocol,
                title=title,
                url=url,
                content=text,
                doc_type="docs",
                metadata={
                    "source": "web",
                    "char_count": len(text)
                },
                content_hash=content_hash
            )
            
        except Exception as e:
            logger.error(f"Error fetching web docs {url}: {e}")
            return None
    
    def crawl_all(self) -> List[DeFiDocument]:
        """Crawl all configured DeFi protocols"""
        documents = []
        
        for key, source in self.DEFI_SOURCES.items():
            protocol_name = source.get("protocol", key)
            
            # Fetch whitepaper/litepaper
            if "whitepaper" in source:
                doc = self.fetch_pdf(
                    source["whitepaper"],
                    protocol_name,
                    f"{protocol_name} Whitepaper"
                )
                if doc:
                    documents.append(doc)
            elif "litepaper" in source:
                doc = self.fetch_web_docs(
                    source["litepaper"],
                    protocol_name,
                    f"{protocol_name} Litepaper"
                )
                if doc:
                    documents.append(doc)
            
            # Fetch documentation (limited for now to avoid overwhelming)
            if "docs" in source:
                doc = self.fetch_web_docs(
                    source["docs"],
                    protocol_name,
                    f"{protocol_name} Documentation"
                )
                if doc:
                    documents.append(doc)
        
        logger.info(f"Crawled {len(documents)} documents from {len(self.DEFI_SOURCES)} protocols")
        return documents
    
    def save_metadata(self, documents: List[DeFiDocument]):
        """Save document metadata for tracking"""
        metadata_file = self.storage_path / "metadata.json"
        
        metadata = []
        for doc in documents:
            metadata.append({
                "protocol": doc.protocol,
                "title": doc.title,
                "url": doc.url,
                "doc_type": doc.doc_type,
                "content_hash": doc.content_hash,
                "metadata": doc.metadata
            })
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata for {len(documents)} documents")


if __name__ == "__main__":
    # Test crawler
    crawler = DeFiCrawler()
    docs = crawler.crawl_all()
    crawler.save_metadata(docs)
    
    print(f"\nCrawled {len(docs)} documents:")
    for doc in docs:
        print(f"  - {doc.title}: {len(doc.content)} chars")