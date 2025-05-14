"""Code for scraping Solutions data"""
import logging

from typing import List, Dict, TypedDict
from tqdm import tqdm


import pandas as pd

from data_scraper.core.scraper import Scraper
from data_scraper.processors.solutions_provider import SolutionsProvider


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class SolutionsRecord(TypedDict):
    """Represents a record extracted from Solutions"""
    kb_id: str
    kind: str
    topic: str
    url: str
    issue: str
    diagnosticsteps: str
    text: str
    components: list[str]


class SolutionsScraper(Scraper):
    """Main class for Solutions scraping and processing."""

    def __init__(self, config: dict):
        super().__init__(config=config)
        self.config = config
        self.kb_provider = SolutionsProvider(
            self.config["solutions_url"],
            self.config["solutions_token"]
            )

    def get_documents(self) -> list[dict]:
        documents = self.kb_provider.get_solutions(
            self.config["product_name"],
            self.config["max_results"])
        return documents

    def get_records(self, documents: List[Dict]) -> list[SolutionsRecord]:
        """Convert Solution API responses to SolutionsRecord"""
        solutions_records: list[SolutionsRecord] = []
        for raw_result in tqdm(documents, desc="Processing issues"):
            solutions_records.append(
                {
                    "kb_id": raw_result.get('id', ''),
                    "url": raw_result.get('view_uri', ''),
                    "topic": raw_result.get('publishedTitle', ''),
                    "issue": ''.join(raw_result.get('issue', '')),
                    "diagnosticsteps": ''.join(raw_result.get('solution_diagnosticsteps', 'N/A')),
                    "text": ''.join(raw_result.get('solution_resolution', 'N/A')),
                    "components": raw_result.get('component', []),
                    "kind": "solution",
                }
            )

        return solutions_records

    def get_chunks(self, record: dict) -> list[str]:
        chunks = []

        for kb_field in ["topic", "issue"]:
            chunks += self.text_processor.split_text(record[kb_field])

        return chunks

    def record_postprocessing(self, record):
        # Postprocessing is not required for Errata records
        pass

    def cleanup_records(
        self, records: list, backup: bool, backup_path: str
    ) -> list:
        df = pd.DataFrame(records)

        LOG.info("Records stats BEFORE cleanup: %d", df.shape[0])

        df = df.dropna()
        df = df.drop_duplicates(subset=["text"])

        LOG.info("Records stats AFTER cleanup: %d", df.shape[0])

        if backup:
            LOG.info("Saving backup to: %s", backup_path)
            df.to_csv(backup_path)

        return [SolutionsRecord(**row) for row in df.to_dict(orient="records")]
