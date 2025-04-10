
"""Jira Scraper"""
import uuid
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from jira_scraper.processors.jira_provider import JiraProvider
from jira_scraper.processors.vector_store import QdrantVectorStoreManager
from jira_scraper.processors.text_processor import TextProcessor


class JiraScraper:
    """Main class for JIRA scraping and processing."""

    def __init__(self, config: Dict):
        self.config = config
        self.jira_client = JiraProvider(
            config["jira_url"], config["jira_token"])
        self.db_manager = QdrantVectorStoreManager(
            config["database_client_url"],
            config["database_api_key"]
        )
        self.text_processor = TextProcessor(
            config["embedding_model"],
            config["chunk_size"]
        )
        self.llm_client = OpenAI(
            base_url=config["llm_server_url"],
            organization="",
            api_key=config["llm_api_key"],
        )

    # pylint: disable=too-many-arguments
    def insert_row(self, data: str, record_id: str,
                   jira_url: str, field: str,
                   dataset: pd.DataFrame):
        """
        Insert row into dataset, keep information
        about source field and URL.
        """
        row = {}
        row["id"] = record_id
        row["url"] = jira_url
        row["text"] = data
        row["kind"] = field
        dataset.append(row)

    def build_query(self, projects: List[str]) -> str:
        """Build JQL query from project dictionary.

        Args:
            projects_list: List of project names.

        Returns:
            JQL query string with appropriate filters
        """
        projects_str = " OR ".join([f"project={e}" for e in projects])

        # 2 years ago from now
        two_years_ago = datetime.now() - timedelta(days=365*2)
        date_filter = f'created >= "{two_years_ago.strftime("%Y-%m-%d")}"'

        return f"({projects_str}) AND {date_filter}"

    def fetch_all_issues(self, query: str, max_results: int) -> List[Dict]:
        """Fetch all issues matching the query."""
        # Get initial batch to determine total count
        initial_issues = self.jira_client.get_issues(query, max_results)
        if not initial_issues:
            return []

        total = len(initial_issues)
        print(f"{total} items found for query {query}")

        # Fetch remaining issues in parallel
        with mp.Pool(10) as pool:
            results = pool.starmap(
                self.jira_client.get_issues,
                [(query, max_results, page) for page in range(
                    max_results, total, max_results)]
            )

        # Combine all issues
        all_issues = initial_issues + [
            issue for batch in results for issue in batch]
        return all_issues

    def process_issues_to_dataframe(self, issues: List[Dict]) -> pd.DataFrame:
        """Convert JIRA issues to pandas DataFrame."""
        dataset = []

        for issue in tqdm(issues, desc="Processing issues"):

            jira_url = f"{self.config['jira_url']}/browse/{issue['key']}"
            for text_field in ["summary", "description"]:
                self.insert_row(
                    issue["fields"][text_field],
                    issue["id"],
                    jira_url,
                    text_field,
                    dataset)

            # Concatenate all comments for a jira
            comment_text = ""
            for comment in issue["fields"]["comment"]["comments"]:
                comment_text += f"\n\n{comment['body']}"

            self.insert_row(
                comment_text,
                issue["id"],
                jira_url,
                "comment",
                dataset)

        df = pd.DataFrame(dataset)
        df = df.dropna()
        return df.drop_duplicates(subset=["text"])

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for the model."""
        response = self.llm_client.embeddings.create(
            model=self.config["embedding_model"],
            input="test"
        )
        return len(response.data[0].embedding)

    def process_and_store_embeddings(self, df: pd.DataFrame):
        """Process text and store embeddings in database."""
        vector_size = self.get_embedding_dimension()

        self.db_manager.recreate_collection(
            self.config["db_collection_name"],
            vector_size
        )

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc="Processing embeddings"):
            chunks = self.text_processor.split_text(row["text"])

            for chunk in chunks:
                embedding = self.llm_client.embeddings.create(
                    model=self.config["embedding_model"],
                    input=chunk
                ).data[0].embedding

                point = self.db_manager.build_record(
                    record_id=str(uuid.uuid4()),
                    payload={"url": row["url"],
                             "text": row["text"],
                             "kind": row["kind"]},
                    vector=embedding,
                )

                self.db_manager.upsert_data(
                    self.config["db_collection_name"],
                    [point]
                )

    def run(self, backup_path: str = "jira_all_bugs.pickle"):
        """Main execution method."""
        query = self.build_query(self.config["jira_projects"])
        issues = self.fetch_all_issues(query, self.config["max_results"])

        if not issues:
            print("No issues found to process.")
            return

        df = self.process_issues_to_dataframe(issues)
        print(df.info())

        # Save backup
        df.to_pickle(backup_path)

        # Process and store embeddings
        self.process_and_store_embeddings(df)

        # Print final stats
        stats = self.db_manager.get_collection_stats(
            self.config["db_collection_name"])
        print(f"Number of records: {stats.points_count}")
