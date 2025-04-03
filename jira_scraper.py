#!/usr/bin/env python
# coding: utf-8

import uuid
import logging
from argparse import ArgumentParser
import json
import multiprocessing as mp
from typing import List, Dict, Any

import requests
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, pipeline
from openai import OpenAI
from qdrant_client import QdrantClient, models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "all-jira-tickets"
TOKENIZER_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_EMBEDDING_MODEL_ID = "BAAI/bge-large-en-v1.5"
DEFAULT_JIRA_PROJECTS = ["OSP", "RHOSINFRA", "OSPCIX", "RHOSBUGS", "OSPK8", "RHOSPRIO", "OSPRH"]
MAX_SUMMARY_LENGTH = 150  # Reduced from 1000 to more reasonable length
BATCH_SIZE = 32  # For embedding processing

def get_jira_data(jira_url: str, headers: dict, query: str, max_results: int, start_at: int) -> List[Dict[str, Any]]:
    """Retrieve data from JIRA with proper error handling"""
    full_url = f"{jira_url}/rest/api/2/search?jql={query}&maxResults={max_results}&fields=*all&startAt={start_at}"
    logger.info(f"Fetching data from {full_url}")

    try:
        response = requests.get(full_url, headers=headers, timeout=(3.05, 180))
        response.raise_for_status()
        data = response.json()
        return data.get("issues", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {full_url}: {str(e)}")
        return []

def summarize_text(text: str, summarizer) -> str:
    """Summarize text with error handling"""
    if not text.strip():
        return ""

    try:
        summary = summarizer(
            text,
            max_length=MAX_SUMMARY_LENGTH,
            min_length=30,
            do_sample=False,
            truncation=True
        )
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        return text[:MAX_SUMMARY_LENGTH]  # Fallback to truncation

def process_row(row: pd.Series, summarizer) -> Dict[str, Any]:
    """Process a single JIRA row to create text content"""
    try:
        description = row.get('description', ' ') or ' '
        comments = "\n".join(row.get('comments', [])) if isinstance(row.get('comments', []), list) else row.get('comments', '')

        full_text = f"{row['summary']}\n{description}\n{comments}"
        summary = summarize_text(full_text, summarizer)

        return {
            "id": row["id"],
            "url": row["url"],
            "summary": row["summary"],
            "description": description,
            "comments": comments,
            "text": summary
        }
    except Exception as e:
        logger.error(f"Error processing row {row.get('id', 'unknown')}: {str(e)}")
        return None

def update_database(
    database_client_url: str,
    llm_server_url: str,
    llm_api_key: str,
    jira_token: str,
    embedding_model: str,
    database_api_key: str,
    max_results: int = 10000,
    jira_url: str = "https://issues.redhat.com",
    chunk_size: int = 1024,
    jira_database_bkp_path: str = "jira_all_bugs.pickle",
    jira_projects: List[str] = DEFAULT_JIRA_PROJECTS,
):
    """Main function to update the database with JIRA data"""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {jira_token}",
    }

    # Build and encode query
    projects = " OR ".join([f"project={e}" for e in jira_projects])
    query = requests.utils.quote(f"{projects} AND type=bug AND status=Closed")

    # Initial data fetch
    try:
        initial_url = f"{jira_url}/rest/api/2/search?jql={query}&maxResults={max_results}&fields=*all"
        response = requests.get(initial_url, headers=headers, timeout=(3.05, 180))
        response.raise_for_status()
        data = response.json()
        total = data["total"]
        logger.info(f"{total} items found for query {query}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Initial JIRA request failed: {str(e)}")
        raise

    # Fetch all pages in parallel
    with mp.Pool(10) as pool:
        results = pool.starmap(
            get_jira_data,
            [(jira_url, headers, query, max_results, page) for page in range(0, total, max_results)]
        )

    # Flatten results and process into DataFrame
    all_issues = [issue for sublist in results for issue in sublist]

    # Initialize summarizer
    summarizer = pipeline("summarization", model="t5-small")

    # Process each issue
    processed_data = []
    for raw_result in tqdm(all_issues, desc="Processing JIRA issues"):
        try:
            row = {
                "id": raw_result["id"],
                "url": f"{jira_url}/browse/{raw_result['key']}",
                "summary": raw_result["fields"].get("summary", ""),
                "description": raw_result["fields"].get("description", ""),
                "comments": [c["body"] for c in raw_result["fields"].get("comment", {}).get("comments", [])]
            }
            processed_row = process_row(row, summarizer)
            if processed_row:
                processed_data.append(processed_row)
        except Exception as e:
            logger.error(f"Error processing issue {raw_result.get('id', 'unknown')}: {str(e)}")

    df = pd.DataFrame(processed_data)
    df = df.drop_duplicates(subset=["id"])
    logger.info(f"Processed {len(df)} unique issues")

    # Backup data
    df.to_pickle(jira_database_bkp_path)
    logger.info(f"Data backup saved to {jira_database_bkp_path}")

    # Initialize clients
    llm = OpenAI(base_url=llm_server_url, api_key=llm_api_key)
    client = QdrantClient(database_client_url, api_key=database_api_key)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=chunk_size, chunk_overlap=0
    )

    # Get embedding dimension by testing one sample
    test_embedding = llm.embeddings.create(
        model=embedding_model,
        input="test"
    ).data[0].embedding
    embedding_dim = len(test_embedding)

    # Create collection with correct dimension
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE),
    )

    # Process and upload in batches
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Uploading to Qdrant"):
        try:
            chunks = splitter.split_text(row["text"])
            for chunk in chunks:
                embedding = llm.embeddings.create(
                    model=embedding_model,
                    input=chunk
                ).data[0].embedding

                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            payload={
                                "url": row["url"],
                                "text": chunk,
                                "summary": row["summary"],
                                "source": "jira"
                            },
                            vector=embedding,
                        ),
                    ],
                )
        except Exception as e:
            logger.error(f"Error processing {row['url']}: {str(e)}")

    logger.info(
        f"Final record count: {client.get_collection(collection_name=COLLECTION_NAME).points_count}"
    )

def main():
    parser = ArgumentParser("jira_scraper")
    parser.add_argument("--jira_url", type=str, default="https://issues.redhat.com")
    parser.add_argument("--jira_token", type=str, required=True)
    parser.add_argument("--max_results", type=int, default=10000)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--database_client_url", type=str, required=True)
    parser.add_argument("--database_api_key", type=str, required=True)
    parser.add_argument("--llm_server_url", type=str, required=True)
    parser.add_argument("--llm_api_key", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL_ID)
    parser.add_argument("--jira_projects", nargs='+', type=str, default=DEFAULT_JIRA_PROJECTS)
    args = parser.parse_args()

    update_database(
        llm_server_url=args.llm_server_url,
        llm_api_key=args.llm_api_key,
        jira_token=args.jira_token,
        embedding_model=args.embedding_model,
        max_results=args.max_results,
        jira_url=args.jira_url,
        chunk_size=args.chunk_size,
        database_client_url=args.database_client_url,
        jira_projects=args.jira_projects,
        database_api_key=args.database_api_key,
    )

if __name__ == "__main__":
    main()
