
"""Client to fetch jira issues."""
import abc
import json
from typing import List, Dict

import requests


# pylint: disable=too-few-public-methods
class IssueProvider(abc.ABC):
    """Abstract class defining `IssueProvider` interface."""

    @abc.abstractmethod
    def get_issues(self, query: str,
                   max_results: int,
                   start_at: int = 0) -> List[Dict]:
        """Retrieve issues from the provider."""


class JiraProvider(IssueProvider):
    """Provider for JIRA."""

    def __init__(self, query_url: str, query_token: str):
        self.query_url = query_url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {query_token}",
        }

    def _get_issues(self, query: str,
                    max_results: int,
                    start_at: int = 0) -> str:
        full_url = (
            f"{self.query_url}/rest/api/2/search?"
            f"jql={query}&maxResults={max_results}&"
            f"fields=*all&startAt={start_at}"
        )

        response = requests.get(
            full_url,
            headers=self.headers,
            timeout=(3.05, 180),
        )
        response.raise_for_status()
        return json.loads(response.text)

    def get_initial_issues(self, query: str,
                           max_results: int) -> tuple[List[Dict], int]:
        """Retrieve intial issues and total count."""
        try:
            data = self._get_issues(query, max_results)
            return (data["issues"], data['total'])
        except requests.exceptions.Timeout:
            print(f"Request to jira query {query} timed out.")
            return ([], 0)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching JIRA data: {e}")
            return ([], 0)

    def get_issues(self, query: str,
                   max_results: int,
                   start_at: int = 0) -> List[Dict]:
        """Retrieve issues from JIRA with pagination."""
        try:
            data = self._get_issues(query, max_results,
                                    start_at)
            return data["issues"]
        except requests.exceptions.Timeout:
            print(f"Request to jira query {query} timed out.")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching JIRA data: {e}")
            return []
