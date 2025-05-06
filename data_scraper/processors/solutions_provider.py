"""Client to fetch Solutions."""
import logging
import requests

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# pylint: disable=too-few-public-methods
class SolutionsProvider:
    """Provider for Solutions"""

    def __init__(self, query_url: str, query_token: str):
        self.query_url = query_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {query_token}",
        }

    def get_solutions(self, product_name: str,
                      max_results: int,
                      start_at: int = 0) -> list[dict]:
        """Get solutions from Knowledge Base.

        Gets solutions from Knowledge Base and returns list of all the solutions and number
        of retrieved records.

        Args:
            product_name: Search for Solutions for the specific product name
            (e.g., product_name="*OpenStack*")
            max_results: Maximum number of solutions that should be retrieved
            start_at: Specifies a start page you want to download.
        """

        url = f"{self.query_url}/hydra/rest/search/v2/kcs"

        query = (f"fq=(documentKind:Solution AND product: *{product_name}* AND "
                 "solution_resolution:*)&sort=lastModifiedDate desc")

        payload = {
            "clientName": "cli",
            "expression": query,
            "q": "*",
            "rows": max_results,
            "start": start_at
        }

        LOG.info("Processing Solutions request [product: %s, max_results: %d, "
                 "start_at: %d]", query, max_results, start_at)

        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                verify=False,
                timeout=(3.05, 180),
            )
        except requests.exceptions.Timeout:
            LOG.error("Request to Knowledge base %s timed out.", query)
            return [{}]
        except requests.exceptions.RequestException as e:
            LOG.error("Error fetching KB data: %s", e)
            return [{}]
        parsed_response = response.json()['response']
        LOG.info("Found %d Solution records matching the query and retrieved %d " \
                 "of them. [query: %s, max_results: %d, start_at: %d]",
                 parsed_response["numFound"],
                 len(parsed_response["docs"]),
                 query, max_results, start_at)

        return parsed_response["docs"]
