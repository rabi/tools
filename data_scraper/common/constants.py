"""Module for config constants."""

JIRA_COLLECTION_NAME = "rca-knowledge-base"
OSP_DOCS_COLLECTION_NAME = "rca-osp-docs-knowledge-base"
ERRATA_COLLECTION_NAME = "rca-errata"
CI_LOGS_COLLECTION_NAME = "rca-ci"
SOLUTIONS_COLLECTION_NAME = "rca-solutions"
SOLUTIONS_PRODUCT_NAME = "OpenStack"
SOLUTIONS_MAX_RESULTS = 9999
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_JIRA_URL = "https://issues.redhat.com"
DEFAULT_JIRA_PROJECTS = {
    "OSP",
    "OSPCIX",
    "OSPRH",
}
DEFAULT_ZULL_PIPELINES = {
    "openstack-uni-jobs-periodic-integration-rhoso18.0-rhel9",
    "openstack-uni-update-jobs-periodic-integration-rhoso18.0-rhel9",
    "ci-framework-integrity",
}

DEFAULT_ZULL_TENANTS = {
    "components-integration",
}
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_MAX_RESULTS = 10000
DEFAULT_DATE_CUTOFF = "2000-01-01T00:00:00Z"
DEFAULT_NUM_SCRAPER_PROCESSES=10
DEFAULT_ERRATA_PUBLIC_URL="https://access.redhat.com/errata"
DEFAULT_SOLUTIONS_PUBLIC_URL="https://access.redhat.com"
