.PHONY: help install-deps tox

install-pdm: ## Install required utilities/tools
	@command -v pdm > /dev/null || { echo >&2 "pdm is not installed. Installing..."; pip install --upgrade pip pdm; }

install-global: install-pdm pdm-lock-check ## Install rca-accelerator-chatbot to global Python directories
	pdm install --global --project .

install-deps: install-pdm ## Install Python dependencies
	pdm sync

install-dev-deps: install-pdm ## Install Python dependencies
	pdm sync --dev

pdm-lock-check: ## Check that the pdm.lock file is in a good shape
	pdm lock --check

tox: ## Run tox
	tox

install-openstack-docs-deps: ## Install dependencies for pulling openstack docs
	python3.11 -m venv .openstackvenv && source .openstackvenv/bin/activate
	pip install tox

pull-openstack-docs: ## Pull and convert Openstack docs
	./openstack_retriever/get_openstack_plaintext_docs.sh

help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ''
