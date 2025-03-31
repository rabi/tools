# RCAccelerator Tools

This repository contains useful tools that can be useful when deploying the
[RCAccelerator/chatbot](https://github.com/RCAccelerator/chatbot). As of now,
this repository contains:

- `jira_scraper.py`: a script that scrapes Jira and stores the collected data
in a vector database.


## Getting Started

2. Install dependencies (which also takes care of installing PDM if needed):
   ```bash
   make install-deps
   ```

3. Run a tool:
   ```bash
   pdm run jira_scraper.py --help
   ```
