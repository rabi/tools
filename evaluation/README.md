# Chatbot Evaluation

This script for evaluating chatbot performance. It processes a dataset of queries, sends them to a chatbot API, and computes performance metrics including semantic similarity and retrieval accuracy.

## Features

- **Configurable Concurrency**: Limits the number of simultaneous API calls (default: 5)
- **Multiple Metrics**: Supports semantic similarity scoring and Hit@K retrieval accuracy
- **Progress Tracking**: Shows progress with a progress bar during evaluation

## Usage

```bash
python evaluation.py \
  -i input_dataset.csv \
  -o results.csv \
  --similarity_threshold 0.8 \
  --chatbot_api_url "https://chatbot-api-url/prompt" \
  --llm_server_url "https://llm-api-url/v1" \
  --llm_api_key "llm-api-key" \
  --semantic_similarity \
  --retrieval_metric \
  -k 5
```

## Arguments

| Argument | Description |
|----------|-------------|
| `-i, --input` | Path to the evaluation dataset CSV file |
| `-o, --output` | Path to save the results CSV file |
| `--similarity_threshold` | Threshold for chatbot's search sensitivity |
| `--chatbot_api_url` | URL endpoint for the chatbot API |
| `--llm_server_url` | URL endpoint for the LLM API |
| `--llm_api_key` | API key for the LLM service |
| `-k, --num_k` | K value for Hits@K metric (default: 5) |
| `--semantic_similarity` | Enable semantic similarity calculation |
| `--retrieval_metric` | Enable retrieval Hit@K metric calculation |
| `--chatbot_api_timeout` | Timeout for chatbot API calls in seconds (default: 30) |
| `--llm_model_name` | Specific LLM model name for embeddings |
| `--concurrency_limit` | Maximum number of concurrent API calls (default: 10) |

## Input Dataset Format

The script expects a CSV file with at least the following columns:
- `user_prompt`: The query text to send to the chatbot
- `comments`: Expected response text (for semantic similarity)
- `url`: Expected URL that should be retrieved (for Hit@K metric)

## Output

The script produces a CSV file with the original data plus:
- `chatbot_response`: The response from the chatbot
- `similarity_score`: Semantic similarity score (if enabled)
- `hit_at_k`: Whether the expected URL was in the top K results (if enabled)

The script also prints summary metrics:
- Average semantic similarity score
- Average Hit@K score

## Metrics

### Semantic Similarity

Calculates cosine similarity between of the chatbot response and the reference text using the LLM's embedding model.

### Hit@K

Measures if the ground truth URL appears in the top K URLs returned by the chatbot.
