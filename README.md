## Help Centre Assistant

A lightweight FastAPI service that answers questions about the Typeform Help Centre using hybrid retrieval (dense + BM25 in Pinecone) and an OpenAI LLM. Designed for quick local runs and simple Docker deployment.

### Install and setup
- **Python**: 3.11+
- **Create a virtual environment** (recommended inside the project) and install the requirememts:
  ```bash
  pip install -r requirements.txt
  ```

### Environment variables
- **Required**:
  - `OPENAI_API_KEY`: for the LLM
  - `PINECONE_API_KEY`: for the vector index
- **Optional (LangSmith via LangChain)**:
  - `LANGCHAIN_API_KEY`: enables tracing/logging automatically; if omitted, logging is off

Create a `.env` file in the project root (it’s used locally and copied into the Docker image). Note encapsulating double quotes are NOT used:
```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pc-...
# Optional LangSmith
LANGCHAIN_API_KEY=ls-...
```

### Knowledge base
The knowledge base used for the vector database consists of chunks scraped from the Typeform Help Centre documentation and converted to sparse vectors for the BM25 model. A standard set of URLs has already been chunked and vectorised:
  - `data/chunks.json`: text chunks extracted from the default help centre pages
  - `data/bm25_values.json`: BM25 sparse model fitted on those chunks
- Default seed URLs live in `src/constants.py` under `TARGET_URLS`.
- To add URLs or change chunking, regenerate the knowledge base:
  ```bash
  python -m scripts.prepare_knowledge_base
  ```
  This will overwrite `data/chunks.json` and `data/bm25_values.json`.

### Docker
Build and run the API (maps container port 80 to localhost:8000):
```bash
docker build -t help-centre-assistant .
docker run --env-file .env -p 8000:80 help-centre-assistant
```
At container startup, the Pinecone index (`help-centre-hybrid`) is created/populated automatically by `scripts/start.sh` (it runs `python -m scripts.init_vector_db`)—no extra steps required.

### Query the API
- **cURL**:
  ```bash
  curl -X POST 'http://localhost:8000/ask_question' \
    -H 'Content-Type: application/json' \
    -d '{"question": "How do I add a Multi-Question Page?"}'
  ```
- **Interactive docs**: Open the FastAPI Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs) and try the `POST /ask_question` endpoint (optional `thread_id` is supported).

#### Request/Response schema
- **Endpoint**: `POST /ask_question`
- **Content-Type**: `application/json`

- **Request body**:
  ```json
  {
    "question": "How do I add a Multi-Question Page?",
    "thread_id": "a2c6e4a4-1c8f-4a68-a3f7-8f1d2a3b7c1f" // optional
  }
  ```
  - `question` (string): your query.
  - `thread_id` (string, optional): pass a persistent ID to maintain conversation state across turns. If omitted, the server generates one.

- **Response body**:
  ```json
  {
    "answer": "To add a Multi-Question Page, go to...",
    "thread_id": "a2c6e4a4-1c8f-4a68-a3f7-8f1d2a3b7c1f"
  }
  ```
  - `answer` (string): assistant’s response.
  - `thread_id` (string): the conversation ID to reuse on subsequent requests.

- **Example continuing a thread**:
  ```bash
  curl -X POST 'http://localhost:8000/ask_question' \
    -H 'Content-Type: application/json' \
    -d '{"question": "Can I combine it with logic?", "thread_id": "a2c6e4a4-1c8f-4a68-a3f7-8f1d2a3b7c1f"}'
  ```

### Evaluation
**Synthetic dataset generation**:
This script creates an LLM-generated test dataset which can be used for evaluation. It also generates a set of unanswerable questions, to see how well the LLM handles these scenarios.
  ```bash
  python -m scripts.generate_synthetic_qa \
    --chunks data/chunks.json \
    --out data/eval_synthetic.jsonl \
    --model gpt-4.1 \
    --max_items 2 \
    --max_chunks 100 \
    --neg_ratio 0.1
  ```
  - Produces line-delimited JSON (`.jsonl`) with fields: `id`, `question`, `reference_answer`, `source_chunk_ids`, `type` (fact/howto/unanswerable), `evidence_spans`.

**Retrieval evaluation**:
  - Ensure the Pinecone index exists (run once if needed):
    ```bash
    python -m scripts.init_vector_db
    ```
  - Run eval and write results to `data/eval_runs/run_*/retrieval_eval.json`:
    ```bash
    python -m scripts.eval_retrieval \
      --dataset data/eval_synthetic.jsonl \
      --k 5 \
      --alpha 0.5 \
      --use_reranker \
      --reranker_top_n 3 \
      --limit 100
    ```
  - Reports `hit@k`, `mrr@k`, `ndcg@k`, `precision@k`, `recall@k` (unanswerables are skipped).

**LLM evaluation**:
  ```bash
  python -m scripts.eval_llm \
    --dataset data/eval_synthetic.jsonl \
    --judge_model gpt-4.1 \
    --limit 100
  ```
  - Writes `data/eval_runs/run_*/llm_eval.json` with summary and per-example details.

- **Explore results in a notebook**:
  - Use `notebooks/evaluation.ipynb` to load and explore outputs under `data/eval_runs/`.
  - Existing runs and a sample dataset are already included: see `data/eval_runs/` and `data/eval_synthetic.jsonl`.