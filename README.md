# AI-RAG

A beginner-friendly AI Document Chat project built one day at a time.

## Day 1: Text Extraction

Goal: read a local document and print its raw text in the console.

Supported today:

- `.txt`
- `.md`
- `.pdf` after installing `pypdf`

Run the sample:

```powershell
cd "F:\AI Project\AI-RAG"
python backend/main.py sample_docs/day1.txt --show-text
```

Run your own text or Markdown file:

```powershell
python backend/main.py path\to\your-file.txt --show-text
```

For PDF support:

```powershell
python -m pip install -r requirements.txt
python backend/main.py path\to\your-file.pdf
```

## Day 2: Chunking

Goal: split extracted text into smaller overlapping chunks.

Run the Day 2 sample:

```powershell
python backend/main.py sample_docs/day2_long.txt --chunk-size 600 --overlap 120
```

Show the full extracted text before the chunks:

```powershell
python backend/main.py sample_docs/day2_long.txt --show-text
```

Why chunks matter:

- LLMs and embedding models work better with smaller text pieces.
- Search needs focused passages, not one giant document.
- Overlap keeps important sentences from getting lost between chunk boundaries.

## Day 3: Embeddings

Goal: turn each chunk into a vector, which is a list of numbers.

Run local embeddings without an API key:

```powershell
python -B backend/main.py sample_docs/day2_long.txt --chunk-size 600 --overlap 120 --embed
```

Use a smaller local vector to make the output easier to inspect:

```powershell
python -B backend/main.py sample_docs/day2_long.txt --chunk-size 600 --overlap 120 --embed --embedding-dimensions 16
```

Use Gemini embeddings after installing dependencies and setting an API key:

```powershell
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY="your-api-key"
python -B backend/main.py sample_docs/day2_long.txt --embed --embedding-provider gemini
```

The local embedding provider is for learning and offline testing. The Gemini
provider is the real embedding path for production-style RAG work.

## Day 4: Vector Storage

Goal: store chunk vectors in a local in-memory Qdrant collection.

Install the vector database client:

```powershell
python -m pip install -r requirements.txt
```

Store local embeddings in Qdrant:

```powershell
python -B backend/main.py sample_docs/day2_long.txt --chunk-size 600 --overlap 120 --store-vectors --embedding-dimensions 16
```

What gets stored:

- vector values
- document id
- chunk index
- chunk start and end positions
- original chunk text
- embedding provider and model name

For now, Qdrant runs in memory. That means the collection exists only while the
program is running. This is perfect for learning Day 4 without Docker, services,
or a separate database process.

## Day 5: Hybrid Search

Goal: ask a question and retrieve the most relevant stored chunks.

Run a search:

```powershell
python -B backend/main.py sample_docs/day2_long.txt --chunk-size 600 --overlap 120 --query "What is chunking?" --top-k 3
```

Another example:

```powershell
python -B backend/main.py sample_docs/day2_long.txt --chunk-size 600 --overlap 120 --query "Why do chunks overlap?" --top-k 2
```

Day 5 process:

1. Extract document text.
2. Split the document into chunks.
3. Embed each chunk.
4. Store chunk vectors in Qdrant.
5. Embed the user query.
6. Search Qdrant for the closest chunk vectors.
7. Score the real chunk text for keyword matches using BM25-style local search.
8. Combine both scores and print the highest-scoring chunks.

This is the retrieval part of Retrieval-Augmented Generation.

Local search quality note:

- Local embeddings now use 384 dimensions by default instead of 64, reducing
  hash collisions between unrelated words.
- Retrieval is hybrid: vector similarity helps with meaning, while lexical
  scoring catches exact document labels such as `Frameworks/Libraries`,
  `Databases`, `DevOps/Tools`, or specific skill names.

CV example:

```powershell
python -B backend/main.py sample_docs/My_CV.pdf --chunk-size 600 --overlap 120 --query "What are the Frameworks/Libraries here?" --top-k 3
```

That query should retrieve the `TECHNICAL SKILLS` chunk containing:

```text
Frameworks/Libraries: Laravel, CodeIgniter, Express.js, React.js, Vue.js, Redux, jQuery
```

## Day 6: Grounded Gemini Answers

Goal: send the retrieved chunks to Gemini and ask it to answer using only that
document context.

Create a `.env` file in the project root with one of these key names:

```text
GEMINI_API_KEY=your-api-key
```

This project also accepts `GOOGLE_API_KEY` or `API_KEY`.

Generate a grounded answer:

```powershell
python -B backend/main.py sample_docs/day2_long.txt --chunk-size 600 --overlap 120 --query "Why do chunks overlap?" --top-k 3 --answer
```

Show the exact prompt sent to Gemini:

```powershell
python -B backend/main.py sample_docs/day2_long.txt --chunk-size 600 --overlap 120 --query "Why do chunks overlap?" --top-k 3 --answer --show-prompt
```

Build the prompt without calling Gemini:

```powershell
python -B backend/main.py sample_docs/day2_long.txt --chunk-size 600 --overlap 120 --query "Why do chunks overlap?" --top-k 3 --dry-run-answer
```

Day 6 process:

1. Extract document text.
2. Split text into chunks.
3. Create embeddings for chunks.
4. Store vectors in Qdrant.
5. Embed the user question.
6. Retrieve the top matching chunks.
7. Build a grounded prompt from those chunks.
8. Ask Gemini to answer using only that context.

This is where the project becomes a real RAG loop:

```text
retrieve context -> augment prompt -> generate answer
```

## Day 7: FastAPI Backend

Goal: expose the project logic through HTTP API routes.

Run the backend server:

```powershell
python -B -m uvicorn api:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Core routes:

- `GET /health` checks that the API is running.
- `POST /chunks` chunks raw text from JSON.
- `POST /documents/extract` extracts text from an uploaded `.txt`, `.md`, or `.pdf`.
- `POST /rag/search` searches a provided text document with a question.
- `POST /rag/prompt` builds the grounded prompt that would be sent to Gemini.

Example health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Example chunk request:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/chunks `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"RAG splits documents into searchable chunks.","chunk_size":30,"overlap":5}'
```

## Day 8: Connected Backend Pipeline

Goal: upload a document once, store its vectors in memory, then ask questions
against that uploaded document by `document_id`.

Upload and process a document:

```powershell
$upload = curl.exe -s -X POST `
  -F "file=@sample_docs/day2_long.txt" `
  -F "chunk_size=600" `
  -F "overlap=120" `
  http://127.0.0.1:8000/documents/upload | ConvertFrom-Json

$upload.document_id
```

Ask a question about that uploaded document:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/documents/$($upload.document_id)/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"Why do chunks overlap?","top_k":2}'
```

Build the Gemini prompt without calling Gemini:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/documents/$($upload.document_id)/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"Why do chunks overlap?","top_k":2,"dry_run_answer":true}'
```

Ask Gemini for a grounded answer:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/documents/$($upload.document_id)/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"Why do chunks overlap?","top_k":1,"answer":true}'
```

Use Gemini 2.5 Pro for a stronger answer model:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/documents/$($upload.document_id)/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"Why do chunks overlap?","top_k":1,"answer":true,"llm_model":"gemini-2.5-pro"}'
```

List uploaded documents:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/documents
```

Day 8 process:

1. `POST /documents/upload` extracts text from the uploaded file.
2. The API chunks that text.
3. The API embeds each chunk.
4. The API stores chunk vectors in the shared in-memory Qdrant store.
5. The API returns a `document_id`.
6. `POST /documents/{document_id}/query` embeds the question.
7. Qdrant searches only that document's chunks.
8. The API returns matching chunks, a grounded prompt, or a Gemini answer.

The uploaded document data is still in memory only. Restarting the backend clears
processed documents, which keeps this learning version simple.

## Day 9: Frontend UI

Goal: use the RAG backend from a browser.

Start the backend:

```powershell
python -B -m uvicorn api:app --app-dir backend --host 127.0.0.1 --port 8000
```

Start the frontend:

```powershell
python -B -m http.server 5173 --directory frontend
```

Open:

```text
http://127.0.0.1:5173
```

Day 9 flow:

1. Upload a `.txt`, `.md`, or `.pdf` file.
2. The frontend calls `POST /documents/upload`.
3. The backend returns a `document_id`.
4. Ask a question in the browser.
5. The frontend calls `POST /documents/{document_id}/query`.
6. The UI renders the answer and source chunks.

## Day 10: Full Integration

Goal: run the whole system as one app and verify the important paths.

Start the integrated server:

```powershell
python -B -m uvicorn api:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open the app:

```text
http://127.0.0.1:8000/app
```

API docs are still available:

```text
http://127.0.0.1:8000/docs
```

Run the smoke test in another terminal:

```powershell
python -B scripts/smoke_test.py
```

The smoke test checks:

- API version and system status
- frontend serving from `/app`
- document upload and vector storage
- document-specific hybrid search
- grounded prompt creation without spending a Gemini API call
- document deletion and vector cleanup

Day 10 process:

1. FastAPI serves both API routes and the frontend.
2. The browser app uploads documents to the backend.
3. The backend stores vectors in memory.
4. The browser asks questions by `document_id`.
5. The backend retrieves sources and optionally asks Gemini for a grounded answer.
6. The smoke test confirms the main workflow is still healthy.

In the browser, use the `Model` selector to switch between `gemini-2.5-flash`
and `gemini-2.5-pro`. Flash is the fast default; Pro is the stronger reasoning
option and may use more quota.

Delete unused documents:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/documents/$($upload.document_id)" `
  -Method Delete
```

Deleting a document removes its metadata from API memory and deletes its vectors
from the in-memory Qdrant collection.

## Day 11: RAG Evaluation Baseline

Goal: create a repeatable question set and measure whether retrieval finds the
right evidence.

Run the evaluation:

```powershell
python -B scripts/evaluate_retrieval.py
```

Show retrieved passage previews for debugging:

```powershell
python -B scripts/evaluate_retrieval.py --show-passages
```

What the evaluation checks:

- loads questions from `eval/eval_questions.json`
- indexes each referenced sample document
- runs the same local hybrid retrieval pipeline
- checks whether the expected source terms appear in the top retrieved chunks
- prints `Recall@3` as the first retrieval quality baseline

Why this matters:

- RAG quality should be measured, not guessed.
- Failed questions show where chunking, embeddings, or search ranking need work.
- This baseline gives future improvements something concrete to beat.

## Day 12: Retrieval Metrics

Goal: turn the Day 11 evaluation set into measurable retrieval scores.

Run the metrics report:

```powershell
python -B scripts/evaluate_retrieval.py
```

The report now includes:

- `Recall@1`: the expected source was the first retrieved chunk.
- `Recall@3`: the expected source appeared anywhere in the top 3 chunks.
- `Recall@5`: the expected source appeared anywhere in the top 5 chunks.
- `MRR@5`: mean reciprocal rank, which rewards correct sources appearing higher.

Use a different retrieval depth:

```powershell
python -B scripts/evaluate_retrieval.py --top-k 10
```

Why this matters:

- `Recall@k` tells you whether the retriever found the right evidence.
- `MRR` tells you whether the right evidence appeared near the top.
- These metrics let you compare future chunking, embedding, hybrid search, and
  reranking changes objectively.

## Day 13: Structure-Aware Chunking

Goal: compare basic character chunking with chunking that understands document
structure.

Run the original character-based baseline:

```powershell
python -B scripts/evaluate_retrieval.py --chunk-strategy character
```

Compare structure-aware strategies:

```powershell
python -B scripts/evaluate_retrieval.py --chunk-strategy paragraph
python -B scripts/evaluate_retrieval.py --chunk-strategy heading
python -B scripts/evaluate_retrieval.py --chunk-strategy page
```

Show retrieved passages with page and section metadata:

```powershell
python -B scripts/evaluate_retrieval.py --chunk-strategy heading --show-passages
```

Inspect chunks directly from the CLI:

```powershell
python -B backend/main.py sample_docs/My_CV.pdf --chunk-size 600 --overlap 120 --chunk-strategy heading
```

Available chunk strategies:

- `character`: the original overlapping character splitter.
- `paragraph`: groups nearby text lines into cleaner paragraph-aware chunks.
- `heading`: keeps detected section headings attached to the text below them.
- `page`: keeps PDF page boundaries separate while preserving headings.

Chunk metadata now includes:

- document id
- chunk index
- start and end character positions
- page number when available
- section title when detected

The metadata is stored in Qdrant payloads, returned by API search results, and
included in grounded prompts.

Current metrics with the eval set:

| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
| :--- | ---: | ---: | ---: | ---: |
| character | 60% | 100% | 100% | 0.7667 |
| paragraph | 70% | 90% | 100% | 0.8250 |
| heading | 60% | 90% | 100% | 0.7750 |
| page | 60% | 90% | 100% | 0.7750 |

Why this matters:

- Better chunks make retrieved evidence easier to inspect.
- Headings help section-specific questions find cleaner context.
- Page metadata makes PDF sources easier to cite and debug.
- Running the same Day 12 metrics proves whether a chunking strategy helped.
