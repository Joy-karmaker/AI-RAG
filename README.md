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

## Day 5: Semantic Search

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
7. Print the highest-scoring chunks.

This is the retrieval part of Retrieval-Augmented Generation.

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
