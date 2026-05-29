from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.chunking import chunk_text
from rag.embeddings import DEFAULT_LOCAL_DIMENSIONS, embed_texts
from rag.extractor import extract_file_text
from rag.prompt import build_grounded_prompt
from rag.vector_store import DEFAULT_COLLECTION_NAME, InMemoryVectorStore


app = FastAPI(
    title="AI-RAG API",
    version="0.7.0",
    description="Day 7 FastAPI backend for the AI-RAG learning project.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChunkRequest(BaseModel):
    text: str = Field(..., min_length=1)
    chunk_size: int = Field(default=1000, gt=0)
    overlap: int = Field(default=200, ge=0)


class ChunkPayload(BaseModel):
    index: int
    start: int
    end: int
    text: str


class ChunkResponse(BaseModel):
    characters: int
    chunk_count: int
    chunks: list[ChunkPayload]


class ExtractResponse(BaseModel):
    filename: str
    characters: int
    text: str


class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    chunk_size: int = Field(default=1000, gt=0)
    overlap: int = Field(default=200, ge=0)
    top_k: int = Field(default=3, gt=0)
    embedding_dimensions: int = Field(default=DEFAULT_LOCAL_DIMENSIONS, gt=0)
    document_id: Optional[str] = None


class SearchResultPayload(BaseModel):
    rank: int
    chunk_index: int
    score: float
    start: int
    end: int
    text: str
    text_preview: str


class SearchResponse(BaseModel):
    query: str
    document_id: str
    chunk_count: int
    vector_count: int
    results: list[SearchResultPayload]


class PromptResponse(SearchResponse):
    prompt: str


@app.get("/")
def read_root() -> dict[str, object]:
    return {
        "name": "AI-RAG API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chunks", response_model=ChunkResponse)
def create_chunks(request: ChunkRequest) -> ChunkResponse:
    try:
        chunks = chunk_text(
            request.text,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ChunkResponse(
        characters=len(request.text),
        chunk_count=len(chunks),
        chunks=[
            ChunkPayload(
                index=chunk.index,
                start=chunk.start,
                end=chunk.end,
                text=chunk.text,
            )
            for chunk in chunks
        ],
    )


@app.post("/documents/extract", response_model=ExtractResponse)
async def extract_document(file: UploadFile = File(...)) -> ExtractResponse:
    filename = Path(file.filename or "uploaded_document").name
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / filename
            file_path.write_bytes(content)
            text = extract_file_text(file_path)
    except (FileNotFoundError, RuntimeError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ExtractResponse(
        filename=filename,
        characters=len(text),
        text=text,
    )


@app.post("/rag/search", response_model=SearchResponse)
def search_document(request: SearchRequest) -> SearchResponse:
    try:
        search_response, _ = _run_search_pipeline(request)
        return search_response
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rag/prompt", response_model=PromptResponse)
def build_rag_prompt(request: SearchRequest) -> PromptResponse:
    try:
        search_response, search_results = _run_search_pipeline(request)
        prompt = build_grounded_prompt(request.query, search_results)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PromptResponse(
        **search_response.model_dump(),
        prompt=prompt,
    )


def _run_search_pipeline(request: SearchRequest) -> tuple[SearchResponse, list]:
    chunks = chunk_text(
        request.text,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
    )
    embeddings = embed_texts(
        [chunk.text for chunk in chunks],
        dimensions=request.embedding_dimensions,
    )
    document_id = request.document_id or "api-document"
    vector_store = InMemoryVectorStore(collection_name=DEFAULT_COLLECTION_NAME)
    vector_count = vector_store.store_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id=document_id,
    )
    query_embedding = embed_texts(
        [request.query],
        dimensions=request.embedding_dimensions,
    )[0]
    search_results = vector_store.search(
        query_vector=query_embedding.values,
        limit=request.top_k,
    )

    return (
        SearchResponse(
            query=request.query,
            document_id=document_id,
            chunk_count=len(chunks),
            vector_count=vector_count,
            results=[
                SearchResultPayload(
                    rank=rank,
                    chunk_index=result.chunk_index,
                    score=result.score,
                    start=result.start,
                    end=result.end,
                    text=result.text,
                    text_preview=result.text_preview,
                )
                for rank, result in enumerate(search_results, start=1)
            ],
        ),
        search_results,
    )
