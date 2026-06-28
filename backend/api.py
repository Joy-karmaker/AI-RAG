from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.chunking import chunk_text
from rag.embeddings import DEFAULT_GEMINI_MODEL, DEFAULT_LOCAL_DIMENSIONS, embed_texts
from rag.extractor import extract_file_text
from rag.generation import DEFAULT_LLM_MODEL, generate_grounded_answer
from rag.prompt import build_grounded_prompt
from rag.vector_store import DEFAULT_COLLECTION_NAME, InMemoryVectorStore


app = FastAPI(
    title="AI-RAG API",
    version="1.0.0",
    description="Integrated API and frontend for the AI-RAG learning project.",
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

vector_store = InMemoryVectorStore(collection_name=DEFAULT_COLLECTION_NAME)
documents: dict[str, "DocumentRecord"] = {}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    filename: str
    characters: int
    chunk_count: int
    vector_count: int
    embedding_dimensions: int
    embedding_provider: str
    embedding_model: str


class ChunkRequest(BaseModel):
    text: str = Field(..., min_length=1)
    chunk_size: int = Field(default=1000, gt=0)
    overlap: int = Field(default=200, ge=0)
    strategy: str = "character"


class ChunkPayload(BaseModel):
    index: int
    start: int
    end: int
    text: str
    page: Optional[int] = None
    section_title: Optional[str] = None


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
    strategy: str = "character"
    top_k: int = Field(default=3, gt=0)
    embedding_dimensions: int = Field(default=DEFAULT_LOCAL_DIMENSIONS, gt=0)
    embedding_provider: str = "local"
    gemini_model: str = DEFAULT_GEMINI_MODEL
    lexical_weight: Optional[float] = None
    vector_weight: Optional[float] = None
    query_mode: str = "auto"
    document_id: Optional[str] = None


class SearchResultPayload(BaseModel):
    rank: int
    chunk_index: int
    score: float
    start: int
    end: int
    text: str
    text_preview: str
    page: Optional[int] = None
    section_title: Optional[str] = None
    vector_score: Optional[float] = None
    lexical_score: Optional[float] = None
    query_type: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    document_id: str
    chunk_count: int
    vector_count: int
    results: list[SearchResultPayload]


class PromptResponse(SearchResponse):
    prompt: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    characters: int
    chunk_count: int
    vector_count: int
    embedding_dimensions: int
    embedding_provider: str
    embedding_model: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentUploadResponse]


class DocumentDeleteResponse(BaseModel):
    document_id: str
    filename: str
    vectors_deleted: int
    remaining_documents: int


class DocumentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, gt=0)
    answer: bool = False
    dry_run_answer: bool = False
    llm_model: str = DEFAULT_LLM_MODEL
    temperature: float = 0.2
    lexical_weight: Optional[float] = None
    vector_weight: Optional[float] = None
    query_mode: str = "auto"


class DocumentQueryResponse(BaseModel):
    document_id: str
    filename: str
    query: str
    results: list[SearchResultPayload]
    answer: Optional[str] = None
    answer_model: Optional[str] = None
    prompt: Optional[str] = None


@app.get("/")
def read_root() -> dict[str, object]:
    return {
        "name": "AI-RAG API",
        "version": app.version,
        "app": "/app",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check() -> dict[str, object]:
    return {
        "status": "ok",
        "processed_documents": len(documents),
    }


@app.get("/system/status")
def system_status() -> dict[str, object]:
    return {
        "status": "ok",
        "api_version": app.version,
        "frontend_available": FRONTEND_DIR.exists(),
        "processed_documents": len(documents),
        "vector_collection": DEFAULT_COLLECTION_NAME,
    }


@app.get("/app")
def frontend_redirect() -> RedirectResponse:
    return RedirectResponse(url="/app/")


@app.post("/chunks", response_model=ChunkResponse)
def create_chunks(request: ChunkRequest) -> ChunkResponse:
    try:
        chunks = chunk_text(
            request.text,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            strategy=request.strategy,
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
                page=chunk.page,
                section_title=chunk.section_title,
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


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=1000),
    overlap: int = Form(default=200),
    strategy: str = Form(default="heading"),
    embedding_dimensions: int = Form(default=DEFAULT_LOCAL_DIMENSIONS),
    embedding_provider: str = Form(default="local"),
    gemini_model: str = Form(default=DEFAULT_GEMINI_MODEL),
    document_id: Optional[str] = Form(default=None),
) -> DocumentUploadResponse:
    if chunk_size <= 0:
        raise HTTPException(status_code=400, detail="chunk_size must be greater than 0.")

    if overlap < 0:
        raise HTTPException(status_code=400, detail="overlap cannot be negative.")

    if embedding_dimensions <= 0:
        raise HTTPException(
            status_code=400,
            detail="embedding_dimensions must be greater than 0.",
        )

    if embedding_provider not in {"local", "gemini"}:
        raise HTTPException(
            status_code=400,
            detail="embedding_provider must be 'local' or 'gemini'.",
        )

    filename = Path(file.filename or "uploaded_document").name
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = _extract_upload_text(filename, content)
        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=strategy,
        )
        if not chunks:
            raise ValueError("No text chunks were created from this document.")

        embeddings = embed_texts(
            [chunk.text for chunk in chunks],
            provider=embedding_provider,
            dimensions=embedding_dimensions,
            gemini_model=gemini_model,
        )
        resolved_document_id = document_id or _new_document_id(filename)
        vector_count = vector_store.store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=resolved_document_id,
        )
    except (RuntimeError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = DocumentRecord(
        document_id=resolved_document_id,
        filename=filename,
        characters=len(text),
        chunk_count=len(chunks),
        vector_count=vector_count,
        embedding_dimensions=embeddings[0].dimensions if embeddings else embedding_dimensions,
        embedding_provider=embeddings[0].provider if embeddings else embedding_provider,
        embedding_model=embeddings[0].model if embeddings else _requested_embedding_model(
            embedding_provider,
            embedding_dimensions,
            gemini_model,
        ),
    )
    documents[resolved_document_id] = record

    return _document_record_to_response(record)


@app.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    return DocumentListResponse(
        documents=[
            _document_record_to_response(record)
            for record in documents.values()
        ]
    )


@app.get("/documents/{document_id}", response_model=DocumentUploadResponse)
def get_document(document_id: str) -> DocumentUploadResponse:
    record = _get_document_record(document_id)
    return _document_record_to_response(record)


@app.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(document_id: str) -> DocumentDeleteResponse:
    record = _get_document_record(document_id)

    try:
        vectors_deleted = vector_store.delete_document(document_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    del documents[document_id]

    return DocumentDeleteResponse(
        document_id=document_id,
        filename=record.filename,
        vectors_deleted=vectors_deleted,
        remaining_documents=len(documents),
    )


@app.post("/documents/{document_id}/query", response_model=DocumentQueryResponse)
def query_uploaded_document(
    document_id: str,
    request: DocumentQueryRequest,
) -> DocumentQueryResponse:
    if request.answer and request.dry_run_answer:
        raise HTTPException(
            status_code=400,
            detail="Use either answer or dry_run_answer, not both.",
        )

    record = _get_document_record(document_id)

    try:
        query_embedding = embed_texts(
            [request.query],
            provider=record.embedding_provider,
            dimensions=record.embedding_dimensions,
            gemini_model=record.embedding_model,
        )[0]
        search_results = vector_store.search(
            query_vector=query_embedding.values,
            limit=request.top_k,
            document_id=document_id,
            query_text=request.query,
            lexical_weight=request.lexical_weight,
            vector_weight=request.vector_weight,
            query_mode=request.query_mode,
        )

        prompt = None
        answer = None
        answer_model = None

        if request.dry_run_answer:
            prompt = build_grounded_prompt(request.query, search_results)
        elif request.answer:
            grounded_answer = generate_grounded_answer(
                question=request.query,
                search_results=search_results,
                model=request.llm_model,
                temperature=request.temperature,
            )
            answer = grounded_answer.answer
            answer_model = grounded_answer.model
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DocumentQueryResponse(
        document_id=record.document_id,
        filename=record.filename,
        query=request.query,
        results=_search_results_to_payload(search_results),
        answer=answer,
        answer_model=answer_model,
        prompt=prompt,
    )


def _run_search_pipeline(request: SearchRequest) -> tuple[SearchResponse, list]:
    chunks = chunk_text(
        request.text,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        strategy=request.strategy,
    )
    embeddings = embed_texts(
        [chunk.text for chunk in chunks],
        provider=request.embedding_provider,
        dimensions=request.embedding_dimensions,
        gemini_model=request.gemini_model,
    )
    document_id = request.document_id or "api-document"
    request_vector_store = InMemoryVectorStore(collection_name=DEFAULT_COLLECTION_NAME)
    vector_count = request_vector_store.store_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id=document_id,
    )
    query_embedding = embed_texts(
        [request.query],
        provider=request.embedding_provider,
        dimensions=request.embedding_dimensions,
        gemini_model=request.gemini_model,
    )[0]
    search_results = request_vector_store.search(
        query_vector=query_embedding.values,
        limit=request.top_k,
        query_text=request.query,
        lexical_weight=request.lexical_weight,
        vector_weight=request.vector_weight,
        query_mode=request.query_mode,
    )

    return (
        SearchResponse(
            query=request.query,
            document_id=document_id,
            chunk_count=len(chunks),
            vector_count=vector_count,
            results=_search_results_to_payload(search_results),
        ),
        search_results,
    )


def _extract_upload_text(filename: str, content: bytes) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / filename
        file_path.write_bytes(content)
        return extract_file_text(file_path)


def _requested_embedding_model(
    provider: str,
    dimensions: int,
    gemini_model: str,
) -> str:
    if provider == "local":
        return f"local-hash-{dimensions}"

    return gemini_model


def _new_document_id(filename: str) -> str:
    safe_stem = Path(filename).stem.lower().replace(" ", "-") or "document"
    return f"{safe_stem}-{uuid.uuid4().hex[:8]}"


def _get_document_record(document_id: str) -> DocumentRecord:
    record = documents.get(document_id)

    if not record:
        raise HTTPException(status_code=404, detail="Document not found.")

    return record


def _document_record_to_response(record: DocumentRecord) -> DocumentUploadResponse:
    return DocumentUploadResponse(
        document_id=record.document_id,
        filename=record.filename,
        characters=record.characters,
        chunk_count=record.chunk_count,
        vector_count=record.vector_count,
        embedding_dimensions=record.embedding_dimensions,
        embedding_provider=record.embedding_provider,
        embedding_model=record.embedding_model,
    )


def _search_results_to_payload(search_results: list) -> list[SearchResultPayload]:
    return [
        SearchResultPayload(
            rank=rank,
            chunk_index=result.chunk_index,
            score=result.score,
            start=result.start,
            end=result.end,
            text=result.text,
            text_preview=result.text_preview,
            page=result.page,
            section_title=result.section_title,
            vector_score=result.vector_score,
            lexical_score=result.lexical_score,
            query_type=result.query_type,
        )
        for rank, result in enumerate(search_results, start=1)
    ]


if FRONTEND_DIR.exists():
    app.mount(
        "/app",
        StaticFiles(directory=FRONTEND_DIR, html=True),
        name="frontend",
    )
