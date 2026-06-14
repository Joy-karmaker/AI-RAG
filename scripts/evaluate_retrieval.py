from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rag.chunking import chunk_text
from rag.embeddings import DEFAULT_LOCAL_DIMENSIONS, embed_texts
from rag.extractor import extract_file_text
from rag.vector_store import InMemoryVectorStore


DEFAULT_EVAL_FILE = PROJECT_ROOT / "eval" / "eval_questions.json"


@dataclass(frozen=True)
class EvalCase:
    id: str
    document: Path
    question: str
    expected_answer: str
    expected_source_terms: list[str]


@dataclass(frozen=True)
class IndexedDocument:
    document_id: str
    path: Path
    chunk_count: int
    vector_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small retrieval evaluation set against the local RAG pipeline."
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_FILE,
        help=f"Path to the eval JSON file. Default: {DEFAULT_EVAL_FILE}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="Chunk size used while indexing eval documents. Default: 600.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Chunk overlap used while indexing eval documents. Default: 120.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to inspect for each question. Default: 3.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=DEFAULT_LOCAL_DIMENSIONS,
        help=f"Local embedding dimensions. Default: {DEFAULT_LOCAL_DIMENSIONS}.",
    )
    parser.add_argument(
        "--show-passages",
        action="store_true",
        help="Print retrieved passage previews for every question.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        eval_cases = load_eval_cases(args.eval_file)
        vector_store = InMemoryVectorStore(collection_name="ai_rag_eval")
        indexed_documents = index_documents(
            eval_cases=eval_cases,
            vector_store=vector_store,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            embedding_dimensions=args.embedding_dimensions,
        )
        results = evaluate_cases(
            eval_cases=eval_cases,
            vector_store=vector_store,
            top_k=args.top_k,
            embedding_dimensions=args.embedding_dimensions,
            show_passages=args.show_passages,
        )
    except (FileNotFoundError, RuntimeError, UnicodeDecodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print_header("Indexed Documents")
    for indexed in indexed_documents:
        print(
            f"- {indexed.document_id}: {indexed.chunk_count} chunks, "
            f"{indexed.vector_count} vectors"
        )

    print_header("Retrieval Evaluation")
    hits = sum(1 for result in results if result["hit"])
    total = len(results)
    recall_at_k = hits / total if total else 0.0

    for result in results:
        status = "PASS" if result["hit"] else "FAIL"
        print(
            f"{status} {result['id']} | rank={result['rank'] or '-'} | "
            f"question={result['question']}"
        )

    print_header("Summary")
    print(f"Questions: {total}")
    print(f"Hits@{args.top_k}: {hits}")
    print(f"Recall@{args.top_k}: {recall_at_k:.2%}")

    failed = [result for result in results if not result["hit"]]
    if failed:
        print("\nFailed questions to inspect:")
        for result in failed:
            print(f"- {result['id']}: {result['question']}")


def load_eval_cases(eval_file: Path) -> list[EvalCase]:
    path = resolve_project_path(eval_file)
    raw_cases = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw_cases, list):
        raise ValueError("eval file must contain a JSON list.")

    cases = []
    for raw_case in raw_cases:
        case = parse_eval_case(raw_case)
        cases.append(case)

    if not cases:
        raise ValueError("eval file must contain at least one question.")

    return cases


def parse_eval_case(raw_case: Any) -> EvalCase:
    if not isinstance(raw_case, dict):
        raise ValueError("each eval case must be a JSON object.")

    required_fields = [
        "id",
        "document",
        "question",
        "expected_answer",
        "expected_source_terms",
    ]
    missing = [field for field in required_fields if field not in raw_case]
    if missing:
        raise ValueError(f"eval case is missing required field(s): {', '.join(missing)}")

    source_terms = raw_case["expected_source_terms"]
    if not isinstance(source_terms, list) or not source_terms:
        raise ValueError("expected_source_terms must be a non-empty list.")

    return EvalCase(
        id=str(raw_case["id"]),
        document=resolve_project_path(Path(str(raw_case["document"]))),
        question=str(raw_case["question"]),
        expected_answer=str(raw_case["expected_answer"]),
        expected_source_terms=[str(term) for term in source_terms],
    )


def index_documents(
    eval_cases: list[EvalCase],
    vector_store: InMemoryVectorStore,
    chunk_size: int,
    overlap: int,
    embedding_dimensions: int,
) -> list[IndexedDocument]:
    indexed_documents = []
    unique_documents = sorted({case.document for case in eval_cases})

    for document_path in unique_documents:
        document_id = document_to_id(document_path)
        text = extract_file_text(document_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        embeddings = embed_texts(
            [chunk.text for chunk in chunks],
            dimensions=embedding_dimensions,
        )
        vector_count = vector_store.store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
        )
        indexed_documents.append(
            IndexedDocument(
                document_id=document_id,
                path=document_path,
                chunk_count=len(chunks),
                vector_count=vector_count,
            )
        )

    return indexed_documents


def evaluate_cases(
    eval_cases: list[EvalCase],
    vector_store: InMemoryVectorStore,
    top_k: int,
    embedding_dimensions: int,
    show_passages: bool,
) -> list[dict[str, object]]:
    results = []

    for case in eval_cases:
        query_embedding = embed_texts(
            [case.question],
            dimensions=embedding_dimensions,
        )[0]
        search_results = vector_store.search(
            query_vector=query_embedding.values,
            limit=top_k,
            document_id=document_to_id(case.document),
            query_text=case.question,
        )
        rank = first_matching_rank(search_results, case.expected_source_terms)
        results.append(
            {
                "id": case.id,
                "question": case.question,
                "hit": rank is not None,
                "rank": rank,
            }
        )

        if show_passages:
            print_header(case.id)
            print(f"Question: {case.question}")
            print(f"Expected terms: {', '.join(case.expected_source_terms)}")
            for result_rank, result in enumerate(search_results, start=1):
                print(
                    f"{result_rank}. chunk={result.chunk_index} "
                    f"score={result.score:.4f} preview={result.text_preview}"
                )

    return results


def first_matching_rank(search_results, expected_terms: list[str]) -> int | None:
    normalized_terms = [normalize_text(term) for term in expected_terms]

    for rank, result in enumerate(search_results, start=1):
        text = normalize_text(result.text)
        if all(term in text for term in normalized_terms):
            return rank

    return None


def document_to_id(document_path: Path) -> str:
    return document_path.name


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def normalize_text(value: str) -> str:
    return " ".join(value.lower().replace(" -", "-").replace("- ", "-").split())


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


if __name__ == "__main__":
    main()
