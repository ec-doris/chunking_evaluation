import logging
from importlib.resources import files
from pathlib import Path
from pprint import pprint
from typing import Annotated, Any, cast

import mlflow
import typer
from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from openai import OpenAI
from typer import Typer

from chunking_evaluation import GeneralEvaluation, SyntheticEvaluation
from chunking_evaluation.chunking import RecursiveTokenChunker

app = Typer()

logging.basicConfig(level=logging.INFO, format="%(asctime)s -  %(name)s - %(levelname)s - %(message)s")


class NomicSentenceTransformerEmbeddingFunction(SentenceTransformerEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = False,
        prefix: str = "clustering",
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, device=device, normalize_embeddings=normalize_embeddings, **kwargs)
        self.prefix: str = prefix

    def __call__(self, input: Documents) -> Embeddings:
        return cast(
            Embeddings,
            [  # noqa: C416
                embedding
                for embedding in self._model.encode(
                    [f"{self.prefix}: {d}" for d in input],
                    convert_to_numpy=True,
                    normalize_embeddings=self._normalize_embeddings,
                )
            ],
        )


class TaskedSentenceTransformerEmbeddingFunction(SentenceTransformerEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = False,
        task: str = "",
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, device=device, normalize_embeddings=normalize_embeddings, **kwargs)
        assert task != "", "Task cannot be empty"
        self.task: str = task

    def __call__(self, input: Documents) -> Embeddings:
        return cast(
            Embeddings,
            [  # noqa: C416
                embedding
                for embedding in self._model.encode(
                    list(input),
                    task=self.task,
                    convert_to_numpy=True,
                    normalize_embeddings=self._normalize_embeddings,
                )
            ],
        )


@app.command()
def evaluate():
    tracking_path = (files("chunking_evaluation") / "../mlflow.db").resolve()

    mlflow.tracking.set_tracking_uri("sqlite:///" + str(tracking_path))
    with mlflow.start_run() as run:
        params = dict(chunk_size=400, chunk_overlap=200)  # noqa: C408
        chunker = RecursiveTokenChunker(**params)
        mlflow.log_param("chunker", chunker.__class__.__name__)
        mlflow.log_params(params)

        evaluation = GeneralEvaluation()
        mlflow.log_param("corpora_paths", sorted(evaluation.corpora_id_paths))
        mlflow.log_param("questions_path", evaluation.questions_csv_path)

        chunk_embedding = NomicSentenceTransformerEmbeddingFunction(
            model_name="lightonai/modernbert-embed-large",
            device="cuda",
            trust_remote_code=True,
            prefix="search_document",
        )
        query_embedding = NomicSentenceTransformerEmbeddingFunction(
            model_name="lightonai/modernbert-embed-large",
            device="cuda",
            trust_remote_code=True,
            prefix="search_query",
        )

        mlflow.log_param("query_embedding", query_embedding.model_name)
        mlflow.log_param("chunk_embedding", chunk_embedding.model_name)
        results = evaluation.run(
            chunker, chunk_embedding_function=query_embedding, query_embedding_function=chunk_embedding
        )

        del results["corpora_scores"]  # TODO log more details
        # TODO: log dataset
        mlflow.log_metrics(results)
    pprint(results)


@app.command()
def generate_data(
    corpora_path: Path,
    queries_path: Path,
    generation_base_url: Annotated[str, typer.Option()],
    generation_api_key: Annotated[str, typer.Option()],
    generation_model_name: Annotated[str, typer.Option()],
    embedding_base_url: Annotated[str, typer.Option()],
    embedding_api_key: Annotated[str, typer.Option()],
    embedding_model_name: Annotated[str, typer.Option()],
    n_rounds: int = -1,
    n_queries: int = 5,
):
    evaluation = SyntheticEvaluation(
        corpora_paths=[str(p) for p in corpora_path.glob("*.txt")],
        queries_csv_path=str(queries_path),
        completion_client=OpenAI(base_url=generation_base_url, api_key=generation_api_key),
        completion_model_name=generation_model_name,
        embedding_client=OpenAI(base_url=embedding_base_url, api_key=embedding_api_key),
        embedding_model_name=embedding_model_name,
    )

    # Generate queries and excerpts, and save to CSV
    evaluation.generate_queries_and_excerpts(num_rounds=-1)

    # Apply filter to remove queries with poor excerpts
    evaluation.filter_poor_excerpts(threshold=0.36)

    # Apply filter to remove duplicates
    evaluation.filter_duplicates(threshold=0.6)


if __name__ == "__main__":
    app()
