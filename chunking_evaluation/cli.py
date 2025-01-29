import logging
from importlib.resources import files
from pathlib import Path
from pprint import pprint
from typing import Annotated

import mlflow
import typer
from openai import OpenAI
from typer import Typer

from chunking_evaluation import SyntheticEvaluation
from chunking_evaluation.chunking import RecursiveTokenChunker
from chunking_evaluation.embedding import NomicSentenceTransformerEmbeddingFunction
from chunking_evaluation.evaluation_framework.base_evaluation import BaseEvaluation

app = Typer()

logging.basicConfig(level=logging.INFO, format="%(asctime)s -  %(name)s - %(levelname)s - %(message)s")


@app.command()
def evaluate(corpus_path: Path, questions_path: Path, experiment: str = ""):
    tracking_path = (files("chunking_evaluation") / "../mlflow.db").resolve()

    mlflow.tracking.set_tracking_uri("sqlite:///" + str(tracking_path))
    with mlflow.start_run(experiment_id=experiment or None) as run:
        params = dict(chunk_size=400, chunk_overlap=200)  # noqa: C408
        chunker = RecursiveTokenChunker(**params)
        mlflow.log_param("chunker", chunker.__class__.__name__)
        mlflow.log_params(params)

        evaluation = BaseEvaluation(
            questions_csv_path=str(questions_path.resolve()),
            corpora_id_paths={str(doc): str(doc) for doc in corpus_path.glob("*.txt")},
        )
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
        mlflow.log_metrics(results)
    pprint(results)


@app.command()
def generate_data(
    corpora_path: Path,
    queries_path: Path,
    generation_base_url: Annotated[str, typer.Option()],
    generation_api_key: Annotated[str, typer.Option(envvar="GEN_API_KEY")],
    generation_model_name: Annotated[str, typer.Option()],
    embedding_base_url: Annotated[str, typer.Option()],
    embedding_api_key: Annotated[str, typer.Option(envvar="EMBEDDING_API_KEY")],
    embedding_model_name: Annotated[str, typer.Option()],
    n_rounds: int = -1,
    n_queries: int = 5,
    max_tokens: int = 1024,
    poor_excerpts_threshold: float = 0.36,
    duplicate_threshold: float = 0.6,
):
    evaluation = SyntheticEvaluation(
        corpora_paths=[str(p) for p in corpora_path.glob("*.txt")],
        queries_csv_path=str(queries_path),
        completion_client=OpenAI(base_url=generation_base_url, api_key=generation_api_key),
        completion_model_name=generation_model_name,
        embedding_client=OpenAI(base_url=embedding_base_url, api_key=embedding_api_key),
        embedding_model_name=embedding_model_name,
        completion_max_tokens=max_tokens,
    )

    # Generate queries and excerpts, and save to CSV
    evaluation.generate_queries_and_excerpts(num_rounds=n_rounds, queries_per_corpus=n_queries)

    # Apply filter to remove queries with poor excerpts
    evaluation.filter_poor_excerpts(threshold=poor_excerpts_threshold)

    # Apply filter to remove duplicates
    evaluation.filter_duplicates(threshold=duplicate_threshold)


if __name__ == "__main__":
    app()
