import logging
from importlib.resources import files
from pprint import pprint
from typing import Any, cast

import mlflow
from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from typer import Typer

from chunking_evaluation import GeneralEvaluation
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
            model_name="nomic-ai/nomic-embed-text-v1.5", device="cuda", trust_remote_code=True, prefix="search_document"
        )
        query_embedding = NomicSentenceTransformerEmbeddingFunction(
            model_name="nomic-ai/nomic-embed-text-v1.5", device="cuda", trust_remote_code=True, prefix="search_query"
        )

        # embedding = SentenceTransformerEmbeddingFunction(
        #     model_name="billatsectorflow/stella_en_400M_v5", device="cuda", trust_remote_code=True
        # )

        # query_embedding = TaskedSentenceTransformerEmbeddingFunction(
        #     model_name="jinaai/jina-embeddings-v3", device="cuda", trust_remote_code=True, task="retrieval.query"
        # )
        #
        # chunk_embedding = TaskedSentenceTransformerEmbeddingFunction(
        #     model_name="jinaai/jina-embeddings-v3", device="cuda", trust_remote_code=True, task="retrieval.passage"
        # )

        results = evaluation.run(
            chunker, chunk_embedding_function=query_embedding, query_embedding_function=chunk_embedding
        )

        del results["corpora_scores"]  # TODO log more details
        # TODO: log dataset
        mlflow.log_metrics(results)
    pprint(results)


if __name__ == "__main__":
    app()
