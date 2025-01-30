import logging
from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from statistics import mean
from typing import Annotated

import mlflow
import typer
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from openai import OpenAI
from typer import Typer

from chunking_evaluation import SyntheticEvaluation
from chunking_evaluation.chunking import RecursiveTokenChunker
from chunking_evaluation.chunking.fixed_token_chunker import FixedTokenChunker
from chunking_evaluation.embedding import (
    InstructedSentenceTransformerEmbeddingFunction,
    PromptedSentenceTransformerEmbeddingFunction,
    TaskedSentenceTransformerEmbeddingFunction,
)
from chunking_evaluation.evaluation_framework.base_evaluation import BaseEvaluation

app = Typer(pretty_exceptions_show_locals=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s -  %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cli")


class ChunkingExperimentator:
    EMBEDDINGS = [
        (
            InstructedSentenceTransformerEmbeddingFunction(
                model_name="lightonai/modernbert-embed-large",
                device="cuda",
                trust_remote_code=True,
                prefix="search_document",
            ),
            InstructedSentenceTransformerEmbeddingFunction(
                model_name="lightonai/modernbert-embed-large",
                device="cuda",
                trust_remote_code=True,
                prefix="search_query",
            ),
        ),
        (
            SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-large-en-v1.5", device="cuda", normalize_embeddings=True
            ),
            SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-large-en-v1.5",
                device="cuda",
                trust_remote_code=True,
                prefix="Represent this sentence for searching relevant passages",
                normalize_embeddings=True,
            ),
        ),
        (
            SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cuda"),
            SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2",
                device="cuda",
            ),
        ),
        (
            PromptedSentenceTransformerEmbeddingFunction(
                model_name="jxm/cde-small-v2", device="cuda", prompt="document", trust_remote_code=True
            ),
            PromptedSentenceTransformerEmbeddingFunction(
                model_name="jxm/cde-small-v2", device="cuda", prompt="query", trust_remote_code=True
            ),
        ),
        (
            InstructedSentenceTransformerEmbeddingFunction(
                model_name="intfloat/e5-large-v2",
                normalize_embeddings=True,
                prefix="passage",
                device="cuda",
            ),
            InstructedSentenceTransformerEmbeddingFunction(
                model_name="intfloat/e5-large-v2", normalize_embeddings=True, prefix="query", device="cuda"
            ),
        ),
        (
            SentenceTransformerEmbeddingFunction(
                model_name="jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, device="cuda"
            ),
            SentenceTransformerEmbeddingFunction(
                model_name="jinaai/jina-embeddings-v2-base-en", trust_remote_code=True, device="cuda"
            ),
        ),
        (
            TaskedSentenceTransformerEmbeddingFunction(
                model_name="jinaai/jina-embeddings-v3", trust_remote_code=True, task="retrieval.passage", device="cuda"
            ),
            TaskedSentenceTransformerEmbeddingFunction(
                model_name="jinaai/jina-embeddings-v3", trust_remote_code=True, task="retrieval.query", device="cuda"
            ),
        ),
    ]

    CHUNKERS = [
        (
            RecursiveTokenChunker,
            [
                {"chunk_size": 200, "chunk_overlap": 0},
                {"chunk_size": 200, "chunk_overlap": 100},
                {"chunk_size": 400, "chunk_overlap": 200},
                {"chunk_size": 400, "chunk_overlap": 100},
                {"chunk_size": 400, "chunk_overlap": 0},
                {"chunk_size": 800, "chunk_overlap": 200},
                {"chunk_size": 800, "chunk_overlap": 400},
                {"chunk_size": 800, "chunk_overlap": 100},
                {"chunk_size": 800, "chunk_overlap": 0},
            ],
        ),
        (
            FixedTokenChunker,
            [
                {"chunk_size": 200, "chunk_overlap": 0},
                {"chunk_size": 200, "chunk_overlap": 100},
                {"chunk_size": 400, "chunk_overlap": 200},
                {"chunk_size": 400, "chunk_overlap": 100},
                {"chunk_size": 400, "chunk_overlap": 0},
                {"chunk_size": 800, "chunk_overlap": 200},
                {"chunk_size": 800, "chunk_overlap": 400},
                {"chunk_size": 800, "chunk_overlap": 100},
                {"chunk_size": 800, "chunk_overlap": 0},
            ],
        ),
        # (
        #     LLMSemanticChunker,
        #     [
        #         {
        #             "organisation": None,
        #             "client": OpenAI(),  # Set the parameters through env vars
        #             "model_name": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        #         }
        #     ],
        # ),
    ]

    def __init__(self, mlflow_tracking_uri: str):
        mlflow.tracking.set_tracking_uri(mlflow_tracking_uri)

    def run_experiment(
        self,
        corpus_path: Path,
        questions_path: Path,
        experiment_name: str,
    ):
        mlflow.set_experiment(experiment_name)

        for chunker_cls, chunker_params_list in self.CHUNKERS:
            for chunker_params in chunker_params_list:
                chunker = chunker_cls(**chunker_params)
                run_name = f"{chunker.__class__.__name__}_" + "_".join(
                    f"{k}_{v}" for k, v in chunker_params.items() if v is not None
                )
                logger.info(f"Running {run_name}")

                with mlflow.start_run(run_name=run_name or None) as run:
                    metrics: dict[str, list[float]] = defaultdict(list)
                    for chunk_embedding, query_embedding in self.EMBEDDINGS:
                        evaluation = BaseEvaluation(
                            questions_csv_path=str(questions_path.resolve()),
                            corpora_id_paths={str(doc): str(doc) for doc in corpus_path.glob("*.txt")},
                        )
                        logger.info("Embedding with: {}".format(query_embedding.model_name))
                        with mlflow.start_run(nested=True, run_name=query_embedding.model_name) as nested_run:
                            mlflow.log_param("chunker", chunker.__class__.__name__)
                            mlflow.log_params(chunker_params)
                            mlflow.log_param("corpora_paths", sorted(evaluation.corpora_id_paths))
                            mlflow.log_param("questions_path", evaluation.questions_csv_path)
                            mlflow.log_param("query_embedding", query_embedding.model_name)
                            mlflow.log_param("chunk_embedding", chunk_embedding.model_name)
                            results = evaluation.run(
                                chunker,
                                chunk_embedding_function=query_embedding,
                                query_embedding_function=chunk_embedding,
                            )

                            del results["corpora_scores"]
                            mlflow.log_metrics(results)
                            for k, v in results.items():
                                metrics[k].append(v)

                    mean_results: dict[str, float] = {k: mean(v) for k, v in metrics.items()}
                    mlflow.log_metrics(mean_results)
                    logger.info(f"IOU : {mean_results['iou_mean']:.4}")
                    logger.info(f"PREC: {mean_results['precision_mean']:.4}")
                    logger.info(f"REC : {mean_results['recall_mean']:.4}")
                    logger.info(f"PREC_OMEGA: {mean_results['precision_omega_mean']:.4}")


@app.command()
def evaluate(corpus_path: Path, questions_path: Path, experiment: str = ""):
    tracking_path = (files("chunking_evaluation") / "../mlflow.db").resolve()
    mlflow_tracking_uri = "sqlite:///" + str(tracking_path)

    ChunkingExperimentator(mlflow_tracking_uri=mlflow_tracking_uri).run_experiment(
        corpus_path=corpus_path,
        questions_path=questions_path,
        experiment_name=experiment,
    )


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
    duplicate_threshold: float = 0.78,
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
