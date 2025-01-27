from pprint import pprint
from typing import Any, cast

from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from typer import Typer

from chunking_evaluation import GeneralEvaluation
from chunking_evaluation.chunking import RecursiveTokenChunker

app = Typer()


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


@app.command()
def evaluate():
    chunker = RecursiveTokenChunker(chunk_size=400, chunk_overlap=200)
    evaluation = GeneralEvaluation()
    chunk_embedding = NomicSentenceTransformerEmbeddingFunction(
        model_name="nomic-ai/nomic-embed-text-v1.5", device="cuda", trust_remote_code=True, prefix="search_document"
    )
    query_embedding = NomicSentenceTransformerEmbeddingFunction(
        model_name="nomic-ai/nomic-embed-text-v1.5", device="cuda", trust_remote_code=True, prefix="search_query"
    )

    results = evaluation.run(
        chunker, chunk_embedding_function=chunk_embedding, query_embedding_function=query_embedding
    )
    pprint(results)


if __name__ == "__main__":
    app()
