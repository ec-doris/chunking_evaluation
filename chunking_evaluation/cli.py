from pprint import pprint
from typing import cast

from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from typer import Typer

from chunking_evaluation import GeneralEvaluation
from chunking_evaluation.chunking import RecursiveTokenChunker

app = Typer()


class NomicSentenceTransformerEmbeddingFunction(SentenceTransformerEmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return cast(
            Embeddings,
            [  # noqa: C416
                embedding
                for embedding in self._model.encode(
                    [f"clustering: {d}" for d in input],
                    convert_to_numpy=True,
                    normalize_embeddings=self._normalize_embeddings,
                )
            ],
        )


@app.command()
def evaluate():
    chunker = RecursiveTokenChunker(chunk_size=400, chunk_overlap=200)
    evaluation = GeneralEvaluation()
    embedding = NomicSentenceTransformerEmbeddingFunction(
        model_name="nomic-ai/nomic-embed-text-v1.5", device="cuda", trust_remote_code=True
    )

    results = evaluation.run(chunker, embedding)
    pprint(results)


if __name__ == "__main__":
    app()
