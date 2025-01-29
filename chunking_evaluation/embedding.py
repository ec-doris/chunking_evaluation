from typing import Any, cast

from chromadb import Documents, Embeddings
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)


class InstructedSentenceTransformerEmbeddingFunction(SentenceTransformerEmbeddingFunction):
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
