import gc
from typing import Any, Callable, cast

import torch
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer


class TempModel:
    def __init__(
        self,
        model_creator: Callable[[], SentenceTransformer],
    ):
        self.model_creator = model_creator

    def __enter__(self) -> "SentenceTransformer":
        self.model = self.model_creator()
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = False,
        **kwargs: Any,
    ):
        """Initialize SentenceTransformerEmbeddingFunction.

        Args:
            model_name (str, optional): Identifier of the SentenceTransformer model, defaults to "all-MiniLM-L6-v2"
            device (str, optional): Device used for computation, defaults to "cpu"
            normalize_embeddings (bool, optional): Whether to normalize returned vectors, defaults to False
            **kwargs: Additional arguments to pass to the SentenceTransformer model.
        """
        self.model_name = model_name
        self._model_creator = lambda: SentenceTransformer(self.model_name, device=device, **kwargs)
        self._normalize_embeddings = normalize_embeddings

    def __call__(self, input: Documents) -> Embeddings:
        with TempModel(self._model_creator) as model:
            return cast(
                Embeddings,
                list(
                    model.encode(
                        list(input),
                        convert_to_numpy=True,
                        normalize_embeddings=self._normalize_embeddings,
                    )
                ),
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
        with TempModel(self._model_creator) as model:
            return cast(
                Embeddings,
                [  # noqa: C416
                    embedding
                    for embedding in model.encode(
                        [f"{self.prefix}: {d}" for d in input],
                        convert_to_numpy=True,
                        normalize_embeddings=self._normalize_embeddings,
                    )
                ],
            )


class PromptedSentenceTransformerEmbeddingFunction(SentenceTransformerEmbeddingFunction):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = False,
        prompt: str = "",
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, device=device, normalize_embeddings=normalize_embeddings, **kwargs)
        self.prompt: str = prompt

    def __call__(self, input: Documents) -> Embeddings:
        with TempModel(self._model_creator) as model:
            return cast(
                Embeddings,
                [  # noqa: C416
                    embedding
                    for embedding in model.encode(
                        list(input),
                        convert_to_numpy=True,
                        normalize_embeddings=self._normalize_embeddings,
                        prompt_name=self.prompt,
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
        with TempModel(self._model_creator) as model:
            return cast(
                Embeddings,
                [  # noqa: C416
                    embedding
                    for embedding in model.encode(
                        list(input),
                        task=self.task,
                        convert_to_numpy=True,
                        normalize_embeddings=self._normalize_embeddings,
                    )
                ],
            )
