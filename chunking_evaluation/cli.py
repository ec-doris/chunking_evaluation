from pprint import pprint

from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from typer import Typer

from chunking_evaluation import GeneralEvaluation
from chunking_evaluation.chunking import RecursiveTokenChunker

app = Typer()


@app.command()
def evaluate():
    chunker = RecursiveTokenChunker(chunk_size=400, chunk_overlap=200)
    evaluation = GeneralEvaluation()
    embedding = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5", device="cuda")

    results = evaluation.run(chunker, embedding)
    pprint(results)


if __name__ == "__main__":
    app()
