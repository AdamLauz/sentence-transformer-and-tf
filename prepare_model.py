from pathlib import Path
from sentence_transformers import SentenceTransformer

SENTENCE_TRANSFORMER_PATH = str(Path("./", "models", "sentence_transformer.h5"))


def save_sentence_transformer():
    sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # save model locally
    sentence_transformer.save(path=SENTENCE_TRANSFORMER_PATH)


if __name__ == "__main__":
    save_sentence_transformer()