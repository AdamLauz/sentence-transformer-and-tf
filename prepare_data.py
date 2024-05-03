import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import re

DATA_PATH = "./dataset"
TRAIN_PATH = Path(DATA_PATH, "train.json")
TEST_PATH = Path(DATA_PATH, "test.json")
SENTENCE_TRANSFORMER_PATH = str(Path("./", "models", "sentence_transformer.h5"))
INGREDIENT_ENCODE_DICT = dict()


def load_sentence_transformer() -> SentenceTransformer:
    return SentenceTransformer.load(SENTENCE_TRANSFORMER_PATH)


def normalize_ingredient(ingredient: str) -> str:
    ingredient_c = [" ".join([WordNetLemmatizer().lemmatize(q) for q in p.split()]) for p in [ingredient]]  # Lemmatization
    ingredient_c = list(map(lambda x: re.sub(r'\(.*oz.\)|crushed|crumbles|ground|minced|powder|chopped|finely|sliced', '', x), ingredient_c))
    ingredient_c = list(map(lambda x: re.sub("[^a-zA-Z]", " ", x),
                          ingredient_c))  # To remove everything except a-z and A-Z
    ingredient_c = " ".join(ingredient_c)  # To make list element a string element
    ingredient_c = ingredient_c.lower().strip()
    return ingredient_c


def get_ingredients_embedding(model: SentenceTransformer, ingredients: List[str]):
    embeddings = None
    for ingredient in ingredients:
        ingredient = normalize_ingredient(ingredient)

        if ingredient in INGREDIENT_ENCODE_DICT:
            ingredient_encoded = INGREDIENT_ENCODE_DICT[ingredient]
        else:
            ingredient_encoded = model.encode([ingredient])
            INGREDIENT_ENCODE_DICT.update({ingredient: ingredient_encoded.copy()})

        if embeddings is None:
            embeddings = ingredient_encoded.copy()
        else:
            embeddings += ingredient_encoded

    return embeddings


def prepare_data():
    # load data and embed the ingredients to vectors

    train = json.load(open(TRAIN_PATH))
    test = json.load(open(TEST_PATH))

    sentence_transformer = load_sentence_transformer()

    def add_embeddings(data: json) -> json:
        total_records = len(data)
        for idx, record in enumerate(data):
            print(f"Processing record {idx + 1}/{total_records}...")
            ingredients = record["ingredients"]
            ingredients_embedding = get_ingredients_embedding(sentence_transformer, ingredients)
            record.update({"ingredients_embedding": ingredients_embedding[0].tolist()})
        return data

    train_prepared = add_embeddings(train)

    with open(Path(DATA_PATH, "train_prepared.json"), "w") as fp:
        json.dump(train_prepared, fp, indent=4)

    test_prepared = add_embeddings(test)

    with open(Path(DATA_PATH, "test_prepared.json"), "w") as fp:
        json.dump(test_prepared, fp, indent=4)


if __name__ == "__main__":
    prepare_data()
