import json
import numpy as np
from keras.models import load_model
from model import LABELS, MODEL_SAVE_PATH
from pathlib import Path
import pandas as pd

TEST_DATA_PATH = str(Path("./dataset/test_prepared.json"))
RESULTS_PATH = str(Path("dataset", "alauz_whats-cooking-submission-2.csv"))


def get_test_data():
    with open(TEST_DATA_PATH, "r") as fp:
        data = json.load(fp)

    X = np.array([row["ingredients_embedding"] for row in data])
    ID = np.array([row["id"] for row in data])

    return X, ID


def get_model():
    return load_model(MODEL_SAVE_PATH)


if __name__ == "__main__":
    X, ID = get_test_data()
    model = get_model()

    predictions = model.predict(X)
    predicted_labels = np.argmax(predictions, axis=1)
    results = [(id, LABELS[label_enc]) for id, label_enc in zip(ID, predicted_labels)]

    results_df = pd.DataFrame(results, columns=["id", "cuisine"])
    results_df.to_csv(RESULTS_PATH, index=False)
