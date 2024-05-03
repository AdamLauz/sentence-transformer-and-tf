import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras import layers, Model, regularizers
from pathlib import Path

TRAIN_DATA_PATH = str(Path("./dataset/train_prepared.json"))
LABELS = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek', 'indian', 'irish', 'italian', 'jamaican', 'japanese', 'korean', 'mexican', 'moroccan', 'russian', 'southern_us', 'spanish', 'thai', 'vietnamese']
MODEL_SAVE_PATH = str(Path("models", "cuisine_model.h5"))


def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array([row["ingredients_embedding"] for row in data])
    y = np.array([LABELS.index(row["cuisine"]) for row in data])

    return X, y


def prepare_datasets(validation_size):
    # load data
    X, y = load_data(TRAIN_DATA_PATH)

    # create train, validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size)

    return X_train, X_validation, y_train, y_validation


def build_model(input_shape, output_shape):
    """Generates NN model

    :param input_shape (tuple): Shape of input set
    :return model: NN model
    """

    # Input layer
    input_layer = layers.Input(shape=input_shape, name="embeddings_input")

    # Hidden layers
    layer_1 = layers.Dense(output_shape * 14, activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_layer)
    layer_1 = layers.Dropout(0.3)(layer_1)

    layer_2 = layers.Dense(output_shape * 7, activation='relu', kernel_regularizer=regularizers.l2(0.001))(layer_1)
    layer_2 = layers.Dropout(0.3)(layer_2)

    layer_output = layers.Dense(output_shape, activation='softmax', name='cuisine_output')(layer_2)

    # Create the model
    model = Model(inputs=[input_layer], outputs=[layer_output])

    return model


if __name__ == "__main__":
    # get train, validation, test splits
    X_train, X_validation, y_train, y_validation = prepare_datasets(0.2)

    # create network
    input_shape = X_train.shape[1]
    output_shape = np.unique(y_train).shape[0]

    model = build_model(input_shape, output_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss={'cuisine_output': 'sparse_categorical_crossentropy'},
                  metrics={'cuisine_output': 'accuracy'})

    model.summary()

    # train model
    history = model.fit(x={'embeddings_input': X_train}, y={'cuisine_output': y_train},
                        validation_data=[{'embeddings_input': X_validation},
                                         {'cuisine_output': y_validation}], batch_size=32, epochs=100)

    # Save the model
    model.save(MODEL_SAVE_PATH)


