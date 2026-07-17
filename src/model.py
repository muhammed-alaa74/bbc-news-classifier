"""Model architecture for the BBC News classifier."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

from . import config


def build_model(num_classes: int = config.NUM_CLASSES) -> Sequential:
    model = Sequential([
        # mask_zero=True: most training articles are long enough to fill
        # max_length with almost no padding, so the model barely learns
        # what a padding (index 0) token "means". Without masking,
        # GlobalAveragePooling1D still averages those near-random padding
        # embeddings into short inputs (e.g. one-sentence queries), which
        # can swamp the real signal and bias predictions toward a single
        # class. Masking makes the pooling layer skip padding entirely.
        Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM, mask_zero=True),
        GlobalAveragePooling1D(),
        Dense(24, activation="relu"),
        Dense(num_classes, activation="softmax"),  # fixed: matches real class count
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model
