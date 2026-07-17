"""
Data loading & preprocessing for the BBC News classifier.

Bug fix vs. the original notebook
----------------------------------
The original notebook encoded the 5 category labels with a Keras
`Tokenizer`. Keras tokenizers reserve index 0 and start counting words
(here: labels) at 1, so the labels ended up as {1, 2, 3, 4, 5} while the
model's output (softmax) layer used indices {0, 1, 2, 3, 4, 5} (6 units).
That meant:
  - one output neuron (index 0) was never trained against any real label
    and only "stole" learning capacity from the network,
  - decoding argmax==0 back to a category name silently failed (no match
    in label_tokenizer.word_index), producing an empty prediction.

The fix here is to use a plain 0-indexed label encoding (5 classes, 5
output units), which removes the off-by-one mismatch entirely.
"""

import csv
import pickle
from dataclasses import dataclass

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from . import config


class LabelEncoder:
    """A tiny, explicit 0-indexed label encoder.

    Replaces the Keras `Tokenizer` that was previously (mis)used for
    label encoding. Keeps a similar `word_index` / `index_word` API so
    the rest of the code (and the Streamlit app) reads naturally.
    """

    def __init__(self):
        self.word_index = {}   # label -> 0-based index
        self.index_word = {}   # 0-based index -> label

    def fit(self, labels):
        unique_labels = sorted(set(labels))
        self.word_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_word = {idx: label for label, idx in self.word_index.items()}
        return self

    def transform(self, labels):
        return np.array([self.word_index[label] for label in labels])

    def inverse_transform(self, indices):
        return [self.index_word[int(i)] for i in indices]

    @property
    def num_classes(self):
        return len(self.word_index)


@dataclass
class Dataset:
    train_padded: np.ndarray
    train_labels: np.ndarray
    validation_padded: np.ndarray
    validation_labels: np.ndarray
    tokenizer: Tokenizer
    label_encoder: LabelEncoder


def clean_sentence(sentence: str) -> str:
    """Remove stopwords the same way the original notebook did."""
    for word in config.STOPWORDS:
        token = " " + word + " "
        sentence = sentence.replace(token, " ")
    return sentence


def load_raw_data(csv_path: str = config.DATA_PATH):
    sentences, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)  # skip header
        for row in reader:
            labels.append(row[0])
            sentences.append(clean_sentence(row[1]))
    return sentences, labels


def build_dataset(csv_path: str = config.DATA_PATH) -> Dataset:
    sentences, labels = load_raw_data(csv_path)

    train_size = int(len(sentences) * config.TRAINING_PORTION)

    train_sentences = sentences[:train_size]
    train_labels_raw = labels[:train_size]

    validation_sentences = sentences[train_size:]
    validation_labels_raw = labels[train_size:]

    # --- text tokenizer -------------------------------------------------
    tokenizer = Tokenizer(num_words=config.VOCAB_SIZE, oov_token=config.OOV_TOKEN)
    tokenizer.fit_on_texts(train_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(
        train_sequences,
        maxlen=config.MAX_LENGTH,
        padding=config.PADDING_TYPE,
        truncating=config.TRUNC_TYPE,
    )

    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(
        validation_sequences,
        maxlen=config.MAX_LENGTH,
        padding=config.PADDING_TYPE,
        truncating=config.TRUNC_TYPE,
    )

    # --- label encoder (0-indexed, bug fixed) ---------------------------
    label_encoder = LabelEncoder().fit(labels)
    train_labels = label_encoder.transform(train_labels_raw)
    validation_labels = label_encoder.transform(validation_labels_raw)

    assert label_encoder.num_classes == config.NUM_CLASSES, (
        f"Expected {config.NUM_CLASSES} classes, found {label_encoder.num_classes}. "
        "Update config.NUM_CLASSES if the dataset's categories changed."
    )

    return Dataset(
        train_padded=train_padded,
        train_labels=train_labels,
        validation_padded=validation_padded,
        validation_labels=validation_labels,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
    )


def save_artifacts(tokenizer: Tokenizer, label_encoder: LabelEncoder):
    with open(config.TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(config.LABEL_TOKENIZER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)


def load_artifacts():
    with open(config.TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(config.LABEL_TOKENIZER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return tokenizer, label_encoder
