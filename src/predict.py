"""Inference helpers shared by training evaluation and the Streamlit app."""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from . import config


def predict_category(text: str, model, tokenizer, label_encoder):
    """Return (predicted_label, confidence, full_probability_dict)."""
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequence,
        maxlen=config.MAX_LENGTH,
        padding=config.PADDING_TYPE,
        truncating=config.TRUNC_TYPE,
    )

    probabilities = model.predict(padded, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    predicted_label = label_encoder.index_word[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    prob_dict = {
        label_encoder.index_word[i]: float(probabilities[i])
        for i in range(len(probabilities))
    }
    return predicted_label, confidence, prob_dict
