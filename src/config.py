"""
Central configuration for the BBC News classifier project.
Keeping these values in one place avoids the training/inference
mismatch bugs that show up when constants are copy-pasted around.
"""

import os

# --- Paths -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "bbc-text.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "bbc_news_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
LABEL_TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "label_tokenizer.pkl")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# --- Text preprocessing --------------------------------------------------
VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
MAX_LENGTH = 120
TRUNC_TYPE = "post"
PADDING_TYPE = "post"
OOV_TOKEN = "<OOV>"
TRAINING_PORTION = 0.8

# --- Training --------------------------------------------------------
NUM_EPOCHS = 30
NUM_CLASSES = 5  # business, entertainment, politics, sport, tech

STOPWORDS = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "he'd", "he'll",
    "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
    "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
    "into", "is", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "she'd", "she'll", "she's", "should", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd",
    "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll",
    "we're", "we've", "were", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "would", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves",
]
