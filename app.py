"""
BBC News Classifier — Streamlit App (production styling)

Run:
    streamlit run app.py
"""

import pickle
import textwrap
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def html_block(raw: str) -> str:
    """Flatten a multi-line HTML/CSS string to a single line.

    Streamlit passes markdown through a CommonMark parser before applying
    unsafe_allow_html. A line indented 4+ spaces is treated as a literal
    code block and rendered as plain text instead of being parsed as HTML
    -- exactly what happened with the indented f-strings used to build the
    result cards. Stripping indentation on every line (and joining with no
    separator) makes that mis-detection impossible regardless of how the
    call site is formatted/indented in the source.
    """
    return "".join(line.strip() for line in textwrap.dedent(raw).splitlines())

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "models" / "bbc_news_model.keras"
TOKENIZER_PATH = Path(__file__).parent / "models" / "tokenizer.pkl"
LABEL_TOKENIZER_PATH = Path(__file__).parent / "models" / "label_tokenizer.pkl"
MAX_LENGTH = 120
PADDING_TYPE = "post"
TRUNC_TYPE = "post"

# Muted, harmonized hues — same saturation/lightness band so no single
# category color fights for attention. No emoji anywhere; color + a small
# dot marker carry the category identity instead.
CATEGORY_COLORS = {
    "business": "#3B6E8F",
    "entertainment": "#8E4B6B",
    "politics": "#6B5B95",
    "sport": "#3E7D5A",
    "tech": "#A56A3A",
}

SAMPLE_TEXT = (
    "Manchester United secured a dramatic late win in yesterday's match, "
    "with the winning goal arriving in stoppage time to the delight of the "
    "home crowd."
)

st.set_page_config(
    page_title="BBC News Classifier",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    html_block("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

    :root {
        --bg: #F7F7F5;
        --surface: #FFFFFF;
        --border: #E3E1DC;
        --ink: #1C1F26;
        --muted: #6B7280;
        --accent: #2F5D62;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background-color: var(--bg);
    }

    .block-container {
        max-width: 720px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* Masthead */
    .masthead {
        text-align: center;
        border-bottom: 2px solid var(--ink);
        padding-bottom: 1.1rem;
        margin-bottom: 0.4rem;
    }
    .masthead-title {
        font-family: 'Source Serif 4', serif;
        font-weight: 700;
        font-size: 2.4rem;
        letter-spacing: -0.01em;
        margin: 0;
        color: var(--ink);
    }
    .masthead-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
        color: var(--muted);
        margin-top: 0.35rem;
        letter-spacing: 0.02em;
    }

    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 2.2rem 0 0.6rem 0;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: var(--ink);
    }
    .stTextArea textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }

    /* Buttons */
    .stButton button {
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        border: 1px solid var(--border);
        padding: 0.5rem 1rem;
    }
    .stButton button[kind="primary"] {
        background-color: var(--ink);
        border-color: var(--ink);
        color: var(--bg);
    }
    .stButton button[kind="primary"]:hover {
        background-color: var(--accent);
        border-color: var(--accent);
    }
    .stButton button[kind="secondary"] {
        background-color: var(--surface);
        color: var(--ink);
    }
    .stButton button[kind="secondary"]:hover {
        border-color: var(--ink);
        color: var(--ink);
    }

    /* Result card */
    .result-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-left: 4px solid var(--result-color, var(--accent));
        border-radius: 6px;
        padding: 1.4rem 1.6rem;
        margin-top: 0.4rem;
    }
    .result-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .result-label {
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        font-size: 1.7rem;
        margin: 0.15rem 0 0 0;
        color: var(--ink);
    }
    .result-confidence {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.95rem;
        color: var(--muted);
        margin-top: 0.2rem;
    }

    /* Breakdown rows */
    .breakdown-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border);
    }
    .breakdown-row:last-child {
        border-bottom: none;
    }
    .breakdown-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        flex-shrink: 0;
        background-color: var(--dot-color, var(--muted));
    }
    .breakdown-name {
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        color: var(--ink);
        width: 108px;
        flex-shrink: 0;
    }
    .breakdown-track {
        flex-grow: 1;
        height: 6px;
        background-color: var(--bg);
        border-radius: 3px;
        overflow: hidden;
    }
    .breakdown-fill {
        height: 100%;
        border-radius: 3px;
        background-color: var(--fill-color, var(--muted));
    }
    .breakdown-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: var(--muted);
        width: 44px;
        text-align: right;
        flex-shrink: 0;
    }

    .architecture-note {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        color: var(--muted);
        text-align: center;
        margin-top: 2.5rem;
        line-height: 1.6;
    }

    hr {
        border-color: var(--border);
    }
    </style>
    """),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Loading tokenizers...")
def load_tokenizers():
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(LABEL_TOKENIZER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    return tokenizer, label_encoder


def predict(text: str, model, tokenizer, label_encoder):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequence, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE
    )
    probabilities = model.predict(padded, verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    ranked = [(label_encoder.index_word[i], float(probabilities[i])) for i in order]
    return ranked


model = load_model()
tokenizer, label_encoder = load_tokenizers()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    html_block("""
    <div class="masthead">
        <p class="masthead-title">BBC News Classifier</p>
        <p class="masthead-subtitle">Neural text classification &middot; five categories &middot; instant inference</p>
    </div>
    """),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
st.markdown('<p class="section-label">Article text</p>', unsafe_allow_html=True)

if "article_text" not in st.session_state:
    st.session_state.article_text = ""

# A widget's session_state value can't be reassigned after that widget has
# been instantiated in the same run. So the "load sample" button only sets
# a pending flag and reruns; the flag is applied here, *before* the
# text_area below is created, which is the one point in the run where
# writing to st.session_state.article_text is still allowed.
if st.session_state.get("_load_sample_pending", False):
    st.session_state.article_text = SAMPLE_TEXT
    st.session_state._load_sample_pending = False

text = st.text_area(
    "Article text",
    height=200,
    placeholder="Paste or type a news article here...",
    key="article_text",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 1])
with col1:
    classify_clicked = st.button("Classify", type="primary", use_container_width=True)
with col2:
    if st.button("Load sample article", type="secondary", use_container_width=True):
        st.session_state._load_sample_pending = True
        st.rerun()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if classify_clicked:
    if not text or not text.strip():
        st.warning("Enter a news article before classifying.")
    else:
        ranked = predict(text, model, tokenizer, label_encoder)
        top_label, top_conf = ranked[0]
        color = CATEGORY_COLORS.get(top_label, "#2F5D62")

        st.markdown('<p class="section-label">Prediction</p>', unsafe_allow_html=True)
        st.markdown(
            html_block(f"""
            <div class="result-card" style="--result-color: {color};">
                <p class="result-eyebrow">Predicted category</p>
                <p class="result-label">{top_label.capitalize()}</p>
                <p class="result-confidence">{top_conf * 100:.1f}% confidence</p>
            </div>
            """),
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-label">Full breakdown</p>', unsafe_allow_html=True)
        rows_html = ""
        for label, prob in ranked:
            c = CATEGORY_COLORS.get(label, "#6B7280")
            rows_html += (
                f'<div class="breakdown-row">'
                f'<span class="breakdown-dot" style="--dot-color: {c};"></span>'
                f'<span class="breakdown-name">{label.capitalize()}</span>'
                f'<span class="breakdown-track">'
                f'<span class="breakdown-fill" style="--fill-color: {c}; width: {prob * 100:.1f}%;"></span>'
                f'</span>'
                f'<span class="breakdown-value">{prob * 100:.1f}%</span>'
                f'</div>'
            )
        st.markdown(f'<div class="result-card">{rows_html}</div>', unsafe_allow_html=True)

st.markdown(
    html_block("""
    <p class="architecture-note">
        Embedding(1000, 16, masked) &rarr; GlobalAveragePooling1D &rarr; Dense(24, relu) &rarr; Dense(5, softmax)
    </p>
    """),
    unsafe_allow_html=True,
)
