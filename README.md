<div align="center">

# BBC News Classifier

**A neural network that reads a news article and tells you what it's about — instantly.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-2F5D62?style=for-the-badge)](https://bbc-news-classifier-nlp.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-1C1F26?style=flat-square)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-1C1F26?style=flat-square)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-1C1F26?style=flat-square)](#license)

**[Try it live →](https://bbc-news-classifier-nlp.streamlit.app/)**

</div>

---

## What it does

Paste any English news article and the model classifies it into one of five
categories — **Business**, **Entertainment**, **Politics**, **Sport**, or
**Tech** — with a confidence score, in under a second. No API keys, no
sign-up, no setup. [Open the live app](https://bbc-news-classifier-nlp.streamlit.app/)
and try it on a real headline right now.

<div align="center">

| Accuracy | Loss |
|---|---|
| ![accuracy](assets/accuracy.png) | ![loss](assets/loss.png) |

**~94.8% validation accuracy**

![confusion matrix](assets/confusion_matrix.png)

</div>

---

## How it works

```mermaid
flowchart LR
    A[Raw article text] --> B[Clean & tokenize]
    B --> C[Pad / truncate to 120 tokens]
    C --> D[Embedding 1000 x 16, masked]
    D --> E[GlobalAveragePooling1D]
    E --> F[Dense 24, relu]
    F --> G[Dense 5, softmax]
    G --> H[Predicted category]

    classDef stage fill:#F7F7F5,stroke:#1C1F26,stroke-width:1px,color:#1C1F26;
    classDef output fill:#2F5D62,stroke:#2F5D62,stroke-width:1px,color:#FFFFFF;
    class A,B,C,D,E,F stage;
    class G,H output;
```

A lightweight embedding network — no transformers, no GPU required — trained
on 2,225 labeled BBC articles. Small enough to retrain in under a minute,
accurate enough to use in production for well-defined categories.

---

## Where this is actually useful

This isn't just a classroom exercise — the underlying primitive (map text to
a category, instantly, cheaply) is the backbone of several real products.

```mermaid
flowchart TB
    M[Text classifier core]
    M --> N[News & publishing]
    M --> S[Short-form video & social]
    M --> C[Messaging & chat]

    N --> N1[Auto-tag incoming wire articles by section]
    N --> N2[Route stories to the right newsroom desk]
    N --> N3[Personalize a reader's feed by category weighting]

    S --> S1[Auto-suggest a category or hashtag set for a reel caption]
    S --> S2[Group user-generated clips into topic shelves]

    C --> C1[Auto-sort forwarded news links in a group chat]
    C --> C2[Flag off-topic messages in a topic-specific channel]

    classDef core fill:#1C1F26,stroke:#1C1F26,color:#FFFFFF;
    classDef domain fill:#F7F7F5,stroke:#1C1F26,color:#1C1F26;
    classDef leaf fill:#FFFFFF,stroke:#E3E1DC,color:#1C1F26;
    class M core;
    class N,S,C domain;
    class N1,N2,N3,S1,S2,C1,C2 leaf;
```

**News & publishing platforms.** The most direct fit — this is exactly the
kind of classifier a news aggregator or CMS uses to auto-tag incoming
articles by section instead of relying on manual tagging, or to route a
story to the right editorial desk before a human ever touches it.

**Short-form video & social apps.** Applied to a caption, transcript, or
auto-generated subtitle rather than the video itself, the same architecture
can suggest a topic for a clip — useful for organizing a content library or
powering topic-based recommendations on a reels-style feed.

**Chat & messaging products.** Applied to forwarded links or long text
messages, a classifier like this can auto-sort or auto-label content inside
a group chat — for example, flagging a message as Sport-related in a chat
that's mostly about Politics, or auto-archiving news links by topic.

> **Before reusing this exact model:** it was trained only on English,
> article-length BBC text. Arabic content, captions, or chat messages would
> need a retrained model on in-domain, in-language data — the architecture
> transfers directly, the trained weights do not.

---

## Project structure

```
bbc-news-classifier/
├── app.py                     Streamlit app (deployable entry point)
├── requirements.txt
├── data/
│   └── bbc-text.csv           BBC News dataset, 2,225 articles
├── models/                    Trained artifacts loaded by app.py
│   ├── bbc_news_model.keras
│   ├── tokenizer.pkl
│   └── label_tokenizer.pkl
├── src/                       Reusable training / inference code
│   ├── config.py              All hyperparameters and paths
│   ├── data_preprocessing.py  Loading, cleaning, tokenizing, label encoding
│   ├── model.py                Model architecture
│   ├── train.py                 python -m src.train — trains & saves everything
│   └── predict.py               Shared inference helper used by app.py
├── notebooks/
│   └── BBC_NEWS.ipynb         Exploration notebook using the src/ modules
└── assets/                    Training curve plots referenced above
```

---

## Getting started

```bash
git clone https://github.com/muhammed-alaa74/bbc-news-classifier.git
cd bbc-news-classifier
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`.

**Retrain from scratch:**

```bash
python -m src.train
```

Rebuilds the tokenizer and label encoder, trains the model, and overwrites
the artifacts in `models/` plus the plots in `assets/`.

---

## Deployment

**Live now:** [bbc-news-classifier-nlp.streamlit.app](https://bbc-news-classifier-nlp.streamlit.app/) — deployed on Streamlit Community Cloud.

To deploy your own copy:

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo/branch, set **Main file path** to `app.py`.
4. Click **Deploy**.

<details>
<summary>Docker (alternative deployment)</summary>

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

</details>

---

## Dataset

[BBC News dataset](https://raw.githubusercontent.com/PacktPublishing/Python-Natural-Language-Processing-Cookbook-Second-Edition/main/data/bbc-text.csv) —
2,225 articles across 5 categories, from the Python Natural Language
Processing Cookbook companion repository.

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**[Try the live demo →](https://bbc-news-classifier-nlp.streamlit.app/)**

</div>
