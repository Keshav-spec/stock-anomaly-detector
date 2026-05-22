import numpy as np
import streamlit as st

from transformers import pipeline


@st.cache_resource
def load_finbert():

    print("Loading FinBERT model...")

    model = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        return_all_scores=True
    )

    print("FinBERT loaded.")

    return model


finbert = load_finbert()


def get_sentiment_score(texts):

    if len(texts) == 0:
        return 0

    results = finbert(
        texts,
        truncation=True,
        max_length=512
    )

    scores = []

    for r in results:

        if isinstance(r, list):

            pos = next(
                (
                    x["score"]
                    for x in r
                    if x["label"].lower() == "positive"
                ),
                0
            )

            neg = next(
                (
                    x["score"]
                    for x in r
                    if x["label"].lower() == "negative"
                ),
                0
            )

            scores.append(pos - neg)

        elif isinstance(r, dict):

            label = r["label"].lower()

            score = r["score"]

            if label == "positive":
                scores.append(score)

            elif label == "negative":
                scores.append(-score)

            else:
                scores.append(0)

    return np.mean(scores)