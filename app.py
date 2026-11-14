"""
app.py
Task 8: Streamlit Web Application — Airline Tweet Sentiment Analyzer
Supports Heuristic (Lexicon + Negation), API-based (NLTK VADER), and
Machine Learning (TF-IDF + Balanced Logistic Regression Pipeline) approaches.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

from nlp_utils import (
    ensure_nltk_resources,
    clean_text,
    heuristic_sentiment,
)

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Airline Tweet Sentiment Analyzer",
    page_icon="✈️",
    layout="wide",
)

# 2. Safely download NLTK data at startup inside try/except with quiet=True (Task 8.4)
ensure_nltk_resources()

# Penn Treebank POS Tag descriptions for the collapsible POS expander
POS_TAG_MEANINGS = {
    "CC": "Coordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "IN": "Preposition / subordinating conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "MD": "Modal auxiliary",
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "RP": "Particle",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund / present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WRB": "Wh-adverb",
}


# 3. Cached Resource Loaders (Task 8.3)
@st.cache_resource
def load_ml_pipeline(model_path: str = "model/sentiment_model.joblib"):
    """Load the trained scikit-learn Pipeline once and cache in memory."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_resource
def load_vader_analyzer():
    """Load NLTK VADER SentimentIntensityAnalyzer once and cache in memory."""
    return SentimentIntensityAnalyzer()


ml_pipeline = load_ml_pipeline()
vader_sia = load_vader_analyzer()

# 4. Sidebar — About the Project & Dataset (Task 8.1)
with st.sidebar:
    st.header("✈️ About the Project")
    st.markdown(
        """
        **Practical Report 1 (PR 1)**  
        *NLP Fundamentals: From Raw Text to a Deployed App*  
        **Institute:** Red & White Skill Education

        ---
        ### 📊 Dataset Overview
        - **Dataset:** [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
        - **Size:** ~14,640 tweets across 6 US airlines
        - **Classes:** `negative` (63%), `neutral` (21%), `positive` (16%)
        - **Primary Metric:** **Macro-F1** (handles class imbalance fairly)

        ---
        ### ⚙️ NLP Pipeline Stages
        1. **RegEx Cleaning (`clean_text`):** Lowercase, remove URLs & `@mentions`, strip `#` symbol, remove numbers/punctuation.
        2. **Negation-Safe Stop-words:** Preserve `not`, `no`, `never`, `nor`.
        3. **Approaches Compared:**
           - **Heuristic:** 35+ positive & negative airline lexicon words + 2-word negation flip.
           - **API-based:** NLTK VADER compound polarity score.
           - **ML Pipeline:** `TfidfVectorizer(ngram_range=(1,2), min_df=2)` + `LogisticRegression(class_weight='balanced')`.
        """
    )

# 5. Main Area — Title & Quick Example Buttons (Task 8.1 & 8.3)
st.title("✈️ Airline Tweet Sentiment Analyzer")
st.markdown(
    "Analyze customer tweets using **Heuristic Rules**, **NLTK VADER**, or our trained **Machine Learning Pipeline** (`TF-IDF + Balanced Logistic Regression`)."
)

if "tweet_input" not in st.session_state:
    st.session_state.tweet_input = (
        "@JetBlue Thank you for the super smooth flight and friendly crew today! #GreatService"
    )

st.subheader("1. Try an Example Tweet or Enter Your Own")
col_ex1, col_ex2, col_ex3 = st.columns(3)

with col_ex1:
    if st.button("😊 Positive Example", use_container_width=True):
        st.session_state.tweet_input = (
            "@SouthwestAir Huge thanks to the gate agent in Denver! Amazing service and early arrival."
        )

with col_ex2:
    if st.button("😡 Negative (with Negation)", use_container_width=True):
        st.session_state.tweet_input = (
            "@united My flight was delayed 4 hours and my luggage is lost! Not happy with customer service at all. https://t.co/xyz"
        )

with col_ex3:
    if st.button("😐 Neutral Inquiry", use_container_width=True):
        st.session_state.tweet_input = (
            "@AmericanAir What time does boarding start for flight 415 from Chicago to Dallas tomorrow?"
        )

# Text Area Input
user_text = st.text_area(
    "Enter an airline tweet below (up to 500 characters):",
    key="tweet_input",
    height=110,
    placeholder="e.g., @united flight delayed 3 hours and rude staff, not happy!",
)

# Approach Selector (Task 8.3)
approach = st.radio(
    "2. Choose an NLP Sentiment Approach:",
    options=[
        "ML Model (TF-IDF Bigrams + Logistic Regression)",
        "API-based (NLTK VADER Compound Score)",
        "Heuristic (Airline Lexicon + Negation Flip)",
    ],
    horizontal=True,
)

# Predict Button
predict_clicked = st.button("🔍 Predict Sentiment", type="primary")


def display_coloured_sentiment(label: str, subtitle: str = "") -> None:
    """Display the predicted label using st.success / st.warning / st.error (Task 8.3)."""
    extra = f" — {subtitle}" if subtitle else ""
    if label == "positive":
        st.success(f"### 😊 Predicted Sentiment: **POSITIVE**{extra}")
    elif label == "neutral":
        st.warning(f"### 😐 Predicted Sentiment: **NEUTRAL**{extra}")
    else:
        st.error(f"### 😡 Predicted Sentiment: **NEGATIVE**{extra}")


if predict_clicked:
    st.divider()

    # Edge Case 1: Empty or whitespace-only input (Task 8.4)
    if not user_text or not user_text.strip():
        st.warning("⚠️ Please enter a non-empty tweet before clicking **Predict Sentiment**.")
        st.stop()

    # Edge Case 2: Very long input > 500 characters -> Truncate to 500 chars with a message (Task 8.4)
    working_text = user_text
    if len(working_text) > 500:
        st.info(
            f"ℹ️ Your input was **{len(working_text)} characters** long and has been truncated to the first **500 characters** for analysis."
        )
        working_text = working_text[:500]

    # Clean the text using shared nlp_utils.clean_text
    cleaned = clean_text(working_text)

    st.subheader("📊 Analysis Results")
    st.markdown("**Cleaned Text (`clean_text` output):**")
    if cleaned:
        st.code(cleaned, language="text")
    else:
        st.code("<Empty after removing URLs, @mentions, numbers, and emojis>", language="text")

    # Edge Case 3: Input that becomes empty after cleaning (e.g. URL-only or emoji-only) (Task 8.4)
    if not cleaned:
        st.warning(
            "⚠️ After removing URLs, `@mentions`, numbers, and emojis, no alphabetical words remained in the tweet."
        )
        if approach.startswith("API-based"):
            scores = vader_sia.polarity_scores(working_text)
            comp = scores["compound"]
            v_label = "positive" if comp >= 0.05 else ("negative" if comp <= -0.05 else "neutral")
            display_coloured_sentiment(v_label, f"VADER Compound Score on raw text: {comp:.3f}")
        else:
            display_coloured_sentiment("neutral", "Fallback prediction for empty cleaned text")
    else:
        # Normal Prediction Flow across the 3 Approaches
        if approach.startswith("Heuristic"):
            pred_label, pos_cnt, neg_cnt = heuristic_sentiment(working_text, return_details=True)
            display_coloured_sentiment(
                pred_label,
                f"Positive matches: {pos_cnt} | Negative matches: {neg_cnt} (with 2-word negation flip)",
            )

        elif approach.startswith("API-based"):
            scores = vader_sia.polarity_scores(working_text)
            comp = scores["compound"]
            if comp >= 0.05:
                pred_label = "positive"
            elif comp <= -0.05:
                pred_label = "negative"
            else:
                pred_label = "neutral"

            display_coloured_sentiment(pred_label, f"VADER Compound Score: {comp:.3f}")
            vader_df = pd.DataFrame(
                {
                    "Score Component": ["Negative (neg)", "Neutral (neu)", "Positive (pos)"],
                    "Value": [scores["neg"], scores["neu"], scores["pos"]],
                }
            ).set_index("Score Component")
            st.markdown("**VADER Polarity Breakdown:**")
            st.bar_chart(vader_df)

        else:
            # ML Model Approach
            if ml_pipeline is None:
                st.error(
                    "⚠️ `sentiment_model.joblib` was not found in the application directory. "
                    "Please run **Task 6.7** in `NLP_PR1.ipynb` to generate `sentiment_model.joblib`."
                )
            else:
                pred_label = ml_pipeline.predict([cleaned])[0]
                probs = ml_pipeline.predict_proba([cleaned])[0]
                classes = list(ml_pipeline.classes_)
                confidence = float(np.max(probs)) * 100.0

                display_coloured_sentiment(pred_label, f"Model Confidence: {confidence:.1f}%")

                st.markdown("**Predicted Class Probabilities (`predict_proba`):**")
                prob_df = pd.DataFrame(
                    {
                        "Sentiment Class": classes,
                        "Probability": [round(float(p), 4) for p in probs],
                    }
                ).set_index("Sentiment Class")
                st.bar_chart(prob_df)

    # Collapsible POS Tagging Section (Task 8.3)
    with st.expander("🔤 View Part-of-Speech (POS) Tags (NLTK Averaged Perceptron Tagger)"):
        if not cleaned:
            st.info("No alphabetical tokens available for POS tagging after cleaning.")
        else:
            try:
                tokens = word_tokenize(cleaned)
            except LookupError:
                tokens = cleaned.split()
            tagged_tokens = nltk.pos_tag(tokens)
            pos_rows = [
                {
                    "Token #": i + 1,
                    "Word": word,
                    "POS Tag": tag,
                    "Grammatical Description": POS_TAG_MEANINGS.get(tag, "Other / Symbol"),
                }
                for i, (word, tag) in enumerate(tagged_tokens)
            ]
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)