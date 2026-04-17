"""
Sentiment Analysis & Fake Review Detection — Flask Backend
Uses LSTM (TensorFlow/Keras) for both sentiment and fake-review classification,
combined with heuristic rules for improved fake-review detection.
"""

import os
import re
import json
import numpy as np
import mysql.connector

# Suppress TF warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────
MAX_WORDS = 5000
MAX_LEN = 100
EMBEDDING_DIM = 64

# ─── Database Configuration ──────────────────────────────────────────────────
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'mathanmani@123',
    'database': 'fake_review_db'
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Database Error: {err}")
        return None

# ─── Sample Training Data ────────────────────────────────────────────────────
# Sentiment labels:  0 = Negative, 1 = Neutral, 2 = Positive
sentiment_texts = [
    # Positive
    "This product is absolutely amazing and I love it",
    "Wonderful quality and fast shipping, highly recommend",
    "Best purchase I have ever made, exceeded expectations",
    "Fantastic experience, will definitely buy again",
    "Great value for money, works perfectly",
    "I am very happy with this purchase, five stars",
    "Excellent product, exactly as described",
    "Super impressed with the quality and durability",
    "Outstanding service and product quality",
    "Love this item, it works like a charm",
    "Incredible build quality, worth every penny",
    "Very satisfied customer here, amazing product",
    "Perfect gift, my friend loved it",
    "Top notch quality, fast delivery, great seller",
    "This made my life so much easier, thank you",

    # Negative
    "Terrible product, broke after one day of use",
    "Waste of money, do not buy this garbage",
    "Awful quality, feels cheap and flimsy",
    "Worst purchase ever, completely disappointed",
    "Product stopped working after a week, very bad",
    "Horrible experience, took forever to arrive and was damaged",
    "Do not recommend, customer service was unhelpful",
    "Very poor quality, nothing like the pictures",
    "Regret buying this, total waste of money",
    "Extremely disappointed, product is defective",
    "Broke immediately, cheapest materials ever used",
    "Unusable product, arrived broken and scratched",
    "Scam product, looks nothing like advertised",
    "Terrible build quality, falls apart easily",
    "Would give zero stars if I could",

    # Neutral
    "The product is okay, nothing special",
    "Average quality, does what it says",
    "It is fine for the price, not great not bad",
    "Decent product, meets basic expectations",
    "Works as expected, nothing remarkable",
    "Standard quality, you get what you pay for",
    "It is alright, could be better could be worse",
    "Mediocre product, serves its purpose",
    "Not bad but not impressive either",
    "Fair quality for the price point",
    "Acceptable product, no complaints no praise",
    "Middle of the road, does the job",
]

sentiment_labels = (
    [2]*15 +   # Positive
    [0]*15 +   # Negative
    [1]*12     # Neutral
)

# Fake labels: 0 = Genuine, 1 = Fake
fake_texts = [
    # Genuine reviews
    "I bought this for my kitchen and it works well for daily use",
    "Good product but the delivery was a bit slow",
    "The quality is decent for this price range",
    "Used it for three months now, still going strong",
    "Nice product, though the color is slightly different from photo",
    "My daughter loves this toy, good quality plastic",
    "Works fine for basic tasks, battery life is average",
    "Solid build quality, heavier than expected",
    "Pretty good value, I would buy again at this price",
    "Arrived on time, packaging was good, product as expected",
    "After two weeks of use I can say it is reliable",
    "Decent quality, instructions could be clearer though",
    "Good for the price but don't expect premium quality",
    "My second purchase from this brand, consistent quality",
    "Fits perfectly, material feels comfortable",

    # Fake reviews
    "BEST PRODUCT EVER!!! BUY NOW!!! AMAZING AMAZING AMAZING!!!",
    "This is the greatest thing in the world you must buy it now",
    "Five stars five stars five stars absolutely perfect no flaws",
    "OMG this changed my life completely buy buy buy!!!",
    "Perfect perfect perfect I bought 50 of these amazing",
    "Greatest product on earth, nothing compares, buy immediately",
    "THIS IS AMAZING BUY IT NOW BEST DEAL EVER WOW WOW WOW",
    "I love it so much I bought one for everyone I know best ever",
    "Totally real review this product is the best in the universe",
    "Cannot believe how amazing this is, life changing purchase!!!",
    "Everyone should buy this right now, 100 percent perfect!!!",
    "asdf great product asdf would recommend asdf",
    "best best best best best best best best best product",
    "Seller asked me to leave five stars so here it is great product",
    "Got a discount for this review but the product is genuinely great",
]

fake_labels = [0]*15 + [1]*15

# ─── Build & Train Models ────────────────────────────────────────────────────

def build_lstm_model(vocab_size: int, num_classes: int) -> Sequential:
    """Build a Bidirectional LSTM classifier."""
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


print("[*] Training Sentiment LSTM model...")
sent_tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
sent_tokenizer.fit_on_texts(sentiment_texts)
sent_sequences = sent_tokenizer.texts_to_sequences(sentiment_texts)
sent_padded = pad_sequences(sent_sequences, maxlen=MAX_LEN, padding="post", truncating="post")
sent_labels_arr = np.array(sentiment_labels)

sentiment_model = build_lstm_model(MAX_WORDS, 3)
sentiment_model.fit(sent_padded, sent_labels_arr, epochs=50, batch_size=4, verbose=0)
print("[OK] Sentiment model ready")

print("[*] Training Fake-Review LSTM model...")
fake_tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
fake_tokenizer.fit_on_texts(fake_texts)
fake_sequences = fake_tokenizer.texts_to_sequences(fake_texts)
fake_padded = pad_sequences(fake_sequences, maxlen=MAX_LEN, padding="post", truncating="post")
fake_labels_arr = np.array(fake_labels)

fake_model = build_lstm_model(MAX_WORDS, 2)
fake_model.fit(fake_padded, fake_labels_arr, epochs=50, batch_size=4, verbose=0)
print("[OK] Fake-review model ready")

# ─── Heuristic Helpers ───────────────────────────────────────────────────────

FAKE_SIGNALS = {
    "excessive_caps": (
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1) > 0.5,
        "Excessive use of capital letters",
    ),
    "excessive_exclamation": (
        lambda t: t.count("!") > 3,
        "Excessive exclamation marks",
    ),
    "repetitive_words": (
        lambda t: _has_repetition(t),
        "Highly repetitive language",
    ),
    "very_short": (
        lambda t: len(t.split()) < 4,
        "Review is suspiciously short",
    ),
    "buy_now_pressure": (
        lambda t: any(p in t.lower() for p in ["buy now", "buy it now", "buy immediately", "must buy"]),
        "Contains aggressive purchase pressure language",
    ),
    "incentivised": (
        lambda t: any(p in t.lower() for p in ["discount for review", "free product", "asked me to leave"]),
        "Appears to be an incentivised review",
    ),
}


def _has_repetition(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 4:
        return False
    from collections import Counter
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(words) > 0.4


def heuristic_fake_score(text: str):
    """Return (fake_probability_boost, list_of_reasons)."""
    reasons = []
    score = 0.0
    for _key, (test_fn, reason) in FAKE_SIGNALS.items():
        if test_fn(text):
            reasons.append(reason)
            score += 0.15
    return min(score, 0.6), reasons


# ─── Sentiment → Star mapping ────────────────────────────────────────────────

def confidence_to_stars(pos_conf: float, neg_conf: float, neu_conf: float) -> float:
    """Map sentiment confidence values to a 1-5 star rating."""
    raw = pos_conf * 5.0 + neu_conf * 3.0 + neg_conf * 1.0
    return round(max(1.0, min(5.0, raw)), 1)


def sentiment_reason(label: str, confidence: float) -> str:
    """Generate a short human-readable explanation."""
    strength = "strongly" if confidence > 0.80 else "moderately" if confidence > 0.55 else "slightly"
    if label == "Positive":
        return f"The review expresses {strength} positive sentiment with favorable language and tone."
    elif label == "Negative":
        return f"The review expresses {strength} negative sentiment indicating dissatisfaction."
    else:
        return f"The review is {strength} neutral, expressing neither strong praise nor criticism."


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = data.get("review", "").strip()
    if not text:
        return jsonify({"error": "Review text is required."}), 400

    # ── Sentiment prediction ──
    seq = sent_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    sent_pred = sentiment_model.predict(padded, verbose=0)[0]

    neg_conf, neu_conf, pos_conf = float(sent_pred[0]), float(sent_pred[1]), float(sent_pred[2])
    sent_idx = int(np.argmax(sent_pred))
    sent_label = ["Negative", "Neutral", "Positive"][sent_idx]
    sent_confidence = float(sent_pred[sent_idx])

    # ── Fake detection (DL) ──
    f_seq = fake_tokenizer.texts_to_sequences([text])
    f_padded = pad_sequences(f_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    fake_pred = fake_model.predict(f_padded, verbose=0)[0]
    dl_genuine_conf = float(fake_pred[0])
    dl_fake_conf = float(fake_pred[1])

    # ── Fake detection (heuristic boost) ──
    h_boost, h_reasons = heuristic_fake_score(text)
    combined_fake_conf = min(dl_fake_conf + h_boost, 1.0)
    combined_genuine_conf = max(1.0 - combined_fake_conf, 0.0)

    is_fake = combined_fake_conf > 0.5
    fake_label = "Fake" if is_fake else "Genuine"
    fake_confidence = combined_fake_conf if is_fake else combined_genuine_conf

    # ── Fake reason ──
    if is_fake:
        if h_reasons:
            fake_reason = "Flagged as potentially fake: " + "; ".join(h_reasons) + "."
        else:
            fake_reason = "The deep-learning model detected patterns commonly seen in fake reviews."
    else:
        fake_reason = "The review appears authentic with natural language patterns and specific details."

    # ── Star rating ──
    stars = confidence_to_stars(pos_conf, neg_conf, neu_conf)

    # ── Database insertion ──
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            # Storing the input sentence (text), sentiments, and fake labels into the 'review' table.
            # Adjust column names below if they differ in your actual database table.
            sql = """
                INSERT INTO reviews 
                (review_text, sentiment_label, sentiment_confidence, fake_label, fake_confidence, stars)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            val = (text, sent_label, float(sent_confidence), fake_label, float(fake_confidence), float(stars))
            cursor.execute(sql, val)
            conn.commit()
            cursor.close()
            conn.close()
            print("[INFO] Review stored in database successfully.")
        else:
            print("[WARN] Could not connect to database, review not stored.")
    except Exception as e:
        print(f"[ERROR] Failed to insert into database: {e}")

    return jsonify({
        "sentiment": {
            "label": sent_label,
            "confidence": round(sent_confidence * 100, 2),
            "reason": sentiment_reason(sent_label, sent_confidence),
        },
        "fake_detection": {
            "label": fake_label,
            "confidence": round(fake_confidence * 100, 2),
            "reason": fake_reason,
        },
        "scores": {
            "positive": round(pos_conf * 100, 2),
            "negative": round(neg_conf * 100, 2),
            "neutral": round(neu_conf * 100, 2),
            "fake": round(combined_fake_conf * 100, 2),
            "genuine": round(combined_genuine_conf * 100, 2),
        },
        "stars": stars,
    })


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n>>> Server running at http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
