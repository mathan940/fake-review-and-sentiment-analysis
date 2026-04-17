"""
Sentiment Analysis & Fake Review Detection — Flask Backend
Uses TF-IDF + Logistic Regression for fast, reliable classification.
Combined with heuristic rules for improved fake-review detection.
"""

import os
import re
import numpy as np
import mysql.connector
from collections import Counter

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)

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

# ─── Train Models (fast: TF-IDF + Logistic Regression) ───────────────────────

print("[*] Training Sentiment model...")
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ('clf', LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')),
])
sentiment_pipeline.fit(sentiment_texts, sentiment_labels)
print("[OK] Sentiment model ready")

print("[*] Training Fake-Review model...")
fake_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ('clf', LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')),
])
fake_pipeline.fit(fake_texts, fake_labels)
print("[OK] Fake-review model ready")

# ─── Heuristic Helpers ───────────────────────────────────────────────────────

def _has_repetition(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 4:
        return False
    counts = Counter(words)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(words) > 0.4

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
    sent_proba = sentiment_pipeline.predict_proba([text])[0]
    # Classes order from pipeline: 0=Negative, 1=Neutral, 2=Positive
    classes = list(sentiment_pipeline.classes_)
    neg_conf = float(sent_proba[classes.index(0)])
    neu_conf = float(sent_proba[classes.index(1)])
    pos_conf = float(sent_proba[classes.index(2)])

    sent_idx = int(np.argmax(sent_proba))
    sent_label = ["Negative", "Neutral", "Positive"][classes[sent_idx]]
    sent_confidence = float(sent_proba[sent_idx])

    # ── Fake detection (ML) ──
    fake_proba = fake_pipeline.predict_proba([text])[0]
    fake_classes = list(fake_pipeline.classes_)
    dl_genuine_conf = float(fake_proba[fake_classes.index(0)])
    dl_fake_conf = float(fake_proba[fake_classes.index(1)])

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
            fake_reason = "The model detected patterns commonly seen in fake reviews."
    else:
        fake_reason = "The review appears authentic with natural language patterns and specific details."

    # ── Star rating ──
    stars = confidence_to_stars(pos_conf, neg_conf, neu_conf)

    # ── Database insertion ──
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
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
