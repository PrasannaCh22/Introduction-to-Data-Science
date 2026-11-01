"""
PROJECT 1: SPAM HAM CLASSIFICATION
Classifies a text message as SPAM or HAM using TF-IDF + Multinomial Naive Bayes.

Run:
    pip install -r requirements.txt
    python 1_spam_ham_classification.py

No external dataset is required. A small demonstration dataset is included.
For a larger dataset, place a CSV named spam.csv beside this file with columns:
    label, message
Labels should be "spam" or "ham".
"""

from pathlib import Path
import sys

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except ImportError as exc:
    print("A required package is missing.")
    print("Install dependencies with: pip install pandas scikit-learn")
    raise SystemExit(1) from exc


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "spam.csv"

SAMPLE_DATA = [
    ("ham", "Hey, are we still meeting for lunch today?"),
    ("ham", "Please send me the notes from today's class."),
    ("ham", "I will reach the college in ten minutes."),
    ("ham", "Can you call me when you are free?"),
    ("ham", "The assignment submission is tomorrow."),
    ("ham", "Thanks for helping me with the project."),
    ("ham", "Your parcel has been delivered to the hostel office."),
    ("ham", "Let's meet at the library after class."),
    ("ham", "Don't forget to bring your laptop."),
    ("ham", "The lecture starts at nine in the morning."),
    ("ham", "I am travelling home this weekend."),
    ("ham", "Please share the presentation with our group."),
    ("spam", "Congratulations! You have won a free prize. Claim now!"),
    ("spam", "URGENT! You have been selected for a cash reward."),
    ("spam", "Win a brand new phone by clicking this link."),
    ("spam", "You are a lucky winner. Claim your free gift today."),
    ("spam", "Get instant cash. Send your bank details now."),
    ("spam", "Limited offer! Buy now and get a huge discount."),
    ("spam", "You have won a lottery. Call this number immediately."),
    ("spam", "Free entry in a weekly prize draw. Reply now."),
    ("spam", "Earn money fast from home. Sign up today."),
    ("spam", "Claim your exclusive reward before it expires."),
    ("spam", "Congratulations! Your account won a special bonus."),
    ("spam", "Click here now to receive your free voucher."),
]


def load_data():
    """Load spam.csv when available; otherwise use the built-in sample."""
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        required = {"label", "message"}
        if not required.issubset(df.columns):
            raise ValueError("spam.csv must contain 'label' and 'message' columns.")
        df = df[["label", "message"]].dropna()
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df["message"] = df["message"].astype(str).str.strip()
        df = df[df["label"].isin(["spam", "ham"])]
    else:
        print("spam.csv not found. Using the built-in demonstration dataset.")
        df = pd.DataFrame(SAMPLE_DATA, columns=["label", "message"])

    if df["label"].nunique() < 2 or len(df) < 8:
        raise ValueError("Dataset must contain at least 8 messages from both classes.")
    return df


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.25,
        random_state=42, stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\n========== SPAM / HAM CLASSIFICATION ==========")
    print(f"Total messages: {len(df)}")
    print(f"Training messages: {len(X_train)}")
    print(f"Testing messages: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (labels: ham, spam):")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))

    print("\nTry your own messages (type 'quit' to stop).")
    while True:
        message = input("\nEnter a message: ").strip()
        if message.lower() in {"quit", "exit"}:
            break
        if not message:
            print("Please enter a non-empty message.")
            continue
        result = model.predict([message])[0]
        print("Prediction:", result.upper())


if __name__ == "__main__":
    main()
