import pandas as pd

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print("Fake news data loaded:", len(fake))
print("Genuine news data loaded:", len(true))


fake["label"] = 0
true["label"] = 1


data = pd.concat([fake, true], axis=0)


data = data[["text", "label"]]


data = data.dropna()
data = data.drop_duplicates()


data = data.sample(frac=1, random_state=42)

print("\nDataset loaded successfully!")
print(data.head())

print("\nLabel distribution:")
print(data["label"].value_counts())


import re

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

data["text"] = data["text"].apply(clean_text)

print(data["text"].head())

from sklearn.model_selection import train_test_split

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

print("Model training completed!")



from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation Results")

print("Accuracy:", accuracy * 100, "%")

# Classification report
print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Fake News", "Genuine News"]
))

# Confusion matrix
print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))


import joblib
import os

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/fake_news_model.pkl")

print("Model saved successfully!")