import pandas as pd
import re
import joblib
import time

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

start_time = time.time()

print("=" * 60)
print("ANALISIS SENTIMEN GOJEK")
print("TF-IDF + MACHINE LEARNING")
print("BINARY SENTIMENT CLASSIFICATION")
print("=" * 60)

print("\n[1] Loading dataset...")

df = pd.read_csv("gojek_reviews.csv")

print("Dataset loaded successfully")
print(f"Original total data: {len(df)} rows")

print("\n[2] Removing neutral reviews (score = 3)...")

df = df[df["score"] != 3]

print(f"Remaining data: {len(df)} rows")

print("\n[3] Creating sentiment labels...")

def labeling(score):
    if score <= 2:
        return "Negatif"
    else:
        return "Positif"

df["sentiment"] = df["score"].apply(labeling)

print("Labeling completed")

print("\nSentiment Distribution Before Balancing:")
print(df["sentiment"].value_counts())

print("\n[4] Balancing dataset...")

negative_df = df[df["sentiment"] == "Negatif"]
positive_df = df[df["sentiment"] == "Positif"]

positive_df = positive_df.sample(
    n=len(negative_df),
    random_state=42
)

df = pd.concat([
    positive_df,
    negative_df
])

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nBalanced Sentiment Distribution:")
print(df["sentiment"].value_counts())

print("\n[5] Initializing preprocessing...")

def preprocessing(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text

print("Preprocessing ready")

print("\n[6] Preprocessing text data...")

tqdm.pandas()

df["clean_text"] = df["content"].progress_apply(
    preprocessing
)

print("Preprocessing completed")

print("\nExample preprocessing:")

print("\nBefore:")
print(df["content"].iloc[0])

print("\nAfter:")
print(df["clean_text"].iloc[0])

print("\n[7] TF-IDF Vectorization...")

X = df["clean_text"]
y = df["sentiment"]

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_tfidf = tfidf.fit_transform(X)

print("TF-IDF completed")
print(f"Feature shape: {X_tfidf.shape}")

print("\n[8] Splitting training and testing data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training data: {X_train.shape[0]}")
print(f"Testing data: {X_test.shape[0]}")

print("\n[9] Initializing models...")

models = {

    "Naive Bayes": MultinomialNB(
        alpha=0.5
    ),

    "SVM": LinearSVC(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = {}

trained_models = {}

print("\n[10] Training and evaluating models...")

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    print("\nTraining model...")

    model.fit(X_train, y_train)

    trained_models[model_name] = model

    print("Training completed")

    print("\nPredicting test data...")

    y_pred = model.predict(X_test)

    print("Prediction completed")

    print("\nEvaluating model...")

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    results[model_name] = accuracy

    print(f"\nAccuracy Score: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    model_filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + ".pkl"
    )

    joblib.dump(
        model,
        model_filename
    )

    print(f"\nModel saved: {model_filename}")

print("\n[11] Saving TF-IDF vectorizer...")

joblib.dump(
    tfidf,
    "tfidf_vectorizer.pkl"
)

print("Vectorizer saved successfully")

print("\n[12] Model Comparison")

print("\nAccuracy Summary:")

for model_name, accuracy in results.items():

    print(f"{model_name}: {accuracy:.4f}")

best_model = max(
    results,
    key=results.get
)

best_accuracy = results[best_model]

print("\nBest Performing Model:")
print(f"Model    : {best_model}")
print(f"Accuracy : {best_accuracy:.4f}")

print("\n[13] Testing manual predictions...")

sample_reviews = [

    "aplikasi sangat membantu dan cepat",

    "driver ramah dan cepat",

    "fitur gofood sangat bagus",

    "aplikasi lemot dan sering error",

    "gopay saya diblokir",

    "customer service tidak membantu",

    "update terbaru membuat aplikasi berat",

    "pengalaman buruk menggunakan aplikasi ini"
]

for review in sample_reviews:

    clean_review = preprocessing(review)

    vector = tfidf.transform([clean_review])

    print("\n" + "=" * 60)
    print(f"Review     : {review}")
    print(f"Clean Text : {clean_review}")
    print("=" * 60)

    for model_name, model in trained_models.items():

        prediction = model.predict(vector)[0]

        print(f"{model_name:<25}: {prediction}")

end_time = time.time()

total_time = end_time - start_time

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(f"Total execution time: {total_time:.2f} seconds")
print(f"Total execution time: {total_time / 60:.2f} minutes")