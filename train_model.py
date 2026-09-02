import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("dataset/documents.csv")
print("==========================================")
print("DATASET INFORMATION")
print("==========================================")

print("Dataset loaded successfully!")
print("Total documents:", len(data))


# ============================================================
# CHECK DATASET
# ============================================================

print("\nCategory distribution:")

print(
    data["Category"].value_counts()
)


# ============================================================
# CLEAN DATA
# ============================================================

data["Text"] = (
    data["Text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

data["Category"] = (
    data["Category"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Remove empty documents

data = data[
    data["Text"] != ""
]


print("\nDocuments after cleaning:", len(data))


# ============================================================
# INPUT AND OUTPUT
# ============================================================

X = data["Text"]

y = data["Category"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining documents:", len(X_train))
print("Testing documents:", len(X_test))


# ============================================================
# TF-IDF + LOGISTIC REGRESSION
# ============================================================

model = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(

            lowercase=True,

            # Keep important words such as:
            # invoice, certificate, balance, resume, etc.
            stop_words=None,

            ngram_range=(1, 2),

            min_df=1,

            sublinear_tf=True,

            max_features=10000
        )
    ),

    (
        "classifier",

        LogisticRegression(

            max_iter=3000,

            C=10,

            class_weight="balanced",

            random_state=42
        )
    )
])


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n==========================================")
print("TRAINING MODEL")
print("==========================================")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully!")


# ============================================================
# TEST MODEL
# ============================================================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# TEST SAMPLE DOCUMENTS
# ============================================================

print("\n==========================================")
print("SAMPLE PREDICTIONS")
print("==========================================")


sample_documents = {

    "Invoice":
        "Invoice Number INV9999 Customer ABC Technologies "
        "Subtotal 15000 GST 2700 Total Amount 17700",

    "Bank Statement":
        "Bank Statement Account Number 123456 "
        "Transaction Date Credit Debit Balance",

    "Certificate":
        "Certificate of completion awarded for successfully "
        "completing Python programming training",

    "Research Paper":
        "Research paper presenting an analysis of machine "
        "learning algorithms for classification",

    "Resume":
        "Resume of a software engineer with Python Java "
        "SQL machine learning skills and experience"
}


for expected_category, document in sample_documents.items():

    prediction = model.predict(
        [document]
    )[0]

    print(
        f"Expected: {expected_category:16} "
        f"Predicted: {prediction}"
    )


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = "model/classifier.pkl"


joblib.dump(
    model,
    model_path
)


print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY")
print("==========================================")

print(
    f"File: {model_path}"
)

print("\nYou can now run:")
print("streamlit run app.py")