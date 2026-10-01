# ==========================================================
# CREDIT SCORING MODEL  (CodeAlpha ML Internship - Task 1)
# Dataset: UCI "Default of Credit Card Clients"
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ----------------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------------
# Put the file inside a folder named "data" next to this script.

pd.set_option("display.max_columns", None)

df = pd.read_excel("data/default of credit card clients.xls", header=1)

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)


# ----------------------------------------------------------
# 2. Remove ID column
# ----------------------------------------------------------

df = df.drop("ID", axis=1)
print("\nShape after removing ID:", df.shape)


# ----------------------------------------------------------
# 3. Check missing values and target balance
# ----------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum().sum(), "missing values in total")

TARGET = "default payment next month"

print("\nTarget distribution (0 = no default, 1 = default):")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True).round(3))


# ----------------------------------------------------------
# 4. Data cleaning
# ----------------------------------------------------------
# EDUCATION has undocumented values (0, 5, 6) -> group into "others" (4)
# MARRIAGE has undocumented value 0 -> group into "others" (3)

df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})


# ----------------------------------------------------------
# 5. Feature engineering (from financial history)
# ----------------------------------------------------------

bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
delay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# Average bill and average payment over 6 months
df["AVG_BILL"] = df[bill_cols].mean(axis=1)
df["AVG_PAYMENT"] = df[pay_cols].mean(axis=1)

# How much of the billed amount the customer actually paid
df["PAYMENT_RATIO"] = df[pay_cols].sum(axis=1) / (df[bill_cols].sum(axis=1).abs() + 1)

# How much of the credit limit is being used
df["CREDIT_UTILIZATION"] = df["AVG_BILL"] / (df["LIMIT_BAL"] + 1)

# Payment delay behaviour
df["MAX_DELAY"] = df[delay_cols].max(axis=1)
df["MONTHS_DELAYED"] = (df[delay_cols] > 0).sum(axis=1)

print("\nNew engineered features added:")
print(["AVG_BILL", "AVG_PAYMENT", "PAYMENT_RATIO",
       "CREDIT_UTILIZATION", "MAX_DELAY", "MONTHS_DELAYED"])


# ----------------------------------------------------------
# 6. Separate input features (X) and target (y)
# ----------------------------------------------------------

X = df.drop(TARGET, axis=1)
y = df[TARGET]

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ----------------------------------------------------------
# 7. Define categorical and numerical columns
# ----------------------------------------------------------

categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

numerical_features = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    # engineered features
    "AVG_BILL", "AVG_PAYMENT", "PAYMENT_RATIO",
    "CREDIT_UTILIZATION", "MAX_DELAY", "MONTHS_DELAYED",
]

print("\nCategorical columns:", categorical_features)
print("Number of numerical columns:", len(numerical_features))


# ----------------------------------------------------------
# 8. Split data into training and testing (stratified)
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining and testing sizes:")
print("X_train:", X_train.shape, "| X_test:", X_test.shape)


# ----------------------------------------------------------
# 9. Data preprocessing
# ----------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", StandardScaler(), numerical_features),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed.")
print("Processed X_train shape:", X_train_processed.shape)
print("Processed X_test shape:", X_test_processed.shape)


# ----------------------------------------------------------
# 10. Train the models
# ----------------------------------------------------------
# class_weight="balanced" handles the imbalance (only ~22% defaulters)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
}

predictions = {}
probabilities = {}

for name, model in models.items():
    model.fit(X_train_processed, y_train)
    predictions[name] = model.predict(X_test_processed)
    probabilities[name] = model.predict_proba(X_test_processed)[:, 1]
    print(f"{name} training completed.")


# ----------------------------------------------------------
# 11. Evaluate the models
# ----------------------------------------------------------

results_list = []

for name in models:
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions[name]),
        "Precision": precision_score(y_test, predictions[name]),
        "Recall": recall_score(y_test, predictions[name]),
        "F1 Score": f1_score(y_test, predictions[name]),
        "ROC-AUC": roc_auc_score(y_test, probabilities[name]),
    }
    results_list.append(row)

results = pd.DataFrame(results_list)

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")
print(results.round(4).to_string(index=False))

results.round(4).to_csv("model_comparison.csv", index=False)


# ----------------------------------------------------------
# 12. Choose the best model
# ----------------------------------------------------------

best_model_name = results.loc[results["ROC-AUC"].idxmax(), "Model"]
best_f1_name = results.loc[results["F1 Score"].idxmax(), "Model"]

print("\nBest model by ROC-AUC:", best_model_name)
print("Best model by F1 Score:", best_f1_name)


# ----------------------------------------------------------
# 13. Confusion matrices
# ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, name in zip(axes, models):
    cm = confusion_matrix(y_test, predictions[name])
    print(f"\nConfusion Matrix - {name}:")
    print(cm)
    ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"]).plot(
        ax=ax, colorbar=False
    )
    ax.set_title(name)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()


# ----------------------------------------------------------
# 14. ROC curves
# ----------------------------------------------------------

plt.figure(figsize=(8, 6))

for name in models:
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    auc = roc_auc_score(y_test, probabilities[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png", dpi=150)
plt.show()


# ----------------------------------------------------------
# 15. Feature importance (Random Forest)
# ----------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()
importances = pd.Series(
    models["Random Forest"].feature_importances_, index=feature_names
).sort_values(ascending=False)

print("\nTop 10 most important features:")
print(importances.head(10).round(4))

plt.figure(figsize=(9, 6))
importances.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()


# ----------------------------------------------------------
# 16. Save the models and preprocessor
# ----------------------------------------------------------

joblib.dump(
    {"preprocessor": preprocessor, **{k: v for k, v in models.items()}},
    "credit_scoring_models.pkl",
)

print("\nModels and preprocessor saved successfully.")
print("Credit Scoring Project Completed!")