import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest

from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier


# =========================
# 1. Load data
# =========================

np.random.seed(42)

red_df = pd.read_csv("data/winequality-red.csv", sep=";")
white_df = pd.read_csv("data/winequality-white.csv", sep=";")

# Phân biệt loại rượu
red_df["wine_type"] = 0
white_df["wine_type"] = 1

# Gộp 2 bộ dữ liệu
df = pd.concat([red_df, white_df], ignore_index=True)

print("Dataset shape:", df.shape)
print(df.head())


# =========================
# 2. Chuẩn bị X, y
# =========================

X = df.drop("quality", axis=1).values

# Binary classification:
# quality >= 6: good wine = 1
# quality < 6 : bad wine = 0
y = (df["quality"] >= 6).astype(int).values

print("\nClass distribution:")
print(pd.Series(y).value_counts())


# =========================
# 3. Chia train/test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 4. Hàm đánh giá
# =========================

results = []

def evaluate_model(name, model):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"{name:25s} | Accuracy: {acc:.4f} | Precision: {pre:.4f} | Recall: {rec:.4f} | F1-score: {f1:.4f}")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1-score": f1
    })


# =========================
# 5. Train model tự code
# =========================

dt_numpy = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=None
)

dt_numpy.fit(X_train, y_train)


rf_numpy = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt"
)

rf_numpy.fit(X_train, y_train)


# =========================
# 6. Train model sklearn
# =========================

dt_sklearn = SklearnDecisionTree(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

dt_sklearn.fit(X_train, y_train)


rf_sklearn = SklearnRandomForest(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42
)

rf_sklearn.fit(X_train, y_train)


# =========================
# 7. So sánh kết quả
# =========================

print("\n========== MODEL COMPARISON ==========")

evaluate_model("Decision Tree NumPy", dt_numpy)
evaluate_model("Random Forest NumPy", rf_numpy)
evaluate_model("Decision Tree sklearn", dt_sklearn)
evaluate_model("Random Forest sklearn", rf_sklearn)

os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv("results/metrics.csv", index=False)

print("\nSaved results to results/metrics.csv")