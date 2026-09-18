"""
train_baseline.py
Step 2: train classical baselines (Random Forest, XGBoost) on the manifest features.
Reads:  data/manifests/manifest.csv
Reports: accuracy, per-class P/R/F1, macro-F1, confusion matrix (on the test split).
"""
 
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
 

# Config

MANIFEST = Path("data/manifests/manifest.csv")
SEED = 42
 

# Load the dataset

df = pd.read_csv(MANIFEST)
print("Loaded", len(df), "rows from", MANIFEST)
 

# Separate metadata from features

meta_cols = ["utt_id", "speaker_id", "wav_path", "label", "split", "duration"]
feature_cols = [c for c in df.columns if c not in meta_cols]
print("Number of feature columns:", len(feature_cols))
 

# Split into train / val / test

train_df = df[df["split"] == "train"]
val_df   = df[df["split"] == "val"]
test_df  = df[df["split"] == "test"]
 
X_train, y_train = train_df[feature_cols], train_df["label"]
X_val,   y_val   = val_df[feature_cols],   val_df["label"]
X_test,  y_test  = test_df[feature_cols],  test_df["label"]
print("Train:", len(X_train), " Val:", len(X_val), " Test:", len(X_test))
 
labels = ["angry", "happy", "sad"]
 

# MODEL 1 — Random Forest

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
print("\nRandom Forest trained.")
 
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average="macro")
 
print("\n=== Random Forest — Test Results ===")
print("Accuracy:", round(acc_rf, 3))
print("Macro-F1:", round(f1_rf, 3))
print("\nPer-class report:")
print(classification_report(y_test, y_pred_rf))
 
# Confusion matrix (RF)
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=labels)
plt.figure(figsize=(9, 7))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 26}, cbar_kws={"label": "Count"},
            linewidths=1, linecolor="white")
plt.xlabel("Predicted", fontsize=18, labelpad=12)
plt.ylabel("True", fontsize=18, labelpad=12)
plt.title("Random Forest — Confusion Matrix (Test)", fontsize=20, pad=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15, rotation=0)
plt.tight_layout()
plt.savefig("results/figures/confusion_matrix_rf.png", dpi=300)
plt.close()
print("Saved -> results/figures/confusion_matrix_rf.png")
 
# Feature importance (RF)
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nTop 10 features:")
print(importance_df.head(10))
 
top_n = 15
top_features = importance_df.head(top_n)
plt.figure(figsize=(10, 8))
sns.barplot(data=top_features, x="importance", y="feature", color="#C59B27")
plt.xlabel("Importance", fontsize=16, labelpad=10)
plt.ylabel("Feature", fontsize=16, labelpad=10)
plt.title(f"Random Forest — Top {top_n} Feature Importances", fontsize=18, pad=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("results/figures/feature_importance_rf.png", dpi=300)
plt.close()
print("Saved -> results/figures/feature_importance_rf.png")
 

# MODEL 2 — XGBoost

# XGBoost needs numeric labels (0,1,2), not text. Encode them.
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)  ## learns angry->0, happy->1, sad->2 (alphabetical)
y_test_enc = le.transform(y_test)
 
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED,
    n_jobs=-1,
    eval_metric="mlogloss",
)
xgb.fit(X_train, y_train_enc)
print("\nXGBoost trained.")
 
y_pred_xgb_enc = xgb.predict(X_test)
# turn numeric predictions back into text labels for readable reports
y_pred_xgb = le.inverse_transform(y_pred_xgb_enc)
 
acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average="macro")
 
print("\n=== XGBoost — Test Results ===")
print("Accuracy:", round(acc_xgb, 3))
print("Macro-F1:", round(f1_xgb, 3))
print("\nPer-class report:")
print(classification_report(y_test, y_pred_xgb))
 
# Confusion matrix (XGBoost)
cm_xgb = confusion_matrix(y_test, y_pred_xgb, labels=labels)
plt.figure(figsize=(9, 7))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 26}, cbar_kws={"label": "Count"},
            linewidths=1, linecolor="white")
plt.xlabel("Predicted", fontsize=18, labelpad=12)
plt.ylabel("True", fontsize=18, labelpad=12)
plt.title("XGBoost — Confusion Matrix (Test)", fontsize=20, pad=16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15, rotation=0)
plt.tight_layout()
plt.savefig("results/figures/confusion_matrix_xgb.png", dpi=300)
plt.close()
print("Saved -> results/figures/confusion_matrix_xgb.png")
 

# Comparison table

compare = pd.DataFrame({
    "model": ["Random Forest", "XGBoost"],
    "accuracy": [round(acc_rf, 3), round(acc_xgb, 3)],
    "macro_f1": [round(f1_rf, 3), round(f1_xgb, 3)],
})
print("\n=== Model Comparison ===")
print(compare.to_string(index=False))
 
Path("results/metrics").mkdir(parents=True, exist_ok=True)
compare.to_csv("results/metrics/model_comparison.csv", index=False)
print("\nSaved -> results/metrics/model_comparison.csv")