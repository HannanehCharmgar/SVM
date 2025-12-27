import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.inspection import permutation_importance


# ===============================
# 1. Load Dataset
# ===============================
csv_file = 'breast_cancer_wisconsin.csv'
df = pd.read_csv(csv_file)


# ===============================
# 2. Outlier Removal using Z-Score
# ===============================
features = df.drop('target', axis=1)

z_scores = np.abs(stats.zscore(features))
df_clean = df[(z_scores < 3).all(axis=1)]

print(f"Original shape: {df.shape}")
print(f"After outlier removal: {df_clean.shape}\n")


# ===============================
# 3. Correlation Analysis
# ===============================
corr_matrix = df_clean.drop('target', axis=1).corr()

plt.figure(figsize=(12,10))
sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    linewidths=0.5
)
plt.title("Feature Correlation Matrix")
plt.show()


# ===============================
# 4. Feature / Target Separation
# ===============================
X = df_clean.drop('target', axis=1).values
y = df_clean['target'].values   # 0: Malignant, 1: Benign


# ===============================
# 5. Train / Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ===============================
# 6. Feature Scaling
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===============================
# 7. Train SVM Model
# ===============================
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train_scaled, y_train)


# ===============================
# 8. Model Evaluation
# ===============================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision_b = precision_score(y_test, y_pred, pos_label=1)
recall_b = recall_score(y_test, y_pred, pos_label=1)
f1_b = f1_score(y_test, y_pred, pos_label=1)

precision_m = precision_score(y_test, y_pred, pos_label=0)
recall_m = recall_score(y_test, y_pred, pos_label=0)
f1_m = f1_score(y_test, y_pred, pos_label=0)

print(f"Accuracy: {accuracy*100:.4f}%\n")
print("Class-wise metrics:")
print(f"Benign -> Precision: {precision_b:.4f}, Recall: {recall_b:.4f}, F1-Score: {f1_b:.4f}")
print(f"Malignant -> Precision: {precision_m:.4f}, Recall: {recall_m:.4f}, F1-Score: {f1_m:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))


# ===============================
# 9. Confusion Matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Malignant', 'Benign'],
    yticklabels=['Malignant', 'Benign']
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ===============================
# 10. ROC Curve
# ===============================
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# ===============================
# 11. Feature Importance (Permutation Importance)
# ===============================

result = permutation_importance(
    model,
    X_test_scaled,
    y_test,
    n_repeats=20,
    random_state=42,
    scoring='accuracy'
)

feature_names = df_clean.drop('target', axis=1).columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance based on Permutation Importance:")
print(importance_df)


# --- Color mapping: Positive = Blue, Negative = Red
colors = importance_df['Importance'].apply(
    lambda x: 'steelblue' if x > 0 else 'indianred'
)

# --- Plot
plt.figure(figsize=(12, 7))
plt.barh(
    importance_df['Feature'],
    importance_df['Importance'],
    color=colors
)

plt.axvline(0, color='black', linewidth=1)
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Analysis (Positive vs Negative Effects)")

plt.gca().invert_yaxis()   # Most important feature on top
plt.tight_layout()
plt.show()
