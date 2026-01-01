# ============================================================================
# COMPLETE SVM IMPLEMENTATION 
# ============================================================================

# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Display settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
#  LOAD DATA
# ============================================================================

print("="*60)
print("SVM FOR DIABETES PREDICTION")
print("="*60)

# Load data
df = pd.read_csv('diabetes.csv')
print("\n Data Information:")
print(f"Data size: {df.shape}")
print(f"\nClass distribution:\n{df['Outcome'].value_counts()}")
print(f"\nPositive class percentage: {(df['Outcome'].mean()*100):.1f}%")

# ============================================================================
#  DATA PREPROCESSING
# ============================================================================

print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Replace illogical zero values
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df[col] = df[col].replace(0, df[col].median())

print("✓ Zero values replaced with median")

# ============================================================================
#  DATA SPLITTING
# ============================================================================

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

# ============================================================================
#  FEATURE SCALING
# ============================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled")

# ============================================================================
#  SVM MODEL TRAINING
# ============================================================================

print("\n" + "="*60)
print("TRAINING SVM MODEL")
print("="*60)

# Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['linear', 'rbf']
}

svm = GridSearchCV(
    SVC(random_state=42, probability=True, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

svm.fit(X_train_scaled, y_train)

print(f"✓ Best parameters: {svm.best_params_}")
print(f"✓ Best CV score: {svm.best_score_:.3f}")

best_svm = svm.best_estimator_

# ============================================================================
#  PREDICTIONS
# ============================================================================

y_pred = best_svm.predict(X_test_scaled)
y_pred_proba = best_svm.predict_proba(X_test_scaled)[:, 1]

# ============================================================================
#  COMPREHENSIVE EVALUATION
# ============================================================================

print("\n" + "="*50)
print(" COMPREHENSIVE SVM MODEL EVALUATION")
print("="*50)

# A) Main metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n Main Metrics:")
print(f" Accuracy: {accuracy:.3f}")
print(f" Precision: {precision:.3f}")
print(f" Recall: {recall:.3f}")
print(f" F1-Score: {f1:.3f}")

# B) Complete classification report
print(f"\n Complete Classification Report:")
print(classification_report(y_test, y_pred, 
                           target_names=['No Diabetes', 'Has Diabetes'],
                           digits=3))

# C) Detailed confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n Confusion Matrix Details:")
print(f"True Positive (TP): {tp}")
print(f"True Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")

# Additional metrics
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\n Additional Metrics:")
print(f"Specificity: {specificity:.3f}")
print(f"Negative Predictive Value (NPV): {npv:.3f}")

# ============================================================================
# VISUALIZATION SECTION - همه پلات‌های اصلی
# ============================================================================

print("\n" + "="*60)
print("VISUALIZATION SECTION")
print("="*60)

# ایجاد figure اصلی با 4 پلات 
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('SVM Model Performance Analysis', fontsize=16, fontweight='bold', y=1.02)

# 1. Confusion Matrix Heatmap
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
ax1.set_ylabel('Actual', fontsize=12)
ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax1.set_xticklabels(['No Diabetes', 'Has Diabetes'])
ax1.set_yticklabels(['No Diabetes', 'Has Diabetes'])

# 2. Metrics Comparison Bar Chart
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']

bars = ax2.bar(metrics, values, color=colors, edgecolor='black')
ax2.set_title('Evaluation Metrics Comparison', fontsize=12, fontweight='bold')
ax2.set_ylabel('Value', fontsize=12)
ax2.set_ylim(0, 1)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{value:.3f}', ha='center', va='bottom', fontsize=11)

# 3. ROC Curve
ax3 = axes[1, 0]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

ax3.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax3.set_xlabel('False Positive Rate', fontsize=12)
ax3.set_ylabel('True Positive Rate', fontsize=12)
ax3.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax3.legend(loc="lower right")
ax3.grid(True, alpha=0.3)

# 4. Precision-Recall Curve
ax4 = axes[1, 1]
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

ax4.plot(recall_vals, precision_vals, color='purple', lw=2,
         label=f'Precision-Recall (AP = {avg_precision:.3f})')
ax4.set_xlabel('Recall', fontsize=12)
ax4.set_ylabel('Precision', fontsize=12)
ax4.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
ax4.legend(loc="best")
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
#  Feature Importance 
# ============================================================================

print("\n" + "="*60)
print("MODEL INTERPRETATION - FEATURE IMPORTANCE")
print("="*60)

plt.figure(figsize=(10, 6))

if best_svm.kernel == 'linear':
    # Linear coefficients
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': best_svm.coef_[0],
        'Abs_Coefficient': np.abs(best_svm.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\n Feature Importance (by absolute coefficient values):")
    print(coefficients.drop('Abs_Coefficient', axis=1).to_string(index=False))
    
    # Coefficient plot
    colors = ['red' if coef < 0 else 'green' for coef in coefficients['Coefficient']]
    plt.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.title('SVM Coefficients\n(Negative: decreases risk, Positive: increases risk)', 
              fontsize=14, fontweight='bold')
    
else:
    # Permutation importance for non-linear kernels
    perm_importance = permutation_importance(
        best_svm, X_test_scaled, y_test, 
        n_repeats=10, random_state=42
    )
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    print("\n Feature Importance (Permutation Importance):")
    print(feature_importance.drop('Std', axis=1).to_string(index=False))
    
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], 
            xerr=feature_importance['Std'], color='steelblue', edgecolor='black')
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('SVM Feature Importance (Permutation Method)', 
             fontsize=14, fontweight='bold')

plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
#   Threshold Analysis 
# ============================================================================

print("\n" + "="*60)
print("THRESHOLD ANALYSIS")
print("="*60)

# Performance at different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\nThreshold | Precision | Recall | F1-Score")
print("-" * 40)

precisions = []
recalls = []
f1_scores = []

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    p = precision_score(y_test, y_pred_thresh, zero_division=0)
    r = recall_score(y_test, y_pred_thresh)
    f = f1_score(y_test, y_pred_thresh)
    
    precisions.append(p)
    recalls.append(r)
    f1_scores.append(f)
    
    print(f"{thresh:.1f}       | {p:.3f}     | {r:.3f}  | {f:.3f}")

# نمودار Threshold Analysis
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, 'b-', lw=2, label='Precision', marker='o')
plt.plot(thresholds, recalls, 'g-', lw=2, label='Recall', marker='s')
plt.plot(thresholds, f1_scores, 'r-', lw=3, label='F1-Score', marker='^')

plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('SVM - Threshold Analysis', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Mark optimal threshold
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
plt.axvline(x=optimal_threshold, color='black', linestyle='--', alpha=0.7)
plt.text(optimal_threshold + 0.02, 0.1, f'Optimal: {optimal_threshold:.2f}', 
        fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
#   Probability Distribution
# ============================================================================

print("\n" + "="*60)
print("PROBABILITY DISTRIBUTION ANALYSIS")
print("="*60)

plt.figure(figsize=(12, 5))

# هیستوگرام
plt.subplot(1, 2, 1)
bins = np.linspace(0, 1, 21)
plt.hist(y_pred_proba[y_test == 0], bins=bins, alpha=0.7, 
        label='No Diabetes', color='blue', edgecolor='black', density=True)
plt.hist(y_pred_proba[y_test == 1], bins=bins, alpha=0.7, 
        label='Has Diabetes', color='red', edgecolor='black', density=True)
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Probability Distribution by Class', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Box plot
plt.subplot(1, 2, 2)
data_to_plot = [y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]]
box = plt.boxplot(data_to_plot, patch_artist=True, 
                  labels=['No Diabetes', 'Has Diabetes'],
                  medianprops={'color': 'black', 'linewidth': 2})

colors = ['lightblue', 'lightcoral']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.ylabel('Predicted Probability', fontsize=12)
plt.title('Probability Distribution - Box Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
#    Error Analysis
# ============================================================================

print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

plt.figure(figsize=(8, 6))
error_types = ['False Positives', 'False Negatives', 'True Positives', 'True Negatives']
error_counts = [fp, fn, tp, tn]
colors_error = ['red', 'orange', 'green', 'blue']

bars = plt.bar(error_types, error_counts, color=colors_error, 
               edgecolor='black', linewidth=1.5, alpha=0.8)

plt.title('SVM Error Analysis', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Count', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# Add count labels
for bar, count in zip(bars, error_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             str(count), ha='center', va='bottom', 
             fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
#   Class Distribution
# ============================================================================

print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)

plt.figure(figsize=(10, 4))

# توزیع کلاس در داده‌های train و test
plt.subplot(1, 2, 1)
train_counts = y_train.value_counts()
test_counts = y_test.value_counts()

x = np.arange(2)
width = 0.35

plt.bar(x - width/2, train_counts.values, width, label='Train', color='skyblue', edgecolor='black')
plt.bar(x + width/2, test_counts.values, width, label='Test', color='lightcoral', edgecolor='black')

plt.xlabel('Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Class Distribution: Train vs Test', fontsize=14, fontweight='bold')
plt.xticks(x, ['No Diabetes', 'Has Diabetes'])
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

# Pie chart
plt.subplot(1, 2, 2)
class_counts = df['Outcome'].value_counts()
colors_pie = ['lightblue', 'lightcoral']
wedges, texts, autotexts = plt.pie(class_counts, labels=['No Diabetes', 'Has Diabetes'],
                                   autopct='%1.1f%%', colors=colors_pie,
                                   startangle=90, explode=(0.05, 0), 
                                   textprops={'fontsize': 11})

plt.title('Overall Class Distribution', fontsize=14, fontweight='bold')

# Make autotexts bold
for autotext in autotexts:
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.show()

# ============================================================================
#  FINAL SUMMARY
# ============================================================================

print("\n" + "="*50)
print(" RESULTS SUMMARY")
print("="*50)

print(f"\n Model Strengths:")
if recall > 0.7:
    print(f"  • High Recall ({recall:.3f}): Good ability to identify actual patients")
if precision > 0.7:
    print(f"  • High Precision ({precision:.3f}): Good accuracy in positive predictions")
if f1 > 0.7:
    print(f"  • Good balance between Precision and Recall (F1-Score: {f1:.3f})")

print(f"\n Areas for Improvement:")
if fp > fn:
    print(f"  • {fp} False Positives: May misclassify healthy people as diabetic")
if fn > fp:
    print(f"  • {fn} False Negatives: May miss actual diabetic patients")

print(f"\n Recommendation:")
if recall < precision:
    print("  • If identifying all patients is important, lower the threshold")
else:
    print("  • If avoiding false alarms is important, increase the threshold")

# ============================================================================
# FINAL PERFORMANCE SUMMARY TABLE
# ============================================================================

print("\n" + "="*50)
print(" FINAL PERFORMANCE SUMMARY")
print("="*50)

summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
               'Specificity', 'NPV', 'ROC-AUC'],
    'Value': [accuracy, precision, recall, f1, 
              specificity, npv, roc_auc],
    'Interpretation': [
        'Overall correctness',
        'Correct positive predictions',
        'Ability to find all positives',
        'Balance of Precision and Recall',
        'Ability to identify negatives',
        'Correct negative predictions',
        'Overall classification ability'
    ]
})

print(summary_df.to_string(index=False))

print("\n" + "="*60)
print("✅ ALL VISUALIZATIONS COMPLETED - 9 PLOTS GENERATED")
print("="*60)