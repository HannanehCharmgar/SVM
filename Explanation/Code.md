# توضیحات کد الگوریتم SVM 

##  بخش 1: وارد کردن کتابخانه‌ها
```
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
```
pandas/numpy: برای مدیریت و پردازش داده‌ها

matplotlib/seaborn: برای مصورسازی نتایج

scikit-learn: برای پیاده‌سازی الگوریتم‌های یادگیری ماشین

GridSearchCV: برای بهینه‌سازی خودکار پارامترها

SVC: برای پیاده‌سازی Support Vector Classifier

Evaluation: دقت، precision، recall، F1-score و دیگر متریک‌ها

## بخش 2: بارگذاری داده

```
# ============================================================================
#  LOAD DATA
# ============================================================================

print("="*60)
print("SVM FOR DIABETES PREDICTION")
print("="*60)

df = pd.read_csv('diabetes.csv')

print("\n Data Information:")
print(f"Data size: {df.shape}")
print(f"\nClass distribution:\n{df['Outcome'].value_counts()}")
print(f"\nPositive class percentage: {(df['Outcome'].mean()*100):.1f}%")
```
## output:
```
============================================================
SVM FOR DIABETES PREDICTION
============================================================

 Data Information:
Data size: (768, 9)

Class distribution:
Outcome
0    500
1    268
Name: count, dtype: int64

Positive class percentage: 34.9%
```

بارگذاری داده‌های دیابت از فایل CSV

نمایش ابعاد داده (768 نمونه، 9 ویژگی)

نمایش توزیع کلاس‌ها:

کلاس 0 (بدون دیابت): 500 نمونه (65.1%)

کلاس 1 (دیابت): 268 نمونه (34.9%)

داده نامتوازن است (Imbalanced Data)

## بخش 3: پیش‌پردازش داده

```
# ============================================================================
#  DATA PREPROCESSING
# ============================================================================

print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df[col] = df[col].replace(0, df[col].median())

print("✓ Zero values replaced with median")
```
شناسایی ستون‌های مشکل‌دار: مقادیر صفر در این ستون‌ها غیرمنطقی هستند

جایگزینی با میانه: مقادیر صفر با میانه هر ستون جایگزین می‌شوند

ستون‌های پردازش شده:

Glucose (گلوکز)

BloodPressure (فشار خون)

SkinThickness (ضخامت پوست)

Insulin (انسولین)

BMI (شاخص توده بدنی)

## بخش 4: تقسیم داده

```
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
```
## output:
```

✓ Train set: (614, 8)
✓ Test set: (154, 8)
```
جداسازی ویژگی‌ها و برچسب:

X: همه ویژگی‌ها به جز Outcome

y: ستون Outcome (برچسب 0 یا 1)

تقسیم داده:

80% برای آموزش (614 نمونه)

20% برای آزمون (154 نمونه)

stratify=y: حفظ نسبت کلاس‌ها در هر بخش

random_state=42: برای تکرارپذیری نتایج

## بخش 5: استانداردسازی ویژگی‌ها

```
# ============================================================================
#  FEATURE SCALING
# ============================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled")
```
StandardScaler: استانداردسازی ویژگی‌ها به میانگین 0 و انحراف معیار 1

fit_transform: روی داده آموزش محاسبه پارامترها + تبدیل

transform: روی داده آزمون فقط تبدیل با پارامترهای آموزش

دلیل استانداردسازی: SVM به مقیاس ویژگی‌ها حساس است.

##  بخش 6: آموزش مدل SVM با GridSearchCV
```
# ============================================================================
#  SVM MODEL TRAINING
# ============================================================================

print("\n" + "="*60)
print("TRAINING SVM MODEL")
print("="*60)

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

best_svm = svm.best_estimator
```
## output:
```

============================================================
TRAINING SVM MODEL
============================================================
✓ Best parameters: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
✓ Best CV score: 0.675
```
Grid Search: جستجوی بهترین ترکیب پارامترها

پارامترهای جستجو:

C: [0.1, 1, 10, 100] - پارامتر جریمه

gamma: ['scale', 'auto', 0.1, 0.01] - پارامتر کرنل RBF

kernel: ['linear', 'rbf'] - نوع کرنل

تنظیمات GridSearchCV:

cv=5: اعتبارسنجی متقابل ۵-تایی

scoring='f1': استفاده از F1-score برای ارزیابی

n_jobs=-1: استفاده از تمام هسته‌های CPU

class_weight='balanced': مدیریت عدم توازن کلاس

نتایج:

بهترین پارامترها: C=1, gamma='scale', kernel='rbf'

بهترین امتیاز: 0.675

## بخش 7: پیش‌بینی
```
# ============================================================================
#  PREDICTIONS
# ============================================================================

y_pred = best_svm.predict(X_test_scaled)
y_pred_proba = best_svm.predict_proba(X_test_scaled)[:, 1]
```
d: پیش‌بینی کلاس (0 یا 1)

y_pred_proba: احتمال تعلق به کلاس مثبت (کلاس 1)

[:, 1]: فقط ستون دوم که مربوط به کلاس مثبت است

## بخش 8: ارزیابی جامع مدل

```
# ============================================================================
#  COMPREHENSIVE EVALUATION
# ============================================================================

print("\n" + "="*50)
print(" COMPREHENSIVE SVM MODEL EVALUATION")
print("="*50)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n Main Metrics:")
print(f" Accuracy: {accuracy:.3f}")
print(f" Precision: {precision:.3f}")
print(f" Recall: {recall:.3f}")
print(f" F1-Score: {f1:.3f}")

print(f"\n Complete Classification Report:")
print(classification_report(
    y_test, y_pred, 
    target_names=['No Diabetes', 'Has Diabetes'],
    digits=3
))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n Confusion Matrix Details:")
print(f"True Positive (TP): {tp}")
print(f"True Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\n Additional Metrics:")
print(f"Specificity: {specificity:.3f}")
print(f"Negative Predictive Value (NPV): {npv:.3f}")
```
## output:
```

==================================================
 COMPREHENSIVE SVM MODEL EVALUATION
==================================================

 Main Metrics:
 Accuracy: 0.753
 Precision: 0.618
 Recall: 0.778
 F1-Score: 0.689

 Complete Classification Report:
              precision    recall  f1-score   support

 No Diabetes      0.860     0.740     0.796       100
Has Diabetes      0.618     0.778     0.689        54

    accuracy                          0.753       154
   macro avg      0.739     0.759     0.742       154
weighted avg      0.775     0.753     0.758       154


 Confusion Matrix Details:
True Positive (TP): 42
True Negative (TN): 74
False Positive (FP): 26
False Negative (FN): 12

 Additional Metrics:
Specificity: 0.740
Negative Predictive Value (NPV): 0.860
```
متریک‌های اصلی:

Accuracy: 0.753

Precision: 0.618 (نرخ مثبت واقعی)

Recall: 0.778 (حساسیت مدل)

F1-Score: 0.689 (میانگین هارمونیک precision و recall)

ماتریس درهم‌ریختگی:

TP: 42 (دیابتی که درست تشخیص داده شد)

TN: 74 (سالم که درست تشخیص داده شد)

FP: 26 (سالم که اشتباه دیابتی تشخیص داده شد)

FN: 12 (دیابتی که تشخیص داده نشد)

متریک‌های اضافی:

Specificity: 0.740 (نرخ منفی واقعی)

NPV: 0.860 (ارزش پیش‌بینی منفی)

## بخش 9 : پلات های ارزیابی

<img width="1189" height="1025" alt="image" src="https://github.com/user-attachments/assets/8f7ade47-21a7-4b15-ad0b-f0375ba3bfeb" />

##  1. ماتریس درهم‌ریختگی (Confusion Matrix)
- True Negative (TN) = 74 → مواردی که واقعاً دیابت نداشتند و مدل درست تشخیص داده.
- False Positive (FP) = 26 → مواردی که دیابت نداشتند، اما مدل اشتباهی گفته "دارد".
- False Negative (FN) = 12 → مواردی که دیابت داشتند، اما مدل اشتباهی گفته "ندارد".
- True Positive (TP) = 42 → مواردی که دیابت داشتند و مدل درست تشخیص داده.

## 2. نمودار معیارهای عملکرد (Accuracy, Precision, Recall, F1-Score)
- Accuracy = 0.753 → 75.3% از تمام نمونه‌ها درست طبقه‌بندی شده‌اند.
- Precision = 0.618 → وقتی مدل می‌گوید "دارد"، 61.8% از این پیش‌بینی‌ها درست هستند.
- Recall = 0.778 → از تمام افراد واقعی با دیابت، مدل 77.8% را تشخیص داده.
- F1-Score = 0.689 → میانگین هارمونیک precision و recall — نشان‌دهنده تعادل بین دقت و حساسیت.

## 3. منحنی ROC و AUC

- AUC = 0.810 → نشان‌دهنده عملکرد خوب مدل در تمایز بین کلاس‌ها است.
- AUC > 0.8 = خوب
- AUC > 0.9 = عالی
 این عدد نشان می‌دهد مدل در مقایسه با یک حدس تصادفی (خط زرد)، عملکرد قابل قبولی دارد.

## 4. منحنی Precision-Recall و AP

AP (Average Precision) = 0.668 → معیاری برای ارزیابی مدل‌هایی که کلاس‌ها نامتعادل هستند (مثل اینجا که احتمالاً افراد بدون دیابت بیشتر هستند).
منحنی نشان می‌دهد که با افزایش recall، precision کاهش پیدا می‌کند — که طبیعی است.

 این نشان می‌دهد که مدل در شرایطی که کلاس مثبت کم‌تعداد است (بیماران)، عملکردش کمی ضعیف‌تر از AUC است. 

 نقاط قوت مدل:

Accuracy و AUC خوبی دارد (75% و 81%).
Recall بالا — یعنی بیشتر بیماران را تشخیص می‌دهد (مهم در پزشکی!).
 نقاط ضعف:

Precision پایین — مدل زیادی اشتباه مثبت می‌دهد (افراد سالم را بیمار می‌گیرد).
F1-Score متوسط — نشان‌دهنده عدم تعادل بین دقت و حساسیت.


## بخش 10 پلات اهمیت ویژگی ها - FEATURE IMPORTANCE
```
# ============================================================================
#  Feature Importance 
# ============================================================================

print("\n" + "="*60)
print("MODEL INTERPRETATION - FEATURE IMPORTANCE")
print("="*60)

plt.figure(figsize=(10, 6))

if best_svm.kernel == 'linear':
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': best_svm.coef_[0],
        'Abs_Coefficient': np.abs(best_svm.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)

    print(coefficients.drop('Abs_Coefficient', axis=1).to_string(index=False))
    plt.barh(coefficients['Feature'], coefficients['Coefficient'])
else:
    perm_importance = permutation_importance(
        best_svm, X_test_scaled, y_test, n_repeats=10, random_state=42
    )
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)

    print(feature_importance.to_string(index=False))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])

plt.tight_layout()
plt.show()
```
## output:

<img width="990" height="590" alt="image" src="https://github.com/user-attachments/assets/ded88789-ee5f-490f-bd06-cbe14cb4db2f" />


Glucose (گلوکز) → بالاترین اهمیت (~0.09)
✅ این نشان می‌دهد که سطح گلوکز خون قوی‌ترین پیش‌بینی‌کننده برای تشخیص دیابت است — که کاملاً منطقی و مطابق با دانش پزشکی است.

Age (سن) → اهمیت ~0.05
✅ سن عامل خطر مهمی در دیابت نوع 2 است — مدل این را درک کرده.

BMI (شاخص توده بدنی) → اهمیت ~0.035
✅ چاقی و اضافه وزن از عوامل اصلی دیابت هستند.

DiabetesPedigreeFunction → اهمیت ~0.03
✅ این ویژگی نشان‌دهنده سابقه خانوادگی دیابت است — مدل آن را مهم تشخیص داده.

Pregnancies (بارداری‌ها) → اهمیت ~0.03
✅ تعداد بارداری‌ها (به خصوص در زنان) با دیابت بارداری و بعد از آن مرتبط است.

Insulin (انسولین) → اهمیت ~0.01
⚠️ کمتر از انتظار — ممکن است داده‌های انسولین نوسان زیادی داشته باشند یا مقادیر زیادی NaN/صفر داشته باشند.

BloodPressure (فشار خون) → اهمیت بسیار کم
❗ ممکن است این ویژگی در داده‌ها همبستگی ضعیفی با هدف داشته باشد یا مقادیر آن نسبتاً ثابت باشند.

SkinThickness (ضخامت پوست) → کمترین اهمیت (~0.002)
❌ این ویژگی تقریباً هیچ نقشی در پیش‌بینی ندارد — می‌توانی آن را حذف کنی یا بررسی کنی که آیا داده‌هایش ناسالم هستند یا خیر.


## بخش 11 تحلیل آستانه - threshold

```
# ============================================================================
#   Threshold Analysis 
# ============================================================================

print("\n" + "="*60)
print("THRESHOLD ANALYSIS")
print("="*60)

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

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, lw=2, label='Precision', marker='o')
plt.plot(thresholds, recalls, lw=2, label='Recall', marker='s')
plt.plot(thresholds, f1_scores, lw=3, label='F1-Score', marker='^')

plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('SVM - Threshold Analysis', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
plt.axvline(x=optimal_threshold, linestyle='--', alpha=0.7)
plt.text(optimal_threshold + 0.02, 0.1,
         f'Optimal: {optimal_threshold:.2f}',
         fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
```
## output:

<img width="989" height="583" alt="image" src="https://github.com/user-attachments/assets/5b7a2086-5301-45f4-ba82-e0bf812efe28" />

```
============================================================
THRESHOLD ANALYSIS
============================================================

Threshold | Precision | Recall | F1-Score
----------------------------------------
0.3       | 0.575     | 0.852  | 0.687
0.4       | 0.618     | 0.778  | 0.689
0.5       | 0.633     | 0.574  | 0.602
0.6       | 0.657     | 0.426  | 0.517
0.7       | 0.714     | 0.278  | 0.400
```

### محورهای نمودار:

- محور افقی (X): مقدار آستانه (Threshold) از 0.30 تا 0.70
- محور عمودی (Y): مقدار معیارهای عملکرد (از 0.25 تا 0.85)
  
### خطوط:

- قرمز (Precision): با افزایش آستانه، precision افزایش پیدا می‌کند.
- زرد (Recall): با افزایش آستانه، recall کاهش می‌یابد.
- سبز (F1-Score): میانگین هارمونیک precision و recall — ابتدا ثابت است، سپس کاهش می‌یابد.
- 
### خط نقطه‌چین قرمز:

نشان‌دهنده آستانه بهینه (Optimal Threshold = 0.40) است.
در این نقطه، F1-Score بیشترین مقدار خود را دارد (حدود 0.69).


## یخش 11 وزیع احتمال پیش‌بینی شده توسط مدل SVM را بر اساس کلاس‌ها

```
# ============================================================================
#   Probability Distribution
# ============================================================================

print("\n" + "="*60)
print("PROBABILITY DISTRIBUTION ANALYSIS")
print("="*60)

plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
bins = np.linspace(0, 1, 21)
plt.hist(y_pred_proba[y_test == 0], bins=bins, alpha=0.7,
         label='No Diabetes', density=True)
plt.hist(y_pred_proba[y_test == 1], bins=bins, alpha=0.7,
         label='Has Diabetes', density=True)
plt.axvline(x=0.5, linestyle='--', linewidth=2, label='Threshold = 0.5')

plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Probability Distribution by Class', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Box Plot
plt.subplot(1, 2, 2)
data_to_plot = [y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]]
plt.boxplot(data_to_plot, patch_artist=True,
            labels=['No Diabetes', 'Has Diabetes'])

plt.ylabel('Predicted Probability')
plt.title('Probability Distribution - Box Plot', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## output:

<img width="1189" height="489" alt="image" src="https://github.com/user-attachments/assets/1196c6d4-136b-4ebc-8b13-ad43182845ab" />

 1. نمودار سمت چپ: Probability Distribution by Class (هیستوگرام چگالی)

### محورهای نمودار:

محور افقی (X): احتمال پیش‌بینی شده برای کلاس "Has Diabetes" (از 0 تا 1)
محور عمودی (Y): چگالی (Density) — نشان‌دهنده تعداد نمونه‌ها در هر بازه احتمال

### رنگ‌ها:

صورتی: نمونه‌های واقعی بدون دیابت (No Diabetes)
زرد: نمونه‌های واقعی با دیابت (Has Diabetes)
خط چین صورتی: آستانه 0.5 — مرزی که مدل بر اساس آن تصمیم می‌گیرد.

### برای کلاس Has Diabetes (زرد):
اکثر نمونه‌ها در احتمال بالاتر از 0.5 قرار دارند — یعنی مدل برای افراد واقعی با دیابت، احتمال بالایی پیش‌بینی می‌کند.
یک پیک قوی در حدود 0.8–0.9 دیده می‌شود — یعنی مدل خوبی در تشخیص بیماران است.
اما چند نمونه در زیر 0.5 هم هستند — این‌ها False Negatives هستند (بیمارانی که مدل اشتباهی سالم تشخیص داده).
### برای کلاس No Diabetes (صورتی):
بخش زیادی از نمونه‌ها در احتمال پایین‌تر از 0.5 هستند — یعنی مدل افراد سالم را درست تشخیص می‌دهد.
اما تعداد قابل توجهی از افراد سالم در احتمال بالاتر از 0.5 قرار دارند — این‌ها False Positives هستند (افراد سالمی که مدل اشتباهی بیمار تشخیص داده).
این مشکل منجر به Precision پایین می‌شود — چون وقتی مدل می‌گوید "دارد"، در واقع تعدادی از آن‌ها سالم هستند.

2. نمودار سمت راست: Probability Distribution - Box Plot

### محورهای نمودار:

محور افقی (X): کلاس‌ها (No Diabetes و Has Diabetes)
محور عمودی (Y): احتمال پیش‌بینی شده برای کلاس "Has Diabetes"

### عناصر جعبه:

خط وسط جعبه: میانه (Median) احتمال
جعبه: چارک اول تا سوم (IQR)
خطوط (Whiskers): دامنه داده‌ها (بدون آ웃لایر)
نقطه‌های جدا شده: آ웃لایرها (اگر وجود داشته باشند)

 ## برای Has Diabetes (جعبه صورتی راست):
میانه ≈ 0.5 — یعنی نیمی از بیماران احتمال بالاتر از 0.5 دارند.
چارک سوم (Q3) ≈ 0.75 — یعنی 75% از بیماران احتمال بالاتر از 0.75 دارند → خوب!
حداقل ≈ 0.1 — یعنی بعضی بیماران احتمال بسیار پایینی دریافت کرده‌اند → این‌ها False Negatives هستند.

### برای No Diabetes (جعبه صورتی چپ):
میانه ≈ 0.15 — یعنی نیمی از افراد سالم احتمال پایینی دارند → خوب.
چارک سوم (Q3) ≈ 0.4 — یعنی 75% از افراد سالم احتمال کمتر از 0.4 دارند → خوب.
حداکثر ≈ 0.8 — یعنی بعضی افراد سالم احتمال بسیار بالایی (حتی بالاتر از 0.8) دریافت کرده‌اند → این‌ها False Positives هستند.

این جعبه‌ها نشان می‌دهند که توزیع احتمال‌ها برای دو کلاس همپوشانی دارد — یعنی مدل نمی‌تواند به طور کامل این دو کلاس را از هم جدا کند. این هم‌پوشانی دلیل اصلی خطاهای طبقه‌بندی است.


🔹 نقاط قوت:
مدل برای بیشتر بیماران احتمال بالایی پیش‌بینی می‌کند (میانه 0.5 و Q3=0.75).
برای بیشتر افراد سالم احتمال پایینی پیش‌بینی می‌کند (میانه 0.15 و Q3=0.4).

🔹نقاط ضعف:
همپوشانی توزیع احتمال‌ها → منجر به False Positives و False Negatives می‌شود.
تعداد قابل توجهی از افراد سالم در احتمال بالاتر از 0.5 قرار دارند → Precision پایین.
تعدادی از بیماران در احتمال پایین‌تر از 0.5 قرار دارند → Recall کمی کاهش می‌یابد.

  نتیجه نهایی:
این نمودارها نشان می‌دهند که مدل SVM شما در تشخیص بیماران خوب عمل می‌کند، اما در تشخیص افراد سالم ضعیف‌تر است — چون توزیع احتمال‌ها برای دو کلاس همپوشانی دارد.
با تنظیم آستانه به 0.40، می‌تونی تعادل بهتری بین precision و recall ایجاد کنی — و این یک گام بسیار مهم برای بهبود عملکرد مدل در کاربردهای پزشکی است.

## بخش12 تحلیل خطا 

```
# ============================================================================
#    Error Analysis
# ============================================================================

print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

plt.figure(figsize=(8, 6))

error_types = ['False Positives', 'False Negatives',
               'True Positives', 'True Negatives']
error_counts = [fp, fn, tp, tn]

bars = plt.bar(error_types, error_counts,
               edgecolor='black', linewidth=1.5, alpha=0.8)

plt.title('SVM Error Analysis', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Count', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, error_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             str(count), ha='center', va='bottom',
             fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
```

## output:

<img width="790" height="589" alt="image" src="https://github.com/user-attachments/assets/d4e8c0e3-cb81-4e25-8f8b-f60e0a7a659c" />

## محورهای نمودار:

محور افقی (X): نوع خطا یا پیش‌بینی (False Positives, False Negatives, True Positives, True Negatives)
محور عمودی (Y): تعداد نمونه‌ها (Count)

مقادیر:

- False Positives (FP) = 26 → افراد سالمی که مدل اشتباهی "دارد دیابت" تشخیص داده.
- False Negatives (FN) = 12 → افراد بیماری که مدل اشتباهی "ندارد دیابت" تشخیص داده.
- True Positives (TP) = 42 → افراد بیماری که مدل درست "دارد دیابت" تشخیص داده.
- True Negatives (TN) = 74 → افراد سالمی که مدل درست "ندارد دیابت" تشخیص داده.

این اعداد دقیقاً همان اعدادی هستند که در ماتریس درهم‌ریختگی دیدیم — پس این نمودار فقط یک نمایش بصری از همان داده‌هاست، اما با تمرکز بر خطاهای مدل.


1. False Positives (FP = 26)

این‌ها اشتباه مثبت هستند — یعنی مدل افراد سالم را بیمار تشخیص داده.
در کاربردهای پزشکی، این خطا ممکنه منجر به:
اضطراب غیرضروری برای بیمار
هزینه‌های اضافی برای تست‌های تکمیلی
اشتغال غیرضروری منابع پزشکی

⚠️ این عدد نسبتاً بالاست — و علت اصلی Precision پایین (0.618) است.



2. False Negatives (FN = 12)

این‌ها اشتباه منفی هستند — یعنی مدل افراد بیمار را سالم تشخیص داده.
در کاربردهای پزشکی، این خطا خطرناک‌تر است — چون ممکنه باعث تأخیر در درمان شود.
✅ خوشبختانه این عدد کم است (تنها 12 مورد) — و نشان‌دهنده Recall بالا (0.778) است.



3. True Positives (TP = 42) و True Negatives (TN = 74)

این‌ها پیش‌بینی‌های درست هستند.
مدل در تشخیص افراد سالم (TN=74) عملکرد بهتری دارد — اما در تشخیص بیماران (TP=42) کمی ضعیف‌تر است.



## بخش 13 تقسیم کلاس ها
 ```
# ============================================================================
#   Class Distribution
# ============================================================================

print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)

plt.figure(figsize=(10, 4))

# Train vs Test
plt.subplot(1, 2, 1)
train_counts = y_train.value_counts()
test_counts = y_test.value_counts()

x = np.arange(2)
width = 0.35

plt.bar(x - width/2, train_counts.values, width,
        label='Train', edgecolor='black')
plt.bar(x + width/2, test_counts.values, width,
        label='Test', edgecolor='black')

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution: Train vs Test', fontweight='bold')
plt.xticks(x, ['No Diabetes', 'Has Diabetes'])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Pie chart
plt.subplot(1, 2, 2)
class_counts = df['Outcome'].value_counts()

plt.pie(class_counts,
        labels=['No Diabetes', 'Has Diabetes'],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0))

plt.title('Overall Class Distribution', fontweight='bold')

plt.tight_layout()
plt.show()
```
## output:

<img width="954" height="390" alt="image" src="https://github.com/user-attachments/assets/6284e3e8-0f45-4360-971e-07b89d05a396" />

### تقسیم داده‌ها (Train/Test Split)
نسبت کلاس‌ها در Train و Test تقریباً یکسان است — یعنی:
در Train: ~65% No Diabetes, ~35% Has Diabetes
در Test: ~67% No Diabetes, ~33% Has Diabetes
این یک تقسیم خوب است — چون مدل روی یک توزیع مشابه آموزش دیده و روی یک توزیع مشابه تست می‌شود. این باعث می‌شود که نتایج تست قابل اعتماد باشد.


قسیم داده‌ها به درستی انجام شده — یعنی مجموعه آموزش و آزمون توزیع مشابهی دارند.
این نشان‌دهنده روش صحیح تقسیم داده است — و باعث می‌شود که نتایج تست قابل اعتماد باشد.
با توجه به نامتعادل بودن داده‌ها، معیارهایی مثل F1-Score و AUC برای ارزیابی مدل مناسب‌تر از Accuracy هستند.


## بخش 14 خلاصه عملکرد نهایی مدل

```
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


```

## output:
```
==================================================
 FINAL PERFORMANCE SUMMARY
==================================================
     Metric    Value                  Interpretation
   Accuracy 0.753247             Overall correctness
  Precision 0.617647    Correct positive predictions
     Recall 0.777778   Ability to find all positives
   F1-Score 0.688525 Balance of Precision and Recall
Specificity 0.740000   Ability to identify negatives
        NPV 0.860465    Correct negative predictions
    ROC-AUC 0.809630  Overall classification ability
```


## 📊 جدول عملکرد نهایی — تحلیل عمیق

| متریک | مقدار | تفسیر و اهمیت در تشخیص دیابت |
|-------|--------|-----------------------------|
| **Accuracy** | `0.753` | 75.3% از تمام موارد درست پیش‌بینی شده‌اند.<br>⚠️ نباید تنها معیار قضاوت باشد — چون داده‌ها نامتعادل هستند (65% سالم). مدل می‌تونه فقط با گفتن *"همه سالم‌اند"* به Accuracy ≈ 0.65 برسه! |
| **Precision** | `0.618` | وقتی مدل می‌گوید *"دارد دیابت"*, فقط در **61.8% موارد** درست است.<br>🔴 یعنی **38.2% از موارد مثبت، اشتباه (False Positive)** هستند → افراد سالم را بیمار تشخیص داده. در عمل: آزمایش‌های اضافی، اضطراب، هزینه. |
| **Recall (Sensitivity)** | `0.778` | از همه بیماران واقعی، **77.8%** را شناسایی کرده.<br>🟢 این عدد **نسبتاً خوب** است — یعنی فقط **22.2% از بیماران (12 نفر)** از دست رفته‌اند (False Negative). در پزشکی، این مقدار معمولاً قابل قبول است، ولی هدف باید >0.85 یا حتی >0.9 باشد. |
| **F1-Score** | `0.689` | میانگین هارمونیک Precision و Recall.<br>🟡 عددی **متوسط** — نشان می‌دهد مدل در تعادل بین دقت و حساسیت **ضعف دارد**. برای مسائل حساس پزشکی، F1 > 0.75 ترجیح داده می‌شود. |
| **Specificity (TNR)** | `0.740` | از همه افراد سالم واقعی، **74%** را درست تشخیص داده.<br>🟠 یعنی **26% از افراد سالم (26 نفر)** اشتباه بیمار گرفته شده‌اند — همان FPها. |
| **NPV** | `0.860` | وقتی مدل می‌گوید *"ندارد دیابت"*, در **86% موارد** درست است.<br>🟢 این عدد **خوب** است — یعنی اگر پیام *"سالم هستید"* داد، احتمالاً واقعاً سالم است. برای آرامش بیمار مهم است. |
| **ROC-AUC** | `0.810` | توانایی کلی مدل در تمایز بین کلاس‌ها.<br>🟢 **بالاتر از 0.8 = خوب**. نشان می‌دهد مدل بهتر از حدس تصادفی عمل می‌کند. |

> 💡 **نکته کلیدی**:  
> در کاربردهای پزشکی، **کاهش False Negative (افزایش Recall)** اولویت بالاتری نسبت به کاهش False Positive دارد — چون از دست دادن یک بیمار (عدم تشخیص دیابت) پیامدهای جدی‌تری نسبت به تشخیص اشتباه یک فرد سالم دارد.

## بخش 15 تصویر عملکرد نهایی SVM روی کلاس ها با ایجاد مرز ها

```
# ============================================================================
#  CLEAN SVM DECISION BOUNDARY VISUALIZATION (PCA - FULL & FIXED)
# ============================================================================

from sklearn.decomposition import PCA

print("\n" + "="*60)
print("SVM DECISION BOUNDARY VISUALIZATION (PCA)")
print("="*60)

# 1. PCA to 2D (visualization only)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train_scaled)

# 2. Train SVM on PCA data

svm_vis = SVC(
    kernel=best_svm.kernel,
    C=best_svm.C,
    gamma=best_svm.gamma,
    class_weight='balanced'
)
svm_vis.fit(X_pca, y_train)


# 3. Create mesh grid
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

# 4. Decision function values
Z = svm_vis.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 5. Plot
plt.figure(figsize=(10, 8))

# Decision boundary & margins
plt.contour(
    xx, yy, Z,
    levels=[-1, 0, 1],
    linestyles=['--', '-', '--'],
    linewidths=[1.2, 2.5, 1.2],
    colors='black'
)

# Classes
plt.scatter(
    X_pca[y_train == 0, 0],
    X_pca[y_train == 0, 1],
    c='royalblue',
    edgecolor='k',
    s=50,
    alpha=0.7,
    label='No Diabetes'
)

plt.scatter(
    X_pca[y_train == 1, 0],
    X_pca[y_train == 1, 1],
    c='crimson',
    edgecolor='k',
    s=50,
    alpha=0.7,
    label='Has Diabetes'
)

# Support vectors
plt.scatter(
    svm_vis.support_vectors_[:, 0],
    svm_vis.support_vectors_[:, 1],
    s=120,
    facecolors='none',
    edgecolors='black',
    linewidths=2,
    label='Support Vectors'
)

# Labels
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.title(
    'SVM Decision Boundary with Support Vectors\n(PCA Projection - Visualization Only)',
    fontsize=14,
    fontweight='bold'
)

plt.legend(loc='upper left', fontsize=11, frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## output:
<img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/b23f2a39-5d79-4b5a-8d94-45bbd2346ce4" />


## 🧭 عناصر نمودار

| عنصر | توضیح |
|------|--------|
| **محور X**: `PCA Component 1` | مؤلفه اصلی اول — حاوی بیشترین واریانس داده‌ها |
| **محور Y**: `PCA Component 2` | مؤلفه اصلی دوم — حاوی دومین بیشترین واریانس |
| **نقاط آبی (No Diabetes)** | نمونه‌های واقعی بدون دیابت |
| **نقاط قرمز (Has Diabetes)** | نمونه‌های واقعی با دیابت |
| **دایره‌های مشکی با حاشیه سفید (Support Vectors)** | نمونه‌هایی که مرز تصمیم‌گیری روی آن‌ها متکی است — مهم‌ترین نقاط برای تشکیل مرز |
| **خط سیاه منحنی (Decision Boundary)** | مرزی که مدل SVM برای جدا کردن دو کلاس استفاده می‌کند |
| **خط چین سیاه (Margin)** | حاشیه (Margin) حول مرز تصمیم‌گیری — فاصله بین مرز و نزدیک‌ترین نقاط از هر کلاس |


تفسیر و تحلیل 

✅ نقاط قوت:
مرز تصمیم‌گیری غیرخطی است — یعنی مدل از یک کرنل غیرخطی (مثل RBF یا Polynomial) استفاده کرده — که برای داده‌های پیچیده مثل دیابت مناسب است.
بردارهای پشتیبان (Support Vectors) در اطراف مرز جمع شده‌اند — نشان‌دهنده این است که مدل روی نقاط مرزی تمرکز دارد.
تعداد زیادی از نقاط همپوشانی دارند — که با تحلیل قبلی (Probability Distribution) همخوانی دارد — یعنی مدل نمی‌تواند دو کلاس را کاملاً از هم جدا کند.
⚠️ نقاط ضعف / چالش‌ها:
همپوشانی زیاد بین دو کلاس — یعنی نقاط آبی و قرمز خیلی به هم نزدیک هستند — این همان دلیل خطاهای طبقه‌بندی (FP و FN) است.
مرز تصمیم‌گیری در مناطق چگال — یعنی مدل در مناطقی که داده‌ها زیاد هستند، مرز را می‌کشد — اما در مناطق کم‌چگال، ممکنه دقت کمتری داشته باشه.
تصویر فقط برای بصری‌سازی است — چون PCA فقط حدود 50-70% واریانس داده‌ها را حفظ می‌کند — بنابراین نمی‌توان از این نمودار برای تحلیل دقیق استفاده کرد.


 پیشنهادات :
1. بررسی Overfitting:
اگر مدل روی داده‌های آموزشی Accuracy بالایی دارد اما روی تست پایین است — ممکنه Overfit شده باشد.
می‌تونی از Validation Curve یا Learning Curve استفاده کنی تا این موضوع رو بررسی کنی.
2. استفاده از مدل‌های دیگر:
مدل‌هایی مثل Random Forest یا XGBoost ممکنه در فضای ویژگی‌ها بهتر عمل کنند — چون می‌تونن الگوهای غیرخطی را بهتر تشخیص بدن.
3. افزودن ویژگی‌های تعاملی:
مثلاً Glucose × BMI یا Age × Pregnancies — که ممکنه در فضای PCA بهتر جدا شوند.
