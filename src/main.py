# Loan Approval Prediction: A Binary Classification Project
# Converted from Jupyter Notebook to main.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For preprocessing and modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support

# For handling imbalance (bonus)
from imblearn.over_sampling import SMOTE

# Set style for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading and Initial Exploration
# Load the dataset
df = pd.read_csv('../data/loan_approval_dataset.csv')

print("Dataset Shape:", df.shape)
print(df.head())

print(df.info())

print(df.describe())

# 2. Exploratory Data Analysis (EDA)
# Check missing values
print(df.isna().sum())

# Drop Loan_ID (not useful for modeling)
df = df.drop('loan_id', axis=1)

# removing the extra spaces
df.columns = df.columns.str.strip()

# Strip spaces from values
df['loan_status'] = df['loan_status'].str.strip()

# Encode Approved as 1, Rejected as 0 (for example)
df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

# Visualize target distribution (class imbalance)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='loan_status')
plt.title('Distribution of Loan Status (Imbalanced)')
plt.xlabel('Loan Status')
plt.ylabel('Count')

# Replace tick labels
plt.xticks(ticks=[0, 1], labels=['Rejected', 'Approved'])

plt.show()

# Key categorical features vs. target
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
categorical_cols = ['education', 'self_employed']
for i, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, hue='loan_status', ax=axes[i])
    axes[i].set_title(f'{col.title()} vs Loan Status')
plt.tight_layout()
plt.show()

# Numerical features distribution
numerical_cols = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                  'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
df[numerical_cols].hist(bins=20, figsize=(16, 12))
plt.suptitle('Distribution of Numerical Features')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols + ['loan_status']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 3. Data Preprocessing
# Encode categorical variables
categorical_cols = ['education', 'self_employed', 'loan_status']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 4. Feature Preparation
# Features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Scale numerical features (important for Logistic Regression)
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# 5. Handling Imbalanced Data
# Apply SMOTE (bonus)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("SMOTE Train Class Distribution:\n", pd.Series(y_train_smote).value_counts(normalize=True))

# 6. Model Training and Evaluation
# Function to evaluate model
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{model_name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")

    # Get metrics for class 0 (Rejected)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    prec_rejected = precision[0]
    rec_rejected = recall[0]
    f1_rejected = f1[0]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return y_pred, y_pred_proba, prec_rejected, rec_rejected, f1_rejected

# Logistic Regression (original data)
lr = LogisticRegression(random_state=42, max_iter=1000)
lr_pred, lr_proba, lr_prec, lr_rec, lr_f1 = evaluate_model(lr, X_train, y_train, X_test, y_test, "Logistic Regression (Original)")

# Decision Tree (original data)
dt = DecisionTreeClassifier(random_state=42, max_depth=5)  # Limit depth to avoid overfitting
dt_pred, dt_proba, dt_prec, dt_rec, dt_f1 = evaluate_model(dt, X_train, y_train, X_test, y_test, "Decision Tree (Original)")

# Logistic Regression with SMOTE
lr_smote = LogisticRegression(random_state=42, max_iter=1000)
lr_smote_pred, lr_smote_proba, lr_smote_prec, lr_smote_rec, lr_smote_f1 = evaluate_model(lr_smote, X_train_smote, y_train_smote, X_test, y_test, "Logistic Regression (SMOTE)")

# Decision Tree with SMOTE
dt_smote = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_smote_pred, dt_smote_proba, dt_smote_prec, dt_smote_rec, dt_smote_f1 = evaluate_model(dt_smote, X_train_smote, y_train_smote, X_test, y_test, "Decision Tree (SMOTE)")

# Compare models
results = {
    'Model': ['LR Original', 'DT Original', 'LR SMOTE', 'DT SMOTE'],
    'Precision (Rejected)': [lr_prec, dt_prec, lr_smote_prec, dt_smote_prec],
    'Recall (Rejected)': [lr_rec, dt_rec, lr_smote_rec, dt_smote_rec],
    'F1 (Rejected)': [lr_f1, dt_f1, lr_smote_f1, dt_smote_f1],
    'ROC-AUC': [roc_auc_score(y_test, lr_proba), roc_auc_score(y_test, dt_proba), roc_auc_score(y_test, lr_smote_proba), roc_auc_score(y_test, dt_smote_proba)]
}

results_df = pd.DataFrame(results)
print("Model Comparison:")
print(results_df.round(3))

# Feature Importance for Decision Tree (SMOTE)
feat_importance = pd.Series(dt_smote.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_importance.plot(kind='barh', figsize=(10, 6))
plt.title('Feature Importance - Decision Tree (SMOTE)')
plt.show()

# Plot ROC curves
plt.figure(figsize=(8, 6))
probabilities = [
    ('LR Original', lr_proba),
    ('DT Original', dt_proba),
    ('LR SMOTE', lr_smote_proba),
    ('DT SMOTE', dt_smote_proba)
]
for name, proba in probabilities:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, proba):.3f})')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()