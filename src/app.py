from utils import db_connect
engine = db_connect()

# your code here
# ===== 1. IMPORTS & DATA LOADING =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load data
file_path = '/workspaces/BakuDShaggy-EDA-mas-proyecto-final/data/raw/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
data = pd.read_csv(file_path)

# ===== 2. INITIAL CHECKS =====
print("Initial Checks:")
print("-"*40)
print("Class Balance:\n", data['Diabetes_binary'].value_counts(normalize=True))
print("\nMissing Values:", data.isnull().sum().sum())
print("\nBinary Features:", [col for col in data.columns if sorted(data[col].unique()) in [[0,1], [0.0,1.0]]])

# ===== 3. FEATURE ENGINEERING =====
# Create interaction feature
data['BP_Chol_Combo'] = data['HighBP'] * data['HighChol']

# Split data
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===== 4. PHYSHEALTH BINNING (CRITICAL FIX) =====
bins = [-1, 7, 14, 30]  # Fix: Include 0 values with -1 lower bound
labels = ['0-7_days', '8-14_days', '15-30_days']

for df in [X_train, X_test]:
    df['PhysHlth_grouped'] = pd.cut(df['PhysHlth'], bins=bins, labels=labels, include_lowest=True)
    df.drop('PhysHlth', axis=1, inplace=True)

# ===== 5. BMI SCALING (NEW REQUIRED STEP) =====
scaler = StandardScaler()
X_train['BMI_scaled'] = scaler.fit_transform(X_train[['BMI']])
X_test['BMI_scaled'] = scaler.transform(X_test[['BMI']])

# Create full datasets
train_data = X_train.copy()
train_data['Diabetes_binary'] = y_train
test_data = X_test.copy()
test_data['Diabetes_binary'] = y_test

# ===== 6. CORRELATION MATRIX (FIXED) =====
# Use only numerical columns
numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = train_data[numerical_cols].corr()

plt.figure(figsize=(20,15))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap='coolwarm',
    center=0,
    annot=True,
    fmt=".2f",
    vmin=-0.4,
    vmax=0.4
)
plt.title("Numerical Features Correlation Matrix")
plt.show()

# ===== 7. ORDINAL ENCODING =====
ordinal_cols = ['GenHlth', 'Education', 'Income', 'Age']
ordinal_categories = [
    [1,2,3,4,5],  # GenHlth
    [1,2,3,4,5,6],  # Education
    [1,2,3,4,5,6,7,8],  # Income
    list(range(1,14))  # Age
]

encoder = OrdinalEncoder(
    categories=ordinal_categories,
    handle_unknown='use_encoded_value',
    unknown_value=-1
)
X_train_ordinal = encoder.fit_transform(X_train[ordinal_cols])
X_test_ordinal = encoder.transform(X_test[ordinal_cols])

# ===== 8. MODEL TRAINING =====
# Combine features correctly
X_combined_train = np.hstack([
    X_train_ordinal,
    X_train[['HighBP', 'HighChol', 'BMI_scaled', 'BP_Chol_Combo']].values
])

model = LogisticRegressionCV(
    Cs=20,
    penalty='elasticnet',
    solver='saga',
    l1_ratios=[0.2, 0.5, 0.8],
    cv=5,
    max_iter=1000,
    class_weight='balanced'
)
model.fit(X_combined_train, y_train)

# ===== 9. EVALUATION =====
y_pred = model.predict(X_combined_train)
print("\nModel Performance Report:")
print("-"*40)
print(classification_report(y_train, y_pred))