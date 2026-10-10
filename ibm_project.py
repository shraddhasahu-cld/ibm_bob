# =============================================================
#  Salary Prediction Project
#  Uses a synthetic dataset to train and evaluate a regression
#  model that predicts annual salary based on employee features.
# =============================================================

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 1.  Load dataset (salary_dataset.csv lives in the same folder)
# ------------------------------------------------------------------

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salary_dataset.csv")
df = pd.read_csv(DATASET_PATH)
N = len(df)

# ------------------------------------------------------------------
# 2.  Explore the data
# ------------------------------------------------------------------

print("=" * 60)
print("           SALARY PREDICTION PROJECT")
print("=" * 60)

print(f"   Loaded from: salary_dataset.csv")
print("\n[*] Dataset Overview")
print(f"   Shape      : {df.shape}")
print(f"   Features   : {list(df.columns[:-1])}")
print(f"   Target     : salary")

print("\n[+] Salary Statistics (USD)")
print(df["salary"].describe().apply(lambda x: f"   {x:,.0f}").to_string())

print("\n[>] Sample Records")
print(df.head(5).to_string(index=False))

# ------------------------------------------------------------------
# 3.  Preprocessing
# ------------------------------------------------------------------

print("\n\n[~] Preprocessing")

# Encode categorical columns
le = LabelEncoder()
cat_cols = ["education", "job_title", "department", "gender"]
df_encoded = df.copy()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df[col])
    print(f"   Encoded '{col}' -> integer labels")

# Feature / target split
X = df_encoded.drop(columns=["salary"])
y = df_encoded["salary"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n   Train samples : {len(X_train)}")
print(f"   Test  samples : {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4.  Model training & evaluation
# ------------------------------------------------------------------

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    """Train, predict, and return evaluation metrics."""
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "predictions": preds}


models = {
    "Linear Regression":        LinearRegression(),
    "Random Forest":            RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":        GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print("\n\n[AI] Model Training & Evaluation")
print("-" * 60)

results = []
best_model_name = None
best_r2 = -np.inf
best_preds = None

for name, model in models.items():
    # Linear Regression benefits from scaling; tree models don't need it
    if "Linear" in name:
        res = evaluate(name, model, X_train_scaled, X_test_scaled, y_train, y_test)
    else:
        res = evaluate(name, model, X_train, X_test, y_train, y_test)

    results.append(res)
    print(f"\n   {name}")
    print(f"     MAE  : ${res['MAE']:>10,.0f}")
    print(f"     RMSE : ${res['RMSE']:>10,.0f}")
    print(f"     R2   : {res['R2']:.4f}")

    if res["R2"] > best_r2:
        best_r2 = res["R2"]
        best_model_name = name
        best_preds = res["predictions"]

print("\n" + "-" * 60)
print(f"[OK] Best Model : {best_model_name}  (R2 = {best_r2:.4f})")

# ------------------------------------------------------------------
# 5.  Feature importance (tree-based best model)
# ------------------------------------------------------------------

if best_model_name in ("Random Forest", "Gradient Boosting"):
    best_model_obj = models[best_model_name]
    importances = best_model_obj.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    print("\n\n[i] Feature Importances (Best Model)")
    print("-" * 40)
    for feat, imp in feat_imp.items():
        bar = "#" * int(imp * 50)
        print(f"   {feat:<20s} {imp:.4f}  {bar}")

# ------------------------------------------------------------------
# 6.  Sample Predictions vs Actual
# ------------------------------------------------------------------

print("\n\n[?] Sample Predictions vs Actual (first 10 test records)")
print("-" * 55)
print(f"   {'Actual':>12}   {'Predicted':>12}   {'Difference':>12}")
print("-" * 55)
for actual, pred in zip(list(y_test)[:10], list(best_preds)[:10]):
    diff = actual - pred
    print(f"   ${actual:>10,}   ${pred:>10,.0f}   ${diff:>+10,.0f}")

# ------------------------------------------------------------------
# 7.  Predict salary for a new employee
# ------------------------------------------------------------------

print("\n\n[+] Predict Salary for a New Employee")
print("-" * 45)

new_employee = {
    "age": 35,
    "years_experience": 10,
    "education": "Master's",
    "job_title": "Engineer",
    "department": "Engineering",
    "gender": "Female",
    "performance_score": 4,
    "num_projects": 8,
}

print("   Profile:")
for k, v in new_employee.items():
    print(f"     {k:<20s}: {v}")

# Encode the new record using the same label encoders
# Build fresh label encoders per column (consistent mapping)
le_map = {}
for col in cat_cols:
    le_tmp = LabelEncoder()
    le_tmp.fit(df[col])
    le_map[col] = le_tmp

new_row = {k: v for k, v in new_employee.items()}
for col in cat_cols:
    new_row[col] = le_map[col].transform([new_row[col]])[0]

new_df = pd.DataFrame([new_row])

# Use the best model for the prediction
if best_model_name == "Linear Regression":
    new_scaled = scaler.transform(new_df)
    predicted_salary = models[best_model_name].predict(new_scaled)[0]
else:
    predicted_salary = models[best_model_name].predict(new_df)[0]

print(f"\n   Predicted Annual Salary: ${predicted_salary:,.0f}")

# ------------------------------------------------------------------
# 8.  Summary
# ------------------------------------------------------------------

print("\n\n" + "=" * 60)
print("                     SUMMARY")
print("=" * 60)
print(f"  Dataset size       : {N} records")
print(f"  Features used      : {X.shape[1]}")
print(f"  Models evaluated   : {len(models)}")
print(f"  Best model         : {best_model_name}")
print(f"  Best R2 score      : {best_r2:.4f}")
print(f"  Best MAE           : ${[r for r in results if r['Model']==best_model_name][0]['MAE']:,.0f}")
print("=" * 60)
print("\nProject complete!\n")
