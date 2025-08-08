
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Load dataset
df = pd.read_excel("C:/Users/bngog/Desktop/intern/css pj/large_synthetic_test_dataset.xlsx")
# Ensure 'Account' column is not empty or NaN
df=df[df['Account'].notna()]
df=df[df['Account'] != '']  

# Drop completely empty rows (if any)
df.dropna(how='all', inplace=True)


# --- compute once, as floats ---
DAYS_MAX = max(df['DaysInArrears'].quantile(0.95), 1.0)
PIA_MAX  = max((df['Principal Arrears'] + df['Interest Arrears']).quantile(0.95), 1.0)
OUT_MAX  = max(df['Outstanding'].quantile(0.95), 1.0)
SAL_MAX  = max(df['Salary'].quantile(0.95), 1.0)
CS_MAX   = max(df['Compulsory saving'].quantile(0.95), 1.0)
COL_MAX  = max(df['Collateral'].quantile(0.95), 1.0) * 0.8
VS_MAX   = max(df.get('voluntary saving', pd.Series(0, index=df.index)).quantile(0.95), 1.0)



# Step 1: Simulate Credit Score
def simulate_score(row, DAYS_MAX, PIA_MAX, OUT_MAX, SAL_MAX, CS_MAX, COL_MAX, VS_MAX):
    RANGE = 550.0
    score = 850.0

    days = np.nan_to_num(row['DaysInArrears'])
    score -= (days / DAYS_MAX +1) * (0.30 * RANGE)
    early = max(DAYS_MAX - days, 0.0)
    score += (early / DAYS_MAX+1) * (0.10 * RANGE)

    pia = np.nan_to_num(row['Principal Arrears'] + row['Interest Arrears'])
    score -= (pia / PIA_MAX+1) * (0.15 * RANGE)

    out = np.nan_to_num(row['Outstanding'])
    score -= (out / OUT_MAX+1 ) * (0.15 * RANGE)

    sal = np.nan_to_num(row['Salary'])
    score += (sal / SAL_MAX +1 ) * (0.10 * RANGE)

    cs  = np.nan_to_num(row['Compulsory saving'])
    score += (cs / CS_MAX + 1 ) * (0.10 * RANGE)

    col = np.nan_to_num(row['Collateral'])
    score += (col / COL_MAX +1 ) * (0.05 * RANGE)

    vs  = np.nan_to_num(row.get('voluntary saving', 0.0))
    score += (vs / VS_MAX +1) * (0.05 * RANGE)
#scale the score in daysInArrers according to the class column
#if customer creditline is in class 5 ,whole account is affected
# if customer creditline is in class 1,2,3, only the creditline is affected with certain percentage
#script base on daysInArrears column
#classes are 1,2,3,4,5
#class 1 -->4%
#class 2 -->3%
#class 3 -->2%
#class 4 -->1%
#class 5 -->0%

    return round(np.clip(score, 300.0, 850.0))


df['SimulatedScore'] = df.apply(simulate_score, axis=1,args=(DAYS_MAX, PIA_MAX, OUT_MAX, SAL_MAX, CS_MAX, COL_MAX, VS_MAX))

# Step 2: Feature Engineering 
# Fill NaNs
df.fillna(0, inplace=True)

# Create new engineered features
df['ArrearsRatio'] = (df['Principal Arrears'] + df['Interest Arrears']) / df['Outstanding']
df['SavingsToSalary'] = (df['Compulsory saving'] + df['voluntary saving']) / (df['Salary'] + 1)
df['InstallmentRatio'] = df['Payment plan'] / (df['Salary'] + 1)

# Replace infinities and fill NaNs
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

#  Step 3: Prepare Features & Target 
features = [
    'Outstanding', 'Principal Arrears', 'Interest Arrears', 'Payment plan',
    'DaysInArrears', 'Duration', 'Remaining Period', 'Periodicity',
    'Compulsory saving', 'voluntary saving', 'Collateral', 'Salary',
    'ArrearsRatio', 'SavingsToSalary', 'InstallmentRatio'
]
target = 'SimulatedScore'

X = df[features]
y = df[target]

#  Step 4: Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Step 5: Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#  Step 6: Train Model 
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#predict on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate Model
def test_performance(model, scaler, X_train, X_test, y_train, y_test, X_full, y_full):
    # 1) Evaluate on hold-out test set
    y_pred_test = model.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_pred_test)
    mae_test  = mean_absolute_error(y_test, y_pred_test)
    r2_test   = r2_score(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

    print("=== Test Set Performance ===")
    print(f"RMSE:  {rmse_test:.2f}")
    print(f"MAE:   {mae_test:.2f}")
    print(f"R²:    {r2_test:.2f}")
    print(f"MAPE:  {mape_test:.1f}%\n")

    # 2) 5-fold CV on full dataset
    cv_scores = cross_val_score(
        model, 
        scaler.transform(X_full), 
        y_full, 
        cv=5, 
        scoring='neg_root_mean_squared_error'
    )
    print("5-Fold CV ")
    print(f"RMSE (mean ± std): {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}\n")

    # 3) Feature importances
    importances = pd.Series(model.feature_importances_, index=X_full.columns)
    importances = importances.sort_values(ascending=False)
    print("Top 10 Feature Importances")
    print(importances.head(10).to_frame("importance"))


test_performance(model, scaler, X_train, X_test, y_train, y_test, X, y)


# Step 7: Predict and Aggregate 
predictions = model.predict(X_scaled)
#print(predictions  )
final_score = round(np.mean(predictions), 2)
#print(final_score)

def score_to_band(score):
    if score < 580: return "Poor"
    elif score < 670: return "Fair"
    elif score < 740: return "Good"
    elif score < 800: return "Very Good"
    else: return "Exceptional"

fico_band = score_to_band(final_score)

# Step 8: Output Result 
final_result = {
    "Account": int(df['Account'].dropna().iloc[0]),
    "Predicted Credit Score": final_score,
    "FICO Band": fico_band,

}

print(final_result)
# Step 9: Save Model and Scaler


joblib.dump(model, "C:/Users/bngog/Desktop/intern/css pj/credit_score_model.pkl")
joblib.dump(scaler, "C:/Users/bngog/Desktop/intern/css pj/credit_score_scaler.pkl")

print(" Model and scaler saved successfully.")
