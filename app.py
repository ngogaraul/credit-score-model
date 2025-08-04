
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_excel("C:/Users/bngog/Desktop/intern/synthetic_credit_data_10000_rows.xlsx")

# Drop completely empty rows (if any)
df.dropna(how='all', inplace=True)

# Step 1: Simulate Credit Score
def simulate_score(row):
    score = 850
    score -= np.nan_to_num(row['DaysInArrears']) * 2
    score -= np.nan_to_num(row['Principal Arrears'] + row['Interest Arrears']) * 0.0005
    score -= (np.nan_to_num(row['Outstanding']) / 1e5) * 1.5
    score += np.nan_to_num(row['Salary']) / 1e5
    score += np.nan_to_num(row['Compulsory saving']) / 1e6
    score += np.nan_to_num(row['Collateral']) / 1e5
    score = np.clip(score, 300, 850)
    return round(score)

df['SimulatedScore'] = df.apply(simulate_score, axis=1)

# Step 2: Feature Engineering 
# Fill NaNs
df.fillna(0, inplace=True)

# Create new engineered features
df['ArrearsRatio'] = (df['Principal Arrears'] + df['Interest Arrears']) / df['Outstanding']
df['SavingsToSalary'] = (df['Compulsory saving'] + df['Voluntary Saving']) / (df['Salary'] + 1)
df['InstallmentRatio'] = df['Payment plan'] / (df['Salary'] + 1)

# Replace infinities and fill NaNs
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

#  Step 3: Prepare Features & Target 
features = [
    'Outstanding', 'Principal Arrears', 'Interest Arrears', 'Payment plan',
    'DaysInArrears', 'Duration', 'Remaining Period', 'Periodicity',
    'Compulsory saving', 'Voluntary Saving', 'Collateral', 'Salary',
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
    "Comment": "Model trained using this person's 10,000 records only"
}

print(final_result)
# Step 9: Save Model and Scaler


joblib.dump(model, "C:/Users/bngog/Desktop/intern/css pj/credit_score_model.pkl")
joblib.dump(scaler, "C:/Users/bngog/Desktop/intern/css pj/credit_score_scaler.pkl")

print(" Model and scaler saved successfully.")
