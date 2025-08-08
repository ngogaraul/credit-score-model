import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model_path = "C:/Users/bngog/Desktop/intern/css pj/credit_score_model.pkl"
scaler_path = "C:/Users/bngog/Desktop/intern/css pj/credit_score_scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def preprocess_customer_data(df):
    # Fill missing values
    df.fillna(0, inplace=True)

    # Feature engineering
    df['ArrearsRatio'] = (df['Principal Arrears'] + df['Interest Arrears']) / (df['Outstanding'] + 1)
    df['SavingsToSalary'] = (df['Compulsory saving'] + df['Voluntary Saving']) / (df['Salary'] + 1)
    df['InstallmentRatio'] = df['Payment plan'] / (df['Salary'] + 1)

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    # List of features used during training
    features = [
        'Outstanding', 'Principal Arrears', 'Interest Arrears', 'Payment plan',
        'DaysInArrears', 'Duration', 'Remaining Period', 'Periodicity',
        'Compulsory saving', 'Voluntary Saving', 'Collateral', 'Salary',
        'ArrearsRatio', 'SavingsToSalary', 'InstallmentRatio'
    ]
    
    X = df[features]
    X_scaled = scaler.transform(X)
    return X_scaled, df['Account'].iloc[0]

def predict_credit_score(df):
    X_scaled, account = preprocess_customer_data(df)
    predictions = model.predict(X_scaled)
    final_score = round(np.mean(predictions), 2)

    def score_to_band(score):
        if score < 580: return "Poor"
        elif score < 670: return "Fair"
        elif score < 740: return "Good"
        elif score < 800: return "Very Good"
        else: return "Exceptional"

    fico_band = score_to_band(final_score)

    return {
        "Account": int(account),
        "Predicted Credit Score": final_score,
        "FICO Band": fico_band,
        #"Comment": "Predicted using saved model and scaler"
    }

# Example use:
new_df = pd.read_excel("C:/Users/bngog/Desktop/intern/css pj/dataset.xlsx") 
result = predict_credit_score(new_df)
print(result)
