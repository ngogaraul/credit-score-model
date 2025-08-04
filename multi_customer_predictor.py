
import pandas as pd
import numpy as np
import joblib

def load_model_and_predict_for_multiple_customers(data_path, model_path, scaler_path, output_path):
    # Load trained model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load dataset with many customers
    df = pd.read_excel(data_path)

    # Step 1: Fill missing values
    df.fillna(0, inplace=True)

    # Step 2: Feature Engineering
    df['ArrearsRatio'] = (df['Principal Arrears'] + df['Interest Arrears']) / (df['Outstanding'] + 1)
    df['SavingsToSalary'] = (df['Compulsory saving'] + df['Voluntary Saving']) / (df['Salary'] + 1)
    df['InstallmentRatio'] = df['Payment plan'] / (df['Salary'] + 1)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # Step 3: Aggregate per customer
    agg_df = df.groupby("Account").agg({
        'Outstanding': 'mean',
        'Principal Arrears': 'sum',
        'Interest Arrears': 'sum',
        'Payment plan': 'mean',
        'DaysInArrears': 'mean',
        'Duration': 'mean',
        'Remaining Period': 'mean',
        'Periodicity': 'mean',
        'Compulsory saving': 'sum',
        'Voluntary Saving': 'sum',
        'Collateral': 'mean',
        'Salary': 'mean',
        'ArrearsRatio': 'mean',
        'SavingsToSalary': 'mean',
        'InstallmentRatio': 'mean'
    }).reset_index()

    # Step 4: Scale features
    features = [
        'Outstanding', 'Principal Arrears', 'Interest Arrears', 'Payment plan',
        'DaysInArrears', 'Duration', 'Remaining Period', 'Periodicity',
        'Compulsory saving', 'Voluntary Saving', 'Collateral', 'Salary',
        'ArrearsRatio', 'SavingsToSalary', 'InstallmentRatio'
    ]
    X = agg_df[features]
    X_scaled = scaler.transform(X)

    # Step 5: Predict
    predictions = model.predict(X_scaled)

    # Step 6: Map to FICO band
    def score_to_band(score):
        if score < 580: return "Poor"
        elif score < 670: return "Fair"
        elif score < 740: return "Good"
        elif score < 800: return "Very Good"
        else: return "Exceptional"

    fico_bands = [score_to_band(score) for score in predictions]

    # Step 7: Save results
    agg_df["Predicted Credit Score"] = np.round(predictions, 2)
    agg_df["FICO Band"] = fico_bands
    #agg_df["Comment"] = "Predicted using full customer history"

    agg_df.to_excel(output_path, index=False)
    return agg_df[['Account', 'Predicted Credit Score', 'FICO Band']]#removed "Comment" for simplicity
