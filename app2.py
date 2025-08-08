import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import scipy.stats as stats
import joblib

# 1) Load & initial clean
df = pd.read_excel("C:/Users/bngog/Desktop/intern/css pj/large_synthetic_test_dataset.xlsx")
df = df[df['Account'].notna() & (df['Account'] != '')].dropna(how='all')

# 2) Simulate target
def simulate_score(r):
    RANGE, BASE = 550.0, 300.0
    days = np.nan_to_num(r.get('DaysInArrears', 0))
    part1 = (days / 30) * (0.30 * RANGE)
    early = max((30 - days) / 30, 0)
    part2 = early * (0.10 * RANGE)

    pia = np.nan_to_num(r.get('Principal Arrears', 0) + r.get('Interest Arrears', 0))
    part3 = (pia / (pia + 1)) * (0.15 * RANGE)

    out = np.nan_to_num(r.get('Outstanding', 0))
    part4 = (out / (out + 1e3)) * (0.15 * RANGE)

    sal = np.nan_to_num(r.get('Salary', 0))
    cs  = np.nan_to_num(r.get('Compulsory saving', 0))
    vs  = np.nan_to_num(r.get('voluntary saving', 0))
    part5 = (sal / (sal + 1e3)) * (0.10 * RANGE)
    part6 = (cs  / (cs  + 1e3)) * (0.10 * RANGE)
    part7 = (vs  / (vs  + 1e3)) * (0.05 * RANGE)

    col = np.nan_to_num(r.get('Collateral', 0))
    part8 = (col / (col + 1e3)) * (0.05 * RANGE)

    score = 850 - part1 + part2 - part3 - part4 + part5 + part6 + part7 + part8
    return np.clip(score, BASE, BASE + RANGE)

df['SimulatedScore'] = df.apply(simulate_score, axis=1)
df['SimulatedScore'] = (
    df['SimulatedScore']
      .replace([np.inf, -np.inf], np.nan)
      .fillna(300.0)
      .round()
      .astype(int)
)

# 3) Imputation + missing flags
num_impute = ['Salary','Outstanding','Principal Arrears','Interest Arrears']
for col in num_impute:
    df[f"{col}_missing"] = df[col].isna().astype(int)
imp = SimpleImputer(strategy='median')
df[num_impute] = imp.fit_transform(df[num_impute])

# 4) Winsorize continuous vars
for c in ['DaysInArrears','Compulsory saving','voluntary saving','Collateral']:
    if c in df.columns:
        lo, hi = df[c].quantile([0.01,0.99])
        df[c] = df[c].clip(lo, hi)

# 5) Core ratios
df['ArrearsRatio']    = (df['Principal Arrears']+df['Interest Arrears'])/(df['Outstanding']+1)
df['SavingsToSalary'] = (df['Compulsory saving']+df['voluntary saving'])/(df['Salary']+1)
df['InstallmentRatio'] = df.get('Payment plan',0)/(df['Salary']+1)

# 6) TotalArrearsCount
if 'TotalArrearsCount' not in df.columns:
    if 'DaysInArrears' in df.columns:
        df['InArrears'] = (df['DaysInArrears'] > 0).astype(int)
        counts = df.groupby('Account')['InArrears'].sum().rename('TotalArrearsCount')
        df = df.join(counts, on='Account')
    else:
        df['TotalArrearsCount'] = 0

# 7) DaysSinceMaturity
if 'LastLoanMaturityDate' not in df.columns:
    # derive from Start date + Duration (days)
    if 'Start date' in df.columns and 'Duration' in df.columns:
        df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')
        # assume Duration is in days
        df['LastLoanMaturityDate'] = df['Start date'] + pd.to_timedelta(df['Duration'], unit='D')
    else:
        df['LastLoanMaturityDate'] = pd.NaT

df['LastLoanMaturityDate'] = pd.to_datetime(df['LastLoanMaturityDate'], errors='coerce')
df['DaysSinceMaturity'] = (
    pd.Timestamp.today() - df['LastLoanMaturityDate']
).dt.days.fillna(0).astype(int)

# 8) LoanAgeDays (optional but similar)
if 'LoanAgeDays' not in df.columns:
    if 'Start date' in df.columns:
        df['LoanAgeDays'] = (
            pd.Timestamp.today() - pd.to_datetime(df['Start date'], errors='coerce')
        ).dt.days.fillna(0).astype(int)
    else:
        df['LoanAgeDays'] = 0

# 9) Final cleanup
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# 10) Feature list
features = [
    'DaysInArrears','Principal Arrears','Interest Arrears','Outstanding','Salary',
    'Compulsory saving','voluntary saving','Collateral','Payment plan',
    'Duration','Remaining Period','Periodicity',
    'Salary_missing','Outstanding_missing','Principal Arrears_missing','Interest Arrears_missing',
    'ArrearsRatio','SavingsToSalary','InstallmentRatio',
    'TotalArrearsCount','DaysSinceMaturity','LoanAgeDays'
]
features = [f for f in features if f in df.columns]
X = df[features]
y = df['SimulatedScore']

# 11) Correlation check
corrs = X.assign(Target=y).corr()['Target'].drop('Target')
print("Correlations with target:")
print(corrs.sort_values(key=lambda s: s.abs(), ascending=False))

# 12) Split & RF pipeline
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42))
])
param_dist = {
    'rf__n_estimators': stats.randint(100,300),
    'rf__max_depth':    stats.randint(5,15),
    'rf__min_samples_split': stats.randint(2,8),
    'rf__min_samples_leaf':  stats.randint(1,4)
}
rs = RandomizedSearchCV(
    pipeline, param_dist, n_iter=20,
    scoring='neg_root_mean_squared_error', cv=5,
    n_jobs=-1, random_state=42
)
rs.fit(X_train, y_train)
model = rs.best_estimator_

# 13) Evaluate
y_pred = model.predict(X_test)
print("\nRMSE:", mean_squared_error(y_test,y_pred))
print("R²:  ", r2_score(y_test,y_pred))
print("MAE: ", mean_absolute_error(y_test,y_pred))

# 14) Feature importances
fi = pd.Series(model.named_steps['rf'].feature_importances_, index=features)
print("\nTop features:\n", fi.sort_values(ascending=False).head(10))

# 15) Save augmented excel and model
df.to_excel("C:/Users/bngog/Desktop/intern/css pj/augmented_dataset.xlsx", index=False)
joblib.dump(model, "C:/Users/bngog/Desktop/intern/css pj/credit_score_rf_model.pkl")

print("\nSaved augmented_dataset.xlsx and trained model.")
