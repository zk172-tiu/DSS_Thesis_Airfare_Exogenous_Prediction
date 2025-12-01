import pandas as pd
import numpy as np
import math
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("merged_df.csv", index_col=0)
df = df.drop_duplicates().reset_index(drop=True)
df['Date_of_journey'] = pd.to_datetime(df['Date_of_journey'])
df['pax_load_factor_percent'] = df['pax_load_factor_percent'].str.rstrip('%').astype(float)

df['journey_month'] = df['Date_of_journey'].dt.month
df['journey_weekofyear'] = df['Date_of_journey'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['Date_of_journey'].dt.dayofweek.isin([5, 6]).astype(int)

df['route'] = df['Source'] + "_" + df['Destination']
df['airline_route'] = df['Airline'] + "_" + df['Source'] + "_" + df['Destination']

df['airline_freq'] = df.groupby('Airline')['Airline'].transform('count')
df['route_freq'] = df.groupby('route')['route'].transform('count')

df['is_nonstop'] = (df['Total_stops'] == 'non-stop').astype(int)
df['num_stops'] = df['Total_stops'].apply(lambda x: 0 if x == 'non-stop' else (1 if '1 stop' in x else 2))

df['is_red_eye'] = ((df['Departure'] == 'After 10 PM') & (df['Arrival'] == 'Before 6 AM')).astype(int)

df['duration_bin'] = pd.cut(
    df['Duration_in_hours'],
    bins=[0, 3, 6, 12, 20, np.inf],
    labels=['short', 'medium', 'long', 'extra_long', 'ultra_long']
)

df['is_premium'] = df['Class'].isin(['Business', 'First', 'Premium Economy']).astype(int)

df['Fare_log'] = np.log1p(df['Fare'])

le = LabelEncoder()
for col in ['Flight_code', 'airline_route']:
    df[col] = le.fit_transform(df[col].astype(str))
    df[col] = df[col].astype('category')

cat_cols = [
    'Journey_day', 'Airline', 'Class', 'Source', 'Departure',
    'Arrival', 'Destination', 'Total_stops',
    'Flight_code', 'route', 'airline_route', 'duration_bin', 'Month'
]
for col in cat_cols:
    df[col] = df[col].astype('category')

X = df.drop(['Fare', 'Fare_log', 'Date_of_journey'], axis=1)
y = df['Fare_log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_params = {
    'n_estimators': 595,
    'max_depth': 11,
    'learning_rate': 0.06304206171990798,
    'subsample': 0.9967142394753343,
    'colsample_bytree': 0.9959073051791871,
    'reg_alpha': 0.5526463609665203,
    'reg_lambda': 2.646133180471669,
    'min_child_weight': 4,
    'n_jobs': -1,
    'random_state': 42,
    'enable_categorical': True,
    'verbosity': 0
}

final_model = XGBRegressor(**best_params)
final_model.fit(X_train, y_train)

y_pred_log = final_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

baseline_mae = mean_absolute_error(y_true, y_pred)
baseline_rmse = math.sqrt(mean_squared_error(y_true, y_pred))
baseline_r2 = r2_score(y_true, y_pred)

print(f"MAE={baseline_mae:.2f}, RMSE={baseline_rmse:.2f}, R²={baseline_r2:.5f}")

ablation_features = [
    'Total_departures', 'Total_flighthours', 'total_flown_km',
    'total_passenger_carried', 'available_seat_kilometre',
    'pax_load_factor_percent', 'freight_tonne', 'mail_tonne'
]

results = []

for feat in ablation_features:
    print(f"Evaluating without feature: {feat}")
    X_train_drop = X_train.drop(columns=[feat])
    X_test_drop = X_test.drop(columns=[feat])

    model = XGBRegressor(**best_params)
    model.fit(X_train_drop, y_train)

    y_pred_log = model.predict(X_test_drop)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    results.append({
        "Feature Dropped": feat,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "ΔR² (vs baseline)": r2 - baseline_r2,
        "ΔMAE": mae - baseline_mae,
        "ΔRMSE": rmse - baseline_rmse
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="ΔR² (vs baseline)", ascending=True).reset_index(drop=True)

print(results_df)

