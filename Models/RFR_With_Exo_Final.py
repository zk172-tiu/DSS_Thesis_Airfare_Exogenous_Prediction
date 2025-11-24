import pandas as pd
from carbontracker.tracker import CarbonTracker
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

df = pd.read_csv("merged_df.csv", index_col=0)
df = df.drop_duplicates()
df = df.reset_index(drop=True)

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

df = df.drop('Date_of_journey', axis=1)

df['Fare_log'] = np.log1p(df['Fare'])

label_cols = ['Flight_code', 'airline_route']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col].astype(str))

cat_cols = [
    'Journey_day', 'Airline', 'Class', 'Source', 'Departure',
    'Arrival', 'Destination', 'Total_stops', 'route', 'duration_bin', 'Month'
]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop(['Fare', 'Fare_log'], axis=1)
y = df['Fare_log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_params = {
    'n_estimators': 124,
    'max_depth': 29,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': None,
    'random_state': 42,
    'n_jobs': -1
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_list, mae_list, rmse_list = [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = RandomForestRegressor(**best_params)
    model.fit(X_tr, y_tr)

    y_pred_log = model.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_val)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    r2_list.append(r2)
    mae_list.append(mae)
    rmse_list.append(rmse)

    print(f"Fold {fold}: R²={r2}, MAE={mae}, RMSE={rmse}")

print("Mean CV=5 results (training set)")
print(f"Mean R²: {np.mean(r2_list)}")
print(f"Mean MAE: {np.mean(mae_list)}")
print(f"Mean RMSE: {np.mean(rmse_list)}")

tracker = CarbonTracker(epochs=1, monitor_epochs=1, verbose=1)
tracker.epoch_start()

final_model = RandomForestRegressor(**best_params)
final_model.fit(X_train, y_train)

tracker.epoch_end()
tracker.stop()

y_pred_log_test = final_model.predict(X_test)
y_pred_test = np.expm1(y_pred_log_test)
y_true_test = np.expm1(y_test)

mae_test = mean_absolute_error(y_true_test, y_pred_test)
rmse_test = math.sqrt(mean_squared_error(y_true_test, y_pred_test))
r2_test = r2_score(y_true_test, y_pred_test)

print("Test set results")
print(f"R²={r2_test}, MAE={mae_test}, RMSE={rmse_test}")
