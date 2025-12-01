import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

df = pd.read_csv("clean.csv")
df = df.drop_duplicates().reset_index(drop=True)
df['Date_of_journey'] = pd.to_datetime(df['Date_of_journey'])

df['journey_month'] = df['Date_of_journey'].dt.month
df['journey_weekofyear'] = df['Date_of_journey'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['Date_of_journey'].dt.dayofweek.isin([5, 6]).astype(int)

df['route'] = df['Source'] + "_" + df['Destination']
df['airline_route'] = df['Airline'] + "_" + df['Source'] + "_" + df['Destination']

df['airline_freq'] = df.groupby('Airline')['Airline'].transform('count')
df['route_freq'] = df.groupby('route')['route'].transform('count')

df['is_nonstop'] = (df['Total_stops'] == 'non-stop').astype(int)
df['num_stops'] = df['Total_stops'].apply(lambda x: 0 if x == 'non-stop'
                                          else (1 if '1 stop' in x else 2))

df['is_red_eye'] = ((df['Departure'] == 'After 10 PM') &
                    (df['Arrival'] == 'Before 6 AM')).astype(int)

df['duration_bin'] = pd.cut(df['Duration_in_hours'],
                            bins=[0, 3, 6, 12, 20, np.inf],
                            labels=['short', 'medium', 'long', 'extra_long', 'ultra_long'])

df['is_premium'] = df['Class'].isin(['Business', 'First', 'Premium Economy']).astype(int)

df['Fare_log'] = np.log1p(df['Fare'])

label_cols = ['Flight_code', 'airline_route']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    df[col] = df[col].astype('category')

cat_cols = [
    'Journey_day', 'Airline', 'Class', 'Source', 'Departure',
    'Arrival', 'Destination', 'Total_stops',
    'Flight_code', 'route', 'airline_route', 'duration_bin'
]

for col in cat_cols:
    df[col] = df[col].astype('category')

X = df.drop(['Fare', 'Fare_log', 'Date_of_journey'], axis=1)
y = df['Fare_log']

best_params_base = {
    'n_estimators': 691,
    'max_depth': 12,
    'learning_rate': 0.07574225743711721,
    'subsample': 0.9977092878441579,
    'colsample_bytree': 0.6666309761131906,
    'reg_alpha': 0.6257912057988657,
    'reg_lambda': 0.6693561004312016,
    'min_child_weight': 8,
    'n_jobs': -1,
    'enable_categorical': True,
    'verbosity': 0
}

results = []

for seed in range(42, 47):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    best_params = best_params_base.copy()
    best_params['random_state'] = seed

    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)

    y_pred_train = np.expm1(model.predict(X_train))
    y_true_train = np.expm1(y_train)

    y_pred_test = np.expm1(model.predict(X_test))
    y_true_test = np.expm1(y_test)

    r2 = r2_score(y_true_test, y_pred_test)
    mae = mean_absolute_error(y_true_test, y_pred_test)
    rmse = math.sqrt(mean_squared_error(y_true_test, y_pred_test))

    results.append([seed, r2, mae, rmse])

    print(f"\nSeed {seed}")
    print(f"R2   : {r2}")
    print(f"MAE  : {mae}")
    print(f"RMSE : {rmse}")

df_results = pd.DataFrame(results, columns=['seed', 'R2', 'MAE', 'RMSE'])
print(df_results)
