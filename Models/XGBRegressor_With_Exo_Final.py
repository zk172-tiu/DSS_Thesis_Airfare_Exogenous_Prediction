import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

df = pd.read_csv("merged_df.csv", index_col=0)
df = df.drop_duplicates().reset_index(drop=True)
df['Date_of_journey'] = pd.to_datetime(df['Date_of_journey'])
df['pax_load_factor_percent'] = df['pax_load_factor_percent'].str.rstrip('%').astype(float)

df['journey_month'] = df['Date_of_journey'].dt.month
df['journey_weekofyear'] = df['Date_of_journey'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['Date_of_journey'].dt.dayofweek.isin([5,6]).astype(int)
df['route'] = df['Source'] + "_" + df['Destination']
df['airline_route'] = df['Airline'] + "_" + df['Source'] + "_" + df['Destination']
df['airline_freq'] = df.groupby('Airline')['Airline'].transform('count')
df['route_freq'] = df.groupby('route')['route'].transform('count')
df['is_nonstop'] = (df['Total_stops'] == 'non-stop').astype(int)
df['num_stops'] = df['Total_stops'].apply(lambda x: 0 if x=='non-stop' else (1 if '1 stop' in x else 2))
df['is_red_eye'] = ((df['Departure']=='After 10 PM') & (df['Arrival']=='Before 6 AM')).astype(int)
df['duration_bin'] = pd.cut(df['Duration_in_hours'], bins=[0,3,6,12,20,np.inf],
                            labels=['short','medium','long','extra_long','ultra_long'])
df['is_premium'] = df['Class'].isin(['Business','First','Premium Economy']).astype(int)
df['Fare_log'] = np.log1p(df['Fare'])

label_cols = ['Flight_code','airline_route']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    df[col] = df[col].astype('category')

cat_cols = ['Journey_day','Airline','Class','Source','Departure','Arrival',
            'Destination','Total_stops','Flight_code','route','airline_route','duration_bin','Month']
for col in cat_cols:
    df[col] = df[col].astype('category')

X = df.drop(['Fare','Fare_log','Date_of_journey'], axis=1)
y = df['Fare_log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_params_v2 = {
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

kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_list, mae_list, rmse_list = [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train),1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = XGBRegressor(**best_params_v2)
    model.fit(X_tr, y_tr)

    y_pred = np.expm1(model.predict(X_val))
    y_true = np.expm1(y_val)

    r2_list.append(r2_score(y_true, y_pred))
    mae_list.append(mean_absolute_error(y_true, y_pred))
    rmse_list.append(math.sqrt(mean_squared_error(y_true, y_pred)))

print("Mean CV=5 results (training set)")
print(f"R²: {np.mean(r2_list)}, MAE: {np.mean(mae_list)}, RMSE: {np.mean(rmse_list)}")

final_model_v2 = XGBRegressor(**best_params_v2)
final_model_v2.fit(X_train, y_train)

y_pred_test = np.expm1(final_model_v2.predict(X_test))
y_true_test = np.expm1(y_test)

print("Test set results")
print(f"R²={r2_score(y_true_test, y_pred_test)}, MAE={mean_absolute_error(y_true_test, y_pred_test)}, RMSE={math.sqrt(mean_squared_error(y_true_test, y_pred_test))}")
