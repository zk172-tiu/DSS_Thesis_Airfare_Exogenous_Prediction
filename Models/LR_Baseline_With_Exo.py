import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from carbontracker.tracker import CarbonTracker

df = pd.read_csv("merged_df.csv", index_col = 0)

X = df.drop(columns=["Fare", "Date_of_journey", "Flight_code"])
y = df["Fare"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_test = lr.predict(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = []
mae_scores = []
rmse_scores = []

for train_index, val_index in kf.split(X_train):
    X_tr = X_train.iloc[train_index]
    X_val = X_train.iloc[val_index]
    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[val_index]

    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_val)

    r2_scores.append(r2_score(y_val, y_pred))
    mae_scores.append(mean_absolute_error(y_val, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

print(f"Mean R²: {np.mean(r2_scores)}")
print(f"Mean MAE: {np.mean(mae_scores)}")
print(f"Mean RMSE: {np.mean(rmse_scores)}")

test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Test R²: {test_r2}")
print(f"Test MAE: {test_mae}")
print(f"Test RMSE: {test_rmse}")
