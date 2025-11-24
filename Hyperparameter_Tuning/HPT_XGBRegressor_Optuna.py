# file: HPT_XGBRegressor_Optuna.py

"""
This module implements hyperparameter optimisation for an XGBRegressor
using Optuna. The function `tune_xgb` performs K-Fold
cross-validation exclusively on the training data to prevent data leakage and
returns the optimal hyperparameters that minimise mean absolute error (MAE).
"""

import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

def tune_xgb(X_train, y_train, n_trials=50, n_splits=3, random_seed=42):
    """
    Conducts hyperparameter tuning for an XGBRegressor model using Optuna.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (log-transformed if applicable).
    n_trials : int, optional (default=50)
        Number of trials to perform in the Optuna study.
    n_splits : int, optional (default=3)
        Number of folds to use in K-Fold cross-validation.
    random_seed : int, optional (default=42)
        Random seed for reproducibility.

    Returns
    -------
    dict
        Optimal hyperparameters yielding the lowest MAE across cross-validation folds.
    """

    def objective(trial):
        """
        Objective function for Optuna. Evaluates a given set of hyperparameters
        using K-Fold cross-validation on the training set and returns the mean MAE.
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': random_seed,
            'n_jobs': -1,
            'enable_categorical': True,
            'verbosity': 0
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        mae_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)

            
            y_pred_log = model.predict(X_val)
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(y_val)

            mae_scores.append(mean_absolute_error(y_true, y_pred))

        return np.mean(mae_scores)

    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Optimal trial identified:")
    trial = study.best_trial
    print(f"Best MAE: {trial.value:.4f}")
    print("Optimal hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    return trial.params


def main():
    """
    Example standalone execution of XGBRegressor hyperparameter tuning.
    Assumes that 'clean_train.csv' contains only the training data and that
    the target has been log-transformed if applicable.
    """

   
    df = pd.read_csv("clean.csv")
    X_train = df.drop(['Fare', 'Fare_log'], axis=1)
    y_train = df['Fare_log'] 

    best_params = tune_xgb(X_train, y_train, n_trials=50)
    print("Hyperparameter tuning completed. Optimal parameters:")
    print(best_params)


if __name__ == "__main__":
    main()
