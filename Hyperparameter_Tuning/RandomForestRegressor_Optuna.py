# file: RandomForestRegressor_Optuna.py

"""
This module implements hyperparameter optimisation for a RandomForestRegressor
using Optuna. The function `tune_random_forest` performs K-Fold
cross-validation exclusively on the training data to prevent data leakage and
returns the optimal hyperparameters that minimise mean absolute error (MAE).
"""

import optuna
from optuna.pruners import MedianPruner
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

def tune_random_forest(X_train, y_train, n_trials=30, n_splits=3, n_jobs=2):
    """
    Conducts hyperparameter tuning for a RandomForestRegressor model using Optuna.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (log-transformed if applicable).
    n_trials : int, optional (default=30)
        The number of trials to perform in the Optuna study.
    n_splits : int, optional (default=3)
        The number of folds to employ in K-Fold cross-validation.
    n_jobs : int, optional (default=2)
        The number of CPU cores to utilise for RandomForest training.

    Returns
    -------
    dict
        The dictionary of hyperparameters yielding the lowest MAE across cross-validation folds.
    """

    def objective(trial):
        """
        Objective function for Optuna. Evaluates a given set of hyperparameters
        using K-Fold cross-validation on the training set and returns the mean MAE.
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 8, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': n_jobs
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mae_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = RandomForestRegressor(**params)
            model.fit(X_tr, y_tr)


            y_pred_log = model.predict(X_val)
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(y_val)

            mae_scores.append(mean_absolute_error(y_true, y_pred))


            trial.report(np.mean(mae_scores), step=len(mae_scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(mae_scores)


    study = optuna.create_study(direction='minimize', pruner=MedianPruner(n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    print("Optimal trial identified:")
    trial = study.best_trial
    print(f"Best MAE: {trial.value:.4f}")
    print("Optimal hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    return trial.params


def main():
    """
    Example standalone execution of the hyperparameter tuning procedure.
    Assumes that 'clean_train.csv' contains only the training data and that
    the target has been log-transformed if necessary.
    """

    df = pd.read_csv("clean.csv")
    X_train = df.drop(['Fare', 'Fare_log'], axis=1)
    y_train = df['Fare_log']  

    best_params = tune_random_forest(X_train, y_train, n_trials=30)
    print("Hyperparameter tuning completed. Optimal parameters:")
    print(best_params)


if __name__ == "__main__":
    main()
