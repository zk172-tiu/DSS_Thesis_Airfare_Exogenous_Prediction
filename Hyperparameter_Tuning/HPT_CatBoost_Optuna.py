# file: CatBoost_Optuna.py

"""
This module implements hyperparameter optimisation for a CatBoostRegressor
using Optuna. The function `tune_catboost` performs K-Fold
cross-validation exclusively on the training data to prevent data leakage and
returns the optimal hyperparameters that minimise mean absolute error (MAE).
"""

import optuna
from optuna.pruners import MedianPruner
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

def tune_catboost(X_train, y_train, cat_features=None, n_trials=30, n_splits=3, random_seed=42):
    """
    Conducts hyperparameter tuning for a CatBoostRegressor model using Optuna.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (log-transformed if applicable).
    cat_features : list, optional
        List of categorical feature column names or indices.
    n_trials : int, optional (default=30)
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
            'iterations': trial.suggest_categorical('iterations', [800, 1000, 1200, 1400, 1600]),
            'depth': trial.suggest_int('depth', 7, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.08, 0.12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 6),
            'random_strength': trial.suggest_float('random_strength', 0, 5),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1.5),
            'border_count': trial.suggest_categorical('border_count', [32, 64, 128]),
            'loss_function': 'MAE',
            'task_type': 'GPU',
            'verbose': 0,
            'random_seed': random_seed
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        mae_scores = []

        for train_idx, valid_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            model = CatBoostRegressor(**params)
            model.fit(
                X_tr, y_tr,
                cat_features=cat_features,
                eval_set=(X_val, y_val),
                early_stopping_rounds=20,
                verbose=0
            )

            
            y_pred_log = model.predict(X_val)
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(y_val)

            mae_scores.append(mean_absolute_error(y_true, y_pred))

        return np.mean(mae_scores)

    
    pruner = MedianPruner(n_startup_trials=1, n_warmup_steps=5)
    study = optuna.create_study(direction='minimize', pruner=pruner)
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
    Example standalone execution of CatBoost hyperparameter tuning.
    Assumes that 'clean_train.csv' contains only the training data and that
    the target has been log-transformed if applicable.
    """

    
    df = pd.read_csv("clean.csv")
    X_train = df.drop(['Fare', 'Fare_log'], axis=1)
    y_train = df['Fare_log']  

   
    cat_features = [
        'Airline', 'Source', 'Destination', 'Class', 'Total_stops'
    ]

    best_params = tune_catboost(X_train, y_train, cat_features=cat_features, n_trials=30)
    print("Hyperparameter tuning completed. Optimal parameters:")
    print(best_params)


if __name__ == "__main__":
    main()
