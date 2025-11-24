import optuna
from optuna.pruners import MedianPruner
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 8, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': 2 
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
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

        # 🔹 Report partial progress for pruning
        trial.report(np.mean(mae_scores), step=len(mae_scores))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(mae_scores)  # Minimaliseer MAE


study = optuna.create_study(direction='minimize', pruner=MedianPruner(n_startup_trials=5))
study.optimize(objective, n_trials=30, n_jobs=2, show_progress_bar=True)


print("Best trial")
trial = study.best_trial
print(f"Best MAE: {trial.value}")
print("Best Params:")
for key, value in trial.params.items():
    print(f"  {key}: {value}")
