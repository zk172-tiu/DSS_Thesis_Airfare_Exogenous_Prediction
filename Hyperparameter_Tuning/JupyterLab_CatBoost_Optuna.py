import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np

def objective(trial):
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
        'random_seed': 42
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
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

pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=5)

study = optuna.create_study(direction='minimize', pruner=pruner)
study.optimize(objective, n_trials=30, show_progress_bar=True)  


print("Best MAE:", study.best_trial.value)
print("Best params:", study.best_trial.params)
