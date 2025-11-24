# file: HPT_MLP_Optuna.py

"""
Hyperparameter tuning for a 4-layer MLP using Optuna.
This script performs K-Fold cross-validation exclusively on the training set
to prevent data leakage and returns the optimal hyperparameters that minimise
mean absolute error (MAE).
"""

import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np
import torch
import torch.nn as nn

def objective(trial):
    """
    Objective function for Optuna hyperparameter tuning.
    Evaluates a 4-layer MLP using K-Fold CV on the training set.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object to suggest hyperparameters.

    Returns
    -------
    float
        Mean MAE across K-Fold validation.
    """

    hidden1 = trial.suggest_int("hidden1", 256, 1024, step=64)
    hidden2 = trial.suggest_int("hidden2", 128, 512, step=32)
    hidden3 = trial.suggest_int("hidden3", 64, 256, step=16)
    hidden4 = trial.suggest_int("hidden4", 32, 128, step=16)
    dropout = trial.suggest_float("dropout", 0.1, 0.25, step=0.05)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_list = []

    for train_idx, val_idx in kf.split(X_num_train):
        X_num_tr, X_num_val = X_num_train[train_idx], X_num_train[val_idx]
        X_cat_tr, X_cat_val = X_cat_train[train_idx], X_cat_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        embeddings = nn.ModuleList([
            nn.Embedding(categories, size).to(device)
            for categories, size in cat_embedding_sizes
        ])

        num_features = X_num_tr.shape[1]
        embedding_dim = sum([size for _, size in cat_embedding_sizes])

        model = nn.Sequential(
            nn.Linear(num_features + embedding_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, hidden4),
            nn.ReLU(),
            nn.Linear(hidden4, 1)
        ).to(device)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(embeddings.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        loss_fn = nn.MSELoss()
        epochs = 50

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(X_num_tr.size(0))
            for i in range(0, X_num_tr.size(0), batch_size):
                idx = perm[i:i + batch_size]
                X_num_batch = X_num_tr[idx]
                X_cat_batch = X_cat_tr[idx]
                y_batch = y_tr[idx]

                X_cat_emb = torch.cat([emb(X_cat_batch[:, j]) for j, emb in enumerate(embeddings)], dim=1)
                X_batch = torch.cat([X_num_batch, X_cat_emb], dim=1)
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_cat_emb_val = torch.cat([emb(X_cat_val[:, j]) for j, emb in enumerate(embeddings)], dim=1)
            X_val_full = torch.cat([X_num_val, X_cat_emb_val], dim=1)
            y_pred_val = model(X_val_full)

        y_pred_np = np.expm1(y_pred_val.cpu().numpy().flatten())
        y_true_np = np.expm1(y_val.cpu().numpy().flatten())
        mae_list.append(mean_absolute_error(y_true_np, y_pred_np))

    return np.mean(mae_list)

def main():
    """
    Run Optuna hyperparameter tuning for the 4-layer MLP.
    Assumes the following global variables are defined:
    X_num_train, X_cat_train, y_train, cat_embedding_sizes, device
    """

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=60, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial
    print(trial.params)
    print(f"Best MAE: {trial.value}")

if __name__ == "__main__":
    main()
