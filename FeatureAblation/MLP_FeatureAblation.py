import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

hidden1, hidden2, hidden3, hidden4 = 640, 512, 128, 96
dropout = 0.2
lr = 0.0004922437203478091
batch_size = 512
weight_decay = 7.566924510458423e-06
epochs = 300
patience = 10

df = pd.read_csv("merged_df.csv", index_col=0).drop_duplicates().reset_index(drop=True)

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
df['is_premium'] = df['Class'].isin(['Business', 'First', 'Premium Economy']).astype(int)
df['Fare_log'] = np.log1p(df['Fare'])

cat_cols = [c for c in df.columns if df[c].dtype == 'object' or str(df[c].dtype) == 'category']
num_cols = [c for c in df.columns if c not in cat_cols + ['Fare', 'Fare_log', 'Date_of_journey']]

for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X_num_full = torch.tensor(df[num_cols].values, dtype=torch.float32).to(device)
X_cat_full = torch.tensor(df[cat_cols].values, dtype=torch.long).to(device)
y_full = torch.tensor(df['Fare_log'].values, dtype=torch.float32).view(-1, 1).to(device)

total_size = len(X_num_full)
test_size = int(0.1 * total_size)
val_size = int(0.1 * total_size)
train_size = total_size - test_size - val_size

def build_model(num_features, cat_embedding_sizes):
    embeddings = nn.ModuleList([
        nn.Embedding(categories, size).to(device)
        for categories, size in cat_embedding_sizes
    ])
    embedding_dim = sum(size for _, size in cat_embedding_sizes)
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
    return model, embeddings

def train_model(model, embeddings, train_loader, val_loader):
    optimizer = torch.optim.Adam(list(model.parameters()) + list(embeddings.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    best_val_loss = float('inf')
    counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb_num, xb_cat, yb in train_loader:
            optimizer.zero_grad()
            xb_emb = torch.cat([embeddings[i](xb_cat[:, i]) for i in range(len(embeddings))], dim=1)
            xb_full = torch.cat([xb_num, xb_emb], dim=1)
            preds = model(xb_full)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb_num.size(0)
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for xb_num, xb_cat, yb in val_loader:
                xb_emb = torch.cat([embeddings[i](xb_cat[:, i]) for i in range(len(embeddings))], dim=1)
                xb_full = torch.cat([xb_num, xb_emb], dim=1)
                preds = model(xb_full)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb_num.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "tmp_best_model.pt")
            torch.save(embeddings.state_dict(), "tmp_best_emb.pt")
        else:
            counter += 1
            if counter >= patience:
                break
    model.load_state_dict(torch.load("tmp_best_model.pt"))
    embeddings.load_state_dict(torch.load("tmp_best_emb.pt"))
    return model, embeddings

def evaluate(model, embeddings, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb_num, xb_cat, yb in loader:
            xb_emb = torch.cat([embeddings[i](xb_cat[:, i]) for i in range(len(embeddings))], dim=1)
            xb_full = torch.cat([xb_num, xb_emb], dim=1)
            out = model(xb_full)
            preds.append(out.cpu().numpy())
            targets.append(yb.cpu().numpy())
    preds = np.expm1(np.vstack(preds))
    targets = np.expm1(np.vstack(targets))
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    return mae, rmse, r2

cat_embedding_sizes = [(len(df[col].unique()), min(50, (len(df[col].unique()) + 1) // 2)) for col in cat_cols]

dataset = TensorDataset(X_num_full, X_cat_full, y_full)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

baseline_model, baseline_emb = build_model(X_num_full.shape[1], cat_embedding_sizes)
baseline_model, baseline_emb = train_model(baseline_model, baseline_emb, train_loader, val_loader)
baseline_mae, baseline_rmse, baseline_r2 = evaluate(baseline_model, baseline_emb, test_loader)
print(f"Baseline MAE={baseline_mae:.2f}, RMSE={baseline_rmse:.2f}, R²={baseline_r2:.5f}")

ablation_features = [
    'Total_departures', 'Total_flighthours', 'total_flown_km',
    'total_passenger_carried', 'available_seat_kilometre',
    'pax_load_factor_percent', 'freight_tonne', 'mail_tonne'
]

results = []
for feat in ablation_features:
    print(f"Removing feature: {feat}")
    drop_idx = num_cols.index(feat)
    keep_indices = [i for i in range(len(num_cols)) if i != drop_idx]
    X_num_drop = X_num_full[:, keep_indices]

    dataset_drop = TensorDataset(X_num_drop, X_cat_full, y_full)
    train_dataset, val_dataset, test_dataset = random_split(dataset_drop, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model, emb = build_model(X_num_drop.shape[1], cat_embedding_sizes)
    model, emb = train_model(model, emb, train_loader, val_loader)
    mae, rmse, r2 = evaluate(model, emb, test_loader)

    results.append({
        "Feature Dropped": feat,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "ΔR² (vs baseline)": r2 - baseline_r2,
        "ΔMAE": mae - baseline_mae,
        "ΔRMSE": rmse - baseline_rmse
    })

results_df = pd.DataFrame(results).sort_values(by="ΔR² (vs baseline)", ascending=True).reset_index(drop=True)
print(results_df)
