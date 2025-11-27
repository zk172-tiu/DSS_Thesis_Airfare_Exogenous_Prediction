import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from carbontracker.tracker import CarbonTracker
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
generator = torch.Generator().manual_seed(SEED)


hidden1 = 640
hidden2 = 512
hidden3 = 128
hidden4 = 96
dropout = 0.2
lr = 0.0004922437203478091
batch_size = 512
weight_decay = 7.566924510458423e-06
epochs = 300
patience = 10

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
df['is_premium'] = df['Class'].isin(['Business','First','Premium Economy']).astype(int)
df['Fare_log'] = np.log1p(df['Fare'])

cat_cols = [c for c in df.columns if df[c].dtype=='object' or str(df[c].dtype)=='category']
num_cols = [c for c in df.columns if c not in cat_cols + ['Fare','Fare_log','Date_of_journey']]

for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X_num = torch.tensor(df[num_cols].values, dtype=torch.float32).to(device)
X_cat = torch.tensor(df[cat_cols].values, dtype=torch.long).to(device)
y = torch.tensor(df['Fare_log'].values, dtype=torch.float32).view(-1,1).to(device)

cat_embedding_sizes = [(len(df[col].unique()), min(50, (len(df[col].unique())+1)//2)) for col in cat_cols]
embedding_dim = sum([size for _, size in cat_embedding_sizes])
num_features = X_num.shape[1]

total_size = len(X_num)
test_size = int(0.1 * total_size)
val_size = int(0.1 * total_size)
train_size = total_size - test_size - val_size

dataset = TensorDataset(X_num, X_cat, y)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

embeddings = nn.ModuleList([nn.Embedding(categories, size).to(device) for categories, size in cat_embedding_sizes])

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

optimizer = torch.optim.Adam(list(model.parameters()) + list(embeddings.parameters()), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

tracker = CarbonTracker(epochs=1, monitor_epochs=1, verbose=1)
tracker.epoch_start()

best_val_loss = float('inf')
counter = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for xb_num, xb_cat, yb in train_loader:
        optimizer.zero_grad()
        xb_emb = torch.cat([embeddings[i](xb_cat[:,i]) for i, emb in enumerate(embeddings)], dim=1)
        xb_full = torch.cat([xb_num, xb_emb], dim=1)
        preds = model(xb_full)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb_num.size(0)
    train_loss = running_loss / train_size

    model.eval()
    val_running_loss = 0
    with torch.no_grad():
        for xb_num, xb_cat, yb in val_loader:
            xb_emb = torch.cat([embeddings[i](xb_cat[:,i]) for i, emb in enumerate(embeddings)], dim=1)
            xb_full = torch.cat([xb_num, xb_emb], dim=1)
            preds = model(xb_full)
            loss = criterion(preds, yb)
            val_running_loss += loss.item() * xb_num.size(0)
    val_loss = val_running_loss / val_size

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_4layer_mlp.pt")
    else:
        counter += 1
        if counter >= patience:
            break

tracker.epoch_end()
tracker.stop()

model.load_state_dict(torch.load("best_4layer_mlp.pt"))
model.eval()

train_preds, train_targets = [], []
with torch.no_grad():
    for xb_num, xb_cat, yb in train_loader:
        xb_emb = torch.cat([embeddings[i](xb_cat[:,i]) for i, emb in enumerate(embeddings)], dim=1)
        xb_full = torch.cat([xb_num, xb_emb], dim=1)
        preds = model(xb_full)
        train_preds.append(preds.cpu().numpy())
        train_targets.append(yb.cpu().numpy())

train_preds = np.vstack(train_preds)
train_targets = np.vstack(train_targets)
train_preds_exp = np.expm1(train_preds)
train_targets_exp = np.expm1(train_targets)
train_mae = mean_absolute_error(train_targets_exp, train_preds_exp)
train_rmse = np.sqrt(mean_squared_error(train_targets_exp, train_preds_exp))
train_r2 = r2_score(train_targets_exp, train_preds_exp)

test_preds, test_targets = [], []
with torch.no_grad():
    for xb_num, xb_cat, yb in test_loader:
        xb_emb = torch.cat([embeddings[i](xb_cat[:,i]) for i, emb in enumerate(embeddings)], dim=1)
        xb_full = torch.cat([xb_num, xb_emb], dim=1)
        preds = model(xb_full)
        test_preds.append(preds.cpu().numpy())
        test_targets.append(yb.cpu().numpy())

test_preds = np.vstack(test_preds)
test_targets = np.vstack(test_targets)
test_preds_exp = np.expm1(test_preds)
test_targets_exp = np.expm1(test_targets)
test_mae = mean_absolute_error(test_targets_exp, test_preds_exp)
test_rmse = np.sqrt(mean_squared_error(test_targets_exp, test_preds_exp))
test_r2 = r2_score(test_targets_exp, test_preds_exp)

print(f"Final MAE: {train_mae}, RMSE: {train_rmse}, R²: {train_r2}")
print(f"Final  MAE: {test_mae}, RMSE: {test_rmse}, R²: {test_r2}")
