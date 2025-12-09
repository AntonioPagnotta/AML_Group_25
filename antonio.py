import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, mean_squared_error, r2_score
import os
from tqdm import tqdm


# --- CONFIGURATION ---
class Config:
    # UPDATE THIS PATH TO YOUR FILE
    data_path = 'embeddings.pkl'

    epochs = 15
    batch_size = 256

    # --- DATA LIMITS (Use these to fix memory/speed issues) ---
    max_samples = 1000  # Example: Set to 2000 to train on a subset. None = use all.
    max_seq_len = 1024  # Max length for embeddings (truncates if longer)
    # ----------------------------------------------------------

    # Hyperbolic Params
    hyp_lr = 5e-4  # Slightly higher for regression
    hyp_dim = 128

    # Euclidean Params
    euc_lr = 1e-3
    euc_hidden = [1024, 512]

    val_frac = 0.2
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- UTILS ---

def discretize_fitness(fitness_array):
    """Transforms continuous fitness into 5 discrete classes for F1 comparison."""
    bins = [-np.inf, -3.0, -1.0, 1.0, 3.0, np.inf]
    labels = np.digitize(fitness_array, bins) - 1
    return labels


class MutationDataset(Dataset):
    def __init__(self, pkl_path, max_samples=None, max_seq_len=None):
        print(f"Loading {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 1. Load Data
        if hasattr(data, 'loc'):
            self.fitness = data['fitness'].values
            embeddings_raw = data['embedding'].values
            self.file_names = data['file_name'].values
        else:
            self.fitness = np.array(data['fitness'])
            embeddings_raw = np.array(data['embedding'])
            self.file_names = np.array(data['file_name'])

        # 2. Limit Number of Samples (Optional)
        if max_samples is not None and max_samples < len(self.fitness):
            print(f"Limiting dataset to first {max_samples} samples.")
            self.fitness = self.fitness[:max_samples]
            embeddings_raw = embeddings_raw[:max_samples]
            self.file_names = self.file_names[:max_samples]

        self.labels = discretize_fitness(self.fitness)

        # 3. Analyze Shapes and Padding
        np_embs = [np.array(e) for e in embeddings_raw]

        # Determine max dimensions
        max_h_data = max(s[0] for s in [e.shape for e in np_embs])

        if max_seq_len:
            self.max_h = min(max_h_data, max_seq_len)
        else:
            self.max_h = max_h_data

        # Handle 1D vs 2D embeddings
        if len(np_embs[0].shape) > 1:
            self.max_w = max(s[1] for s in [e.shape for e in np_embs])
        else:
            self.max_w = 1

        print(f"Processing Embeddings: Target Shape=({self.max_h}, {self.max_w})")

        processed_embs = []
        for arr in tqdm(np_embs, desc="Padding/Truncating"):
            # Create placeholder
            if len(arr.shape) == 1:
                # 1D Case
                current_len = arr.shape[0]
                limit = min(current_len, self.max_h)
                padded = np.zeros((self.max_h,), dtype=np.float32)
                padded[:limit] = arr[:limit]
            else:
                # 2D Case
                h, w = arr.shape
                limit_h = min(h, self.max_h)
                limit_w = min(w, self.max_w)
                padded = np.zeros((self.max_h, self.max_w), dtype=np.float32)
                padded[:limit_h, :limit_w] = arr[:limit_h, :limit_w]

            processed_embs.append(padded.flatten())

        self.embeddings = np.array(processed_embs, dtype=np.float32)

        # 4. Normalize (Z-Score) - Critical for Convergence
        mean = self.embeddings.mean(axis=0)
        std = self.embeddings.std(axis=0) + 1e-6
        self.embeddings = (self.embeddings - mean) / std

        self.input_dim = self.embeddings.shape[1]
        self.proteins = np.array([str(fn).split('_')[0] for fn in self.file_names])

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            'embedding': torch.tensor(self.embeddings[idx]),
            'fitness': torch.tensor(self.fitness[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'protein': self.proteins[idx]
        }


def collate_fn(batch):
    embeddings = torch.stack([b['embedding'] for b in batch])
    fitness = torch.stack([b['fitness'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    proteins = [b['protein'] for b in batch]
    return embeddings, fitness, labels, proteins


# ==========================================
# 1. HYPERBOLIC REGRESSOR
# ==========================================

# Use float64 for Hyperbolic stability
torch.set_default_dtype(torch.float64)


class HyperbolicUtils:
    EPS = 1e-5

    @staticmethod
    def artanh(x):
        x = torch.clamp(x, min=-1.0 + 1e-5, max=1.0 - 1e-5)
        return 0.5 * torch.log((1 + x) / (1 - x))

    @staticmethod
    def exp_map_zero(v):
        """Maps tangent vector v at origin to the Poincare ball."""
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        scaled = torch.tanh(v_norm) * (v / v_norm)
        return scaled

    @staticmethod
    def log_map_zero(y):
        """Maps point y in Poincare ball back to tangent space at origin."""
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True).clamp(min=1e-6, max=1.0 - 1e-5)
        scale = HyperbolicUtils.artanh(y_norm) / y_norm
        return y * scale


class HyperbolicRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, hyp_dim):
        super().__init__()

        layers = []
        in_d = input_dim

        # 1. Euclidean Encoder
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_d, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_d = h_dim

        self.encoder = nn.Sequential(*layers)

        # 2. Projection to Tangent Space of the Ball
        self.to_tangent = nn.Linear(in_d, hyp_dim)

        # 3. Regression Head (Linear)
        # We perform regression on the tangent plane output
        self.regressor = nn.Linear(hyp_dim, 1)

    def forward(self, x):
        x = x.double()

        # Encode
        feat = self.encoder(x)
        tangent = self.to_tangent(feat)

        # --- HYPERBOLIC BOTTLENECK ---
        # 1. Clip tangent norm (Stability)
        norm = torch.norm(tangent, p=2, dim=-1, keepdim=True)
        scale = torch.clamp(norm, max=5.0) / (norm + 1e-6)
        tangent = tangent * scale

        # 2. Project to Poincare Ball (Hyperbolic Space)
        hyp_emb = HyperbolicUtils.exp_map_zero(tangent)

        # 3. Project back to Tangent Space for Regression
        # This enforces that the latent representation `hyp_emb` strictly
        # adheres to hyperbolic geometry constraints before we read it out.
        tangent_out = HyperbolicUtils.log_map_zero(hyp_emb)

        # Output Scalar
        output = self.regressor(tangent_out)

        return output.squeeze()


# ==========================================
# 2. EUCLIDEAN REGRESSOR
# ==========================================

class EuclideanRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512]):
        super().__init__()

        layers = []
        in_d = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_d, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_d = h_dim

        # Output dimension 1
        layers.append(nn.Linear(in_d, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()  # Use float32
        return self.model(x).squeeze()


# ==========================================
# 3. TRAINING LOOPS
# ==========================================

def evaluate_regression(model, loader, device, model_type="euclidean"):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for embs, fitness, _, _ in loader:
            if model_type == "euclidean":
                embs = embs.to(device).float()
            else:
                embs = embs.to(device).double()

            fitness = fitness.to(device)

            # Predict
            preds = model(embs)

            all_preds.append(preds.cpu().numpy())
            all_true.append(fitness.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate Equivalent F1
    y_pred_cls = discretize_fitness(y_pred)
    y_true_cls = discretize_fitness(y_true)
    f1 = f1_score(y_true_cls, y_pred_cls, average='weighted', zero_division=0)

    return mse, r2, f1


def train_hyperbolic(config, train_loader, val_loader, input_dim):
    print("\n--- Training Hyperbolic Model (Regression) ---")
    torch.set_default_dtype(torch.float64)  # Float64 context

    model = HyperbolicRegressor(input_dim, [512], config.hyp_dim).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.hyp_lr)
    criterion = nn.MSELoss()

    best_mse = float('inf')
    best_stats = (0, 0, 0)  # mse, r2, f1

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for embs, fitness, _, _ in train_loader:
            embs = embs.to(config.device)
            fitness = fitness.to(config.device).double()  # Target must be double too

            pred = model(embs)
            loss = criterion(pred, fitness)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        mse, r2, f1 = evaluate_regression(model, val_loader, config.device, "hyperbolic")

        print(
            f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f} | Val MSE: {mse:.4f} | R2: {r2:.4f} | Eq F1: {f1:.4f}")

        if mse < best_mse:
            best_mse = mse
            best_stats = (mse, r2, f1)

    return best_stats


def train_euclidean(config, train_loader, val_loader, input_dim):
    print("\n--- Training Euclidean Model (Regression) ---")
    torch.set_default_dtype(torch.float32)  # Switch context

    model = EuclideanRegressor(input_dim, config.euc_hidden).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.euc_lr)
    criterion = nn.MSELoss()

    best_mse = float('inf')
    best_stats = (0, 0, 0)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for embs, fitness, _, _ in train_loader:
            embs = embs.to(config.device).float()
            fitness = fitness.to(config.device).float()

            pred = model(embs)
            loss = criterion(pred, fitness)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        mse, r2, f1 = evaluate_regression(model, val_loader, config.device, "euclidean")

        print(f"Epoch {epoch + 1} | MSE: {mse:.4f} | R2: {r2:.4f} | Eq F1: {f1:.4f}")

        if mse < best_mse:
            best_mse = mse
            best_stats = (mse, r2, f1)

    return best_stats


# --- MAIN ---

def main():
    if not os.path.exists(Config.data_path):
        print(f"File not found: {Config.data_path}")
        return

    # Data Loading
    dataset = MutationDataset(
        Config.data_path,
        max_samples=Config.max_samples,
        max_seq_len=Config.max_seq_len
    )

    train_size = int((1 - Config.val_frac) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Dataset Loaded. Input Dim: {dataset.input_dim}")

    # 1. Train Hyperbolic
    hyp_mse, hyp_r2, hyp_f1 = train_hyperbolic(Config, train_loader, val_loader, dataset.input_dim)

    # 2. Train Euclidean
    euc_mse, euc_r2, euc_f1 = train_euclidean(Config, train_loader, val_loader, dataset.input_dim)

    print("\n================ FINAL COMPARISON (REGRESSION) ================")
    print(f"{'Metric':<10} | {'Hyperbolic':<15} | {'Euclidean':<15}")
    print("-" * 45)
    print(f"{'MSE (low)':<10} | {hyp_mse:<15.4f} | {euc_mse:<15.4f}")
    print(f"{'R2 (high)':<10} | {hyp_r2:<15.4f} | {euc_r2:<15.4f}")
    print(f"{'Eq F1':<10} | {hyp_f1:<15.4f} | {euc_f1:<15.4f}")

    if euc_mse < hyp_mse:
        print("\n>> Euclidean Model performed better (Lower MSE).")
    else:
        print("\n>> Hyperbolic Model performed better (Lower MSE).")


if __name__ == '__main__':
    main()