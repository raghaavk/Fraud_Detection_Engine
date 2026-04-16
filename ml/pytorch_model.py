import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import joblib
from data_loader import load_training_data, engineer_features
import config  # this loads all paths
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'data'))


# ── Neural Network Architecture ───────────────────────────────────────────────

class FraudDetectionNet(nn.Module):
    """
    Deep neural network for fraud detection.
    Uses batch normalisation and dropout to prevent overfitting.
    """

    def __init__(self, input_dim):
        super(FraudDetectionNet, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze()


# ── Training Function ──────────────────────────────────────────────────────────

def train_pytorch():
    # ── Load data ─────────────────────────────────────────────────
    df = load_training_data()
    X, y, feature_cols = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=0.2, random_state=42, stratify=y
    )

    # ── Scale features ────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ── Convert to tensors ────────────────────────────────────────
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)

    # ── Handle class imbalance with weighted sampler ───────────────
    fraud_count = y_train.sum()
    normal_count = len(y_train) - fraud_count
    weights = np.where(y_train == 1,
                       len(y_train) / (2 * fraud_count),
                       len(y_train) / (2 * normal_count))
    sampler = WeightedRandomSampler(
        torch.FloatTensor(weights), len(weights)
    )

    # ── DataLoaders ───────────────────────────────────────────────
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset, batch_size=512, sampler=sampler
    )

    # ── Model setup ───────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Training on: {device}")

    model = FraudDetectionNet(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── Training loop ─────────────────────────────────────────────
    print("🚀 Training PyTorch neural network...")
    epochs = 30

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_test_t.to(device)).cpu().numpy()
                auc = roc_auc_score(y_test, val_preds)
            print(f"   Epoch {epoch+1}/{epochs} | "
                  f"Loss: {total_loss/len(train_loader):.4f} | "
                  f"Val AUC: {auc:.4f}")

    # ── Final evaluation ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test_t.to(device)).cpu().numpy()

    y_pred = (y_prob >= 0.5).astype(int)
    final_auc = roc_auc_score(y_test, y_prob)

    print(f"\n📈 Final PyTorch AUC: {final_auc:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=['Normal', 'Fraud']))

    # ── Save model ────────────────────────────────────────────────
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/pytorch_fraud.pth")
    joblib.dump(scaler, "saved_models/scaler.pkl")
    joblib.dump(feature_cols, "saved_models/feature_cols.pkl")
    print("✅ PyTorch model saved to ml/saved_models/")

    return model, scaler, feature_cols


if __name__ == "__main__":
    train_pytorch()
