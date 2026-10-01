from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from copycat.models.base import BaseCreatorModel
from copycat.data.schema import OBSERVABLE_CATEGORICAL_COLS, OBSERVABLE_NUMERIC_COLS

class LatentTasteModel(BaseCreatorModel):
    def __init__(
        self,
        k_factors: int = 5,
        learning_rate: float = 0.015,
        epochs: int = 300,
        l2_reg: float = 1e-4,
        random_seed: int = 42,
    ):
        self.k_factors = k_factors
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.random_seed = random_seed
        self.encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_net: Optional[nn.Linear] = None
        self.creator_tastes: Dict[str, np.ndarray] = {}
        self.creator_baselines: Dict[str, float] = {}
        self.global_baseline: float = 10.0

    def _extract_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        cat_data = df[OBSERVABLE_CATEGORICAL_COLS].values
        num_data = df[OBSERVABLE_NUMERIC_COLS].values
        
        if fit:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.scaler = StandardScaler()
            cat_enc = self.encoder.fit_transform(cat_data)
            num_enc = self.scaler.fit_transform(num_data)
        else:
            cat_enc = self.encoder.transform(cat_data)
            num_enc = self.scaler.transform(num_data)
            
        return np.hstack([cat_enc, num_enc])

    def get_latent_factors(self, df: pd.DataFrame) -> np.ndarray:
        X = self._extract_features(df, fit=False)
        self.feature_net.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            raw_v = self.feature_net(x_t).numpy()
            
        norms = np.linalg.norm(raw_v, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return raw_v / norms

    def fit(self, train_df: pd.DataFrame) -> "LatentTasteModel":
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        X = self._extract_features(train_df, fit=True)
        y = train_df["performance"].values
        log_y = np.log(np.maximum(y, 1.0))
        
        creators = train_df["creator_id"].values
        unique_creators = np.unique(creators)
        creator_to_idx = {cid: idx for idx, cid in enumerate(unique_creators)}
        c_indices = np.array([creator_to_idx[cid] for cid in creators])
        
        n_features = X.shape[1]
        n_creators = len(unique_creators)
        
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(log_y, dtype=torch.float32)
        c_tensor = torch.tensor(c_indices, dtype=torch.long)
        
        self.feature_net = nn.Linear(n_features, self.k_factors)
        creator_embeddings = nn.Embedding(n_creators, self.k_factors)
        creator_biases = nn.Embedding(n_creators, 1)
        
        nn.init.orthogonal_(creator_embeddings.weight)
        creator_biases.weight.data.fill_(float(np.mean(log_y)))
        
        optimizer = optim.Adam(
            list(self.feature_net.parameters())
            + list(creator_embeddings.parameters())
            + list(creator_biases.parameters()),
            lr=self.learning_rate,
            weight_decay=self.l2_reg,
        )
        
        criterion = nn.MSELoss()
        
        for _ in range(self.epochs):
            optimizer.zero_grad()
            v = self.feature_net(x_tensor)
            v_norm = v / (torch.norm(v, p=2, dim=1, keepdim=True) + 1e-8)
            
            u = creator_embeddings(c_tensor)
            u_norm = u / (torch.norm(u, p=2, dim=1, keepdim=True) + 1e-8)
            
            bias = creator_biases(c_tensor).squeeze(-1)
            dot = torch.sum(u_norm * v_norm, dim=1)
            
            pred = bias + 2.0 * dot
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
            
        learned_u = creator_embeddings.weight.detach().cpu().numpy()
        learned_b = creator_biases.weight.detach().cpu().numpy().squeeze(-1)
        
        for cid, idx in creator_to_idx.items():
            u_vec = learned_u[idx]
            norm = np.linalg.norm(u_vec)
            self.creator_tastes[cid] = u_vec / (norm if norm > 1e-9 else 1.0)
            self.creator_baselines[cid] = float(learned_b[idx])
            
        self.global_baseline = float(np.mean(log_y))
        return self

    def fit_creator_adaptation(self, creator_id: str, creator_train_df: pd.DataFrame) -> None:
        y = creator_train_df["performance"].values
        log_y = np.log(np.maximum(y, 1.0))
        baseline = float(np.mean(log_y))
        
        V = self.get_latent_factors(creator_train_df)
        target = (log_y - baseline) / 2.0
        
        ridge_eye = 0.05 * np.eye(self.k_factors)
        u_unnorm = np.linalg.solve(V.T @ V + ridge_eye, V.T @ target)
        norm = np.linalg.norm(u_unnorm)
        
        self.creator_tastes[creator_id] = u_unnorm / (norm if norm > 1e-9 else 1.0)
        self.creator_baselines[creator_id] = baseline

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        V = self.get_latent_factors(test_df)
        creators = test_df["creator_id"].values
        preds = np.zeros(len(test_df))
        
        for idx in range(len(test_df)):
            cid = creators[idx]
            v_vec = V[idx]
            
            if cid in self.creator_tastes:
                u_vec = self.creator_tastes[cid]
                baseline = self.creator_baselines[cid]
            else:
                u_vec = np.zeros(self.k_factors)
                baseline = self.global_baseline
                
            dot = float(np.dot(u_vec, v_vec))
            log_pred = baseline + 2.0 * dot
            preds[idx] = float(np.exp(log_pred))
            
        return preds
