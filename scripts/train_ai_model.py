#!/usr/bin/env python3
"""
Train the BitNet AI Fusion model.
Uses ternary weights (-1, 0, 1) and 8-bit activations.
Now includes real-time PnL backtesting and Confusion Matrix logging.
"""

import sys
import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from tqdm.auto import tqdm
import logging
from datasets import load_dataset
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo, hf_hub_download, snapshot_download
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trading_cli.strategy.ai.model import create_model
from scripts.generate_ai_dataset import build_dataset, SEQ_LEN, FEATURE_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hyperparameters
EPOCHS = 100
LR = 5e-6 # Cut in half (was 1e-5)
HIDDEN_DIM = 512
LAYERS = 8
LABEL_SMOOTHING = 0.1 # Increased from 0.05
GRAD_CLIP_NORM = 1.0
MAX_LOGIT_MAGNITUDE = 10.0
USE_MIXED_PRECISION = False

# HF Configuration
HF_REPO_ID = os.getenv("HF_REPO_ID") # e.g. "username/BitFin"
HF_DATASET_ID = "luohoa97/BitFin" # User's dataset repo
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_PATH = "data/trading_dataset"
DATASET_CACHE_PATH = "data/trading_dataset/dataset_cache.pt"
MODEL_OUTPUT_PATH = "models/ai_fusion_model.safetensors"

class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def get_max_batch_size(model, input_dim, seq_len, device, start_batch=128):
    """Automatically find the largest batch size that fits in VRAM."""
    if device.type == 'cpu':
        return 64
        
    tqdm.write("🔍 Searching for optimal batch size for your GPU...")
    batch_size = start_batch
    last_success = batch_size
    
    pbar = tqdm(total=16384, desc="Hardware Probe", unit="batch")
    pbar.update(batch_size)
    
    try:
        while batch_size <= 16384: # Ceiling
            mock_X = torch.randn(batch_size, seq_len, input_dim).to(device)
            mock_y = torch.randint(0, 3, (batch_size,)).to(device)
            
            outputs = model(mock_X)
            loss = nn.CrossEntropyLoss()(outputs, mock_y)
            loss.backward()
            model.zero_grad()
            
            last_success = batch_size
            batch_size *= 2
            pbar.update(batch_size - last_success)
            torch.cuda.empty_cache()
            
    except RuntimeError as e:
        pbar.close()
        if "out of memory" in str(e).lower():
            tqdm.write(f"💡 GPU Hit limit at {batch_size}. Using {last_success} as optimal batch.")
            torch.cuda.empty_cache()
        else:
            raise e
            
    pbar.close()
    return last_success


def compute_balanced_class_weights(labels, num_classes=3):
    """Compute inverse-frequency class weights with sane lower bounds."""
    counts = torch.bincount(labels.cpu(), minlength=num_classes).float()
    safe_counts = counts.clamp(min=1.0)
    weights = labels.numel() / (num_classes * safe_counts)
    return counts, weights / weights.mean()


def build_weighted_sampler(labels, num_classes=3):
    """Sample minority classes more often to reduce prediction collapse."""
    _, class_weights = compute_balanced_class_weights(labels, num_classes=num_classes)
    sample_weights = class_weights[labels.cpu()]
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def load_dataset_from_parquet(dataset_path, cache_path=DATASET_CACHE_PATH):
    """Reconstruct tensors from parquet dataset files."""
    
    # Load metadata if available to get feature columns and seq_len
    metadata_path = os.path.join(dataset_path, "metadata.json") if os.path.isdir(dataset_path) else None
    
    feature_cols = FEATURE_COLUMNS
    sequence_length = SEQ_LEN
    
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            feature_cols = meta.get("feature_columns", FEATURE_COLUMNS)
            sequence_length = meta.get("seq_len", SEQ_LEN)
            print(f"Loaded metadata: {len(feature_cols)} features, seq_len={sequence_length}")

    if os.path.isdir(dataset_path):
        shard_paths = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
        if not shard_paths:
            raise FileNotFoundError(f"No parquet shards found in {dataset_path}")
        dataset = load_dataset("parquet", data_files={"train": shard_paths}, split="train")
    else:
        dataset = load_dataset("parquet", data_files={"train": dataset_path}, split="train")

    ordered_feature_cols = []
    for step in range(sequence_length):
        for feature_name in feature_cols:
            col = f"{feature_name}_t{step:02d}"
            if col not in dataset.column_names:
                raise ValueError(f"Missing parquet feature column: {col}")
            ordered_feature_cols.append(col)

    dataset = dataset.with_format("numpy")
    feature_arrays = [np.asarray(dataset[col], dtype=np.float32) for col in ordered_feature_cols]
    X = np.stack(feature_arrays, axis=1).reshape(len(dataset), sequence_length, len(feature_cols))
    y = np.asarray(dataset["label"], dtype=np.int64)
    rets = np.asarray(dataset["future_ret"], dtype=np.float32)
    symbols = sorted({str(symbol) for symbol in dataset["symbol"] if symbol is not None}) if "symbol" in dataset.column_names else []

    # Sanity Checks
    print(f"🔍 Performing data sanity checks...", flush=True)
    if torch.isnan(X).any():
        raise ValueError("Dataset contains NaNs in features (X)!")
    if torch.isinf(X).any():
        raise ValueError("Dataset contains Infs in features (X)!")
    
    # Feature distribution logging
    X_flat = X.view(-1, X.shape[-1]).numpy()
    df_desc = pd.DataFrame(X_flat, columns=feature_cols).describe().loc[['mean', 'std', 'min', 'max']]
    print("\n--- Feature Distributions ---")
    print(df_desc)
    print("----------------------------\n")

    data = {
        "X": X,
        "y": torch.tensor(y, dtype=torch.long),
        "rets": torch.tensor(rets, dtype=torch.float32),
        "symbols": symbols,
        "feature_columns": feature_cols,
    }
    torch.save(data, cache_path)
    return data

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    epochs = args.epochs
    model_type = args.model_type
    lr = args.lr

    # 1. Load or Generate Dataset
    if not os.path.exists(DATASET_PATH):
        try:
            print("📦 Fetching parquet dataset shards from Hugging Face...", flush=True)
            snapshot_download(
                repo_id=HF_DATASET_ID,
                repo_type="dataset",
                allow_patterns=["trading_dataset/*.parquet", "trading_dataset/metadata.json"],
                local_dir="data",
                local_dir_use_symlinks=False,
            )
        except Exception:
            print("🚀 Starting on-the-fly dataset generation...", flush=True)
            build_dataset(output_path=DATASET_PATH, cache_path=DATASET_CACHE_PATH)

    if os.path.isdir(DATASET_PATH):
        parquet_files = glob.glob(os.path.join(DATASET_PATH, "*.parquet"))
        if not parquet_files:
             parquet_mtime = 0
        else:
             parquet_mtime = max(os.path.getmtime(path) for path in parquet_files)
    else:
        parquet_mtime = os.path.getmtime(DATASET_PATH)

    if os.path.exists(DATASET_CACHE_PATH) and os.path.getmtime(DATASET_CACHE_PATH) >= parquet_mtime:
        print(f"🚀 Loading dataset cache from {DATASET_CACHE_PATH}...", flush=True)
        data = torch.load(DATASET_CACHE_PATH)
    else:
        print(f"🚀 Building dataset cache from {DATASET_PATH}...", flush=True)
        data = load_dataset_from_parquet(DATASET_PATH, cache_path=DATASET_CACHE_PATH)

    X, y, rets = data["X"], data["y"], data["rets"]
    
    # 2. Split Data
    dataset = TensorDataset(X, y, rets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # 3. Create Model
    input_dim = X.shape[2]
    model = create_model(
        input_dim=input_dim, 
        hidden_dim=HIDDEN_DIM, 
        layers=LAYERS, 
        seq_len=SEQ_LEN,
        model_type=model_type
    )
    model.to(device)
    
    # 4. Dynamic Batch Sizing
    batch_size = get_max_batch_size(model, input_dim, SEQ_LEN, device)
    
    train_labels = y[train_ds.indices]
    train_counts, class_weights = compute_balanced_class_weights(train_labels)
    train_sampler = build_weighted_sampler(train_labels)

    tqdm.write(
        "Class balance (train): "
        f"HOLD={int(train_counts[0].item())}, "
        f"BUY={int(train_counts[1].item())}, "
        f"SELL={int(train_counts[2].item())}"
    )
    tqdm.write(
        "Class weights: "
        f"{class_weights[0].item():.3f}, "
        f"{class_weights[1].item():.3f}, "
        f"{class_weights[2].item():.3f}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=0)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=5)
    
    # 5. Balance the loss using the actual training split
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    # Mixed Precision Setup
    dtype = torch.float32
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_autocast = USE_MIXED_PRECISION and device.type == 'cuda' and dtype != torch.float32
    scaler = torch.amp.GradScaler(device_type, enabled=use_autocast)

    tqdm.write(f"🚀 Starting training (Batch Size: {batch_size}, Precision: {dtype})...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_X, batch_y, _ in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=use_autocast):
                outputs = model(batch_X)
                outputs = outputs.clamp(min=-MAX_LOGIT_MAGNITUDE, max=MAX_LOGIT_MAGNITUDE)
                loss = criterion(outputs, batch_y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss detected at epoch {epoch+1}: {loss.item()}"
                )

            if use_autocast:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm detected at epoch {epoch+1}: {grad_norm.item()}"
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm detected at epoch {epoch+1}: {grad_norm.item()}"
                    )
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.*correct/total:.1f}%",
                "grad": f"{float(grad_norm):.2f}",
            })
            
        # Validation & Backtest
        model.eval()
        val_loss = 0
        all_preds, all_true, all_rets = [], [], []
        
        with torch.no_grad():
            for batch_X, batch_y, batch_r in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=use_autocast):
                    outputs = model(batch_X)
                    outputs = outputs.clamp(min=-MAX_LOGIT_MAGNITUDE, max=MAX_LOGIT_MAGNITUDE)
                    loss = criterion(outputs, batch_y)

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite validation loss detected at epoch {epoch+1}: {loss.item()}"
                    )
                
                val_loss += loss.item()
                
                # Apply Probability Threshold (0.6)
                probs = torch.softmax(outputs, dim=-1)
                conf, preds = torch.max(probs, dim=-1)
                
                # If confidence < 0.6, force HOLD (0)
                # This reduces noisy trades and targets high-conviction signals
                threshold = 0.6
                final_preds = preds.clone()
                mask = (conf < threshold) & (preds != 0)
                final_preds[mask] = 0
                
                all_preds.extend(final_preds.cpu().numpy())
                all_true.extend(batch_y.cpu().numpy())
                all_rets.extend(batch_r.numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate Backtest Metrics
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        all_rets = np.array(all_rets)
        
        buys = int((all_preds == 1).sum())
        sells = int((all_preds == 2).sum())
        
        buy_pnl = float(np.sum(all_rets[all_preds == 1]))
        sell_pnl = float(-np.sum(all_rets[all_preds == 2])) # Future return is inverse for SELL
        total_pnl = buy_pnl + sell_pnl
        
        buy_win_rate = float(np.sum((all_preds == 1) & (all_true == 1)) / (buys + 1e-6))
        sell_win_rate = float(np.sum((all_preds == 2) & (all_true == 2)) / (sells + 1e-6))
        
        tqdm.write(f"\n--- Epoch {epoch+1} Statistics ---")
        tqdm.write(f"Val Loss: {avg_val_loss:.4f} | Total PnL: {total_pnl:+.4f}")
        tqdm.write(f"BUYs: {buys} | PnL: {buy_pnl:+.4f} | Win Rate: {buy_win_rate:.1%}")
        tqdm.write(f"SELLs: {sells} | PnL: {sell_pnl:+.4f} | Win Rate: {sell_win_rate:.1%}")
        tqdm.write(f"Activity: {(buys+sells)/len(all_preds):.1%}")
        
        if buys + sells > 0:
            cm = confusion_matrix(all_true, all_preds, labels=[0, 1, 2])
            tqdm.write(f"Confusion Matrix (HOLD/BUY/SELL):\n{cm}")

            save_file(model.state_dict(), MODEL_OUTPUT_PATH)
            
        # Early Stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            tqdm.write(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

    print(f"✅ Training complete. Best model saved to {MODEL_OUTPUT_PATH}.")
    
    # Upload to HF
    if HF_REPO_ID and HF_TOKEN:
        try:
            print(f"📤 Uploading to HF: {HF_REPO_ID}...", flush=True)
            api = HfApi()
            create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model")
            api.upload_file(
                path_or_fileobj=MODEL_OUTPUT_PATH,
                path_in_repo="ai_fusion_bitnet.safetensors",
                repo_id=HF_REPO_ID,
                token=HF_TOKEN
            )
            print("✅ Upload successful!", flush=True)
        except Exception as e:
            print(f"⚠️ Upload failed: {e}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Trading Model")
    parser.add_argument("--model_type", type=str, default="bitnet", choices=["bitnet", "lstm"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    args = parser.parse_args()
    
    train(args)
