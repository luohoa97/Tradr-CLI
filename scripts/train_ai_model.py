#!/usr/bin/env python3
"""
Train the BitNet AI Fusion model.
Uses ternary weights (-1, 0, 1) and 8-bit activations.
Now includes real-time PnL backtesting and Confusion Matrix logging.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm
import logging
from safetensors.torch import save_file, load_file
from huggingface_hub import HfApi, create_repo, hf_hub_download
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trading_cli.strategy.ai.model import create_model
from scripts.generate_ai_dataset import build_dataset, SEQ_LEN, LOOKAHEAD

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 4096 # Starting point for dynamic search
LR = 0.0003
HIDDEN_DIM = 512
LAYERS = 8

# HF Configuration
HF_REPO_ID = os.getenv("HF_REPO_ID") # e.g. "username/BitFin"
HF_DATASET_ID = "luohoa97/BitFin" # User's dataset repo
HF_TOKEN = os.getenv("HF_TOKEN")

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

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 1. Load or Generate Dataset
    if not os.path.exists("data/trading_dataset.pt"):
        try:
            print("📦 Fetching dataset from Hugging Face...", flush=True)
            hf_hub_download(repo_id=HF_DATASET_ID, filename="trading_dataset.pt", local_dir="data", repo_type="dataset")
        except Exception:
            print("🚀 Starting on-the-fly dataset generation...", flush=True)
            build_dataset()

    print("🚀 Loading dataset from data/trading_dataset.pt...", flush=True)
    data = torch.load("data/trading_dataset.pt")
    X, y, rets = data["X"], data["y"], data["rets"]
    
    # 2. Split Data
    dataset = TensorDataset(X, y, rets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # 3. Create Model
    input_dim = X.shape[2]
    model = create_model(input_dim=input_dim, hidden_dim=HIDDEN_DIM, layers=LAYERS, seq_len=SEQ_LEN)
    model.to(device)
    
    # 4. Dynamic Batch Sizing
    batch_size = get_max_batch_size(model, input_dim, SEQ_LEN, device)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=True, num_workers=2)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed Precision Setup
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_bf16 = (dtype == torch.bfloat16)
    scaler = torch.amp.GradScaler(device_type, enabled=(not use_bf16 and device.type == 'cuda'))

    tqdm.write(f"🚀 Starting training (Batch Size: {batch_size}, Precision: {dtype})...")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_X, batch_y, _ in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(device.type == 'cuda')):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            if not use_bf16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})
            
        # Validation & Backtest
        model.eval()
        val_loss = 0
        all_preds, all_true, all_rets = [], [], []
        
        with torch.no_grad():
            for batch_X, batch_y, batch_r in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(device.type == 'cuda')):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(batch_y.cpu().numpy())
                all_rets.extend(batch_r.numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate Backtest Metrics
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        all_rets = np.array(all_rets)
        
        buys = int((all_preds == 1).sum())
        sells = int((all_preds == 2).sum())
        pnl = float(np.sum(all_rets[all_preds == 1]) - np.sum(all_rets[all_preds == 2]))
        win_rate = float(np.sum((all_preds == 1) & (all_true == 1)) / (buys + 1e-6))
        
        tqdm.write(f"\n--- Epoch {epoch+1} Statistics ---")
        tqdm.write(f"Val Loss: {avg_val_loss:.4f} | Total PnL: {pnl:+.4f} | Win Rate: {win_rate:.1%}")
        tqdm.write(f"Signals: {buys} BUY | {sells} SELL | Activity: {(buys+sells)/len(all_preds):.1%}")
        
        if buys + sells > 0:
            cm = confusion_matrix(all_true, all_preds, labels=[0, 1, 2])
            tqdm.write(f"Confusion Matrix (HOLD/BUY/SELL):\n{cm}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            save_file(model.state_dict(), "models/ai_fusion_bitnet.safetensors")

    print("✅ Training complete. Final model saved.")
    
    # Upload to HF
    if HF_REPO_ID and HF_TOKEN:
        try:
            print(f"📤 Uploading to HF: {HF_REPO_ID}...", flush=True)
            api = HfApi()
            create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model")
            api.upload_file(
                path_or_fileobj="models/ai_fusion_bitnet.safetensors",
                path_in_repo="ai_fusion_bitnet.safetensors",
                repo_id=HF_REPO_ID,
                token=HF_TOKEN
            )
            print("✅ Upload successful!", flush=True)
        except Exception as e:
            print(f"⚠️ Upload failed: {e}", flush=True)

if __name__ == "__main__":
    train()
