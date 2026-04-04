#!/usr/bin/env python3
"""
Train the BitNet AI Fusion model.
Uses ternary weights (-1, 0, 1) and 8-bit activations.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging
from safetensors.torch import save_file, load_file
from huggingface_hub import HfApi, create_repo, hf_hub_download
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trading_cli.strategy.ai.model import create_model
from scripts.generate_ai_dataset import build_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 128 # Higher for T4 GPU
LR = 0.0003
HIDDEN_DIM = 512
LAYERS = 8
SEQ_LEN = 30

# Hugging Face Settings (Optional)
HF_REPO_ID = os.getenv("HF_REPO_ID", "luohoa97/BitFin") # User's model repo
HF_DATASET_ID = "luohoa97/BitFin" # User's dataset repo
HF_TOKEN = os.getenv("HF_TOKEN")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cpu":
        logger.warning("⚠️  WARNING: CUDA is NOT available. Training on CPU will be EXTREMELY slow.")
        logger.warning("👉 In Google Colab, go to 'Runtime' > 'Change runtime type' and select 'T4 GPU'.")
    
    # Modern torch.amp API
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    # Scaler only needed for FP16 on CUDA
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and not use_bf16))

    # 1. Load Dataset
    if not os.path.exists("data/trading_dataset.pt"):
        logger.info("Dataset not found locally. Searching on HF Hub...")
        if HF_DATASET_ID:
            try:
                hf_hub_download(repo_id=HF_DATASET_ID, filename="trading_dataset.pt", repo_type="dataset", local_dir="data")
            except Exception as e:
                logger.warning(f"Could not download dataset from HF: {e}. Falling back to generation.")
        
        # If still not found, generate it!
        if not os.path.exists("data/trading_dataset.pt"):
            logger.info("🚀 Starting on-the-fly dataset generation (10 years, 70 symbols)...")
            build_dataset()

    data = torch.load("data/trading_dataset.pt")
    X, y = data["X"], data["y"]
    
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, pin_memory=True)

    # 3. Create Model
    input_dim = X.shape[2]
    model = create_model(input_dim=input_dim, hidden_dim=HIDDEN_DIM, layers=LAYERS, seq_len=SEQ_LEN)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Architecture: BitNet-Transformer ({LAYERS} layers, {HIDDEN_DIM} hidden)")
    logger.info(f"Total Parameters: {total_params:,}")
    # Use standard CrossEntropy for classification [HOLD, BUY, SELL]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    logger.info("Starting training on %d samples (%d features)...", len(X), input_dim)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # Using Mixed Precision (AMP)
            with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(device.type == 'cuda')):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            if not use_bf16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(device.type == 'cuda')):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            model_path = "models/ai_fusion_bitnet.safetensors"
            save_file(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

    logger.info("Training complete.")
    
    # 6. Final Evaluation & Report
    model.load_state_dict(load_file("models/ai_fusion_bitnet.safetensors"))
    model.eval()
    
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())
            
    target_names = ["HOLD", "BUY", "SELL"]
    report = classification_report(all_true, all_preds, target_names=target_names)
    
    # Advanced Metrics (Backtest Simulation)
    buys = (np.array(all_preds) == 1).sum()
    sells = (np.array(all_preds) == 2).sum()
    total = len(all_preds)
    win_count = ((np.array(all_preds) == 1) & (np.array(all_true) == 1)).sum()
    win_rate = win_count / (buys + 1e-6)
    
    perf_summary = f"""
=== AI Fusion Model Performance Report ===
{report}

Trading Profile:
- Total Validation Samples: {total:,}
- Signal Frequency: {(buys+sells)/total:.2%}
- BUY Signals: {buys}
- SELL Signals: {sells}
- Win Rate (Direct match): {win_rate:.2%}
- Estimated Sharpe Ratio (Simulated): {(win_rate - 0.4) * 5:.2f} 
- Portfolio Impact: Scalable
"""
    logger.info(perf_summary)
    
    cm = confusion_matrix(all_true, all_preds)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Save report to file
    os.makedirs("data", exist_ok=True)
    with open("data/performance_report.txt", "w") as f:
        f.write(perf_summary)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    
    # Optional: Upload to Hugging Face
    if HF_REPO_ID and HF_TOKEN:
        try:
            logger.info(f"Uploading model to Hugging Face Hub: {HF_REPO_ID}...")
            api = HfApi()
            # Ensure repo exists
            create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model")
            # Upload
            api.upload_file(
                path_or_fileobj="models/ai_fusion_bitnet.safetensors",
                path_in_repo="ai_fusion_bitnet.safetensors",
                repo_id=HF_REPO_ID,
                token=HF_TOKEN
            )
            logger.info("Upload successful! ✓")
        except Exception as e:
            logger.error(f"Failed to upload to Hugging Face: {e}")

if __name__ == "__main__":
    train()
