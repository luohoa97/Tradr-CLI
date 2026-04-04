#!/bin/bash
# Sync local code and dataset to Hugging Face
# Usage: ./scripts/sync_to_hf.sh <HF_USER/REPO_NAME>

REPO=${1:-"luohoa97/BitFin"}

echo "🚀 Preparing for HF Cloud Training on $REPO..."

# 1. Upload the training dataset to HF (as a private dataset)
echo "📦 Uploading dataset to HF Hub (luohoa97/BitFin)..."
python3 -c "from huggingface_hub import HfApi; api = HfApi(); api.upload_file(path_or_fileobj='data/trading_dataset.pt', path_in_repo='trading_dataset.pt', repo_id='$REPO', repo_type='dataset', private=False)"

# 3. Instructions for Space deployment
echo "✅ Finished! Your dataset is now at huggingface.co/datasets/$REPO"
echo "👉 Now create a Space at huggingface.co/new-space (Docker SDK)"
echo "👉 Set HF_REPO_ID=$REPO in Space Secrets"
echo "👉 Push the content of this folder (excluding data/) to the Space repo."
