echo "🚀 Synchronizing with Hugging Face Space (luohoa97/BitFinTrainer)..."

# Use hf upload to bypass git credential issues
# This respects .gitignore and excludes heavy folders
hf upload luohoa97/BitFinTrainer . . --repo-type space \
    --exclude="data/*" \
    --exclude="models/*" \
    --exclude=".venv/*" \
    --exclude=".gemini/*" \
    --commit-message="Deploy BitNet-Transformer Trainer"

echo "✅ Finished! Your Space is building at: https://huggingface.co/spaces/luohoa97/BitFinTrainer"
