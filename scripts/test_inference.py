import torch
from safetensors.torch import load_file
from trading_cli.strategy.ai.model import create_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_inference():
    model = create_model(input_dim=9)
    try:
        model.load_state_dict(load_file("models/ai_fusion_bitnet.safetensors"))
        model.eval()
        logger.info("Model loaded successfully ✓")
        
        # Test with random input
        x = torch.randn(1, 9)
        with torch.no_grad():
            output = model(x)
            logger.info(f"Output: {output}")
            action = torch.argmax(output, dim=-1).item()
            logger.info(f"Action: {action}")
    except Exception as e:
        logger.error(f"Inference test failed: {e}")

if __name__ == "__main__":
    test_inference()
