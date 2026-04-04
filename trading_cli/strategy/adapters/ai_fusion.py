"""AI Fusion Strategy — BitNet-optimized adaptive trading."""

from __future__ import annotations

import logging
import os
import torch
import pandas as pd
import numpy as np

from trading_cli.strategy.adapters.base import SignalResult, StrategyAdapter, StrategyInfo
from trading_cli.strategy.adapters.registry import register_strategy
from trading_cli.strategy.ai.model import create_model
from safetensors.torch import load_file
from trading_cli.strategy.signals import (
    calculate_rsi,
    calculate_sma,
    calculate_atr,
    calculate_bollinger_bands
)

logger = logging.getLogger(__name__)


@register_strategy
class AIFusionStrategy(StrategyAdapter):
    """AI Fusion Strategy.

    Uses a ternary-quantized BitNet model to fuse technical and sentiment data.
    Adapts to market conditions by learning patterns from historical data.
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.model = None
    def _load_model(self):
        """Lazy load the BitNet model."""
        try:
            # [rsi2, rsi14, dist_sma20, dist_sma50, dist_sma200, bb_pos, atr_pct, vol_ratio, sentiment]
            input_dim = 9
            self.model = create_model(input_dim=input_dim, hidden_dim=512, layers=8, seq_len=30)
            
            # Prefer safetensors
            st_path = self.model_path.replace(".pt", ".safetensors")
            if os.path.exists(st_path):
                self.model.load_state_dict(load_file(st_path, device="cpu"))
                self.model.eval()
                logger.info("AI Fusion BitNet model loaded (safetensors) ✓")
            elif os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
                self.model.eval()
                logger.info("AI Fusion BitNet model loaded (legacy .pt) ✓")
            else:
                logger.warning("AI Fusion model file not found. Run training first.")
        except Exception as exc:
            logger.error("Failed to load AI Fusion model: %s", exc)
            self.model = None

    @property
    def strategy_id(self) -> str:
        return "ai_fusion"

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name="AI Fusion (BitNet Ternary)",
            description=(
                "Ultra-efficient AI strategy using BitNet (ternary weights). "
                "Fuses 8 technical indicators with real-time sentiment analysis. "
                "Learns non-linear market regimes for better adaptation."
            ),
            params_schema={
                "model_path": {"type": "str", "default": "models/ai_fusion_bitnet.pt", "desc": "Path to .pt model"},
            },
        )

    def generate_signal(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        sentiment_score: float = 0.0,
        **kwargs,
    ) -> SignalResult:
        if self.model is None:
            return SignalResult(symbol, "HOLD", 0.0, 0.0, "AI model not loaded")

        try:
            # 1. Extract Features (Same logic as generate_ai_dataset.py)
            close = ohlcv["close" if "close" in ohlcv.columns else "Close"]
            if len(close) < 200:
                return SignalResult(symbol, "HOLD", 0.0, 0.0, "insufficient data")

            # Technicals
            r2 = calculate_rsi(close, 2).iloc[-1] / 100.0
            r14 = calculate_rsi(close, 14).iloc[-1] / 100.0
            s20 = calculate_sma(close, 20)
            s50 = calculate_sma(close, 50)
            s200 = calculate_sma(close, 200)
            
            d20 = float((close.iloc[-1] / s20.iloc[-1]) - 1.0)
            d50 = float((close.iloc[-1] / s50.iloc[-1]) - 1.0)
            d200 = float((close.iloc[-1] / s200.iloc[-1]) - 1.0)
            
            up, mid, lo = calculate_bollinger_bands(close, 20, 2.0)
            # Ensure we get scalars
            last_up = up.iloc[-1]
            # 2. Build Sequence
            # We need the last 30 days of OHLCV to compute features
            if len(ohlcv) < 60: # 30 for sequence + room for indicators
                return SignalResult(symbol, "HOLD", 0.0, 0.0, "Insufficient data history")
                
            # Predict for the last SEQ_LEN days
            last_30 = ohlcv.tail(60) 
            
            # Helper to generate features for a dataframe
            def generate_features(df):
                close = df["close" if "close" in df.columns else "Close"]
                r2 = calculate_rsi(close, 2) / 100.0
                r14 = calculate_rsi(close, 14) / 100.0
                s20 = calculate_sma(close, 20)
                s50 = calculate_sma(close, 50)
                s200 = calculate_sma(close, 200)
                d20 = (close / s20) - 1.0
                d50 = (close / s50) - 1.0
                d200 = (close / s200) - 1.0
                up, mid, lo = calculate_bollinger_bands(close, 20, 2.0)
                bbp = (close - lo) / (up - lo + 1e-6)
                atr = calculate_atr(df, 14)
                atrp = atr / close
                vol = df["volume" if "volume" in df.columns else "Volume"]
                vsma = vol.rolling(20).mean()
                vr = (vol / (vsma + 1e-6)).clip(0, 5) / 5.0
                return pd.DataFrame({
                    "r2": r2, "r14": r14, "d20": d20, "d50": d50, "d200": d200,
                    "bbp": bbp, "atrp": atrp, "vr": vr
                })

            full_features = generate_features(last_30)
            full_features["sentiment"] = sentiment_score
            full_features = full_features.dropna().tail(30)
            
            if len(full_features) < 30:
                 return SignalResult(symbol, "HOLD", 0.0, 0.0, "Insufficient data after indicator calculation")
            
            input_tensor = torch.tensor(full_features.values, dtype=torch.float32).unsqueeze(0)
            
            # 3. Inference
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=-1)
                action_idx = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, action_idx].item()

            action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
            action = action_map[action_idx]

            reason = f"AI Prediction: {action} (conf={confidence:.1%}, sent={sentiment_score:+.2f})"
            
            return SignalResult(
                symbol=symbol,
                action=action,
                confidence=confidence,
                score=float(logits[0, action_idx]),
                reason=reason,
                metadata={"probs": probs.tolist(), "regime": "AI-detected"}
            )

        except Exception as exc:
            logger.error("AI Fusion inference error: %s", exc)
            return SignalResult(symbol, "HOLD", 0.0, 0.0, f"Inference error: {exc}")
