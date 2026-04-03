"""FinBERT sentiment analysis — lazy-loaded singleton, cached inference."""
 
from __future__ import annotations
 
import logging
import threading
from typing import Callable
 
logger = logging.getLogger(__name__)
 
_MODEL_NAME = "ProsusAI/finbert"
_LABELS = ["positive", "negative", "neutral"]
 
 
class FinBERTAnalyzer:
    """
    Lazy-loaded FinBERT wrapper.
 
    Usage:
        analyzer = FinBERTAnalyzer()
        analyzer.load(progress_callback=lambda msg: print(msg))
        results = analyzer.analyze_batch(["Apple beats earnings", "Market crashes"])
    """
 
    _instance: FinBERTAnalyzer | None = None
    _lock = threading.Lock()
 
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load_error: str | None = None
 
    @classmethod
    def get_instance(cls) -> "FinBERTAnalyzer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = FinBERTAnalyzer()
        return cls._instance
 
    @property
    def is_loaded(self) -> bool:
        return self._loaded
 
    @property
    def load_error(self) -> str | None:
        return self._load_error
 
    def load(self, progress_callback: Callable[[str], None] | None = None) -> bool:
        """
        Load model from HuggingFace Hub (or local cache).
        Returns True on success, False on failure.
        """
        if self._loaded:
            return True
 
        def _cb(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)
 
        try:
            import transformers
            transformers.logging.set_verbosity_error()
 
            _cb("Loading FinBERT tokenizer...")
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
 
            _cb("Loading FinBERT model weights (~500MB)...")
            from transformers import AutoModelForSequenceClassification
            self._model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
            self._model.eval()
 
            _cb("FinBERT ready ✓")
            self._loaded = True
            return True
 
        except Exception as exc:
            msg = f"FinBERT load failed: {exc}"
            logger.error(msg)
            self._load_error = msg
            if progress_callback:
                progress_callback(msg)
            return False
 
    def analyze_with_cache(self, headlines: list[str], conn) -> list[dict]:
        """
        Analyze headlines, checking SQLite cache first to avoid re-inference.
        Uncached headlines are batch-processed and then stored in the cache.
        """
        from trading_cli.data.db import get_cached_sentiment, cache_sentiment
 
        results: list[dict] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
 
        for i, text in enumerate(headlines):
            cached = get_cached_sentiment(conn, text)
            if cached:
                results.append(cached)
            else:
                results.append(None)  # placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)
 
        if uncached_texts:
            fresh = self.analyze_batch(uncached_texts)
            for idx, text, res in zip(uncached_indices, uncached_texts, fresh):
                results[idx] = res
                try:
                    cache_sentiment(conn, text, res["label"], res["score"])
                except Exception:
                    pass
 
        return [r or {"label": "neutral", "score": 0.5} for r in results]
 
    def analyze_batch(
        self,
        headlines: list[str],
        batch_size: int = 50,
    ) -> list[dict]:
        """
        Run FinBERT inference on a list of headlines.
 
        Returns list of {"label": str, "score": float} dicts,
        one per input headline.  Falls back to {"label": "neutral", "score": 0.5}
        if model is not loaded.
        """
        if not headlines:
            return []
        if not self._loaded:
            logger.warning("FinBERT not loaded — returning neutral for all headlines")
            return [{"label": "neutral", "score": 0.5}] * len(headlines)
 
        import torch
 
        results: list[dict] = []
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]
            try:
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                for prob_row in probs:
                    idx = int(prob_row.argmax())
                    label = self._model.config.id2label[idx].lower()
                    # Normalise label variants (ProsusAI uses "positive","negative","neutral")
                    if label not in _LABELS:
                        label = "neutral"
                    results.append({"label": label, "score": float(prob_row[idx])})
            except Exception as exc:
                logger.error("FinBERT inference error on batch %d: %s", i, exc)
                results.extend([{"label": "neutral", "score": 0.5}] * len(batch))
        return results
