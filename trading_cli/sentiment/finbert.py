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
        self._device = None  # Will be set to 'cuda' or 'cpu'
 
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
 
    def reload(self, progress_callback: Callable[[str], None] | None = None) -> bool:
        """
        Reset error state and attempt to load again.
        Returns True on success, False on failure.
        """
        self._loaded = False
        self._load_error = None  # Will be set by load() if it fails
        self._model = None
        self._tokenizer = None
        self._tried_fds_workaround = False  # Reset workaround flag for fresh attempt
        return self.load(progress_callback)

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
            import os
            import sys

            # Suppress warnings
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            # Disable tqdm to avoid threading issues
            os.environ["TQDM_DISABLE"] = "1"

            # Proactively set multiprocessing start method to 'spawn' on Linux
            # This prevents fds_to_keep errors in multithreaded contexts
            if sys.platform.startswith('linux'):
                try:
                    import multiprocessing
                    try:
                        multiprocessing.set_start_method('spawn', force=True)
                        logger.info("Set multiprocessing method to 'spawn'")
                    except RuntimeError:
                        pass  # Already set, which is fine
                except (ImportError, AttributeError):
                    pass

            import transformers
            transformers.logging.set_verbosity_error()

            # Auto-detect device
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
                _cb(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
                _cb("Using Apple Metal (MPS)")
            elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
                self._device = "cuda"  # ROCm uses cuda device type
                _cb("Using AMD ROCm GPU")
            else:
                self._device = "cpu"
                # Enable multi-threaded CPU inference for Intel/AMD CPUs
                # Don't restrict threads - let PyTorch use available cores
                _cb(f"Using CPU ({torch.get_num_threads()} threads)")

            _cb("Loading FinBERT tokenizer...")
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_NAME,
                use_fast=True,  # Fast tokenizer is much quicker
            )

            _cb("Loading FinBERT model weights (~500MB)...")
            from transformers import AutoModelForSequenceClassification

            # Use low_cpu_mem_usage for faster loading with meta tensors
            # device_map="auto" will place model on GPU if available
            if self._device == "cuda":
                device_map = "auto"
            else:
                device_map = None

            self._model = AutoModelForSequenceClassification.from_pretrained(
                _MODEL_NAME,
                low_cpu_mem_usage=True,  # Faster loading with meta tensors
                device_map=device_map,
            )
            self._model.eval()

            # Move to device if not using device_map
            if device_map is None:
                self._model = self._model.to(self._device)

            _cb(f"FinBERT ready on {self._device.upper()} ✓")
            self._loaded = True
            return True

        except Exception as exc:
            import traceback
            msg = f"FinBERT load failed: {exc}"
            logger.error(msg)
            logger.debug("Load traceback:\n%s", traceback.format_exc())
            self._load_error = msg
            if progress_callback:
                progress_callback(msg)

            # If it's the fds_to_keep error, try once more with additional workarounds
            if "fds_to_keep" in str(exc) and not getattr(self, '_tried_fds_workaround', False):
                self._tried_fds_workaround = True
                logger.info("Attempting retry with fds_to_keep workaround...")
                # Preserve original error if workaround also fails
                original_error = msg
                success = self._load_with_fds_workaround(progress_callback)
                if not success and not self._load_error:
                    self._load_error = original_error
                return success

            return False
    
    def _load_with_fds_workaround(self, progress_callback) -> bool:
        """Fallback loading method with additional workarounds for fds_to_keep error."""
        if self._loaded:
            return True

        def _cb(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        try:
            import os
            import sys

            # Suppress warnings
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            # Critical: Disable multiprocessing in tokenizers to avoid fds_to_keep errors
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            # Disable tqdm to avoid threading issues
            os.environ["TQDM_DISABLE"] = "1"

            # Try multiple strategies for fds_to_keep error
            
            # Strategy 1: Set multiprocessing start method to 'spawn'
            # This avoids inheriting file descriptors from parent process
            try:
                import multiprocessing
                if sys.platform.startswith('linux'):
                    # Only set on Linux where the issue occurs
                    try:
                        multiprocessing.set_start_method('spawn', force=True)
                        _cb("Set multiprocessing method to 'spawn'")
                    except RuntimeError:
                        pass  # Already set, which is fine
            except (ImportError, AttributeError):
                pass

            # Strategy 2: Lower file descriptor limit if it's very high
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                _cb(f"Current file descriptor limit: soft={soft}, hard={hard}")
                # If the soft limit is extremely high, reduce it to avoid issues
                if soft > 1048576:  # > 1M file descriptors
                    new_limit = min(soft, 65536)
                    resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
                    _cb(f"Lowered file descriptor limit from {soft} to {new_limit}")
            except (ImportError, ValueError, OSError) as e:
                logger.debug(f"Could not adjust file descriptor limit: {e}")

            import transformers
            transformers.logging.set_verbosity_error()

            # Auto-detect device (same as main load)
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
                # Explicitly set thread count for CPU
                torch.set_num_threads(min(torch.get_num_threads(), 4))
                _cb(f"Retrying FinBERT load on {self._device.upper()} ({torch.get_num_threads()} threads)")

            _cb(f"Retrying FinBERT load on {self._device.upper()}...")

            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # Use fast tokenizer and optimized loading
            self._tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_NAME,
                use_fast=True,
            )

            # Use device_map for auto placement
            device_map = "auto" if self._device == "cuda" else None
            self._model = AutoModelForSequenceClassification.from_pretrained(
                _MODEL_NAME,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
            self._model.eval()

            if device_map is None:
                self._model = self._model.to(self._device)

            _cb(f"FinBERT ready on {self._device.upper()} ✓")
            self._loaded = True
            return True

        except Exception as exc:
            msg = f"FinBERT load failed (workaround attempt): {exc}"
            logger.error(msg)
            self._load_error = msg
            if progress_callback:
                progress_callback(msg)
            # Log additional context for debugging
            import traceback
            logger.debug("Workaround load traceback:\n%s", traceback.format_exc())
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
                ).to(self._device)  # Move inputs to correct device
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
