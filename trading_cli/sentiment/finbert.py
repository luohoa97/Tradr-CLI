"""FinBERT sentiment analysis — lazy-loaded singleton, cached inference."""
 
from __future__ import annotations
 
import logging
import threading
from typing import Callable
 
logger = logging.getLogger(__name__)
 
# File descriptor limit is set in __main__.py at startup
# This module-level code is kept for backward compatibility when imported directly
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_limit = 256
    if soft > target_limit:
        new_soft = min(target_limit, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        logger.info(f"Auto-adjusted file descriptor limit from {soft} to {new_soft}")
except Exception as e:
    if logger:
        logger.debug(f"Could not adjust file descriptor limit: {e}")
 
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
        self._device: str = "cpu"
        self._tried_fds_workaround: bool = False
 
    @classmethod
    def get_instance(cls) -> FinBERTAnalyzer:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = FinBERTAnalyzer()
        assert cls._instance is not None
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

            # Suppress warnings
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            # Disable tqdm to avoid threading issues
            os.environ["TQDM_DISABLE"] = "1"

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
            # CRITICAL: Do NOT use device_map="auto" as it can trigger subprocess issues
            # Instead, load on CPU first, then move to device manually
            self._model = AutoModelForSequenceClassification.from_pretrained(
                _MODEL_NAME,
                low_cpu_mem_usage=True,
                device_map=None,  # Avoid subprocess spawning
                # Disable features that might use subprocesses
                trust_remote_code=False,
            )
            self._model.eval()

            # Move to device after loading
            self._model = self._model.to(self._device)

            _cb(f"FinBERT ready on {self._device.upper()} ✓")
            self._loaded = True
            return True

        except Exception as exc:
            import traceback
            import sys as sys_mod
            full_traceback = traceback.format_exc()
            msg = f"FinBERT load failed: {exc}"
            logger.error(msg)
            logger.error("Full traceback:\n%s", full_traceback)
            self._load_error = msg
            if progress_callback:
                progress_callback(msg)

            # If it's the fds_to_keep error, try once more with additional workarounds
            if "fds_to_keep" in str(exc) and not getattr(self, '_tried_fds_workaround', False):
                self._tried_fds_workaround = True
                logger.info("Attempting retry with fds_to_keep workaround...")
                logger.info("Original traceback:\n%s", full_traceback)
                # Preserve original error if workaround also fails
                original_error = msg
                success = self._load_with_fds_workaround(progress_callback)
                if not success and not self._load_error:
                    # Add helpful context about Python version
                    python_version = sys_mod.version
                    self._load_error = (
                        f"{original_error}\n"
                        f"\n"
                        f"This is a known issue with Python 3.12+ and transformers.\n"
                        f"Your Python version: {python_version}\n"
                        f"\n"
                        f"To fix this, consider:\n"
                        f"  1. Downgrade to Python 3.11 (recommended)\n"
                        f"  2. Or upgrade transformers: pip install -U transformers>=4.45.0\n"
                        f"  3. Or use the --no-sentiment flag to skip FinBERT loading"
                    )
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

            # Suppress warnings
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["TQDM_DISABLE"] = "1"

            # Try to lower file descriptor limit if it's very high
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                _cb(f"Current file descriptor limit: soft={soft}, hard={hard}")
                # Force lower limit for workaround attempt - must be very low for Python 3.14
                target_limit = 128
                if soft > target_limit:
                    new_soft = min(target_limit, hard)
                    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
                    _cb(f"Lowered file descriptor limit from {soft} to {new_soft} (emergency fallback)")
            except (ImportError, ValueError, OSError) as e:
                logger.debug(f"Could not adjust file descriptor limit: {e}")

            import transformers
            transformers.logging.set_verbosity_error()

            # Auto-detect device
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
                # Limit CPU threads for more stable loading
                torch.set_num_threads(min(torch.get_num_threads(), 4))

            _cb(f"Retrying FinBERT load on {self._device.upper()} ({torch.get_num_threads()} threads)...")

            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # Use fast tokenizer and optimized loading
            # Disable subprocess-based tokenization
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self._tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_NAME,
                use_fast=True,
            )

            # Use device_map for auto placement
            # For Python 3.14+, avoid using device_map="auto" which can trigger subprocess issues
            device_map = None
            self._model = AutoModelForSequenceClassification.from_pretrained(
                _MODEL_NAME,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
            self._model.eval()

            # Manually move to device
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
            
            # If still failing with fds_to_keep, try one more time with subprocess isolation
            if "fds_to_keep" in str(exc):
                logger.info("Attempting final retry with subprocess isolation...")
                return self._load_with_subprocess_isolation(progress_callback)
            
            return False

    def _load_with_subprocess_isolation(self, progress_callback) -> bool:
        """Final attempt: load model with maximum subprocess isolation for Python 3.14+."""
        if self._loaded:
            return True

        def _cb(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        try:
            import os
            import subprocess
            import sys

            # Set maximum isolation before loading
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["TQDM_DISABLE"] = "1"
            
            # Additional isolation for Python 3.14
            os.environ["RAYON_RS_NUM_CPUS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "1"

            # Force file descriptor limit to minimum
            try:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
                _cb("Set file descriptor limit to 64 (maximum isolation)")
            except Exception:
                pass

            import transformers
            transformers.logging.set_verbosity_error()

            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
                torch.set_num_threads(1)  # Single thread for maximum isolation

            _cb(f"Loading with subprocess isolation on {self._device.upper()}...")

            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # Use slow tokenizer to avoid Rust subprocess issues
            self._tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_NAME,
                use_fast=False,  # Use slow tokenizer
            )

            self._model = AutoModelForSequenceClassification.from_pretrained(
                _MODEL_NAME,
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            self._model = self._model.to(self._device)

            _cb(f"FinBERT ready on {self._device.upper()} ✓")
            self._loaded = True
            return True

        except Exception as exc:
            msg = f"FinBERT load failed (subprocess isolation): {exc}"
            logger.error(msg)
            self._load_error = msg
            if progress_callback:
                progress_callback(msg)
            import traceback
            logger.debug("Subprocess isolation traceback:\n%s", traceback.format_exc())
            
            # Add helpful context
            import sys as sys_mod
            python_version = sys_mod.version
            self._load_error = (
                f"{msg}\n"
                f"\n"
                f"This is a known compatibility issue between Python 3.12+ and the transformers library.\n"
                f"Your Python version: {python_version}\n"
                f"\n"
                f"To resolve this issue:\n"
                f"  1. Downgrade to Python 3.11 (most reliable solution)\n"
                f"     - Use pyenv: pyenv install 3.11 && pyenv local 3.11\n"
                f"  2. Or upgrade to the latest transformers: pip install -U transformers\n"
                f"     - Note: As of now, you have transformers 5.5.0\n"
                f"  3. Or run with sentiment disabled: trading-cli --no-sentiment\n"
                f"\n"
                f"The app will continue without sentiment analysis."
            )
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
