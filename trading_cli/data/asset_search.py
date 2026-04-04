"""Asset search with embedding-based semantic autocomplete."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_cli.execution.adapters.alpaca import AlpacaAdapter

logger = logging.getLogger(__name__)


class AssetSearchEngine:
    """Searchable asset index with optional semantic embeddings.
    
    Supports:
    - Symbol search (e.g., "AAPL")
    - Company name search (e.g., "Apple")
    - Fuzzy/partial matching (e.g., "appl" → Apple)
    - Semantic search via embeddings (optional, requires sentence-transformers)
    """

    def __init__(self, cache_dir: Path | None = None):
        self._assets: list[dict[str, str]] = []
        self._symbol_index: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()
        self._cache_dir = cache_dir or Path.home() / ".cache" / "trading_cli"
        self._cache_file = self._cache_dir / "assets.json"
        self._embeddings = None
        self._embedding_model = None
        self._initialized = False

    def load_assets(self, adapter: AlpacaAdapter) -> int:
        """Load assets from adapter (with caching).
        
        Returns:
            Number of assets loaded.
        """
        # Try cache first
        if self._load_from_cache():
            logger.info("Loaded %d assets from cache", len(self._assets))
            self._initialized = True
            return len(self._assets)

        # Fetch from adapter
        try:
            assets = adapter.get_all_assets()
            if assets:
                with self._lock:
                    self._assets = assets
                    self._symbol_index = {
                        asset["symbol"].upper(): asset for asset in assets
                    }
                self._save_to_cache()
                logger.info("Loaded %d assets from adapter", len(assets))
                self._initialized = True
                return len(assets)
        except Exception as exc:
            logger.warning("Failed to load assets: %s", exc)

        return 0

    def _load_from_cache(self) -> bool:
        """Load cached assets. Returns True if successful."""
        if not self._cache_file.exists():
            return False
        try:
            data = json.loads(self._cache_file.read_text())
            with self._lock:
                self._assets = data["assets"]
                self._symbol_index = {
                    asset["symbol"].upper(): asset for asset in self._assets
                }
            return True
        except Exception as exc:
            logger.warning("Cache load failed: %s", exc)
            return False

    def _save_to_cache(self) -> None:
        """Save assets to cache."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(
                json.dumps({"assets": self._assets}, indent=2)
            )
        except Exception as exc:
            logger.warning("Cache save failed: %s", exc)

    def search(
        self,
        query: str,
        max_results: int = 10,
        use_semantic: bool = True,
    ) -> list[dict[str, str]]:
        """Search assets by symbol or company name.
        
        Args:
            query: Search query (symbol fragment or company name).
            max_results: Maximum number of results to return.
            use_semantic: Whether to use semantic embeddings if available.
        
        Returns:
            List of dicts with 'symbol', 'name', and optionally 'score'.
        """
        if not query.strip():
            return []

        query_upper = query.upper().strip()
        query_lower = query.lower().strip()

        results: list[dict[str, str]] = []

        with self._lock:
            # Exact symbol match (highest priority)
            if query_upper in self._symbol_index:
                asset = self._symbol_index[query_upper]
                results.append({
                    "symbol": asset["symbol"],
                    "name": asset["name"],
                    "score": 1.0,
                })
                if len(results) >= max_results:
                    return results

            # Text-based matching (symbol prefix or name substring)
            for asset in self._assets:
                symbol = asset["symbol"]
                name = asset.get("name", "")

                # Symbol starts with query
                if symbol.upper().startswith(query_upper):
                    score = 0.9 if symbol.upper() == query_upper else 0.8
                    results.append({
                        "symbol": symbol,
                        "name": name,
                        "score": score,
                    })
                    if len(results) >= max_results:
                        return results

            # Name contains query (case-insensitive)
            if len(results) < max_results and len(query_lower) >= 2:
                for asset in self._assets:
                    name = asset.get("name", "")
                    if query_lower in name.lower():
                        # Check not already in results
                        if not any(r["symbol"] == asset["symbol"] for r in results):
                            results.append({
                                "symbol": asset["symbol"],
                                "name": name,
                                "score": 0.7,
                            })
                            if len(results) >= max_results:
                                return results

        # Semantic search (optional, for fuzzy matching)
        if use_semantic and len(results) < max_results:
            semantic_results = self._search_semantic(query, max_results - len(results))
            # Merge, avoiding duplicates
            existing_symbols = {r["symbol"] for r in results}
            for sr in semantic_results:
                if sr["symbol"] not in existing_symbols:
                    results.append(sr)
                    if len(results) >= max_results:
                        break

        return results[:max_results]

    def _search_semantic(
        self,
        query: str,
        max_results: int,
    ) -> list[dict[str, str]]:
        """Search using semantic similarity (requires embeddings)."""
        if not self._embedding_model or not self._embeddings:
            return []

        try:
            # Encode query
            query_embedding = self._embedding_model.encode(
                [query],
                normalize_embeddings=True,
            )[0]

            # Compute cosine similarity
            import numpy as np
            embeddings_matrix = np.array(self._embeddings)
            similarities = embeddings_matrix @ query_embedding

            # Get top results
            top_indices = np.argsort(similarities)[::-1][:max_results]
            
            results = []
            for idx in top_indices:
                if similarities[idx] < 0.3:  # Minimum similarity threshold
                    break
                asset = self._assets[idx]
                results.append({
                    "symbol": asset["symbol"],
                    "name": asset["name"],
                    "score": float(similarities[idx]),
                })

            return results
        except Exception as exc:
            logger.warning("Semantic search failed: %s", exc)
            return []

    def load_embedding_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load a sentence transformer model for semantic search.
        
        This is optional and will only be used if successfully loaded.
        Falls back to text-based matching if unavailable.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' (80MB, fast, good quality).
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("Loading embedding model '%s'...", model_name)
            self._embedding_model = SentenceTransformer(model_name)
            
            # Precompute embeddings for all assets
            texts = [
                f"{asset['symbol']} {asset['name']}"
                for asset in self._assets
            ]
            embeddings = self._embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._embeddings = embeddings.tolist()
            logger.info(
                "Loaded embedding model: %d assets embedded",
                len(self._embeddings),
            )
        except ImportError:
            logger.info(
                "sentence-transformers not installed. "
                "Install with: uv add sentence-transformers (optional)"
            )
        except Exception as exc:
            logger.warning("Failed to load embedding model: %s", exc)

    @property
    def is_ready(self) -> bool:
        """Whether the search engine has assets loaded."""
        return self._initialized

    @property
    def has_semantic_search(self) -> bool:
        """Whether semantic search is available."""
        return self._embedding_model is not None and self._embeddings is not None
