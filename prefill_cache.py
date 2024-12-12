from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from model import Transformer, Attention, KVCache


@dataclass
class PrefillCacheContext:
    context: str
    context_token_size: int
    model: Transformer


class PrefillCache:

    def __init__(self, model_device: str, cache_device: str, cache_size: int):
        self._cache: Dict[str, List[Tensor]] = {}
        self._model_device = model_device
        self._cache_device = cache_device
        self._cache_size = cache_size
        self._ctx: Optional[PrefillCacheContext] = None

    def get_context_token_size(self) -> Optional[int]:
        return self._ctx.context_token_size if self._ctx is not None else None

    def _enforce_cache_size_limit(self) -> None:
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

    def insert(self, key: str, tensors: List[Tensor]) -> None:
        assert isinstance(key, str)
        self._enforce_cache_size_limit()
        self._cache[key] = tensors

    def get(self, key: str) -> Optional[List[Tensor]]:
        return self._cache.get(key)

    def exists(self, key: str) -> bool:
        return key in self._cache

    def _extract_kv_caches(self, model: Transformer) -> List[Tensor]:
        cached_tensors = []
        for layer in model.layers:
            attn: Attention = layer.attention
            kv_cache: KVCache = attn.kv_cache
            cached_tensors.extend(
                [
                    kv_cache.k_cache.clone().to(self._cache_device),
                    kv_cache.v_cache.clone().to(self._cache_device),
                ]
            )
        return cached_tensors

    def insert_with_transformer(self, key: str, model: Transformer) -> None:
        cached_tensors = self._extract_kv_caches(model)
        self.insert(key, cached_tensors)

    def load_to_transformer(self, key: str, model: Transformer) -> None:
        cached_tensors = self.get(key)
        if cached_tensors is None:
            return

        for i, layer in enumerate(model.layers):
            kv_cache = layer.attention.kv_cache
            k_idx, v_idx = i * 2, i * 2 + 1

            # In-place update of cache tensors
            kv_cache.k_cache[:, :, :, :] = cached_tensors[k_idx][:, :, :, :]
            kv_cache.v_cache[:, :, :, :] = cached_tensors[v_idx][:, :, :, :]

    @property
    def context(self) -> Optional[PrefillCacheContext]:
        return self._ctx

    def set_context(self, ctx: PrefillCacheContext) -> None:
        self._ctx = ctx

    def need_to_prefill(self) -> bool:
        return self._ctx is not None and not self.exists(self._ctx.context)

    def save(self) -> None:
        if self._ctx is not None:
            self.insert_with_transformer(self._ctx.context, self._ctx.model)

    def load(self) -> None:
        if self._ctx is not None:
            self.load_to_transformer(self._ctx.context, self._ctx.model)
