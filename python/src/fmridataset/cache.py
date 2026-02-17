"""Lightweight LRU cache wrapper.

Uses ``cachetools`` if available, falls back to ``functools.lru_cache``.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:
    from cachetools import LRUCache

    _HAS_CACHETOOLS = True
except ImportError:  # pragma: no cover
    _HAS_CACHETOOLS = False


_DEFAULT_MAXSIZE = 128


def lru_cache(maxsize: int = _DEFAULT_MAXSIZE) -> Callable[[F], F]:
    """Decorator: LRU cache backed by *cachetools* or stdlib fallback."""
    if _HAS_CACHETOOLS:

        def decorator(fn: F) -> F:
            cache: LRUCache[Any, Any] = LRUCache(maxsize=maxsize)

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                key = (args, tuple(sorted(kwargs.items())))
                try:
                    return cache[key]
                except KeyError:
                    result = fn(*args, **kwargs)
                    cache[key] = result
                    return result

            wrapper.cache = cache  # type: ignore[attr-defined]
            wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
            return wrapper  # type: ignore[return-value]

        return decorator
    else:
        return functools.lru_cache(maxsize=maxsize)  # type: ignore[return-value]
