"""
Optimized data structures for enhanced performance in critical paths.
These structures are designed to reduce memory allocations and improve cache locality.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
import threading


class OptimizedCounterPool:
    """Memory pool for Counter-like objects to reduce allocations"""
    
    def __init__(self, pool_size: int = 100):
        self.pool_size = pool_size
        self.available_pools: List[Dict[Any, int]] = []
        self.lock = threading.Lock()
        
        # Pre-allocate pool
        for _ in range(pool_size):
            self.available_pools.append({})
    
    def get_counter(self) -> Dict[Any, int]:
        """Get a counter from the pool"""
        with self.lock:
            if self.available_pools:
                counter = self.available_pools.pop()
                counter.clear()  # Ensure clean state
                return counter
            else:
                return {}  # Fallback to regular dict
    
    def return_counter(self, counter: Dict[Any, int]) -> None:
        """Return a counter to the pool"""
        with self.lock:
            if len(self.available_pools) < self.pool_size:
                counter.clear()
                self.available_pools.append(counter)


class FastStringCache:
    """LRU-like cache optimized for string operations in tight loops"""
    
    def __init__(self, maxsize: int = 50000):
        self.maxsize = maxsize
        self.cache: Dict[str, str] = {}
        self.access_order: List[str] = []
        self.lock = threading.Lock()
    
    def get_or_compute(self, key: str, compute_func) -> str:
        """Get cached value or compute and cache it"""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            
            # Compute new value
            value = compute_func(key)
            
            # Add to cache
            if len(self.cache) >= self.maxsize:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
            return value


class VectorizedHashComputer:
    """Vectorized hash computation for better performance"""
    
    def __init__(self, cache_size: int = 10000):
        self.string_cache = FastStringCache(cache_size)
    
    def batch_hash_strings(self, strings: List[str], seed: int = 123) -> np.ndarray:
        """Compute hashes for a batch of strings efficiently"""
        import xxhash
        
        # Use cache for repeated strings
        hashes = []
        for s in strings:
            hash_val = self.string_cache.get_or_compute(
                f"{s}_{seed}",
                lambda x: xxhash.xxh64(s, seed=seed).hexdigest()
            )
            hashes.append(hash_val)
        
        return np.array(hashes)


class OptimizedFeatureCombiner:
    """Optimized feature combination with memory reuse"""
    
    def __init__(self, initial_capacity: int = 1000):
        self.initial_capacity = initial_capacity
        self.hash_computer = VectorizedHashComputer()
        self.temp_arrays = []  # Pre-allocated arrays for reuse
        
        # Pre-allocate some temporary arrays
        for size in [1000, 5000, 10000, 50000]:
            self.temp_arrays.append(np.empty(size, dtype=object))
    
    def get_temp_array(self, size: int) -> np.ndarray:
        """Get a temporary array of appropriate size"""
        # Find the smallest array that fits
        for arr in self.temp_arrays:
            if len(arr) >= size:
                return arr[:size]  # Return a view of the right size
        
        # If no pre-allocated array is big enough, create new one
        return np.empty(size, dtype=object)
    
    def combine_features_vectorized(
        self, 
        feature_arrays: List[np.ndarray], 
        batch_size: int = 1000
    ) -> np.ndarray:
        """Combine features with vectorized operations and memory reuse"""
        
        if not feature_arrays:
            return np.array([])
        
        array_len = len(feature_arrays[0])
        
        # Use pre-allocated temporary array if possible
        temp_array = self.get_temp_array(array_len)
        
        # Process in batches for better memory management
        combined_results = []
        
        for start_idx in range(0, array_len, batch_size):
            end_idx = min(start_idx + batch_size, array_len)
            batch_size_actual = end_idx - start_idx
            
            # Batch string concatenation
            if len(feature_arrays) == 1:
                batch_combined = feature_arrays[0][start_idx:end_idx].astype(str)
            else:
                batch_combined = np.char.add(
                    feature_arrays[0][start_idx:end_idx].astype(str),
                    feature_arrays[1][start_idx:end_idx].astype(str)
                )
                
                for i in range(2, len(feature_arrays)):
                    batch_combined = np.char.add(
                        batch_combined,
                        feature_arrays[i][start_idx:end_idx].astype(str)
                    )
            
            combined_results.append(batch_combined)
        
        # Concatenate all batches
        return np.concatenate(combined_results)


class MemoryEfficientDataFrame:
    """Memory-efficient alternative for specific DataFrame operations"""
    
    def __init__(self, data: Dict[str, np.ndarray] = None):
        self.data = data or {}
        self.index_map = {}  # Map for fast column lookup
        
        # Create index map for O(1) column access
        for i, col in enumerate(self.data.keys()):
            self.index_map[col] = i
    
    def add_column(self, name: str, values: np.ndarray):
        """Add a column efficiently"""
        self.data[name] = values
        self.index_map[name] = len(self.index_map)
    
    def get_column(self, name: str) -> np.ndarray:
        """Get column data efficiently"""
        return self.data[name]
    
    def get_columns(self, names: List[str]) -> Dict[str, np.ndarray]:
        """Get multiple columns efficiently"""
        return {name: self.data[name] for name in names if name in self.data}
    
    def memory_usage(self) -> Dict[str, int]:
        """Get memory usage information"""
        usage = {}
        for name, arr in self.data.items():
            usage[name] = arr.nbytes
        return usage


# Global instances for reuse across the application
_counter_pool = OptimizedCounterPool()
_feature_combiner = OptimizedFeatureCombiner()

def get_counter_pool() -> OptimizedCounterPool:
    """Get the global counter pool instance"""
    return _counter_pool

def get_feature_combiner() -> OptimizedFeatureCombiner:
    """Get the global feature combiner instance"""
    return _feature_combiner