"""KV cache utilities: estimates and a safe paged block allocator."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


def estimate_kv_cache_bytes(
    *,
    num_layers: int,
    batch_size: int,
    num_key_value_heads: int,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    """Estimate K+V storage; excludes allocator and tensor metadata overhead."""
    dimensions = (num_layers, batch_size, num_key_value_heads, sequence_length, head_dim)
    if any(size <= 0 for size in dimensions):
        raise ValueError("all cache dimensions must be positive")
    element_size = torch.empty((), dtype=dtype).element_size()
    return 2 * num_layers * batch_size * num_key_value_heads * sequence_length * head_dim * element_size


@dataclass
class PagedKVCache:
    """Fixed-size KV pages with per-request block tables and explicit release.

    Tensor layout is ``(layers, K_or_V, blocks, kv_heads, block_size, head_dim)``.
    This simple educational implementation gathers a request's pages to a
    contiguous sequence when read; production engines use fused paged kernels.
    """

    num_layers: int
    num_blocks: int
    num_key_value_heads: int
    block_size: int
    head_dim: int
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float16
    free_blocks: list[int] = field(init=False)
    request_blocks: dict[str, list[int]] = field(init=False, default_factory=dict)
    lengths: dict[str, int] = field(init=False, default_factory=dict)
    storage: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if min(self.num_layers, self.num_blocks, self.num_key_value_heads,
               self.block_size, self.head_dim) <= 0:
            raise ValueError("cache dimensions must be positive")
        self.storage = torch.empty(
            self.num_layers, 2, self.num_blocks, self.num_key_value_heads,
            self.block_size, self.head_dim, device=self.device, dtype=self.dtype,
        )
        self.free_blocks = list(range(self.num_blocks - 1, -1, -1))

    @property
    def free_count(self) -> int:
        return len(self.free_blocks)

    def create_request(self, request_id: str) -> None:
        if request_id in self.request_blocks:
            raise ValueError(f"request already exists: {request_id}")
        self.request_blocks[request_id] = []
        self.lengths[request_id] = 0

    def append(self, request_id: str, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append one token across all layers; K/V shape is (L, Hkv, D)."""
        if request_id not in self.request_blocks:
            raise KeyError(f"unknown request: {request_id}")
        expected = (self.num_layers, self.num_key_value_heads, self.head_dim)
        if keys.shape != expected or values.shape != expected:
            raise ValueError(f"each token's K/V must have shape {expected}")
        offset = self.lengths[request_id]
        page_offset = offset % self.block_size
        if page_offset == 0:
            if not self.free_blocks:
                raise MemoryError("paged KV cache is full")
            block = self.free_blocks.pop()
            self.request_blocks[request_id].append(block)
        block = self.request_blocks[request_id][-1]
        self.storage[:, 0, block, :, page_offset, :].copy_(keys)
        self.storage[:, 1, block, :, page_offset, :].copy_(values)
        self.lengths[request_id] = offset + 1

    def read(self, request_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        if request_id not in self.request_blocks:
            raise KeyError(f"unknown request: {request_id}")
        length = self.lengths[request_id]
        if not length:
            empty = self.storage.new_empty(self.num_layers, self.num_key_value_heads, 0, self.head_dim)
            return empty, empty.clone()
        blocks = self.request_blocks[request_id]
        k = self.storage[:, 0, blocks].permute(0, 2, 1, 3, 4).reshape(
            self.num_layers, self.num_key_value_heads, -1, self.head_dim
        )[:, :, :length]
        v = self.storage[:, 1, blocks].permute(0, 2, 1, 3, 4).reshape(
            self.num_layers, self.num_key_value_heads, -1, self.head_dim
        )[:, :, :length]
        return k, v

    def release(self, request_id: str) -> None:
        if request_id not in self.request_blocks:
            return
        self.free_blocks.extend(self.request_blocks.pop(request_id))
        del self.lengths[request_id]
