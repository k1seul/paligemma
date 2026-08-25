import torch
from typing import List

class KVCache:
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def update(self, key, value, layer_idx):
        if layer_idx < len(self.key_cache):
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=2)
        elif layer_idx == len(self.key_cache):
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            raise(IndexError("New key value index must match the layer number of saved key value cache."))

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get(self, layer_idx):
        if layer_idx >= len(self.key_cache):
            raise(IndexError("Layer index must be less than the length of saved key-value cache."))

        return (self.key_cache[layer_idx], self.value_cache[layer_idx])

    @property
    def num_items(self):
        if not self.key_cache:
            return 0
        _, _, token_len, *_ = self.key_cache[0].shape

        return token_len
