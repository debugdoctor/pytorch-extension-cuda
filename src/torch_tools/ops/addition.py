import torch
from .. import _C

def add_two(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

    assert a.shape == b.shape, "a and b are must be the same shape, but got the shapes {} and {}".format(a.shape, b.shape)

    return _C.add(a, b)