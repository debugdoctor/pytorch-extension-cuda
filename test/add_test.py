import torch
import torch_tools

if __name__ == "__main__":
    a = torch.randn(4,4)
    b = torch.randn(4,4)

    print(a)
    print(b)
    print(torch_tools.add_two(a, b))