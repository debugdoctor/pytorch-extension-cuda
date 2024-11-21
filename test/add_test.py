import torch
import torch_tools

if __name__ == "__main__":
    # huggingface parallelism : auto device_map is supported
    a = torch.randn(4,4).cuda(1)
    b = torch.randn(4,4).cuda(1)
    c = torch.randn(4,4).cuda(2)

    print(a)
    print(b)
    print(c)
    print(torch_tools.add_two(a, b))
    print(torch_tools.add_two(b.cuda(2), c))