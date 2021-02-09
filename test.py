import torch

a = torch.arange(0,9)
b = torch.zeros(9, dtype=torch.bool)
b[0] = True

print(a)
print(a[b])