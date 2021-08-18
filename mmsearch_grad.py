import torch

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

a = torch.tensor([2.,3], requires_grad=True)

b = torch.tensor([6.,4],requires_grad=True)

Q = 3*a**3 - b**2

extern_grad = torch.tensor([1.,1])

Q.backward(gradient=extern_grad)

print(a.grad)
print(9*a*a)
print(extern_grad)