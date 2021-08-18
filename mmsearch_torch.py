import torch

shape2 = (2,2,)
one_tensor = torch.ones(shape2)

# two_tensor = torch.ones_like(one_tensor)

zero_tensor = torch.zeros_like(one_tensor)

one_tensor[1:,:] = 0
print(one_tensor)

cat_tensor = torch.cat([one_tensor, one_tensor,zero_tensor],dim=1)
# print(f'cat:\n{cat_tensor}')

two_tensor = torch.tensor([[2,4],[0,0]])
thrid_tensor = torch.tensor([[4,2],[1,2]])
a = thrid_tensor.matmul(two_tensor)
print(a)
# print(one_tensor*one_tensor)
# print(two_tensor)
# print(zero_tensor)


print(f'one_tensor shape:{one_tensor.shape}, datatype:{one_tensor.dtype},device:{one_tensor.device}')

