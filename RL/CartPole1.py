import time

import gym
import torch

a = torch.arange(12)

a.numel()

print(a.numel())

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b

# env = gym.make("CartPole-v0")
#
# env.reset()
# env.render()


