import random

import numpy as np
import torch
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):  # @save
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    # print(f"X:{X}")
    y = torch.matmul(X, w) + b
    # print(f"y.shape:{y.shape}")
    # print(f"y:{y}")
    # 矩阵相乘 非常友好
    y += torch.normal(0, 0.01, y.shape)
    # 将 y变成一列
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2.,])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 10)

print(f"features:{features}\n labels:{labels}")


'''
1。python yield这里的话 相当于是一个迭代器， 每一次调用data——iter就返回下一个对象
2。然后这里的代码我重写了

'''


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))

    # random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# features = [11, 12, 13, 21, 22, 23, 31, 32, 33]
# labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for x, y in data_iter(3, features, labels):
    print(x, '\n', y)
