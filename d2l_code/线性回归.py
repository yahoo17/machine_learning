import numpy as np
import torch
from d2l import torch as d2l

# 对于每一个小批量，我们会进行以下步骤:
#
# 通过调用net(X)生成预测并计算损失l（正向传播）。
# 通过进行反向传播来计算梯度。
# 通过调用优化器来更新模型参数。
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


true_w = torch.tensor([2.,-3.4])
true_b = 4.
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


def linereg(X, w, b):
    return  torch.matmul(X,w) + b


def squared_loss(y_hat, y):
    # 均方损失
    return (y_hat - y.reshape(y_hat.shape)) **2 / 2

# torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
def sgd(params, lr, batch_size):
    ## 小批量梯度
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


w = torch.normal(0,0.01, size=(2,1), requires_grad=True)

b = torch.zeros(1, requires_grad=True)


lr = 0.03
num_epochs = 100
net = linereg
loss = squared_loss
batch_size = 10
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = net(X,w,b)
        l = loss(y_hat,y)

        l.sum().backward()

        sgd([w,b],lr,batch_size)

        with torch.no_grad():
            train_l = loss(net(features,w,b), labels)
            print(f"epoch{epoch+1}, loss:{float(train_l.mean()):f}")


print(f"我们估计的w：{w},实际的w：{w} ，w的估计误差：{true_w - w.reshape(true_w.shape)}")
print(f"我们估计的b：{b},实际的b：{b}, b的估计误差：{true_b - b}")