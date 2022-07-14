# -*- coding: utf-8 -*
import numpy as np
from sklearn import preprocessing
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
x = [[237,	128,72],[254,128,71], [25,128,72],
    [11,129,73], [181,129,73], [231,128,73],
    [231,129,74], [228,	128,73]]
# x = [[1.0, 0.3393939393939394, 0.0],
#      [1.0, 0.3114754098360656, 0.0],
#      [0.0, 1.0, 0.4563106796116505],
#      [0.0, 1.0, 0.5254237288135594],
#      [1.0, 0.5185185185185185, 0.0],
#      [1.0, 0.34810126582278483, 0.0],
#      [1.0, 0.3503184713375796, 0.0],
#     [1.0, 0.3548387096774194, 0.0]]
# 0.3114754098360656
# 0.0]
A=[]
for j in range(0,8):
    for i in range(0,3):
        a=x[j]
        b=(a[i]-min(a))/(max(a)-min(a))
        print(b)
        A.append(b)
        print(A)
y= [[25],[48],[65],[80],[40],[20],[50],[45]]
# x = [
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1],
#         ]
# y=[[0], [1], [1], [0]]
x=np.array(x)
x = preprocessing.MinMaxScaler().fit_transform(x) #归一化
print(x)
y=np.array(y)
# y = preprocessing.MinMaxScaler().fit_transform(y)
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
train_data = zip(x, y)
test_data = zip(x, y)
train_data = DataLoader(list(train_data), batch_size=32, shuffle=True)  # 训练数据
test_data  = DataLoader(list(test_data), batch_size=64, shuffle=False)  # 测试数据

class BPNNModel(torch.nn.Module):
    def __init__(self):
        # 调用父类的初始化函数，必须要的
        super(BPNNModel, self).__init__()

        # 创建四个Sequential对象，Sequential是一个时序容器，将里面的小的模型按照序列建立网络
        self.layer1 = nn.Sequential(nn.Linear(3, 9), nn.ReLU())
        # self.layer2 = nn.Sequential(nn.Linear(9, 12), nn.ReLU())
        # self.layer3 = nn.Sequential(nn.Linear(12, 15), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(9, 1))

    def forward(self, img):
        # 每一个时序容器都是callable的，因此用法也是一样。
        img = self.layer1(img)
        # img = self.layer2(img)
        # img = self.layer3(img)
        img = self.layer4(img)
        return img


# 创建和实例化一个整个模型类的对象
model = BPNNModel()
# 打印出整个模型
print(model)




# Step 3:============================定义损失函数和优化器===================
criterion = nn.MSELoss()
# 我们优先使用随机梯度下降，lr是学习率:
optimizer = torch.optim.SGD(model.parameters(), 3e-3)






# Step 4:============================开始训练网络===================
# 为了实时观测效果，我们每一次迭代完数据后都会，用模型在测试数据上跑一次，看看此时迭代中模型的效果。
# 用数组保存每一轮迭代中，训练的损失值和精确度，也是为了通过画图展示出来。
train_losses = []
train_acces = []
# 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
eval_losses = []
eval_acces = []

for e in range(10000):


    # 4.1==========================训练模式==========================
    train_loss = 0
    train_acc = 0
    model.train()   # 将模型改为训练模式

    # 每次迭代都是处理一个小批量的数据，batch_size是64
    for im, label in train_data:

        # 计算前向传播，并且得到损失函数的值
        out = model(im)
        loss = criterion(out, label)

        # 反向传播，记得要把上一次的梯度清0，反向传播，并且step更新相应的参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录误差
        train_loss += loss.item()

    train_losses.append(train_loss / len(train_data))
    torch.save(model, 'bp2.pt')


    # 4.2==========================每次进行完一个训练迭代，就去测试一把看看此时的效果==========================
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    model.eval()  # 将模型改为预测模式

# x = [[23,	12,7],]
# x=np.array(x)
# x = torch.from_numpy(x).float()
# out = model(x)
# print(out)

    # 每次迭代都是处理一个小批量的数据，batch_size是128
    for im, label in test_data:
        im = Variable(im)  # torch中训练需要将其封装即Variable，此处封装像素即784
        label = Variable(label)  # 此处为标签

        out = model(im)  # 经网络输出的结果
        # label = label.unsqueeze(1)
        loss = criterion(out, label)  # 得到误差

        # 记录误差
        eval_loss += loss.item()

    eval_losses.append(eval_loss / len(test_data))
    # eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f},Eval Loss: {:.6f}'
          .format(e, train_loss / len(train_data),eval_loss / len(test_data)))
plt.title('train loss')
plt.plot(np.arange(len(train_losses)), train_losses)
plt.plot(np.arange(len(train_acces)), train_acces)
plt.title('train acc')
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.title('test loss')
plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('test acc')
plt.show()
for i in range(8):
    out = model(x[i, :])
    print("predict:","   ",out.detach().numpy())
