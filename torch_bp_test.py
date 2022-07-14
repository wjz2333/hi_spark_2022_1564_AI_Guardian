# -*- coding: utf-8 -*
import numpy as np
import torch
from torch import nn
from sklearn import preprocessing
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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
new_m = torch.load('bp2.pt')



print(new_m)
# new_m.eval()  # 将模型改为预测模式
x  = [[237,	128,72],[254,128,71], [25,128,72],
    [11,129,73], [181,129,73], [231,128,73],
    [231,129,74], [228,	128,73] ]
x=np.array(x)
x = preprocessing.MinMaxScaler().fit_transform(x) #归一化
x = torch.from_numpy(x).float()
out = new_m(x[0])
print(out)

