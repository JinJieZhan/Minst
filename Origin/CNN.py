# minist_test.py
import torch
import torchvision
from torch import optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST('../data/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = torchvision.datasets.MNIST(root='../data/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# 模型类设计
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)  # n
        x = F.relu(self.pooling(self.conv1(x)))  # 第一轮卷积池化激活
        x = F.relu(self.pooling(self.conv2(x)))  # 第二轮卷积池化激活
        x = x.view(batch_size, -1)  # 吧数据变成线性层接受的
        x = self.fc(x)  # 去线性层
        return x


model = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将训练模型放到GPU
print(device)
# 损失函数
criterion = torch.nn.CrossEntropyLoss()
# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    runing_loss = 0.0
    for i, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        # 清零 正向传播  损失函数  反向传播 更新
        optimizer.zero_grad()
        y_pre = model(x)
        loss = criterion(y_pre, y)
        loss.backward()
        optimizer.step()
        runing_loss += loss.item()
    # 每轮训练一共训练1W个样本，这里的runing_loss是1W个样本的总损失值，要看每一个样本的平均损失值， 记得除10000

    print("这是第 %d轮训练，当前损失值 %.5f" % (epoch + 1, runing_loss / 10000))


def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            pre_y = model(x)
            # 这里拿到的预测值 每一行都对应10个分类，这10个分类都有对应的概率，
            # 我们要拿到最大的那个概率和其对应的下标。
            j, pre_y = torch.max(pre_y.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度

            total += y.size(0)  # 统计方向0上的元素个数 即样本个数
            correct += (pre_y == y).sum().item()  # 张量之间的比较运算
    print("第%d轮测试结束，当前正确率:%f %%" % (epoch + 1, correct / total * 100))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test(epoch)
