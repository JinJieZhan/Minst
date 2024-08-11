import torch
import torch.nn.functional as F
from itertools import product
from InPytorch.Vit import ViT
from InPytorch.dataset import MNIST_Train


# 设定设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义模型评估函数
def evaluate(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 累加损失
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)  # 平均损失
    accuracy = correct / len(dataloader.dataset)  # 计算准确率
    return test_loss, accuracy

bool_values = [True, False]
combinations = list(product(bool_values, repeat=3))

for combination in combinations:
    flag_value, CLSisUsed, PostionisUsed = combination
    print(f"flag: {flag_value}, CLSisUsed: {CLSisUsed}, PostionisUsed: {PostionisUsed}")

    # 初始化数据集和数据加载器
    dataset = MNIST_Train(flag=flag_value)
    dataloader = dataset.get_loader()

    # 初始化模型
    model = ViT(CLSisUsed=CLSisUsed, PostionisUsed=PostionisUsed).to(DEVICE)

    # 加载模型权重
    file_name = f'Torch_flag{flag_value}_CLSisUsed{CLSisUsed}_PostionisUsed{PostionisUsed}.pth'
    try:
        model.load_state_dict(torch.load(file_name, map_location=DEVICE))
        print(f"Successfully loaded {file_name}")
    except FileNotFoundError:
        print(f"Pre-trained model {file_name} not found, starting from scratch.")
        continue  # 如果模型文件不存在，跳过当前组合

    # 执行评估
    test_loss, accuracy = evaluate(model, dataloader)

    # 打印评估结果
    print(f"Results for flag={flag_value}, CLSisUsed={CLSisUsed}, PostionisUsed={PostionisUsed}")
    print(f'Test Loss: {test_loss:.6f}, Accuracy: {accuracy:.6f}')

'''
flag: True, CLSisUsed: True, PostionisUsed: True
Successfully loaded Torch_flagTrue_CLSisUsedTrue_PostionisUsedTrue.pth
Results for flag=True, CLSisUsed=True, PostionisUsed=True
Test Loss: 0.122825, Accuracy: 0.963167
flag: True, CLSisUsed: True, PostionisUsed: False
Successfully loaded Torch_flagTrue_CLSisUsedTrue_PostionisUsedFalse.pth
Results for flag=True, CLSisUsed=True, PostionisUsed=False
Test Loss: 0.532457, Accuracy: 0.822633
flag: True, CLSisUsed: False, PostionisUsed: True
Successfully loaded Torch_flagTrue_CLSisUsedFalse_PostionisUsedTrue.pth
Results for flag=True, CLSisUsed=False, PostionisUsed=True
Test Loss: 0.096981, Accuracy: 0.970117
flag: True, CLSisUsed: False, PostionisUsed: False
Successfully loaded Torch_flagTrue_CLSisUsedFalse_PostionisUsedFalse.pth
Results for flag=True, CLSisUsed=False, PostionisUsed=False
Test Loss: 0.553621, Accuracy: 0.813850
flag: False, CLSisUsed: True, PostionisUsed: True
Successfully loaded Torch_flagFalse_CLSisUsedTrue_PostionisUsedTrue.pth
Results for flag=False, CLSisUsed=True, PostionisUsed=True
Test Loss: 0.088695, Accuracy: 0.973500
flag: False, CLSisUsed: True, PostionisUsed: False
Successfully loaded Torch_flagFalse_CLSisUsedTrue_PostionisUsedFalse.pth
Results for flag=False, CLSisUsed=True, PostionisUsed=False
Test Loss: 0.575870, Accuracy: 0.805883
flag: False, CLSisUsed: False, PostionisUsed: True
Successfully loaded Torch_flagFalse_CLSisUsedFalse_PostionisUsedTrue.pth
Results for flag=False, CLSisUsed=False, PostionisUsed=True
Test Loss: 0.077278, Accuracy: 0.975900
flag: False, CLSisUsed: False, PostionisUsed: False
Successfully loaded Torch_flagFalse_CLSisUsedFalse_PostionisUsedFalse.pth
Results for flag=False, CLSisUsed=False, PostionisUsed=False
Test Loss: 0.572997, Accuracy: 0.812250
'''