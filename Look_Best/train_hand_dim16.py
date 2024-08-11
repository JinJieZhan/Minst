import torch

import torch.nn.functional as F
from dataset import MNIST_Train
from dataset import MNIST_Check
from Look_Best.Vit_Hand import VisionTransformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define your dataset
dataset = MNIST_Train()

# Initialize your ViT model
model = VisionTransformer().to(DEVICE)

# Attempt to load pretrained model
try:
    model.load_state_dict(torch.load('Torch_flagTrue_CLSisUsedFalse_PostionisUsedTrue_dim16.pth', map_location=DEVICE))
    print("Successfully loaded model.pth")
except FileNotFoundError:
    print("Pre-trained model not found, starting from scratch.")

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training parameters
EPOCHS = 10
BATCH_SIZE = 64

# Data loader
try:
    dataloader = dataset.get_loader()
    iter_count = 0
    for epoch in range(EPOCHS):
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            iter_count += 1
            # Print loss every 1000 iterations
            if iter_count % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{EPOCHS}], Iteration [{iter_count}], Loss: {loss.item()}')
        torch.save(model.state_dict(), 'Torch_flagTrue_CLSisUsedFalse_PostionisUsedTrue_dim16.pth')
        print(f'Saved model after epoch {epoch + 1}')
    print("Training finished!")

except Exception as e:
    print(f"Error during training: {e}")
    raise e

dataset = MNIST_Check()
dataloader = dataset.get_loader()

# 将模型设置为评估模式
model.eval()


# 定义评估函数
def evaluate(model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 适应你的损失函数
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return test_loss, accuracy


# 执行评估
test_loss, accuracy = evaluate(model, dataloader)

# 打印结果
print(f'Test Loss: {test_loss:.6f}, Accuracy: {accuracy:.6f}')
# Test Loss: 0.310295, Accuracy: 0.901200