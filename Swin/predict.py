import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
from model import swin_mnist as create_model


class MNISTDataset(Dataset):
    def __init__(self, images_path, images_label, transform=None):
        self.images_path = images_path
        self.images_label = images_label
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = self.images_label[index]
        img = Image.open(img_path).convert("L")  # MNIST是灰度图像

        if self.transform:
            img = self.transform(img)
        return img, label


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define data transform
    img_size = 28
    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 保持图像为28x28
        transforms.ToTensor(),  # 转换为Tensor并保持单通道
        transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST数据集的均值和标准差
    ])

    # Load dataset
    mnist_test = datasets.MNIST(root='./data/', train=False, download=True, transform=data_transform)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)

    # Create model
    model = create_model(num_classes=10).to(device)
    weights_dir = './weights'
    epochs = opt.epochs
    # Iterate over a range of model files
    for i in range(epochs):  # Adjust the range as needed
        model_weight_path = os.path.join(weights_dir, f"model-{i}.pth")

        if os.path.isfile(model_weight_path):
            print(f"Processing file: {model_weight_path}")
        # Load model weights

        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()

        # Define loss function
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)

                # Compute loss
                loss = criterion(output, labels)
                total_loss += loss.item()

                # Compute predictions
                predict = torch.softmax(output, dim=1)
                _, predicted_classes = torch.max(predict, dim=1)

                # Compute accuracy
                correct_predictions += (predicted_classes == labels).sum().item()
                total_samples += labels.size(0)

            average_loss = total_loss / len(test_loader)
            accuracy = correct_predictions / total_samples * 100  # percentage

        print(f'Average Loss: {average_loss:.4f} Accuracy: {accuracy:.6f}%')

