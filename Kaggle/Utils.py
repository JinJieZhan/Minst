import random
import sys
from math import floor
from torch.utils.data import Dataset
from torchvision import datasets
from tqdm import tqdm
import torch
import os
from PIL import Image


def read_split_data(root: str, val_rate: float = 0.2):
    # 如果根目录不存在，则创建它
    if not os.path.exists(root):
        os.makedirs(root)
        # 下载MNIST数据集
        mnist_train = datasets.MNIST(root=root, train=True, download=True)

        # 为每个类别创建一个目录
        for i in range(10):
            class_dir = os.path.join(root, f'number_{i}')
            os.makedirs(class_dir, exist_ok=True)

        # 将训练集图像保存到相应的类别目录中
        for idx, (image, label) in enumerate(mnist_train):
            image_path = os.path.join(root, f'number_{label}', f'train_img_{idx}.jpeg')
            image.save(image_path)

    # 确保随机性可复现
    random.seed(0)

    # 确保根目录存在
    assert os.path.exists(root), f"Dataset root directory '{root}' does not exist."

    # 初始化存储图像路径和标签的列表
    all_images_path, all_images_label = [], []

    # 收集所有训练集图像的路径和标签
    for label in range(10):
        class_dir = os.path.join(root, f'number_{label}')
        for img_name in os.listdir(class_dir):
            if img_name.startswith('train_img'):
                all_images_path.append(os.path.join(class_dir, img_name))
                all_images_label.append(label)

    # 初始化训练集和验证集
    train_images_path, train_images_label, val_images_path, val_images_label = [], [], [], []

    for label in range(10):
        # 获取当前类别的所有图像
        current_class_images = [(p, l) for p, l in zip(all_images_path, all_images_label) if l == label]
        num_val = floor(len(current_class_images) * val_rate)

        # 打乱图像顺序
        random.shuffle(current_class_images)

        # 分配验证集和训练集
        val_images_path.extend([p for p, l in current_class_images[:num_val]])
        val_images_label.extend([l for p, l in current_class_images[:num_val]])
        train_images_path.extend([p for p, l in current_class_images[num_val:]])
        train_images_label.extend([l for p, l in current_class_images[num_val:]])

    # 返回训练集和验证集的路径及其对应的标签
    return train_images_path, train_images_label, val_images_path, val_images_label


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


def train_one_epoch(model, optimizer, load, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss, accu_num = 0.0, 0
    sample_num = 0

    data_loader = tqdm(load, file=sys.stdout, desc=f"[train epoch {epoch}]")

    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        sample_num += images.size(0)

        optimizer.zero_grad()

        pred = model(images)
        pred_classes = torch.argmax(pred, dim=1)
        accu_num += (pred_classes == labels).sum().item()

        loss = loss_function(pred, labels)
        loss.backward()
        optimizer.step()

        accu_loss += loss.item()

        # 更新 tqdm 描述
        data_loader.set_description(
            f"[train epoch {epoch}] loss: {accu_loss / (step + 1):.3f}, acc: {accu_num / sample_num:.3f}")

    return accu_loss / (step + 1), accu_num / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1, device=device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1, device=device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        sample_num += images.size(0)

        pred = model(images)
        pred_classes = torch.argmax(pred, dim=1)
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        # 更新 tqdm 描述
        data_loader.set_description(f"[valid epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, "
                                    f"acc: {accu_num.item() / sample_num:.3f}")

    # 将损失和准确度返回为标量值
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
