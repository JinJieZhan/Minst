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
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])

        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
