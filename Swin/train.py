import argparse
import os
import random
import sys
from math import floor
from torchvision import datasets
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from model import swin_mnist as create_model
import predict


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
def evaluate(model, load, device, epoch):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss, accu_num, sample_num = 0.0, 0, 0

    data_loader = tqdm(load, file=sys.stdout, desc=f"[valid epoch {epoch}]")

    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        sample_num += images.size(0)

        pred = model(images)
        pred_classes = torch.argmax(pred, dim=1)
        accu_num += (pred_classes == labels).sum().item()

        loss = loss_function(pred, labels)
        accu_loss += loss.item()

        # 更新 tqdm 描述
        data_loader.set_description(
            f"[valid epoch {epoch}] loss: {accu_loss / (step + 1):.3f}, acc: {accu_num / sample_num:.3f}")

    return accu_loss / (step + 1), accu_num / sample_num


def main(opt):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    # 数据预处理
    img_size = 28  # MNIST图像尺寸为28x28
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST的均值和标准差
        ]),
        "val": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    }

    # 使用自定义的数据集加载
    train_paths, train_labels, val_paths, val_labels = read_split_data(root='./data/', val_rate=0.2)
    train_dataset = MNISTDataset(train_paths, train_labels, transform=data_transform["train"])
    val_dataset = MNISTDataset(val_paths, val_labels, transform=data_transform["val"])

    batch_size = opt.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print(f'Using {nw} dataloader workers every process')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)

    model = create_model(num_classes=opt.num_classes).to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=opt.lr, weight_decay=5E-2)

    for epoch in range(opt.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                load=train_loader,
                                                device=device,
                                                epoch=epoch)

        # 验证
        val_loss, val_acc = evaluate(model=model,
                                     load=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), f"./weights/model-{epoch}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-path', type=str, default="./data/MNIST")
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
    predict.main(opt)
