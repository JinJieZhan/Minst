import argparse
import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from model import swin_mnist as create_model

from Utils import train_one_epoch, evaluate, read_split_data, MNISTDataset


def objective(trial, args, train_loader, val_loader, device):
    # 定义超参数搜索空间
    patch_size = trial.suggest_int('patch_size', 4, 8)
    window_size = trial.suggest_int('window_size', 2, 4)
    embed_dim = trial.suggest_categorical('embed_dim', [48, 96, 192])
    depths_str = trial.suggest_categorical('depths', ['depth1', 'depth2', 'depth3'])

    # 根据 depth 选择对应的配置
    depths_map = {
        'depth1': (2, 2, 6, 2),
        'depth2': (2, 2, 18, 2),
        'depth3': (2, 2, 18, 6)
    }
    depths = depths_map[depths_str]

    # 创建模型
    model = create_model(num_classes=args.num_classes,
                         patch_size=patch_size,
                         window_size=window_size,
                         embed_dim=embed_dim,
                         depths=depths).to(device)

    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5E-2)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, device, epoch)

        # Optuna 记录当前的验证精度
        trial.report(val_acc, epoch)

        # 如果某个 trial 不再有改进，提前停止
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc


def main(args):
    # 自动选择设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 划分训练集和验证集
    train_paths, train_labels, val_paths, val_labels = read_split_data(args.data_path)

    img_size = 28
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        "val": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    }

    # 创建训练集和验证集
    train_dataset = MNISTDataset(train_paths, train_labels, transform=data_transform["train"])
    val_dataset = MNISTDataset(val_paths, val_labels, transform=data_transform["val"])

    batch_size = args.batch_size

    # 根据 CPU 核心数设置 DataLoader 的工作线程数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers per process')

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=nw)

    # 创建 Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args, train_loader, val_loader, device), n_trials=50)

    # 尝试保存优化结果
    try:
        import optuna.visualization as vis
        vis.plot_optimization_history(study).savefig("optimization_history.png")
        vis.plot_parallel_coordinate(study).savefig("parallel_coordinate.png")
        vis.plot_param_importances(study).savefig("param_importances.png")
    except ImportError:
        print("Optuna visualization libraries are not available.")

    print(f"Best trial: {study.best_trial.params}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data-path', type=str, default="./data/MNIST")
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()
    main(args)
