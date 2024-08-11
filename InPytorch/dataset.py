import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


class MNIST_Train(Dataset):
    def __init__(self, root: str = '../data/', flag=False, train=True, download=True, batch_size=64):
        self.root = root
        self.train = train
        self.download = download
        self.batch_size = batch_size
        self._transform = transforms.ToTensor() if flag else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(root=self.root, train=self.train, download=self.download,
                                      transform=self._transform)
        self.loader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def get_loader(self):
        return self.loader


class MNIST_Check(Dataset):
    def __init__(self, root: str = '../data/', flag=False, batch_size=64):
        self.root = root
        self.batch_size = batch_size
        self._transform = transforms.ToTensor() if flag else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.test_dataset = datasets.MNIST(root=self.root, train=False, download=True, transform=self._transform)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=self.batch_size)

    def __len__(self):
        return len(self.test_dataset)

    def __getitem__(self, index):
        return self.test_dataset[index]

    def get_loader(self):
        return self.test_loader
