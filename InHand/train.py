import torch
import torch.nn.functional as F
from itertools import product

from InHand.Vit import VisionTransformer
from InHand.dataset import MNIST_Train

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
times = 0
bool_values = [True, False]
combinations = list(product(bool_values, repeat=3))
for combination in combinations:
    flag, CLSisUsed, PostionisUsed = combination
    print(f"flag: {flag}, CLSisUsed: {CLSisUsed}, PostionisUsed: {PostionisUsed}")

    dataset = MNIST_Train(flag=flag)
    model = VisionTransformer(CLSisUsed=CLSisUsed, PostionisUsed=PostionisUsed).to(DEVICE)

    file_name = f'Torch_flag{flag}_CLSisUsed{CLSisUsed}_PostionisUsed{PostionisUsed}.pth'
    try:
        model.load_state_dict(torch.load(file_name, map_location=DEVICE))
        print(f"Successfully loaded {file_name}")
    except FileNotFoundError:
        print(f"Pre-trained model {file_name} not found, starting from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS = 10
    BATCH_SIZE = 64
    print(f"训练次数：{times}")

    try:
        dataloader = dataset.get_loader()
        for epoch in range(EPOCHS):
            for batch_idx, (imgs, labels) in enumerate(dataloader):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()

            print(f'第{times}中第epoch{epoch}的Loss: {loss.item()}')
            torch.save(model.state_dict(), file_name)
        times+=1
        print("Training finished!")

    except Exception as e:
        print(f"Error during training: {e}")
        raise e
