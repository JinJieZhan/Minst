import torch
from torch import nn


class ViT(nn.Module):
    def __init__(self, emb_size=16, CLSisUsed=True, PostionisUsed=True,nhead=2,num_layer=3):
        super().__init__()
        self.CLSisUsed = CLSisUsed
        self.PostionisUsed = PostionisUsed
        self.nhead = nhead
        self.num_layer = num_layer
        self.patch_size = 4
        self.patch_count = 28 // self.patch_size
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.patch_size ** 2, kernel_size=self.patch_size, padding=0,
                              stride=self.patch_size)
        self.patch_emb = nn.Linear(in_features=self.patch_size ** 2, out_features=emb_size)

        if self.CLSisUsed:
            self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size))
        if self.PostionisUsed:
            if self.CLSisUsed:
                self.pos_emb = nn.Parameter(torch.rand(1, self.patch_count ** 2 + 1, emb_size))
            else:
                self.pos_emb = nn.Parameter(torch.rand(1, self.patch_count ** 2, emb_size))

        self.transformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, batch_first=True), num_layers=num_layer)
        self.cls_linear = nn.Linear(in_features=emb_size, out_features=10)

    def forward(self, x):
        x = self.conv(x)  # 输出形状为 (batch_size, 16, height, width)
        x = x.view(x.size(0), x.size(1), -1)  # 变形为 (batch_size, 16, 49) 假设 height=7, width=7
        x = x.permute(0, 2, 1)  # 变形为 (batch_size, 49, 16)
        x = self.patch_emb(x)  # 输出形状为 (batch_size, 49, emb_size

        if self.PostionisUsed:
            if self.CLSisUsed:
                cls_token = self.cls_token.expand(x.size(0), 1, x.size(2))
                x = torch.cat((cls_token, x), dim=1)
            x = x + self.pos_emb
        elif self.CLSisUsed:
            cls_token = self.cls_token.expand(x.size(0), 1, x.size(2))
            x = torch.cat((cls_token, x), dim=1)

        y = self.transformer_enc(
            x)  # (batch_size, seq_len+1, emb_size) if CLSisUsed, else (batch_size, seq_len, emb_size)

        if self.CLSisUsed:
            return self.cls_linear(y[:, 0, :])  # (batch_size, num_classes)
        else:
            x = y.mean(dim=1)  # (batch_size, emb_size)
            return self.cls_linear(x)  # (batch_size, num_classes)
