from torch import nn
import torch


# 定义了一个名为 PatchEmbed 的 PyTorch 模型类，用于将图像转换为patch嵌入（Patch Embedding）
class PatchEmbed(nn.Module):

    def __init__(self, img_size, patch_size, in_c, embed_dim, norm_layer=None):
        '''
        img_size: 输入图像的大小，默认为 28x28 像素。
        patch_size: patch的大小，默认为 7x7 像素。
        in_c: 输入图像的通道数，默认为 1（适用于灰度图像）。
        embed_dim: 嵌入维度，默认为 64，即每个patch被嵌入到的特征向量的维度。
        norm_layer: 归一化层的类型，默认为 None。如果提供了，则用它来对嵌入向量进行归一化处理；否则使用恒等映射 nn.Identity()
        '''
        super().__init__()
        img_size = (img_size, img_size)  # 图像的高度和宽度
        patch_size = (patch_size, patch_size)  # patch的高度和宽度
        # grid_size 计算了图像可以被分成多少个patch。具体来说，它是一个元组，
        # 其中第一个元素是图像宽度除以patch宽度的整数部分，第二个元素是图像高度除以patch高度的整数部分。
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 计算了图像被划分为多少个patch
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 代码计算了总的patch数量

        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# 定义注意力机制模块
class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, proj_drop_ratio):
        '''
        self.num_heads 记录了注意力头的数量。
        head_dim 计算每个头的维度。
        self.scale 是注意力分数的缩放因子。
        self.qkv 是一个线性变换层，将输入向量转换为查询（q）、键（k）、值（v）三部分。
        self.attn_drop 和 self.proj_drop 是用于在计算过程中应用的dropout层。
        self.proj 是用于将注意力加权后的结果进行投影的线性变换层。
        '''
        super(Attention, self).__init__()
        self.num_heads = num_heads  # 将输入向量分为多个头以并行计算注意力
        head_dim = dim // num_heads  # 通过整数除法，计算每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 设置注意力分数的缩放因子
        # 输入向量的维度是 dim，输出的维度是 dim * 3，因为每个头需要三个不同的映射（q、k、v）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # 用于在计算完注意力后对加权结果进行投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # num_patches 是指图像被划分为的图块数目，或者序列被划分为的片段数目。
        # 在某些实现中，为了兼容位置编码（position encoding）或其他需要添加的特殊符号
        B, N, C = x.shape
        '''
        B：批量大小（batch_size）
        N：patch 数量加上一个额外的位置编码（num_patches + 1）
        C：总嵌入维度（total_embed_dim）
        '''
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # 对张量的最后一个维度应用 softmax 函数
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, drop=0.0):

        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # 注意这里调用了 nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)  # 应用 dropout 在激活函数后面
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 qkv_bias,
                 qk_scale,
                 drop_ratio,
                 attn_drop_ratio,
                 ):
        super(Block, self).__init__()
        # self.norm1: 第一个归一化层，使用指定的 norm_layer 对输入进行归一化。
        self.norm1 = nn.LayerNorm(dim)

        # self.attn: 注意力机制，使用 Attention 类处理输入，包括注意力计算和投影的dropout。
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

        # self.norm2: 第二个归一化层，再次对输入进行归一化。
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp: 多层感知机模块，使用指定的 Mlp 类进行处理，包括激活函数和dropout。
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_c=1, num_classes=10,
                 embed_dim=16, depth=3, num_heads=2, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=0.5, drop_ratio=0.2, attn_drop_ratio=0.2,
                 embed_layer=PatchEmbed, CLSisUsed=False, PostionisUsed=True):
        super(VisionTransformer, self).__init__()
        self.CLSisUsed = CLSisUsed
        self.PostionisUsed = PostionisUsed
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if self.CLSisUsed:
            self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))

        if self.PostionisUsed:
            if self.CLSisUsed:
                self.pos_emb = nn.Parameter(torch.rand(1, num_patches + 1, embed_dim))
            else:
                self.pos_emb = nn.Parameter(torch.rand(1, num_patches, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes)

        if self.PostionisUsed:
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        if self.CLSisUsed:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)

        if self.PostionisUsed:
            if self.CLSisUsed:
                cls_token = self.cls_token.expand(x.size(0), 1, x.size(2))
                x = torch.cat((cls_token, x), dim=1)
            x = x + self.pos_emb
        elif self.CLSisUsed:
            cls_token = self.cls_token.expand(x.size(0), 1, x.size(2))
            x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        y = self.forward_features(x)
        if self.CLSisUsed:
            return self.head(y[:, 0, :])  # (batch_size, num_classes)
        else:
            x = y.mean(dim=1)  # (batch_size, emb_size)
            return self.head(x)  # (batch_size, num_classes)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # 卷积层 (nn.Conv2d) 的处理
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # 层归一化 (nn.LayerNorm) 的处理
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
