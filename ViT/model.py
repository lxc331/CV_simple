import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

class DropPath(nn.Module):
    """按样本随机丢弃残差分支。"""

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvResidualBlock(nn.Module):
    """在进入 Transformer 前提取局部纹理和轮廓。"""

    def __init__(self, channels, expansion=2):
        super(ConvResidualBlock, self).__init__()
        hidden_channels = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class PatchEmbedding(nn.Module):
    """卷积特征提取后，以较小 patch 保留服饰边缘细节。"""

    def __init__(self, image_size=28, patch_size=2, in_channels=1, embed_dim=192):
        super(PatchEmbedding, self).__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能够被 patch_size 整除")

        stem_dim = embed_dim // 2
        self.patch_size = patch_size
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            ConvResidualBlock(stem_dim),
            ConvResidualBlock(stem_dim),
        )
        self.projection = nn.Conv2d(
            stem_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        height, width = x.shape[-2:]
        pad_height = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height))

        x = self.conv_stem(x)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class ConvolutionalPositionEncoding(nn.Module):
    """使用深度卷积为 patch token 注入局部二维位置信息。"""

    def __init__(self, embed_dim):
        super(ConvolutionalPositionEncoding, self).__init__()
        self.projection = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim
        )

    def forward(self, patch_tokens, grid_size):
        batch_size, _, embed_dim = patch_tokens.shape
        feature = patch_tokens.transpose(1, 2).reshape(
            batch_size, embed_dim, grid_size[0], grid_size[1]
        )
        feature = feature + self.projection(feature)
        return feature.flatten(2).transpose(1, 2)


class MultiHeadAttention(nn.Module):
    """带 QK 归一化的多头自注意力。"""

    def __init__(self, embed_dim=192, num_heads=6, dropout=0.05):
        super(MultiHeadAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能够被 num_heads 整除")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.query_norm = nn.LayerNorm(self.head_dim)
        self.key_norm = nn.LayerNorm(self.head_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, token_num, embed_dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, token_num, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = self.query_norm(query)
        key = self.key_norm(key)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attention_dropout(attention.softmax(dim=-1))
        x = attention @ value
        x = x.transpose(1, 2).reshape(batch_size, token_num, embed_dim)
        return self.projection_dropout(self.projection(x))


class MLP(nn.Module):
    """SwiGLU 前馈网络。"""

    def __init__(self, embed_dim=192, mlp_dim=512, dropout=0.05):
        super(MLP, self).__init__()
        self.input_projection = nn.Linear(embed_dim, mlp_dim * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.output_projection = nn.Linear(mlp_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        value, gate = self.input_projection(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        x = self.dropout1(x)
        return self.dropout2(self.output_projection(x))


class EncoderBlock(nn.Module):
    """带 LayerScale 和随机深度的 Pre-Norm Transformer Block。"""

    def __init__(
        self,
        embed_dim=192,
        num_heads=6,
        mlp_dim=512,
        dropout=0.05,
        drop_path=0.0,
        layer_scale_init=1e-2,
    ):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        self.layer_scale1 = nn.Parameter(layer_scale_init * torch.ones(embed_dim))
        self.layer_scale2 = nn.Parameter(layer_scale_init * torch.ones(embed_dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale1 * self.attention(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale2 * self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """面向 FashionMNIST 细粒度分类的混合卷积 Vision Transformer。"""

    def __init__(
        self,
        image_size=28,
        patch_size=2,
        in_channels=1,
        num_classes=10,
        embed_dim=192,
        depth=8,
        num_heads=6,
        mlp_dim=512,
        dropout=0.05,
        drop_path_rate=0.12,
    ):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        token_num = self.patch_embedding.num_patches + 1
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, token_num, embed_dim))
        self.conv_position = ConvolutionalPositionEncoding(embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        drop_path_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_dim,
                    dropout,
                    drop_path_rates[index],
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def interpolate_position_embedding(self, grid_size):
        base_grid_size = self.patch_embedding.grid_size
        if grid_size == base_grid_size:
            return self.position_embedding

        class_position = self.position_embedding[:, :1]
        patch_position = self.position_embedding[:, 1:].reshape(
            1, base_grid_size[0], base_grid_size[1], -1
        )
        patch_position = patch_position.permute(0, 3, 1, 2)
        patch_position = F.interpolate(
            patch_position, size=grid_size, mode="bicubic", align_corners=False
        )
        patch_position = patch_position.flatten(2).transpose(1, 2)
        return torch.cat((class_position, patch_position), dim=1)

    def forward(self, x):
        patch_size = self.patch_embedding.patch_size
        grid_size = (
            (x.shape[-2] + patch_size - 1) // patch_size,
            (x.shape[-1] + patch_size - 1) // patch_size,
        )

        patch_tokens = self.patch_embedding(x)
        patch_tokens = self.conv_position(patch_tokens, grid_size)
        class_token = self.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((class_token, patch_tokens), dim=1)
        x = self.embedding_dropout(
            x + self.interpolate_position_embedding(grid_size)
        )

        for block in self.encoder:
            x = block(x)

        x = self.norm(x)
        class_feature = x[:, 0]
        patch_feature = x[:, 1:].mean(dim=1)
        return self.classifier(torch.cat((class_feature, patch_feature), dim=1))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer().to(device)
    summary(model, input_size=(1, 28, 28))

