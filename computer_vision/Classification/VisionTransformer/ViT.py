from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
import torch.nn.functional as F


img = torch.rand(8, 3, 224, 224)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 16, emb_size = 768, img_size = 224):
        super().__init__()

        self.patch_size = patch_size

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, patch_size, stride = patch_size),
            Rearrange('b e h w -> b (h w) e') #(batch_size, 14*14, 768)
        )

        self.class_token = nn.Parameter(torch.rand((1, 1, emb_size))) #(1, 1, 768)

        self.positions = nn.Parameter(torch.rand((img_size // patch_size ** 2 +1, emb_size)))

    def forward(self, x):
        b = x.shape[0] #batch size
        x = self.projection((x)) #(batch_size, 196, 768)
        class_token = repeat(self.class_token, '() n e -> b n e', b = b) #(batch_size, 1, 768)
        x = torch.cat([class_token, x], dim = 1)
        x += self.positions

        return x #output shape : (8, 197, 768) (batch, 1+patch_output, embedding) 1: class token


class MultiheadAttention(nn.Module):
    def __init__(self, emb_size = 768, num_heads = 8, dropout = 0):
        super().__init__()

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask = None): #x : output of patch embedding (batch_size, 1+ num of patches, embedding)
        queries = rearrange(self.queries(x), 'b n (h d) -> b h n d', h = self.num_heads) #b, 197, 768 -> b, 8, 197, 96
        keys = rearrange(self.keys(x), 'b n (h d) -> b h n d', h = self.num_heads)
        values = rearrange(self.values(x), 'b n (h d) -> b h n d', h = self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        scaling = self.emb_size **(1/2)
        att = F.softmax(energy, dim = 1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)') #b, 8, 197, 96 -> b, 197, 768
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()

        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion = 4, drop = 0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(expansion * emb_size, emb_size)
        )



class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size = 768, drop = 0, forward_expansion = 4, forward_drop = 0, **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiheadAttention(emb_size, **kwargs),
                nn.Dropout(drop)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion = forward_expansion, drop = forward_drop),
                nn.Dropout(drop)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size = 768, n_classes = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction = 'mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )


class ViT(nn.Sequential):
    def __init__(self, in_channels = 3, patch_size = 16, emb_size = 768, img_size = 224, depth = 12, n_classes = 10, **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size = emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
x = torch.rand(8, 3, 224, 224)
model = ViT()
output = model(x)
print(output.shape)