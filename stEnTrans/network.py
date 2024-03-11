import torch
import math

class PatchEmbed(torch.nn.Module):
    def __init__(self, patch_size=16, in_channel=1, embed_dim=1024):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # conv: (b, 1, h, w) -> (b, 1024, h/16, w/16)
        # flatten: (b, 256, h/16, w/16) -> (b, 256, hw/256)
        # reshape:(b, 256, hw/256) -> (b, hw/256, 256) (batch, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(torch.nn.Module):
    #dim = patch_size * patch_size * 4, which is quadrupled in PatchEmbed.
    def __init__(self, dim=16, num_heads=8, drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim / num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = torch.nn.Linear(dim, dim*3, bias=True)
        self.drop1 = torch.nn.Dropout(drop_ratio)
        self.proj = torch.nn.Linear(dim, dim)
        self.drop2 = torch.nn.Dropout(drop_ratio)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, int(D / self.num_heads)).permute(2, 0, 3, 1, 4)
        # (batch, num_heads, num_patches, dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * self.scale  # (batch, num_heads, num_patches, num_patches)
        att = att.softmax(dim=-1)
        att = self.drop1(att)

        x = (att @ v).transpose(1, 2).flatten(2)  # B,N,dim
        x = self.drop2(self.proj(x))
        return x

class Mlp(torch.nn.Module):
    def __init__(self, in_dim=1024, drop_ratio=0.):
        super(Mlp, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, in_dim*2)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(in_dim*2, in_dim)
        self.drop = torch.nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(torch.nn.Module):
    def __init__(self, in_dim=1024, num_heads=8, drop_ratio=0.):
        super(Block, self).__init__()
        # This step is very important, otherwise it will be difficult to converge.
        self.norm1 = torch.nn.LayerNorm(in_dim)
        self.attn = Attention(dim=in_dim, num_heads=num_heads, drop_ratio=drop_ratio)
        self.norm2 = torch.nn.LayerNorm(in_dim)
        self.mlp = Mlp(in_dim=in_dim,drop_ratio=drop_ratio)
        self.drop = torch.nn.Dropout(0.)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

def absolute_position_encoding(seq_len, embed_dim):
    """
    Generate absolute position coding
    :param seq_len: Sequence length
    :param embed_dim: PatchEmbed length
    :return: absolute position coding
    """
    # (10000 ** ((2 * i) / embed_dim))
    seq_len = int(seq_len)
    pos_enc = torch.zeros((seq_len, embed_dim))
    for pos in range(seq_len):
        for i in range(0, embed_dim, 2):
            pos_enc[pos, i] = math.sin(pos / (10000 ** (2*i / embed_dim)))
            if i + 1 < embed_dim:
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** (2*i / embed_dim)))
    return pos_enc

class VisionTransformer(torch.nn.Module):
    def __init__(self, patch_size=16, in_c=1, embed_dim=1024, depth=12, num_heads=8, drop_ratio=0.):
        super(VisionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channel=in_c, embed_dim=embed_dim)
        self.pos_drop = torch.nn.Dropout(p=drop_ratio)

        # depth transformer code blocks.
        self.blocks = torch.nn.Sequential(*[
            Block(in_dim=embed_dim, num_heads=num_heads, drop_ratio=drop_ratio)
            for _ in range(depth)
        ])

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        b, _, h, w = x.shape
        num_patches = (h / self.patch_size) * (w / self.patch_size)
        pri_x = x
        pos = absolute_position_encoding(num_patches, self.embed_dim).to(device)
        x = self.patch_embed(x)
        # Add position encoding.
        x = self.pos_drop(x + pos)
        # （batch, num_patches, embed_dim）
        x = self.blocks(x)
        # (b, 4, dim, ph, pw)
        x = x.reshape(b, -1, int(self.embed_dim//4), 4).transpose(1, 3).reshape(b, 4, int(self.embed_dim//4), int(h / self.patch_size), int(w / self.patch_size))
        fina_x = torch.zeros((b, 4, h, w)).to(device)
        k = 0
        # Reverse PatchEmbedding.
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                fina_x[:, :, i::self.patch_size, j::self.patch_size] = x[:, :, k, :, :]
                k += 1
        x = pri_x + fina_x
        x = fina_x
        return x