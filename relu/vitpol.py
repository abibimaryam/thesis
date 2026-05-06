import torch
import torch.nn as nn
import torch.nn.functional as F

    
    # изначальное изображение (1, 28, 28)
    # делится на патчи по 7x7 (28/7) 
    # 4^2 = 16 
    # [ 7x7 ][ 7x7 ][ 7x7 ][ 7x7 ]
    # [ 7x7 ][ 7x7 ][ 7x7 ][ 7x7 ]
    # [ 7x7 ][ 7x7 ][ 7x7 ][ 7x7 ]
    # [ 7x7 ][ 7x7 ][ 7x7 ][ 7x7 ]
    # 16 патчей
    # каждый — вектор 128 из-за dim = 128 который мы выбрали
    # добавляется специальный CLS-токен (ещё один вектор 128)
    # теперь всего 17 токенов (16 патчей + 1 CLS)
    # добавляются позиционные эмбеддинги
    # чтобы модель понимала, где находится каждый патч
    # Теперь размер становится:(B, 17, 128)
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return self.dropout(x) 


# последовательность из 17 токенов проходит через attention
# каждый токен сравнивает себя со всеми остальными
# вычисляется матрица внимания размером 17×17
# каждый токен получает взвешенную сумму информации от других
# class Attention(nn.Module):
#     def __init__(self, dim, heads, dropout=0.1,p=2):
#         super().__init__()
#         self.heads = heads
#         self.scale = (dim // heads) ** -0.5
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.proj = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)
#         self.attn_dropout = nn.Dropout(dropout) 

#         # обучаемый параметр
#         self.p_raw = nn.Parameter(torch.tensor(0.0))

#         self.last_attn = None
        

#     def forward(self, x):
#         # B — размер батча
#         # N = 17 — число токенов
#         # C = 128 — размер эмбеддинга
#         B, N, C = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: t.reshape(B, N, self.heads, C // self.heads).transpose(1, 2), qkv)
#         attn = (q @ k.transpose(-2, -1)) * self.scale

#         p = 1 + F.softplus(self.p_raw)

#         attn = F.relu(attn) ** p / (N ** 0.5)
#         # attn = (attn ** p) / (N ** 0.5)
#         attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

#         self.last_attn = attn.detach()
        
#         attn = self.attn_dropout(attn) 
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         return self.dropout(self.proj(out))


# class Attention(nn.Module):
#     def __init__(self, dim, heads, dropout=0.1):
#         super().__init__()
#         self.heads = heads
#         self.scale = (dim // heads) ** -0.5

#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.proj = nn.Linear(dim, dim)

#         self.dropout = nn.Dropout(dropout)
#         self.attn_dropout = nn.Dropout(dropout)

#         # per-head learnable exponent
#         self.p_raw = nn.Parameter(torch.zeros(heads))

#         self.last_attn = None

#     def forward(self, x):
#         B, N, C = x.shape

#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(
#             lambda t: t.reshape(B, N, self.heads, C // self.heads).transpose(1, 2),
#             qkv
#         )

#         attn = (q @ k.transpose(-2, -1)) * self.scale

#         p = 1 + F.softplus(self.p_raw).view(1, self.heads, 1, 1)

#         attn = F.relu(attn) ** p / (N ** 0.5)
#         attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

#         self.last_attn = attn.detach()

#         attn = self.attn_dropout(attn)
#         out = (attn @ v).transpose(1, 2).reshape(B, N, C)

#         return self.dropout(self.proj(out))

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # per-head learnable exponent
        self.p_raw = nn.Parameter(torch.zeros(heads))

        self.last_attn = None
        self.last_energy = None

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(B, N, self.heads, C // self.heads).transpose(1, 2),
            qkv
        )

        sim = (q @ k.transpose(-2, -1)) * self.scale

 
        p = 1 + F.softplus(self.p_raw).view(1, self.heads, 1, 1)


        num = F.relu(sim) ** p

        den = sim.abs().sum(dim=-1, keepdim=True) + 1e-6

        attn = num / den

        self.last_attn = attn.detach()
        self.last_energy = den.detach()

        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.dropout(self.proj(out))



class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        # MLP (Multi-Layer Perceptron)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):


    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10, dim=128, depth=4, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, dim, dropout)
        self.transformer = nn.Sequential(*[TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(self.dropout(x))  


# Пример использования
model = ViT()
print(model)