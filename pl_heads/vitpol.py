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

#         # p для каждой головы
#         self.p_raw = nn.Parameter(torch.zeros(heads))

#         self.last_attn = None
        

#     def forward(self, x):
#         # B — размер батча
#         # N = 17 — число токенов
#         # C = 128 — размер эмбеддинга
#         B, N, C = x.shape
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: t.reshape(B, N, self.heads, C // self.heads).transpose(1, 2), qkv)
#         attn = (q @ k.transpose(-2, -1)) * self.scale

#         p = (1 + F.softplus(self.p_raw)).view(1, self.heads, 1, 1)

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
#         self.dim_head = dim // heads
#         self.scale = self.dim_head ** -0.5

#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.proj = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)

#         # 🔥 КАЖДАЯ ГОЛОВА - СВОЯ φ с уникальным параметром!
#         self.p_raw = nn.Parameter(torch.zeros(heads))  # [heads] вместо scalar
        
#         # Бонус: разные scale для каждой головы (опционально)
#         self.head_scales = nn.Parameter(torch.ones(heads))


#     # def phi(self, x):
#     #     B, H, N, D = x.shape
        
#     #     # p_raw: [H] → используем БЕЗ softplus для большего диапазона!
#     #     p = self.p_raw.view(1, H, 1, 1)  # [-inf, +inf] вместо [1, +inf]
        
#     #     scale = self.scale * self.head_scales.view(1, H, 1, 1)

#     #     x = x * scale
#     #     x = torch.clamp(x, -5, 5)
        
#     #     # 🔥 p_raw ПРЯМО МОДУЛИРУЕТ вход ELU!
#     #     return F.elu(x * (1 + p)) + 1 

    

#     # def phi(self, x):
#     #     B, H, N, D = x.shape

#     #     p = F.softplus(self.p_raw).view(1, H, 1, 1)
#     #     scale = self.scale * self.head_scales.view(1, H, 1, 1)

#     #     x = x * scale
#     #     x = F.layer_norm(x, (D,))
#     #     x = x / (1 + x.abs())

#     #     # 🔥 ПРОСТОЙ ReLU^p — идеально!
#     #     relu_part = F.relu(x)
#     #     kernel = torch.pow(relu_part + 1e-6, p)  # ReLU^p
        
#     #     return kernel
    
#     def phi(self, x, global_ctx=None):
#         B, H, N, D = x.shape
#         p = F.softplus(self.p_raw).view(1, H, 1, 1)
#         scale = self.scale * self.head_scales.view(1, H, 1, 1)

#         x = x * scale
#         x = F.layer_norm(x, (D,))
#         x = x / (1 + x.abs())

#         # 🔥 АДАПТИВНЫЙ THRESHOLD ИЗ КОНТЕКСТА
#         if global_ctx is not None:
#             thresh = torch.sigmoid(self.thresh_proj(global_ctx)).view(B, H, 1, 1)
#         else:
#             thresh = 0.1
            
#         # 🔥 ReLU^p + Swish + Sparsity
#         relu_part = F.relu(x - thresh)
#         power_part = torch.pow(relu_part + 1e-6, p)
#         swish_gate = x * F.sigmoid(x)
        
#         kernel = power_part * swish_gate * (relu_part > 0.1).float()
        
#         return kernel
    
#     def forward(self, x):
#         B, N, C = x.shape

#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(
#             lambda t: t.reshape(B, N, self.heads, self.dim_head).transpose(1, 2),
#             qkv
#         )

#         q = self.phi(q)
#         k = self.phi(k)

#         kv = k.transpose(-2, -1) @ v
#         k_sum = k.sum(dim=2) 
#         z = 1 / (q @ k_sum.unsqueeze(-1) + 1e-6)

#         out = (q @ kv) * z
#         out = out.transpose(1, 2).reshape(B, N, C)

#         return self.dropout(self.proj(out))


# class Attention(nn.Module):
#     def __init__(self, dim, heads, dropout=0.1):
#         super().__init__()
#         assert dim % heads == 0
        
#         self.heads = heads
#         self.dim_head = dim // heads
#         self.scale = self.dim_head ** -0.5
        
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
#         self.proj = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)
        
#         # 🔥 Базовые параметры по головам
#         self.p_raw = nn.Parameter(torch.zeros(heads))
#         self.head_scales = nn.Parameter(torch.ones(heads))
#         self.head_pos_encoding = nn.Parameter(torch.zeros(heads))
        
#         # 🔥 КОНТЕКСТНЫЕ ПРОЕКЦИИ (новое!)
#         self.context_proj_p = nn.Linear(dim, heads)      # global_ctx → p_mod [B,H]
#         self.context_proj_scale = nn.Linear(dim, heads)  # global_ctx → scale_mod [B,H]
    
#     def phi(self, x, global_ctx=None):
#         B, H, N, D = x.shape
        
#         # БАЗОВЫЕ параметры
#         p_base = F.softplus(self.p_raw).view(1, H, 1, 1)
#         scale_base = self.scale * self.head_scales.view(1, H, 1, 1)
#         head_pos_emb = F.softplus(self.head_pos_encoding).view(1, H, 1, 1)
        
#         # 🔥 КОНТЕКСТНАЯ МОДУЛЯЦИЯ (если передан global_ctx)
#         if global_ctx is not None:
#             # [B,H] модуляции из контекста
#             p_mod = self.context_proj_p(global_ctx).view(B, H, 1, 1)  # [B,H,1,1]
#             scale_mod = torch.sigmoid(self.context_proj_scale(global_ctx)).view(B, H, 1, 1)
            
#             # Комбинируем: base + context
#             p = p_base + p_mod
#             scale = scale_base * (1 + scale_mod)
#         else:
#             p = p_base
#             scale = scale_base
        
#         # Трансформации (B-размер теперь учитывается!)
#         x = x * scale  # [B,H,N,D]
#         x = F.layer_norm(x, (D,))
#         x = x / (1 + x.abs())
        
#         kernel = F.elu(x * (1 + p + head_pos_emb)) + 1
#         return kernel
    
#     def forward(self, x):
#         B, N, C = x.shape
        
#         # 🔥 ГЛОБАЛЬНЫЙ КОНТЕКСТ (среднее по токенам)
#         global_ctx = x.mean(dim=1)  # [B,C] — агрегация всей последовательности
        
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(
#             lambda t: t.reshape(B, N, self.heads, self.dim_head).transpose(1, 2),
#             qkv
#         )
        
#         # 🔥 phi с контекстом!
#         q = self.phi(q, global_ctx)
#         k = self.phi(k, global_ctx)
        
#         kv = k.transpose(-2, -1) @ v
#         k_sum = k.sum(dim=2) 
#         z = 1 / (q @ k_sum.unsqueeze(-1) + 1e-6)
        
#         out = (q @ kv) * z
#         out = out.transpose(1, 2).reshape(B, N, C)
        
#         return self.dropout(self.proj(out))


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # 🔥 ОДИН общий параметр p
        self.p_raw = nn.Parameter(torch.tensor(0.0))

        # scale можно оставить разный для голов
        self.head_scales = nn.Parameter(torch.ones(heads))


    def phi(self, x):
        B, H, N, D = x.shape

        p = F.softplus(self.p_raw) + 0.5   # scalar > 0
        scale = self.scale * self.head_scales.view(1, H, 1, 1)

        x = x * scale
        x = F.layer_norm(x, (D,))
        x = x / (1 + x.abs())

        relu_part = F.relu(x)
        kernel = torch.pow(relu_part + 1e-6, p)

        return kernel


    def forward(self, x):
        B, N, C = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(B, N, self.heads, self.dim_head).transpose(1, 2),
            qkv
        )

        q = self.phi(q)
        k = self.phi(k)

        kv = k.transpose(-2, -1) @ v
        k_sum = k.sum(dim=2)

        z = 1 / (q @ k_sum.unsqueeze(-1) + 1e-6)

        out = (q @ kv) * z
        out = out.transpose(1, 2).reshape(B, N, C)

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