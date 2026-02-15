## ========================================
## ✅ ВИЗУАЛИЗАЦИЯ МАСОК ATTENTION ПОСЛЕ ОБУЧЕНИЯ
## ========================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import torch
from vit import model

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Загрузка данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

batch_size=64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

model.load_state_dict(torch.load("best_vit_fashion.pth", map_location=device))


# Переключаем модель в режим оценки
model.to(device)
model.eval()
print("\n🔍 ВИЗУАЛИЗАЦИЯ ATTENTION МАСОК...")

class AttentionHook:
    def __init__(self):
        self.attentions = []
        self.hooks = []
    
    def hook_fn(self, module, input, output):
        # Захватываем attention weights (после softmax)
        if isinstance(output, tuple) and len(output) > 1:
            self.attentions.append(output[1].detach())
        # Или если attention сохраняется в модуле
        elif hasattr(module, 'attention_weights'):
            self.attentions.append(module.attention_weights.detach())
    
    def register(self, model):
        # Автоматически находим все attention слои
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(self.hook_fn)
                self.hooks.append(hook)
    
    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attentions = []

def plot_attention_masks(model, test_loader, num_samples=8):
    hook = AttentionHook()
    hook.register(model)
    
    plt.figure(figsize=(20, 4*num_samples))
    
    for i, (images, labels) in enumerate(test_loader):
        if i >= num_samples:
            break
            
        # Одно изображение из батча
        img = images[:1].to(device)
        label = labels[0].item()
        
        hook.clear()
        with torch.no_grad():
            _ = model(img)
        
        if not hook.attentions:
            print("⚠️ Attention не найдены - проверьте структуру модели")
            break
            
        # Последний слой, class token attention к патчам
        last_attn = hook.attentions[-1]  # [B, heads, N, N]
        B, H, N, _ = last_attn.shape
        cls_attn = last_attn[0, :, 0, 1:].mean(dim=0).cpu().numpy()  # Усредняем heads
        
        # 7x7 патчи для FashionMNIST (28/4=7)
        side = int(np.sqrt(len(cls_attn)))
        attn_map = cls_attn.reshape(side, side)
        attn_map = attn_map / attn_map.max()
        
        # subplot
        plt.subplot(num_samples, 4, i*4 + 1)
        img_show = img[0, 0].cpu() * 0.5 + 0.5
        plt.imshow(img_show, cmap='gray')
        plt.title(f'{fashion_classes[label]}\nОригинал')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i*4 + 2)
        sns.heatmap(attn_map, cmap='hot', cbar=True)
        plt.title('Attention Map\n(среднее по heads)')
        
        plt.subplot(num_samples, 4, i*4 + 3)
        plt.imshow(img_show, cmap='gray')
        plt.imshow(attn_map, cmap='jet', alpha=0.6, 
                  extent=[0, 28, 28, 0], interpolation='bilinear')
        plt.title('Overlay')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i*4 + 4)
        # Rollout (если несколько слоев)
        if len(hook.attentions) > 1:
            rollout = np.prod([a[0,0,0,1:].cpu().numpy() for a in hook.attentions], axis=0)
            rollout = rollout.reshape(side, side) / rollout.max()
            sns.heatmap(rollout, cmap='viridis')
            plt.title('Attention Rollout\n(все слои)')
    
    hook.clear()
    plt.tight_layout()
    plt.show()
    print("✅ Визуализация завершена!")

# КЛАССЫ FASHIONMNIST (добавьте перед вызовом)
fashion_classes = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}

# 🚀 ЗАПУСК ВИЗУАЛИЗАЦИИ ПОСЛЕ ОБУЧЕНИЯ
plot_attention_masks(model, test_loader, num_samples=8)

print("🎉 Обучение и визуализация attention завершены!")
