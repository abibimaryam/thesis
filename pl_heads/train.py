import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from vitpol import ViT



def main():

    model = ViT(img_size=32, patch_size=8, in_channels=3, num_classes=10)
    print(model)

    batch_size=64

    # transform = transforms.ToTensor()
    # train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    # test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)


    # Загрузка данных
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),   # mean для CIFAR-10
            (0.2023, 0.1994, 0.2010)    # std для CIFAR-10
        )
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='../data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='../data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Модель, оптимизатор, loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Обучение
    def train_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return total_loss / len(loader), correct / total

    def test_epoch(model, loader, criterion, device):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return total_loss / len(loader), correct / total

    # Основной цикл обучения
    epochs = 50
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_test_loss = float('inf')

    log_file = open("training_log.txt", "w", encoding="utf-8")

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        scheduler.step()
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        log_file.write(
            f"Epoch {epoch +1 }: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}\n"
        )
        log_file.write(
            f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}\n"
        )
        log_file.flush()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_vit_fashion.pth')
            print(f'Эпоха {epoch+1}: Новая лучшая модель! Test Loss: {test_loss:.4f}')

        # if (epoch + 1) % 10 == 0:
        #     torch.save(model.state_dict(), f'vit_epoch_{epoch+1}.pth')
        #     print(f'Модель сохранена на эпохе {epoch+1}')
        


    print(f'Final Test Accuracy: {max(test_accs):.4f}')

if __name__ == "__main__":
    main()