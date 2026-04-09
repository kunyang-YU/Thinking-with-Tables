import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data import build_loaders
from model import build_resnet50

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    data_dir = "your_path/multimodalQA-used/skinca/dataset"
    batch_size = 32
    epochs = 10
    lr = 1e-3
    mode = "layer4"   # 可选：fc / layer4 / full

    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, classes = build_loaders(data_dir, batch_size)
    num_classes = len(classes)
    print(num_classes)

    model = build_resnet50(num_classes, mode=mode).to(device)
    criterion = nn.CrossEntropyLoss()

    # 只优化 requires_grad=True 的参数
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

    # torch.save(model.state_dict(), "resnet50_finetuned.pth")
