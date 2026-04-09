import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from regression_dataset import RegressionDataset
from reg_model import build_resnet50_regressor

def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return train_tf, val_tf


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, targets in tqdm(loader):
        imgs, targets = imgs.to(device), targets.to(device)

        preds = model(imgs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    data_dir = "your_path/multimodalQA-used/pawpularity/dataset"
    train_file = f"{data_dir}/train.txt"
    val_file = f"{data_dir}/test.txt"

    batch_size = 32
    epochs = 20
    lr = 1e-4
    target_dim = 1   # 回归任务输出维度
    mode = "layer4"  # 微调策略：fc / layer4 / full

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    train_tf, val_tf = build_transforms()

    train_set = RegressionDataset(data_dir, train_file, train_tf, target_dim)
    val_set = RegressionDataset(data_dir, val_file, val_tf, target_dim)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = build_resnet50_regressor(target_dim, mode).to(device)

    criterion = nn.MSELoss()    # 回归任务常用
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training loop
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # torch.save(model.state_dict(), "resnet50_regression.pth")
