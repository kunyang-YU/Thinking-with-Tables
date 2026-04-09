import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from regression_dataset import RegressionDataset
from reg_model import build_resnet50_regressor
import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes, mode="fc"):  
    """
    mode:
        fc  -> 只训练最后的全连接层
        layer4 -> 微调 layer4 + fc
        full -> 全量端到端微调
    """

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    if mode == "fc":
        pass  # 默认冻结 backbone

    elif mode == "layer4":
        for p in model.layer4.parameters():
            p.requires_grad = True

    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


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


def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)          # (B, C)
            preds = logits.argmax(dim=1)  # 预测类别

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return acc




def ResNet50Predict(task, image_path, 
                    weight_path="resnet50_regression.pth",
                    target_dim=1,
                    device=None):
    """
    task: 微调策略，例如 "fc" / "layer4" / "full"
    image_path: 输入图片路径
    weight_path: 模型权重路径
    return: logit(float)
    """
    print(task)
    # device
    if task == 'adoption' or task == 'skinca':
        weight_path = f'/ssd/yuky/TwT/data/multimodalQA-used/{task}/resnet/resnet50_finetuned.pth'
    elif task == 'pawpularity' or task == 'paintings':
        weight_path = f'/ssd/yuky/TwT/data/multimodalQA-used/{task}/resnet50/resnet50_regression.pth'
    else:
        raise NotImplementedError

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 构建 val transform
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 2. 加载模型
    if task == 'adoption' or task == 'skinca':
        model = build_resnet50(5 if task == 'adoption' else 6, task)
    elif task == 'pawpularity' or task == 'paintings':
        model = build_resnet50_regressor(target_dim, task)
    else:
        raise NotImplementedError
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    # # 3. 读取图片
    # img = Image.open(image_path).convert("RGB")
    # img = val_tf(img).unsqueeze(0).to(device)

    # # 4. 推理
    # with torch.no_grad():
    #     logits = model(img)

    # # 回归输出 → float
    # if task == 'adoption' or task == 'skinca':
        # return logits
    return model
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms, datasets
def build_loaders(data_dir, batch_size=32, num_workers=4):

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

    train_set = datasets.ImageFolder(f"{data_dir}/train", train_tf)
    val_set = datasets.ImageFolder(f"{data_dir}/test", val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_set.classes


if __name__ == "__main__":
    data_dir = "/ssd/yuky/TwT/data/multimodalQA-used/adoption/images"
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

    train_loader, val_loader, classes = build_loaders(data_dir, batch_size)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model =ResNet50Predict('adoption','')
    criterion = nn.MSELoss()    # 回归任务常用
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training loop
    for epoch in range(epochs):
        # train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss=, val_loss={val_loss:.4f}")

    # torch.save(model.state_dict(), "resnet50_regression.pth")
