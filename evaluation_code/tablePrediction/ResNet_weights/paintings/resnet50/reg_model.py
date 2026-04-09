import torch.nn as nn
from torchvision import models

def build_resnet50_regressor(target_dim=1, mode="layer4"):
    """
    target_dim: 输出维度（回归值数量）
    mode: fc / layer4 / full
    """

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    if mode == "layer4":
        for p in model.layer4.parameters():
            p.requires_grad = True
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    # mode="fc": 默认只训练fc

    # Replace fc for regression
    model.fc = nn.Linear(model.fc.in_features, target_dim)

    return model
