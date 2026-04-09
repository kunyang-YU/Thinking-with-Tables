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
