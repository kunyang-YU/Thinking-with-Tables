import torch
from PIL import Image
from torchvision import transforms
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
    if task == 'adoption':
        weight_path = f'/ssd/yuky/TwT/data/multimodalQA-used/{task}/resnet/resnet50_finetuned.pth'
    elif task == 'pawpularity':
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
    if task == 'adoption':
        model = build_resnet50(5, task)
    elif task == 'pawpularity':
        model = build_resnet50_regressor(target_dim, task)
    else:
        raise NotImplementedError
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. 读取图片
    img = Image.open(image_path).convert("RGB")
    img = val_tf(img).unsqueeze(0).to(device)

    # 4. 推理
    with torch.no_grad():
        logits = model(img)

    # 回归输出 → float
    if task == 'adoption':
        return logits
    return logits.squeeze().cpu().item()
