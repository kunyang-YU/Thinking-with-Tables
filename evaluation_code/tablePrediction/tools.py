import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models
def build_resnet50_regressor(target_dim=1, mode="layer4"):
    """
    target_dim: output dimension (number of regression values)
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
    # mode="fc": only train fc by default

    # Replace fc for regression
    model.fc = nn.Linear(model.fc.in_features, target_dim)

    return model


def build_resnet50(num_classes, mode="fc"):  
    """
    mode:
        fc  -> only train the last fully connected layer
        layer4 -> fine-tune layer4 + fc
        full -> full end-to-end fine-tuning
    """

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    if mode == "fc":
        pass  # freeze backbone by default

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
    task: fine-tuning strategy, e.g. "fc" / "layer4" / "full"
    image_path: input image path
    weight_path: model weight path
    return: logit(float)
    """
    print(task)
    # device
    if task == 'adoption':
        weight_path = f'your_path/data/multimodalQA-used/{task}/resnet/resnet50_finetuned.pth'
    elif task == 'pawpularity':
        weight_path = f'your_path/data/multimodalQA-used/{task}/resnet50/resnet50_regression.pth'
    else:
        raise NotImplementedError

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Build val transform
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 2. Load model
    if task == 'adoption':
        model = build_resnet50(5, task)
    elif task == 'pawpularity':
        model = build_resnet50_regressor(target_dim, task)
    else:
        raise NotImplementedError
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Read image
    img = Image.open(image_path).convert("RGB")
    img = val_tf(img).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        logits = model(img)

    # Regression output -> float
    if task == 'adoption':
        return logits
    return logits.squeeze().cpu().item()
