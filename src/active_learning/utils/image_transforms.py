import numpy as np
import torch
from torchvision import transforms


def model_preprocessing_transform(image):
    transform = torch.nn.Sequential(
        torch.nn.Lambda(lambda x: torch.from_numpy(np.array(x)).float()),  # Convert PIL image to tensor
        torch.nn.Lambda(lambda x: x.permute(2, 0, 1)),  # Change from HWC to CHW format
        torch.nn.Lambda(lambda x: x / 255.0),  # Normalize pixel values to [0, 1]
        torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    )
    return transform(image)


def train_augmentations(image, prob=0.2):
    color_transforms = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)
    
    geometric_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=prob),
        transforms.RandomVerticalFlip(p=prob),
        # transforms.RandomRotation(degrees=15, fill=(0, 0, 0)),
    ])

    combined_transforms = transforms.Compose([
        color_transforms,
        geometric_transforms,
    ])

    return combined_transforms(image)