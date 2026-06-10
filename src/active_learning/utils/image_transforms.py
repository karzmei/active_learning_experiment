import numpy as np
import torch
from torchvision import transforms


def model_preprocessing_transform(image):
    #PIL image proprocessing for resnet18 pretrained on imagenet
    transform = torch.nn.Sequential(
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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