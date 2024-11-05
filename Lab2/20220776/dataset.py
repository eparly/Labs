import os
import ast
import numpy as np
import pandas as pd
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image

import matplotlib.pyplot as plt

#Step 2.2 - Data
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image_formatted = image.convert('RGB').resize((227, 227))
        label = self.img_labels.iloc[idx, 1]
        x, y = ast.literal_eval(label)
        
        scale_x = 227 / image.size[0]
        scale_y = 227 / image.size[1]
        
        x = int(x * scale_x)
        y = int(y * scale_y)
        
        label = torch.tensor([x, y])

        if self.transform:
            image = self.transform(image_formatted)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    
def plot_image_with_label(image, label):
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    
    plt.imshow(image)
    plt.scatter(label[0].item(), label[1].item(), c='red', s=40, marker='x')
    plt.title(f'Label: ({label[0].item()}, {label[1].item()})')
    plt.axis('off')
    plt.show()


def visualize_augmentation(original_img, flip_img, blur_img, original_label, flip_label, blur_label):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    original_img = original_img.permute(1, 2, 0).cpu().numpy()
    flip_img = flip_img.permute(1, 2, 0).cpu().numpy()
    blur_img = blur_img.permute(1, 2, 0).cpu().numpy()
    original_img = (original_img * 255).astype(np.uint8)
    flip_img = (flip_img * 255).astype(np.uint8)
    blur_img = (blur_img * 255).astype(np.uint8)
    
    axes[0].imshow(original_img)  
    axes[0].set_title(f'Original Label: {original_label.cpu().numpy()}')
    axes[0].axis('off')
    
    axes[1].imshow(flip_img)
    axes[1].set_title(f'Flipped Label: {flip_label.cpu().detach().numpy()}')
    axes[1].axis('off')
    
    axes[2].imshow(blur_img)
    axes[2].set_title(f'Blurred Label: {blur_label.cpu().detach().numpy()}')
    axes[2].axis('off')
    
    plt.show()
    
def flip_coords_horizontal(coords):
    x, y = coords
    updated_x = 227 - x
    return torch.tensor([updated_x, y])

# used to check that the dataset and data loader are working correctly
def main():
    transform = v2.Compose([
        v2.Resize((227, 227)),
        v2.ToTensor()
    ])

    flip_transform = v2.Compose([
        v2.Resize((227, 227)),
        v2.RandomHorizontalFlip(p=1.0),
        v2.ToTensor()
    ])

    blur_transform = v2.Compose([
        v2.Resize((227, 227)),
        v2.GaussianBlur(kernel_size=5),
        v2.ToTensor()
    ])

    annotations_file = 'oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.txt'
    img_dir = 'oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images'
    
    original_dataset = CustomImageDataset(annotations_file, img_dir, transform=transform)
    flip_dataset = CustomImageDataset(annotations_file, img_dir, transform=flip_transform, target_transform=flip_coords_horizontal)
    blur_dataset = CustomImageDataset(annotations_file, img_dir, transform=blur_transform)

    for i in range(5):
        original_img, original_label = original_dataset[i]
        flip_img, flip_label = flip_dataset[i]
        blur_img, blur_label = blur_dataset[i]
        visualize_augmentation(original_img, flip_img, blur_img, original_label, flip_label, blur_label)

if __name__ == '__main__':
    main()