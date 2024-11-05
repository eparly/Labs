import argparse
from dataset import CustomImageDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as v2
from PIL import Image
import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


from model import CustomModel

# Step 2.4 - Testing
def evaluate_model(model, test_loader, device, show_images=False):
    model.eval()
    distances = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            distance = torch.sqrt((outputs[:, 0] - labels[:, 0]) ** 2 + (outputs[:, 1] - labels[:, 1]) ** 2)
            distances.extend(distance.cpu().numpy())
            if show_images:     
                for i in range(images.size(0)):
                    img = images[i].cpu().permute(1, 2, 0).numpy() 
                    img = (img * 255).astype(np.uint8) 
                    plt.imshow(img)
                    plt.scatter([outputs[i, 0].cpu().numpy()], [outputs[i, 1].cpu().numpy()], color='red')  # Denormalize coordinates
                    plt.scatter([labels[i, 0].cpu().numpy()], [labels[i, 1].cpu().numpy()], color='green')  # Denormalize coordinates
                    plt.title(f'Predicted: ({outputs[i, 0].item():.2f}, {outputs[i, 1].item():.2f})')
                    plt.show()
    
    distances = np.array(distances)
    min_distance = np.min(distances)
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    std_distance = np.std(distances)
    
    print(f'Min Distance: {min_distance:.4f}')
    print(f'Mean Distance: {mean_distance:.4f}')
    print(f'Max Distance: {max_distance:.4f}')
    print(f'Std Distance: {std_distance:.4f}')
    

def target_transform(coords, scale_x, scale_y):
    x, y = coords
    x = x * scale_x
    y = y * scale_y
    return x, y

def parse_args():
    parser = argparse.ArgumentParser(description='Test a model')
    parser.add_argument('--model', type=str, default='model.pth', help='Path to the model')
    parser.add_argument('--showImages',type=str, default = 'f', help='Display images with predicted and ground truth labels')    
    return parser.parse_args()


def main():
    args = parse_args()
    show_images = False
    if args.showImages == 't':
        show_images = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\t\tusing device ', device)
    
    model = CustomModel()
    model.to(device=device)
    
    transform = v2.Compose([
        v2.ToTensor()
    ])

    model.load_state_dict(torch.load(args.model))
    
    test_dataset = CustomImageDataset(
        annotations_file='oxford-iiit-pet-noses/oxford-iiit-pet-noses/test_noses.txt',
        img_dir='oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images',
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    print(datetime.datetime.now())
    evaluate_model(model, test_loader, device, show_images=show_images)
    print(datetime.datetime.now())

if __name__ == "__main__":
    main()