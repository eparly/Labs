import argparse
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import matplotlib.pyplot as plt
import numpy as np
from model2 import DilatedSkipNet, FeatureDilatedSkipNet

class_colors = [
    [0, 0, 0],         # Background (0)
    [128, 0, 0],       # Aeroplane (1)
    [0, 128, 0],       # Bicycle (2)
    [128, 128, 0],     # Bird (3)
    [0, 0, 128],       # Boat (4)
    [128, 0, 128],     # Bottle (5)
    [0, 128, 128],     # Bus (6)
    [128, 128, 128],   # Car (7)
    [64, 0, 0],        # Cat (8)
    [192, 0, 0],       # Chair (9)
    [64, 128, 0],      # Cow (10)
    [192, 128, 0],     # Dining table (11)
    [64, 0, 128],      # Dog (12)
    [192, 0, 128],     # Horse (13)
    [64, 128, 128],    # Motorbike (14)
    [192, 128, 128],   # Person (15)
    [0, 64, 0],        # Potted plant (16)
    [128, 64, 0],      # Sheep (17)
    [0, 192, 0],       # Sofa (18)
    [128, 192, 0],     # Train (19)
    [0, 64, 128],      # TV monitor (20)
]

ignore_color = [128, 128, 128] 
class_colors = np.vstack([class_colors, ignore_color])
class_colors = np.array(class_colors, dtype=np.uint8)

def label_to_color_image(label_mask):
    """Converts a label mask into a color image."""
    label_mask = label_mask.clip(0, 20)
    colour_image = class_colors[label_mask]
    # Convert label_mask to a color image by applying the class_colors map
    return colour_image

def visualize_predictions(images, targets, outputs, epoch):
    """Visualizes images, ground truths, and predictions."""
    images = images.cpu()
    preds = torch.argmax(outputs, dim=1).cpu()

    for i in range(min(len(images), 3)): 
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        gt_label = targets[i]
        pred_label = preds[i]

        gt_color = label_to_color_image(gt_label)
        pred_color = label_to_color_image(pred_label)

        # Plot the images
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(gt_color)
        ax[1].set_title('Ground Truth')
        ax[1].axis('off')
        
        ax[2].imshow(pred_color)
        ax[2].set_title('Predictions')
        ax[2].axis('off')
        
        print('saving ', f"feature_{epoch}_{i}.png")
        plt.savefig(f"feature_{epoch}_{i}.png")
        plt.close(fig)
        

def evaluate_model(model, dataloader, device, num_classes=21):
    model.to(device)
    model.eval()
    miou_list = []
    intersection_sum = np.zeros(num_classes)
    union_sum = np.zeros(num_classes)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            targets = targets.squeeze(0).cpu().numpy()
            
            ious = []
            for cls in range(num_classes):
                pred_inds = preds == cls
                target_inds = targets == cls
                intersection = np.logical_and(pred_inds, target_inds).sum()
                union = np.logical_or(pred_inds, target_inds).sum()
                
                intersection_sum[cls] += intersection
                union_sum[cls] += union
            if(idx % 25 == 0):
                print(f'Batch {idx} / {len(dataloader)}')
                visualize_predictions(images, targets, outputs, idx)
    iou_per_class = intersection_sum / (union_sum + 1e-6)
            
    valid_classes = ~np.isnan(iou_per_class)
    for i in range(num_classes):
        print(f'Class {i} IoU: {iou_per_class[i]}')
    miou = np.mean(iou_per_class[valid_classes])
    return miou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dilated_skip_net.pth')
    args  = parser.parse_args()
    model_path = args.model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if(model_path == 'dilated_net.pth' or model_path == 'dilated_net_response.pth'):
        model = DilatedSkipNet().to(device)
    elif(model_path == 'feature_dilated_net.pth'):
        model = FeatureDilatedSkipNet().to(device)
    else:
        raise ValueError('Invalid model path')
    batch_size = 4
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
    ])


    test_dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=True,
                                transform=transform, target_transform=target_transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.load_state_dict(torch.load(model_path))
    miou = evaluate_model(model, test_loader, device)
    print(f'mIoU: {miou}')
    
