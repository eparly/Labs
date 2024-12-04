import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import numpy as np
from torchvision.models.segmentation import fcn_resnet50
from datetime import datetime

from matplotlib import pyplot as plt

from model2 import FeatureDilatedSkipNet


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

def visualize_predictions(images, targets, student_output, teacher_output, epoch):
    """Visualizes images, ground truths, and predictions."""
    images = images.cpu()
    targets = targets.cpu()
    student_preds = torch.argmax(student_output, dim=1).cpu()
    teacher_preds = torch.argmax(teacher_output, dim=1).cpu()

    for i in range(min(len(images), 3)):  # Visualize up to 3 examples
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        gt_label = targets[i].cpu().numpy()
        student_pred_label = student_preds[i].cpu().numpy()
        teacher_pred_label = teacher_preds[i].cpu().numpy()

        gt_color = label_to_color_image(gt_label)
        student_pred_color = label_to_color_image(student_pred_label)
        teacher_pred_color = label_to_color_image(teacher_pred_label)

        # Plot the images
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        ax[0].imshow(image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(gt_color)
        ax[1].set_title('Ground Truth')
        ax[1].axis('off')

        ax[2].imshow(teacher_pred_color)
        ax[2].set_title('Teacher Predictions')
        ax[2].axis('off')

        ax[3].imshow(student_pred_color)
        ax[3].set_title('Predictions')
        ax[3].axis('off')

        print('saving ', f"segmentation_{epoch}_{i}.png")
        plt.savefig(f"segmentation_{epoch}_{i}.png")
        plt.close(fig)


def cosine_similarity_loss(student_features, teacher_features, temperature=1.0):
    """
    Compute the cosine similarity loss between student and teacher features.
    
    Args:
        student_features: Feature maps from the student model (N, C, H, W)
        teacher_features: Feature maps from the teacher model (N, C, H, W)
        temperature: Scaling factor for the loss (default 1.0)
    
    Returns:
        Cosine similarity loss between the student and teacher features.
    """
    student_features_flat = student_features.view(student_features.size(0), student_features.size(1), -1)  # (N, C, H*W)
    teacher_features_flat = teacher_features.view(teacher_features.size(0), teacher_features.size(1), -1)  # (N, C, H*W)

    student_features_flat = F.normalize(student_features_flat, p=2, dim=2)  
    teacher_features_flat = F.normalize(teacher_features_flat, p=2, dim=2)

    cosine_sim = torch.sum(student_features_flat * teacher_features_flat, dim=2) 

    loss = 1 - cosine_sim.mean()  # Mean cosine similarity loss
    
    # Scale by temperature
    return loss * temperature


def train_distillation(student_model, teacher_model, train_dataloader, val_dataloader, num_epochs=10, alpha=0.7, beta=0.3, temperature=2.0, lr=1e-4, device='cuda'):
    optimizer = optim.Adam(student_model.parameters(), lr=lr, weight_decay = 0.0005)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=255)  # For segmentation
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    teacher_model.eval() 
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        student_model.train()  
        running_loss = 0.0

        # Training loop
        for i, (images, targets) in enumerate(train_dataloader):
            if(i%25==0):
              time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
              print(f"batch: {i} of {len(train_dataloader)} - {time}")
            images, targets = images.to(device), targets.to(device)
            
            
            with torch.no_grad():
                teacher_dict = teacher_model(images) # need to do forward pass with teacher model to get feature maps
                
            
            student_output = student_model(images)
            
            ce_loss = ce_loss_fn(student_output, targets)
            
            # Compute the cosine similarity loss (distillation loss)
            feature_loss = 0
            # adjust layers as needed. Final model only used layer1, layer2, layer3 (enc1, enc2, enc3)
            for student_layer, teacher_layer in zip(['enc1', 'enc2', 'enc3', 'bottleneck'], ['layer1', 'layer2', 'layer3', 'layer4']):
                student_features = student_model.feature_maps[student_layer]
                teacher_features = teacher_feature_maps[teacher_layer]
                
                conv1x1 = nn.Conv2d(teacher_features.shape[1], student_features.shape[1], kernel_size=1)
                conv1x1 = conv1x1.to(device)
                # Apply the 1x1 convolution to teacher features to match channel depth
                teacher_features_adjusted = conv1x1(teacher_features)
                teacher_features_resized = F.interpolate(teacher_features_adjusted, size=(student_features.shape[2], student_features.shape[3]), mode='bilinear', align_corners=False)

                student_features_flat = student_features.view(student_features.size(0), -1)
                teacher_features_flat = teacher_features_resized.view(teacher_features_resized.size(0), -1)

                # Calculate cosine similarity
                feature_loss += 1 - F.cosine_similarity(student_features_flat, teacher_features_flat, dim=1).mean()
                                
            total_loss = alpha * ce_loss + beta * feature_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()


        train_loss = running_loss / len(train_dataloader)
        train_losses.append(train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")

        val_loss = validate_model(student_model, teacher_model, val_dataloader, ce_loss_fn, alpha, beta, temperature, epoch, device)
        val_losses.append(val_loss)

        scheduler.step(train_loss)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"ce loss: {ce_loss}, response loss: {feature_loss}")
        torch.save(student_model.state_dict(), f'student_model.pth')
        if plot_file != None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(train_losses, label='train')
            plt.plot(val_losses, label='val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)

# Validation loop
def validate_model(student_model, teacher_model, val_dataloader, ce_loss_fn, alpha, beta, temperature, epoch, device='cuda'):
    student_model.eval() 
    running_val_loss = 0.0
    
    with torch.no_grad():
        for images, targets in val_dataloader:
            images, targets = images.to(device), targets.to(device)

            student_output = student_model(images)
            teacher_output = teacher_model(images)['out']
            
            ce_loss = ce_loss_fn(student_output, targets)
            feature_loss = 0
            for student_layer, teacher_layer in zip(['enc1', 'enc2', 'enc3', 'bottleneck'], ['layer1', 'layer2', 'layer3', 'layer4']):
                student_features = student_model.feature_maps[student_layer]
                teacher_features = teacher_feature_maps[teacher_layer]
                
                conv1x1 = nn.Conv2d(teacher_features.shape[1], student_features.shape[1], kernel_size=1)
                conv1x1 = conv1x1.to(device)
                # Apply the 1x1 convolution to teacher features to match channel depth
                teacher_features_adjusted = conv1x1(teacher_features)
                teacher_features_resized = F.interpolate(teacher_features_adjusted, size=(student_features.shape[2], student_features.shape[3]), mode='bilinear', align_corners=False)

                
                student_features_flat = student_features.view(student_features.size(0), -1)
                teacher_features_flat = teacher_features_resized.view(teacher_features_resized.size(0), -1)

                # Calculate cosine similarity
                feature_loss += 1 - F.cosine_similarity(student_features_flat, teacher_features_flat, dim=1).mean()  
            total_loss = alpha * ce_loss + beta * feature_loss
            running_val_loss += total_loss.item()


    avg_val_loss = running_val_loss / len(val_dataloader)
    visualize_predictions(images, targets, student_output, teacher_output, epoch)
    return avg_val_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plot_file = 'loss_plot.png'
batch_size = 16

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
])

train_dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=True,
                                 transform=transform, target_transform=target_transform)

val_dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=True,
                               transform=transform, target_transform=target_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Define models
student_model = FeatureDilatedSkipNet() 
student_model = student_model.to(device)


teacher_model = fcn_resnet50(pretrained=True, num_classes=21)  # Example teacher model

teacher_feature_maps = {}
# gets intermediate feature maps from resnet50
def hook(module, input, output, layer_name):
    teacher_feature_maps[layer_name] = output

hooks = []

# register_forward_hook allows us to get the intermediate feature maps from the teacher model
for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
    hooks.append(getattr(teacher_model.backbone, layer_name).register_forward_hook(
        lambda module, input, output, name=layer_name: hook(module, input, output, name)
    ))


teacher_model = teacher_model.to(device)
for param in teacher_model.parameters():
    param.requires_grad = False  # Freeze teacher weights

epochs = 100
train_distillation(student_model, teacher_model, train_loader, val_loader)

for h in hooks:
    h.remove()
