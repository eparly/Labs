from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import numpy as np
from torchvision.models.segmentation import fcn_resnet50
from datetime import datetime

from model2 import DilatedSkipNet

# Define distillation loss
def distillation_loss(student_output, teacher_output, target, alpha, beta, temperature):
    """
    Compute the distillation loss.

    Args:
        student_output: Logits from the student model (N, C, H, W)
        student_features: Features from the student model (optional, if needed)
        teacher_output: Logits from the teacher model (N, C, H, W)
        teacher_features: Features from the teacher model (optional, if needed)
        target: Ground-truth segmentation mask (N, H, W)
        alpha: Weight for supervised loss
        temperature: Temperature for softening logits
    """

    #Note: the weights were not used when training the final version of the model. However, they can be used to balance the classes.
    base_weights = [0.00063867, 0.06298626, 0.15730574, 0.05390672, 0.07562841,
       0.07695767, 0.02623267, 0.03254918, 0.01731525, 0.04047321,
       0.05522139, 0.03563649, 0.02753334, 0.05005508, 0.04087912,
       0.00969805, 0.07170607, 0.05241907, 0.03215133, 0.02922783,
       0.05147845]

    base_weights_tensor = torch.tensor(base_weights)

    background_weight_factor = 1
    scaled_weights = base_weights_tensor.clone()
    scaled_weights[0] *= background_weight_factor
    scaled_weights = scaled_weights.to(device)


    ce = nn.CrossEntropyLoss(ignore_index=255)
    ce_loss = ce(student_output, target) 

   
    soft_student = nn.functional.log_softmax(student_output / temperature, dim=1)
    soft_teacher = nn.functional.softmax(teacher_output / temperature, dim=1)
    response_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * temperature**2
    
    
    # Total Loss
    total_loss = alpha * ce_loss + beta * response_loss

    # print(f"CE Loss: {ce_loss.item()}, Response Loss: {response_loss.item()}")
    return total_loss, ce_loss, response_loss


# Training Step
def train_step(student_model, teacher_model, dataloader, optimizer, scheduler, alpha, beta, temperature, device):
    """
    Perform a single training step with distillation loss.

    Args:
        student_model: The student model being trained
        teacher_model: The pre-trained teacher model
        dataloader: DataLoader for training data
        optimizer: Optimizer for the student model
        alpha: Weight for supervised loss
        temperature: Temperature for distillation
        device: Device for computation (CPU or GPU)
    """
    student_model.train()
    teacher_model.eval()  
    total_loss = 0.0
    total_ce_loss = 0.0
    total_response_loss = 0.0
    for idx, (images, targets) in enumerate(dataloader):
        if(idx%25==0):
          time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
          print(f"batch: {idx} of {len(dataloader)} - {time}")
        images, targets = images.to(device), targets.to(device)


        student_output = student_model(images)  

       
        with torch.no_grad():
            teacher_dict = teacher_model(images)
            teacher_output = teacher_dict['out']  # Logits from teacher model

        # Compute distillation loss
        loss, ce_loss, response_loss = distillation_loss(
            student_output, teacher_output, targets, alpha, beta, temperature
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_ce_loss += ce_loss.item()
        total_response_loss += response_loss.item()
        total_loss += loss.item()
    return total_loss / len(dataloader), total_ce_loss/len(dataloader), total_response_loss/len(dataloader)

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

def validate_step(student_model, teacher_model, dataloader, alpha, beta, temperature, device, epoch):
    student_model.eval()
    teacher_model.eval()  
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            student_output = student_model(images)

            teacher_dict = teacher_model(images)
            teacher_output = teacher_dict['out']  # Logits from teacher model

            # Compute distillation loss
            loss, _, _ = distillation_loss(
                student_output, teacher_output, targets, alpha, beta, temperature
            )


            total_loss += loss.item()
        if epoch % 1 == 0:
            visualize_predictions(images, targets, student_output, teacher_output, epoch)

    return total_loss / len(dataloader)


def train_model(student_model, teacher_model, train_loader, val_loader, optimizer, scheduler, epochs, temperature, device):
    train_losses = []
    val_losses = []
    alpha = 5 #used to scale the cross entropy loss
    beta = 0.00005 # used to scale the response loss

    for epoch in range(epochs):
        # after some time training, we want to focus more on the response loss
        if(epoch == 25):
          alpha = 1
          beta = 0.00005
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training Step
        train_loss, ce_loss, response_loss = train_step(student_model, teacher_model, train_loader, optimizer, scheduler, alpha, beta, temperature, device)
        train_losses.append(train_loss)

        # Validation Step
        val_loss = validate_step(student_model, teacher_model, val_loader, alpha, beta, temperature, device, epoch)
        val_losses.append(val_loss)

        scheduler.step(train_loss)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"ce loss: {ce_loss}, response loss: {response_loss}")
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

    return train_losses, val_losses


# Example Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plot_file = 'loss_plot.png'
batch_size = 32

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
])

# Dataset and DataLoader
train_dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=True,
                                 transform=transform, target_transform=target_transform)

val_dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=True,
                               transform=transform, target_transform=target_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Define models
student_model = DilatedSkipNet()
student_model = student_model.to(device)


optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)

teacher_model = fcn_resnet50(pretrained=True, num_classes=21) 
teacher_model = teacher_model.to(device)
for param in teacher_model.parameters():
    param.requires_grad = False  # Freeze teacher weights

temperature = 5 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

epochs = 100
train_model(student_model, teacher_model, train_loader, val_loader, optimizer, scheduler, epochs, temperature, device)
