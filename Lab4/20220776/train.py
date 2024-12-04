from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import matplotlib.pyplot as plt
import numpy as np
from dice_loss import CombinedLoss
from model2 import DilatedSkipNet


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

# Training and Validation Loop
def train_and_validate():
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for idx, (images, targets) in enumerate(train_loader):
            if(idx % 50 == 0):
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"Batch {idx} / {len(train_loader)} time: {time}")
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")


        if save_file != None:
            torch.save(model.state_dict(), save_file)

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
        # Adjust learning rate
        scheduler.step()

        # Visualize a few results
        if (epoch + 1) % 5 == 0:
            visualize_predictions(images, targets, outputs, epoch)


def visualize_predictions(images, targets, outputs, epoch):
    """Visualizes images, ground truths, and predictions."""
    images = images.cpu()
    targets = targets.cpu()
    preds = torch.argmax(outputs, dim=1).cpu()

    for i in range(min(len(images), 3)):  # Visualize up to 3 examples
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        gt_label = targets[i].cpu().numpy()
        pred_label = preds[i].cpu().numpy()

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

        print('saving ', f"segmentation_{epoch}_{i}.png")
        plt.savefig(f"segmentation_{epoch}_{i}.png")
        plt.close(fig)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plot_file = 'loss_plot.png'
    save_file = 'dilated_skip_net.pth'
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.01
    num_classes = 21  
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
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

    base_weights = [0.00063867, 0.06298626, 0.15730574, 0.05390672, 0.07562841,
        0.07695767, 0.02623267, 0.03254918, 0.01731525, 0.04047321,
        0.05522139, 0.03563649, 0.02753334, 0.05005508, 0.04087912,
        0.00969805, 0.07170607, 0.05241907, 0.03215133, 0.02922783,
        0.05147845]

    base_weights_tensor = torch.tensor(base_weights)

    background_weight_factor = 2
    scaled_weights = base_weights_tensor.clone()
    scaled_weights[0] *= background_weight_factor
    scaled_weights = scaled_weights.to(device)

    # Model, Loss, Optimizer, Scheduler
    model = DilatedSkipNet().to(device)
    criterion = CombinedLoss(alpha=1, beta=0).to(device) #fully ce loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 0.005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    train_and_validate()
