import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Assuming the dataset is already loaded
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Define the transformation for targets
target_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.int64))
])

# Use the same dataset setup you have
train_dataset = datasets.VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=transform, target_transform=target_transform)
# Create a DataLoader for batching
batch_size = 32  # Choose an appropriate batch size
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Initialize an array to store class frequencies
num_classes = 21  # VOC has 21 classes, including background
class_frequencies = np.zeros(num_classes)

# Iterate through the dataset
for idx, (images, targets) in enumerate(dataloader):
    print(f"Batch {idx} / {len(dataloader)}")
    # Flatten the target mask to a 1D tensor (for each batch)
    targets = targets.view(-1)  # Flatten the tensor (height * width * batch_size)

    # Mask out class 255 (ignore pixels)
    targets = targets[targets != 255]  # Only keep pixels that are not 255
    # Count occurrences of each class in the target mask
    unique_classes, counts = torch.unique(targets, return_counts=True)
    
    # Accumulate the counts for each class
    for cls, count in zip(unique_classes, counts):
        class_frequencies[cls.item()] += count.item()

# Normalize class frequencies to get the relative frequency of each class
total_pixels = np.sum(class_frequencies)
class_weights = class_frequencies / total_pixels

# Print out the class frequencies and weights
print(f"Class Frequencies: {class_frequencies}")
print(f"Class Weights (Normalized): {class_weights}")
