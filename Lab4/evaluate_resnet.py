import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import numpy as np

def evaluate_model(model, dataloader, device, num_classes=21):
    model.to(device)
    model.eval()
    miou_list = []
    intersection_sum = np.zeros(num_classes)
    union_sum = np.zeros(num_classes)

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            if(idx % 1 == 0):
                print(f'Batch {idx} / {len(dataloader)}')
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)['out']
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
    iou_per_class = intersection_sum / (union_sum + 1e-6)
            
    valid_classes = ~np.isnan(iou_per_class)
    for i in range(num_classes):
        print(f'Class {i} IoU: {iou_per_class[i]}')
    miou = np.mean(iou_per_class[valid_classes])
    return miou

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the pretrained FCN-ResNet50 model
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
    model.eval()

    # Define the transformation for images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define the transformation for targets
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.int64))
    ])

    # Download the PASCAL VOC 2012 dataset
    dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=True, transform=transform, target_transform=target_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Evaluate the model using the mean Intersection over Union (mIoU) metric
    miou = evaluate_model(model, dataloader, device)
    print(f'mIoU: {miou}')

if __name__ == "__main__":
    main()