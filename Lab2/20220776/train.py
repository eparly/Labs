import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

from dataset import CustomImageDataset
from model import CustomModel
from torchvision.transforms import v2

#step 2.3 - Training

def train(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, scheduler, device, save_file=None, plot_file=None):
    print('training...')
    model.train()
    losses_train = []
    losses_val = []
        
    for epoch in range(1, n_epochs+1):
        print('Epoch:', epoch)
        loss_train = 0.0
        model.train()
        for data in train_loader: 
            optimizer.zero_grad()
            
            imgs = data[0].to(device=device)
            labels = data[1].to(device=device)
            
            outputs = model(imgs)
            
            loss = loss_fn(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        avg_loss = loss_train / len(train_loader)
        scheduler.step(loss_train)
        
        losses_train += [avg_loss]
        
        print(datetime.datetime.now(), 'epoch:', epoch, 'train loss:', avg_loss)
        
        loss_val = 0.0
        model.eval()
        for data in val_loader:
            imgs = data[0].to(device=device)
            labels = data[1].to(device=device)
            
            outputs = model(imgs)
            
            loss = loss_fn(outputs, labels)
            loss_val += loss.item()
        
        avg_loss = loss_val / len(val_loader)
        print(datetime.datetime.now(), 'epoch:', epoch, 'val loss:', avg_loss)
        
        losses_val += [avg_loss]
        if save_file != None:
            torch.save(model.state_dict(), save_file)

        if plot_file != None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='train')
            plt.plot(losses_val, label='val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)

def visualize_augmentation(original_img, transformed_img, original_label, transformed_label):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    transformed_img = transformed_img.permute(1, 2, 0).cpu().numpy()
    original_img = original_img.permute(1, 2, 0).cpu().numpy()
    original_img = (original_img * 255).astype(np.uint8)
    transformed_img = (transformed_img * 255).astype(np.uint8)

    axes[0].imshow(original_img) 
    axes[0].set_title(f'Original Label: {original_label.cpu().numpy()}')
    axes[0].axis('off')
    
    axes[1].imshow(transformed_img)  
    axes[1].set_title(f'Transformed Label: {transformed_label.cpu().detach().numpy()}')
    axes[1].axis('off')
    
    plt.show()
    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.001)

def target_transform(coords, scale_x, scale_y):
    x, y = coords
    x = x * scale_x
    y = y * scale_y
    return x, y

def loss_fn(outputs, labels):
    loss = torch.sqrt((outputs[:, 0] - labels[:, 0]) ** 2 + (outputs[:, 1] - labels[:, 1]) ** 2)
    return loss.mean()

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with specified augmentations.')
    parser.add_argument('--augmentation', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Specify the augmentation type: 0 (none), 1 (blur), 2 (horizontal flip), 3 (blur and flip)')
    parser.add_argument('--saveFile', type=str, default=None,
                        help='Specify the file to save the model.')
    parser.add_argument('--plotFile', type=str, default=None,
                        help='Specify the file to save the plot.')
    return parser.parse_args()

def flip_coords_horizontal(coords):
    x, y = coords
    updated_x = 227 - x
    return torch.tensor([updated_x, y])
    
    

def main():
    args = parse_args()
    save_file = 'model.pth'
    plot_file = 'plot.png'
    augmentation = 0
    if(args.augmentation != None):
        augmentation = args.augmentation
    if (args.saveFile != None):
        save_file = args.saveFile
    if (args.plotFile != None):
        plot_file = args.plotFile
        
    transforms = v2.Compose([
        v2.ToTensor()
    ])    
    
    #step 2.5 - Data augmentation
    if augmentation == 0:
        train_set = CustomImageDataset(annotations_file='oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.txt', img_dir = 'oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images', transform=transforms)
    elif augmentation == 1:
        original_dataset = CustomImageDataset(annotations_file='oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.txt', img_dir = 'oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images', transform=transforms)
        blur_transform = v2.Compose([
            v2.ToTensor(),
            v2.GaussianBlur(kernel_size=5)
        ])
        blur_dataset = CustomImageDataset(annotations_file='oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.txt', img_dir = 'oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images', transform=blur_transform)
        train_set = torch.utils.data.ConcatDataset([original_dataset, blur_dataset])
    elif augmentation == 2:
        original_dataset = CustomImageDataset(annotations_file='oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.txt', img_dir = 'oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images', transform=transforms)
        flip_transform = v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(p=1.0)
        ])
        flip_dataset = CustomImageDataset(annotations_file='oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.txt', img_dir = 'oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images', transform=flip_transform, target_transform=flip_coords_horizontal)
        train_set = torch.utils.data.ConcatDataset([original_dataset, flip_dataset])
    elif augmentation == 3:
        original_dataset = CustomImageDataset(annotations_file='oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.txt', img_dir = 'oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images', transform=transforms)
        flip_transform = v2.Compose([
            v2.ToTensor(),
            v2.RandomHorizontalFlip(p=1.0)
        ])
        flip_dataset = CustomImageDataset(annotations_file='oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.txt', img_dir = 'oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images', transform=flip_transform, target_transform=flip_coords_horizontal)
        blur_transform = v2.Compose([
            v2.ToTensor(),
            v2.GaussianBlur(kernel_size=5)
        ])
        blur_dataset = CustomImageDataset(annotations_file='oxford-iiit-pet-noses/oxford-iiit-pet-noses/train_noses.txt', img_dir = 'oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images', transform=blur_transform)
        train_set = torch.utils.data.ConcatDataset([original_dataset, flip_dataset, blur_dataset])
    device = 'cpu'
    
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)
    
    model = CustomModel()
    model.to(device=device)
    model.apply(init_weights)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    
    val_set = CustomImageDataset(annotations_file='oxford-iiit-pet-noses/oxford-iiit-pet-noses/test_noses.txt', img_dir = 'oxford-iiit-pet-noses/oxford-iiit-pet-noses/images-original/images', transform=transforms)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    train(n_epochs=100, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader, val_loader=val_loader, scheduler=scheduler, device=device, save_file=save_file, plot_file=plot_file)
    
if __name__ == '__main__':
    main()