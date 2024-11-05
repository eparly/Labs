import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt

# Define transformations
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# Load CIFAR100 dataset
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define models
alexnet = models.alexnet()
vgg16 = models.vgg16()
resnet18 = models.resnet18()

# Modify the final layer to match CIFAR100 classes
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 100)
vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 100)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 100)

# Define training and validation functions
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
            labels = labels.long()
            loss = loss_fn(outputs, labels)
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


# Train models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = [alexnet, vgg16, resnet18]
criterion = nn.CrossEntropyLoss()

for model in models:
    print('training', model.__class__.__name__)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss, val_loss = train(n_epochs=50, optimizer=optimizer, model=model, loss_fn=criterion, train_loader=trainloader, val_loader=testloader, scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min'), device=device, save_file=f'{model.__class__.__name__}.pth', plot_file=f'{model.__class__.__name__}.png')
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(f'{model.__class__.__name__} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{model.__class__.__name__}_loss.png')