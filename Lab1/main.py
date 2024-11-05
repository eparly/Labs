## Note to reader - this file was used for testing and development purposes
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torchsummary.torchsummary
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchsummary import torchsummary
from model import autoencoderMLP4Layer
from train import train
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    model = autoencoderMLP4Layer(N_bottleneck=8).to(device)
    torchsummary.summary(model, (1, 28*28))
    n_epochs = 50
    batch_size = 2048
    learning_rate = 0.001
    
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    losses_train = train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device)
    torch.save(model.state_dict(), 'model.pth')
    
    plt.plot(losses_train)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')
    plt.show()
    
    model = autoencoderMLP4Layer(N_bottleneck=8)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST('./data/mnist', train=False, download=True, transform=transform)
    while(1):
        
        input_idx = int(input('Enter value'))
        input_img, _ = dataset[input_idx]
        print(input_img)
        input_img = input_img.view(1, -1).to(torch.float32)
        
        with torch.no_grad():
            output_img = model(input_img)
            
        input_img = input_img.view(28, 28).cpu().numpy()
        output_img = output_img.view(28, 28).cpu().numpy()
        
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(input_img, cmap='gray')
        plt.title('Input Image')
        f.add_subplot(1, 2, 2)
        plt.imshow(output_img, cmap='gray')
        plt.title('Output Image')
        plt.show()
    
    
    
main()
    