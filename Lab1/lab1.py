#########################################################################################################
#
#   ELEC 475 - Lab 1, Step 4
#   Fall 2024
#

import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer

def main():

    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-z', metavar='bottleneck size', type=int, help='int [32]')

    args = argParser.parse_args()

    save_file = None
    if args.s != None:
        save_file = args.s
    bottleneck_size = 0
    if args.z != None:
        bottleneck_size = args.z

    device = 'cpu'
    # if torch.cuda.is_available():
    #     device = 'cuda'
    print('\t\tusing device ', device)

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = train_transform

    test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)

    N_input = 28 * 28   # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    idx = 0
    # ========================
    # Lab part 4 - Visualize the output of the autoencoder
    print('Lab part 4 - Visualize the output of the autoencoder')
    while idx >= 0:
        print('Enter negative number to move to next part')
        idx = input("Enter index > ")
        idx = int(idx)
        if 0 <= idx <= test_set.data.size()[0]:
            print('label = ', test_set.targets[idx].item())
            img = test_set.data[idx]
            print('break 9', img.shape, img.dtype, torch.min(img), torch.max(img))

            img = img.type(torch.float32)
            print('break 10', img.shape, img.dtype, torch.min(img), torch.max(img))
            img = (img - torch.min(img)) / torch.max(img)
            print('break 11', img.shape, img.dtype, torch.min(img), torch.max(img))


            img = img.to(device=device)
            print('break 8 : ', img.shape, img.dtype)
            img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)
            print('break 9 : ', img.shape, img.dtype)
            
            with torch.no_grad():
                output = model(img)

            output = output.view(28, 28).type(torch.FloatTensor)
            print('break 10 : ', output.shape, output.dtype)
            print('break 11: ', torch.max(output), torch.min(output), torch.mean(output))

            img = img.view(28, 28).type(torch.FloatTensor)

            f = plt.figure()
            f.add_subplot(1,2,1)
            plt.imshow(img, cmap='gray')
            f.add_subplot(1,2,2)
            plt.imshow(output, cmap='gray')
            plt.show()
    # ========================
    # Lab part 5 - Image denoising
    print('Lab part 5 - Image denoising')
    idx = 0
    while idx >= 0:
        print('Enter negative number to move to next part')
        idx = input("Enter index > ")
        idx = int(idx)
        if 0 <= idx <= test_set.data.size()[0]:
            print('label = ', test_set.targets[idx].item())
            img = test_set.data[idx]
            print('break 9', img.shape, img.dtype, torch.min(img), torch.max(img))

            img = img.type(torch.float32)
            print('break 10', img.shape, img.dtype, torch.min(img), torch.max(img))
            img = (img - torch.min(img)) / torch.max(img)
            print('break 11', img.shape, img.dtype, torch.min(img), torch.max(img))

            img = img.to(device=device)
            print('break 8 : ', img.shape, img.dtype)
            img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)
            print('break 9 : ', img.shape, img.dtype)
            
            #Add gaussian noise to image for denoising
            noise_factor = 0.2 # change this to see the effect of noise
            noisy_img = img + torch.randn(img.size()) * noise_factor
            
            with torch.no_grad():
                noisy_output = model(noisy_img)
            noisy_output = noisy_output.view(28, 28).type(torch.FloatTensor)

            img = img.view(28, 28).type(torch.FloatTensor)
            noisy_img = noisy_img.view(28, 28).type(torch.FloatTensor)

            f = plt.figure()
            f.add_subplot(1,3,1)
            plt.imshow(img, cmap='gray')
            f.add_subplot(1,3,2)
            plt.imshow(noisy_img, cmap='gray')
            f.add_subplot(1,3,3)
            plt.imshow(noisy_output, cmap='gray')
            plt.show()    
    
    # ========================
    # Lab part 6 - Interpolation
    print('Lab part 6 - Interpolation')
    index1 = 0
    while index1 >= 0:
        print('Enter negative number to move to next part')
        index1 = int(input('enter index for image 1 > '))
        index2 = int(input('enter index for image 2 > '))
        n_steps = int(input('enter number of steps > '))
        
        image1 = test_set.data[index1].to(torch.float32)/255
        image2 = test_set.data[index2].to(torch.float32)/255
        model.eval()
        bottleneck1 = model.encode(image1.view(1, -1)).to(device=device)
        bottleneck2 = model.encode(image2.view(1, -1)).to(device=device)
                
        interpolated_imgs = []
        alphas = torch.linspace(0, 1, n_steps, device=device)
        
        for a in alphas:
            print(a)
            bottleneck = a * bottleneck1 + (1-a) * bottleneck2
            reconstructed_image = model.decode(bottleneck).view(28, 28)
            interpolated_imgs.append(reconstructed_image)

        f = plt.figure(figsize=(12, 4))
        for i, img in enumerate([image2.view(28, 28)] + interpolated_imgs + [image1.view(28, 28)]):
            f.add_subplot(1, n_steps + 2, i + 1)
            plt.imshow(img.detach().numpy(), cmap='gray')
        plt.show()





###################################################################

if __name__ == '__main__':
    main()