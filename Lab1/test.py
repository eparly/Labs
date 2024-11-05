#########################################################################################################
#
#   ELEC 475 - Lab 1, Step 1
#   Fall 2024
#

## Note to reader: This file has been updated and moved to lab1.py
# This file was used for development purposes

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

    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    N_input = 28 * 28   # MNIST image size
    N_output = N_input
    model = autoencoderMLP4Layer(N_input=N_input, N_bottleneck=bottleneck_size, N_output=N_output)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    idx = 0
    while idx >= 0:
        idx = input("Enter index > ")
        idx = int(idx)
        if 0 <= idx <= train_set.data.size()[0]:
            print('label = ', train_set.targets[idx].item())
            img = train_set.data[idx]
            print('break 9', img.shape, img.dtype, torch.min(img), torch.max(img))

            img = img.type(torch.float32)
            print('break 10', img.shape, img.dtype, torch.min(img), torch.max(img))
            img = (img - torch.min(img)) / torch.max(img)
            print('break 11', img.shape, img.dtype, torch.min(img), torch.max(img))

            # plt.imshow(img, cmap='gray')
            # plt.show()

            img = img.to(device=device)
            # print('break 7: ', torch.max(img), torch.min(img), torch.mean(img))
            print('break 8 : ', img.shape, img.dtype)
            img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)
            print('break 9 : ', img.shape, img.dtype)
            
            #Add gaussian noise to image for denoising
            noise_factor = 1 # change this to see the effect of noise
            noisy_img = img + torch.randn(img.size()) * noise_factor
            
            with torch.no_grad():
                output = model(img)
                noisy_output = model(noisy_img)
            # output = output.view(28, 28).type(torch.ByteTensor)
            # output = output.view(28, 28).type(torch.FloatTensor)
            output = output.view(28, 28).type(torch.FloatTensor)
            noisy_output = noisy_output.view(28, 28).type(torch.FloatTensor)
            print('break 10 : ', output.shape, output.dtype)
            print('break 11: ', torch.max(output), torch.min(output), torch.mean(output))
            # plt.imshow(output, cmap='gray')
            # plt.show()

            # both = np.hstack((img.view(28, 28).type(torch.FloatTensor),output))
            # plt.imshow(both, cmap='gray')
            # plt.show()

            img = img.view(28, 28).type(torch.FloatTensor)
            noisy_img = noisy_img.view(28, 28).type(torch.FloatTensor)

            f = plt.figure()
            f.add_subplot(1,3,1)
            plt.imshow(img, cmap='gray')
            f.add_subplot(1,3,2)
            # plt.imshow(noisy_img, cmap='gray')
            # f.add_subplot(1,3,3)
            # plt.imshow(noisy_output, cmap='gray')
            plt.imshow(output, cmap='gray')
            plt.show()
        ## step 6
    
    while True:
        index1 = int(input('enter index for image 1'))
        index2 = int(input('enter index for image 2'))
        n_steps = int(input('enter number of steps'))
        
        if(index1 < 0 or index1 >= len(test_set.data) or index2 < 0 or index2 >= len(test_set.data)):
            return
        
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
        # for i in range(n_steps):
        #     a = i/(n_steps - 1)
        #     bottleneck = a * bottleneck1 + (1-a) * bottleneck2
        #     interpolated.append(bottleneck)
            
        # interpolated_imgs = [model.decode(b).view(28,28) for b in interpolated]
        f = plt.figure(figsize=(12, 4))
        for i, img in enumerate([image2.view(28, 28)] + interpolated_imgs + [image1.view(28, 28)]):
            f.add_subplot(1, n_steps + 2, i + 1)
            plt.imshow(img.detach().numpy(), cmap='gray')
            # plt.axis('off')
        plt.show()





###################################################################

if __name__ == '__main__':
    main()