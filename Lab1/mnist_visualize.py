from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

## Part 2 of the lab
## =========================================================

train_transform = transforms.Compose([transforms.ToTensor()])

train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

idx = int(input('Enter value'))

plt.imshow(train_set.data[idx], cmap='gray')
plt.show()
