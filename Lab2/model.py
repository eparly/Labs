import torch.nn as nn
import torch.nn.functional as F
import torch

#Step 2.1 - Model
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.input_shape = (227, 227, 3)
        self.type = 'CustomCNN'
    
    def forward(self, X):
        
        X = self.pool(F.relu(self.conv1(X)))  # Output: 64x57x57
        X = self.pool(F.relu(self.conv2(X)))  # Output: 128x15x15
        X = self.pool(F.relu(self.conv3(X)))  # Output: 256x4x4
        # flatten the tensor
        X = X.view(-1, 4096)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X


# used to verify the shape of each layer
if __name__ == '__main__':
    model = CustomModel()
    model.eval()
    dummy_input = torch.randn(1, 3, 227, 227)
    output = model(dummy_input)
    print(output)
    print(output.shape)