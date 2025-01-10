import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Define the neural network structure
class NeuralNets(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)  # Flatten for fully connected layers

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# Class names corresponding to the output indices
classname = ['plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


new_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),       
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_model(model_path='trained_net.pth'):
    net = NeuralNets()
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    net.eval()
    return net

def predict_image(image, net):
    image = new_transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = net(image)
        _, predicted = torch.max(output, 1)
        predicted_class = classname[predicted.item()]
    
    return f'Predicted Class: {predicted_class}'
