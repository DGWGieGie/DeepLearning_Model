import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.ToTensor()
            ])

    elif mode == 'test':
        return transforms.Compose([
            transforms.ToTensor(),
            ])

############################################################################
######                        Define the LeNet                        ######
############################################################################

class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 10)
            )
        
    def forward(self, input):
        return self.net(input)


network = Network()