from torch import nn
import torch.nn.functional as F

class PositionProbe(nn.Module):
    '''
    Decode 2D position from the activations of a predictive coding model.
    '''
    def __init__(self, input_channels=256, output_dim=2):
        super(PositionProbe, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 512, 3, padding=1)
        self.conv2 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(1024)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(1024 * 4 * 4, 2048)
        self.fc2 = nn.Linear(2048, output_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x