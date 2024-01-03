import torch
import torch.nn as nn
import torchvision.models as models 
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier = nn.Sequential(*list(self.vgg16.classifier.children())[:-1])

    def forward(self, x):
        x = self.vgg16(x)
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # out =out[:,-1,:]
        out = self.fc(out[:, -1, :])
        return out


class VGG16_LSTM(nn.Module):
    def __init__(self, input_size=(3, 224, 224), hidden_size=256, num_layers=2, num_classes=5):
        super(VGG16_LSTM, self).__init__()
        self.vgg16 = VGG16()
        self.lstm = LSTM(input_size=4096, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)

    def forward(self, x):
        batch_size, timesteps, channels, height, width = x.size()
        x = x.view(batch_size * timesteps, channels, height, width)
        x = self.vgg16(x)
        x = x.view(batch_size, timesteps, -1)
        x = self.lstm(x)
        return x