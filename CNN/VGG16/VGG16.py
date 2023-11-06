import torch
import torch.nn as nn
import torchvision.models as models
class oneVGG16_0(nn.Module):
    def __init__(self):
        super(oneVGG16_0, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier = nn.Sequential(*list(self.vgg16.classifier.children())[:-1])
        self.fc=nn.Linear(61440,5)
    def forward(self, x):
        batch_size, timesteps, channels, height, width = x.size()
        x = x.view(batch_size * timesteps, channels, height, width)#batchsize *15 *3 *224 *224
        x = self.vgg16(x)
        x = x.view(-1,61440)
        x=self.fc(x)
        return x