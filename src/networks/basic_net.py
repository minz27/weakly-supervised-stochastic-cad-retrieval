import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet34

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet34(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        # self.resnet = resnet50(pretrained = True)
        self.pretrained = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.encoder = nn.Sequential(
            #128x128 240x240
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            #64x64 120x120
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            #32x32 60x60
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            #16x16 30x30
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            #8x8 15x15
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            #3x3 7x7
        )

        self.embedding = nn.Sequential(        
            # nn.Linear(in_features=8192, out_features=1024),
            # nn.ReLU(),
            # nn.Linear(in_features=1024, out_features=512)
            # Resnet50
            # nn.Linear(in_features=2048, out_features=1024),
            # nn.ReLU(),
            # nn.Linear(in_features=1024, out_features=512),
            # nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=256),
            # nn.Tanh()
            # nn.ReLU(),
            # nn.Linear(in_features=256, out_features=128)
        )

        self._initialize_weights()    

    def _initialize_weights(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)

            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)    

    def forward(self,x):
        x = self.pretrained(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = torch.nn.functional.normalize(x)
        return x             