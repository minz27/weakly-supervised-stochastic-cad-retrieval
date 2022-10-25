import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            #128x128
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            #64x64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            #32x32
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            #16x16
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            #8x8
        )

        self.embedding = nn.Sequential(        
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512)
        )

        self._initialize_weights()    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)    

    def forward(self,x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x             