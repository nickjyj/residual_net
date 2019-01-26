import torch.nn as nn
from natsort import natsorted
import torch.nn.functional as F
import torch

class Residual_Block(nn.Module):
    def __init__(self, inc, outc, s=1):
        super(Residual_Block, self).__init__()
        self.para = 0.01
        self.model = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, stride=s, padding=1),
            nn.BatchNorm2d(outc),
            nn.LeakyReLU(self.para),
            nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outc),
            nn.LeakyReLU(self.para)
            )
        self.downsample = nn.Conv2d(inc, outc, kernel_size=1, stride=s, padding=0)
        self.relu = nn.LeakyReLU(self.para)

    def forward(self, x):
        t = self.model(x)
        if t.size(1) != x.size(1):
            x = self.downsample(x)
        out = self.relu(x + t)
        return out


class Residual(nn.Module):
    def __init__(self, inc, num_classes, dt):
        """
        inc: the num_channels of images
        dt: dictionary for the intermediate layers. details: {num_channels : num_iterations}
        """
        super(Residual, self).__init__()
        self.keys = natsorted(dt.keys())
        self.fc = nn.Linear(self.keys[-1], num_classes)

        layers = []
        layers += [nn.Conv2d(inc, self.keys[0], kernel_size=3, stride=1, padding=1),
                   nn.BatchNorm2d(self.keys[0]),
                   nn.LeakyReLU()]
        for i in range(len(dt)):
            k = self.keys[i]
            for _ in range(dt[k]):
                layers += [Residual_Block(k, k)]
            if i+1<len(dt):
                k_next = self.keys[i+1]
                layers += [Residual_Block(k, k_next, s=2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
