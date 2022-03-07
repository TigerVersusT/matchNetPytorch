"""
pytorch implementation of matchNet
"""
from turtle import forward
import torch
from torch import nn
from mobileNetV3 import Block, SeModule, hswish

class FeatureNet(nn.Module):
    def __init__(self) -> None:
        super(FeatureNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=7, padding=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),   
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, stride=1),  
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
    
    def forward(self, x):
        return self.features(x)

class mobileNetV3BlockNet(nn.Module):
    def __init__(self):
        super(mobileNetV3BlockNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        self.features = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 96, 64, nn.ReLU(inplace=True), None, 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.features(out)

        return out

class ClassiFilerNet(nn.Module):
    def __init__(self, extractor="mobileNet"):
        super(ClassiFilerNet, self).__init__()

        if extractor == "mobileNet":
            self.input_1 = mobileNetV3BlockNet()
            self.input_2 = mobileNetV3BlockNet()
        else:
            self.input_1 = FeatureNet()
            self.input_2 = FeatureNet()

        self.matric_network = nn.Sequential(
            nn.Linear(in_features=6272, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2),
            nn.Softmax()
        )
    
    def forward(self, x):
        """
        x has shape 2, N, H, W, 1, where 2 means two patches
        """
        feature_1 = self.input_1(x[0]).reshape((x[0].shape[0], -1))
        feature_2 = self.input_2(x[1]).reshape((x[1].shape[0], -1))

        # test
        #print("features.shape:{}".format(feature_1.cpu().shape))

        features = torch.cat((feature_1, feature_2), 1)

        # test
        #print("features.shape:{}".format(features.cpu().shape))

        return self.matric_network(features)

