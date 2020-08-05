import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import math
import time
import numpy as np
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
import random


class StickNet(nn.Module):

    def __init__(self, n_classes=10):
        
        super(StickNet, self).__init__()

        n_intermediates = 32

        self.features = nn.Sequential(
            nn.Conv2d(3, n_intermediates, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(7*7*n_intermediates, n_intermediates*2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(n_intermediates*2, n_classes)
        )
        self._initialize_weights()

    def forward(self, input_tensor):

        x = self.features(input_tensor)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

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
