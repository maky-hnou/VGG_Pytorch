import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, config, in_channels=3, num_classes=1000):
        super(VGG, self).__init__()
        self.config = config
        self.in_channels = in_channels
        self.num_classes = num_classes
        # select convolutional layer configuration for the VGG net
        self.convolutional_layers = self.make_conv_layers()
        self.fully_connected = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, self.num_classes)
        )

    def make_conv_layers(self, ):
        layers = []
        in_channels = self.in_channels
        for op in self.config:
            if op == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                layers += [
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=op, kernel_size=3,
                              padding=1),
                    nn.BatchNorm2d(op),
                    nn.ReLU()
                ]
                in_channels = op
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convolutional_layers(x)
        # flatten to prepare for the fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
