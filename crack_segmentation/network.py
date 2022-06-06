import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary
from sys import exit

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=not use_norm,
            **kwargs
        )
        self.norm = None
        if use_norm:
            self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = torch.relu(x)
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, **kwargs) -> None:
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            bias=not use_norm,
            **kwargs
        )
        self.norm = None
        if use_norm:
            self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv_trans(x)
        if self.norm:
            x = self.norm(x)
        x = torch.relu(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv_trans = ConvTransBlock(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.conv = ConvBlock(
            out_channels*2,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
    
    def forward(self, x, skip_connect):
        x = self.conv_trans(x)
        x = torch.cat([x, skip_connect], dim=1)
        x = self.conv(x)
        return x

class ResUnet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        for param in backbone.parameters():
            param.require_grad = False

        self.down_layers = self._get_down_layers(backbone)
        self.neck = nn.Sequential(
            nn.Conv2d(512,  512, 1)
        )
        self.up_layers = self._get_up_layers()
        self.conv_in = ConvBlock(3, 64, kernel_size=1, stride=1)
        self.conv_out = nn.Conv2d(64, n_classes, 1)
        
    
    def forward(self, x):
        skip_connections = [self.conv_in(x)]
        for layer in self.down_layers:
            x = layer(x)
            skip_connections.append(x)
        ###################
        x = self.neck(x)
        ############
        skip_connections = reversed(skip_connections[0:-1])
        for up_layer, skip in zip(self.up_layers, skip_connections):
            x = up_layer(x, skip)
        # out
        x = self.conv_out(x)
        return x

    def _get_down_layers(self, backbone):
        backbone = models.resnet18(pretrained=True)
        backbone_layers = list(backbone.children())
        down_layers = nn.ModuleList()
        down_layers += [
            nn.Sequential(*backbone_layers[0:3]),
            nn.Sequential(*backbone_layers[3:5])
        ]
        down_layers += [nn.Sequential(backbone_layers[i]) \
            for i in range(5, 8)]
        
        return down_layers
    
    def _get_up_layers(self):
        channels = [64, 64, 64, 128, 256, 512]
        up_layers = nn.ModuleList()
        in_channels = channels[-1]
        channels = reversed(channels[:-1])
        for feature in channels:
            up_layers += [
                UpBlock(in_channels, feature)
            ]
            in_channels = feature
        return up_layers
            
