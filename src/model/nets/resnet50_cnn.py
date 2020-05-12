"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torchvision.models as models
from .resnet_utils import BasicBlock, BottleNeck

class ResNet50_CNN(nn.Module):

    def __init__(self, num_classes, block=BasicBlock, num_block=[3, 4, 6, 3], pretrained=False):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.conv6_x_output = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv7_x_output = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # self.avg_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.conv8_x_output = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=1, bias=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if pretrained:
            print('########## Loading pretrained model parameters... ##########')
            model_dict = self.state_dict()
            resnet50 = models.resnet50(pretrained=True)
            pretrained_dict = resnet50.state_dict()
            del pretrained_dict['fc.weight']
            del pretrained_dict['fc.bias']
            pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print('########## Finish loading pretrained weights ##########')

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _freeze(self):
        print('########## Freeze the parameters in the model except for output layers ##########')
        for name, child in self.named_children():
            if 'output' not in name:
                for param in child.parameters():
                    param.requires_grad = False
    
    def _unfreeze(self):
        print('########## Unfreeze the parameters in the model ##########')
        # for param in self.parameters():
        #     param.requires_grad = True

        for i, child in enumerate(self.children()):
            if i != 0:
                for param in child.parameters():
                    param.requires_grad = True

    def _show_requires_grad(self):
        for name, child in self.named_children():
            print(name, child)
            for param in child.parameters():
                print(param.requires_grad)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        # output = self.avg_pool(output)
        output = self.conv6_x_output(output)
        output = self.conv7_x_output(output)
        output = self.conv8_x_output(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)

        return output 