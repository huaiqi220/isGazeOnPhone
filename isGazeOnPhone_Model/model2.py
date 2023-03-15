"""resnet18的pytorch实现"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class basic_block(nn.Module):
    """基本残差块,由两层卷积构成"""
    def __init__(self,in_planes,planes,kernel_size=3,stride=1):
        """

        :param in_planes: 输入通道
        :param planes:  输出通道
        :param kernel_size: 卷积核大小
        :param stride: 卷积步长
        """
        super(basic_block, self).__init__()
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=kernel_size,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(planes,planes,kernel_size=kernel_size,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        if stride!=1 or in_planes!=planes:
            self.downsample=nn.Sequential(nn.Conv2d(in_planes,planes,kernel_size=1,stride=stride)
                                          ,nn.BatchNorm2d(planes))
        else:
            self.downsample=nn.Sequential()
    def forward(self,inx):
        x=self.relu(self.bn1(self.conv1(inx)))
        x=self.bn2(self.conv2(x))
        out=x+self.downsample(inx)
        return F.relu(out)


class Resnet(nn.Module):
    def __init__(self,basicBlock,blockNums,nb_classes):
        super(Resnet, self).__init__()
        self.in_planes=64
        #输入层
        self.conv1=nn.Conv2d(3,self.in_planes,kernel_size=(7,7),stride=(2,2),padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.in_planes)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self._make_layers(basicBlock,blockNums[0],64,1)
        self.layer2=self._make_layers(basicBlock,blockNums[1],128,2)
        self.layer3=self._make_layers(basicBlock,blockNums[2],256,2)
        self.layer4=self._make_layers(basicBlock,blockNums[3],512,2)
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc=nn.Linear(512,nb_classes)
        self.sm=nn.Softmax(dim=1)

    def _make_layers(self,basicBlock,blockNum,plane,stride):
        """

        :param basicBlock: 基本残差块类
        :param blockNum: 当前层包含基本残差块的数目,resnet18每层均为2
        :param plane: 输出通道数
        :param stride: 卷积步长
        :return:
        """
        layers=[]
        for i in range(blockNum):
            if i==0:
                layer=basicBlock(self.in_planes,plane,3,stride=stride)
            else:
                layer=basicBlock(plane,plane,3,stride=1)
            layers.append(layer)
        self.in_planes=plane
        return nn.Sequential(*layers)
    def forward(self,inx):
        x=self.maxpool(self.relu(self.bn1(self.conv1(inx))))
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avgpool(x)
        x=x.view(x.shape[0],-1)
        out=self.fc(x)
        return out
