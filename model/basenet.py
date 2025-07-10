import torch
from torch import nn
from model.former import MHSAblock, MHSAblockv2, MHSAblock_pro
from model.SE import LiteSqueezeExcitation, ChannelAttention



class UPDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, up=True, last=True):
        super().__init__()
        self.step2 = nn.Sequential(
            nn.Conv2d(in_channels // 3, in_channels // 3, kernel_size=3, stride=1, padding=1, groups=in_channels // 3),
            # nn.BatchNorm2d(in_channels // 3),
            nn.Conv2d(in_channels // 3, in_channels // 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels // 3),
            nn.PReLU())
        self.step1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 3, kernel_size=3, stride=1, padding=1, groups=in_channels // 3),
            # nn.BatchNorm2d(in_channels // 3),
            nn.Conv2d(in_channels // 3, in_channels // 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels // 3),
            nn.PReLU()
            )
        groups = None
        if last == True:
            groups = in_channels // 3
        elif last == False:
            groups = out_channels
        else:
            print('the value of last is False')
        self.DWconv1 = nn.Conv2d(in_channels // 3, groups, kernel_size=3, padding=1, groups=groups)
        # self.bn0 = nn.BatchNorm2d(groups)
        self.PWconv1 = nn.Conv2d(groups, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu1 = nn.ReLU()
        self.residual = nn.Sequential(nn.Conv2d(in_channels // 3, out_channels, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(out_channels))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upchannel = nn.Sequential(nn.Conv2d(groups, in_channels // 3, kernel_size=1),
                                       nn.BatchNorm2d(in_channels // 3))
        self.tpconv = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels//3, out_channels=in_channels//3, kernel_size=4, stride=2,
                                    padding=1, output_padding=0),
                                    nn.BatchNorm2d(in_channels // 3))
        self.wrl = in_channels // 3

    def _channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups,
                   channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x2, x1):
        # B, C, H, W = x
        # print(x1.shape, x2.shape,self.wrl)
        x = torch.cat([self.upsample(x2), self.tpconv(x2), self.upchannel(x1)], dim=1)
        
        x = self._channel_shuffle(x, 3)
        x = self.step1(x)
        residual = x
        # print(x.shape)
        x = self.step2(x)
        x = self.DWconv1(x)
        # x = self.bn0(x)
        x = self.PWconv1(x)
        x = self.bn1(x)
        return x + self.residual(residual)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, with_bn=False, blocks=None, block1=DownDoubleConv,
                 block2=UPDoubleConv):
        super().__init__()
        init_channels = 64
        self.with_bn = with_bn
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=init_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(init_channels))

        # self.en_1 = DownDoubleConv(1*init_channels, 1*init_channels, with_bn)
        self.en_1 = self._make_layer(block1, 1*init_channels, 1*init_channels, blocks=blocks[0],use_att=False)
        # self.en_2 = DownDoubleConv(1*init_channels, 2*init_channels, with_bn)
        self.en_2 = self._make_layer(block1, 1*init_channels, 2*init_channels, blocks=blocks[1],use_att=True)
        # self.en_3 = DownDoubleConv(2*init_channels, 4*init_channels, with_bn)
        self.en_3 = self._make_layer(block1, 2*init_channels, 4*init_channels, blocks=blocks[2],use_att=True)
        # self.en_4 = DownDoubleConv(4*init_channels, 8*init_channels, with_bn)
        self.en_4 = self._make_layer(block1, 4*init_channels, 8*init_channels, blocks=blocks[3],use_att=True)

        self.de_1 = UPDoubleConv((8+8+8)*init_channels, 4*init_channels, last=False)
        self.de_2 = UPDoubleConv((4+4+4)*init_channels, 2*init_channels, last=False)
        self.de_3 = UPDoubleConv((2+2+2)*init_channels, 1*init_channels, last=False)
        self.de_4 = UPDoubleConv((1+1+1)*init_channels, out_channels, last=True)

        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(in_channels=4*init_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(in_channels=2*init_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels))
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=1*init_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels))
        
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, use_att=False):
        layers = []
        layers.append(block(inplanes, planes, first=True, use_att=use_att, stride=2))
        for i in range(0, blocks):
            # print(i)
            layers.append(block(planes, planes, first=False, use_att=use_att, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        e1 = self.en_1(x)
        e2 = self.en_2(e1)
        e3 = self.en_3(e2)
        e4 = self.en_4(e3)

        # d1 = self.de_1(torch.cat([self.upsample(e4), e3], dim=1))
        d1 = self.de_1(e4, e3)
        # d2 = self.de_2(torch.cat([self.upsample(d1), e2], dim=1))
        d2 = self.de_2(d1, e2)
        # d3 = self.de_3(torch.cat([self.upsample(d2), e1], dim=1))
        d3 = self.de_3(d2, e1)
        # d4 = self.de_4(torch.cat([self.upsample(d3), x], dim=1))
        # print(d3.shape, x.shape)
        d4 = self.de_4(d3, x)

        return [self.up1(d1), self.up2(d2), self.up3(d3), d4]


#         if self.out_channels<2:
#             return torch.sigmoid(d4)
#         return torch.softmax(d4, 1)
def CNNlike36(in_channels=1, out_channels=9):
    blocks = [3, 4, 6, 3]
    # blocks = [1,1,2,2]
    return UNet(in_channels=in_channels, out_channels=out_channels, with_bn=True, blocks=blocks)


def CNNlike50(in_channels=1, out_channels=9):
    blocks = [2, 3, 12, 3]
    return UNet(in_channels=in_channels, out_channels=out_channels, with_bn=True, blocks=blocks)
