#%%
import torch
import torch.nn as nn

#%%
class Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
        
#%%
class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=True, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv00 = Block(input_channels, nb_filter[0], nb_filter[0])
        self.conv10 = Block(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv20 = Block(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv30 = Block(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv40 = Block(nb_filter[3], nb_filter[4], nb_filter[4])


        self.conv01 = Block(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv11 = Block(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv21 = Block(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv31 = Block(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv02 = Block(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv12 = Block(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv22 = Block(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])


        self.conv03 = Block(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv13 = Block(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv04 = Block(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])


        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x00 = self.conv00(input) #3x1024x1024 -> 32x1024x1024
        x10 = self.conv10(self.pool(x00)) #32x512x512 -> 64x512x512
        x01 = self.conv01(torch.cat([x00, self.up(x10)], 1)) #64x1024x1024
        x20 = self.conv20(self.pool(x10)) #64x512x512 -> 64x256x256 -> 128x256x256
        x11 = self.conv11(torch.cat([x10, self.up(x20)], 1)) #128x512x512
        x02 = self.conv02(torch.cat([x00, x01, self.up(x11)], 1)) #128x1024x1024
        x30 = self.conv30(self.pool(x20)) #128x128x128 -> 256x128x128 -> 256x64x64
        x21 = self.conv21(torch.cat([x20, self.up(x30)], 1)) #256x256x256
        x12 = self.conv12(torch.cat([x10, x11, self.up(x21)], 1)) #256x512x512
        x03 = self.conv03(torch.cat([x00, x01, x02, self.up(x12)], 1)) #256x1024x1024

        x40 = self.conv40(self.pool(x30)) #256x64x64 -> 512x64x64
        x31 = self.conv31(torch.cat([x30, self.up(x40)], 1)) #512x128x128
        x22 = self.conv22(torch.cat([x20, x21, self.up(x31)], 1)) #512x256x256
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up(x22)], 1)) #512x512x512
        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up(x13)], 1)) #512x1024x1024

        if self.deep_supervision:
            output1 = self.final1(x01) #64x1024x1024 -> 1x1024x1024
            #print(1,output1.shape)
            output2 = self.final2(x02) #128x1024x1024 -> 1x1024x1024
            #print(2,output2.shape)
            output3 = self.final3(x03) #256x1024x1024 -> 1x1024x1024
            #print(3,output3.shape)
            output4 = self.final4(x04) #512x1024x1024 -> 1x1024x1024
            #print(4,output4.shape)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x04) #512x1024x1024 -> 1x1024x1024
            return output
