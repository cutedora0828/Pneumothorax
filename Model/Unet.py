#%%
import torch
import torch.nn as nn

#%%
class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1=nn.Conv2d(3,64,kernel_size=3, stride=1, padding=1)
        self.conv1_2=nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1)
        self.max1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.conv2_1=nn.Conv2d(64,128,kernel_size=3, stride=1, padding=1)
        self.conv2_2=nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1)
        self.max2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.conv3_1=nn.Conv2d(128,256,kernel_size=3, stride=1, padding=1)
        self.conv3_2=nn.Conv2d(256,256,kernel_size=3, stride=1, padding=1)
        self.max3 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.conv4_1=nn.Conv2d(256,512,kernel_size=3, stride=1, padding=1)
        self.conv4_2=nn.Conv2d(512,512,kernel_size=3, stride=1, padding=1)
        self.max4 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.conv5_1=nn.Conv2d(512,1024,kernel_size=3, stride=1, padding=1)
        self.conv5_2=nn.Conv2d(1024,1024,kernel_size=3, stride=1, padding=1)

        self.upconv1=nn.ConvTranspose2d(1024,512,2,stride=2)

        self.conv6_1=nn.Conv2d(1024,512,3, padding=(1,1))
        self.conv6_2=nn.Conv2d(512,512,3, padding=(1,1))
        self.upconv2=nn.ConvTranspose2d(512,256,2,stride=2)

        self.conv7_1=nn.Conv2d(512,256,3, padding=(1,1))
        self.conv7_2=nn.Conv2d(256,256,3, padding=(1,1))
        self.upconv3=nn.ConvTranspose2d(256,128,2,stride=2)

        self.conv8_1=nn.Conv2d(256,128,3, padding=(1,1))
        self.conv8_2=nn.Conv2d(128,128,3, padding=(1,1))
        self.upconv4=nn.ConvTranspose2d(128,64,2,stride=2)

        self.conv9_1=nn.Conv2d(128,64,3, padding=(1,1))
        self.conv9_2=nn.Conv2d(64,64,3, padding=(1,1))

        self.conv10=nn.Conv2d(64,1,kernel_size=1, stride=1, padding=0)


    def forward(self,X):
        conv1_1 = self.conv1_1(X) #3x256x256 -> 64x256x256
        conv1_2 = self.conv1_2(conv1_1) # 64x256x256 -> 64x256x256
        max1, self.indice1 = self.max1(conv1_2) #64x256x256 -> 64x128x128

        conv2_1 = self.conv2_1(max1) #64x128x128 -> 128x128x128
        conv2_2 = self.conv2_2(conv2_1) #128x128x128 -> 128x128x128        
        max2, self.indice2 = self.max2(conv2_2) #128x128x128 -> 128x64x64

        conv3_1 = self.conv3_1(max2) #128x64x64 -> 256x64x64
        conv3_2 = self.conv3_2(conv3_1) #256x64x64 -> 256x64x64        
        max3, self.indice3 = self.max3(conv3_2) #256x64x64 -> 256x32x32

        conv4_1 = self.conv4_1(max3) #256x32x32 -> 512x32x32
        conv4_2 = self.conv4_2(conv4_1) #512x32x32 ->512x32x32
        max4, self.indice4 = self.max4(conv4_2) #512x32x32 ->512x16x16

        conv5_1 = self.conv5_1(max4) #512x16x16 -> 1024x16x16
        conv5_2 = self.conv5_2(conv5_1) #1024x16x16 ->1024x16x16

        upconv1 = self.upconv1(conv5_2) #1024x16x16 -> 512x32x32
        
        concat1 = torch.cat((upconv1, conv4_2), 1) #1024x32x32
        conv6_1 = self.conv6_1(concat1) #1024x32x32 -> 512x32x32
        conv6_2 = self.conv6_2(conv6_1) #512x32x32 -> 512x32x32

        upconv2 = self.upconv2(conv6_2) #512x32x32 ->256x64x64
        
        concat2 = torch.cat((upconv2, conv3_2),1) #512x64x64
        conv7_1 = self.conv7_1(concat2) #512x64x64 -> 256x64x64
        conv7_2 = self.conv7_2(conv7_1) #256x64x64 -> 256x64x64

        upconv3 = self.upconv3(conv7_2) #256x64x64 -> 128x128x128

        concat3 = torch.cat((upconv3, conv2_2), 1) #256x128x128
        conv8_1 = self.conv8_1(concat3) #256x128x128 -> 128x128x128
        conv8_2 = self.conv8_2(conv8_1) #128x128x128 -> 128x128x128

        upconv4 = self.upconv4(conv8_2) #128x128x128 -> 64x256x256

        concat4 = torch.cat((upconv4, conv1_2), 1) #128x256x256
        conv9_1 = self.conv9_1(concat4) #128x256x256 -> 64x256x256
        conv9_2 = self.conv9_2(conv9_1) #64x256x256 -> 64x256x256

        output = self.conv10(conv9_2) #64x256x256 -> 1x256x256

        return output
