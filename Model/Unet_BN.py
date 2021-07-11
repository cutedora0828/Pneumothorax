#%%
import torch
import torch.nn as nn
 
#%%

def print():
    print("I am dora")


class Unet_bn(nn.Module):
    def __init__(self):
        super(Unet_bn,self).__init__()
############1a
        self.branch1x1_1a = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=1,padding=0), # 3x128x128 -> 64x128x128
            nn.BatchNorm2d(64)
        )
        self.branch3x3_1a = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=3, padding=1), # 3x128x128 ->64x128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3,padding=1),  # 64x128x128 ->64x128x128
            nn.BatchNorm2d(64)
        ) 
        #self.TensorProject1 = TensorProject((64,256,256), (64,128,128))
        self.Max_1 = nn.MaxPool2d(kernel_size=2) # 64x128x128 -> 64x64x64
############2a
        self.branch1x1_2a = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=1,padding=0), # 64x64x64 -> 128x64x64
            nn.BatchNorm2d(128)
        )

        self.branch3x3_2a = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=3, padding=1), # 64x64x64 -> 128x64x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3,padding=1),  # 128x64x64 -> 128x64x64
            nn.BatchNorm2d(128)
        ) 
        #self.TensorProject2 = TensorProject((128,128,128), (128,64,64))
        self.Max_2 = nn.MaxPool2d(kernel_size=2) # 128x64x64 -> 128x32x32
############3a
        self.branch1x1_3a = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=1,padding=0), # 128x32x32 -> 256x32x32
            nn.BatchNorm2d(256)
        )
        self.branch3x3_3a = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=3, padding=1), # 128x32x32 -> 256x32x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3,padding=1),  # 256x32x32 -> 256x32x32
            nn.BatchNorm2d(256)
        ) 
        #self.TensorProject3 = TensorProject((256,64,64), (256,32,32))
        self.Max_3 = nn.MaxPool2d(kernel_size=2) # 256x32x32 -> 256x16x16

############4a
        self.branch1x1_4a = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=1,padding=0), # 256x16x16 -> 512x16x16
            nn.BatchNorm2d(512)
        )
        self.branch3x3_4a = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=3, padding=1), # 256x16x16 -> 512x16x16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=3,padding=1),  # 512x16x16 -> 512x16x16
            nn.BatchNorm2d(512)
        )
        #self.TensorProject4 = TensorProject((512,32,32), (512,16,16))
        self.Max_4 = nn.MaxPool2d(kernel_size=2) # 512x16x16 -> 512x8x8


############5a
        self.branch1x1_5a = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size=1,padding=0), # 512x8x8 -> 1024x8x8
            nn.BatchNorm2d(1024)
        )
        self.branch3x3_5a = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size=3, padding=1), # 512x8x8 -> 1024x8x8
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=3,padding=1),  # 1024x8x8 -> 1024x8x8
            nn.BatchNorm2d(1024)
        )
############1b
        self.up_1_1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512 ,kernel_size=2,stride=2,padding=0) # 1024x8x8 -> 512x16x16
        self.up1_residual = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size=2,stride=2,padding=0) # 1024x8x8 -> 512x16x16
        
        self.up_1x1_1b = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size=1,padding=0), # 1024x16x16 -> 512x16x16
            nn.BatchNorm2d(512)
        )
        self.up_3x3_1b = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size=3, padding=1),# 1024x16x16 -> 512x16x16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=3,padding=1),  # 512x16x16 -> 512x16x16
            nn.BatchNorm2d(512)
        )

############2b
        self.up_2_1 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size=2,stride=2,padding=0) # 512x32x32 -> 256x16x16
        self.up2_residual = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size=2,stride=2,padding=0) # 512x32x32 -> 256x16x16

        self.up_1x1_2b = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size=1,padding=0), # 512x32x32 -> 256x32x32
            nn.BatchNorm2d(256)
        )
        self.up_3x3_2b = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size=3, padding=1),# 512x32x32 -> 256x32x32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=3,padding=1),  # 256x32x32 -> 256x32x32
            nn.BatchNorm2d(256)
        )

############3b
        self.up_3_1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size=2,stride=2,padding=0) # 256x32x32 -> 128x64x64
        self.up3_residual = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size=2,stride=2,padding=0) # 256x32x32 -> 128x64x64
        
        self.up_1x1_3b = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size=1,padding=0), # 256x64x64 -> 128x64x64
            nn.BatchNorm2d(128)
        )
        self.up_3x3_3b = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size=3, padding=1),# 256x64x64 -> 128x64x64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3,padding=1),  # 128x64x64 -> 128x64x64
            nn.BatchNorm2d(128)
        )
############4b
        self.up_4_1 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size=2,stride=2,padding=0) # 128x128x128-> 64x128x128
        self.up4_residual = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size=2,stride=2,padding=0) # 128x128x128-> 64x128x128
        
        self.up_1x1_4b = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size=1,padding=0), # 128x128x128 -> 64x128x128
            nn.BatchNorm2d(64)
        )
        self.up_3x3_4b = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size=3, padding=1),# 128x128x128 -> 64x128x128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3,padding=1),  # 64x128x128 -> 64x128x128
            nn.BatchNorm2d(64)
        )
############5output
        self.output = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size=1, padding=0) # 64x128x128 -> 1x128x128
        
################################################################################################        
    def forward(self, x):
        branch1_1 = self.branch1x1_1a(x) # 3x128x128 -> 64x128x128
        #print(1, branch1_1.shape)
        branch1_2 = self.branch3x3_1a(x) # 3x128x128 -> 64x128x128
        #print(2,branch1_2.shape)
        branch1 = torch.add(branch1_1,branch1_2) # 64x128x128
        #print(3,branch1.shape)
        relu_1 = torch.relu(branch1) # 64x128x128
        #print(4,relu_1.shape)
        #TensorProject1 = self.TensorProject1(relu_1)
        #print(TensorProject1.shape)
        Max_1 = self.Max_1(relu_1) # 64x128x128 -> 64x64x64
        #print(5,Max_1.shape)

############2a
        branch2_1 = self.branch1x1_2a(Max_1) # 64x64x64 -> 128x64x64
        #print(6,branch2_1.shape)
        branch2_2 = self.branch3x3_2a(Max_1) # 64x64x64 -> 128x64x64
        #print(7,branch2_2.shape)
        branch2 = torch.add(branch2_1,branch2_2) # 128x64x64
        #print(8,branch2.shape)
        relu_2 = torch.relu(branch2) # 128x64x64
        #print(9,relu_2.shape)
        #TensorProject2 = self.TensorProject2(relu_2)
        #print(TensorProject2.shape)
        Max_2 = self.Max_2(relu_2) # 128x32x32
        #print(10,Max_2.shape)

############3a
        branch3_1 = self.branch1x1_3a(Max_2) # 128x32x32 -> 256x32x32
        #print(11,branch3_1.shape)
        branch3_2 = self.branch3x3_3a(Max_2) # 128x32x32 -> 256x32x32
        #print(12,branch3_2.shape)
        branch3 = torch.add(branch3_1,branch3_2) # 256x32x32
        #print(13,branch3.shape)
        relu_3 = torch.relu(branch3) # 256x32x32
        #print(14,relu_3.shape)
        #TensorProject3 = self.TensorProject3(relu_3)
        #print(TensorProject3.shape)
        Max_3 = self.Max_3(relu_3) # 256x16x16
        #print(15,Max_3.shape)

############4a
        branch4_1 = self.branch1x1_4a(Max_3) # 256x16x16 -> 512x16x16
        #print(16,branch4_1.shape)
        branch4_2 = self.branch3x3_4a(Max_3) # 256x16x16 -> 512x16x16
        #print(17,branch4_2.shape)
        branch4 = torch.add(branch4_1,branch4_2) # 512x16x16
        #print(18,branch4.shape)
        relu_4 = torch.relu(branch4)# 512x16x16
        #print(19,relu_4.shape)
        #TensorProject4 = self.TensorProject4(relu_4)
        #print(TensorProject4.shape)

        Max_4 = self.Max_4(relu_4)# 512x8x8
        #print(20,Max_4.shape)

############5a
        branch5_1 = self.branch1x1_5a(Max_4) # 512x8x8 -> 1024x8x8
        #print(21,branch5_1.shape)
        branch5_2 = self.branch3x3_5a(Max_4) # 512x8x8 -> 1024x8x8
        #print(22,branch5_2.shape)
        branch5 = torch.add(branch5_1,branch5_2) # 1024x8x8
        #print(23,branch5.shape)
        relu_5 = torch.relu(branch5) # 1024x8x8
        #print(24,relu_5.shape)

############1b
        up_1_1_1b = self.up_1_1(relu_5) #1024x8x8 -> 512x16x16
        #print(26,up_1_1_1b.shape)
        cat_1b = torch.cat((branch4,up_1_1_1b),1) #1024x16x16
        #print(27,cat_1b.shape)
        up_branch_1_1 = self.up_1x1_1b(cat_1b) #1024x16x16 -> 512x16x16
        #print(28,up_branch_1_1.shape)
        up_branch_1_2 = self.up_3x3_1b(cat_1b) #1024x16x16 -> 512x16x16
        #print(29,up_branch_1_2.shape)
        up_branch_1 = torch.add(up_branch_1_1,up_branch_1_2) #512x16x16
        #print(30,up_branch_1.shape)
        up1_residual = self.up1_residual(relu_5) #1024x8x8 -> 512x16x16
        #print(31,up1_residual.shape)
        up1_add = torch.add(up_branch_1,up1_residual) #512x16x16
        #print(32,up1_add.shape)
        relu_up_1 = torch.relu(up1_add) #512x16x16
        #print(33,relu_up_1.shape)

############2b
        up_1_1_2b = self.up_2_1(relu_up_1) #512x16x16 -> 256x32x32
        #print(34,up_1_1_2b.shape)
        cat_2b = torch.cat((branch3,up_1_1_2b),1) #512x32x32
        #print(35,cat_2b.shape)
        up_branch_2_1 = self.up_1x1_2b(cat_2b) #512x32x32 -> 256x32x32
        #print(36,up_branch_2_1.shape)
        up_branch_2_2 = self.up_3x3_2b(cat_2b) #512x32x32 -> 256x32x32
        #print(37,up_branch_2_2.shape)
        up_branch_2 = torch.add(up_branch_2_1,up_branch_2_2) #256x32x32
        #print(38,up_branch_2.shape)
        up2_residual = self.up2_residual(relu_up_1) #512x16x16 -> 256x32x32
        #print(39,up2_residual.shape)
        up2_add = torch.add(up_branch_2,up2_residual) #256x32x32
        #print(40,up2_add.shape)
        relu_up_2 = torch.relu(up2_add) #256x32x32
        #print(41,relu_up_2.shape)

############3b
        up_1_1_3b = self.up_3_1(relu_up_2) #256x32x32 -> 128x64x64 
        #print(42,up_1_1_3b.shape)
        cat_3b = torch.cat((branch2,up_1_1_3b),1) #128x64x64 -> 256x64x64
        #print(43,cat_3b.shape)
        up_branch_3_1 = self.up_1x1_3b(cat_3b) #256x64x64 -> 128x64x64
        #print(44,up_branch_3_1.shape)
        up_branch_3_2 = self.up_3x3_3b(cat_3b) #256x64x64 -> 128x64x64
        #print(45,up_branch_3_2.shape)
        up_branch_3 = torch.add(up_branch_3_1,up_branch_3_2) #128x64x64
        #print(46,up_branch_3.shape)
        up3_residual = self.up3_residual(relu_up_2) #256x32x32 -> 128x64x64 
        #print(47,up3_residual.shape)
        up3_add = torch.add(up_branch_3,up3_residual) #128x64x64
        #print(48,up3_add.shape)
        relu_up_3 = torch.relu(up3_add) #128x64x64
        #print(49,relu_up_3.shape)

############4b
        up_1_1_4b = self.up_4_1(relu_up_3) #128x64x64 -> 64x128x128
        #print(50,up_1_1_4b.shape)
        cat_4b = torch.cat((branch1,up_1_1_4b),1) #64x128x128 -> 128x128x128
        #print(51,cat_4b.shape)
        up_branch_4_1 = self.up_1x1_4b(cat_4b) #128x128x128 -> 64x128x128
        #print(52,up_branch_4_1.shape)
        up_branch_4_2 = self.up_3x3_4b(cat_4b) #128x128x128 -> 64x128x128
        #print(53,up_branch_4_2.shape)
        up_branch_4 = torch.add(up_branch_4_1,up_branch_4_2) #64x128x128
        #print(54,up_branch_4.shape)
        up4_residual = self.up4_residual(relu_up_3) #128x64x64 -> 64x128x128
        #print(55,up4_residual.shape)
        up4_add = torch.add(up_branch_4,up4_residual) #64x128x128
        #print(56,up4_add.shape)
        relu_up_4 = torch.relu(up4_add) #64x128x128
        #print(57,relu_up_4.shape)

############output
        output = self.output(relu_up_4) #64x128x128 -> 1x128x128
        #print(58,output.shape)
        #output = torch.sigmoid(output) #1x128x128
        #print(59,output.shape)

        return output


