
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
from CGNet.ChangeGuideModule import *
# from ChangeFormerV4_MyEncoder import *
from DDLNet import *



class DDLNet_MyModel(nn.Module):
    def __init__(self, device=None):
        super(DDLNet_MyModel, self).__init__()
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]  # 64
        self.down1 = vgg16_bn.features[5:12]  # 128
        self.down2 = vgg16_bn.features[12:22]  # 256
        self.down3 = vgg16_bn.features[22:32]  # 512
        self.down4 = vgg16_bn.features[32:42]  # 512

        self.DDLNet   = DDLNet(1)

        # self.conv_reduce_1 = BasicConv2d(128*2,128,3,1,1, device=device)
        # self.conv_reduce_2 = BasicConv2d(256*2,256,3,1,1, device=device)
        # self.conv_reduce_3 = BasicConv2d(512*2,512,3,1,1, device=device)
        # self.conv_reduce_4 = BasicConv2d(512*2,512,3,1,1, device=device)

        # self.up_layer4 = BasicConv2d(512,512,3,1,1, device=device)
        # self.up_layer3 = BasicConv2d(512,512,3,1,1, device=device)
        # self.up_layer2 = BasicConv2d(256,256,3,1,1, device=device)

        # self.decoder = nn.Sequential(BasicConv2d(512,64,3,1,1, device=device),nn.Conv2d(64,1,3,1,1, device=device))

        # self.decoder_final = nn.Sequential(BasicConv2d(128,64,3,1,1, device=device),nn.Conv2d(64,1,1, device=device))

        # self.cgm_2 = ChangeGuideModule(256, device=device)
        # self.cgm_3 = ChangeGuideModule(512, device=device)
        # self.cgm_4 = ChangeGuideModule(512, device=device)

        # #相比v2 额外的模块   Additional modules compared to v2
        # self.upsample2x=nn.UpsamplingBilinear2d(scale_factor=2)
        # self.decoder_module4 = BasicConv2d(1024,512,3,1,1, device=device)
        # self.decoder_module3 = BasicConv2d(768,256,3,1,1, device=device)
        # self.decoder_module2 = BasicConv2d(384,128,3,1,1, device=device)

        self.sigmoid = nn.Sigmoid()

    # def forward(self, A,B=None):
    #     if B == None:
    #         B = A

    def forward(self,A,B):

        size = A.size()[2:]
        layer1_pre = self.inc(A)                 #[1, 64, 256, 256]
        layer1_A = self.down1(layer1_pre)        #[1, 128, 128, 128]
        layer2_A = self.down2(layer1_A)          #[1, 256, 64, 64]
        layer3_A = self.down3(layer2_A)          #[1, 512, 32, 32]
        layer4_A = self.down4(layer3_A)          #[1, 512, 16, 16]

        layer1_pre = self.inc(B)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        x1 = [layer1_A, layer2_A, layer3_A, layer4_A]
        x2 = [layer1_B, layer2_B, layer3_B, layer4_B]
        x = [x1, x2]

        cp = self.DDLNet(x)



        cp = self.sigmoid (cp)



        return cp   #final_map #, change_map
    




if __name__ == "__main__":

        # model = models.vgg16_bn(pretrained=True)
        # import torchinfo
        # torchinfo.summary(model, input_size=[(1,3,256,256)])


        x1 = torch.rand([1, 3,256, 256])
        x2 = torch.rand([1, 3,256, 256])
        # # print ("x1 size: " , x1.size()[2:])
        # # print ("x1 min: ", x1.min(), "x1 max: ", x1.max())
        # # print ("x mean:", x1.mean(), "x std: ", x1.std())
        model = DDLNet_MyModel()
        y1 = model(x1, x2)
        print ("y1:   ", y1.shape)   #[1, 8,256, 256]


        # import torchinfo
        # torchinfo.summary(model, input_size=[(1,3,256,256), (1,3,256,256)])