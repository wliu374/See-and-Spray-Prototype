from sympy import *
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader

from transformers import SegformerForSemanticSegmentation,SegformerModel
import timm

class segFormer(nn.Module):
    def __init__(self):
        self.segformerForSemanticSegmentation = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    def forward(self, x):
        output = self.segformerForSemanticSegmentation(x)
        output = output.logits()
        output = torch.sigmoid(output)
        output = F.interpolate(output,size = x.shape[-2:], mode="bilinear", align_corners=False)
        return output

import torch
import torch.nn as nn


class MobileNetV4Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV4Encoder, self).__init__()
        self.backbone = timm.create_model(
            'mobilenetv4_conv_small.e2400_r224_in1k',
            pretrained=pretrained,
            features_only=True  # ✅ Extracts multi-scale feature maps
        )

    def forward(self, x):
        features = self.backbone(x)  # Returns multiple feature maps
        return features


# Decoder: Upsampling the feature maps back to input size
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Upsampling layers
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(960, 256, kernel_size=3, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d( 256 + 96, 128, kernel_size=3, padding=1)  # Skip connection

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)  # Skip connection

        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1)  # Skip connection

        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Conv2d(32 + 32, 16, kernel_size=3, padding=1)  # Skip connection

        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)  # Final segmentation mask

        self.weights_init()

    def forward(self, features):
        f0, f1, f2, f3, f4 = features  # Unpack feature maps (deepest to shallowest)

        x = self.upsample1(f4)
        x = self.conv1(x)

        x = F.interpolate(x, size=f3.shape[2:], mode="bilinear", align_corners=True)
        x = self.upsample2(torch.cat([x, f3], dim=1))
        x = self.conv2(x)

        x = F.interpolate(x, size=f2.shape[2:], mode="bilinear", align_corners=True)
        x = self.upsample3(torch.cat([x, f2], dim=1))
        x = self.conv3(x)

        x = F.interpolate(x, size=f1.shape[2:], mode="bilinear", align_corners=True)
        x = self.upsample4(torch.cat([x, f1], dim=1))
        x = self.conv4(x)

        x = F.interpolate(x, size=f0.shape[2:], mode="bilinear", align_corners=True)
        x = self.upsample5(torch.cat([x, f0], dim=1))
        x = self.conv5(x)

        x = self.final_conv(x)  # Final segmentation mask
        return x
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.up1 =  nn.Sequential(
            convBlock(960,64),
            nn.Conv2d(64,64,kernel_size=1),
            nn.Sigmoid()
        )
        self.up2 = nn.Sequential(
            convBlock(64,64),
            nn.Conv2d(64,1,kernel_size=1),
            nn.Sigmoid()
        )
        self.weights_init()
    def forward(self,features,input):
        _, _, f2, _, f4 = features  # Unpack feature maps (deepest to shallowest)
        x = self.up1(f4)
        x = F.interpolate(x, size=f2.shape[2:], mode="bilinear", align_corners=True)
        x = self.up2(f2 + x)
        x = F.interpolate(x, size=input.shape[2:], mode="bilinear", align_corners=True)
        return x
    
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class convBlock(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(convBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1),
            nn.Conv2d(output_channel,output_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

# Full segmentation model
class MobileNetV4Segmentation(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV4Segmentation, self).__init__()
        self.encoder = MobileNetV4Encoder()
        self.decoder = Decoder2()

    def forward(self, x):
        features = self.encoder(x)  # Extract feature maps
        output = self.decoder(features,x)  # Upsample to full resolution
        return output
    
if __name__ == '__main__':
    # net =segFormer().cuda()
    # net = timm.create_model('mobilenetv4_conv_blur_medium.e500_r224_in1k', pretrained=True).cuda()
    # net = MobileNetV4Encoder().cuda()
    # net = MobileNetV4Encoder().cuda().half()
    net = MobileNetV4Segmentation().cuda().half()
    net = net.eval()
    x = net(torch.randn(1,3,320,320).cuda().half())
    # for feature in x:
    #     print(feature.size())
    print("OK")
    print(x.size())