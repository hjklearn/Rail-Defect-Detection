import torch
import torch.nn as nn
from .mobilenetv2 import *
import torch.nn.functional as F
# from .van import *


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # x = torch.sigmoid(x)
        x = torch.sigmoid(x) * residual  #

        return x

class DSConv(nn.Module):
    def __init__(self, in_channel, out_channel, rate):
        super(DSConv, self).__init__()
        self.depth = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=rate, stride=1, dilation=rate, groups=in_channel),
                                   nn.BatchNorm2d(in_channel),
                                   nn.PReLU())
        self.point = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.PReLU())

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)

        return x


# class Boundry(nn.Module):
#     def __init__(self, in_channels):
#         super(Boundry, self).__init__()
#         self.relu = nn.ReLU()
#         self.avg_max = nn.AdaptiveMaxPool2d(1)
#         self.sig = nn.Sigmoid()
#         self.conv3_d1 = nn.Sequential(
#             nn.Conv2d(2 * in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), stride=1, dilation=(1, 1)
#                       ),
#             nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), stride=1, dilation=(1, 1)
#                       ),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU())
#
#         self.conv3_d2 = nn.Sequential(
#             nn.Conv2d(3 * in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), stride=1, dilation=(1, 1)
#                       ),
#             nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), stride=1, dilation=(1, 1)
#                       ),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU())
#
#
#     def forward(self, rgb, d):
#         mul1 = rgb.mul(d)
#         add = torch.cat([rgb, mul1, d], dim=1)
#         add = self.conv3_d2(add)
#         add_max, _ = torch.max(add, dim=1, keepdim=True)
#         mul_add_max = add_max.mul(add)
#
#         add_avg_max = self.avg_max(add)
#         add_avg_max_sig = self.sig(add_avg_max)
#         mul_add_avg_max_sig = add_avg_max_sig.mul(add)
#
#         cat = torch.cat((mul_add_max, mul_add_avg_max_sig), dim=1)
#         cat_conv = self.conv3_d1(cat)
#         out = cat_conv + add
#
#         return  out


class Boundry(nn.Module):
    def __init__(self, in_channels):
        super(Boundry, self).__init__()

        self.conv1x1_1 = nn.Conv2d(in_channels, 2 * in_channels, 1)
        self.conv1x1_2 = nn.Conv2d(in_channels, 2 * in_channels, 1)
        self.conv1x1_3 = nn.Conv2d(1, in_channels, 1)
        self.conv1x1_4 = nn.Conv2d(2 * in_channels, in_channels, 1)

        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.max8 = nn.MaxPool2d(kernel_size=8, stride=8)


        self.conv3_d1 = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=(3, 1), padding=(1, 0), stride=1, dilation=(1, 1)
                      ),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), stride=1, dilation=(1, 1)
                      ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        self.conv3_d2 = nn.Sequential(
            nn.Conv2d(6 * in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), stride=1, dilation=(1, 1)
                      ),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), stride=1, dilation=(1, 1)
                      ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())


    def forward(self, rgb, d):
        rgb_c = self.conv1x1_1(rgb)
        d_c = self.conv1x1_2(d)
        mul1 = rgb_c.mul(d_c)
        # add = torch.cat([mul1, rgb_c, d_c], dim=1)
        # add = self.conv3_d1(add)
        add = mul1 + rgb_c + d_c
        add_c = self.conv1x1_4(add)

        avgmax1 = self.max2(add)
        avgmax3 = self.max4(add)
        avgmax5 = self.max8(add)
        max1, _ = torch.max(add, dim=1, keepdim=True)
        max1 = self.conv1x1_3(max1)

        avgmax1_up = F.interpolate(input=avgmax1, size=(add.size()[2], add.size()[3]))
        avgmax3_up = F.interpolate(input=avgmax3, size=(add.size()[2], add.size()[3]))
        avgmax5_up = F.interpolate(input=avgmax5, size=(add.size()[2], add.size()[3]))

        cat = torch.cat([avgmax1_up, avgmax3_up, avgmax5_up], dim=1)
        cat_conv = self.conv3_d2(cat)
        out = cat_conv + max1 + add_c


        return  out



class fusion(nn.Module):
    def __init__(self, in_channels):
        super(fusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv3_d = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), stride=1, dilation=(1, 1)
                      ),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), stride=1, dilation=(1, 1)
                      ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())

    def forward(self, rgb, t):
        mul_rt = rgb.mul(t)

        rgb_sig = self.sigmoid(rgb)
        t_sig = self.sigmoid(t)

        mul_r = rgb_sig.mul(t)
        add_r = mul_r + rgb
        mul_t = t_sig.mul(rgb)
        add_t = mul_t + t

        r_mul = add_r.mul(mul_rt)
        t_mul = add_t.mul(mul_rt)

        cat_all = torch.cat((r_mul, t_mul), dim=1)
        out = self.conv3_d(cat_all)


        return out



class MFI(nn.Module):
    def __init__(self, in_channels):
        super(MFI, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.avgmax = nn.AdaptiveMaxPool2d(1)
        self.sig = nn.Sigmoid()
        self.conv3_d1 = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), stride=1, dilation=(1, 1)
                      ),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), stride=1, dilation=(1, 1)
                      ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())

    def forward(self, rgb, depth, edge):

        jian_r = rgb - edge
        jian_d = depth - edge
        jian_r = torch.abs(jian_r)
        jian_d = torch.abs(jian_d)

        mul_r_d = jian_r.mul(jian_d)
        add_r_d = jian_r + jian_d

        cat = torch.cat((mul_r_d, add_r_d), dim=1)
        cat_conv = self.conv3_d1(cat)

        out = cat_conv + edge

        return out


class Mirror_model(nn.Module):
    def __init__(self):
        super(Mirror_model, self).__init__()
        self.layer1_rgb = mobilenet_v2().features[0:2]
        self.layer2_rgb = mobilenet_v2().features[2:4]
        self.layer3_rgb = mobilenet_v2().features[4:7]
        self.layer4_rgb = mobilenet_v2().features[7:17]
        self.layer5_rgb = mobilenet_v2().features[17:18]

        self.layer1_t = mobilenet_v2().features[0:2]
        self.layer2_t = mobilenet_v2().features[2:4]
        self.layer3_t = mobilenet_v2().features[4:7]
        self.layer4_t = mobilenet_v2().features[7:17]
        self.layer5_t = mobilenet_v2().features[17:18]

        self.boundary = Boundry(16)



        self.fusion1 = fusion(24)
        self.fusion2 = fusion(32)
        self.fusion3 = fusion(160)
        self.fusion4 = fusion(320)

        self.MFI1 = MFI(160)
        self.MFI2 = MFI(16)
        self.MFI3 = MFI(16)


        self.conv16_1_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(16, 1, 1))
        self.conv16_1_2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(16, 1, 1))
        self.conv16_1_3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(16, 1, 1))
        self.conv16_1_4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(16, 1, 1))
        self.conv16_1_5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(16, 1, 1))

        self.conv16_160 = nn.Sequential(nn.Conv2d(16, 160, 1),
                                        nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True))
        self.conv16_32 = nn.Conv2d(16, 32, 1)
        self.conv16_24 = nn.Conv2d(16, 24, 1)


        self.conv32_16 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                       nn.Conv2d(32, 16, 1)
                                       )
        self.conv24_16 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                       nn.Conv2d(24, 16, 1)
                                       )


        self.conv160_16 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
                                      nn.Conv2d(160, 16, 1),
                                      )
        self.conv32_16 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                       nn.Conv2d(32, 16, 1)
                                       )
        self.conv24_16 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                       nn.Conv2d(24, 16, 1)
                                       )



        self.conv_480_160 = nn.Sequential(nn.Conv2d(480, 160, 1),
                                        nn.BatchNorm2d(160),
                                        nn.ReLU(inplace=True))
        self.conv_192_16 = nn.Sequential(nn.Conv2d(192, 16, 1),
                                          nn.BatchNorm2d(16),
                                          nn.ReLU(inplace=True))
        self.conv_40_16 = nn.Sequential(nn.Conv2d(40, 16, 1),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(inplace=True))

        self.conv_480_160_1 = nn.Sequential(nn.Conv2d(480, 160, 1),
                                          nn.BatchNorm2d(160),
                                          nn.ReLU(inplace=True))
        self.conv_192_16_1 = nn.Sequential(nn.Conv2d(192, 16, 1),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(inplace=True))
        self.conv_40_16_1 = nn.Sequential(nn.Conv2d(40, 16, 1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(inplace=True))

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)





    def forward(self, rgb, depth):
        x1_rgb = self.layer1_rgb(rgb)
        x2_rgb = self.layer2_rgb(x1_rgb)
        x3_rgb = self.layer3_rgb(x2_rgb)
        x4_rgb = self.layer4_rgb(x3_rgb)
        x5_rgb = self.layer5_rgb(x4_rgb)

        depth = torch.cat([depth, depth, depth], dim=1)
        x1_depth = self.layer1_t(depth)
        x2_depth = self.layer2_t(x1_depth)
        x3_depth = self.layer3_t(x2_depth)
        x4_depth = self.layer4_t(x3_depth)
        x5_depth = self.layer5_t(x4_depth)


        edge = self.boundary(x1_rgb, x1_depth)
        edge_conv = self.conv16_1_4(edge)
        edge_160 = self.conv16_160(edge)

        x2_r_t = self.fusion1(x2_rgb, x2_depth)
        x2_rgb_en = x2_rgb + x2_r_t
        x2_depth_en = x2_depth + x2_r_t

        x3_r_t = self.fusion2(x3_rgb, x3_depth)
        x3_rgb_en = x3_rgb + x3_r_t
        x3_depth_en = x3_depth + x3_r_t

        x4_r_t = self.fusion3(x4_rgb, x4_depth)
        x4_rgb_en = x4_rgb + x4_r_t
        x4_depth_en = x4_depth + x4_r_t

        x5_r_t = self.fusion4(x5_rgb, x5_depth)
        x5_rgb_en = x5_rgb + x5_r_t
        x5_depth_en = x5_depth + x5_r_t

        x5_rgb_en_up2 = self.up2(x5_rgb_en)
        x5_depth_en_up2 = self.up2(x5_depth_en)
        cat_5_4_r = torch.cat((x5_rgb_en_up2, x4_rgb_en), dim=1)
        cat_5_4_r_480_160 = self.conv_480_160(cat_5_4_r)
        cat_5_4_t = torch.cat((x5_depth_en_up2, x4_depth_en), dim=1)
        cat_5_4_t_480_160 = self.conv_480_160_1(cat_5_4_t)
        add_5_4 = self.MFI1(cat_5_4_r_480_160, cat_5_4_t_480_160, edge_160)
        add_5_4_conv = self.conv160_16(add_5_4)
        f3 = add_5_4_conv.mul(edge) + edge


        cat_5_4_r_480_160_up2 = self.up2(cat_5_4_r_480_160)
        cat_5_4_3_r = torch.cat((cat_5_4_r_480_160_up2, x3_rgb_en), dim=1)
        cat_5_4_3_r_192_16 = self.conv_192_16(cat_5_4_3_r)
        cat_5_4_3_r_192_16_up4 = self.up4(cat_5_4_3_r_192_16)
        cat_5_4_t_480_160_up2 = self.up2(cat_5_4_t_480_160)
        cat_5_4_3_t = torch.cat((cat_5_4_t_480_160_up2, x3_depth_en), dim=1)
        cat_5_4_3_t_196_16 = self.conv_192_16_1(cat_5_4_3_t)
        cat_5_4_3_t_196_16_up4 = self.up4(cat_5_4_3_t_196_16)
        add_5_4_3 = self.MFI2(cat_5_4_3_r_192_16_up4, cat_5_4_3_t_196_16_up4, f3)
        f2 = add_5_4_3.mul(edge) + edge


        cat_5_4_3_r_192_16_up2 = self.up2(cat_5_4_3_r_192_16)
        cat_5_4_3_t_196_16_up2 = self.up2(cat_5_4_3_t_196_16)
        cat_5_4_3_2_r = torch.cat((cat_5_4_3_r_192_16_up2, x2_rgb_en), dim=1)
        cat_5_4_3_2_r_40_16 = self.conv_40_16(cat_5_4_3_2_r)
        cat_5_4_3_2_r_40_16_up2 = self.up2(cat_5_4_3_2_r_40_16)
        cat_5_4_3_2_t = torch.cat((cat_5_4_3_t_196_16_up2, x2_depth_en), dim=1)
        cat_5_4_3_2_t_40_16 = self.conv_40_16_1(cat_5_4_3_2_t)
        cat_5_4_3_2_t_40_16_up2 = self.up2(cat_5_4_3_2_t_40_16)
        add_5_4_3_2 = self.MFI3(cat_5_4_3_2_r_40_16_up2, cat_5_4_3_2_t_40_16_up2, f2)
        f1 = add_5_4_3_2


        out3 = self.conv16_1_1(add_5_4_conv)
        out2 = self.conv16_1_2(add_5_4_3)
        out1 = self.conv16_1_3(f1)
        edge1 = self.conv16_1_4(f2)
        edge2 = self.conv16_1_5(f3)

        return out1, out2, out3, edge_conv, edge1, edge2


