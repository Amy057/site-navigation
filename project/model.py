
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary as summary


class Inception(nn.Module):
    def __init__(self, in_chans, out_channels):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv3d(in_chans, out_channels // 8, kernel_size=1),
            nn.BatchNorm3d(out_channels // 8),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv3d(in_chans, out_channels // 4, kernel_size=1),
            # nn.BatchNorm3d(out_channels // 4),
            # nn.ReLU(True),
            nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv3d(in_chans, out_channels // 4, kernel_size=1),
            # nn.BatchNorm3d(out_channels // 4),
            # nn.ReLU(True),
            nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            # nn.BatchNorm3d(out_channels // 4),
            # nn.ReLU(True),
            nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(True),
        )

        # 1x1 conv -> 7x7 conv branch
        self.b4 = nn.Sequential(
            nn.Conv3d(in_chans, out_channels // 4, kernel_size=1),
            # nn.BatchNorm3d(out_channels // 4),
            # nn.ReLU(True),
            nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            # nn.BatchNorm3d(out_channels // 4),
            # nn.ReLU(True),
            nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            # nn.BatchNorm3d(out_channels // 4),
            # nn.ReLU(True),
            nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b5 = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(in_chans, out_channels // 8, kernel_size=1),
            nn.BatchNorm3d(out_channels // 8),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        y5 = self.b5(x)
        return torch.cat([y1, y2, y3, y4, y5], 1)


class InceptionV1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionV1, self).__init__()

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.branch1_1 = nn.Conv3d(in_channels, out_channels // 8, kernel_size=1)
        self.bn_branch1_1 = nn.BatchNorm3d(out_channels // 8)
        self.relu_branch1_1 = nn.ReLU(inplace=True)

        self.branch7_7_1 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=1)
        self.branch7_7_2 = nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=7, padding=3)
        self.bn_branch7_7 = nn.BatchNorm3d(out_channels // 4)
        self.relu_branch7_7 = nn.ReLU(inplace=True)

        self.branch5_5_1 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=1)
        self.branch5_5_2 = nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2)
        self.bn_branch5_5 = nn.BatchNorm3d(out_channels // 4)
        self.relu_branch5_5 = nn.ReLU(inplace=True)
        # 定义激活函数ReLU

        self.branch3_3_1 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=1)
        self.branch3_3_2 = nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1)
        self.bn_branch3_3 = nn.BatchNorm3d(out_channels // 4)
        self.relu_branch3_3 = nn.ReLU(inplace=True)

        self.branch1_pool = nn.Conv3d(in_channels, out_channels // 8, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = x.clone()
        branch1_1 = self.branch1_1(x)
        # branch1_1 = self.bn_branch1_1(branch1_1)
        # branch1_1 = self.relu_branch1_1(branch1_1)

        branch7_7 = self.branch7_7_1(x)
        branch7_7 = self.branch7_7_2(branch7_7)
        # branch7_7 = self.bn_branch7_7(branch7_7)
        # branch7_7 = self.relu_branch7_7(branch7_7)

        branch5_5 = self.branch5_5_1(x)
        branch5_5 = self.branch5_5_2(branch5_5)
        # branch5_5 = self.bn_branch5_5(branch5_5)
        # branch5_5 = self.relu_branch5_5(branch5_5)

        branch3_3 = self.branch3_3_1(x)
        branch3_3 = self.branch3_3_2(branch3_3)
        # branch3_3 = self.bn_branch3_3(branch3_3)
        # branch3_3 = self.relu_branch3_3(branch3_3)

        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch1_pool(branch_pool)

        outputs = [branch1_1, branch7_7, branch5_5, branch3_3, branch_pool]
        output = torch.cat(outputs, 1)

        output = torch.add(output, x1)
        # output = self.bn1(output)
        # output = self.relu1(output)

        return output


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):  # inp --> (B,C,D,H,W)
        super(CoordAtt, self).__init__()
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))  # edited
        # B,C,D,H,W                        B,C,1,H,1
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        # B,C,D,H,W                        B,C,1,1,W
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()

        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)  # edited
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, d, h, w = x.size()  # edited

        x_d = self.pool_d(x)  # edited  (n,c,d,1,1) permute(0,1,2,3,4)

        x_h = self.pool_h(x).permute(0, 1, 3, 2, 4)  # (n,c,1,h,1) --> n,c,h,1,1

        x_w = self.pool_w(x).permute(0, 1, 4, 2, 3)  # n,c,1,1,w --> n,c,w,1,1

        # y = torch.cat([x_h, x_w], dim=2)
        y = torch.cat([x_d, x_h, x_w], dim=2)  # edited  shape(y)=(N, C, D+H+W, 1, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # x_h, x_w = torch.split(y, [h, w], dim=2)
        x_d, x_h, x_w = torch.split(y, [d, h, w], dim=2)  # [d, h, w] --> [d], [h], [w]
        # x_w = x_w.permute(0, 1, 3, 2)

        x_h = x_h.permute(0, 1, 3, 2, 4)  # (n,c,1,h,1) --> n,c,h,1,1  01234->01324->01324

        x_w = x_w.permute(0, 1, 3, 4, 2)  # n,c,1,1,w --> n,c,w,1,1   # (n,c,h,w)-->(n,c,w,h)  01234->01423->01342

        a_d = self.conv_d(x_d).sigmoid()

        a_h = self.conv_h(x_h).sigmoid()

        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h * a_d

        return out


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )


class InputTransition(nn.Module):
    def __init__(self, inChans, outChans):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(outChans, outChans, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        # do we want a PRELU here as well?
        x1 = x.clone()
        sobel_x = torch.tensor([[[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]]], dtype=torch.float32)
        sobel_x = sobel_x.to(torch.device('cuda:1'))
        sobel_y = torch.tensor([[[[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]]]], dtype=torch.float32)
        sobel_y = sobel_y.to(torch.device('cuda:1'))

        sobel_result = torch.zeros_like(x1)
        # print("x1.shape:", x1.shape)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[2]):
                image = x1[i, :, j]
                # print("image.shape:", image.shape)

                image = image.unsqueeze(0)
                # print("unsqueeze_image.shape:", image.shape)

                sobel_x_result = F.conv2d(image, sobel_x, padding=1)
                sobel_y_result = F.conv2d(image, sobel_y, padding=1)
                # print("sobel_x_result:", sobel_x_result.shape)
                sobel_layer = torch.sqrt(sobel_x_result ** 2 + sobel_y_result ** 2)
                sobel_result[i, :, j] = sobel_layer.squeeze()
        x1 = sobel_result
        x2 = x.clone()
        x16 = torch.cat((x1, x2, x1, x2, x1, x2, x1, x2,
                         x1, x2, x1, x2, x1, x2, x1, x2), 1)
        return x16



class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool3d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels,
                                   in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # print("x1.shape:", x1.shape)
        # print("x2.shape:", x2.shape)
        x1 = self.up(x1)
        # if x1.size(2) == 1:
        #     if x1.size(2) != x2.size(2):
        #         x1 = x1.repeat(1, 1, x2.size(2), 1, 1)
        diff_z = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        diff_y = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diff_z // 2, diff_z - diff_z // 2,
                        diff_y // 2, diff_y - diff_y // 2,
                        diff_x // 2, diff_x - diff_x // 2])


        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv3d(in_channels, num_classes, kernel_size=1)
        )

class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 16):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.in_conv1 = InputTransition(in_channels, base_c)
        self.in_conv2 = DoubleConv(base_c, base_c)
        self.incept_out16 = InceptionV1(base_c, base_c)
        self.coordatt16 = CoordAtt(base_c, base_c)

        self.down1 = Down(base_c, base_c * 2)
        self.incept_out32 = InceptionV1(base_c * 2, base_c * 2)
        self.coordatt32 = CoordAtt(base_c * 2, base_c * 2)

        self.down2 = Down(base_c * 2, base_c * 4)
        self.incept_out64 = InceptionV1(base_c * 4, base_c * 4)
        self.coordatt64 = CoordAtt(base_c * 4, base_c * 4)

        self.down3 = Down(base_c * 4, base_c * 8)
        self.incept_out128 = InceptionV1(base_c * 8, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
    def forward(self, x):
        x1 = self.in_conv1(x)
        x2_1 = self.in_conv2(x1)
        x_incep_16 = self.incept_out16(x2_1)
        coordatt16 = self.coordatt16(x_incep_16)

        x2_2 = self.down1(coordatt16)
        # print("x2_2.shape:", x2_2.shape)
        x_incep_32 = self.incept_out32(x2_2)
        coordatt32 = self.coordatt32(x_incep_32)

        x3 = self.down2(coordatt32)
        coordatt64 = self.coordatt64(x3)

        x4 = self.down3(coordatt64)

        x5 = self.down4(x4)
        x = self.up1(x5, x4)

        x = self.up2(x, x3)
        # coordatt32_out = self.coordatt32(x)

        x = self.up3(x, x_incep_32)
        up_16_incep = self.incept_out16(x)
        # up_coordatt16 = self.coordatt16(up_16_incep)

        x = self.up4(up_16_incep, x_incep_16)
        up16_incep_out = self.incept_out16(x)

        out = self.out_conv(up16_incep_out)
        return out


if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = UNet().to(device)
    model = nn.DataParallel(model, device_ids=[2, 1, 3])

    input = torch.randn(4, 1, 16, 96, 96)  # BCHW
    input = input.to(device)
    out = model(input)
    # print(model)
    summary.summary(model, (1, 16, 96, 96))
    print(out.shape)
