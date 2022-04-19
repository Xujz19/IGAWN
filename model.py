import torch
import torch.nn as nn
import torch.nn.functional as F


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x1, x2, x3, x4):
    N, C, H, W = x1.size()
    # print([in_batch, in_channel, in_height, in_width])
    x1 = x1 / 2
    x2 = x2 / 2
    x3 = x3 / 2
    x4 = x4 / 2

    h = torch.zeros([N, C, H*2, W*2]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.requires_grad = False

    def forward(self, x_LL, x_HL, x_LH, x_HH):
        return iwt_init(x_LL, x_HL, x_LH, x_HH)


class SA(nn.Module):
    def __init__(self, n_feats, reduction):
        super(SA, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats//reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_feats//reduction, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.body(x)
        out = att * x
        return out


class CA(nn.Module):
    def __init__(self, n_feats, reduction):
        super(CA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // reduction, n_feats, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = self.avg_pool(x)
        att = self.body(x_avg)
        out = att * x
        return out


class AttentionDWT(nn.Module):
    def __init__(self, n_feats, reduction=8):
        super(AttentionDWT, self).__init__()

        self.DWT = DWT()
        # self.DWT = DWTForward(J=1, wave='haar', mode='zero')

        self.sa_LL = SA(n_feats, reduction)
        self.sa_HL = SA(n_feats, reduction)
        self.sa_LH = SA(n_feats, reduction)
        self.sa_HH = SA(n_feats, reduction)

        self.ca = CA(n_feats*4, reduction*2)
        self.compress = nn.Conv2d(n_feats*4, n_feats*2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_LL0, x_HL0, x_LH0, x_HH0 = self.DWT(x)
        x_LL = self.sa_LL(x_LL0)
        x_LH = self.sa_LH(x_LH0)
        x_HL = self.sa_HL(x_HL0)
        x_HH = self.sa_HH(x_HH0)

        out = self.ca(torch.cat([x_LL, x_HL, x_LH, x_HH], dim=1)) + torch.cat([x_LL0, x_HL0, x_LH0, x_HH0], dim=1)
        out = self.compress(out)
        return out


class AttentionIDWT(nn.Module):
    def __init__(self, n_feats, reduction=8):
        super(AttentionIDWT, self).__init__()

        self.IDWT = IDWT()
        # self.IDWT = DWTInverse(wave='haar', mode='zero')

        self.ca = CA(n_feats, reduction*2)

        self.sa_LL = SA(n_feats//4, reduction)
        self.sa_HL = SA(n_feats//4, reduction)
        self.sa_LH = SA(n_feats//4, reduction)
        self.sa_HH = SA(n_feats//4, reduction)

        self.expansion = nn.Conv2d(n_feats//4, n_feats//2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        N, C, H, W = x.size()
        [x_LL0, x_HL0, x_LH0, x_HH0] = torch.split(x, C//4, dim=1)
        x = self.ca(x)
        [x_LL, x_HL, x_LH, x_HH] = torch.split(x, C//4, dim=1)
        x_LL = self.sa_LL(x_LL) + x_LL0
        x_LH = self.sa_LH(x_LH) + x_LH0
        x_HL = self.sa_HL(x_HL) + x_HL0
        x_HH = self.sa_HH(x_HH) + x_HH0

        out = self.IDWT(x_LL, x_HL, x_LH, x_HH)
        out = self.expansion(out)
        return out


class FFTLayer(nn.Module):
    def __init__(self, feat_channel, cond_channel):
        super(FFTLayer, self).__init__()
        self.feat_channel = feat_channel
        self.SFT_scale_conv0 = nn.Conv2d(cond_channel, cond_channel, 1)
        self.SFT_scale_conv1 = nn.Conv2d(cond_channel, feat_channel//4, 1)
        self.SFT_shift_conv0 = nn.Conv2d(cond_channel, cond_channel, 1)
        self.SFT_shift_conv1 = nn.Conv2d(cond_channel, feat_channel, 1)

    def forward(self, x0, x1):
        # x0: feature, x1: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x1), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x1), 0.1, inplace=True))

        [x_LL, x_HL, x_LH, x_HH] = torch.split(x0, self.feat_channel//4, dim=1)
        x_LL = x_LL * (scale + 1)
        out = torch.cat([x_LL, x_HL, x_LH, x_HH], dim=1) + shift
        return out


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, residual=True):
        super(ResBlock, self).__init__()

        self.residual = residual

        self.body = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        identity = x
        x = self.body(x)
        if self.residual:
            out = identity + x
        else:
            out = x
        return out


class ConvBlock(nn.Module):
    def __init__(self, n_feats, bias=True):
        super(ConvBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        res = self.body(x)
        return res


class IGAWN(nn.Module):
    def __init__(self, n_colors=3, n_feats=32, bias=False):
        super(IGAWN, self).__init__()

        # IntensityNet
        self.netE_conv_head = nn.Conv2d(n_colors, n_feats//2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.netE_conv1_1 = AttentionDWT(n_feats//2, 2)
        self.netE_conv1_2 = ResBlock(n_feats, n_feats)

        self.netE_conv2_1 = AttentionDWT(n_feats, 4)
        self.netE_conv2_2 = ResBlock(n_feats*2, n_feats*2)

        self.netE_conv3_1 = AttentionDWT(n_feats*2, 4)
        self.netE_conv3_2 = ResBlock(n_feats*4, n_feats*4)

        self.netE_deconv3_1 = ResBlock(n_feats*4, n_feats*4)
        self.fft3 = FFTLayer(n_feats * 4, n_feats * 4)
        self.netE_deconv3_2 = AttentionIDWT(n_feats*4, 4)

        self.netE_deconv2_1 = ResBlock(n_feats*2, n_feats*2)
        self.fft2 = FFTLayer(n_feats * 2, n_feats * 2)
        self.netE_deconv2_2 = AttentionIDWT(n_feats*2, 4)

        self.netE_deconv1_1 = ResBlock(n_feats, n_feats)
        self.fft1 = FFTLayer(n_feats, n_feats)
        self.netE_deconv1_2 = AttentionIDWT(n_feats, 2)

        self.netE_conv_end = nn.Conv2d(n_feats//2, n_colors, kernel_size=3, stride=1, padding=1, bias=bias)

        # IlluminationNet
        self.netI_conv_head = nn.Conv2d(3, n_feats//2, kernel_size=3, stride=1, padding=1)
        self.netI_conv1_1 = AttentionDWT(n_feats//2, 2)
        self.netI_conv1_2 = ResBlock(n_feats, n_feats)

        self.netI_conv2_1 = AttentionDWT(n_feats, 4)
        self.netI_conv2_2 = ResBlock(n_feats*2, n_feats*2)

        self.netI_conv3_1 = AttentionDWT(n_feats*2, 4)
        self.netI_conv3_2 = ResBlock(n_feats*4, n_feats*4)

        self.netI_deconv3_1 = ResBlock(n_feats*4, n_feats*4)
        self.netI_deconv3_2 = AttentionIDWT(n_feats*4, 4)

        self.netI_deconv2_1 = ResBlock(n_feats*2, n_feats*2)
        self.netI_deconv2_2 = AttentionIDWT(n_feats*2, 4)

        self.netI_deconv1_1 = ResBlock(n_feats, n_feats)
        self.netI_deconv1_2 = AttentionIDWT(n_feats, 2)
        self.netI_conv_end = nn.Conv2d(n_feats//2, 1, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        map0 = self.netI_conv_head(x)
        map1 = self.netI_conv1_2(self.netI_conv1_1(map0))
        map2 = self.netI_conv2_2(self.netI_conv2_1(map1))
        map3 = self.netI_conv3_2(self.netI_conv3_1(map2))
      
        map3 = self.netI_deconv3_1(map3)
        map_dec3 = self.netI_deconv3_2(map3)
        map2 = self.netI_deconv2_1(map2 + map_dec3)
        map_dec2 = self.netI_deconv2_2(map2)
        map1 = self.netI_deconv1_1(map1 + map_dec2)
        map_dec1 = self.netI_deconv1_2(map1)
        map_out = self.netI_conv_end(map_dec1)

        x0 = self.netE_conv_head(x)
        x1 = self.netE_conv1_2(self.netE_conv1_1(x0))
        x2 = self.netE_conv2_2(self.netE_conv2_1(x1))
        x3 = self.netE_conv3_2(self.netE_conv3_1(x2))

        x_dec3 = self.fft3(x3, map3)
        x_dec3 = self.netE_deconv3_2(self.netE_deconv3_1(x_dec3))
        x_dec2 = self.fft2(x_dec3 + x2, map2)
        x_dec2 = self.netE_deconv2_2(self.netE_deconv2_1(x_dec2))
        x_dec1 = self.fft1(x_dec2 + x1, map1)
        x_dec1 = self.netE_deconv1_2(self.netE_deconv1_1(x_dec1))
        x_out = self.netE_conv_end(x_dec1)

        return x_out, 1 - torch.sigmoid(map_out)


if __name__=='__main__':
    def count_params(model):
        count = 0
        for param in model.parameters():
            param_size = param.size()
            count_of_one_param = 1
            for i in param_size:
                count_of_one_param *= i
            count += count_of_one_param
        print('Total parameters: %d' % count)


    data1 = torch.rand((1, 3, 400, 600))
    data2 = torch.rand((1, 1, 128, 128))
    data1 = data1.cuda()
    data2 = data2.cuda()
    model = IGAWN()
    print(model)
    model.cuda()
    count_params(model)
    for epoch in range(1):
        print(epoch)
        out1, out2 = model(data1)
    print(out1.shape)
    print(out2.shape)


    import time
    time_start = time.time()
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    out1, out2 = model(data1)
    print(time.time() - time_start)






