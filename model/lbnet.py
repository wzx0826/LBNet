from model import common
import torch
import torch.nn as nn
from util.tools import extract_image_patches, reduce_mean, reduce_sum, same_padding, reverse_patches
from util.rlutrans import  Mlp, TransBlock
import torch.nn.functional as F

def make_model(args, parent=False):
    return LBNet(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        y = self.sigmoid(x)
        return y * res


class FRDAB(nn.Module):
    def __init__(self, n_feats=32):
        super(FRDAB, self).__init__()

        self.c1 = common.default_conv(n_feats, n_feats, 1)
        self.c2 = common.default_conv(n_feats, n_feats // 2, 3)
        self.c3 = common.default_conv(n_feats, n_feats // 2, 3)
        self.c4 = common.default_conv(n_feats*2, n_feats, 3)
        self.c5 = common.default_conv(n_feats // 2, n_feats // 2, 3)
        self.c6 = common.default_conv(n_feats*2, n_feats, 1)

        self.se = CALayer(channel=2*n_feats, reduction=16)
        self.sa = SpatialAttention()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x

        y1 = self.act(self.c1(x))

        y2 = self.act(self.c2(y1))
        y3 = self.act(self.c3(y1))

        cat1 = torch.cat([y1, y2, y3], 1)

        y4 = self.act(self.c4(cat1))

        y5 = self.c5(y3)  # 16

        cat2 = torch.cat([y2, y5, y4], 1)

        ca_out = self.se(cat2)
        sa_out = self.sa(cat2)

        y6 = ca_out + sa_out

        y7 = self.c6(y6)

        output = res + y7

        return output


class LFFM(nn.Module):
    def __init__(self, n_feats=32):
        super(LFFM, self).__init__()

        self.b1 = FRDAB(n_feats=n_feats)
        self.b2 = FRDAB(n_feats=n_feats)
        self.b3 = FRDAB(n_feats=n_feats)

        self.c1 = nn.Conv2d(2 * n_feats, n_feats, 1, stride=1, padding=0, groups=2)
        self.c2 = nn.Conv2d(3 * n_feats, n_feats, 1, stride=1, padding=0, groups=2)
        self.c3 = nn.Conv2d(4 * n_feats, n_feats, 1, stride=1, padding=0, groups=1)

    def forward(self, x):
        res = x

        out1 = self.b1(x)
        dense1 = torch.cat([x, out1], 1)

        out2 = self.b2(self.c1(dense1))

        dense2 = torch.cat([x, out1, out2], 1)
        out3 = self.b3(self.c2(dense2))

        dense3 = torch.cat([x, out1, out2, out3], 1)
        out4 = self.c3(dense3)

        output = res + out4

        return output


## Residual Channel Attention Network (RCAN)
class LBNet(nn.Module):
    def __init__(self, args, conv=common.default_conv, norm_layer=nn.LayerNorm):
        super(LBNet, self).__init__()

        kernel_size = 3
        scale = args.scale[0]
        n_feat = args.n_feats
        num_head = args.num_heads

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        self.head = conv(args.n_colors, n_feat, kernel_size)

        self.r1 = LFFM(n_feats=n_feat)
        self.r2 = LFFM(n_feats=n_feat)
        self.r3 = LFFM(n_feats=n_feat)

        self.se1 = CALayer(channel=n_feat, reduction=16)
        self.se2 = CALayer(channel=n_feat, reduction=16)
        self.se3 = CALayer(channel=n_feat, reduction=16)

        self.attention = TransBlock(n_feat=n_feat, dim=n_feat*9, num_heads = num_head)
        self.attention2 = TransBlock(n_feat=n_feat, dim=n_feat*9, num_heads = num_head)

        self.c1 = common.default_conv(6 * n_feat, n_feat, 1)
        self.c2 = common.default_conv(n_feat, n_feat, 3)
        self.c3 = common.default_conv(n_feat, n_feat, 3)

        modules_tail = [
            conv(n_feat, 4 * 4 * 3, 3),
            nn.PixelShuffle(4),
        ]
        self.tail = nn.Sequential(*modules_tail)

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)


    def forward(self, x):
        (H, W) = (x.shape[2], x.shape[3])
        y_input1 = self.sub_mean(x)
        y_input = self.head(y_input1)
        res = y_input

        y1 = self.r1(y_input)
        y2 = self.r2(y1)
        y3 = self.r3(y2)

        y5 = self.r1(y3 + self.se1(y1))
        y6 = self.r2(y5 + self.se2(y2))
        y6_1 = self.r3(y6 + self.se3(y3))

        y7 = torch.cat([y1, y2, y3, y5, y6, y6_1], dim=1)
        y8 = self.c1(y7)
        
        
        b, c, h, w = y8.shape
        y8 = extract_image_patches(y8, ksizes=[3, 3],
                                      strides=[1,1],
                                      rates=[1, 1],
                                      padding='same') # 16*2304*576
        y8 = y8.permute(0,2,1)
        out_transf1 = self.attention(y8)
        out_transf1 = self.attention(out_transf1)
        out_transf1 = self.attention(out_transf1)
        out1 = out_transf1.permute(0, 2, 1)
        out1 = reverse_patches(out1, (h, w), (3, 3), 1, 1)
        y9 = self.c2(out1)
        
        
        b2, c2, h2, w2 = y9.shape
        y9 = extract_image_patches(y9, ksizes=[3, 3],
                                      strides=[1,1],
                                      rates=[1, 1],
                                      padding='same') # 16*2304*576
        y9 = y9.permute(0,2,1)
        out_transf2 = self.attention2(y9)
        out_transf2 = self.attention2(out_transf2)
        out_transf2 = self.attention2(out_transf2)
        out2 = out_transf2.permute(0, 2, 1)
        out2 = reverse_patches(out2, (h, w), (3, 3), 1, 1)
        
        y10 = self.c3(out2)
        

        output = y10 + res #y10 + res
        output = self.tail(output)

        y = self.add_mean(output)

        return y

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
