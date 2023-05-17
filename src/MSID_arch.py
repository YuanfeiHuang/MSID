import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from . import Upsamplers as Upsamplers


class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()

        self.dw = torch.nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_ch,
                bias=bias,
                padding_mode=padding_mode,
        )

        self.pw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, input):
        out = self.dw(input)
        out = self.pw(out)
        return out


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw=torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        )

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'silu':
        layer = nn.SiLU(inplace)
    elif act_type == 'gelu':
        layer = nn.GELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class SLKA(nn.Module):
    def __init__(self, n_feats, k=21, d=3, shrink=0.5, scale=2):
        super().__init__()
        f = int(n_feats*shrink)
        self.head = nn.Conv2d(n_feats, f, 1)
        self.proj_2 = nn.Conv2d(f, f, kernel_size=1)
        self.activation = nn.GELU()
        self.LKA = nn.Sequential(
            nn.Conv2d(f, f, 1),
            conv_layer(f, f, k // d, dilation=1, groups=f),
            self.activation,
            nn.Conv2d(f, f, 1),
            conv_layer(f, f, 2*d-1, groups=f),
            self.activation,
            conv_layer(f, f, kernel_size=1),
        )
        self.tail = nn.Conv2d(f, n_feats, 1)
        self.scale = scale

    def forward(self, x):
        c1 = self.head(x)
        c2 = F.max_pool2d(c1, kernel_size=self.scale * 2 + 1, stride=self.scale)
        c2 = self.LKA(c2)
        c3 = F.interpolate(c2, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        a = self.tail(c3 + self.proj_2(c1))
        a = F.sigmoid(a)
        return x * a


class AFD(nn.Module):
    def __init__(self, in_channels, conv=nn.Conv2d, attn_shrink=0.25, act_type='silu', attentionScale=2):
        super(AFD, self).__init__()

        kwargs = {'padding': 1}
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = conv_block(in_channels, self.dc, 1, act_type=act_type)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **kwargs)
        self.act = activation(act_type)

        self.c2_d = conv_block(self.remaining_channels, self.dc, 1, act_type=act_type)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3,  **kwargs)

        self.c3 = conv(self.remaining_channels, self.rc, kernel_size=5,  **{'padding': 2})

        self.c3_d = conv_block(self.remaining_channels, self.dc, 1, act_type=act_type)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=7, **{'padding': 3})

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)

        self.esa = SLKA(in_channels, k=21, d=3, shrink=attn_shrink, scale=attentionScale)


    def forward(self, input):
        distilled_c1 = self.c1_d(input)
        r_c1 = self.act(self.c1_r(input))

        distilled_c2 = self.c2_d(r_c1)
        r_c2 = self.act(self.c2_r(r_c1))
        r_c3 = self.act(self.c3(r_c2))

        distilled_c3 = self.c3_d(r_c3)
        r_c4 = self.act(self.c3_r(r_c3))
        r_c5 = self.act(self.c4(r_c4))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c5], dim=1)
        out = self.c5(out)
        out_fused = self.esa(out)
        out_fused = out_fused + input

        return out_fused


class MSID(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=56, num_block=10, num_out_ch=3, upscale=3,
                 conv='BSConvU', upsampler='pixelshuffledirect', attn_shrink=0.25, act_type='gelu'):
        super(MSID, self).__init__()
        kwargs = {'padding': 1}
        if conv == 'DepthWiseConv':
            self.conv = DepthWiseConv
        elif conv == 'BSConvU':
            self.conv = BSConvU
        else:
            self.conv = nn.Conv2d
        self.fea_conv = self.conv(num_in_ch * 4, num_feat, kernel_size=3, **kwargs)

        self.B1 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=2)
        self.B2 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=2)
        self.B3 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=2)
        self.B4 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=2)
        self.B5 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=3)
        self.B6 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=3)
        self.B7 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=3)
        self.B8 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=4)
        self.B9 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type, attentionScale=4)
        self.B10 = AFD(in_channels=num_feat, conv=self.conv, attn_shrink=attn_shrink, act_type=act_type,attentionScale=4)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()
        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)

        if upsampler == 'pixelshuffledirect':
            self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pixelshuffleblock':
            self.upsampler = Upsamplers.PixelShuffleBlcok(in_feat=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'nearestconv':
            self.upsampler = Upsamplers.NearestConv(in_ch=num_feat, num_feat=num_feat, num_out_ch=num_out_ch)
        elif upsampler == 'pa':
            self.upsampler = Upsamplers.PA_UP(nf=num_feat, unf=24, out_nc=num_out_ch)
        else:
            raise NotImplementedError(("Check the Upsampeler. None or not support yet"))

    def forward(self, input):
        input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)
        out_B9 = self.B9(out_B8)
        out_B10 = self.B10(out_B9)

        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8, out_B9, out_B10], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)
        out_lr = self.c2(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output
