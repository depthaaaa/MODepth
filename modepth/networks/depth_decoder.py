import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from .graph_unet_layers import Graphformer
from .uper_graph_head import PSP
import numpy as np
from collections import OrderedDict
from modepth.layers import ConvBlock, Conv3x3, DWConv, upsample
from fvcore.nn import FlopCountAnalysis, parameter_count_table


class DispHead(nn.Module):
    def __init__(self, num_output_channels=1, input_dim=100):
        super(DispHead, self).__init__()

        self.num_output_channels = num_output_channels
        self.conv1 = Conv3x3(input_dim, self.num_output_channels)

    def forward(self, x, scale):
        x = self.conv1(x)
        return x


class ViTDepthDecoder(nn.Module):
    def __init__(self, scales=range(4), in_channels=[64, 128, 192, 256], num_output_channels=1, bins=False):
        super(ViTDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales
        self.win = 7
        self.convs = OrderedDict()
        self.with_auxiliary_head = False
        self.in_channels = in_channels
        self.bins = bins
        self.norm_cfg = dict(type='BN', requires_grad=True)  # {'type': 'BN', 'requires_grad': True}
        self.embed_dim = in_channels[-1] // 2
        self.decoder_cfg = dict(
            in_channels=self.in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=self.embed_dim,
            dropout_ratio=0.,
            norm_cfg=self.norm_cfg,
            align_corners=False,
            num_groups=in_channels[-1] // 2
        )
        self.PSP = PSP(**self.decoder_cfg)


        self.graph_dims = in_channels
        self.v_dims = [v_channel // 2 for v_channel in in_channels]     # [32, 64, 96, 128]
        self.num_ch_dec = np.array([self.v_dims[0]//2] + self.v_dims[:-1])    # [32//2, 32, 64, 96]

        self.graph3 = Graphformer(input_dim=self.in_channels[3], chn_dim=self.graph_dims[2], embed_dim=self.graph_dims[3], window_size=self.win,
                                  v_dim=self.v_dims[3], num_heads=32, up_sample=True, depth=1)
        self.graph2 = Graphformer(input_dim=self.in_channels[2], chn_dim=self.graph_dims[1], embed_dim=self.graph_dims[2], window_size=self.win,
                                  v_dim=self.v_dims[2], num_heads=16, up_sample=True, depth=1)
        self.graph1 = Graphformer(input_dim=self.in_channels[1], chn_dim=self.graph_dims[0], embed_dim=self.graph_dims[1], window_size=self.win,
                                  v_dim=self.v_dims[1], num_heads=8, up_sample=True, depth=1)
        self.graph0 = Graphformer(input_dim=self.in_channels[0], chn_dim=self.v_dims[0], embed_dim=self.graph_dims[0], window_size=self.win,
                                  v_dim=self.v_dims[0], num_heads=4, up_sample=True, depth=1)

        if bins:
            print("Using bins!!")
            for s in self.scales:
                self.convs[("bins", s)] = Conv3x3(self.num_ch_dec[s], bins)
        else:
            print("Without bins!!")
            for s in self.scales:
                self.convs[("dispconv", s)] = DispHead(input_dim=self.num_ch_dec[s],
                                                   num_output_channels=self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats):

        self.outputs = {}
        # decoder
        scale = 1
        # print(x[-1].size())
        ppm_out = self.PSP(feats)  # torch.Size([2, 256, 12, 40])
        e = []
        e.append(self.graph3(feats[3], ppm_out))  # torch.Size([2, 512, 12, 40])
        # e3 = nn.PixelShuffle(2)(e3)

        e.append(self.graph2(feats[2], e[-1]))  # torch.Size([2, 256, 24, 80])
        # e2 = nn.PixelShuffle(2)(e2)

        e.append(self.graph1(feats[1], e[-1]))
        # e1 = nn.PixelShuffle(2)(e1)

        e.append(self.graph0(feats[0], e[-1]))
        e = e[::-1]
        for i in self.scales:
            if self.bins:
                # print(e[i].shape)
                self.outputs[("bins", i)] = self.sigmoid(self.convs[("bins", i)](e[i]))
            else:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](e[i], scale))
        return self.outputs


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]

        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs


if __name__ == '__main__':
    device = "cuda:0"
    
    img = []
    # img.append(torch.rand([2, 64, 96, 320]).to(device))
    # img.append(torch.rand([2, 128, 48, 160]).to(device))
    # img.append(torch.rand([2, 192, 24, 80]).to(device))
    # img.append(torch.rand([2, 256, 12, 40]).to(device))
    
    img.append(torch.rand([2, 64, 160, 512]).to(device))
    img.append(torch.rand([2, 128, 80, 256]).to(device))
    img.append(torch.rand([2, 192, 40, 128]).to(device))
    img.append(torch.rand([2, 256, 20, 64]).to(device))
    scales = [0, 1, 2, 3]
    # restmp = DepthDecoderUnet(scales=scales, in_channels=[64, 128, 192, 256], bins=32).to(device)

    # img = []
    # img.append(torch.rand([2, 64, 96, 320]).to(device))
    # img.append(torch.rand([2, 64, 48, 160]).to(device))
    # img.append(torch.rand([2, 128, 24, 80]).to(device))
    # img.append(torch.rand([2, 256, 12, 40]).to(device))
    # img.append(torch.rand([2, 512, 6, 20]).to(device))

    scales = [0, 1, 2]
    restmp = ViTDepthDecoder(scales=scales, in_channels=[64, 128, 192, 256], bins=0).to(device)
    # num_ch_enc = np.array([64, 64, 128, 256, 512])
    # restmp = DepthDecoder(num_ch_enc, scales=scales).to(device)
    # FLOPs
    # flops = FlopCountAnalysis(restmp, img)
    # print("FLOPs: ", flops.total())

    # parameters
    print(parameter_count_table(restmp))

    output = restmp(img)
    # print(output[("disp", 0)].size(), output[("disp", 1)].size(), output[("disp", 2)].size())
    for i in range(3):
        print(output["disp", i].shape)

