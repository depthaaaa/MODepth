from __future__ import absolute_import, division, print_function
# from visualizer import get_local
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
sys.path.append("../")
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.utils import Registry, build_from_cfg
from .pix_transformer_decoder import PixelTransformerDecoder, PixelTransformerDecoderLayer, SinePositionalEncoding


class AdaPlanes(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, norm='linear', min_val=0.001, max_val=10) -> None:
        super(AdaPlanes, self).__init__()
        self.norm = norm

        self.positional_encoding = SinePositionalEncoding(num_feats=in_channels//2, normalize=True)
        self.level_embed = nn.Embedding(3, in_channels)
        
        transformerlayers=dict(
            attn_cfgs=dict(
                type='MultiheadAttention',
                embed_dims=in_channels,
                num_heads=8,
                dropout=0.0),
            ffn_cfgs=dict(
                feedforward_channels=256,
                dropout=0.0),
            operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm'))
        self.transformer_decoder = PixelTransformerDecoder(hidden_dim=in_channels,
                                                           return_intermediate=True,
                                                           num_layers=3,
                                                           num_feature_levels=3,
                                                           operation='//',
                                                           transformerlayers=transformerlayers
                                                           )
        
        self.query_feat = nn.Embedding(embedding_dim, in_channels)
        # learnable query p.e.
        self.query_embed = nn.Embedding(embedding_dim, in_channels)

        self.min_val = min_val
        self.max_val = max_val

    # @get_local('x0')
    def forward(self, mlvl_feats):
        feat = []
        for i in range(len(mlvl_feats)):
            feat.append(mlvl_feats[("bins", i)])
        src = []
        pos = []
        size_list = []
        per_pixel_feat = feat[0]

        feat = feat[::-1]

        batch_size = feat[0].size(0)
        input_img_h, input_img_w = feat[0].size(2), feat[0].size(3)
        img_masks = feat[0].new_zeros(
            (batch_size, input_img_h, input_img_w))

        feat = feat[:3]

        mlvl_masks = []
        for idx, feat in enumerate(feat):
            size_list.append(feat.shape[-2:])
            
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                            size=feat.shape[-2:]).to(torch.bool).squeeze(0))

            # print(self.positional_encoding(mlvl_masks[-1]).flatten(2).shape)
            # print(self.level_embed.weight[idx][None, :, None].shape)
            # print(((mlvl_masks[-1]).flatten(2) + self.level_embed.weight[idx][None, :, None]).shape)
            pos.append(
                self.positional_encoding(mlvl_masks[-1]).flatten(2) + self.level_embed.weight[idx][None, :, None])
            src.append(feat.flatten(2))

            # 4, 256, HW -> HW, N, C
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        
        multi_scale_infos = {'src':src, 'pos':pos}
        bs = per_pixel_feat.shape[0]
        
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        query_pe = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # print(query_feat.shape, query_pe.shape)
        predictions_bins, predictions_logits = \
             self.transformer_decoder(multi_scale_infos, query_feat, query_pe, per_pixel_feat)
        # print(len(predictions_logits))

        # NOTE: depth estimation module
        self.norm = 'softmax'
        self.outputs = {}

        for i, (item_bin, pred_logit) in enumerate(zip(predictions_bins, predictions_logits)):
            # torch.Size([2, 64, 1]) torch.Size([2, 64, 192, 640])
            y = item_bin.squeeze(dim=2)

            if self.norm == 'linear':
                y = torch.relu(y)
                eps = 0.1
                y = y + eps
            elif self.norm == 'softmax':
                y = torch.softmax(y, dim=1)
            else:
                y = torch.sigmoid(y)
            y = y / y.sum(dim=1, keepdim=True)  # normalization
            
            # out = self.convert_to_prob(energy_maps)
            bin_widths = (self.max_val - self.min_val) * y 
            bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
            bin_edges = torch.cumsum(bin_widths, dim=1)     #  Cumulative depth

            centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

            n, dout = centers.size()

            centers = centers.view(n, dout, 1, 1)

            pred_logit = pred_logit.softmax(dim=1)
            
            pred_depth = torch.sum(pred_logit * centers, dim=1, keepdim=True)

            self.outputs["disp", (len(mlvl_feats) - 1 - i)] = F.interpolate(pred_depth,
                                                                            size=[int(x // (2 ** (len(mlvl_feats) - 1 - i))) for x in pred_depth.size()[-2:]],
                                                                            mode="bilinear", align_corners=False)


        return self.outputs
   
if __name__ == '__main__':
    x0={}
    # x0[("bins", 0)] = torch.rand([2, 64, 192, 640])
    # x0[("bins", 1)] = (torch.rand([2, 64, 96, 320]))
    # x0[("bins", 2)] = (torch.rand([2, 64, 48, 160]))
    x0[("bins", 0)] = torch.rand([2, 64, 320, 1024])
    x0[("bins", 1)] = (torch.rand([2, 64, 160, 512]))
    x0[("bins", 2)] = (torch.rand([2, 64, 80, 256]))
    # x0[("bins", 3)] = (torch.rand([2, 32, 24, 80]))
    depth = AdaPlanes(in_channels=64, embedding_dim=64, min_val=0.001, 
                          max_val=80.0)
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    print(parameter_count_table(depth))
    outputs = depth(x0)
    for i in range(3):
        print(i, outputs["disp", i].shape)    # torch.Size([2, 1, 96, 320])

