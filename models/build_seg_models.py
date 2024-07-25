from models.fpn_seg_head import FPNHead
from models.mask_rcnn_seg_head import MaskRCNNSegmentationHead
from models.build_seg_backbone import repnext_m1, repnext_m2, repnext_m3, repnext_m4, repnext_m5

import torch
import torch.nn.functional as F
import torch.nn as nn

from timm.models import register_model


class SegRepNextModel(nn.Module):
    def __init__(self, backbone, num_classes=19, use_maskrcnn_head=False, **kwargs):
        super(SegRepNextModel, self).__init__()

        self.backbone = eval(backbone + '()')

        if use_maskrcnn_head == True:
            self.decode_head = MaskRCNNSegmentationHead(self.backbone.embed_dim,
                                                        256 if 'm1' in backbone or 'm2' in backbone else 768,
                                                        num_classes)
        else:
            self.decode_head = FPNHead(self.backbone.embed_dim, 128 if 'm1' in backbone or 'm2' in backbone else 768,
                                        num_classes)

    def forward(self, x):
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        return y


@register_model
def SegRepNext_m1(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    backbone = 'repnext_m1'
    model = SegRepNextModel(backbone, **kwargs)
    return model


@register_model
def SegRepNext_m2(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    backbone = 'repnext_m2'
    model = SegRepNextModel(backbone, **kwargs)
    return model


@register_model
def SegRepNext_m3(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    backbone = 'repnext_m3'
    model = SegRepNextModel(backbone, **kwargs)
    return model


@register_model
def SegRepNext_m4(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    backbone = 'repnext_m4'
    model = SegRepNextModel(backbone, **kwargs)
    return model


@register_model
def SegRepNext_m5(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    backbone = 'repnext_m5'
    model = SegRepNextModel(backbone, **kwargs)
    return model


# if __name__ == '__main__':
#     import torch
#     input_data = torch.randn(2, 3, 224, 224)
#     model = SegRepNext_m1()
#     y = model(input_data)
#     print(y.shape)