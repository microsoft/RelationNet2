# This file is added for back-compatibility. Thus, downstream codebase
# could still use and import mmdet.ops.

# yapf: disable
try:
    from mmcv.ops import (ContextBlock, ConvWS2d, conv_ws_2d,
                          GeneralizedAttention)
except ImportError:
    from mmcv.cnn import (ContextBlock, ConvWS2d, conv_ws_2d,
                          GeneralizedAttention)
from mmcv.cnn import NonLocal2d as NonLocal2D
from mmcv.cnn import build_plugin_layer
from mmcv.ops import Conv2d, ConvTranspose2d, CornerPool
from mmcv.ops import DeformConv2d as DeformConv
from mmcv.ops import DeformConv2dPack as DeformConvPack
from mmcv.ops import DeformRoIPool as DeformRoIPooling
from mmcv.ops import DeformRoIPoolPack as DeformRoIPoolingPack
from mmcv.ops import Linear, MaskedConv2d, MaxPool2d
from mmcv.ops import ModulatedDeformConv2d as ModulatedDeformConv
from mmcv.ops import ModulatedDeformConv2dPack as ModulatedDeformConvPack
from mmcv.ops import \
    ModulatedDeformRoIPoolPack as ModulatedDeformRoIPoolingPack
from mmcv.ops import (RoIAlign, RoIPool, SAConv2d, SigmoidFocalLoss,
                      SimpleRoIAlign, batched_nms)
from mmcv.ops import deform_conv2d as deform_conv
from mmcv.ops import deform_roi_pool as deform_roi_pooling
from mmcv.ops import get_compiler_version, get_compiling_cuda_version
from mmcv.ops import modulated_deform_conv2d as modulated_deform_conv
from mmcv.ops import (nms, nms_match, point_sample,
                      rel_roi_point_to_rel_img_point, roi_align, roi_pool,
                      sigmoid_focal_loss, soft_nms)
from .corner_pool import TLPool,BRPool
# yapf: enable

__all__ = [
    'nms', 'soft_nms', 'RoIAlign', 'roi_align', 'RoIPool', 'roi_pool',
    'DeformConv', 'DeformConvPack', 'DeformRoIPooling', 'DeformRoIPoolingPack',
    'ModulatedDeformRoIPoolingPack', 'ModulatedDeformConv',
    'ModulatedDeformConvPack', 'deform_conv', 'modulated_deform_conv',
    'deform_roi_pooling', 'SigmoidFocalLoss', 'sigmoid_focal_loss',
    'MaskedConv2d', 'ContextBlock', 'GeneralizedAttention', 'NonLocal2D',
    'get_compiler_version', 'get_compiling_cuda_version', 'ConvWS2d',
    'conv_ws_2d', 'build_plugin_layer', 'batched_nms', 'Conv2d',
    'ConvTranspose2d', 'MaxPool2d', 'Linear', 'nms_match', 'CornerPool',
    'point_sample', 'rel_roi_point_to_rel_img_point', 'SimpleRoIAlign',
    'SAConv2d','TLPool','BRPool'
]
