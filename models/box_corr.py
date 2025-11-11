import numpy as np
import skimage
import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from torchvision.transforms import Resize

from .query_generator import C_base
from .sam_mask import MaskProcessor


class Box_correction(nn.Module):

    def __init__(
            self,
            reduction,
            image_size,
            emb_dim,
    ):

        super(Box_correction, self).__init__()

        self.sam_mask = MaskProcessor(emb_dim, image_size, reduction)
        self.sam_corr = True

    def forward(self, feats, outputs, x):

        # mask processing
        masks, ious, corrected_bboxes = self.sam_mask(feats, outputs)
        for i in range(len(outputs)):
            outputs[i]["scores"] = ious[i]
            outputs[i]["pred_boxes"] = corrected_bboxes[i].to(outputs[i]["pred_boxes"].device).unsqueeze(0) /x.shape[-1]
        return outputs, masks
