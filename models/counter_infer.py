import numpy as np
import skimage
import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from utils.box_ops import boxes_with_scores
from .query_generator import C_base
from .sam_mask import MaskProcessor


class CNT(nn.Module):

    def __init__(
            self,
            image_size: int,
            num_objects: int,
            emb_dim: int,
            kernel_dim: int,
            reduction: int,
            zero_shot: bool,
            training: bool=False,
    ):

        super(CNT, self).__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot

        self.class_embed = nn.Sequential(nn.Linear(emb_dim, 1), nn.LeakyReLU())
        self.bbox_embed = MLP(emb_dim, emb_dim, 4, 3)

        self.adapt_features = C_base(
            transformer_dim=self.emb_dim,
            num_prototype_attn_steps=3,
            num_image_attn_steps=2,
        )
        from .prompt_encoder import PromptEncoder
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.emb_dim,
            image_embedding_size=(
                self.image_size // self.reduction,
                self.image_size // self.reduction,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        config_name = '../configs/sam2_hiera_base_plus.yaml'
        cfg = compose(config_name=config_name)
        OmegaConf.resolve(cfg)
        self.backbone = instantiate(cfg.backbone, _recursive_=True)
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://dl.fbaipublicfiles.com/segment_anything_2/072824/' + config_name.split('/')[-1].replace('.yaml',
                                                                                                             '.pt'),
            map_location="cpu"
        )['model']
        state_dict = {k.replace("image_encoder.", ""): v for k, v in checkpoint.items()}
        self.backbone.load_state_dict(state_dict, strict=False)

        self.shape_or_objectness = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1 ** 2 * emb_dim)
        )
        self.validate = not training
        if self.validate:
            self.sam_mask = MaskProcessor(self.emb_dim, self.image_size, reduction)


    def forward(self, x, bboxes):
        self.num_objects = bboxes.size(1) 
        with torch.no_grad():
            feats = self.backbone(x)
        src = feats['vision_features']
        bs, c, w, h = src.shape
        self.reduction = 1024 / w

        bboxes_roi = torch.cat([
            torch.arange(
                bs, requires_grad=False
            ).to(bboxes.device).repeat_interleave(self.num_objects).reshape(-1, 1),
            bboxes.flatten(0, 1),
        ], dim=1)
        self.kernel_dim = 1

        # # NORMAL
        exemplars = roi_align(
            src,
            boxes=bboxes_roi, output_size=self.kernel_dim,
            spatial_scale=1.0 / self.reduction, aligned=True
        ).permute(0, 2, 3, 1).reshape(bs, self.num_objects * self.kernel_dim ** 2, self.emb_dim)

        l1 = feats['backbone_fpn'][0]
        l2 = feats['backbone_fpn'][1]
        r1 = 1.0 / self.reduction * 2 * 2
        exemplars_l1 = roi_align(
            l1,
            boxes=bboxes_roi, output_size=self.kernel_dim,
            spatial_scale=1.0 / self.reduction * 2 * 2, aligned=True
        ).permute(0, 2, 3, 1).reshape(bs, self.num_objects * self.kernel_dim ** 2, self.emb_dim)

        exemplars_l2 = roi_align(
            l2,
            boxes=bboxes_roi, output_size=self.kernel_dim,
            spatial_scale=1.0 / self.reduction * 2, aligned=True
        ).permute(0, 2, 3, 1).reshape(bs, self.num_objects * self.kernel_dim ** 2, self.emb_dim)

        box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device)
        box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]
        box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]

        # Encode shape
        shape = self.shape_or_objectness(box_hw).reshape(
            bs, -1, self.emb_dim
        )
        prototype_embeddings = torch.cat([exemplars, shape], dim=1)
        prototype_embeddings_l1 = torch.cat([exemplars_l1, shape], dim=1)
        prototype_embeddings_l2 = torch.cat([exemplars_l2, shape], dim=1)
        hq_prototype_embeddings = [prototype_embeddings_l1, prototype_embeddings_l2]

        # adapt image feature with prototypes
        adapted_f, adapted_f_aux = self.adapt_features(
            image_embeddings=src,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            prototype_embeddings=prototype_embeddings,
            hq_features=feats['backbone_fpn'],
            hq_prototypes=hq_prototype_embeddings,
            hq_pos=feats['vision_pos_enc'],
        )
        # Predict class [fg, bg] and l,r,t,b
        bs, c, w, h = adapted_f.shape
        adapted_f = adapted_f.view(bs, self.emb_dim, -1).permute(0, 2, 1)
        centerness = self.class_embed(adapted_f).view(bs, w, h, 1).permute(0, 3, 1, 2)
        outputs_coord = self.bbox_embed(adapted_f).sigmoid().view(bs, w, h, 4).permute(0, 3, 1, 2)
        outputs, ref_points = boxes_with_scores(centerness, outputs_coord,sort=False, validate=True)

        if not self.validate:
            adapted_f_aux = adapted_f_aux.view(bs, self.emb_dim, -1).permute(0, 2, 1)
            centerness_aux = self.class_embed_aux(adapted_f_aux).view(bs, w, h, 1).permute(0, 3, 1, 2)
            outputs_coord_aux = self.bbox_embed_aux(adapted_f_aux).sigmoid().view(bs, w, h, 4).permute(0, 3, 1, 2)
            outputs_aux, ref_points_aux = boxes_with_scores(centerness_aux, outputs_coord_aux, sort=False, validate=self.validate)
            for i in range(len(outputs)):
                outputs[i]["scores"] = outputs[i]["box_v"]
            return outputs, ref_points, centerness, outputs_coord, (outputs_aux, ref_points_aux, centerness_aux, outputs_coord_aux)

        else:
            # mask processing
            masks, ious, corrected_bboxes = self.sam_mask(feats, outputs)
            for i in range(len(outputs)):
                outputs[i]["scores"] = ious[i]
                outputs[i]["pred_boxes"] = corrected_bboxes[i].to(outputs[i]["pred_boxes"].device).unsqueeze(0) /x.shape[ -1]

            return outputs, ref_points, centerness, outputs_coord, masks


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(args):
    assert args.reduction in [4, 8, 16]

    return CNT(
        image_size=args.image_size,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        kernel_dim=args.kernel_dim,
        reduction=args.reduction,
    )