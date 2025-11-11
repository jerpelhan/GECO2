import torch
import torch.nn.functional as F
from torch import nn

from sam2.sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.sam2.modeling.sam.transformer import TwoWayTransformer


class MaskProcessor(nn.Module):
    def __init__(self, hidden_dim, image_size, reduction, **kwargs):
        super().__init__()

        self.sam_prompt_embed_dim = hidden_dim
        self.reduction = reduction
        self.sam_image_embedding_size = image_size // self.reduction
        self.image_size = image_size
        self.prompt_encoder_sam = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            **({}),
        )
        self.num_feature_levels = 3
        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # TODO change loading, this is ugly
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt',
            map_location="cpu"
        )['model']
        state_dict = {k.replace("mask_decoder.", "").replace("sam_", ""): v for k, v in
                      checkpoint.items() if "mask_decoder" in k}
        self.mask_decoder.load_state_dict(state_dict)
        state_dict = {k.replace("prompt_encoder.", "").replace("sam_", ""): v for k, v in
                      checkpoint.items() if "prompt_encoder" in k}
        self.prompt_encoder_sam.load_state_dict(state_dict)
        state_dict = {k: v for k, v in checkpoint.items() if "no_mem_embed" in k}
        self.load_state_dict(state_dict, strict=False)

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def forward_feats(self, feats: torch.Tensor):
        """Get the image feature on the input batch."""
        # precompute projected level 0 and level 1 features in SAM decoder
        # to avoid running it again on every SAM click
        feats["backbone_fpn"][0] = self.mask_decoder.conv_s0(
            feats["backbone_fpn"][0]
        )
        feats["backbone_fpn"][1] = self.mask_decoder.conv_s1(
            feats["backbone_fpn"][1]
        )
        _, vision_feats, _, _ = self._prepare_backbone_features(feats)

        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed
        bs = vision_feats[0].shape[1]
        feats = [
                    feat.permute(1, 2, 0).view(bs, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
                ][::-1]
        return feats

    def forward(self, features_orig, outputs):


        batch_masks = []
        batch_iou = []
        batch_bboxes = []
        for img_idx in range(len(outputs)):
            only_score = False
            if len((outputs[img_idx]['pred_boxes'][0])) > 800:
                only_score = True
                batch_masks.append([])
                batch_bboxes.append(outputs[img_idx]['pred_boxes'].squeeze()*self.image_size)
                batch_iou.append(outputs[img_idx]['box_v'])
                continue
            features = {
                'vision_features':  features_orig['vision_features'][img_idx].unsqueeze(0),
                'vision_pos_enc': [x[img_idx].unsqueeze(0) for x in features_orig['vision_pos_enc']],
                'backbone_fpn': [x[img_idx].unsqueeze(0) for x in features_orig['backbone_fpn']],
            }
            features = self.forward_feats(features)
            step = 50
            low_res_masks = []
            iou_predictions = []
            corrected_bboxes_ = []
            masks_ = []
            for box_i in range(step, len(outputs[img_idx]['pred_boxes'][0]) + step, step):
                box = outputs[img_idx]['pred_boxes'][0][(box_i - step):box_i] * self.image_size
                box_coords = box.reshape(-1, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=box.device)
                box_labels = box_labels.repeat(box.size(0), 1)
                sparse_embeddings, dense_embeddings = self.prompt_encoder_sam(
                    points= (box_coords, box_labels),
                    boxes=None,
                    masks=None,
                )

                low_res_masks_, iou_predictions_, _, _ = self.mask_decoder(
                    image_embeddings=features[-1],
                    image_pe=self.prompt_encoder_sam.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=True,
                    high_res_features=features[:-1],
                )
                low_res_masks.append(low_res_masks_)
                iou_predictions.append(iou_predictions_[:, 2])

                masks = F.interpolate(low_res_masks_, (self.image_size, self.image_size),
                mode = "bilinear",
                align_corners = False)
                masks = masks > 0

                corrected_bboxes = torch.zeros((masks.shape[0], 4), dtype=torch.float)
                masks = masks[:, 2]
                for index, mask in enumerate(masks):
                    y, x = torch.where(mask != 0)
                    if y.shape[0] > 0 and x.shape[0] > 0:
                        corrected_bboxes[index, 0] = torch.min(x)
                        corrected_bboxes[index, 1] = torch.min(y)
                        corrected_bboxes[index, 2] = torch.max(x)
                        corrected_bboxes[index, 3] = torch.max(y)
                masks_.append(masks)
                corrected_bboxes_.append(corrected_bboxes)
            if only_score:
                batch_masks.append([])  
                batch_bboxes.append(outputs[img_idx]['pred_boxes'].squeeze()*self.image_size)
                batch_iou.append(torch.cat(iou_predictions).unsqueeze(0))

            if len(corrected_bboxes_) > 0:
                batch_masks.append(masks_) 
                batch_bboxes.append(torch.cat(corrected_bboxes_))
                batch_iou.append(torch.cat(iou_predictions).unsqueeze(0))
            else:
                batch_masks.append([])
                batch_bboxes.append(torch.tensor([]).to(features[0].device))
                batch_iou.append(torch.tensor([]).to(features[0].device))


        batch_masks = [torch.cat(masks) if len(masks)>0 else torch.zeros((1,1024,1024)) for masks in batch_masks]
        return batch_masks, batch_iou, batch_bboxes