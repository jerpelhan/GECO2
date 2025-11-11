import torch
import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCELoss

from utils import box_ops



class ObjectNormalizedL2Loss(nn.Module):

    def __init__(self):
        super(ObjectNormalizedL2Loss, self).__init__()

    def forward(self, output, dmap, num_objects):
        return ((output - dmap) ** 2).sum() / num_objects


class Detection_criterion(nn.Module):

    def __init__(
            self, sizes, iou_loss_type, center_sample, fpn_strides, pos_radius, aux=False
    ):
        super().__init__()

        self.sizes = sizes
        self.box_loss = IOULoss(iou_loss_type)
        self.aux = aux
        self.center_sample = center_sample
        self.strides = fpn_strides
        self.radius = pos_radius

    def prepare_target(self, points, targets):
        ex_size_of_interest = []

        for i, point_per_level in enumerate(points):
            size_of_interest_per_level = point_per_level.new_tensor(self.sizes[i])
            ex_size_of_interest.append(
                size_of_interest_per_level[None].expand(len(point_per_level), -1)
            )

        ex_size_of_interest = torch.cat(ex_size_of_interest, 0)
        n_point_per_level = [len(point_per_level) for point_per_level in points]
        point_all = torch.cat(points, dim=0)
        label, box_target = self.compute_target_for_location(
            point_all, targets, ex_size_of_interest, n_point_per_level
        )

        for i in range(len(label)):
            label[i] = torch.split(label[i], n_point_per_level, 0)
            box_target[i] = torch.split(box_target[i], n_point_per_level, 0)

        label_level_first = []
        box_target_level_first = []

        for level in range(len(points)):
            label_level_first.append(
                torch.cat([label_per_img[level] for label_per_img in label], 0).to(points[0].device)
            )
            box_target_level_first.append(
                torch.cat(
                    [box_target_per_img[level] for box_target_per_img in box_target], 0
                )
            )

        return label_level_first, box_target_level_first

    def get_sample_region(self, gt, strides, n_point_per_level, xs, ys, radius=1):
        n_gt = gt.shape[0]
        n_loc = len(xs)
        gt = gt[None].expand(n_loc, n_gt, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        # y_stride = torch.min((gt[..., 3] - gt[..., 1]) / 2)/2
        # x_stride = torch.min((gt[..., 2] - gt[..., 0]) / 2)/2

        if center_x[..., 0].sum() == 0:
            return xs.new_zeros(xs.shape, dtype=torch.uint8)

        begin = 0

        center_gt = gt.new_zeros(gt.shape)

        for level, n_p in enumerate(n_point_per_level):
            end = begin + n_p
            stride = strides[level] * radius

            x_min = center_x[begin:end] - stride
            y_min = center_y[begin:end] - stride
            x_max = center_x[begin:end] + stride
            y_max = center_y[begin:end] + stride

            center_gt[begin:end, :, 0] = torch.where(
                x_min > gt[begin:end, :, 0], x_min, gt[begin:end, :, 0]
            )
            center_gt[begin:end, :, 1] = torch.where(
                y_min > gt[begin:end, :, 1], y_min, gt[begin:end, :, 1]
            )
            center_gt[begin:end, :, 2] = torch.where(
                x_max > gt[begin:end, :, 2], gt[begin:end, :, 2], x_max
            )
            center_gt[begin:end, :, 3] = torch.where(
                y_max > gt[begin:end, :, 3], gt[begin:end, :, 3], y_max
            )

            begin = end

        left = xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - xs[:, None]
        top = ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - ys[:, None]

        center_bbox = torch.stack((left, top, right, bottom), -1)
        is_in_boxes = center_bbox.min(-1)[0] > 0

        return is_in_boxes

    def compute_target_for_location(
            self, locations, targets, sizes_of_interest, n_point_per_level
    ):
        labels = []
        box_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        for i in range(len(targets)):
            targets_per_img = targets[i]
            targets_per_img=targets_per_img.clip(remove_empty=True)
            assert targets_per_img.mode == 'xyxy'
            targets_per_img = targets_per_img[:50]
            bboxes = targets_per_img.box

            labels_per_img = torch.tensor([1]*len(bboxes)).to(locations.device)
            area = targets_per_img.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            box_targets_per_img = torch.stack([l, t, r, b], 2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, n_point_per_level, xs, ys, radius=self.radius
                )

            else:
                is_in_boxes = box_targets_per_img.min(2)[0] > 0

            max_box_targets_per_img = box_targets_per_img.max(2)[0]

            is_cared_in_level = (
                                        max_box_targets_per_img >= sizes_of_interest[:, [0]]
                                ) & (max_box_targets_per_img <= sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_level == 0] = INF

            locations_to_min_area, locations_to_gt_id = locations_to_gt_area.min(1)

            box_targets_per_img = box_targets_per_img[
                range(len(locations)), locations_to_gt_id
            ]

            labels_per_img = labels_per_img.to(locations_to_gt_id.device)[locations_to_gt_id]
            labels_per_img[locations_to_min_area == INF] = 0

            labels.append(labels_per_img)
            box_targets.append(box_targets_per_img)

        return labels, box_targets

    def compute_centerness_targets(self, box_targets):
        left_right = box_targets[:, [0, 2]]
        top_bottom = box_targets[:, [1, 3]]
        centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * (
                top_bottom.min(-1)[0] / top_bottom.max(-1)[0]
        )

        return torch.sqrt(centerness)

    def forward(self, locations, box_pred, targets):
        batch = box_pred[0].shape[0]
        labels, box_targets = self.prepare_target(locations, targets)
        box_flat = []

        labels_flat = []
        box_targets_flat = []

        for i in range(len(labels)):
            box_flat.append(box_pred.permute(0, 2, 3, 1).reshape(-1, 4))

            labels_flat.append(labels[i].reshape(-1))
            box_targets_flat.append(box_targets[i].reshape(-1, 4))
        box_flat = torch.cat(box_flat, 0)
        labels_flat = torch.cat(labels_flat, 0)
        box_targets_flat = torch.cat(box_targets_flat, 0)
        pos_id = torch.nonzero(labels_flat > 0).squeeze(1)
        box_flat = box_flat[pos_id]
        box_targets_flat = box_targets_flat[pos_id]

        if pos_id.numel() > 0:
            center_targets = self.compute_centerness_targets(box_targets_flat)
            box_loss = self.box_loss(box_flat, box_targets_flat, center_targets)
        else:
            box_loss = box_flat.sum()

        return box_loss


INF = 100000000

class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super().__init__()

        self.loc_loss_type = loc_loss_type

    def forward(self, out, target, weight=None):

        pred_left, pred_top, pred_right, pred_bottom = out.unbind(1)

        target_left, target_top, target_right, target_bottom = target.unbind(1)

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1) / (area_union + 1)

        if self.loc_loss_type == 'iou':
            loss = -torch.log(ious)

        elif self.loc_loss_type == 'giou':
            g_w_intersect = torch.max(pred_left, target_left) + torch.max(
                pred_right, target_right
            )
            g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
                pred_top, target_top
            )
            g_intersect = g_w_intersect * g_h_intersect + 1e-7
            gious = ious - (g_intersect - area_union) / g_intersect

            loss = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (loss * weight).sum() / weight.sum()

        else:
            return loss.mean()


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.cross_entropy = BCELoss()


    def loss_boxes(self, outputs, targets, indices, num_boxes, centerness, centerness_gt,mask):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            (src_boxes),
            (target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def ce_loss(self, outputs, targets, indices, num_boxes, centerness, centerness_gt, mask):

        l2 = ((centerness[mask > 0] - centerness_gt[mask > 0]) ** 2)
        losses = {}
        losses['loss_ce'] = l2.sum() / num_boxes
        return losses



    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, centerness, centerness_gt, mask, **kwargs):
        loss_map = {
            'bboxes': self.loss_boxes,
            'ce': self.ce_loss
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes,centerness, centerness_gt, mask, **kwargs)
    
    # def generate_centerness_gt(self, indices, FN_idx, FP_idx, outputs, targets, centerness, ref_points):
    #     # TP_bboxes = outputs['pred_boxes'][0][indices[0][0]] * centerness.shape[1]
    #     # FP_bboxes = outputs['pred_boxes'][0][FP_idx] * centerness.shape[1]
    #     FN_bboxes = targets[0]['boxes'][FN_idx] * centerness.shape[1]
    #     centerness_gt = torch.zeros_like(centerness)
    #     mask = torch.ones_like(centerness)
    #     # FP -> Non-matched PRED bboxes get 0 in the reference point, so 1 in mask
    #     FP_locs = ref_points.permute(1, 0)[FP_idx]
    #     mask[0][FP_locs[:, 0], FP_locs[:, 1]] = 1
    #     bounding_boxes = (targets[0]['boxes'] * centerness.shape[1]).type(torch.int64)
    #     for box in bounding_boxes:
    #         x_min, y_min, x_max, y_max = box
    #         mask[:, y_min:y_max, x_min:x_max] = 0
    #     # FN -> Non-matched GT bboxes get 1 in center of bbox
    #     if len(FN_bboxes) > 0:
    #         FN_y_loc = torch.clamp(((FN_bboxes[:, 3] + FN_bboxes[:, 1]) / 2).int(), min=0, max=centerness.shape[1]-1)
    #         FN_x_loc = torch.clamp(((FN_bboxes[:, 2] + FN_bboxes[:, 0]) / 2).int(), min=0, max=centerness.shape[1]-1)
    #         centerness_gt[0][FN_y_loc, FN_x_loc] = 1
    #         mask[0][FN_y_loc, FN_x_loc] = 1
    #     # TP -> Matched PRED bboxes get 1 in the reference point
    #     TP_locs = ref_points.permute(1, 0)[indices[0][0]]
    #     centerness_gt[0][TP_locs[:, 0], TP_locs[:, 1]] = 1
    #     mask[0][TP_locs[:, 0], TP_locs[:, 1]] = 1
    #     return centerness_gt, mask
    
    def generate_centerness_gt(self, indices, FN_idx, FP_idx, outputs, targets, centerness, ref_points):
        FN_bboxes = targets[0]['boxes'][FN_idx] * centerness.shape[1]

        centerness_gt = torch.zeros_like(centerness)
        mask = torch.zeros_like(centerness)

        # FP -> Non-matched PRED bboxes get 0 in the reference point, so 1 in mask
        FP_locs = ref_points.permute(1, 0)[FP_idx]
        mask[0][FP_locs[:, 0], FP_locs[:, 1]] = 1

        # bounding_boxes = (targets[0]['boxes'] * centerness.shape[1]).type(torch.int64)
        # for box in bounding_boxes:
        #     x_min, y_min, x_max, y_max = box
        #     mask[:, y_min:y_max, x_min:x_max] = 0

        # FN -> Non-matched GT bboxes get 1 in center of bbox
        if len(FN_bboxes) > 0:
            FN_y_loc = torch.clamp(((FN_bboxes[:, 3] + FN_bboxes[:, 1]) / 2).int(), min=0, max=centerness.shape[1]-1)
            FN_x_loc = torch.clamp(((FN_bboxes[:, 2] + FN_bboxes[:, 0]) / 2).int(), min=0, max=centerness.shape[1]-1)

            centerness_gt[0][FN_y_loc, FN_x_loc] = 1
            mask[0][FN_y_loc, FN_x_loc] = 1

        # TP -> Matched PRED bboxes get 1 in the reference point
        TP_locs = ref_points.permute(1, 0)[indices[0][0]]
        centerness_gt[0][TP_locs[:, 0], TP_locs[:, 1]] = 1
        mask[0][TP_locs[:, 0], TP_locs[:, 1]] = 1

        if centerness_gt.sum() < targets[0]['boxes'].shape[0]:
            centerness_gt = torch.zeros_like(centerness)
            FN_bboxes = targets[0]['boxes'] * centerness.shape[1]
            
            FN_y_loc = torch.clamp(((FN_bboxes[:, 3] + FN_bboxes[:, 1]) / 2).int(), min=0, max=centerness.shape[1]-1)
            FN_x_loc = torch.clamp(((FN_bboxes[:, 2] + FN_bboxes[:, 0]) / 2).int(), min=0, max=centerness.shape[1]-1)

            centerness_gt[0][FN_y_loc, FN_x_loc] = 1
            mask = torch.ones_like(centerness)


        return centerness_gt, mask

    # def generate_centerness_gt(self, indices, FN_idx, FP_idx, outputs, targets, centerness, ref_points):
    #     # TP_bboxes = outputs['pred_boxes'][0][indices[0][0]] * centerness.shape[1]
    #     # FP_bboxes = outputs['pred_boxes'][0][FP_idx] * centerness.shape[1]
    #     FN_bboxes = targets[0]['boxes'][FN_idx] * centerness.shape[1]

    #     centerness_gt = torch.zeros_like(centerness)
    #     mask = torch.zeros_like(centerness)

    #     # FN -> Non-matched GT bboxes get 1 in center of bbox
    #     if len(FN_bboxes) > 0:
    #         FN_y_loc = ((FN_bboxes[:, 3] + FN_bboxes[:, 1]) / 2 ).int()
    #         FN_x_loc = ((FN_bboxes[:, 2] + FN_bboxes[:, 0]) / 2 ).int()

    #         centerness_gt[0][FN_y_loc, FN_x_loc] = 1
    #         # mask[0][FN_y_loc, FN_x_loc] = 1

    #     # FP -> Non-matched PRED bboxes get 0 in the reference point, so 1 in mask
    #     FP_locs = ref_points.permute(1, 0)[FP_idx]
    #     # mask[0][FP_locs[:, 0], FP_locs[:, 1]] = 1

    #     # TP -> Matched PRED bboxes get 1 in the reference point
    #     print(indices[0][0])
    #     TP_locs = ref_points.permute(1, 0)[indices[0][0]]
    #     centerness_gt[0][TP_locs[:, 0], TP_locs[:, 1]] = 1
    #     # mask[0][TP_locs[:, 0], TP_locs[:, 1]] = 1

    #     return centerness_gt, mask

    # # from matplotlib import pyplot as plt
    # # plt.clf()
    # # plt.imshow(centerness_gt.cpu()[0], cmap='jet')
    # # plt.imshow(mask.cpu()[0], cmap='jet', alpha=0.3)
    # # for i in range(TP_bboxes.shape[0]):
    # #     box = TP_bboxes[i].cpu()
    # #     plt.plot([box[0], box[0], box[2], box[2], box[0]],
    # #              [box[1], box[3], box[3], box[1], box[1]], color='g')
    # #
    # # for i in range(FP_bboxes.shape[0]):
    # #     box = FP_bboxes[i].cpu()
    # #     plt.plot([box[0], box[0], box[2], box[2], box[0]],
    # #              [box[1], box[3], box[3], box[1], box[1]], color='orange')
    # #
    # # for i in range(FN_bboxes.shape[0]):
    # #     box = FN_bboxes[i].cpu()
    # #     plt.plot([box[0], box[0], box[2], box[2], box[0]],
    # #              [box[1], box[3], box[3], box[1], box[1]], color='red')
    # # plt.savefig("T")

    def forward(self, outputs, targets, centerness, ref_points):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, FN_idx, FP_idx = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes = torch.clamp(num_boxes, min=1).item()

        centerness_gt, mask = self.generate_centerness_gt(indices, FN_idx, FP_idx, outputs, targets, centerness, ref_points)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, centerness, centerness_gt, mask, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


