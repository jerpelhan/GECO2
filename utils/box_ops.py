
import torch
from torchvision import ops
from torchvision.ops.boxes import box_area
import torch.nn.functional as F


def boxes_with_scores(density_map, tlrb, sort=False, validate=False):
    B, C, _, _ = density_map.shape  # B, 1, H, W

    # maxpool instead of scikit local peak
    pooled = F.max_pool2d(density_map, 3, 1, 1)
    # medians over batch
    if validate:
        batch_thresh = torch.max(density_map.reshape(B, -1), dim=-1).values.view(B, C, 1, 1) / 8
    else:
        batch_thresh = torch.median(density_map.reshape(B, -1), dim=-1).values.view(B, C, 1, 1)

    # binary mask of selected boxes
    mask = (pooled == density_map) & (density_map > batch_thresh)

    # need this for loop to have the same output structure
    # can be vectorized otherwise
    out_batch = []
    ref_points_batch = []
    for i in range(B):
        # select the masked density maps and box offsets
        bbox_scores = density_map[i, mask[i]]
        ref_points = mask[i].nonzero()[:, -2:]

        # normalize center locations
        bbox_centers = ref_points / torch.tensor(mask.shape[2:], device=mask.device)

        # select masked box offsets, permute to keep channels last
        tlrb_ = tlrb[i].permute(1, 2, 0)
        bbox_offsets = tlrb_[mask[i].permute(1, 2, 0).expand_as(tlrb_)].reshape(-1, 4)

        # vectorised calculation of the boxes = [ref_points_transposed[1] / ...] in original
        sign = torch.tensor([-1, -1, 1, 1], device=mask.device)
        bbox_xyxy = bbox_centers.flip(-1).repeat(1, 2) + sign * bbox_offsets

        # sort by bbox score if needed -- this matches the original
        if sort:
            perm = torch.argsort(bbox_scores, descending=True)
            bbox_scores = bbox_scores[perm]
            bbox_xyxy = bbox_xyxy[perm]
            ref_points = ref_points[perm]

        out_batch.append({
            "pred_boxes": bbox_xyxy.unsqueeze(0),
            "box_v": bbox_scores.unsqueeze(0)
        })
        ref_points_batch.append(ref_points.T)

    return out_batch, ref_points_batch

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter + 1e-16  # [N,M]

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1] + 1e-16  # [N,M]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)



import numpy as np
class BoxList:
    def __init__(self, box, image_size, mode='xyxy'):
        device = box.device if hasattr(box, 'device') else 'cpu'
        if torch.is_tensor(box):
            box = torch.as_tensor(box, dtype=torch.float32, device=device)
        else:
            box = torch.as_tensor(np.array(box), dtype=torch.float32, device=device)

        self.box = box
        self.size = image_size
        self.mode = mode

        self.fields = {}

    def convert(self, mode):
        if mode == self.mode:
            return self

        x_min, y_min, x_max, y_max = self.split_to_xyxy()

        if mode == 'xyxy':
            box = torch.cat([x_min, y_min, x_max, y_max], -1)
            box = BoxList(box, self.size, mode=mode)

        elif mode == 'xywh':
            remove = 1
            box = torch.cat(
                [x_min, y_min, x_max - x_min + remove, y_max - y_min + remove], -1
            )
            box = BoxList(box, self.size, mode=mode)

        box.copy_field(self)

        return box

    def copy_field(self, box):
        for k, v in box.fields.items():
            self.fields[k] = v

    def area(self):
        box = self.box

        if self.mode == 'xyxy':
            remove = 1

            area = (box[:, 2] - box[:, 0] + remove) * (box[:, 3] - box[:, 1] + remove)

        elif self.mode == 'xywh':
            area = box[:, 2] * box[:, 3]

        return area

    def split_to_xyxy(self):
        if self.mode == 'xyxy':
            x_min, y_min, x_max, y_max = self.box.split(1, dim=-1)

            return x_min, y_min, x_max, y_max

        elif self.mode == 'xywh':
            remove = 1
            x_min, y_min, w, h = self.box.split(1, dim=-1)

            return (
                x_min,
                y_min,
                x_min + (w - remove).clamp(min=0),
                y_min + (h - remove).clamp(min=0),
            )

    def __len__(self):
        return self.box.shape[0]

    def __getitem__(self, index):
        box = BoxList(self.box[index], self.size, self.mode)

        return box

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))

        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled = self.box * ratio
            box = BoxList(scaled, size, mode=self.mode)

            for k, v in self.fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)

                box.fields[k] = v

            return box

        ratio_w, ratio_h = ratios
        x_min, y_min, x_max, y_max = self.split_to_xyxy()
        scaled_x_min = x_min * ratio_w
        scaled_x_max = x_max * ratio_w
        scaled_y_min = y_min * ratio_h
        scaled_y_max = y_max * ratio_h
        scaled = torch.cat([scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max], -1)
        box = BoxList(scaled, size, mode='xyxy')

        for k, v in self.fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)

            box.fields[k] = v

        return box.convert(self.mode)

    def clip(self, remove_empty=True):
        remove = 1

        max_width = self.size[0] - remove
        max_height = self.size[1] - remove

        self.box[:, 0].clamp_(min=0, max=max_width)
        self.box[:, 1].clamp_(min=0, max=max_height)
        self.box[:, 2].clamp_(min=0, max=max_width)
        self.box[:, 3].clamp_(min=0, max=max_height)

        if remove_empty:
            box = self.box
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])

            return self[keep]

        else:
            return self

    def to(self, device):
        box = BoxList(self.box.to(device), self.size, self.mode)

        for k, v in self.fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)

            box.fields[k] = v

        return box


def remove_small_box(boxlist, min_size):
    box = boxlist.convert('xywh').box
    _, _, w, h = box.unbind(dim=1)
    keep = (w >= min_size) & (h >= min_size)
    keep = keep.nonzero().squeeze(1)

    return boxlist[keep]



def boxlist_nms(boxlist, scores, threshold, max_proposal=-1):
    if threshold <= 0:
        return boxlist

    mode = boxlist.mode
    boxlist = boxlist.convert('xyxy')
    box = boxlist.box
    keep = ops.nms(box, scores, threshold)

    if max_proposal > 0:
        keep = keep[:max_proposal]

    boxlist = boxlist[keep]
    return boxlist.convert(mode)

def compute_location(features):
    locations = []
    _, _, height, width = features.shape
    location_per_level = compute_location_per_level(
        height, width, 1, features.device
    )
    locations.append(location_per_level)

    return locations

def compute_location_per_level(height, width, stride, device):
    shift_x = torch.arange(
        0, width * stride, step=stride, dtype=torch.float32, device=device
    )
    shift_y = torch.arange(
        0, height * stride, step=stride, dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    location = torch.stack((shift_x, shift_y), 1) + stride // 2

    return location