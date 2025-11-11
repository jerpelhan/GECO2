import argparse
import json
import os
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.ops import box_convert
from torchvision.transforms import functional as TVF
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

def xywh_to_x1y1x2y2(xywh):
    x, y, w, h = xywh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]



def resize_and_pad(img, bboxes, density_map=None, gt_bboxes=None, size=1024.0, zero_shot=False, train=False):
    resize512 = T.Resize((512, 512), antialias=True)
    channels, original_height, original_width = img.shape
    longer_dimension = max(original_height, original_width)
    scaling_factor = size / longer_dimension
    scaled_bboxes = bboxes * scaling_factor
    if not zero_shot and not train:
        a_dim = ((scaled_bboxes[:, 2] - scaled_bboxes[:, 0]).mean() + (
                scaled_bboxes[:, 3] - scaled_bboxes[:, 1]).mean()) / 2
        scaling_factor = min(1.0, 80 / a_dim.item()) * scaling_factor
    resized_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scaling_factor, mode='bilinear',
                                                  align_corners=False)

    size = int(size)
    pad_height = max(0, size - resized_img.shape[2])
    pad_width = max(0, size - resized_img.shape[3])

    padded_img = torch.nn.functional.pad(resized_img, (0, pad_width, 0, pad_height), mode='constant', value=0)[0]
    if density_map is not None:
        original_sum = density_map.sum()
        _, w0, h0 = density_map.shape
        _, W, H = img.shape
        resized_density_map = torch.nn.functional.interpolate(density_map.unsqueeze(0), size=(W, H), mode='bilinear',
                                                            align_corners=False)
        resized_density_map = torch.nn.functional.interpolate(resized_density_map, scale_factor=scaling_factor,
                                                            mode='bilinear',
                                                            align_corners=False)
        padded_density_map = \
            torch.nn.functional.pad(resized_density_map, (0, pad_width, 0, pad_height), mode='constant', value=0)[0]
        padded_density_map = resize512(padded_density_map)
        padded_density_map = padded_density_map / padded_density_map.sum() * original_sum

    bboxes = bboxes * torch.tensor([scaling_factor, scaling_factor, scaling_factor, scaling_factor]).to(bboxes.device)
    if gt_bboxes is None and density_map is None:
        return padded_img, bboxes, scaling_factor
    gt_bboxes = gt_bboxes * torch.tensor([scaling_factor, scaling_factor, scaling_factor, scaling_factor])
    return padded_img, bboxes, padded_density_map, gt_bboxes, scaling_factor, (pad_width, pad_height)



class FSC147DATASET(Dataset):
    def __init__(
            self, data_path, img_size, split='train', num_objects=3,
            tiling_p=0.5, zero_shot=False, return_ids=False, training=False
    ):
        self.split = split
        self.data_path = data_path
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size), antialias=True)
        self.resize512 = T.Resize((512, 512), antialias=True)
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        self.return_ids = return_ids
        self.training = training

        with open(
                os.path.join(self.data_path, 'annotations', 'Train_Test_Val_FSC_147.json'), 'rb'
        ) as file:
            splits = json.load(file)
            self.image_names = splits[split]
        with open(
                os.path.join(self.data_path, 'annotations', 'annotation_FSC147_384.json'), 'rb'
        ) as file:
            self.annotations = json.load(file)

        self.labels = COCO(os.path.join(self.data_path, 'annotations', 'instances_' + split + '.json'))
        self.img_name_to_ori_id = self.map_img_name_to_ori_id()

    def get_gt_bboxes(self, idx):
        coco_im_id = self.img_name_to_ori_id[self.image_names[idx]]
        anno_ids = self.labels.getAnnIds([coco_im_id])
        annotations = self.labels.loadAnns(anno_ids)
        bboxes = []
        for a in annotations:
            bboxes.append(xywh_to_x1y1x2y2(a['bbox']))
        return bboxes

    def __getitem__(self, idx: int):
        img = Image.open(os.path.join(
            self.data_path,
            'images_384_VarV2',
            self.image_names[idx]
        )).convert("RGB")

        gt_bboxes = torch.tensor(self.get_gt_bboxes(idx))

        img = T.Compose([
            T.ToTensor(),
        ])(img)

        bboxes = torch.tensor(
            self.annotations[self.image_names[idx]]['box_examples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]

        density_map = torch.from_numpy(np.load(os.path.join(
            self.data_path,
            'gt_density_map_adaptive_512_512_object_VarV2',
            os.path.splitext(self.image_names[idx])[0] + '.npy',
        ))).unsqueeze(0)


        img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = tile_multiscale(img, bboxes, density_map,
                                         gt_bboxes=gt_bboxes)

        original_sum = density_map.sum()
        density_map = self.resize512(density_map)
        density_map = density_map / density_map.sum() * original_sum
        gt_bboxes = torch.clamp(gt_bboxes, min=0, max=1024)


        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        return img, bboxes, density_map, torch.tensor(idx), gt_bboxes, torch.tensor(scaling_factor), padwh

    def __len__(self):
        return len(self.image_names)

    def map_img_name_to_ori_id(self, ):
        all_coco_imgs = self.labels.imgs
        map_name_2_id = dict()
        for k, v in all_coco_imgs.items():
            img_id = v["id"]
            img_name = v["file_name"]
            map_name_2_id[img_name] = img_id
        return map_name_2_id

