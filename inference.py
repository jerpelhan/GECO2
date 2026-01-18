import argparse
import json
import math
import os

import numpy as np
import skimage
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import ops
from torchvision.transforms import Resize
from tqdm import tqdm
from models.counter_infer import build_model
from models.matcher import build_matcher
from utils.arg_parser import get_argparser
from utils.box_ops import BoxList
from utils.data import FSC147DATASET, pad_collate_test
from utils.losses import SetCriterion


@torch.no_grad()
def evaluate(args):
    gpu=0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )

    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pth'))['model']
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    for split in ['val', 'test']:
        test = FSC147DATASET(
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
            return_ids=True,
            training=False
        )
        test_loader = DataLoader(
            test,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers,
            collate_fn=pad_collate_test,
        )
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        model.eval()
        matcher = build_matcher(args)
        criterion = SetCriterion(0, matcher, {"loss_giou":args.giou_loss_coef}, ["bboxes", "ce"], focal_alpha=args.focal_alpha)
        criterion.to(device)


        predictions = dict()
        predictions["categories"] = [{"name": "fg", "id": 1}]
        predictions["images"] = list()
        predictions["annotations"] = list()
        anno_id = 1

        for img, bboxes, density_map, ids, gt_bboxes, scaling_factor, padwh in test_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            gt_bboxes = gt_bboxes.to(device)

            outputs, ref_points, _, _, masks = model(img, bboxes)

            w, h = img.shape[-1], img.shape[-2]
            losses = []
            num_objects_gt = []
            num_objects_pred = []
            nms_bboxes = []
            nms_scores = []
            nms_masks = []
            for idx in range(img.shape[0]):

                thr = 1/0.11

                if len(outputs[idx]['pred_boxes'][-1]) == 0:
                    nms_bboxes.append(torch.zeros((0, 4)))
                    nms_scores.append(torch.zeros((0)))
                    num_objects_pred.append(0)

                else:

                    # threshold and NMS
                    v = outputs[idx]["box_v"]
                    v_thr = v.max() / thr
                    mask = v > v_thr
                    keep = ops.nms(
                        outputs[idx]["pred_boxes"][mask],
                        v[mask],
                        0.5,
                    )
                    boxes = outputs[idx]["pred_boxes"][mask][keep]
                    boxes = torch.clamp(boxes, 0, 1)
                    scores = outputs[idx]["scores"][mask][keep]

                    # remove bboxes in padded area
                    maxw = (img.shape[-1] - padwh[idx][0]).to(device)
                    maxh = (img.shape[-2] - padwh[idx][1]).to(device)
                    center = (boxes[:, :2] + boxes[:, 2:]) / 2
                    valid = (center[:, 0] * h < maxw) & (center[:, 1] * w < maxh)
                    scores = scores[valid]
                    boxes = boxes[valid]

                    nms_bboxes.append(boxes)
                    nms_scores.append(scores)
                    num_objects_pred.append(len(boxes))

                    if False:
                        from matplotlib import pyplot as plt
                        fig1 = plt.figure(figsize=(8, 8))
                        ((ax1_11, ax1_12), (ax1_21, ax1_22)) = fig1.subplots(2, 2)
                        fig1.tight_layout(pad=2.5)
                        img_ = np.array((img).cpu()[idx].permute(1, 2, 0))  
                        img_ = img_ - np.min(img_)
                        img_ = img_ / np.max(img_)
                        ax1_11.imshow(img_)
                        ax1_11.set_title("Input", fontsize=8)
                        bboxes_ = np.array(bboxes.cpu())[idx]
                        for i in range(3):
                            ax1_11.plot([bboxes_[i][0], bboxes_[i][0], bboxes_[i][2], bboxes_[i][2], bboxes_[i][0]],
                                        [bboxes_[i][1], bboxes_[i][3], bboxes_[i][3], bboxes_[i][1], bboxes_[i][1]], c='r')
                        ax1_12.imshow(img_)
                        ax1_12.set_title("gt bboxes", fontsize=8)
                        target_bboxes = gt_bboxes[idx][torch.logical_not((gt_bboxes[idx] == 0).all(dim=1))]
                        bboxes_ = ((target_bboxes)).detach().cpu()
                        for i in range(len(bboxes_)):
                            ax1_12.plot([bboxes_[i][0], bboxes_[i][0], bboxes_[i][2], bboxes_[i][2], bboxes_[i][0]],
                                        [bboxes_[i][1], bboxes_[i][3], bboxes_[i][3], bboxes_[i][1], bboxes_[i][1]], c='g')
                        ax1_21.imshow(img_)

                        bboxes_pred = nms_bboxes[idx]
                        bboxes_ = ((bboxes_pred * img_.shape[0])).detach().cpu()
                        for i in range(len(bboxes_)):
                            ax1_21.plot([bboxes_[i][0], bboxes_[i][0], bboxes_[i][2], bboxes_[i][2], bboxes_[i][0]],
                                        [bboxes_[i][1], bboxes_[i][3], bboxes_[i][3], bboxes_[i][1], bboxes_[i][1]],
                                        c='orange', linewidth=0.5)
                        ax1_21.set_title("#GT-#PRED=" + str(len(target_bboxes) - len(bboxes_pred)))
                        from torchvision import transforms as T
                        res = T.Resize((1024, 1024))
                        ax1_21.imshow(res(centerness).detach().cpu()[idx][0], alpha=0.6)
                        plt.savefig(test.image_names[ids[idx].item()], dpi=200)
                        plt.close()

            for idx in range(img.shape[0]):
                img_info = {
                    "id": test.map_img_name_to_ori_id()[test.image_names[ids[idx].item()]],  
                    "file_name": "None",
                }
                bboxes = ops.box_convert(nms_bboxes[idx], 'xyxy', 'xywh')
                bboxes = bboxes * img.shape[-1] / scaling_factor[idx]
                for idxi in range(len(nms_bboxes[idx])):
                    box = bboxes[idxi].detach().cpu()
                    anno = {
                        "id": anno_id,
                        "image_id": test.map_img_name_to_ori_id()[test.image_names[ids[idx].item()]],  
                        "area": int((box[2] * box[3]).item()),
                        "bbox": [int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())],
                        "category_id": 1,
                        "score": float(nms_scores[idx][idxi].item()),
                    }
                    anno_id += 1
                    predictions["annotations"].append(anno)
                predictions["images"].append(img_info)
            num_objects_gt =  density_map.flatten(1).sum(dim=1)
            num_objects_pred = torch.tensor(num_objects_pred)
            ae += torch.abs(
                num_objects_gt - num_objects_pred
            ).sum()
            se += torch.pow(
                num_objects_gt - num_objects_pred, 2
            ).sum()
        print(
            f"{split.capitalize()} set",
            f"MAE: {ae.item() / len(test):.2f}",
            f"RMSE: {torch.sqrt(se / len(test)).item():.2f}",
        )

        with open("geco2_" + split + ".json", "w") as handle:
            json.dump(predictions, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GECO2', parents=[get_argparser()])
    args = parser.parse_args()
    print(args)
    print("model_name: ", args.model_name)
    evaluate(args)
