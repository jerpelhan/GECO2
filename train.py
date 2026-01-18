import argparse
import os
import random
from time import perf_counter

import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import ops

from models.counter import build_model
from models.matcher import build_matcher
from utils.arg_parser import get_argparser
from utils.data import FSC147DATASET
from utils.data import pad_collate
from utils.losses import SetCriterion

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

DATASETS = {
    'fsc147': FSC147DATASET
}

def train(args):
    if 'SLURM_PROCID' in os.environ:
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        print("Running on SLURM", world_size, rank, gpu)
    else:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
  

    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )
    print("init dist: ", dist.is_initialized(), rank, world_size, gpu)

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu,
        find_unused_parameters=True
    )

    backbone_params = dict()
    non_backbone_params = dict()
    for n, p in model.named_parameters():
        if 'backbone' in n:
            backbone_params[n] = p
        else:
            non_backbone_params[n] = p

    optimizer = torch.optim.AdamW(
        [
            {'params': non_backbone_params.values()},
            {'params': backbone_params.values(), 'lr': args.backbone_lr}
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)
    if args.resume_training:
        checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name_resume_from}.pth'))
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = 0
        best = 10000000000000
    else:
        start_epoch = 0
        best = 10000000000000
    matcher = build_matcher(args)
    criterion = SetCriterion(0, matcher, {"loss_giou": args.giou_loss_coef}, ["bboxes", "ce"],
                             focal_alpha=args.focal_alpha)
    criterion.to(device)

    train = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split='train',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot,
        training=True
    )
    val = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split='val',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        training=True
    )
    test = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split='test',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        training=True
    )
    train_loader = DataLoader(
        train,
        sampler=DistributedSampler(train),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val,
        sampler=DistributedSampler(val),
        batch_size=1,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate
    )
    test_loader = DataLoader(
        test,
        sampler=DistributedSampler(test),
        batch_size=1,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate
    )

    print(rank)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if rank == 0:
            start = perf_counter()
        train_loss = torch.tensor(0.0).to(device)
        val_loss = torch.tensor(0.0).to(device)
        val_ae = torch.tensor(0.0).to(device)
        val_rmse = torch.tensor(0.0).to(device)
        train_ae = torch.tensor(0.0).to(device)
        test_loss = torch.tensor(0.0).to(device)
        test_ae = torch.tensor(0.0).to(device)
        test_rmse = torch.tensor(0.0).to(device)
        index_val = torch.tensor(0.0).to(device)

        train_loader.sampler.set_epoch(epoch)
        model.train()
        criterion.train()
        for img, bboxes, density_map, img_name, gt_bboxes in train_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            gt_bboxes = gt_bboxes.to(device)

            optimizer.zero_grad()
            outputs, ref_points, centerness, outputs_coord, aux = model(img, bboxes)
            outputs_aux, ref_points_aux, centerness_aux, outputs_coord_aux = aux

            losses = []
            num_objects_pred = []

            nms_bboxes = []
            for idx in range(img.shape[0]):
                target_bboxes = gt_bboxes[idx][torch.logical_not((gt_bboxes[idx] == 0).all(dim=1))] / 1024
                l = criterion(outputs[idx],
                              [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                              centerness[idx], ref_points[idx])

                l1 = criterion(outputs_aux[idx],
                               [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                               centerness_aux[idx], ref_points_aux[idx])
                alpha = 0
                # if min width or height is less than 10px, calculate loss
                if min((target_bboxes[:, 3] - target_bboxes[:, 1]).mean() * args.image_size,
                       (target_bboxes[:, 2] - target_bboxes[:, 0]).mean() * args.image_size) < 25:
                    alpha = 0.3
                # else:
                #     losses.append(centerness_aux[idx].sum()/len(target_bboxes)*0.5)
                keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8],
                               outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8], 0.5)

                boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8])[keep]
                nms_bboxes.append(boxes)
                num_objects_pred.append(len(boxes))
                losses.append(l['loss_giou'] + l["loss_ce"]+  l["loss_bbox"])
                losses.append(l1['loss_giou'] * alpha + l1["loss_ce"] * alpha + l["loss_bbox"]*alpha)
            num_objects_gt = density_map.flatten(1).sum(dim=1)
            loss = sum(losses)

            loss.backward()

            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += loss
            train_ae += torch.abs(num_objects_gt - torch.tensor(num_objects_pred)).sum()

        criterion.eval()
        model.eval()
        with torch.no_grad():
            index_val = 0
            for img, bboxes, density_map, img_name, gt_bboxes in val_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                gt_bboxes = gt_bboxes.to(device)

                optimizer.zero_grad()
                outputs, ref_points, centerness, outputs_coord, aux = model(img, bboxes)
                outputs_aux, ref_points_aux, centerness_aux, outputs_coord_aux = aux
                losses = []

                num_objects_pred = []
                nms_bboxes = []

                for idx in range(img.shape[0]):
                    target_bboxes = gt_bboxes[idx][torch.logical_not((gt_bboxes[idx] == 0).all(dim=1))] / 1024

                    l1 = criterion(outputs_aux[idx],
                                   [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                                   centerness_aux[idx], ref_points_aux[idx])
                    alpha = 0
                    # if min width or height is less than 10px, calculate loss
                    if min((target_bboxes[:, 3] - target_bboxes[:, 1]).mean() * args.image_size,
                           (target_bboxes[:, 2] - target_bboxes[:, 0]).mean() * args.image_size) < 25:
                        alpha = 1

                    l = criterion(outputs[idx],
                                  [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                                  centerness[idx], ref_points[idx])
                    keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8],
                                   outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8], 0.5)

                    boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8])[keep]
                    nms_bboxes.append(boxes)
                    num_objects_pred.append(len(boxes))
                    losses.append(l['loss_giou'] + l["loss_ce"])
                    losses.append(l1['loss_giou'] + l1["loss_ce"])
                num_objects_gt = density_map.flatten(1).sum(dim=1)
                loss = sum(losses)

                train_loss += loss

                num_objects_pred = torch.tensor(num_objects_pred)

                val_loss += loss * img.size(0)
                val_ae += torch.abs(
                    num_objects_gt - num_objects_pred
                ).sum()
                val_rmse += torch.pow(
                    num_objects_gt - num_objects_pred, 2
                ).sum()

                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            index_val = 0
            for img, bboxes, density_map, img_name, gt_bboxes in test_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)
                gt_bboxes = gt_bboxes.to(device)

                optimizer.zero_grad()
                outputs, ref_points, centerness, outputs_coord, aux = model(img, bboxes)

                losses = []

                num_objects_pred = []
                nms_bboxes = []

                for idx in range(img.shape[0]):
                    target_bboxes = gt_bboxes[idx][torch.logical_not((gt_bboxes[idx] == 0).all(dim=1))] / 1024

                    l = criterion(outputs[idx],
                                  [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                                  centerness[idx], ref_points[idx])
                    keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8],
                                   outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8], 0.5)

                    boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / 8])[keep]
                    nms_bboxes.append(boxes)
                    num_objects_pred.append(len(boxes))
                    losses.append(l['loss_giou'] + l["loss_ce"])
                num_objects_gt = density_map.flatten(1).sum(dim=1)
                loss = sum(losses)


                num_objects_pred = torch.tensor(num_objects_pred)

                test_loss += loss * img.size(0)
                test_ae += torch.abs(
                    num_objects_gt - num_objects_pred
                ).sum()
                test_rmse += torch.pow(
                    num_objects_gt - num_objects_pred, 2
                ).sum()

                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        dist.all_reduce(train_loss)
        dist.all_reduce(val_loss)
        dist.all_reduce(val_rmse)
        dist.all_reduce(val_ae)
        dist.all_reduce(train_ae)
        dist.all_reduce(test_loss)
        dist.all_reduce(test_rmse)
        dist.all_reduce(test_ae)

        scheduler.step()

        if rank == 0:

            end = perf_counter()
            best_epoch = False


            if val_rmse.item() / len(val) < best:
                best = val_rmse.item() / len(val)
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict()
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.model_path, f'{args.model_name}.pth')
                )

                best_epoch = True

            print(
                f"Epoch: {epoch}",
                f"Train loss: {train_loss.item():.3f}",
                f"Val loss: {val_loss.item():.3f}",
                f"Train MAE: {train_ae.item() / len(train):.3f}",
                f"Val MAE: {val_ae.item() / len(val):.3f}",
                f"Val RMSE: {torch.sqrt(val_rmse / len(val)).item():.2f}",
                f"Test MAE: {test_ae.item() / len(test):.3f}",
                f"Test RMSE: {torch.sqrt(test_rmse / len(test)).item():.2f}",
                f"Epoch time: {end - start:.3f} seconds",
                'best' if best_epoch else ''
            )
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CNT', parents=[get_argparser()])
    args = parser.parse_args()
    print(args)
    train(args)
