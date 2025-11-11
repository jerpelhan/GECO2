import argparse


def get_argparser():

    parser = argparse.ArgumentParser("CNT", add_help=False)

    parser.add_argument('--model_name', default='base_3_shot_softmax1', type=str)
    parser.add_argument('--model_name_resumed', default='base_3_shot_softmax1', type=str)
    parser.add_argument(
        '--data_path',
        default='/storage/datasets/fsc147',
        type=str
    )
    parser.add_argument(
        '--model_path',
        default='/d/hpc/projects/FRI/pelhanj/CNT_SAM2/models/',
        type=str
    )
    parser.add_argument('--dataset', default='fsc147', type=str)
    parser.add_argument('--backbone', default='SAM', type=str)
    parser.add_argument('--reduction', default=16, type=int)
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--num_enc_layers', default=3, type=int)
    parser.add_argument('--num_ope_iterative_steps', default=3, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--kernel_dim', default=3, type=int)
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--backbone_lr', default=0, type=float)
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=0.1, type=float)
    parser.add_argument('--aux_weight', default=0.3, type=float)
    parser.add_argument('--tiling_p', default=0.5, type=float)
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--encode', action='store_true')
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--cost_class", default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument("--cost_bbox", default=1, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument("--cost_giou", default=2, type=float, help="giou box coefficient in the matching cost")
    parser.add_argument("--focal_alpha", default=0.25, type=float)
    parser.add_argument("--model_name_resume_from", default='base_3_shot_softmax1', type=str)

    return parser
