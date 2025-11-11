import os
import torch
import gradio as gr
from gradio_image_prompter import ImagePrompter
from torch.nn import DataParallel
from models.counter_infer import build_model
from utils.arg_parser import get_argparser
from utils.data import resize_and_pad
import torchvision.ops as ops
from torchvision import transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# Load model (once, to avoid reloading)

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_argparser().parse_args()
    args.zero_shot = True
    model = DataParallel(build_model(args).to(device))
    model.load_state_dict(torch.load('CNTQG_multitrain_ca44.pth', weights_only=True)['model'], strict=False)
    model.eval()
    return model, device


model, device = load_model()


# **Function to Process Image Once**
def process_image_once(inputs, enable_mask):
    model.module.return_masks = enable_mask

    image = inputs['image']
    drawn_boxes = inputs['points']
    image_tensor = torch.tensor(image).to(device)
    image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
    image_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)

    bboxes_tensor = torch.tensor([[box[0], box[1], box[3], box[4]] for box in drawn_boxes], dtype=torch.float32).to(
        device)

    img, bboxes, scale = resize_and_pad(image_tensor, bboxes_tensor, size=1024.0)
    img = img.unsqueeze(0).to(device)
    bboxes = bboxes.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _, _, _, masks = model(img, bboxes)

    # move ALL outputs to CPU, key-by-key (handles lists/dicts safely)
    outputs = [
        {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in out.items()}
        for out in outputs
    ]

    # make sure masks is on CPU and in a consistent structure
    if enable_mask and masks is not None:
        if torch.is_tensor(masks):
            masks = masks.detach().cpu()
        elif isinstance(masks, (list, tuple)):
            masks = [m.detach().cpu() for m in masks]
    else:
        masks = None

    return image, outputs, masks, img, scale, drawn_boxes


# **Post-process and Update Output**
def post_process(image, outputs, masks, img, scale, drawn_boxes, enable_mask, threshold):
    idx = 0
    thr_inv = 1.0 / threshold  # keep your original intent

    # --- pull tensors & drop batch dim if present ---
    pred_boxes = outputs[idx]['pred_boxes']          # [1, N, 4] or [N, 4]
    box_v      = outputs[idx]['box_v']               # [1, N]    or [N]

    if pred_boxes.dim() == 3 and pred_boxes.size(0) == 1:
        pred_boxes = pred_boxes[0]                   # -> [N, 4]
    if box_v.dim() == 2 and box_v.size(0) == 1:
        box_v = box_v[0]                             # -> [N]

    # --- selection mask over N ---
    sel = box_v > (box_v.max() / thr_inv)            # [N] bool

    # handle no survivors cleanly
    if sel.sum().item() == 0:
        # just draw the user boxes and 0 count
        image_pil = Image.fromarray(image.astype(np.uint8))
        draw = ImageDraw.Draw(image_pil)
        for box in drawn_boxes:
            draw.rectangle([box[0], box[1], box[3], box[4]], outline="red", width=3)
        # counter badge
        w, h = image_pil.size
        sq = int(0.05 * w)
        x1, y1 = 10, h - sq - 10
        draw.rectangle([x1, y1, x1+sq, y1+sq], outline="black", fill="black")
        font = ImageFont.load_default()
        txt = "0"
        text_x = x1 + (sq - draw.textlength(txt, font=font)) / 2
        text_y = y1 + (sq - 10) / 2
        draw.text((text_x, text_y), txt, fill="white", font=font)
        return image_pil, 0

    # --- NMS expects [N,4] boxes and [N] scores ---
    keep = ops.nms(pred_boxes[sel], box_v[sel], 0.5)
    pred_boxes = pred_boxes[sel][keep]               # [M,4]
    box_v = box_v[sel][keep]                         # [M]

    # clamp/scale to original image coords
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    pred_boxes = (pred_boxes / scale * img.shape[-1]).tolist()

    # to PIL
    image_pil = Image.fromarray(image.astype(np.uint8))

    # --- masks (optional) ---
    if enable_mask and masks is not None:
        from matplotlib import pyplot as plt
        # get batch slice, drop batch dim if present
        base = masks[idx]
        if base.dim() == 4 and base.size(0) == 1:
            base = base[0]                           # -> [N, H, W]

        if masks is not None:
            masks_ = base[sel][keep]
            N_masks = masks_.shape[0]
            indices = torch.randint(1, N_masks + 1, (1, N_masks), device=masks_.device).view(-1, 1, 1)
            mask_lbl = (masks_ * indices).sum(dim=0) # [H, W]
            mask_display = (
                T.Resize(
                    (int(img.shape[2] / scale), int(img.shape[3] / scale)),
                    interpolation=T.InterpolationMode.NEAREST
                )(mask_lbl.unsqueeze(0))[0]
            )[:image_pil.size[1], :image_pil.size[0]]
            cmap = plt.cm.tab20
            norm = plt.Normalize(vmin=0, vmax=N_masks)
            rgba = cmap(norm(mask_display))
            rgba[mask_display == 0, -1] = 0
            rgba[mask_display != 0, -1] = 0.5
            overlay = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")
            image_pil = image_pil.convert("RGBA")
            image_pil = Image.alpha_composite(image_pil, overlay)

    # --- draw boxes & user input ---
    draw = ImageDraw.Draw(image_pil)
    for box in pred_boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline="orange", width=2)
    for box in drawn_boxes:
        draw.rectangle([box[0], box[1], box[3], box[4]], outline="red", width=3)

    # counter badge
    w, h = image_pil.size
    sq = int(0.05 * w)
    x1, y1 = 10, h - sq - 10
    draw.rectangle([x1, y1, x1+sq, y1+sq], outline="black", fill="black")
    font = ImageFont.load_default()
    txt = str(len(pred_boxes))
    text_x = x1 + (sq - draw.textlength(txt, font=font)) / 2
    text_y = y1 + (sq - 10) / 2
    draw.text((text_x, text_y), txt, fill="white", font=font)

    return image_pil, len(pred_boxes)

iface = gr.Blocks()

with iface:
    # States
    gr.Markdown(
        """
# GECO Demo — Generalized-Scale Object Counting with Gradual Query Aggregation  
*(Accepted at AAAI 2026)*  

Welcome to the **GeCo2** demo — a **Generalized-Scale Object Counting with Gradual Query Aggregation** method capable of detecting and counting **any object** from just a few examples.  

### 🧩 How to use this demo:
1. **Upload an image** below.  
2. **Draw one or more bounding boxes** around example instances of your target category (you can give **1 to N examples** — e.g., one car, three apples, etc.).  
3. Click **"Count"** to detect and count all similar objects in the image.  
4. (Optional) Toggle **"Predict masks"** to visualize segmentation masks for detected objects.  
        """
    )
    image_input = gr.State()
    outputs_state = gr.State()
    masks_state = gr.State()
    img_state = gr.State()
    scale_state = gr.State()
    drawn_boxes_state = gr.State()

    with gr.Row():
        ip = ImagePrompter(label="Upload & draw boxes")   # <-- use this in .click
        image_output = gr.Image(type="pil", label="Result")

    with gr.Row():
        count_output = gr.Number(label="Total Count", precision=0)
        enable_mask = gr.Checkbox(label="Predict masks", value=True)
        threshold = gr.Slider(0.05, 0.95, value=0.33, step=0.01, label="Threshold")

    count_button = gr.Button("Count")

    def initial_process(ip_data, enable_mask_val, thr):
        # ip_data is a dict from ImagePrompter: {'image': np.ndarray, 'points': [...]}
        image, outputs, masks, img, scale, drawn_boxes = process_image_once(ip_data, enable_mask_val)
        out_img, n = post_process(image, outputs, masks, img, scale, drawn_boxes, enable_mask_val, thr)
        return (
            out_img, n,            # visible outputs
            image, outputs, masks, img, scale, drawn_boxes  # states
        )

    def update_threshold(thr, image, outputs, masks, img, scale, drawn_boxes, enable_mask_val):
        return post_process(image, outputs, masks, img, scale, drawn_boxes, enable_mask_val, thr)

    # Click: use the prompter directly as input (like your working minimal example)
    count_button.click(
        initial_process,
        [ip, enable_mask, threshold],
        [image_output, count_output,
         image_input, outputs_state, masks_state, img_state, scale_state, drawn_boxes_state]
    )

    # Slider/checkbox update without re-inference
    threshold.change(
        update_threshold,
        [threshold, image_input, outputs_state, masks_state, img_state, scale_state, drawn_boxes_state, enable_mask],
        [image_output, count_output]
    )
    enable_mask.change(
        update_threshold,
        [threshold, image_input, outputs_state, masks_state, img_state, scale_state, drawn_boxes_state, enable_mask],
        [image_output, count_output]
    )

# queue helps avoid frontend “stuck” feelings under GPU load
iface.queue(max_size=8)  # only keep max_size
iface.launch(show_error=True, debug=True, share=True, max_threads=1)

