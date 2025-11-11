import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    bboxes_tensor = torch.tensor([[box[0], box[1], box[3], box[4]] for box in drawn_boxes], dtype=torch.float32).to(device)

    img, bboxes, scale = resize_and_pad(image_tensor, bboxes_tensor, size=1024.0)
    img = img.unsqueeze(0).to(device)
    bboxes = bboxes.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _, _, _, masks = model(img, bboxes)

    return image, outputs, masks, img, scale, drawn_boxes

# **Post-process and Update Output**
def post_process(image, outputs, masks, img, scale, drawn_boxes, enable_mask, threshold):
    idx = 0
    threshold = 1/threshold
    keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / threshold],
                   outputs[idx]['box_v'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / threshold], 0.5)

    pred_boxes = outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / threshold][keep]
    pred_boxes = torch.clamp(pred_boxes, 0, 1)

    pred_boxes = (pred_boxes.cpu() / scale * img.shape[-1]).tolist()

    image = Image.fromarray((image).astype(np.uint8))

    if enable_mask:
        from matplotlib import pyplot as plt
        masks_ = masks[idx][(outputs[idx]['box_v'] > outputs[idx]['box_v'].max() / threshold)[0]]
        N_masks = masks_.shape[0]
        indices = torch.randint(1, N_masks + 1, (1, N_masks), device=masks_.device).view(-1, 1, 1)
        masks = (masks_ * indices).sum(dim=0)
        mask_display = (
            T.Resize((int(img.shape[2] / scale), int(img.shape[3] / scale)), interpolation=T.InterpolationMode.NEAREST)(
                masks.cpu().unsqueeze(0))[0])[:image.size[1], :image.size[0]]
        cmap = plt.cm.tab20
        norm = plt.Normalize(vmin=0, vmax=N_masks)
        del masks
        del masks_
        del outputs
        rgba_image = cmap(norm(mask_display))
        rgba_image[mask_display == 0, -1] = 0
        rgba_image[mask_display != 0, -1] = 0.5

        overlay = Image.fromarray((rgba_image * 255).astype(np.uint8), mode="RGBA")
        image = image.convert("RGBA")
        image = Image.alpha_composite(image, overlay)


    draw = ImageDraw.Draw(image)
    for box in pred_boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline="orange", width=5)
    # for box in drawn_boxes:
    #     draw.rectangle([box[0], box[1], box[3], box[4]], outline="red", width=3)

    width, height = image.size
    square_size = int(0.05 * width)
    x1, y1 = 10, height - square_size - 10
    x2, y2 = x1 + square_size, y1 + square_size

    # draw.rectangle([x1, y1, x2, y2], outline="black", fill="black", width=1)
    # font = ImageFont.load_default()
    # txt = str(len(pred_boxes))
    # w = draw.textlength(txt, font=font)
    # text_x = x1 + (square_size - w) / 2
    # text_y = y1 + (square_size - 10) / 2
    # draw.text((text_x, text_y), txt, fill="white", font=font)

    return image, len(pred_boxes)


iface = gr.Blocks()

with iface:
    # Store intermediate states
    image_input = gr.State()
    outputs_state = gr.State()
    masks_state = gr.State()
    img_state = gr.State()
    scale_state = gr.State()
    drawn_boxes_state = gr.State()

    # UI Layout: Input Section
    with gr.Row():
        image_prompter = ImagePrompter()
        image_output = gr.Image(type="pil")
        

    # UI Layout: Output Section
    with gr.Row():
        count_output = gr.Number(label="Total Count")
        enable_mask = gr.Checkbox(label="Predict masks", value=True)  # Mask enabled by default
        threshold = gr.Slider(0.05, 0.95, value=0.33, step=0.01, label="Threshold")  # Updated range and default
        

    # Create the 'Count' button
    count_button = gr.Button("Count")

    # Process image once when "Count" button is pressed
    def initial_process(inputs, enable_mask, threshold):
        # Perform inference once
        image, outputs, masks, img, scale, drawn_boxes = process_image_once(inputs, enable_mask)

        # Save intermediate states
        return (
            *post_process(image, outputs, masks, img, scale, drawn_boxes, enable_mask, threshold),  # Processed outputs
            image, outputs, masks, img, scale, drawn_boxes  # Store in states for later use
        )

    # Update image and count when the threshold slider changes (post-process only)
    def update_threshold(threshold, image, outputs, masks, img, scale, drawn_boxes, enable_mask):
        return post_process(image, outputs, masks, img, scale, drawn_boxes, enable_mask, threshold)

    # Run initial inference and post-process when "Count" button is clicked
    count_button.click(
        initial_process,
        [image_prompter, enable_mask, threshold],  # Inputs
        [image_output, count_output, image_input, outputs_state, masks_state, img_state, scale_state, drawn_boxes_state]  # Outputs + States
    )

    # Adjust the output dynamically based on the threshold slider (no re-inference)
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

iface.launch(share=True)

