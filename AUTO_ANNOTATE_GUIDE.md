# Auto Annotate Guide (GECO2)

This guide explains how to run `auto_annotate_from_coco_prompt.py` end-to-end.

## 1) Open project directory

```bash
cd path/to/GECO2
```

## 2) Create and activate conda environment (first time only)

```bash
conda create -n coco-auto python=3.10 -y
conda activate coco-auto
```

## 3) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r req.txt
```

## 4) Build required CUDA extension

This is required for GECO2 attention ops.

```bash
cd models/ops
python -m pip install -e . --no-build-isolation
cd ../..
```

If you skip this step, you will get:

- `ModuleNotFoundError: No module named 'MultiScaleDeformableAttention'`

## 5) Download model checkpoint (if not already available)

```bash
wget -O CNTQG_multitrain_ca44.pth "https://huggingface.co/datasets/jerpelhan/geco2-assets/resolve/main/weights/CNTQG_multitrain_ca44.pth?download=true"
```

## 6) Prepare required inputs

You need:

- Prompt image annotated in CVAT
- Prompt COCO JSON export from CVAT
- Target folder containing images to auto-annotate

Example:

- Prompt image: `/opt/workspace_zeeshan/GECO2_for_Object_detection/cabbage_prompt/images/crop_1290_342909_5553221.jpg`
- Prompt COCO: `/opt/workspace_zeeshan/GECO2_for_Object_detection/cabbage_prompt/annotations/instances_default.json`
- Target images: `/opt/workspace_zeeshan/GECO2_for_Object_detection/cabbage_/images`

## 7) Run auto annotation

```bash
python auto_annotate_from_coco_prompt.py \
  --prompt-image /opt/workspace_zeeshan/GECO2_for_Object_detection/cabbage_prompt/images/crop_1290_342909_5553221.jpg \
  --prompt-coco-json /opt/workspace_zeeshan/GECO2_for_Object_detection/cabbage_prompt/annotations/instances_default.json \
  --target-images-dir /opt/workspace_zeeshan/GECO2_for_Object_detection/cabbage_/images \
  --weights /opt/workspace_zeeshan/GECO2_for_Object_detection/GECO2/CNTQG_multitrain_ca44.pth \
  --output-json /opt/workspace_zeeshan/yoloe_exploring/images/yoloe_images/annotations/instances_default.json
```

## 8) Verify output

On success, the script prints:

- output JSON path
- number of processed images
- number of created annotations
- weights path used

Expected output path:

- `/opt/workspace_zeeshan/yoloe_exploring/images/yoloe_images/annotations/instances_default.json`

## Optional arguments

- `--use-all-categories`
- `--max-exemplars-per-category 20`
- `--score-threshold 0.33`
- `--nms-iou 0.5`
- `--min-confidence 0.0`
- `--extensions ".jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff"`

## Troubleshooting

- **Error**: `No module named MultiScaleDeformableAttention`  
  **Fix**: run step 4 again in the same active conda env.

- **Error**: `Checkpoint not found`  
  **Fix**: download checkpoint (step 5) or pass correct `--weights`.

- **Error**: `CUDA is required for GECO2 inference`  
  **Fix**: run on a CUDA-enabled machine/env and verify:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

- **Error**: prompt image not resolved in COCO JSON  
  **Fix**: pass explicit image id:
  ```bash
  --prompt-image-id <id>
  ```
