#!/bin/bash

conda create -n test_geco2 python=3.10 -y
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
cd ./Deformable-DETR/models/ops
CUDA_VISIBLE_DEVICES=0 python -m pip install .
cd ../../..
cp -r ./Deformable-DETR/models/ops ./models/ops
pip install hydra-core
pip install scikit-image
pip install pycocotools
pip install einops
pip install "numpy<2"
pip install gradio
pip install gradio_image_prompter
pip install huggingface-hub==0.34.3
pip install --force-reinstall "pydantic<2.11"