#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config (edit if needed)
# -----------------------------
ENV_NAME="coco-auto"
PYTHON_VERSION="3.10"
PROJECT_DIR="/opt/workspace_zeeshan/GECO2_for_Object_detection/GECO2"
WEIGHTS_PATH="${PROJECT_DIR}/CNTQG_multitrain_ca44.pth"
WEIGHTS_URL="https://huggingface.co/datasets/jerpelhan/geco2-assets/resolve/main/weights/CNTQG_multitrain_ca44.pth?download=true"

# -----------------------------
# Helpers
# -----------------------------
have_cmd() { command -v "$1" >/dev/null 2>&1; }

if ! have_cmd conda; then
  echo "[ERROR] conda not found in PATH. Install Anaconda/Miniconda first."
  exit 1
fi

# Ensure conda shell integration is available in this script
eval "$(conda shell.bash hook)"

echo "[INFO] Project dir: ${PROJECT_DIR}"
cd "${PROJECT_DIR}"

# -----------------------------
# Create env if missing
# -----------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Conda env '${ENV_NAME}' already exists."
else
  echo "[INFO] Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION})..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

echo "[INFO] Activating env '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

# -----------------------------
# Install core dependencies
# -----------------------------
echo "[INFO] Installing PyTorch (CUDA 12.6 wheels)..."
python -m pip install --upgrade pip
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126

echo "[INFO] Installing Python dependencies..."
if [[ -f "${PROJECT_DIR}/req.txt" ]]; then
  python -m pip install -r "${PROJECT_DIR}/req.txt"
else
  # fallback minimal deps
  python -m pip install hydra-core scikit-image pycocotools einops "numpy<2" gradio gradio_image_prompter huggingface-hub==0.34.3 "pydantic<2.11"
fi

# -----------------------------
# Build MultiScaleDeformableAttention extension
# -----------------------------
echo "[INFO] Building CUDA ops extension..."
cd "${PROJECT_DIR}/models/ops"
python -m pip install -e . --no-build-isolation
cd "${PROJECT_DIR}"

# -----------------------------
# Download checkpoint if missing
# -----------------------------
if [[ -f "${WEIGHTS_PATH}" ]]; then
  echo "[INFO] Weights already exist: ${WEIGHTS_PATH}"
else
  echo "[INFO] Downloading model weights to ${WEIGHTS_PATH} ..."
  if have_cmd wget; then
    wget -O "${WEIGHTS_PATH}" "${WEIGHTS_URL}"
  elif have_cmd curl; then
    curl -L "${WEIGHTS_URL}" -o "${WEIGHTS_PATH}"
  else
    echo "[ERROR] Neither wget nor curl is available to download weights."
    exit 1
  fi
fi

# -----------------------------
# Quick sanity checks
# -----------------------------
echo "[INFO] Running sanity checks..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
import MultiScaleDeformableAttention as m
print("msda_loaded:", hasattr(m, "ms_deform_attn_forward"))
PY

echo "[DONE] Environment is ready."
echo "[NEXT] Example run:"
echo "python auto_annotate_from_coco_prompt.py \\"
echo "  --prompt-image /path/to/prompt_image.jpg \\"
echo "  --prompt-coco-json /path/to/prompt_instances_default.json \\"
echo "  --target-images-dir /path/to/target_images \\"
echo "  --weights ${WEIGHTS_PATH}"