# GeCo2 - Generalized-Scale Object Counting with Gradual Query Aggregation

> Official repository of **GeCo2**  
> 🏆 **Accepted to AAAI 2026**  
> 📄 Read the paper: [GeCo2 PDF](https://arxiv.org/pdf/2511.08048)

---


## Abstract

Few-shot detection-based counters estimate the number of category instances in an image using only a few test-time exemplars. Existing methods often rely on ad-hoc image upscaling and tiling to detect small, densely packed objects, and they struggle when object sizes vary widely within a single image.  **GeCo2** introduces a generalized-scale dense query map that is gradually aggregated across multiple backbone resolutions. Scale-specific query encoders interact with exemplar appearance and shape prototypes at each feature level and then fuse them into a high-resolution query map for detection. This avoids heuristic upscaling/tiling, improves counting and detection accuracy, and reduces memory and runtime. A lightweight SAM2-based mask refinement further polishes box quality.  On standard few-shot counting/detection benchmarks, GeCo2 achieves strong gains in MAE/RMSE and AP/AP50, while running ~3× faster with a smaller GPU footprint.



## Live Demo
Try the interactive demo on Hugging Face:  
👉 [DEMO HERE](https://huggingface.co/spaces/jerpelhan/GECO2-demo)


https://github.com/user-attachments/assets/8b5f3f06-45f8-439f-9333-b7a747db28a5




## Highlights
<img width="2575" height="912" alt="GECO2_first_image_motivation_neurips-1" src="https://github.com/user-attachments/assets/adf4dcfd-aa17-4cff-9113-8b8a0e37de31" />

- 🔁 **Gradual cross-scale query aggregation** → one high-res dense query map without tiling.  
- 🧩 **Per-scale exemplar interaction** with **appearance** + **shape** prototypes.  
- ⚡ **Fast & memory-efficient** inference.  
- 📈 Strong results on **FSCD147**, **FSCD-LVIS**, and **MCAC** (few-shot & multi-class).


<img width="2097" height="587" alt="Geco2_architevture-1" src="https://github.com/user-attachments/assets/88d27ee8-e84e-409a-a87d-095ca24e8a89" />


## Demo Installation

You can easily install and run the demo using the provided `install.sh` script.

```bash
bash install.sh
```

#### Download Weights

Download the model weights from:

👉 [CNTQG_multitrain_ca44.pth](https://huggingface.co/datasets/jerpelhan/geco2-assets/resolve/main/weights/CNTQG_multitrain_ca44.pth?download=true)

and place the file in the **project root directory**.

#### Launch the Demo

Then run:

```bash
python demo_gradio.py
```
---
<img width="1832" height="2661" alt="GeCoV2Qualitative_segmentation-1" src="https://github.com/user-attachments/assets/8797ada0-e8a7-4e4c-8967-4ebbb365f63f" />

---

    
## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{pelhan2026generalized,
  title={Generalized-Scale Object Counting with Gradual Query Aggregation},
  author={Pelhan, Jer and Lukezic, Alan and Kristan, Matej},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={},
  year={2026}
}
```
