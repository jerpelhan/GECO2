# GeCo2 (Generalized-Scale Object Counting with Gradual Query Aggregation)

> Official repository of **GeCo2**  
> 📄 Read the paper: [GeCo2 PDF](https://prints.vicos.si/publications/files/461)

---
<img width="2097" height="587" alt="Geco2_architevture-1" src="https://github.com/user-attachments/assets/88d27ee8-e84e-409a-a87d-095ca24e8a89" />

## Abstract

Few-shot detection-based counters estimate the number of category instances in an image using only a few test-time exemplars. Existing methods often rely on ad-hoc image upscaling and tiling to detect small, densely packed objects, and they struggle when object sizes vary widely within a single image.  
**GeCo2** introduces a **generalized-scale dense query map** that is **gradually aggregated across multiple backbone resolutions**. Scale-specific query encoders interact with exemplar **appearance** and **shape** prototypes at each feature level and then fuse them into a high-resolution query map for detection. This avoids heuristic upscaling/tiling, **improves counting and detection accuracy**, and **reduces memory and runtime**. A lightweight SAM2-based mask refinement further polishes box quality.  
On standard few-shot counting/detection benchmarks, GeCo2 achieves strong gains in **MAE/RMSE** and **AP/AP50**, while running **~3× faster** with a smaller GPU footprint.

---

## Highlights

<img width="2575" height="912" alt="GECO2_first_image_motivation_neurips-1" src="https://github.com/user-attachments/assets/adf4dcfd-aa17-4cff-9113-8b8a0e37de31" />


- 🔁 **Gradual cross-scale query aggregation** → one high-res dense query map without tiling.  
- 🧩 **Per-scale exemplar interaction** with **appearance** + **shape** prototypes.  
- ⚡ **Fast & memory-efficient** inference.  
- 📈 Strong results on **FSCD147**, **FSCD-LVIS**, and **MCAC** (few-shot & multi-class).
<img width="1832" height="2661" alt="GeCoV2Qualitative_segmentation-1" src="https://github.com/user-attachments/assets/8797ada0-e8a7-4e4c-8967-4ebbb365f63f" />

---

## News

- 2026-XX-XX — Paper under submission.  
- 2026-XX-XX — Code release planned.

---

## Repository structure

