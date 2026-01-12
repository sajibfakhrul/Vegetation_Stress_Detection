

# Vegetation Stress Detection Using Deep Learning (Sentinel-2)

This project focuses on detecting and quantifying **vegetation stress** in the **Coconino National Forest** using **Sentinel-2 multispectral satellite imagery** and deep learning models. The goal is to identify areas of declining vegetation health by analyzing changes in spectral reflectance and vegetation indices over time, and to evaluate how well modern neural networks can learn these patterns.

We frame the problem in two complementary ways:

1. **Binary classification** – predicting whether a patch of land is **healthy** or **stressed**.  
2. **Continuous regression** – predicting a **stress severity score** representing the fraction of stressed vegetation within each patch.

The project combines remote-sensing preprocessing in **Google Earth Engine** with deep-learning pipelines in **TensorFlow** and **PyTorch**.

---

## Study Area and Data

- **Region:** Coconino National Forest, Arizona, USA  
- **Satellite source:** Sentinel-2 Surface Reflectance  
  - Dataset: `COPERNICUS/S2_SR_HARMONIZED`  
- **Time periods analyzed:**  
  - May–August **2023**  
  - May–August **2025**

### Spectral Bands
For each month, the following bands are used:
- **B2** – Blue  
- **B3** – Green  
- **B4** – Red  
- **B8** – Near-infrared (NIR)

### Vegetation Index
- **NDVI** is computed for every month:
  \[
  NDVI = \frac{NIR - Red}{NIR + Red}
  \]

---

## Labeling Strategy: Stress Proxy from NDVI Change

Because large-scale field labels are not available, this project uses a **relative vegetation-change proxy** derived from NDVI.

### Step 1: Cloud and land masking
- Clouds and cirrus removed using **QA60** bits.
- Non-vegetated pixels filtered using the **SCL** land classes.

### Step 2: Monthly composites
- For each month, a **median composite** is created.
- Reflectance values are scaled to the range [0, 1].

### Step 3: Mean NDVI comparison
- Mean NDVI is computed for:
  - May–Aug 2023  
  - May–Aug 2025  

### Step 4: NDVI ratio
A per-pixel ratio map is created:

\[
R = \frac{NDVI_{2025}}{NDVI_{2023}}
\]

Pixels with very low baseline NDVI are masked to avoid unstable ratios.

### Step 5: Dynamic thresholding
- A **single regional threshold** is computed as:
  \[
  T = \frac{\text{mean NDVI}_{2025}}{\text{mean NDVI}_{2023}}
  \]
- Pixels are labeled as **stressed** if:
  \[
  R < T
  \]

This produces a spatially detailed **stress proxy map** representing relative vegetation decline between 2023 and 2025.

---

## Patch Generation and Dataset Construction

### Patch tiling
- The forest is tiled into **64 × 64 pixel patches** at **10 m resolution**.
- Each patch covers approximately **640 m × 640 m**.

### Feature stack
Each patch contains **40 input channels**:
- **2023 spectral:** 16 bands (4 months × 4 bands)  
- **2023 NDVI:** 4 bands  
- **2025 spectral:** 16 bands  
- **2025 NDVI:** 4 bands  

Plus **1 label band** derived from the stress proxy.

### Patch-level labels
For each patch:
- The fraction of stressed pixels is computed.
- **Binary label**
  - Healthy = 0  
  - Stressed = 1  
  - Based on majority of pixels.
- **Regression label**
  - Continuous value in [0, 1] representing stress severity.

### Purity filtering
To reduce ambiguity:
- **Healthy patches:** at least 60% healthy pixels.  
- **Stressed patches:** at least 60% stressed pixels.  

Only high-confidence patches are exported for training.

---

## Models

Three deep-learning approaches are evaluated.

### 1. Convolutional Neural Network (CNN)
- Input: 64 × 64 × 40  
- Tasks:
  - Binary classification
  - Stress-severity regression  
- Uses convolutional blocks with global average pooling.

### 2. Vision Transformer (ViT)
- Image split into **8 × 8 patches** → 64 tokens.
- Transformer encoder learns long-range spatial relationships.
- Supports both classification and regression heads.

### 3. DINOv2 (Transfer Learning)
- Based on the **DINOv2 foundation model**.
- Modified to:
  - Accept **40-channel input**.
  - Resize positional embeddings for 64-token grids.
- Fine-tuned for:
  - Binary stress classification.
  - Continuous stress regression.

---

## Results

### Classification Performance (Test Set)

| Model   | Accuracy | Precision | Recall | F1-score |
|---------|----------|-----------|--------|----------|
| CNN     | 0.967    | 0.952     | 0.983  | 0.967    |
| ViT     | 0.925    | 0.918     | 0.933  | 0.926    |
| DINOv2  | 0.9417   | 0.9206    | 0.9667 | 0.9431   |

### Regression Performance (Stress Severity)

| Model   | RMSE | MAE  | R²   |
|---------|------|------|------|
| CNN     | 0.082| 0.066| 0.95 |
| ViT     | ~0.164| —   | 0.81 |
| DINOv2  | 0.159| —   | 0.82 |

These results show that both convolutional and transformer-based models can effectively learn vegetation-stress patterns from multi-temporal Sentinel-2 data, with CNN achieving the strongest overall performance.

---

### Google Earth Engine script
https://code.earthengine.google.com/f9a27de36dc038d0190217b066d2e3a6

---

### Acknowledgements

Sentinel-2 data provided by Copernicus via Google Earth Engine.

DINOv2 model by Meta AI Research.

Open-source tools: TensorFlow, PyTorch, scikit-learn, rasterio, and geemap.

---
