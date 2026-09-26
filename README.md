# PlantVillage Disease Classification

An end-to-end image classification project on the **PlantVillage** leaf-disease dataset, built for **Winter in Data Science (WiDS) 5.0** — Analytics Club IIT Bombay's mentorship program. The project moves through the full pipeline: exploratory data analysis, classical ML baselines, and a deep learning model, with a companion technical deep-dive on why data quality — not architecture — is usually what breaks detection/segmentation pipelines in production.

## Pipeline

### 1. Exploratory data analysis (`01-eda-plantvillage-ipynb.ipynb`, `refactored-eda-plantvillage-ipynb.ipynb`)
- Builds a tabular index of the dataset by walking class subfolders (color, grayscale, segmented variants)
- Quantifies class imbalance (bar charts, pie charts of the top classes by frequency)
- Analyzes image size/resolution spread across classes in 2D/3D and as frequency distributions
- Summarizes per-image mean RGB intensity to check lighting/color consistency across the dataset, and validates resolution consistency per class via box plots

### 2. Classical ML baselines (`02_Baseline_Comparison_PlantVillage.ipynb`)
Establishes how far non-deep methods get on flattened pixel features before reaching for a CNN:
- Most-frequent-class baseline as a sanity floor
- **Linear SVM** on standardized pixel vectors, and on **PCA-reduced** features once full-dimensional training proved computationally prohibitive at ~43k samples × 12k dimensions
- **Random Forest**, both on raw standardized features and on PCA-compressed features, to probe the ceiling of tree-based, non-linear classical methods
- **RBF-kernel SVM** on a stratified subsample, demonstrating where kernel methods stop scaling

### 3. Deep learning (`03_Deep_Learning.ipynb`)
- A **CNN trained from scratch** (224×224 input, on-disk batch loading, [0,1] normalization, augmentation) to establish what convolutional feature learning buys over the classical baselines
- **Transfer learning with MobileNetV2** (frozen ImageNet backbone) fine-tuned on the disease classes
- Early stopping and learning-rate reduction callbacks for stable training
- Confusion-matrix analysis to identify which disease classes are most frequently confused, and how class imbalance affects them

### 4. Technical writeup (`docs/data-analysis-&-augmentation.md`)
A first-principles deep dive into *why* object detection and segmentation datasets fail more often from data issues than architecture choices — covering spatial binding constraints, how transformations (resize/crop/rotate/flip) must be propagated to annotations, and practical diagnostic workflows for validating detection/segmentation datasets before training.

## Stack

`TensorFlow/Keras` · `scikit-learn` · `pandas` / `numpy` · `matplotlib`

## Running it

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
jupyter notebook 01-eda-plantvillage-ipynb.ipynb
```

Notebooks expect the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) (color/grayscale/segmented variants) available locally or via a Kaggle-mounted path.
