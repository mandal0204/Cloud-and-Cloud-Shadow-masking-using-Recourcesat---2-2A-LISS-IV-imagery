# Cloud-and-Cloud-Shadow-masking-using-Recourcesat-2-2A-LISS-IV-imagery

This repository provides a complete pipeline for detecting and masking clouds and cloud shadows in Resourcesat-2/2A LISS-IV satellite imagery. The solution uses a U-Net deep learning model for semantic segmentation, implemented in PyTorch. The workflow covers data preprocessing, model training, patch-based inference on large satellite scenes, and post-training analysis.

## Features

*   **Atmospheric Correction**: Converts raw Digital Numbers (DN) to Top-of-Atmosphere (TOA) reflectance, including a Dark Object Subtraction (DOS) method for haze removal.
*   **Deep Learning Model**: Utilizes a U-Net architecture for segmenting images into three classes: background, cloud, and cloud shadow.
*   **Efficient Data Handling**: Creates image patches from large satellite scenes and stores them in an LMDB (Lightning Memory-Mapped Database) for efficient training.
*   **Patch-Based Inference**: Implements a sliding window approach with overlap to perform inference on full-sized satellite images, minimizing edge artifacts.
*   **Comprehensive Training Pipeline**: Includes class weighting for imbalanced datasets, learning rate scheduling (`ReduceLROnPlateau`), early stopping, and detailed metric logging.
*   **In-depth Evaluation**: Generates a suite of performance metrics including IoU, Precision, Recall, F1-Score, and Accuracy for both overall and per-class performance.
*   **Rich Visualization**: Produces a variety of plots for training analysis, such as loss curves, metric progression, confusion matrices, precision-recall curves, and segmentation heatmaps.
*   **Output Generation**: Saves final prediction masks as GeoTIFFs and generates vector shapefiles for the detected cloud and shadow polygons.

## Workflow

The project is structured into three main stages: data preprocessing, model training, and inference.

### 1. Data Preprocessing

The preprocessing step converts raw satellite data into analysis-ready TOA reflectance imagery.

*   **Script**: `code_files/step_1.py`
*   **Process**:
    1.  Reads scene-specific metadata from `BAND_META.txt`.
    2.  Calculates the Earth-Sun distance based on the acquisition date.
    3.  Converts Digital Number (DN) values from BAND2 (Green), BAND3 (Red), and BAND4 (NIR) to at-sensor radiance.
    4.  Converts radiance values to Top-of-Atmosphere (TOA) reflectance.
    5.  Applies a Dark Object Subtraction (DOS) algorithm to correct for atmospheric haze.
    6.  Saves the atmospherically corrected 3-band image as a GeoTIFF.

### 2. Model Training

This stage trains the U-Net model on the preprocessed data.

*   **Script**: `code_files/train_model.py`
*   **Process**:
    1.  Generates patches from the training images and corresponding masks and writes them to an LMDB for fast data loading.
    2.  Initializes the U-Net model, Adam optimizer, and `ReduceLROnPlateau` learning rate scheduler.
    3.  Trains the model using `CrossEntropyLoss` with weights to handle class imbalance.
    4.  At each epoch, it calculates and logs detailed performance metrics (Accuracy, IoU, Precision, Recall, F1-Score) for both the training and a subset of the validation set.
    5.  Saves the best model checkpoint based on validation IoU and implements early stopping to prevent overfitting.
    6.  The final trained model is saved as `final_model.pth`.

### 3. Training Analysis and Visualization

After training, various plots are generated to analyze model performance.

*   **Script**: `code_files/plots.py`
*   **Process**:
    1.  Loads the training metrics from `metrics_log.csv`.
    2.  Generates and saves plots for loss, learning rate, and all evaluation metrics (Accuracy, IoU, etc.) over epochs.
    3.  Loads the trained model to generate a confusion matrix, precision-recall curves, and example segmentation heatmaps on validation data.
    4.  Implements Grad-CAM to visualize the class activation maps, providing insight into which image features the model focuses on.

### 4. Inference

This stage uses the trained model to generate cloud and shadow masks for new, unseen satellite scenes.

*   **Script**: `code_files/inference_code.py`
*   **Process**:
    1.  Loads the trained U-Net model (`trained_model.pth`).
    2.  Iterates through new satellite scenes in the specified test data folder.
    3.  Applies the same atmospheric correction pipeline used during preprocessing.
    4.  Performs patch-wise inference on the full scene using a sliding window (`256x256` patches with `50%` overlap). The predictions for overlapping regions are averaged to produce a smooth final mask.
    5.  Saves the full prediction mask as a GeoTIFF.
    6.  If ground truth masks are available, it computes evaluation metrics and saves them to `results.csv`.
    7.  Generates and saves shapefiles (.shp) containing polygons for the detected clouds and shadows.

## How to Use

### Setup

1.  Clone the repository.
2.  Install the required Python packages, including `torch`, `torchvision`, `rasterio`, `rioxarray`, `geopandas`, `ephem`, and `tqdm`.
3.  Update the file paths at the top of each script (`step_1.py`, `train_model.py`, `inference_code.py`, `plots.py`) to match your system's directory structure.

### Training the Model

1.  Place your raw LISS-IV training images and corresponding masks into the directories specified in `train_model.py`. The script expects images in subfolders (e.g., `.../train/images/`) and masks in (`.../train/masks/`).
2.  Run the training script:
    ```bash
    python code_files/train_model.py
    ```
3.  The script will first create LMDB patch databases if they don't exist, then start the training process. The best model and a final model will be saved in the specified `output_dir`.

### Running Inference

1.  Place your trained model (e.g., `final_model.pth`) in the location specified by `MODEL_PATH` in `inference_code.py`.
2.  Organize your test satellite data into the folder specified by `TEST_PARENT_FOLDER`.
3.  Run the inference script:
    ```bash
    python code_files/inference_code.py
    ```
4.  The script will process each scene and save the output masks (GeoTIFFs), shapefiles, and an evaluation summary (`results.csv`) to the `OUTPUT_PATH` directory.

## Repository Structure

```
.
├── code_files/
│   ├── step_1.py              # Preprocessing: DN to TOA reflectance
│   ├── train_model.py         # Main training script with LMDB creation
│   ├── inference_code.py      # Inference script for new scenes
│   ├── plots.py               # Generates all post-training analysis plots
│   ├── model_code.py          # U-Net model definition
│   └── test_model.py          # Alternate training/testing script
├── debug_code_files/          # Jupyter notebooks and scripts for development and debugging
├── inference.py/
│   └── predict.py             # Copy of the main inference script
├── models/
│   └── unet.py                # Copy of the U-Net model definition
└── training/
    └── train.py               # Copy of the main training script
