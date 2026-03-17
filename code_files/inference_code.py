# This script processes satellite images to detect clouds and shadows using a trained U-Net model.
# It performs patch-wise inference on large images, calculates evaluation metrics, and saves results.

# Standard library imports
import os
import time
import math
import datetime

# Third-party imports
import torch
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import rioxarray as rxr
import xarray as xr
import ephem  # For Earth-Sun distance calculations

# ------------------------- CONFIGURATION SETTINGS -------------------------
# Important: Update these paths according to your system setup
TEST_PARENT_FOLDER = "/home/btech1/isro/upload/SampleTestData"  # Root folder for test data
GROUND_TRUTH_MASKS_PATH = "/home/btech1/isro/upload/test_mask"  # Path to ground truth masks
MODEL_PATH = "/home/btech1/isro/training_output/trained_model.pth"  # Trained model weights
OUTPUT_PATH = "/home/btech1/isro/upload/run_inference"  # Output directory for results

# Hardware configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Image processing parameters
PATCH_SIZE = 256  # Size of image patches for processing
STRIDE = 128      # Stride for sliding window (50% overlap)
BANDS = ['BAND2', 'BAND3', 'BAND4']  # Satellite bands to use (Green, Red, NIR)

# ------------------------- MODEL ARCHITECTURE -------------------------
class UNet(nn.Module):
    """
    U-Net model for semantic segmentation of satellite imagery
    Architecture details:
    - Contracting path (encoder) with repeated convolutions + pooling
    - Bottleneck layer at the deepest level
    - Expanding path (decoder) with transposed convolutions + skip connections
    - Final 1x1 convolution for pixel-wise classification
    """
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        # Downsampling path (encoder)
        self.downs = nn.ModuleList()
        prev_channels = in_channels
        for feature in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(prev_channels, feature, 3, padding=1), 
                nn.InstanceNorm2d(feature, affine=True),
                nn.ReLU(),
                nn.Conv2d(feature, feature, 3, padding=1), 
                nn.InstanceNorm2d(feature, affine=True),
                nn.ReLU()
            ))
            prev_channels = feature
        
        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck layer (deepest part of network)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_channels, prev_channels*2, 3, padding=1), 
            nn.InstanceNorm2d(prev_channels*2, affine=True),
            nn.ReLU(),
            nn.Conv2d(prev_channels*2, prev_channels*2, 3, padding=1), 
            nn.InstanceNorm2d(prev_channels*2, affine=True),
            nn.ReLU()
        )
        bottleneck_channels = prev_channels*2
        
        # Upsampling path (decoder)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        decoder_in_channels = bottleneck_channels
        for feature in reversed(features):
            # Transposed convolution for upsampling
            self.ups.append(nn.ConvTranspose2d(decoder_in_channels, feature, kernel_size=2, stride=2))
            # Convolution blocks after concatenation with skip connections
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(feature*2, feature, 3, padding=1), 
                nn.InstanceNorm2d(feature, affine=True),
                nn.ReLU(),
                nn.Conv2d(feature, feature, 3, padding=1), 
                nn.InstanceNorm2d(feature, affine=True),
                nn.ReLU()
            ))
            decoder_in_channels = feature
        
        # Final convolution to produce class probabilities
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        skip_connections = []
        for down in self.downs:
            x = down(x)       # Apply convolution block
            skip_connections.append(x)  # Save for skip connection
            x = self.pool(x)   # Downsample
        
        # Bottleneck (lowest resolution)
        x = self.bottleneck(x)
        
        # Expanding path
        skip_connections = skip_connections[::-1]  # Reverse skip connections
        for i in range(len(self.ups)):
            x = self.ups[i](x)  # Upsample
            skip = skip_connections[i]  # Get corresponding skip connection
            
            # Handle size mismatches (edge cases)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            # Concatenate skip connection
            x = torch.cat((skip, x), dim=1)
            # Apply convolution block
            x = self.up_convs[i](x)
        
        # Final classification layer
        return self.final_conv(x)

# ------------------------- UTILITY FUNCTIONS -------------------------
def read_metadata(scene_path):
    """
    Read metadata from BAND_META.txt file
    Returns: Dictionary of metadata parameters
    """
    meta = {}
    meta_path = os.path.join(scene_path, "BAND_META.txt")
    try:
        with open(meta_path) as f:
            for line in f.readlines():
                if '=' in line:
                    key, value = line.strip().split('=')
                    meta[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Metadata file not found at {meta_path}")
    return meta

def get_earth_sun_distance(date):
    """
    Calculate Earth-Sun distance for atmospheric correction
    using PyEphem astronomy library
    """
    obs = ephem.Observer()
    obs.date = date
    sun = ephem.Sun(obs)
    return sun.earth_distance

def get_esun_values(sat):
    """
    Get solar irradiance values for different bands
    Note: These values are sensor-specific and should be verified
    """
    # Hardcoded values for demonstration - replace with actual sensor values
    return {'B2': 1850.0, 'B3': 1550.0, 'B4': 1040.0}

def get_raster_profile(scene_path, band_name="BAND2"):
    """
    Get raster profile (metadata) from a band file
    """
    path = os.path.join(scene_path, f"{band_name}.tif")
    with rasterio.open(path) as src:
        return src.profile

def read_bands(scene_path):
    """
    Read satellite bands into xarray DataArrays
    Returns: List of bands and raster profile
    """
    band_data = []
    for band in BANDS:
        band_path = os.path.join(scene_path, f"{band}.tif")
        try:
            arr = rxr.open_rasterio(band_path, masked=True)
            
            # Remove band dimension if present
            if "band" in arr.dims:
                arr = arr.squeeze(dim="band")
                
            if not isinstance(arr, xr.DataArray):
                raise RuntimeError(f"{band} is not a valid xarray.DataArray")
                
            band_data.append(arr)
        except FileNotFoundError:
            print(f"Band file not found: {band_path}")
            return None, None
    
    # Get profile from first band
    profile = get_raster_profile(scene_path, BANDS[0])
    return band_data, profile

def dn_to_radiance(arr, lmin, lmax, qcalmax=65535, qcalmin=0):
    """
    Convert Digital Number (DN) to radiance
    Formula: radiance = ((DN - QCALMIN) / (QCALMAX - QCALMIN)) × (LMAX - LMIN) + LMIN
    """
    return ((arr - qcalmin) / (qcalmax - qcalmin)) * (lmax - lmin) + lmin

def radiance_to_reflectance(rad, esun, d_es, sun_elev):
    """
    Convert radiance to top-of-atmosphere reflectance
    Formula: reflectance = (π × radiance) / (ESUN × cos(sun_zenith) × d_es²)
    """
    sun_zenith = np.pi/2 - sun_elev  # Convert elevation to zenith angle
    return (np.pi * rad) / (esun * np.cos(sun_zenith) * d_es**2)

def apply_dark_object_subtraction(refs, wavelengths):
    """
    Apply dark object subtraction for haze correction
    Args:
        refs: List of reflectance arrays [B2, B3, B4]
        wavelengths: Dictionary of band wavelengths
    """
    # Find dark pixels in red band
    red_band = refs[1].values
    # Get 100 smallest non-zero values and take the median
    valid_pixels = red_band[~np.isnan(red_band)]
    dark_value = np.percentile(valid_pixels[valid_pixels > 0], 1) if valid_pixels.size > 0 else 0.01
    
    # Determine haze model based on band characteristics
    if np.nanmedian(refs[2].values) < 0.02:  # NIR is very dark
        model_exponent = -4
    elif np.nanmedian(refs[0].values) > 0.15:  # Green is bright
        model_exponent = -1
    else:
        model_exponent = -2
    
    # Multiplier based on darkness level
    multiplier = 0.8 if dark_value < 0.01 else (1.2 if dark_value > 0.05 else 1.0)
    
    # Calculate haze for each band
    haze_values = []
    for i, band in enumerate(['BAND2', 'BAND3', 'BAND4']):
        ratio = wavelengths[band] / wavelengths['BAND3']
        haze = dark_value * (ratio ** model_exponent) * multiplier
        haze_values.append(haze)
    
    # Subtract haze and clip to valid range
    corrected = [(ref - h).clip(0, 1.0) for ref, h in zip(refs, haze_values)]
    return corrected

def compute_metrics(ground_truth, prediction):
    """
    Compute evaluation metrics for cloud/shadow detection
    Returns: Tuple of metrics (cloud%, shadow%, IoU, precision, recall, F1, accuracy)
    """
    # Create boolean masks for different classes
    cloud_gt = (ground_truth == 1)
    shadow_gt = (ground_truth == 2)
    cloud_pred = (prediction == 1)
    shadow_pred = (prediction == 2)
    
    # Helper function for metric calculations
    def iou(pred_mask, gt_mask):
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return intersection / (union + 1e-6)  # Avoid division by zero
    
    def precision(pred_mask, gt_mask):
        true_positives = np.logical_and(pred_mask, gt_mask).sum()
        return true_positives / (pred_mask.sum() + 1e-6)
    
    def recall(pred_mask, gt_mask):
        true_positives = np.logical_and(pred_mask, gt_mask).sum()
        return true_positives / (gt_mask.sum() + 1e-6)
    
    def f1_score(prec, rec):
        return 2 * prec * rec / (prec + rec + 1e-6)
    
    # Calculate metrics for clouds
    cloud_precision = precision(cloud_pred, cloud_gt)
    cloud_recall = recall(cloud_pred, cloud_gt)
    cloud_f1 = f1_score(cloud_precision, cloud_recall)
    cloud_iou = iou(cloud_pred, cloud_gt)
    
    # Overall accuracy
    overall_accuracy = (ground_truth == prediction).sum() / ground_truth.size
    
    # Percentage coverage
    cloud_percentage = cloud_gt.sum() / ground_truth.size * 100
    shadow_percentage = shadow_gt.sum() / ground_truth.size * 100
    
    return (cloud_percentage, shadow_percentage, cloud_iou, 
            cloud_precision, cloud_recall, cloud_f1, overall_accuracy)

def save_shapefiles(mask, transform, crs, out_dir, scene_id):
    """
    Save detected features as shapefiles
    Generates separate files for clouds and shadows
    """
    # Class labels and corresponding filenames
    classes = {
        1: ("cloudshapes", "Cloud polygons"),
        2: ("shadowshapes", "Shadow polygons")
    }
    
    for label, (filename, desc) in classes.items():
        # Extract shapes from raster mask
        shape_gen = shapes(mask, mask == label, transform=transform)
        
        # Convert to GeoDataFrame
        records = [{"geometry": shape(geom), "value": value} for geom, value in shape_gen]
        
        if records:
            gdf = gpd.GeoDataFrame(records, geometry=[r["geometry"] for r in records])
            gdf.set_crs(crs, inplace=True)
            
            # Save to shapefile
            output_path = os.path.join(out_dir, f"{filename}_{scene_id}.shp")
            gdf.to_file(output_path)
            print(f"Saved {desc} to {output_path}")
        else:
            print(f"No {desc} found for {scene_id}")

# ------------------------- INFERENCE CORE -------------------------
def predict_scene_patchwise(reflectance_bands, profile):
    """
    Perform patch-wise prediction on large satellite image
    Uses sliding window approach with overlap and averages predictions
    """
    height, width = profile['height'], profile['width']
    
    # Initialize accumulation arrays
    probability_accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    
    # Transformation for input data
    to_tensor = transforms.ToTensor()

    # Sliding window over the image
    for y in tqdm(range(0, height, STRIDE), desc='Processing image patches'):
        for x in range(0, width, STRIDE):
            # Extract patch from each band
            patch_data = []
            for band in reflectance_bands:
                # Get patch with boundary checks
                band_patch = band.isel(
                    x=slice(x, x + PATCH_SIZE),
                    y=slice(y, y + PATCH_SIZE)
                ).values
                patch_data.append(band_patch)
            
            # Stack bands along channel dimension
            raw_patch = np.stack(patch_data, axis=-1).astype(np.float32)
            patch_height, patch_width = raw_patch.shape[:2]
            
            # Pad patch if smaller than required size
            if patch_height < PATCH_SIZE or patch_width < PATCH_SIZE:
                padded_patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
                padded_patch[:patch_height, :patch_width, :] = raw_patch
                current_patch = padded_patch
            else:
                current_patch = raw_patch
            
            # Convert to tensor and add batch dimension
            input_tensor = to_tensor(current_patch).unsqueeze(0).to(DEVICE)
            
            # Model prediction
            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            
            # Accumulate predictions only in the valid region
            for channel in range(3):
                probability_accumulator[y:y+patch_height, x:x+patch_width, channel] += (
                    probabilities[channel, :patch_height, :patch_width]
                )
            weight_map[y:y+patch_height, x:x+patch_width] += 1
    
    # Normalize by overlap count
    weight_map[weight_map == 0] = 1  # Avoid division by zero
    average_probabilities = probability_accumulator / weight_map[..., np.newaxis]
    
    # Final classification (0=clear, 1=cloud, 2=shadow)
    final_mask = np.argmax(average_probabilities, axis=-1).astype(np.uint8)
    return final_mask

# ------------------------- MAIN EXECUTION -------------------------
def main():
    """
    Main processing pipeline:
    1. Load trained model
    2. Process each satellite scene
    3. Perform atmospheric correction
    4. Run inference
    5. Evaluate results
    6. Save outputs
    """
    # Load model weights
    print(f"Loading model from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Prepare output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Outputs will be saved to {OUTPUT_PATH}")
    
    # Collect all scene paths
    scenes = []
    for date_folder in os.listdir(TEST_PARENT_FOLDER):
        date_path = os.path.join(TEST_PARENT_FOLDER, date_folder)
        if os.path.isdir(date_path):
            for scene_folder in os.listdir(date_path):
                scene_path = os.path.join(date_path, scene_folder)
                if os.path.isdir(scene_path):
                    scenes.append(scene_path)
    
    print(f"Found {len(scenes)} scenes to process")
    
    # Process each scene
    results = []
    for idx, scene_path in enumerate(tqdm(scenes, desc="Processing scenes")):
        start_time = time.time()
        scene_id = os.path.basename(scene_path)
        print(f"\nProcessing scene {idx+1}/{len(scenes)}: {scene_id}")
        
        # Read metadata
        metadata = read_metadata(scene_path)
        if not metadata:
            print(f"Skipping {scene_id} due to missing metadata")
            continue
        
        # Parse acquisition date
        try:
            acquisition_date = datetime.datetime.strptime(metadata['DateOfPass'], '%d-%b-%Y')
            earth_sun_distance = get_earth_sun_distance(acquisition_date)
            sun_elevation = math.radians(float(metadata['SunElevationAtCenter']))
        except (KeyError, ValueError) as e:
            print(f"Error processing metadata for {scene_id}: {str(e)}")
            continue
        
        # Load band data
        digital_number_bands, raster_profile = read_bands(scene_path)
        if digital_number_bands is None:
            continue
        
        # Atmospheric correction =======================================
        # Step 1: Convert DN to Radiance
        radiance_bands = []
        for i, band in enumerate(BANDS):
            lmin = float(metadata[f'B{i+2}_Lmin'])
            lmax = float(metadata[f'B{i+2}_Lmax'])
            radiance = dn_to_radiance(
                digital_number_bands[i], 
                lmin, 
                lmax
            )
            radiance_bands.append(radiance)
        
        # Step 2: Convert Radiance to Reflectance
        esun_values = get_esun_values(metadata['SatID'])
        reflectance_bands = []
        for i, rad in enumerate(radiance_bands):
            band_key = f'B{i+2}'
            reflectance = radiance_to_reflectance(
                rad,
                esun_values[band_key],
                earth_sun_distance,
                sun_elevation
            )
            reflectance_bands.append(reflectance)
        
        # Step 3: Haze correction
        band_wavelengths = {'BAND2': 0.555, 'BAND3': 0.650, 'BAND4': 0.815}
        corrected_bands = apply_dark_object_subtraction(reflectance_bands, band_wavelengths)
        # =============================================================
        
        # Run inference
        prediction_mask = predict_scene_patchwise(corrected_bands, raster_profile)
        
        # Save prediction mask
        output_raster = os.path.join(OUTPUT_PATH, f"mask_{scene_id}.tif")
        with rasterio.open(output_raster, 'w', **raster_profile) as dst:
            dst.write(prediction_mask.astype(np.uint8), 1)
        print(f"Saved prediction mask to {output_raster}")
        
        # Evaluate against ground truth
        gt_mask_path = os.path.join(GROUND_TRUTH_MASKS_PATH, f"mask_{scene_id}.tif")
        if os.path.exists(gt_mask_path):
            with rasterio.open(gt_mask_path) as src:
                ground_truth_mask = src.read(1)
            
            metrics = compute_metrics(ground_truth_mask, prediction_mask)
            print(f"Metrics for {scene_id}: "
                  f"Clouds={metrics[0]:.2f}%, "
                  f"Shadows={metrics[1]:.2f}%, "
                  f"IoU={metrics[2]:.3f}, "
                  f"Accuracy={metrics[6]:.3f}")
            
            # Save shapefiles
            save_shapefiles(
                prediction_mask,
                raster_profile['transform'],
                raster_profile['crs'],
                OUTPUT_PATH,
                scene_id
            )
            
            # Record results
            processing_time = time.time() - start_time
            results.append([
                idx+1, scene_id, metrics[0], metrics[1], metrics[2], 
                metrics[3], metrics[4], metrics[5], metrics[6], 
                processing_time
            ])
        else:
            print(f"No ground truth found for {scene_id}")
    
    # Save results to CSV
    results_df = pd.DataFrame(
        results,
        columns=[
            "S.No.", "DatasetId", "%Cloud", "%Shadow", "IoU", 
            "Precision", "Recall", "F1", "Accuracy", "Time_sec"
        ]
    )
    results_csv = os.path.join(OUTPUT_PATH, "results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Saved evaluation results to {results_csv}")

# ------------------------- ENTRY POINT -------------------------
if __name__ == "__main__":
    # Initialize model
    model = UNet(in_channels=3, out_channels=3)  # 3 classes: background, cloud, shadow
    main()