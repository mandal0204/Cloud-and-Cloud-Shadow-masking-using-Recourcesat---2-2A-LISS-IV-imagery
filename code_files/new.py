# Automated Inference Pipeline for Cloud & Shadow Detection

import os
import time
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
import math
import ephem
import rioxarray as rxr
import xarray as xr
from rasterio.env import Env
import datetime
import concurrent.futures

# ---------------- USER CONFIG ----------------
MODEL_PATH = "/home/btech1/isro/training_output/trained_model.pth"
TEST_PARENT_FOLDER = "/home/btech1/isro/upload/SampleTestData"
GROUND_TRUTH_MASKS_PATH = "/home/btech1/isro/upload/test_mask"
OUTPUT_PATH = "/home/btech1/isro/upload/run_inference"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 256
STRIDE = 128  # 50% overlap
BANDS = ['BAND2', 'BAND3', 'BAND4']


# ---------------- MODEL ----------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64,128,256,512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        prev_channels = in_channels
        for feature in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(prev_channels, feature, 3, padding=1), 
                nn.InstanceNorm2d(feature, affine=True),
                nn.ReLU(),
                nn.Conv2d(feature, feature, 3, padding=1), 
                nn.InstanceNorm2d(feature, affine=True),
                nn.ReLU()))
            prev_channels = feature
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(prev_channels, prev_channels*2, 3, padding=1), 
            nn.InstanceNorm2d(prev_channels*2, affine=True),
            nn.ReLU(),
            nn.Conv2d(prev_channels*2, prev_channels*2, 3, padding=1), 
            nn.InstanceNorm2d(prev_channels*2, affine=True),
            nn.ReLU())
        bottleneck_channels = prev_channels*2
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        decoder_in_channels = bottleneck_channels
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(decoder_in_channels, feature, kernel_size=2, stride=2))
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(feature*2, feature, 3, padding=1), 
                nn.InstanceNorm2d(feature, affine=True),
                nn.ReLU(),
                nn.Conv2d(feature, feature, 3, padding=1), 
                nn.InstanceNorm2d(feature, affine=True),
                nn.ReLU()))
            decoder_in_channels = feature
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((skip, x), dim=1)
            x = self.up_convs[i](x)
        return self.final_conv(x)

# ---------------- UTILITIES ----------------
def read_metadata(scene_path):
    with open(os.path.join(scene_path, "BAND_META.txt")) as f:
        lines = f.readlines()
    meta = {}
    for l in lines:
        if '=' in l:
            k, v = l.strip().split('=')
            meta[k.strip()] = v.strip()
    return meta

def get_earth_sun_distance(date):
    obs = ephem.Observer(); obs.date = date
    sun = ephem.Sun(obs); return sun.earth_distance

def get_esun_values(sat):
    # Example hardcoded values; replace with yours
    return {'B2': 1850.0, 'B3': 1550.0, 'B4': 1040.0}

import rasterio

def get_raster_profile(scene_path, band_name="BAND2"):
    path = os.path.join(scene_path, f"{band_name}.tif")
    with rasterio.open(path) as src:
        profile = src.profile
    return profile


def read_bands(scene_path):
    bands = []
    for band in BANDS:
        path = os.path.join(scene_path, f"{band}.tif")
        arr = rxr.open_rasterio(path, masked=True)

        # Ensure 2D
        if "band" in arr.dims:
            arr = arr.squeeze(dim="band")

        if not isinstance(arr, xr.DataArray):
            raise RuntimeError(f"{band} is not a valid xarray.DataArray")

        bands.append(arr)

    profile = get_raster_profile(scene_path, BANDS[0])
    return bands, profile

def dn_to_radiance(arr, lmin, lmax, qcalmax=65535, qcalmin=0):
    return ((arr - qcalmin) / (qcalmax - qcalmin)) * (lmax - lmin) + lmin

def radiance_to_reflectance(rad, esun, d_es, sun_elev):
    return (np.pi * rad) / (esun * np.cos(np.pi/2 - sun_elev) * d_es**2)

def apply_dark_object_subtraction(refs, wl):
    red = refs[1].values
    dark = next((v for v in np.unique(red[~np.isnan(red)])[:100] if v > 0), 0.01)
    model = -4 if np.nanmedian(refs[2].values) < 0.02 else (-1 if np.nanmedian(refs[0].values) > 0.15 else -2)
    mult = 0.8 if dark < 0.01 else (1.2 if dark > 0.05 else 1.0)
    haze = [dark * ((wl[f'BAND{i+2}']/wl['BAND3'])**model) * mult for i in range(3)]
    return [(ref - h).clip(0,1.0) for ref, h in zip(refs, haze)]

def compute_metrics(gt, pred):
    classes = [1, 2]  # CLOUD, SHADOW
    total = gt.size
    summary = {}

    for cls in classes:
        pred_mask = (pred == cls)
        gt_mask = (gt == cls)
        tp = np.logical_and(pred_mask, gt_mask).sum()
        fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
        fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = tp / (union + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        summary[cls] = {"IoU": iou, "Precision": precision, "Recall": recall, "F1": f1}

    acc = (gt == pred).sum() / total
    pc_cloud = (gt == 1).sum() / total * 100
    pc_shadow = (gt == 2).sum() / total * 100
    return pc_cloud, pc_shadow, acc, summary

# ---------------- SHAPEFILE SAFE WRITER ----------------
def save_shapefiles(mask, transform, crs, out_dir, scene_id):
    for label, name in [(1, "cloudshapes"), (2, "shadowshapes")]:
        gen = shapes(mask, mask == label, transform=transform)
        records = [{"geometry": shape(g), "value": v} for g, v in gen]
        if records:
            gdf = gpd.GeoDataFrame(records, geometry=[r["geometry"] for r in records])
            gdf.set_crs(crs, inplace=True)
            out_path = os.path.join(out_dir, f"{name}_{scene_id}.gpkg")
            gdf.to_file(out_path, driver="GPKG")  # GPKG avoids .shp file size limits

# ---------------- PATCH-WISE INFERENCE WITH OVERLAP ----------------
def predict_scene_patchwise(reflectance_bands, profile, model, tf):
    height, width = profile['height'], profile['width']
    prob_accumulator = np.zeros((height, width, 3), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)

    for i in range(0, height, STRIDE):
        for j in range(0, width, STRIDE):
            raw_patch = []
            for band in reflectance_bands:
                sub = band.isel(x=slice(j, j+PATCH_SIZE), y=slice(i, i+PATCH_SIZE)).values
                raw_patch.append(sub)
            raw_patch = np.stack(raw_patch, axis=-1).astype(np.float32)
            ph, pw = raw_patch.shape[:2]

            if ph != PATCH_SIZE or pw != PATCH_SIZE:
                patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
                patch[:ph, :pw, :] = raw_patch
            else:
                patch = raw_patch

            input_tensor = tf(patch).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(input_tensor)
                pred = torch.softmax(pred, dim=1).squeeze().cpu().numpy()

            for c in range(3):
                prob_accumulator[i:i+ph, j:j+pw, c] += pred[c, :ph, :pw]
            weight_map[i:i+ph, j:j+pw] += 1

    weight_map[weight_map == 0] = 1
    averaged_probs = prob_accumulator / weight_map[..., None]
    final_mask = np.argmax(averaged_probs, axis=-1).astype(np.uint8)
    return final_mask

# ---------------- MAIN ----------------
def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    metrics_rows = []

    model = UNet(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    tf = transforms.ToTensor()

    scene_dirs = [ os.path.join(TEST_PARENT_FOLDER, f, sf) 
                   for f in os.listdir(TEST_PARENT_FOLDER) 
                   for sf in os.listdir(os.path.join(TEST_PARENT_FOLDER, f)) ]

    for idx, scene_path in enumerate(tqdm(scene_dirs, desc="Scenes")):
        t0 = time.time()
        metadata = read_metadata(scene_path)
        scene_id = metadata['OTSProductID']

        date = datetime.datetime.strptime(metadata['DateOfPass'], '%d-%b-%Y')
        d_es = get_earth_sun_distance(date)
        esun = get_esun_values(metadata['SatID'])
        sun_elev = math.radians(float(metadata['SunElevationAtCenter']))

        dn_bands, profile = read_bands(scene_path)
        b2 = radiance_to_reflectance(dn_to_radiance(dn_bands[0], float(metadata['B2_Lmin']), float(metadata['B2_Lmax'])), esun['B2'], d_es, sun_elev)
        b3 = radiance_to_reflectance(dn_to_radiance(dn_bands[1], float(metadata['B3_Lmin']), float(metadata['B3_Lmax'])), esun['B3'], d_es, sun_elev)
        b4 = radiance_to_reflectance(dn_to_radiance(dn_bands[2], float(metadata['B4_Lmin']), float(metadata['B4_Lmax'])), esun['B4'], d_es, sun_elev)

        reflectance_bands = apply_dark_object_subtraction([b2, b3, b4], {'BAND2': 0.555, 'BAND3': 0.650, 'BAND4': 0.815})

        pred_mask = predict_scene_patchwise(reflectance_bands, profile, model, tf)

        # Save predicted mask
        mask_path = os.path.join(OUTPUT_PATH, f"mask_{scene_id}.tif")
        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(pred_mask.astype(np.uint8), 1)

        # Load GT and compute metrics
        gt_path = os.path.join(GROUND_TRUTH_MASKS_PATH, f"mask_{scene_id}.tif")
        with rasterio.open(gt_path) as gt:
            gt_mask = gt.read(1)

        pc_cloud, pc_shadow, acc, per_class = compute_metrics(gt_mask, pred_mask)

        save_shapefiles(pred_mask, profile['transform'], profile['crs'], OUTPUT_PATH, scene_id)

        # Log and display results
        row = [scene_id, pc_cloud, pc_shadow, acc]
        for cls in [1, 2]:
            row.extend([per_class[cls]['IoU'], per_class[cls]['Precision'], per_class[cls]['Recall'], per_class[cls]['F1']])
        metrics_rows.append(row)

        print(f"{scene_id}: Cloud={pc_cloud:.2f}%, Shadow={pc_shadow:.2f}%, Acc={acc:.3f}")
        for cls, name in zip([1,2], ["Cloud", "Shadow"]):
            m = per_class[cls]
            print(f"  {name}: IoU={m['IoU']:.3f}, Precision={m['Precision']:.3f}, Recall={m['Recall']:.3f}, F1={m['F1']:.3f}")

    df = pd.DataFrame(metrics_rows, columns=[
        "DatasetId", "%Cloud", "%Shadow", "Accuracy",
        "Cloud_IoU", "Cloud_Precision", "Cloud_Recall", "Cloud_F1",
        "Shadow_IoU", "Shadow_Precision", "Shadow_Recall", "Shadow_F1"])
    df.to_csv(os.path.join(OUTPUT_PATH, "inference_results.csv"), index=False)

if __name__ == "__main__":
    main()
