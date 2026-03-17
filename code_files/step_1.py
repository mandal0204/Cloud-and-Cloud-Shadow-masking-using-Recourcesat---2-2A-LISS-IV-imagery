import rioxarray as rxr
import xarray as xr
import rasterio
from rasterio.env import Env
import numpy as np
import pandas as pd
import cv2
from skimage.morphology import remove_small_objects
import math
import datetime
import os

# ------------------------------- CONFIGURATION ------------------------------- #

INPUT_PARENT_FOLDER = r"/home/btech1/isro/upload/SampleTestData"
EARTH_SUN_EXCEL_FILE = r"/home/btech1/isro/upload/TOA_help/Earth_Sun_distance.xlsx"
OUTPUT_PARENT_FOLDER = r"/home/btech1/isro/upload/TOA_output"
APPLY_DOS = True

# ------------------------- HELPER FUNCTIONS ------------------------- #

# Function to read the metadata 
def read_metadata(input_folder):
    metadata_path = os.path.join(input_folder, 'BAND_META.txt')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
    else: 
        print(f"Reading metadata file from : {metadata_path}")

    metadata = {}
    with open(metadata_path) as f:
        for line in f:
            key, value = line.split('=')
            metadata[key.strip()] = value.strip()

    return metadata

# Function to get the distances between earth and sun
def get_earth_sun_distance(date, excel_path):
    print(f"Extracting Earth-Sun distance for date: {date.strftime('%d-%b-%Y')}")
    date = date.replace(year=2020)
    df = pd.read_excel(excel_path, sheet_name="Earth_sun_distance", header=1)
    match = df.where(df == date).stack()
    if match.empty:
        raise ValueError(f"Date {date.strftime('%d-%b-%Y')} not found in Excel file.")

    row_idx, col_label = match.index[0]
    col_idx = df.columns.get_loc(col_label)
    return df.iloc[row_idx, col_idx + 2]

# Function to get esun_values
def get_esun_values(sat_id):
    print(f"Extracting E_sun value for: {sat_id}")
    esun = {
        'RS2': {'B2': 185.33, 'B3': 157.766, 'B4': 111.359},
        'RS2A': {'B2': 185.347, 'B3': 158.262, 'B4': 110.81}
    }
    key = 'RS2' if sat_id == 'IRS-R2' else 'RS2A'
    return esun[key]

# Function to read bands
def read_bands(input_folder):
    print("Reading raster (.tif) files for each band...")
    with Env(GTIFF_SRS_SOURCE='GEOKEYS'):
        # Read all bands
        bands = [rxr.open_rasterio(os.path.join(input_folder, f"BAND{i}.tif")).where(lambda x: x != 0, np.nan).squeeze() for i in [2, 3, 4]]
        
        # Get spatial metadata from the first band
        profile = {
            'driver': 'GTiff',
            'dtype': bands[0].values.dtype,
            'width': bands[0].rio.width,
            'height': bands[0].rio.height,
            'count': 1,
            'crs': bands[0].rio.crs,
            'transform': bands[0].rio.transform(),
            'nodata': np.nan
        }
        return bands, profile

# Function to convert DN values to radiance
def dn_to_radiance(dn, lmin, lmax):
    return dn * (lmax - lmin) / 1024

# Function to convert radiance to reflectance
def radiance_to_reflectance(radiance, esun, d_es, sun_elevation):
    return (math.pi * radiance * (d_es ** 2)) / (esun * math.sin(sun_elevation))

# Function for DOS
def apply_dark_object_subtraction(ref_bands, wavelengths):
    print("Applying dark object subtraction for each band")
    red = ref_bands[1].values
    dark = next((px for px in np.unique(red[~np.isnan(red)])[:100] if px > 0), None)
    if dark is None:
        raise ValueError("No valid dark object found.")

    green_median = np.nanmedian(ref_bands[0].values)
    nir_median = np.nanmedian(ref_bands[2].values)

    model = -4 if nir_median < 0.02 else (-1 if green_median > 0.15 else -2)
    multiplier = 0.8 if dark < 0.01 else (1.2 if dark > 0.05 else 1.0)

    haze = [(dark * ((wavelengths[i] / wavelengths['BAND3']) **model) * multiplier) for i in ['BAND2', 'BAND3', 'BAND4']]
    return [(ref - h).clip(0, 1.0) for ref, h in zip(ref_bands, haze)]


# ------------------------- MAIN FUNCTION ------------------------- #

# Function to convert DN to TOA reflectance
def convert_DN_to_reflectance(input_folder, excel_file, output_folder, apply_dos):
    metadata = read_metadata(input_folder)
    scene_id = metadata['OTSProductID']
    date = datetime.datetime.strptime(metadata['DateOfPass'], '%d-%b-%Y')
    d_es = get_earth_sun_distance(date, excel_file)
    esun = get_esun_values(metadata['SatID'])
    sun_elev = math.radians(float(metadata['SunElevationAtCenter']))

    # Read DN values and convert to radiance
    [b2_dn, b3_dn, b4_dn], profile = read_bands(input_folder)
    print("Calculating radiance from DN values for each band...")
    b2_rad = dn_to_radiance(b2_dn, float(metadata['B2_Lmin']), float(metadata['B2_Lmax']))
    b3_rad = dn_to_radiance(b3_dn, float(metadata['B3_Lmin']), float(metadata['B3_Lmax']))
    b4_rad = dn_to_radiance(b4_dn, float(metadata['B4_Lmin']), float(metadata['B4_Lmax']))

    del b2_dn, b3_dn, b4_dn

    # Convert radiance to reflectance
    print("Calculating TOA reflectance from radiance values for each band...")
    b2_ref = radiance_to_reflectance(b2_rad, esun['B2'], d_es, sun_elev)
    b3_ref = radiance_to_reflectance(b3_rad, esun['B3'], d_es, sun_elev)
    b4_ref = radiance_to_reflectance(b4_rad, esun['B4'], d_es, sun_elev)

    del b2_rad, b3_rad, b4_rad

    reflectance_bands = [b2_ref, b3_ref, b4_ref]

    del b2_ref, b3_ref, b4_ref

    # Apply Dark Object Subtraction if apply_dos = True
    if apply_dos:
        wavelengths = {'BAND2': 0.555, 'BAND3': 0.650, 'BAND4': 0.815}
        reflectance_bands = apply_dark_object_subtraction(reflectance_bands, wavelengths)
    
    # Make output file path
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{scene_id}.tif")

    # Stack and export
    scene = xr.concat(reflectance_bands, dim='band').assign_coords(band=['BAND2', 'BAND3', 'BAND4'])
    scene.name = scene_id

    del reflectance_bands

    print(f"Saving output file to: {output_folder}")
    scene.to_dataset('band').rio.to_raster(
        output_file, driver='COG', dtype='float32', nodata=0,
        blockxsize=256, blockysize=256, tiled=True,
        compress='deflate', quality=1, predictor=3,
        BIGTIFF='IF_NEEDED', overviews='auto', windowed=True,
        num_threads='all_cpus')

# ------------------------- EXECUTION ------------------------- #

input_folders = [ os.path.join(INPUT_PARENT_FOLDER, f, sf) for f in os.listdir(INPUT_PARENT_FOLDER) for sf in os.listdir(os.path.join(INPUT_PARENT_FOLDER, f))]
output_folders = [os.path.join(OUTPUT_PARENT_FOLDER, f) for f in os.listdir(INPUT_PARENT_FOLDER)]

for idx, (input_folder_path, output_folder_path) in enumerate(zip(input_folders, output_folders), 1):
    os.makedirs(output_folder_path, exist_ok=True)  # Create the output folder 
    print(f"Processing folder {idx} of {len(input_folders)}: {input_folder_path}")

    # Call the function to convert DN to reflectance
    convert_DN_to_reflectance(input_folder_path, EARTH_SUN_EXCEL_FILE, output_folder_path, APPLY_DOS)
    
    # Print success message
    print(f"Folder {idx} processed successfully. Output saved to: {output_folder_path}\n")

    
print("TOA reflectance conversion successful for all the scenes.")