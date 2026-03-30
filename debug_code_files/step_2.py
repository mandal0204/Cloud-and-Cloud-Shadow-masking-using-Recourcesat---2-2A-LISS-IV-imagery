import os
import numpy as np
import rasterio
import cv2

def csdsi_detection(img, R, G, B, NIR, T1, t2, t3, t4, T5, T6, T7, T8):
    """
    Adapted from GitHub: Cloud/Shadow Detection based on Spectral Indices
    Input `img` shape: (height, width, bands)
    """
    ci_1 = (3 * img[:, :, NIR-1]) / (img[:, :, R-1] + img[:, :, G-1] + img[:, :, B-1] + 1e-6)
    ci_2 = (img[:, :, R-1] + img[:, :, G-1] + img[:, :, B-1] + img[:, :, NIR-1]) / 4
    
    T2 = np.nanmean(ci_2) + t2 * (np.nanmax(ci_2) - np.nanmean(ci_2))
    
    prelim_cloud_mask = np.float32((np.abs(ci_1 - 1) < T1) & (ci_2 > T2))
    final_cloud_mask = cv2.medianBlur(prelim_cloud_mask, T7)
    
    T3 = np.nanmin(img[:, :, NIR-1]) + t3 * (np.nanmean(img[:, :, NIR-1]) - np.nanmin(img[:, :, NIR-1]))
    T4 = np.nanmin(img[:, :, B-1]) + t4 * (np.nanmean(img[:, :, B-1]) - np.nanmin(img[:, :, B-1]))
    
    prelim_cloud_shadow_mask = np.float32((img[:, :, NIR-1] < T3) & (img[:, :, B-1] < T4))
    
    kernel = np.ones((T5, T6), dtype=np.float32)
    non_pseudo_mask = cv2.filter2D(final_cloud_mask, -1, kernel) == 0
    
    refined_shadow_mask = prelim_cloud_shadow_mask * non_pseudo_mask
    final_cloud_shadow_mask = cv2.medianBlur(refined_shadow_mask, T8)
    
    return final_cloud_mask, final_cloud_shadow_mask

def generate_masks(b2, b3, b4, profile, output_folder, scene_id, 
                  T1=0.4, t2=0.5, t3=0.25, t4=0.25, 
                  T5=11, T6=11, T7=3, T8=3):
    """Main function using CSD-SI logic"""
    # Validate input
    for band, name in zip([b2, b3, b4], ['Green', 'Red', 'NIR']):
        if np.all(np.isnan(band)):
            raise ValueError(f"{name} band contains only NaNs")
    
    # Create 4-band array: [Green, Red, Green (as Blue), NIR]
    blue_synthetic = b2.copy()  # Use Green as Blue substitute
    img_stack = np.stack([b2, b3, blue_synthetic, b4], axis=-1)
    
    # Detect clouds/shadows (band indices: R=2, G=1, B=3, NIR=4)
    cloud_mask, shadow_mask = csdsi_detection(
        img_stack, 
        R=2, G=1, B=3, NIR=4,
        T1=T1, t2=t2, t3=t3, t4=t4,
        T5=T5, T6=T6, T7=T7, T8=T8
    )
    
    # Combine masks (cloud priority)
    final_mask = np.zeros_like(b2, dtype=np.uint8)
    final_mask[shadow_mask > 0] = 2  # Shadow = 2
    final_mask[cloud_mask > 0] = 1   # Cloud = 1 (overrides shadow)
    
    # Save outputs (same as original)
    os.makedirs(output_folder, exist_ok=True)
    
    # Training mask
    train_path = os.path.join(output_folder, f"mask_{scene_id}.tif")
    with rasterio.open(train_path, 'w', 
                       driver=profile['driver'],
                       height=profile['height'],
                       width=profile['width'],
                       count=1,
                       dtype='uint8',
                       crs=profile['crs'],
                       transform=profile['transform'],
                       compress='lzw') as dst:
        dst.write(final_mask, 1)
    

    # Save visualization-optimized mask with transparent background
    vis_path = os.path.join(output_folder, f"vis_mask_{scene_id}.tif")
    # Create RGBA representation with transparency
    rgba = np.zeros((final_mask.shape[0], final_mask.shape[1], 4), dtype=np.uint8)

    # Apply color mapping with transparency
    rgba[final_mask == 0] = [0, 0, 0, 0]        # Transparent background
    rgba[final_mask == 1] = [255, 255, 255, 255] # Clouds: White (opaque)
    rgba[final_mask == 2] = [0, 100, 255, 255]   # Shadows: Light blue (opaque)

    # Update metadata for visualization - critical changes for QGIS
    vis_meta = profile.copy()
    vis_meta.update({
        'count': 4,
        'dtype': 'uint8',
        'nodata': None,
        'photometric': 'rgb',  # Indicates RGB color space
        'alpha': 'yes',       # Explicitly marks band 4 as alpha channel
        'compress': 'lzw'     # Better for preserving transparency
    })

    with rasterio.open(vis_path, 'w', **vis_meta) as dst:
        dst.write(rgba.transpose(2, 0, 1))
        # Explicitly set band color interpretations
        dst.set_band_description(1, 'Red')
        dst.set_band_description(2, 'Green')
        dst.set_band_description(3, 'Blue')
        dst.set_band_description(4, 'Alpha')
    
    return train_path, vis_path

# Example Usage
if __name__ == "__main__":
    img_path = r"/home/btech1/isro/R2F30APR2024067624010300067SSANSTUC00GTDB/R2F30APR2024067624010300067SSANSTUC00GTDB.tif"
    output_path = r"/home/btech1/isro/R2F30APR2024067624010300067SSANSTUC00GTDB"
    
    with rasterio.open(img_path) as src:
        b2 = src.read(1)  # Green
        b3 = src.read(2)  # Red
        b4 = src.read(3)  # NIR
        profile = src.profile
    
    # Recommended parameters (tune as needed)
    train_mask, vis_mask = generate_masks(
        b2, b3, b4, profile, output_path, "R2F30APR2024067624010300067SSANSTUC00GTDB",
        T1=0.9,   # 0.95 Cloud index threshold  - inc inc
        t2=0.2,   # 0.1 Cloud brightness adjustment - dec inc
        t3=1.2,  # 0.4 Shadow NIR adjustment - inc inc
        t4=1.2,  # 0.4 Shadow visible band adjustment - inc inc
        T5=7,    # Spatial search height
        T6=7,    # Spatial search width
        T7=3,     # Cloud median filter size
        T8=3      # Shadow median filter size
    )

    print(f"Created masks: {train_mask} and {vis_mask}")