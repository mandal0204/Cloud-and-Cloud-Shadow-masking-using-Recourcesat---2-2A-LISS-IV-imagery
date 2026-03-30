import os
import numpy as np
import tensorflow as tf
import rasterio
from sklearn.model_selection import train_test_split

# ======================= DATA PROCESSING =======================

class PatchGenerator:
    # (Keep existing implementation unchanged)
    
def prepare_training_val_data(image_dir, mask_dir, output_dir, val_split=0.2, patch_size=512, overlap=52, seed=42):
    """
    Prepare dataset with automatic train/val split
    Args:
        val_split: Fraction for validation (0-1)
        seed: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    patch_gen = PatchGenerator(patch_size, overlap)
    
    # Get all image names and split
    image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
    train_names, val_names = train_test_split(image_names, test_size=val_split, random_state=seed)
    
    # Process training set
    print("Processing training set...")
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    process_images_to_tfrecords(image_dir, mask_dir, train_names, train_dir, patch_gen)
    
    # Process validation set
    print("Processing validation set...")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    process_images_to_tfrecords(image_dir, mask_dir, val_names, val_dir, patch_gen)
    
    print(f"Dataset prepared with {len(train_names)} train and {len(val_names)} val images")
    return train_dir, val_dir

def process_images_to_tfrecords(image_dir, mask_dir, image_names, output_dir, patch_gen):
    """Helper function to process images to TFRecords"""
    for name in image_names:
        # Load image
        with rasterio.open(os.path.join(image_dir, name)) as src:
            img = src.read().transpose(1, 2, 0).astype(np.float32) / 65535.0
        
        # Load corresponding mask
        mask_path = os.path.join(mask_dir, name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, f"mask_{name}")
            
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.uint8)
        
        # Create patches
        img_patches, _ = patch_gen.split_image(img)
        mask_patches, _ = patch_gen.split_image(mask[..., np.newaxis])
        
        # Save as TFRecord
        record_file = os.path.join(output_dir, f"{os.path.splitext(name)[0]}.tfrecord")
        with tf.io.TFRecordWriter(record_file) as writer:
            for img_patch, mask_patch in zip(img_patches, mask_patches):
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[img_patch.tobytes()])),
                    'mask': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[mask_patch.tobytes()])),
                    'height': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[img_patch.shape[0]])),
                    'width': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[img_patch.shape[1]]))
                }))
                writer.write(example.SerializeToString())

# (Keep parse_tfrecord and create_dataset functions unchanged)

# ======================= TRAINING =======================

def train_model(train_dir, val_dir, model_save_path, epochs=100, batch_size=2):
    train_ds = create_dataset(train_dir, batch_size)
    val_ds = create_dataset(val_dir, batch_size)
    
    model = MSCFF_V2()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.OneHotMeanIoU(num_classes=3, name='mean_iou'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(name='f1_score', average='weighted')
        ])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_mean_iou', mode='max'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

# ======================= MAIN EXECUTION =======================

if __name__ == "__main__":
    # Configuration
    IMAGE_DIR = "/path/to/dataset/images"
    MASK_DIR = "/path/to/dataset/masks"
    PATCH_DIR = "/path/to/patches"
    MODEL_PATH = "/path/to/model.keras"
    
    # Step 1: Prepare training and validation data
    train_dir, val_dir = prepare_training_val_data(
        IMAGE_DIR,
        MASK_DIR,
        PATCH_DIR,
        val_split=0.2,
        patch_size=512,
        overlap=52
    )
    
    # Step 2: Train model
    history = train_model(
        train_dir,
        val_dir,
        MODEL_PATH,
        epochs=100,
        batch_size=2
    )
    
    # Step 3: (Later) Use separate test data for evaluation
    # test_ds = create_dataset("/path/to/test/patches", batch_size=8)
    # model = tf.keras.models.load_model(MODEL_PATH)
    # results = model.evaluate(test_ds)
    # print("Test results:", results)