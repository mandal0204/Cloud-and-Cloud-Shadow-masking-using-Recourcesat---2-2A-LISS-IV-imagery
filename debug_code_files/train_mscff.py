import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_CUDNN_FRONTEND'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow logging

import tensorflow as tf; layers, register_keras_serializable, K = tf.keras.layers, tf.keras.utils.register_keras_serializable, tf.keras.backend
import numpy as np
import rasterio
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import datetime
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from itertools import cycle
import pickle

# ======================= CONFIGURATION =======================
EPOCHS = 2  # Reduced from 100 to fit GPU memory
PATCH_SIZE = 256  # Reduced from 512 to fit GPU memory
OVERLAP = 25      # Corresponding reduced overlap
BATCH_SIZE = 4    # Adjusted for 12GB GPU
WIDTH_MULTIPLIER = 0.70  # Reduce model complexity
LEARNING_RATE = 1e-3
NUM_CLASSES = 3
CLASS_NAMES = ['BACKGROUND', 'CLOUD', 'SHADOW']

# ======================= FOCAL LOSS IMPLEMENTATION =======================
def focal_loss(gamma=1.0, alpha=[0.2, 0.2, 0.6]):
    """Focal loss for multi-class classification with per-class alpha weights"""
    def focal_loss_fn(y_true, y_pred):
        # Default alpha values if not provided
        if alpha is None:
            alpha_vals = tf.ones(NUM_CLASSES, dtype=tf.float32)
        else:
            if isinstance(alpha, (list, tuple, np.ndarray)):
                alpha_vals = tf.constant(alpha, dtype=tf.float32)
            else:
                alpha_vals = tf.fill([NUM_CLASSES], float(alpha))
        
        # Ensure numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(y_true, tf.int32)
        
        # One-hot encode true labels
        y_true_one_hot = tf.one_hot(y_true, depth=NUM_CLASSES, dtype=tf.float32)
        
        # Calculate cross entropy
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        
        # Calculate focal factor
        p_t = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        
        # Apply class weights and focal factor
        alpha_weights = tf.reduce_sum(alpha_vals * y_true_one_hot, axis=-1)
        
        # Expand dimensions for broadcasting
        modulating_factor = tf.expand_dims(modulating_factor, axis=-1)
        alpha_weights = tf.expand_dims(alpha_weights, axis=-1)
        
        focal_cross_entropy = modulating_factor * cross_entropy * alpha_weights
        
        return tf.reduce_mean(tf.reduce_sum(focal_cross_entropy, axis=-1))
    return focal_loss_fn

# ======================= MSCFF MODEL ARCHITECTURE =======================
def build_mscff(input_shape=(PATCH_SIZE, PATCH_SIZE, 3), num_classes=NUM_CLASSES, width_multiplier=WIDTH_MULTIPLIER):
    """Simplified MSCFF model with reduced complexity"""
    def CBRR_conv_block(inputs, filters, dilation=1, separable=False):
        """Simplified conv block (2 layers instead of 3)"""
        filters = int(filters * width_multiplier)
        conv_layer = layers.SeparableConv2D if separable else layers.Conv2D
        
        # First conv block
        x = conv_layer(filters, (3, 3), padding='same', dilation_rate=dilation)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Second conv block (removed third layer)
        x = conv_layer(filters, (3, 3), padding='same', dilation_rate=dilation)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    inputs = layers.Input(shape=input_shape)
    
    # =================== Encoder =================== 
    # Block 1
    cbrr_1 = CBRR_conv_block(inputs, 64, separable=True)
    maxpool_1 = layers.MaxPooling2D((2, 2))(cbrr_1)
    
    # Block 2
    cbrr_2 = CBRR_conv_block(maxpool_1, 128, separable=True)
    maxpool_2 = layers.MaxPooling2D((2, 2))(cbrr_2)
    
    # Block 3
    cbrr_3 = CBRR_conv_block(maxpool_2, 256, separable=True)
    maxpool_3 = layers.MaxPooling2D((2, 2))(cbrr_3)
    
    # =================== Bottleneck =================== 
    cbrr_4 = CBRR_conv_block(maxpool_3, 512)
    cbrr_5 = CBRR_conv_block(cbrr_4, 512, dilation=2)
    cbrr_6 = CBRR_conv_block(cbrr_5, 512, dilation=4)
    
    # =================== Decoder =================== 
    # Skip connections directly from encoder (no attention)
    up1 = layers.UpSampling2D((2, 2), interpolation='bilinear')(cbrr_6)
    concat1 = layers.Concatenate(axis=-1)([up1, cbrr_3])
    cbrr_7 = CBRR_conv_block(concat1, 512)
    
    up2 = layers.UpSampling2D((2, 2), interpolation='bilinear')(cbrr_7)
    concat2 = layers.Concatenate(axis=-1)([up2, cbrr_2])
    cbrr_8 = CBRR_conv_block(concat2, 256)
    
    up3 = layers.UpSampling2D((2, 2), interpolation='bilinear')(cbrr_8)
    concat3 = layers.Concatenate(axis=-1)([up3, cbrr_1])
    cbrr_9 = CBRR_conv_block(concat3, 128)
    
    # =================== Simplified Multi-scale Fusion =================== 
    scale1 = layers.Conv2D(int(64 * width_multiplier), (1, 1), activation='relu')(cbrr_9)  # 512x512
    scale2 = layers.Conv2D(int(64 * width_multiplier), (1, 1), activation='relu')(cbrr_8)   # 256x256
    scale2_up = layers.UpSampling2D((2, 2), interpolation='bilinear')(scale2)  # 512x512
    
    # Merge only two scales (removed cloud/shadow branches)
    concatenation = layers.Concatenate(axis=-1)([scale1, scale2_up])
    
    # =================== Output =================== 
    x = layers.Conv2D(int(128 * width_multiplier), (3, 3), padding='same')(concatenation)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax', dtype='float32')(x)
    
    return tf.keras.Model(inputs, outputs, name="MSCFF_Simplified")


# ======================= RESUME TRAINING CALLBACK =======================
class FullTrainingCheckpoint(tf.keras.callbacks.Callback):
    """Callback to save full training state after each epoch"""
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        # Save model and optimizer state
        model_path = os.path.join(self.output_dir, f"resume_model_epoch_{epoch+1}.keras")
        self.model.save(model_path, save_format='keras')
        
        # Get generator state and algorithm
        generator = tf.random.get_global_generator()
        tf_state = generator.state.numpy()
        tf_alg = generator.algorithm
        
        # Save training state
        state = {
            'epoch': epoch,
            'history_epochs': self.model.history.epoch,
            'history': self.model.history.history,
            'random_state': np.random.get_state(),
            'tf_random_state': tf_state,
            'tf_random_alg': tf_alg,
            'optimizer_config': tf.keras.optimizers.serialize(self.model.optimizer)
        }
        state_path = os.path.join(self.output_dir, f"training_state_epoch_{epoch+1}.pkl")
        
        # Use pickle for complex data structures
        with open(state_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Update latest pointer
        with open(os.path.join(self.output_dir, "LATEST_CHECKPOINT"), "w") as f:
            f.write(str(epoch+1))
        
        print(f"\nSaved checkpoint for epoch {epoch+1} to {self.output_dir}")


# ======================= OPTIMIZED DATA PROCESSING =======================
class OptimizedPatchGenerator:
    def __init__(self, patch_size=PATCH_SIZE, overlap=OVERLAP):
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        
    def split_image(self, image_path, mask_path=None):
        """Efficient patch generation with memory mapping and oversampling for rare classes"""
        patches = []
        mask_patches = []
        positions = []
        
        with rasterio.open(image_path) as src:
            height, width = src.shape
            
            for y in range(0, height, self.stride):
                for x in range(0, width, self.stride):
                    y_end = min(y + self.patch_size, height)
                    x_end = min(x + self.patch_size, width)
                    
                    # Read image window
                    window = rasterio.windows.Window(x, y, x_end - x, y_end - y)
                    img_patch = src.read(window=window)
                    img_patch = img_patch.transpose(1, 2, 0).astype(np.float32) / 65535.0
                    
                    # Pad if necessary
                    pad_y = self.patch_size - img_patch.shape[0]
                    pad_x = self.patch_size - img_patch.shape[1]
                    if pad_y > 0 or pad_x > 0:
                        img_patch = np.pad(
                            img_patch, 
                            ((0, pad_y), (0, pad_x), (0, 0)),
                            mode='reflect'
                        )
                    
                    patches.append(img_patch)
                    positions.append((y, x))
                    
                    # Read corresponding mask if provided
                    if mask_path:
                        with rasterio.open(mask_path) as mask_src:
                            mask_patch = mask_src.read(1, window=window).astype(np.uint8)
                            
                            if pad_y > 0 or pad_x > 0:
                                mask_patch = np.pad(
                                    mask_patch, 
                                    ((0, pad_y), (0, pad_x)),
                                    mode='reflect'
                                )
                            
                            mask_patches.append(mask_patch)
        
        # Oversample rare classes (cloud and shadow)
        if mask_path:  # Only do oversampling and return 3 values if mask exists
            cloud_shadow_indices = []
            for i, mask in enumerate(mask_patches):
                if np.any(mask == 1) or np.any(mask == 2):
                    cloud_shadow_indices.append(i)
            
            if cloud_shadow_indices:
                num_extra = int(0.6 * len(patches))
                extra_indices = np.random.choice(
                    cloud_shadow_indices, 
                    size=num_extra, 
                    replace=True
                )
                for idx in extra_indices:
                    patches.append(patches[idx])
                    mask_patches.append(mask_patches[idx])
            
            return np.array(patches), np.array(mask_patches), positions
        else:
            return np.array(patches), positions  # No mask case

def process_images_to_tfrecords(image_dir, mask_dir, image_names, output_dir, patch_gen):
    """Optimized TFRecord creation with windowed reading"""
    total_files = len(image_names)
    print(f"Processing {total_files} images to TFRecords...")
    
    for i, name in enumerate(tqdm(image_names), 1):
        img_path = os.path.join(image_dir, name)
        mask_path = os.path.join(mask_dir, f"mask_{name}")
        
        # Create patches with windowed reading
        img_patches, mask_patches, _ = patch_gen.split_image(img_path, mask_path)
        
        # Save as TFRecord
        record_file = os.path.join(output_dir, f"{os.path.splitext(name)[0]}.tfrecord")
        
        with tf.io.TFRecordWriter(record_file) as writer:
            for img_patch, mask_patch in zip(img_patches, mask_patches):
                # Ensure correct shape
                img_patch = img_patch.astype(np.float32)
                
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

def prepare_training_val_data(image_dir, mask_dir, output_dir, val_split=0.2, seed=42):
    """Prepare dataset with train/val split using optimized patch generator"""
    print("\nPreparing optimized dataset:")
    print(f"  Image dir: {image_dir}")
    print(f"  Mask dir: {mask_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Patch size: {PATCH_SIZE}, Overlap: {OVERLAP}")
    
    os.makedirs(output_dir, exist_ok=True)
    patch_gen = OptimizedPatchGenerator()
    
    # Get all image names and split
    image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
    print(f"Found {len(image_names)} images")
    
    # Handle single-image case differently
    if len(image_names) == 1:
        print("Single image detected - splitting patches instead of images")
        img_path = os.path.join(image_dir, image_names[0])
        mask_path = os.path.join(mask_dir, f"mask_{image_names[0]}")
        
        # Create all patches
        img_patches, mask_patches, _ = patch_gen.split_image(img_path, mask_path)
        
        # Split patches
        train_idx, val_idx = train_test_split(
            range(len(img_patches)), 
            test_size=val_split,
            random_state=seed
        )
        
        # Create directories
        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Save training patches
        print("\nProcessing training patches:")
        record_file = os.path.join(train_dir, f"{os.path.splitext(image_names[0])[0]}.tfrecord")
        with tf.io.TFRecordWriter(record_file) as writer:
            for idx in train_idx:
                img_patch = img_patches[idx]
                mask_patch = mask_patches[idx]
                
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
        
        # Save validation patches
        print("\nProcessing validation patches:")
        record_file = os.path.join(val_dir, f"{os.path.splitext(image_names[0])[0]}.tfrecord")
        with tf.io.TFRecordWriter(record_file) as writer:
            for idx in val_idx:
                img_patch = img_patches[idx]
                mask_patch = mask_patches[idx]
                
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
                
        return train_dir, val_dir
    
    # For multiple images: use stratified split
    labels = []
    for name in tqdm(image_names, desc="Analyzing masks"):
        with rasterio.open(os.path.join(mask_dir, f"mask_{name}")) as src:
            mask = src.read(1)
            labels.append(np.median(mask))  # Use median class as label
            
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(sss.split(image_names, labels))
    
    train_names = [image_names[i] for i in train_idx]
    val_names = [image_names[i] for i in val_idx]

    print(f"Split: {len(train_names)} train, {len(val_names)} validation")
    
    # Process training set
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    print("\nProcessing training set:")
    process_images_to_tfrecords(image_dir, mask_dir, train_names, train_dir, patch_gen)
    
    # Process validation set
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    print("\nProcessing validation set:")
    process_images_to_tfrecords(image_dir, mask_dir, val_names, val_dir, patch_gen)
    
    print("\nDataset preparation complete!")
    return train_dir, val_dir

def parse_tfrecord(example):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, features)
    
    height = tf.cast(example['height'], tf.int32)
    width = tf.cast(example['width'], tf.int32)
    
    # Decode image and mask
    image = tf.io.decode_raw(example['image'], tf.float32)
    image = tf.reshape(image, (height, width, 3))
    
    mask = tf.io.decode_raw(example['mask'], tf.uint8)
    mask = tf.reshape(mask, (height, width))  # Ensure 2D shape
    
    # Add NaN/Inf check and normalization
    image = tf.where(
        tf.math.is_finite(image),
        image,
        tf.zeros_like(image)
    )
    
    return image, mask

def create_dataset(tfrecord_dir, batch_size=BATCH_SIZE, repeat=True, drop_remainder=False):
    """Create optimized dataset pipeline"""
    tfrecords = sorted(glob.glob(os.path.join(tfrecord_dir, '*.tfrecord')))
    
    # Create dataset with parallel processing
    dataset = tf.data.TFRecordDataset(
        tfrecords,
        num_parallel_reads=tf.data.AUTOTUNE
    )
    
    # Parse and preprocess
    dataset = dataset.map(
        parse_tfrecord,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Ensure shape - CRITICAL: Maintain 2D mask shape
    dataset = dataset.map(
        lambda img, msk: (img, tf.ensure_shape(msk, [PATCH_SIZE, PATCH_SIZE])),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Repeat if needed
    if repeat:
        dataset = dataset.repeat()
    
    # Batch and prefetch - DO NOT FLATTEN MASKS
    return dataset \
        .batch(batch_size, drop_remainder=drop_remainder) \
        .prefetch(tf.data.AUTOTUNE)

def get_class_weights(dataset, max_batches=200):
    """Calculate class weights from limited samples"""
    class_counts = [1e-7, 1e-7, 1e-7]  # BACKGROUND, CLOUD, SHADOW
    
    batch_count = 0
    for images, masks in dataset:
        # Masks are 3D: [batch, height, width]
        mask_np = masks.numpy()
        for i in range(3):
            class_counts[i] += np.sum(mask_np == i)
        
        batch_count += 1
        if batch_count >= max_batches:
            break
    
    print(f"Class counts: {class_counts}")
    total_pixels = sum(class_counts)
    
    freq = np.array(class_counts) / total_pixels
    median_freq = np.median(freq)
    
    return {i: median_freq / f for i, f in enumerate(freq)}

# ======================= METRICS & VISUALIZATION =======================
@register_keras_serializable(package='CustomMetrics')
class ClassSpecificIoU(tf.keras.metrics.MeanIoU):
    """Class-specific IoU metric that returns IoU for a single class"""
    def __init__(self, class_id, num_classes, name=None, dtype=None, **kwargs):
        # Accept and pass through new ignore_class parameter
        super().__init__(num_classes=num_classes, name=name, dtype=dtype, **kwargs)
        self.class_id = class_id
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probability predictions to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        """Return IoU for the specific class"""
        # Get the confusion matrix and convert to float32
        cm = tf.cast(self.total_cm, tf.float32)
        
        # Compute IoU for each class
        true_positives = tf.linalg.diag_part(cm)
        false_positives = tf.reduce_sum(cm, axis=0) - true_positives
        false_negatives = tf.reduce_sum(cm, axis=1) - true_positives
        
        denominator = true_positives + false_positives + false_negatives
        iou = true_positives / (denominator + tf.keras.backend.epsilon())
        
        # Return IoU for our class
        return iou[self.class_id]
    
    def get_config(self):
        config = super().get_config()
        config.update({'class_id': self.class_id})
        return config

@register_keras_serializable(package='CustomMetrics')
class ClassAccuracy(tf.keras.metrics.Metric):
    def __init__(self, class_id, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.class_id = class_id
        self.total = self.add_weight(name='total', initializer='zeros')
        self.correct = self.add_weight(name='correct', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probability predictions to class indices
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.uint8)
        matches = tf.equal(y_true, y_pred) & tf.equal(y_true, self.class_id)
        self.correct.assign_add(tf.reduce_sum(tf.cast(matches, tf.float32)))
        self.total.assign_add(tf.reduce_sum(tf.cast(tf.equal(y_true, self.class_id), tf.float32)))
    
    def result(self):
        return self.correct / (self.total + tf.keras.backend.epsilon())
    
    def get_config(self):
        config = super().get_config()
        config.update({'class_id': self.class_id})
        return config

@register_keras_serializable(package='CustomMetrics')
class Precision(tf.keras.metrics.Precision):
    def __init__(self, class_id, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.class_id = class_id
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probability predictions to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(
            tf.equal(y_true, self.class_id),
            tf.equal(y_pred, self.class_id),
            sample_weight=sample_weight
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({'class_id': self.class_id})
        return config

@register_keras_serializable(package='CustomMetrics')
class Recall(tf.keras.metrics.Recall):
    def __init__(self, class_id, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.class_id = class_id
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probability predictions to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(
            tf.equal(y_true, self.class_id),
            tf.equal(y_pred, self.class_id),
            sample_weight=sample_weight
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({'class_id': self.class_id})
        return config
    
@register_keras_serializable(package='CustomMetrics')
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, class_id, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.class_id = class_id
        self.precision = Precision(class_id=class_id)
        self.recall = Recall(class_id=class_id)
            
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
            
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * p * r / (p + r + tf.keras.backend.epsilon())
            
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        
    def get_config(self):
        config = super().get_config()
        config.update({'class_id': self.class_id})
        return config

class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lr_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            # Support for optimizers with learning rate schedules
            lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = lr(self.model.optimizer.iterations)
            current_lr = float(lr)
        except AttributeError:
            # Fallback to default method
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            
        self.lr_history.append(current_lr)
        logs['lr'] = current_lr

# ======================= UTILITY FUNCTIONS =======================
def count_examples_in_tfrecord(tfrecord_path):
    """Count number of examples in a TFRecord file"""
    count = 0
    for _ in tf.data.TFRecordDataset(tfrecord_path):
        count += 1
    return count

def count_total_examples(tfrecord_dir):
    """Count total examples in all TFRecords in a directory"""
    total_examples = 0
    tfrecords = glob.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
    for tfrecord in tfrecords:
        total_examples += count_examples_in_tfrecord(tfrecord)
    return total_examples

# ======================= VISUALIZATION FUNCTIONS =======================
import os
import matplotlib.pyplot as plt

def plot_training_history(history, output_dir):
    """Plot training history graphs for loss and metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss in a separate figure
    plt.figure(figsize=(18, 12))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    plt.close()

    # Plot metrics in a separate figure
    plt.figure(figsize=(18, 12))
    for metric in history.history.keys():
        if 'iou_' in metric or 'accuracy' in metric or 'f1_' in metric:
            if not metric.startswith('val_'):
                plt.plot(history.history[metric], label=f'Training {metric}')
                val_metric = f'val_{metric}'
                if val_metric in history.history:
                    plt.plot(history.history[val_metric], label=f'Validation {metric}')
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()


def generate_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Generate and save confusion matrix"""
    # Ensure all classes are present
    labels = np.arange(len(class_names))
    
    cm = confusion_matrix(
        y_true, 
        y_pred,
        labels=labels
    )
    
    plt.figure(figsize=(18, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    return cm

def generate_classification_report(y_true, y_pred, class_names, output_dir):
    """Generate and save classification report"""
    
    # Ensure all classes are present
    labels = np.arange(len(class_names))
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        labels=labels,
        output_dict=False,
        zero_division=0  # Handle potential division by zero
    )
    
    # Save as text file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Save as image
    plt.figure(figsize=(18, 12))
    plt.text(0.1, 0.1, report, {'fontsize': 12}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report.png'), bbox_inches='tight')
    plt.close()
    
    return report

def generate_roc_curve(y_true, y_score, class_names, output_dir):
    """Generate ROC curve for multi-class classification"""
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(18, 12))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    return roc_auc

def generate_precision_recall_curve(y_true, y_score, class_names, output_dir):
    """Generate Precision-Recall curve for multi-class classification"""
    n_classes = len(class_names)
    
    # Compute Precision-Recall curve for each class
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true == i, y_score[:, i])
    
    # Plot all curves
    plt.figure(figsize=(18, 12))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Class {0}'.format(class_names[i]))
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()

def plot_accuracy_history(history, output_dir):
    """Plot accuracy vs epochs graph"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(18, 12))
    
    # Plot training accuracy
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], 'b-', label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    
    # Plot class-wise accuracies if available
    colors = ['g', 'm', 'c', 'y', 'k']
    color_idx = 0
    for metric in history.history.keys():
        if 'accuracy_' in metric and 'val_' not in metric:
            plt.plot(history.history[metric], '--', color=colors[color_idx % len(colors)], 
                     label=f'Training {metric.split("_")[1]} Accuracy')
            color_idx += 1
    
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_history.png'))
    plt.close()

def plot_learning_rate_history(history, output_dir):
    """Plot learning rate vs epochs"""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'lr' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['lr'], 'b-o')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate_history.png'))
        plt.close()

def generate_class_heatmaps(model, sample_images, sample_masks, class_names, output_dir):
    """Generate and save class probability heatmaps for sample images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert list to tensor if needed
    if isinstance(sample_images, list):
        sample_images = tf.stack(sample_images)
    
    # Predict probabilities
    preds = model.predict(sample_images, verbose=0)
    
    # Create a figure for each sample
    for i in range(len(sample_images)):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        img = sample_images[i].numpy().astype(np.float32)
        true_mask = sample_masks[i].numpy().astype(np.uint8)
        pred_mask = np.argmax(preds[i], axis=-1).astype(np.uint8)
        
        # Original image
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth mask
        axes[0, 1].imshow(true_mask, cmap='jet', vmin=0, vmax=len(class_names)-1)
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')
        
        # Predicted mask
        axes[0, 2].imshow(pred_mask, cmap='jet', vmin=0, vmax=len(class_names)-1)
        axes[0, 2].set_title('Predicted Mask')
        axes[0, 2].axis('off')
        
        # Class probability heatmaps
        for class_idx, class_name in enumerate(class_names):
            row = 1
            col = class_idx
            prob_map = preds[i, :, :, class_idx]
            
            im = axes[row, col].imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
            axes[row, col].set_title(f'{class_name} Probability')
            axes[row, col].axis('off')
            
            # Add colorbar
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Sample {i+1} Predictions', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'class_heatmaps_sample_{i+1}.png'))
        plt.close(fig)
    
    print(f"Generated {len(sample_images)} class heatmap figures in {output_dir}")

# ======================= POST-TRAINING EVALUATION =======================
def evaluate_model(model, val_dir, history, class_names, output_dir):
    """Comprehensive model evaluation and visualization"""
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    print("\nStarting post-training evaluation...")
    start_time = time.time()
    
    # 1. Plot training history graphs
    plot_training_history(history, eval_dir)
    print("✓ Training history plots saved")
    
    # 2. Plot accuracy history
    plot_accuracy_history(history, eval_dir)
    print("✓ Accuracy history plot saved")
    
    # 3. Plot learning rate history
    plot_learning_rate_history(history, eval_dir)
    print("✓ Learning rate history plot saved")
    
    # 4. Prepare validation dataset
    val_ds = create_dataset(val_dir, repeat=False)
    
    # 5. Collect samples for visualizations
    sample_images, sample_masks = [], []
    y_true_all, y_prob_all = [], []
    MAX_SAMPLES = 2  # For heatmaps
    MAX_PIXELS = 2000000  # For curves and matrix
    
    print("\nCollecting validation samples for visualizations...")
    for images, masks in tqdm(val_ds, desc="Processing validation data"):
        # Predict on GPU
        preds = model.predict(images, verbose=0)
        
        # Store samples for heatmaps
        if len(sample_images) < MAX_SAMPLES:
            for i in range(min(MAX_SAMPLES - len(sample_images), images.shape[0])):
                sample_images.append(images[i])
                sample_masks.append(masks[i])
        
        # Flatten masks and predictions
        masks_flat = masks.numpy().flatten()
        preds_flat = preds.reshape(-1, preds.shape[-1])
        
        # Store for confusion matrix/report
        y_true_all.append(masks_flat)
        y_prob_all.append(preds_flat)
        
        # Break if we have enough pixels
        total_pixels = sum(len(arr) for arr in y_true_all)
        if total_pixels >= MAX_PIXELS:
            break
    
    # 6. Generate visualizations on CPU
    print("\nGenerating evaluation visualizations...")
    y_true_all = np.concatenate(y_true_all)[:MAX_PIXELS]
    y_prob_all = np.concatenate(y_prob_all)[:MAX_PIXELS]
    y_pred_all = np.argmax(y_prob_all, axis=1)
    
    # Confusion matrix and classification report
    generate_confusion_matrix(y_true_all, y_pred_all, class_names, eval_dir)
    generate_classification_report(y_true_all, y_pred_all, class_names, eval_dir)
    print("✓ Confusion matrix and classification report saved")
    
    # ROC and Precision-Recall curves
    generate_roc_curve(y_true_all, y_prob_all, class_names, eval_dir)
    generate_precision_recall_curve(y_true_all, y_prob_all, class_names, eval_dir)
    print("✓ ROC and Precision-Recall curves saved")
    
    # Class heatmaps
    heatmap_dir = os.path.join(eval_dir, "heatmaps")
    generate_class_heatmaps(model, sample_images, sample_masks, class_names, heatmap_dir)
    print("✓ Class heatmaps saved")
    
    elapsed = time.time() - start_time
    print(f"\nEvaluation complete in {elapsed:.1f} seconds! Results saved to {eval_dir}")

# ======================= COMPREHENSIVE TRAINING FUNCTION =======================
def train_comprehensive_model(train_dir, val_dir, model_save_path, epochs=100, output_dir=None):
    """Comprehensive training function with resume capability"""
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    
    # Create output directory if not provided
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join("training_logs", f"training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets without repeat for weight calculation
    raw_train_ds = create_dataset(train_dir, repeat=False, drop_remainder=False)
    
    # Calculate class weights
    class_weights_dict = get_class_weights(raw_train_ds)
    class_weights_list = [class_weights_dict[0], class_weights_dict[1], class_weights_dict[2]]
    print(f"Class weights: {class_weights_list}")
    
    # Create training dataset WITH repetition
    train_ds = create_dataset(train_dir, repeat=True)
    # Create validation dataset WITHOUT repetition
    val_ds = create_dataset(val_dir, repeat=False)
    
    # Define custom objects for model loading
    custom_objects = {
        'focal_loss_fn': focal_loss(gamma=1.0, alpha=class_weights_list),
        'ClassSpecificIoU': ClassSpecificIoU,
        'ClassAccuracy': ClassAccuracy,
        'Precision': Precision,
        'Recall': Recall,
        'F1Score': F1Score
    }
    
    # Check for existing checkpoint
    initial_epoch = 0
    latest_checkpoint = None
    checkpoint_file = os.path.join(output_dir, "LATEST_CHECKPOINT")
    full_history = None
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            latest_epoch = int(f.read().strip())
        
        # Find latest state files
        model_path = os.path.join(output_dir, f"resume_model_epoch_{latest_epoch}.keras")
        state_path = os.path.join(output_dir, f"training_state_epoch_{latest_epoch}.pkl")
        
        if os.path.exists(model_path) and os.path.exists(state_path):
            print(f"\nResuming training from epoch {latest_epoch}")
            latest_checkpoint = (model_path, state_path)
            initial_epoch = latest_epoch
    
    # Load state if resuming
    if latest_checkpoint:
        model_path, state_path = latest_checkpoint
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )
        
        # Load state using pickle
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
        
        initial_epoch = state['epoch'] + 1
        
        # Restore RNG states
        np.random.set_state(state['random_state'])
        
        # Restore TensorFlow random generator if available
        if 'tf_random_alg' in state and 'tf_random_state' in state:
            tf_generator = tf.random.Generator.from_state(
                alg=state['tf_random_alg'],
                state=state['tf_random_state']
            )
            tf.random.set_global_generator(tf_generator)
            print("Restored TensorFlow RNG state")
        else:
            print("Warning: TF RNG state not found in checkpoint. Using new generator.")
        
        # Restore optimizer state
        optimizer = tf.keras.optimizers.deserialize(
            state['optimizer_config'], custom_objects=custom_objects
        )
        model.optimizer = optimizer
        
        # Create full history object from saved state
        full_history = tf.keras.callbacks.History()
        full_history.history = state['history']
        
        # Handle epoch numbers
        if 'history_epochs' in state:
            full_history.epoch = state['history_epochs']
        else:
            # Reconstruct epoch numbers from history length
            num_epochs = len(next(iter(state['history'].values())))
            full_history.epoch = list(range(num_epochs))
        
        print(f"Resumed training from epoch {initial_epoch}")
    else:
        # Build new model
        model = build_mscff()
        print("Building new model")
        full_history = None
    
    classes = ['BACKGROUND', 'CLOUD', 'SHADOW']
    
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
    ]
    
    # Add per-class metrics
    for i, name in enumerate(classes):
        metrics.extend([
            ClassSpecificIoU(class_id=i, num_classes=3, name=f'iou_{name}'),
            ClassAccuracy(class_id=i, name=f'accuracy_{name}'),
            Precision(class_id=i, name=f'precision_{name}'),
            Recall(class_id=i, name=f'recall_{name}'),
            F1Score(class_id=i, name=f'f1_{name}')
        ])

    # Compile model with Focal Loss if not resuming
    if not latest_checkpoint:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss=focal_loss(gamma=1.0, alpha=class_weights_list),
            metrics=metrics
        )
        print("Compiled new model")

    # Calculate steps per epoch
    print("Counting training examples...")
    total_train_examples = count_total_examples(train_dir)
    train_steps = (total_train_examples + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    print(f"\nDataset statistics:")
    print(f"  Training examples: {total_train_examples}")
    print(f"  Train steps per epoch: {train_steps}")
    
    # Setup CSV logger to append to existing file
    csv_path = os.path.join(output_dir, 'training_log.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(
        csv_path,
        append=latest_checkpoint is not None  # Append if resuming
    )
    
    # Create a custom callback to maintain full history
    class HistoryKeeper(tf.keras.callbacks.Callback):
        def __init__(self, full_history):
            super().__init__()
            self.full_history = full_history
            self.current_history = {'loss': [], 'val_loss': []}
            for metric in metrics:
                name = metric.name
                self.current_history[name] = []
                self.current_history[f'val_{name}'] = []
        
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            # Update current epoch history
            for key, value in logs.items():
                if key in self.current_history:
                    self.current_history[key].append(value)
            
            # Update full history
            if self.full_history:
                for key, value_list in self.current_history.items():
                    # Only update metrics that exist in logs
                    if key in logs:
                        if key in self.full_history.history:
                            self.full_history.history[key].append(logs[key])
                        else:
                            self.full_history.history[key] = [logs[key]]
                
                # Add epoch number
                self.full_history.epoch.append(len(self.full_history.epoch))
    
    # Create callbacks
    lr_logger = LearningRateLogger()
    checkpoint_cb = FullTrainingCheckpoint(output_dir=output_dir)
    
    callbacks = [
        lr_logger,
        checkpoint_cb,
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            monitor='val_accuracy', 
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', 
            factor=0.3, 
            patience=5, 
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'tensorboard'),
            histogram_freq=0
        ),
        csv_logger
    ]
    
    # Train model
    print("\nStarting training...")
    new_history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1,
        steps_per_epoch=train_steps,
        validation_steps=None
    )
    
    # Create full history combining previous and current
    if latest_checkpoint:
        # Create a new history object with complete history
        combined_history = tf.keras.callbacks.History()
        
        # Combine epoch numbers
        combined_history.epoch = full_history.epoch + new_history.epoch
        
        # Combine all metrics
        all_keys = set(full_history.history.keys()) | set(new_history.history.keys())
        for key in all_keys:
            prev_values = full_history.history.get(key, [])
            new_values = new_history.history.get(key, [])
            combined_history.history[key] = prev_values + new_values
    else:
        combined_history = new_history
    
    print(f"\nTraining completed for {epochs} epochs! Final model saved to {model_save_path}")
    return combined_history, output_dir, class_weights_list

# ======================= MAIN EXECUTION =======================
if __name__ == "__main__":
    # Configuration
    IMAGE_DIR = "/home/btech1/isro/dataset/train/images"
    MASK_DIR = "/home/btech1/isro/dataset/train/masks"
    PATCH_DIR = "/home/btech1/isro/dataset/patches"
    MODEL_PATH = "/home/btech1/isro/models/optimized_mscff.keras"
    
    # Define TFRecord directories
    train_tfrecord_dir = os.path.join(PATCH_DIR, "train")
    val_tfrecord_dir = os.path.join(PATCH_DIR, "val")
    
    # Check if TFRecords already exist
    tfrecords_exist = (
        os.path.exists(train_tfrecord_dir) and 
        os.path.exists(val_tfrecord_dir) and
        len(glob.glob(os.path.join(train_tfrecord_dir, '*.tfrecord'))) > 0 and
        len(glob.glob(os.path.join(val_tfrecord_dir, '*.tfrecord'))) > 0
    )
    
    if tfrecords_exist:
        print("Existing TFRecords found. Skipping data preparation.")
        train_dir, val_dir = train_tfrecord_dir, val_tfrecord_dir
    else:
        print("Preparing dataset...")
        os.makedirs(PATCH_DIR, exist_ok=True)
        train_dir, val_dir = prepare_training_val_data(
            IMAGE_DIR,
            MASK_DIR,
            PATCH_DIR,
            val_split=0.16
        )

    # Use fixed output directory for resumability
    OUTPUT_DIR = "/home/btech1/isro/training_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Train model
    print("Starting training with Focal Loss...")
    history, output_dir, class_weights_list = train_comprehensive_model(
        train_dir,
        val_dir,
        MODEL_PATH,
        EPOCHS,
        output_dir=OUTPUT_DIR
    )

    print(f"\nTraining completed successfully for {EPOCHS} epochs! Model saved to {output_dir}")

    # Clear GPU memory before evaluation
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    print("\nLoading best model for evaluation...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'focal_loss_fn': focal_loss(gamma=1.0, alpha=class_weights_list),
            'ClassSpecificIoU': ClassSpecificIoU,
            'ClassAccuracy': ClassAccuracy,
            'Precision': Precision,
            'Recall': Recall,
            'F1Score': F1Score
        }
    )
    
    eval_class_names = CLASS_NAMES

    # Run comprehensive evaluation
    evaluate_model(model, val_dir, history, eval_class_names, output_dir)

    print("\nTraining and evaluation completed successfully!")