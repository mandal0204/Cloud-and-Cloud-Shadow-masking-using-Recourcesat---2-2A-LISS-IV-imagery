import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import rasterio
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_curve
import datetime
import json
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

# ======================= CONFIGURATION =======================
EPOCHS = 2
PATCH_SIZE = 256
OVERLAP = 64
BATCH_SIZE = 4
NUM_CLASSES = 3
CLASS_NAMES = ['BACKGROUND', 'CLOUD', 'SHADOW']
WIDTH_MULTIPLIER = 1  # Reduced model complexity
layers = tf.keras.layers

# User-provided paths
TRAIN_IMAGE_DIR = "/home/btech1/isro/dataset/train1/images"
TRAIN_MASK_DIR = "/home/btech1/isro/dataset/train1/masks"
VAL_IMAGE_DIR = "/home/btech1/isro/dataset/val1/images"
VAL_MASK_DIR = "/home/btech1/isro/dataset/val1/masks"

PATCH_DIR = "/home/btech1/isro/dataset/patches"
MODEL_PATH = "/home/btech1/isro/models/optimized_mscff.keras"
OUTPUT_DIR = f"/home/btech1/isro/training_output/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def validate_images(image_dir):
    """Check source images for NaN/inf values"""
    print("\nValidating source images...")
    image_paths = glob.glob(os.path.join(image_dir, "*.tif"))
    
    for path in tqdm(image_paths):
        try:
            with rasterio.open(path) as src:
                # Read a small portion to check
                sample = src.read(window=rasterio.windows.Window(0, 0, 100, 100))
                
                if np.any(np.isnan(sample)):
                    print(f"WARNING: NaN values found in {os.path.basename(path)}")
                
                if np.any(np.isinf(sample)):
                    print(f"WARNING: Inf values found in {os.path.basename(path)}")
                
                if np.any(sample < 0):
                    print(f"WARNING: Negative values found in {os.path.basename(path)}")
        except Exception as e:
            print(f"ERROR validating {os.path.basename(path)}: {str(e)}")
    
    print("Image validation complete!")

validate_images(TRAIN_IMAGE_DIR)
validate_images(VAL_IMAGE_DIR)

# ======================= FOCAL LOSS =======================
def focal_loss(gamma=2.0, alpha=None):
    def focal_loss_fn(y_true, y_pred):
        # Convert sparse integer labels to one-hot encoding
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=NUM_CLASSES)
        y_true_one_hot = tf.reshape(y_true_one_hot, tf.shape(y_pred))
        
        # Ensure numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate cross entropy
        ce = -y_true_one_hot * tf.math.log(y_pred)
        
        # Calculate focal factor
        p_t = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1)
        focal_factor = tf.pow(1.0 - p_t, gamma)
        
        # Expand dimensions for proper broadcasting
        focal_factor = tf.expand_dims(focal_factor, axis=-1)
        
        if alpha is not None:
            # Ensure alpha is a tensor
            alpha_vals = tf.constant(alpha, dtype=tf.float32)
            alpha_weights = tf.reduce_sum(alpha_vals * y_true_one_hot, axis=-1)
            alpha_weights = tf.expand_dims(alpha_weights, axis=-1)
            focal_ce = focal_factor * ce * alpha_weights
        else:
            focal_ce = focal_factor * ce
            
        # Calculate mean loss
        loss = tf.reduce_mean(tf.reduce_sum(focal_ce, axis=-1))
        
        # Check for NaN/inf and replace if necessary
        loss = tf.cond(tf.math.is_nan(loss) | tf.math.is_inf(loss),
                       lambda: tf.constant(0.0),
                       lambda: loss)
        
        return loss
    return focal_loss_fn

# ======================= MODEL ARCHITECTURE =======================
def build_model(input_shape=(PATCH_SIZE, PATCH_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution with kernel initializer
    x = layers.Conv2D(16, (3, 3), padding='same', 
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(32, (3, 3), padding='same', 
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Bottleneck
    x = layers.Conv2D(64, (3, 3), padding='same', 
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Decoder
    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', 
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    x = layers.Conv2D(16, (3, 3), padding='same', 
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax',
                            kernel_initializer='glorot_uniform')(x)
    
    return tf.keras.Model(inputs, outputs, name="Stable_Model")

# ======================= DATA PROCESSING =======================
def generate_patches(image_path, mask_path=None):
    patches, masks = [], []
    stride = PATCH_SIZE - OVERLAP
    
    with rasterio.open(image_path) as src:
        for y in range(0, src.height, stride):
            for x in range(0, src.width, stride):
                window = rasterio.windows.Window(
                    x, y, 
                    min(PATCH_SIZE, src.width - x), 
                    min(PATCH_SIZE, src.height - y)
                )
                
                try:
                    # Read image with proper error handling
                    img_patch = src.read(window=window)
                    
                    # Check for invalid values
                    if np.any(np.isnan(img_patch)):
                        print(f"WARNING: NaN values found in {image_path} at ({x},{y})")
                        img_patch = np.nan_to_num(img_patch)
                    
                    # Normalize and transpose
                    img_patch = img_patch.transpose(1, 2, 0).astype(np.float32) / 65535.0
                    
                    # Verify normalization
                    if np.any(img_patch < 0) or np.any(img_patch > 1):
                        print(f"WARNING: Invalid pixel values in {image_path} at ({x},{y})")
                        img_patch = np.clip(img_patch, 0, 1)
                    
                    # Pad if necessary
                    if img_patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                        img_patch = np.pad(
                            img_patch, 
                            [(0, PATCH_SIZE - img_patch.shape[0]), 
                             (0, PATCH_SIZE - img_patch.shape[1]), 
                             (0, 0)],
                            mode='reflect'
                        )
                    patches.append(img_patch)
                    
                    if mask_path:
                        with rasterio.open(mask_path) as mask_src:
                            mask_patch = mask_src.read(1, window=window).astype(np.uint8)
                            
                            # Strict validation and correction
                            mask_patch = np.clip(mask_patch, 0, NUM_CLASSES-1)
                            unique_vals = np.unique(mask_patch)
                            
                            # Log invalid values
                            invalid_vals = [v for v in unique_vals if v not in [0, 1, 2]]
                            if invalid_vals:
                                print(f"WARNING: Invalid mask values {invalid_vals} in {mask_path} at ({x},{y})")
                            
                            # Force all values to valid classes
                            mask_patch = np.where(np.isin(mask_patch, [0,1,2]), mask_patch, 0)
                            
                            # Pad if necessary
                            if mask_patch.shape != (PATCH_SIZE, PATCH_SIZE):
                                mask_patch = np.pad(
                                    mask_patch, 
                                    [(0, PATCH_SIZE - mask_patch.shape[0]), 
                                     (0, PATCH_SIZE - mask_patch.shape[1])],
                                    mode='constant',
                                    constant_values=0
                                )
                            masks.append(mask_patch)
                
                except Exception as e:
                    print(f"ERROR processing {image_path} at ({x},{y}): {str(e)}")
                    # Create blank patch if error occurs
                    blank_img = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
                    patches.append(blank_img)
                    if mask_path:
                        blank_mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                        masks.append(blank_mask)
    
    return np.array(patches), np.array(masks) if mask_path else None

def create_tfrecords(image_dir, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_names = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    
    for name in tqdm(image_names, desc=f"Creating TFRecords for {os.path.basename(output_dir)}"):
        img_path = os.path.join(image_dir, name)
        mask_path = os.path.join(mask_dir, f"mask_{name}")
        patches, masks = generate_patches(img_path, mask_path)
        
        record_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}.tfrecord")
        with tf.io.TFRecordWriter(record_path) as writer:
            for patch, mask in zip(patches, masks):
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch.tobytes()])),
                    'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.tobytes()]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

def prepare_datasets():
    train_dir = os.path.join(PATCH_DIR, "train")
    val_dir = os.path.join(PATCH_DIR, "val")
    
    # Create TFRecords if they don't exist
    if not (os.path.exists(train_dir) and glob.glob(os.path.join(train_dir, "*.tfrecord"))):
        os.makedirs(train_dir, exist_ok=True)
        create_tfrecords(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, train_dir)
    
    if not (os.path.exists(val_dir) and glob.glob(os.path.join(val_dir, "*.tfrecord"))):
        os.makedirs(val_dir, exist_ok=True)
        create_tfrecords(VAL_IMAGE_DIR, VAL_MASK_DIR, val_dir)
    
    return train_dir, val_dir

def parse_tfrecord(example):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, features)
    
    image = tf.io.decode_raw(example['image'], tf.float32)
    image = tf.reshape(image, (PATCH_SIZE, PATCH_SIZE, 3))
    
    # Handle NaN/inf in images
    image = tf.where(
        tf.math.is_finite(image),
        image,
        tf.zeros_like(image)
    )
    
    mask = tf.io.decode_raw(example['mask'], tf.uint8)
    mask = tf.reshape(mask, (PATCH_SIZE, PATCH_SIZE))
    
    # Force mask to valid range
    mask = tf.clip_by_value(mask, 0, NUM_CLASSES-1)
    
    return image, mask

def create_dataset(tfrecord_dir, batch_size=BATCH_SIZE):
    tfrecords = glob.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ======================= METRICS =======================
class ClassIoU(tf.keras.metrics.Metric):
    def __init__(self, class_id, num_classes, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.num_classes = num_classes
        self.total_cm = self.add_weight(name='total_confusion_matrix',
                                        shape=(num_classes, num_classes),
                                        initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Flatten tensors and ensure consistent data type
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int64)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.int64)
        
        # Update confusion matrix
        cm = tf.math.confusion_matrix(
            y_true, y_pred, 
            num_classes=self.num_classes,
            weights=sample_weight
        )
        self.total_cm.assign_add(cm)
        
    def result(self):
        true_positives = tf.linalg.diag_part(self.total_cm)
        false_positives = tf.reduce_sum(self.total_cm, axis=0) - true_positives
        false_negatives = tf.reduce_sum(self.total_cm, axis=1) - true_positives
        
        denominator = true_positives + false_positives + false_negatives
        iou = true_positives / (denominator + tf.keras.backend.epsilon())
        return iou[self.class_id]
    
    def reset_state(self):
        self.total_cm.assign(tf.zeros((self.num_classes, self.num_classes)))


class OverallIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(name='total_confusion_matrix',
                                        shape=(num_classes, num_classes),
                                        initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Flatten tensors and ensure consistent data type
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int64)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.int64)
        
        cm = tf.math.confusion_matrix(
            y_true, y_pred, 
            num_classes=self.num_classes,
            weights=sample_weight
        )
        self.total_cm.assign_add(cm)
        
    def result(self):
        true_positives = tf.linalg.diag_part(self.total_cm)
        false_positives = tf.reduce_sum(self.total_cm, axis=0) - true_positives
        false_negatives = tf.reduce_sum(self.total_cm, axis=1) - true_positives
        
        denominator = true_positives + false_positives + false_negatives
        iou = true_positives / (denominator + tf.keras.backend.epsilon())
        return tf.reduce_mean(iou)
    
    def reset_state(self):
        self.total_cm.assign(tf.zeros((self.num_classes, self.num_classes)))


class ClassAccuracy(tf.keras.metrics.Metric):
    def __init__(self, class_id, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Remove extra dimension added by batching
        y_true = tf.cast(tf.squeeze(y_true), tf.int64)  # Add squeeze here
        y_pred = tf.cast(tf.squeeze(y_pred), tf.int64)  # Add squeeze here
        
        # Get indices for this class
        class_mask = tf.equal(y_true, self.class_id)
        correct = tf.equal(y_true, y_pred) & class_mask
        
        self.true_positives.assign_add(tf.reduce_sum(tf.cast(correct, tf.float32)))
        self.total.assign_add(tf.reduce_sum(tf.cast(class_mask, tf.float32)))
        
    def result(self):
        return self.true_positives / (self.total + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.true_positives.assign(0.)
        self.total.assign(0.)


# Keep these as they are - they're correct
class ClassSpecificPrecision(tf.keras.metrics.Precision):
    def __init__(self, class_id, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probability predictions to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(
            tf.equal(y_true, self.class_id),
            tf.equal(y_pred, self.class_id),
            sample_weight=sample_weight
        )


class ClassSpecificRecall(tf.keras.metrics.Recall):
    def __init__(self, class_id, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probability predictions to class indices
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(
            tf.equal(y_true, self.class_id),
            tf.equal(y_pred, self.class_id),
            sample_weight=sample_weight
        )


def get_metrics():
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        OverallIoU(num_classes=NUM_CLASSES, name='mean_iou')
    ]
    
    for i, name in enumerate(CLASS_NAMES):
        metrics.extend([
            ClassIoU(i, NUM_CLASSES, name=f'iou_{name}'),
            ClassAccuracy(i, name=f'accuracy_{name}'),
            ClassSpecificPrecision(i, name=f'precision_{name}'),
            ClassSpecificRecall(i, name=f'recall_{name}'),
        ])
    
    return metrics

# ======================= VISUALIZATION =======================
def plot_loss_curves(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'))
    plt.close()

def plot_accuracy_curves(history):
    plt.figure(figsize=(10, 6))
    
    # Overall accuracy
    plt.plot(history.history['accuracy'], 'b-', label='Overall Accuracy (Train)')
    plt.plot(history.history['val_accuracy'], 'b--', label='Overall Accuracy (Val)')
    
    # Class-wise accuracy
    colors = ['g', 'r', 'c']
    for i, name in enumerate(CLASS_NAMES):
        train_key = f'precision_{name}'  # Using precision as proxy for class accuracy
        val_key = f'val_precision_{name}'
        plt.plot(history.history[train_key], f'{colors[i]}-', label=f'{name} Accuracy (Train)')
        plt.plot(history.history[val_key], f'{colors[i]}--', label=f'{name} Accuracy (Val)')
    
    plt.title('Accuracy Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_curves.png'))
    plt.close()

def plot_iou_curves(history):
    plt.figure(figsize=(10, 6))
    
    # Overall IoU
    plt.plot(history.history['mean_iou'], 'b-', label='Mean IoU (Train)')
    plt.plot(history.history['val_mean_iou'], 'b--', label='Mean IoU (Val)')
    
    # Class-wise IoU
    colors = ['g', 'r', 'c']
    for i, name in enumerate(CLASS_NAMES):
        train_key = f'iou_{name}'
        val_key = f'val_iou_{name}'
        plt.plot(history.history[train_key], f'{colors[i]}-', label=f'{name} IoU (Train)')
        plt.plot(history.history[val_key], f'{colors[i]}--', label=f'{name} IoU (Val)')
    
    plt.title('IoU Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'iou_curves.png'))
    plt.close()

def plot_precision_curves(history):
    plt.figure(figsize=(10, 6))
    
    # Overall precision (average)
    precision_values = []
    val_precision_values = []
    for name in CLASS_NAMES:
        precision_values.append(history.history[f'precision_{name}'])
        val_precision_values.append(history.history[f'val_precision_{name}'])
    
    mean_precision = np.mean(precision_values, axis=0)
    mean_val_precision = np.mean(val_precision_values, axis=0)
    
    plt.plot(mean_precision, 'b-', label='Mean Precision (Train)')
    plt.plot(mean_val_precision, 'b--', label='Mean Precision (Val)')
    
    # Class-wise precision
    colors = ['g', 'r', 'c']
    for i, name in enumerate(CLASS_NAMES):
        train_key = f'precision_{name}'
        val_key = f'val_precision_{name}'
        plt.plot(history.history[train_key], f'{colors[i]}-', label=f'{name} Precision (Train)')
        plt.plot(history.history[val_key], f'{colors[i]}--', label=f'{name} Precision (Val)')
    
    plt.title('Precision Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'precision_curves.png'))
    plt.close()

def plot_recall_curves(history):
    plt.figure(figsize=(10, 6))
    
    # Overall recall (average)
    recall_values = []
    val_recall_values = []
    for name in CLASS_NAMES:
        recall_values.append(history.history[f'recall_{name}'])
        val_recall_values.append(history.history[f'val_recall_{name}'])
    
    mean_recall = np.mean(recall_values, axis=0)
    mean_val_recall = np.mean(val_recall_values, axis=0)
    
    plt.plot(mean_recall, 'b-', label='Mean Recall (Train)')
    plt.plot(mean_val_recall, 'b--', label='Mean Recall (Val)')
    
    # Class-wise recall
    colors = ['g', 'r', 'c']
    for i, name in enumerate(CLASS_NAMES):
        train_key = f'recall_{name}'
        val_key = f'val_recall_{name}'
        plt.plot(history.history[train_key], f'{colors[i]}-', label=f'{name} Recall (Train)')
        plt.plot(history.history[val_key], f'{colors[i]}--', label=f'{name} Recall (Val)')
    
    plt.title('Recall Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'recall_curves.png'))
    plt.close()

def plot_precision_recall_curve(model, dataset):
    y_true, y_prob = [], []
    
    for images, masks in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(masks.numpy().flatten())
        y_prob.extend(preds.reshape(-1, NUM_CLASSES))
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    plt.figure(figsize=(10, 8))
    for i in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(y_true == i, y_prob[:, i])
        plt.plot(recall, precision, lw=2, label=f'{CLASS_NAMES[i]}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_curve.png'))
    plt.close()

def plot_confusion_matrix(model, dataset):
    y_true, y_pred = [], []
    
    for images, masks in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(masks.numpy().flatten())
        y_pred.extend(tf.argmax(preds, axis=-1).numpy().flatten())
    
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

def plot_class_heatmaps(model, dataset, num_samples=3):
    heatmap_dir = os.path.join(OUTPUT_DIR, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Get sample images
    sample_images, sample_masks = [], []
    for images, masks in dataset.take(1):
        sample_images = images[:num_samples]
        sample_masks = masks[:num_samples]
    
    # Predict probabilities
    preds = model.predict(sample_images, verbose=0)
    
    for i in range(num_samples):
        plt.figure(figsize=(20, 15))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(sample_images[i])
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(2, 3, 2)
        plt.imshow(sample_masks[i], cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
        plt.title('Ground Truth Mask')
        plt.axis('off')
        
        # Predicted mask
        plt.subplot(2, 3, 3)
        plt.imshow(np.argmax(preds[i], axis=-1), cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
        plt.title('Predicted Mask')
        plt.axis('off')
        
        # Class probability heatmaps
        for j, class_name in enumerate(CLASS_NAMES):
            plt.subplot(2, 3, j+4)
            plt.imshow(preds[i, :, :, j], cmap='viridis', vmin=0, vmax=1)
            plt.title(f'{class_name} Probability')
            plt.colorbar()
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(heatmap_dir, f'heatmap_sample_{i+1}.png'))
        plt.close()

class DebugCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Checking initial weights...")
        for layer in self.model.layers:
            weights = layer.get_weights()
            for i, w in enumerate(weights):
                if np.any(np.isnan(w)):
                    print(f"NaN in {layer.name} weight {i}")
                if np.any(np.isinf(w)):
                    print(f"Inf in {layer.name} weight {i}")
    
    def on_batch_end(self, batch, logs=None):
        # Check outputs
        if logs and 'loss' in logs and (np.isnan(logs['loss']) or np.isinf(logs['loss'])):
            print(f"Batch {batch}: Invalid loss ({logs['loss']})")
            
            # Check predictions
            sample = self.model.layers[0].input
            outputs = self.model(sample)
            if tf.reduce_any(tf.math.is_nan(outputs)):
                print("  NaN in model outputs")
            if tf.reduce_any(tf.math.is_inf(outputs)):
                print("  Inf in model outputs")
                
            # Check gradients
            with tf.GradientTape() as tape:
                preds = self.model(sample, training=True)
                loss = self.model.loss(sample, preds)
            grads = tape.gradient(loss, self.model.trainable_variables)
            for i, grad in enumerate(grads):
                if grad is not None and (tf.reduce_any(tf.math.is_nan(grad)) or tf.reduce_any(tf.math.is_inf(grad))):
                    print(f"  Invalid gradients in {self.model.trainable_variables[i].name}")

# ======================= TRAINING =======================
def calculate_class_weights(dataset):
    class_counts = np.zeros(NUM_CLASSES)
    
    for _, masks in dataset.take(100):  # Use first 100 batches
        masks = masks.numpy().flatten()
        for i in range(NUM_CLASSES):
            class_counts[i] += np.sum(masks == i)
    
    total = np.sum(class_counts)
    class_weights = total / (NUM_CLASSES * class_counts + 1e-8)
    return class_weights / np.sum(class_weights)

def validate_dataset(dataset):
    """Check for invalid values in dataset"""
    print("Validating dataset...")
    for images, masks in dataset.take(1):
        # Check images
        if tf.reduce_any(tf.math.is_nan(images)):
            print("ERROR: NaN values found in images")
        if tf.reduce_any(tf.math.is_inf(images)):
            print("ERROR: Inf values found in images")
        
        # Check masks
        unique_vals = tf.unique(tf.reshape(masks, [-1])).y
        print("Unique mask values:", unique_vals.numpy())
        
        # Verify mask values are within valid range
        invalid = tf.logical_or(masks < 0, masks >= NUM_CLASSES)
        if tf.reduce_any(invalid):
            invalid_count = tf.reduce_sum(tf.cast(invalid, tf.int32))
            print(f"ERROR: Found {invalid_count} invalid mask values")
            print("Sample invalid mask:", masks[invalid].numpy())
            
    print("Dataset validation complete!")

def recreate_tfrecords_if_needed():
    train_dir = os.path.join(PATCH_DIR, "train")
    val_dir = os.path.join(PATCH_DIR, "val")
    
    # Check if recreation is needed
    needs_recreation = False
    for dir_path in [train_dir, val_dir]:
        tfrecords = glob.glob(os.path.join(dir_path, '*.tfrecord'))
        for tfrecord in tfrecords:
            for example in tf.data.TFRecordDataset(tfrecord).take(1):
                # Check for invalid mask values
                features = {
                    'image': tf.io.FixedLenFeature([], tf.string),
                    'mask': tf.io.FixedLenFeature([], tf.string)
                }
                parsed = tf.io.parse_single_example(example, features)
                mask = tf.io.decode_raw(parsed['mask'], tf.uint8).numpy()
                if np.any(mask > NUM_CLASSES-1):
                    print(f"Invalid mask values found in {tfrecord}")
                    needs_recreation = True
                    break
            if needs_recreation:
                break
        if needs_recreation:
            break
    
    # Recreate if needed
    if needs_recreation:
        print("Recreating TFRecords due to invalid mask values...")
        shutil.rmtree(PATCH_DIR, ignore_errors=True)
        prepare_datasets()

def train_model():
    # Prepare data
    train_dir, val_dir = prepare_datasets()
    train_ds = create_dataset(train_dir)
    val_ds = create_dataset(val_dir)
    
    # Validate datasets
    validate_dataset(train_ds)
    validate_dataset(val_ds)

    recreate_tfrecords_if_needed()
    # Calculate class weights
    class_weights = calculate_class_weights(train_ds)
    print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights))}")
    
    def flattened_sparse_cce(y_true, y_pred):
        # Flatten spatial dimensions and class dimension
        flat_y_true = tf.reshape(y_true, [-1])
        flat_y_pred = tf.reshape(y_pred, [-1, NUM_CLASSES])
        return tf.keras.losses.sparse_categorical_crossentropy(flat_y_true, flat_y_pred)
    
    # Build and compile model
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=flattened_sparse_cce,
        metrics=get_metrics()
    )
    

    # Callbacks
    callbacks = [
        DebugCallback(),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor='val_accuracy'),
        tf.keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, 'training_log.csv')),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=dict(enumerate(class_weights))
    )
    
    # Save final model
    model.save(MODEL_PATH)
    
    return model, history

# ======================= MAIN =======================
if __name__ == "__main__":
    # Train model
    model, history = train_model()
    
    # Save training history
    with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as f:
        json.dump(history.history, f)
    
    # Generate visualizations
    val_ds = create_dataset(os.path.join(PATCH_DIR, "val"))
    
    # Create all requested plots
    plot_loss_curves(history)
    plot_accuracy_curves(history)
    plot_iou_curves(history)
    plot_precision_curves(history)
    plot_recall_curves(history)
    plot_precision_recall_curve(model, val_ds)
    plot_confusion_matrix(model, val_ds)
    plot_class_heatmaps(model, val_ds, num_samples=3)
    
    print(f"\nTraining complete! Results saved to: {OUTPUT_DIR}")
