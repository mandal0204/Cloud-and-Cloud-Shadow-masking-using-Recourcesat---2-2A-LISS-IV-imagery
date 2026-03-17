import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, backend as K
import numpy as np
import rasterio
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import datetime
import json

# ======================= CONFIGURATION =======================
EPOCHS = 2
PATCH_SIZE = 256
OVERLAP = 30
BATCH_SIZE = 4
NUM_CLASSES = 3
CLASS_NAMES = ['BACKGROUND', 'CLOUD', 'SHADOW']
WIDTH_MULTIPLIER = 0.6
IMAGE_DIR = "/home/btech1/isro/dataset/train1/images"
MASK_DIR = "/home/btech1/isro/dataset/train1/masks"
PATCH_DIR = "/home/btech1/isro/dataset/patches"
MODEL_PATH = "/home/btech1/isro/models/optimized_mscff.keras"
OUTPUT_DIR = f"/home/btech1/isro/training_output/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================= FOCAL LOSS =======================
def focal_loss(gamma=2.0, alpha=None):
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_pred * y_true, axis=-1)
        focal_factor = tf.pow(1.0 - p_t, gamma)
        
        if alpha is not None:
            alpha_weights = tf.reduce_sum(alpha * y_true, axis=-1)
            focal_ce = focal_factor * ce * tf.expand_dims(alpha_weights, -1)
        else:
            focal_ce = focal_factor * ce
            
        return tf.reduce_mean(tf.reduce_sum(focal_ce, axis=-1))
    return focal_loss_fn

# ======================= U-NET MODEL ARCHITECTURE =======================
def build_model(input_shape=(PATCH_SIZE, PATCH_SIZE, 3), num_classes=NUM_CLASSES, width_multiplier=WIDTH_MULTIPLIER):
    """U-Net model with width multiplier for reduced complexity"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    def conv_block(x, filters):
        # Apply width multiplier
        filters = int(filters * width_multiplier)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    # Encoder path
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bridge
    b = conv_block(p4, 1024)
    
    # Decoder with skip connections
    u1 = layers.UpSampling2D((2, 2), interpolation='bilinear')(b)
    u1 = layers.concatenate([u1, c4])
    u1 = conv_block(u1, 512)
    
    u2 = layers.UpSampling2D((2, 2), interpolation='bilinear')(u1)
    u2 = layers.concatenate([u2, c3])
    u2 = conv_block(u2, 256)
    
    u3 = layers.UpSampling2D((2, 2), interpolation='bilinear')(u2)
    u3 = layers.concatenate([u3, c2])
    u3 = conv_block(u3, 128)
    
    u4 = layers.UpSampling2D((2, 2), interpolation='bilinear')(u3)
    u4 = layers.concatenate([u4, c1])
    u4 = conv_block(u4, 64)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation="softmax", dtype='float32')(u4)
    
    return tf.keras.Model(inputs, outputs, name="UNet_Optimized")

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
                
                img_patch = src.read(window=window).transpose(1, 2, 0).astype(np.float32) / 65535.0
                
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
                        if mask_patch.shape != (PATCH_SIZE, PATCH_SIZE):
                            mask_patch = np.pad(
                                mask_patch, 
                                [(0, PATCH_SIZE - mask_patch.shape[0]), 
                                 (0, PATCH_SIZE - mask_patch.shape[1])],
                                mode='reflect'
                            )
                        masks.append(mask_patch)
    
    return np.array(patches), np.array(masks) if mask_path else None

def create_tfrecords(image_dir, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_names = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    
    for name in tqdm(image_names, desc="Creating TFRecords"):
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
        os.makedirs(PATCH_DIR, exist_ok=True)
        image_names = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif')]
        train_names, val_names = train_test_split(image_names, test_size=0.2, random_state=42)
        
        create_tfrecords(IMAGE_DIR, MASK_DIR, train_dir)
        create_tfrecords(IMAGE_DIR, MASK_DIR, val_dir)
    
    return train_dir, val_dir

def parse_tfrecord(example):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, features)
    
    image = tf.io.decode_raw(example['image'], tf.float32)
    image = tf.reshape(image, (PATCH_SIZE, PATCH_SIZE, 3))
    
    mask = tf.io.decode_raw(example['mask'], tf.uint8)
    mask = tf.reshape(mask, (PATCH_SIZE, PATCH_SIZE))
    
    return image, mask

def create_dataset(tfrecord_dir, batch_size=BATCH_SIZE):
    tfrecords = glob.glob(os.path.join(tfrecord_dir, "*.tfrecord"))
    dataset = tf.data.TFRecordDataset(tfrecords)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ======================= METRICS =======================
class ClassMetrics(tf.keras.metrics.Metric):
    def __init__(self, class_id, metric_fn, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.metric_fn = metric_fn
        self.metric = self.metric_fn()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        self.metric.update_state(
            tf.equal(y_true, self.class_id),
            tf.equal(y_pred, self.class_id)
        )
    
    def result(self):
        return self.metric.result()
    
    def reset_state(self):
        self.metric.reset_state()

def get_metrics():
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    
    for i, name in enumerate(CLASS_NAMES):
        metrics.extend([
            ClassMetrics(i, lambda: tf.keras.metrics.IoU(num_classes=2), name=f'iou_{name}'),
            ClassMetrics(i, lambda: tf.keras.metrics.Precision(), name=f'precision_{name}'),
            ClassMetrics(i, lambda: tf.keras.metrics.Recall(), name=f'recall_{name}'),
        ])
    
    # Add overall metrics
    metrics.extend([
        tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES, name='mean_iou'),
        tf.keras.metrics.Precision(name='mean_precision'),
        tf.keras.metrics.Recall(name='mean_recall'),
    ])
    
    return metrics

# ======================= VISUALIZATION =======================
def save_history_plots(history):
    plt.figure(figsize=(15, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    # IoU
    plt.subplot(2, 2, 3)
    for metric in history.history:
        if 'iou_' in metric and 'val_' not in metric:
            plt.plot(history.history[metric], label=metric)
    plt.title('Class-wise IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()

def save_confusion_matrix(model, dataset):
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
    
    return cm

def save_roc_curve(model, dataset):
    y_true, y_prob = [], []
    
    for images, masks in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(masks.numpy().flatten())
        y_prob.extend(preds.reshape(-1, NUM_CLASSES))
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    plt.figure(figsize=(10, 8))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, 
                 label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'))
    plt.close()

# ======================= TRAINING =======================
def calculate_class_weights(dataset):
    class_counts = np.zeros(NUM_CLASSES)
    
    for _, masks in dataset.take(100):  # Use first 100 batches
        for i in range(NUM_CLASSES):
            class_counts[i] += tf.reduce_sum(tf.cast(masks == i, tf.float32)).numpy()
    
    total = np.sum(class_counts)
    class_weights = total / (NUM_CLASSES * class_counts + 1e-8)
    return class_weights / np.sum(class_weights)

def train_model():
    # Prepare data
    train_dir, val_dir = prepare_datasets()
    train_ds = create_dataset(train_dir)
    val_ds = create_dataset(val_dir)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_ds)
    print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights))}")
    
    # Build and compile model
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(gamma=2.0, alpha=class_weights),
        metrics=get_metrics()
    )
    
    # Callbacks
    callbacks = [
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
    save_history_plots(history)
    save_confusion_matrix(model, val_ds)
    save_roc_curve(model, val_ds)
    
    print(f"\nTraining complete! Results saved to: {OUTPUT_DIR}")