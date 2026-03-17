import os
import numpy as np
from glob import glob
import tifffile as tiff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import time
import lmdb
import pickle
from tqdm import tqdm
from torch.utils.data import Subset
import random
import gc
import zlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_img_dir = "/home/btech1/isro/dataset/train/images"
train_mask_dir = "/home/btech1/isro/dataset/train/masks"
val_img_dir = "/home/btech1/isro/dataset/val/images"
val_mask_dir = "/home/btech1/isro/dataset/val/masks"
train_lmdb_path = "/home/btech1/isro/dataset/train/patches.lmdb"
val_lmdb_path = "/home/btech1/isro/dataset/val/patches.lmdb"
patch_size = 256
overlap = 0
batch_size = 12
num_classes = 3
classes = ["background", "cloud", "cloud_shadow"]
class_names = ["Background", "Cloud", "Cloud Shadow"]
max_epochs = 45
early_stop_patience = 10
reduce_lr_patience = 5
min_lr = 1e-6
resume = True  # Set to True to enable resume
output_dir = "/home/btech1/isro/training_output"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {os.path.abspath(output_dir)}")

# --- LMDB Patch Writer ---
def write_lmdb(images_dir, masks_dir, lmdb_path, patch_size=256, overlap=128):
    images = sorted(glob(os.path.join(images_dir, "*.tif")))
    map_size = 1024**4  # 1TB
    env = lmdb.open(lmdb_path, map_size=map_size)
    idx = 0
    with env.begin(write=True) as txn:
        for img_path in tqdm(images, desc=f"Writing {lmdb_path}"):
            img_name = os.path.basename(img_path)
            mask_name = f"mask_{img_name}"
            mask_path = os.path.join(masks_dir, mask_name)
            if not os.path.exists(mask_path):
                print(f"Mask not found for {img_name}, skipping.")
                continue
            img = tiff.imread(img_path)
            msk = tiff.imread(mask_path)
            height, width = img.shape[:2]
            for y in range(0, height - patch_size + 1, patch_size - overlap):
                for x in range(0, width - patch_size + 1, patch_size - overlap):
                    patch = img[y:y+patch_size, x:x+patch_size, :]
                    patch_mask = msk[y:y+patch_size, x:x+patch_size]
                    
                    # Robust NaN/Inf handling
                    if np.isnan(patch).any() or np.isinf(patch).any():
                        print(f"NaN or Inf found in patch {idx} from {img_name}")
                        patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    patch = patch.astype(np.float32)

                    # Handle constant patches
                    if np.ptp(patch) > 1e-8:
                        patch -= patch.min()
                        max_val = patch.max()
                        if max_val > 1e-8:
                            patch /= max_val
                        else:
                            patch[:] = 0
                    else:
                        patch[:] = 0
                    
                    patch = (patch * 255).astype(np.uint8)
                    txn.put(f"patch_{idx}".encode(), zlib.compress(patch.tobytes()))
                    txn.put(f"mask_{idx}".encode(), patch_mask.tobytes())
                    txn.put(f"shape_{idx}".encode(), pickle.dumps((patch.shape, patch_mask.shape)))
                    idx += 1
        txn.put(b'length', pickle.dumps(idx))
    env.close()
    print(f"LMDB created: {lmdb_path}, total patches: {idx}")

# --- LMDB Dataset ---
class LMDBPatchDataset(Dataset):
    def __init__(self, lmdb_path, patch_channels=3, patch_size=256):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'length'))
        self.patch_channels = patch_channels
        self.patch_size = patch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            patch_shape, mask_shape = pickle.loads(txn.get(f"shape_{idx}".encode()))
            patch = np.frombuffer(zlib.decompress(txn.get(f"patch_{idx}".encode())), dtype=np.uint8).reshape(patch_shape)
            patch = patch.astype(np.float32) / 255.0
            mask = np.frombuffer(txn.get(f"mask_{idx}".encode()), dtype=np.uint8).reshape(mask_shape)
        
        # Convert to tensors
        patch = torch.from_numpy(patch.transpose(2,0,1).copy()).float()
        mask = torch.from_numpy(mask.copy()).long()
        
        # Additional check for NaNs/Infs
        if torch.isnan(patch).any() or torch.isinf(patch).any():
            print(f"Found NaN value after reading batch {idx}")
            patch = torch.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)
            
        return patch, mask

# --- Check for LMDB ---
def lmdb_exists(lmdb_path):
    return (
        os.path.isdir(lmdb_path) and
        os.path.isfile(os.path.join(lmdb_path, "data.mdb")) and
        os.path.isfile(os.path.join(lmdb_path, "lock.mdb"))
    )


if not lmdb_exists(train_lmdb_path):
    print("Creating training LMDB...")
    write_lmdb(train_img_dir, train_mask_dir, train_lmdb_path, patch_size, overlap)
else:
    print("Training LMDB found.")

if not lmdb_exists(val_lmdb_path):
    print("Creating validation LMDB...")
    write_lmdb(val_img_dir, val_mask_dir, val_lmdb_path, patch_size, overlap)
else:
    print("Validation LMDB found.")

# --- DataLoaders ---
train_dataset = LMDBPatchDataset(train_lmdb_path, patch_channels=3, patch_size=patch_size)
val_dataset_full = LMDBPatchDataset(val_lmdb_path, patch_channels=3, patch_size=patch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
print(f"Train dataset: {len(train_dataset)} patches")
print(f"Validation dataset: {len(val_dataset_full)} patches")
print(f"Batch size: {batch_size}")

# --- Model ---
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

model = UNet(in_channels=3, out_channels=3).to(device)

# Weight classes to handle imbalance
class_weights = torch.tensor([0.1, 1.0, 1.0]).float().to(device)  # Adjust based on your class distribution
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=reduce_lr_patience, min_lr=min_lr)

# --- New Metrics Functions ---
def update_confusion_matrix(cm, y_true, y_pred, num_classes):
    """Update confusion matrix incrementally"""
    flat_true = y_true.flatten()
    flat_pred = y_pred.flatten()
    idx = num_classes * flat_true + flat_pred
    counts = np.bincount(idx, minlength=num_classes**2)
    return cm + counts.reshape(num_classes, num_classes)

def compute_metrics_from_cm(cm, num_classes=3):
    """Compute metrics from confusion matrix"""
    total = cm.sum()
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    
    # Overall accuracy
    overall_acc = TP.sum() / total if total else 0.0
    
    # Precision, recall, F1
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # IoU
    iou = TP / (TP + FP + FN + 1e-10)
    
    # Macro averages
    macro_precision = np.nanmean(precision)
    macro_recall = np.nanmean(recall)
    macro_f1 = np.nanmean(f1)
    macro_iou = np.nanmean(iou)
    
    return {
        "overall_acc": overall_acc,
        "class_acc": (TP + (total - TP - FP - FN)) / total,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "iou": iou.tolist(),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_iou": macro_iou
    }

val_len = len(val_dataset_full)
indices = list(range(val_len))
random.shuffle(indices)
val_groups = np.array_split(indices, max_epochs)  # List of arrays, one per epoch

# --- Resume logic ---
start_epoch = 1
if resume and os.path.exists(os.path.join(output_dir, "metrics_log.csv")):
    df = pd.read_csv(os.path.join(output_dir, "metrics_log.csv"))
    if not df.empty:
        last_epoch = int(df['epoch'].max())
        start_epoch = last_epoch + 1
        checkpoint_path = os.path.join(output_dir, f"best_model_epoch_{last_epoch}.pth")
        if os.path.exists(checkpoint_path):
            print(f"Resuming from epoch {start_epoch}, loading {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Option 1: Support both old and new checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                best_val_iou = checkpoint.get('best_val_iou', 0)
                epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
                print("Loaded full checkpoint (model, optimizer, scheduler).")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model weights only (no optimizer/scheduler state).")
        else:
            print(f"No checkpoint found for epoch {last_epoch}, starting from scratch.")
    else:
        print("metrics_log.csv is empty, starting from scratch.")
else:
    print("No metrics_log.csv found, starting from scratch.")

# --- Training Loop ---
metrics_log = []
train_losses, val_losses, lrs = [], [], []
best_val_iou = 0
epochs_no_improve = 0

for epoch in range(start_epoch, max_epochs+1):
    print(f"\n=== Epoch {epoch}/{max_epochs} ===")
    epoch_start_time = time.time()
    model.train()
    running_train_loss = 0.0
    
    # Initialize confusion matrix for epoch
    train_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Progress bar
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{max_epochs}")

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        unique_vals = torch.unique(masks)

        if not torch.all((unique_vals >= 0) & (unique_vals < num_classes)):
            print(f"Invalid mask values: {unique_vals}")

        # Skip batches with NaN/inf
        if torch.isnan(images).any() or torch.isinf(images).any():
            print(f"Skipping batch {batch_idx} due to NaN/Inf in input")
            pbar.update(1)
            continue
            
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"NaN/Inf in model output at batch {batch_idx}")
            print(f"Batch min: {outputs.min().item()}, max: {outputs.max().item()}, mean: {outputs.mean().item()}")
            pbar.update(1)
            continue

        loss.backward()

        # Skip NaN loss batches
        if torch.isnan(loss):
            print(f"Skipping batch {batch_idx} due to NaN loss")
            pbar.update(1)
            continue
            
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate metrics incrementally
        with torch.no_grad():
            preds = outputs.argmax(dim=1).cpu().numpy()
            trues = masks.cpu().numpy()
            
            # Update confusion matrix
            train_cm = update_confusion_matrix(train_cm, trues, preds, num_classes)
            
            # Compute batch metrics for progress bar
            batch_cm = np.zeros((num_classes, num_classes))
            batch_cm = update_confusion_matrix(batch_cm, trues, preds, num_classes)
            batch_metrics = compute_metrics_from_cm(batch_cm, num_classes)
            
        running_train_loss += loss.item() * images.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}", 
            'Accuracy': f"{batch_metrics['overall_acc']:.4f}",
            'IoU': f"{batch_metrics['macro_iou']:.3f}",
            'Precision': f"{batch_metrics['macro_precision']:.3f}",
            'Recall': f"{batch_metrics['macro_recall']:.3f}",
            'F1_Score': f"{batch_metrics['macro_f1']:.3f}",
            'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        pbar.update(1)
    
    pbar.close()
    
    # Compute final training metrics for epoch
    train_metrics = compute_metrics_from_cm(train_cm, num_classes)
    avg_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    epoch_time = time.time() - epoch_start_time
    
    # --- Validation ---
    val_indices = val_groups[epoch-1]
    val_dataset = Subset(val_dataset_full, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Evaluating on {len(val_dataset)} validation patches...")
    model.eval()
    running_val_loss = 0.0
    val_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Use torch.no_grad for entire validation loop
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Skip NaN batches
            if not torch.isnan(loss):
                running_val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()
                trues = masks.cpu().numpy()
                val_cm = update_confusion_matrix(val_cm, trues, preds, num_classes)
    
    # Handle case where all batches were skipped
    if val_cm.sum() == 0:
        print("WARNING: All validation batches skipped due to NaN loss!")
        avg_val_loss = float('nan')
        metrics = {
            "overall_acc": 0,
            "class_acc": [0, 0, 0],
            "precision": [0, 0, 0],
            "recall": [0, 0, 0],
            "f1": [0, 0, 0],
            "iou": [0, 0, 0],
            "macro_precision": 0,
            "macro_recall": 0,
            "macro_f1": 0,
            "macro_iou": 0
        }
    else:
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        metrics = compute_metrics_from_cm(val_cm, num_classes)
    
    val_losses.append(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)
    
    # Log metrics
    epoch_log = {
        "epoch": epoch,
        "learning_rate": current_lr,
        "train_loss": avg_train_loss,
        "train_accuracy": train_metrics["overall_acc"],
        "train_accuracy_background": train_metrics["class_acc"][0],
        "train_accuracy_cloud": train_metrics["class_acc"][1],
        "train_accuracy_shadow": train_metrics["class_acc"][2],
        "train_iou": train_metrics["macro_iou"],
        "train_iou_background": train_metrics["iou"][0],
        "train_iou_cloud": train_metrics["iou"][1],
        "train_iou_shadow": train_metrics["iou"][2],
        "train_precision": train_metrics["macro_precision"],
        "train_precision_background": train_metrics["precision"][0],
        "train_precision_cloud": train_metrics["precision"][1],
        "train_precision_shadow": train_metrics["precision"][2],
        "train_recall": train_metrics["macro_recall"],
        "train_recall_background": train_metrics["recall"][0],
        "train_recall_cloud": train_metrics["recall"][1],
        "train_recall_shadow": train_metrics["recall"][2],
        "train_f1-score": train_metrics["macro_f1"],
        "train_f1-score_background": train_metrics["f1"][0],
        "train_f1-score_cloud": train_metrics["f1"][1],
        "train_f1-score_shadow": train_metrics["f1"][2],
        "val_loss": avg_val_loss,
        "val_accuracy": metrics["overall_acc"],
        "val_accuracy_background": metrics["class_acc"][0],
        "val_accuracy_cloud": metrics["class_acc"][1],
        "val_accuracy_shadow": metrics["class_acc"][2],
        "val_iou": metrics["macro_iou"],
        "val_iou_background": metrics["iou"][0],
        "val_iou_cloud": metrics["iou"][1],
        "val_iou_shadow": metrics["iou"][2],
        "val_precision": metrics["macro_precision"],
        "val_precision_background": metrics["precision"][0],
        "val_precision_cloud": metrics["precision"][1],
        "val_precision_shadow": metrics["precision"][2],
        "val_recall": metrics["macro_recall"],
        "val_recall_background": metrics["recall"][0],
        "val_recall_cloud": metrics["recall"][1],
        "val_recall_shadow": metrics["recall"][2],
        "val_f1-score": metrics["macro_f1"],
        "val_f1-score_background": metrics["f1"][0],
        "val_f1-score_cloud": metrics["f1"][1],
        "val_f1-score_shadow": metrics["f1"][2],
    }
    metrics_log.append(epoch_log)
    
    # Save metrics immediately after each epoch
    df = pd.DataFrame(metrics_log)
    df.to_csv(os.path.join(output_dir, "metrics_log.csv"), index=False)
    print("Saved metrics_log.csv")

    print(f"Epoch {epoch}: LR={current_lr:.6f}, Train Loss={avg_train_loss:.4f}, "
      f"Accuracy={train_metrics['overall_acc']:.4f}, IoU={train_metrics['macro_iou']:.3f}, "
      f"Precision={train_metrics['macro_precision']:.3f}, Recall={train_metrics['macro_recall']:.3f}, F1_Score={train_metrics['macro_f1']:.3f}, "
      f"Val_Loss={avg_val_loss:.4f}, Val_Accuracy={metrics['overall_acc']:.3f}, "
      f"Val_IoU={metrics['macro_iou']:.3f}, Val_Precision={metrics['macro_precision']:.3f}, "
      f"Val_Recall={metrics['macro_recall']:.3f}, Val_F1={metrics['macro_f1']:.3f}")

    # Memory cleanup
    del images, masks, outputs, preds, trues
    torch.cuda.empty_cache()
    gc.collect()

    # Saving the best model...
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_iou': best_val_iou,
        'epochs_no_improve': epochs_no_improve,
    }, os.path.join(output_dir, f"best_model_epoch_{epoch}.pth"))

    # Update scheduler and check for improvement
    scheduler.step(metrics["macro_iou"])
    if metrics["macro_iou"] > best_val_iou:
        best_val_iou = metrics["macro_iou"]
        epochs_no_improve = 0
        print(f"Saved best model checkpoint at epoch {epoch}.")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs.")
        if epochs_no_improve >= early_stop_patience:
            print("Early stopping triggered.")
            break

# --- Save Final Model ---
torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
print("Saved final model weights.")

# --- Plotting Functions ---
def plot_metric_comparison(metric_name, classes, df, output_dir):
    """Plot overall and per-class metrics for both train and validation"""
    plt.figure(figsize=(14, 12))
    plt.suptitle(f"{metric_name.capitalize()} vs Epoch", fontsize=16)
    
    # Create custom color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes) + 1))
    
    # Training plot
    plt.subplot(2, 1, 1)
    plt.title("Training")
    plt.plot(df['epoch'], df[f'train_{metric_name}'], label='Overall', color=colors[0], linewidth=3)
    for i, cls in enumerate(classes):
        plt.plot(df['epoch'], df[f'train_{metric_name}_{cls}'], label=f'{cls}', color=colors[i+1], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Validation plot
    plt.subplot(2, 1, 2)
    plt.title("Validation")
    plt.plot(df['epoch'], df[f'val_{metric_name}'], label='Overall', color=colors[0], linewidth=3)
    for i, cls in enumerate(classes):
        plt.plot(df['epoch'], df[f'val_{metric_name}_{cls}'], label=f'{cls}', color=colors[i+1], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(output_dir, f"{metric_name}_comparison.png"), dpi=300)
    plt.close()
    print(f"Saved {metric_name}_comparison.png")

def plot_precision_recall_curves(model, train_dataset, val_dataset, output_dir):
    """Generate precision-recall curves for training and validation sets"""
    print("Generating precision-recall curves...")
    
    plt.figure(figsize=(10, 8))
    
    # Process training data
    train_true = [[] for _ in range(num_classes)]
    train_scores = [[] for _ in range(num_classes)]
    
    model.eval()
    with torch.no_grad():
        for idx in tqdm(random.sample(range(len(train_dataset)), min(200, len(train_dataset))), desc="Processing Training Data"):
            img, mask = train_dataset[idx]
            img = img.unsqueeze(0).to(device)
            output = model(img)
            probas = torch.softmax(output, dim=1).cpu().numpy()[0]
            mask = mask.numpy()
            
            for i in range(num_classes):
                train_true[i].append((mask == i).astype(int).flatten())
                train_scores[i].append(probas[i].flatten())
    
    # Plot training curves
    for i, cls in enumerate(classes):
        y_true = np.concatenate(train_true[i])
        y_scores = np.concatenate(train_scores[i])
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, linestyle='-', linewidth=2, 
                 label=f"Train {class_names[i]}", alpha=0.8)
    
    # Process validation data
    val_true = [[] for _ in range(num_classes)]
    val_scores = [[] for _ in range(num_classes)]
    
    with torch.no_grad():
        for idx in tqdm(random.sample(range(len(val_dataset)), min(200, len(val_dataset))), desc="Processing Validation Data"):
            img, mask = val_dataset[idx]
            img = img.unsqueeze(0).to(device)
            output = model(img)
            probas = torch.softmax(output, dim=1).cpu().numpy()[0]
            mask = mask.numpy()
            
            for i in range(num_classes):
                val_true[i].append((mask == i).astype(int).flatten())
                val_scores[i].append(probas[i].flatten())
    
    # Plot validation curves
    for i, cls in enumerate(classes):
        y_true = np.concatenate(val_true[i])
        y_scores = np.concatenate(val_scores[i])
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, linestyle='--', linewidth=2, 
                 label=f"Val {class_names[i]}", alpha=0.8)
    
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=300)
    plt.close()
    print("Saved precision_recall_curves.png")

def plot_confusion_matrix(model, val_dataset, output_dir):
    """Generate confusion matrix for validation set"""
    print("Generating confusion matrix...")
    
    # Get subset of validation data
    indices = random.sample(range(len(val_dataset)), min(1000, len(val_dataset)))
    subset = Subset(val_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_trues = []
    
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(masks.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_trues = np.concatenate(all_trues, axis=0).flatten()
    
    cm = confusion_matrix(all_trues, all_preds, labels=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (Validation Set)")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    print("Saved confusion_matrix.png")
    return cm

def plot_heatmap(model, val_dataset, output_dir):
    """Generate segmentation heatmap for a validation example"""
    print("Generating segmentation heatmap...")
    
    # Find an example with all classes present
    found = False
    for idx in random.sample(range(len(val_dataset)), 50):
        _, mask = val_dataset[idx]
        unique = np.unique(mask.numpy())
        if len(unique) == num_classes:
            found = True
            break
    
    if not found:
        idx = random.randint(0, len(val_dataset) - 1)
        print("Could not find example with all classes, using random sample")
    
    img, true_mask = val_dataset[idx]
    
    model.eval()
    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device))
        pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()
    
    # Create custom colormap
    cmap = ListedColormap(['#000000', '#1f77b4', '#ff7f0e'])  # Black, Blue, Orange
    
    plt.figure(figsize=(15, 6))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    plt.title("Original Image")
    plt.axis('off')
    
    # True mask
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask.numpy(), cmap=cmap, vmin=0, vmax=num_classes-1)
    plt.title("True Mask")
    plt.axis('off')
    
    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap=cmap, vmin=0, vmax=num_classes-1)
    plt.title("Predicted Mask")
    plt.axis('off')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cmap(i), edgecolor='k', label=class_names[i]) 
                      for i in range(num_classes)]
    plt.figlegend(handles=legend_elements, loc='lower center', ncol=num_classes, 
                 bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(output_dir, "segmentation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved segmentation_heatmap.png")

'''
# --- Generate All Plots ---
print("\nGenerating plots...")
df = pd.read_csv(os.path.join(output_dir, "metrics_log.csv"))
if not df.empty:
    # Plot accuracy comparison
    plot_metric_comparison("accuracy", ["background", "cloud", "shadow"], df, output_dir)
    
    # Plot IoU comparison
    plot_metric_comparison("iou", ["background", "cloud", "shadow"], df, output_dir)
    
    # Plot precision comparison
    plot_metric_comparison("precision", ["background", "cloud", "shadow"], df, output_dir)
    
    # Plot recall comparison
    plot_metric_comparison("recall", ["background", "cloud", "shadow"], df, output_dir)
    
    # Plot F1-score comparison
    plot_metric_comparison("f1-score", ["background", "cloud", "shadow"], df, output_dir)
    
    # Loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2)
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "loss_vs_epoch.png"), dpi=300)
    plt.close()
    print("Saved loss_vs_epoch.png")
    
    # Learning rate plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['learning_rate'], label='Learning Rate', linewidth=2)
    plt.title("Learning Rate vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, "lr_vs_epoch.png"), dpi=300)
    plt.close()
    print("Saved lr_vs_epoch.png")
    
    # Precision-Recall curves
    plot_precision_recall_curves(model, train_dataset, val_dataset_full, output_dir)
    
    # Confusion matrix
    plot_confusion_matrix(model, val_dataset_full, output_dir)
    
    # Segmentation heatmap
    plot_heatmap(model, val_dataset_full, output_dir)
else:
    print("No metrics to plot!")
'''    

print("\nTraining complete! All outputs saved to:", os.path.abspath(output_dir))