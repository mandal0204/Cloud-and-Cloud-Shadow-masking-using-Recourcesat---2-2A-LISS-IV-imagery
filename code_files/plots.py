import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import random
from sklearn.metrics import confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay
import tifffile as tiff
import lmdb
import pickle
import zlib
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import cv2
import torch.nn.functional as F
import numpy.core.multiarray  # Required for safe loading

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Update these paths to match your environment
val_lmdb_path = "/home/btech1/isro/dataset/val/patches.lmdb"
weights_path = "/home/btech1/isro/training_output/final_model.pth"
output_dir = "/home/btech1/isro/training_output"
patch_size = 256
batch_size = 12
num_classes = 3
class_names = ["Background", "Cloud", "Cloud Shadow"]
classes = ["background", "cloud", "shadow"]

# --- Model Architecture ---
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
        
        # Store feature maps and gradients for Grad-CAM
        self.gradients = None
        self.feature_maps = None

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        
        # Store feature maps for Grad-CAM
        self.feature_maps = x
        
        # Register hook for gradients
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
            
        skip_connections = skip_connections[::-1]
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((skip, x), dim=1)
            x = self.up_convs[i](x)
        return self.final_conv(x)
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.feature_maps

# --- Dataset Loading ---
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
        for idx in tqdm(random.sample(range(len(train_dataset)), min(500, len(train_dataset))), desc="Processing Training Data"):
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
        for idx in tqdm(random.sample(range(len(val_dataset)), min(500, len(val_dataset))), desc="Processing Validation Data"):
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

def plot_heatmap(model, val_dataset, output_dir, num_heatmaps=5):
    """Generate segmentation heatmaps for validation examples"""
    print(f"Generating {num_heatmaps} segmentation heatmaps...")
    
    # Create custom colormap
    cmap = ListedColormap(['#000000', '#1f77b4', '#ff7f0e'])  # Black, Blue, Orange
    
    for heatmap_idx in range(num_heatmaps):
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
        plt.savefig(os.path.join(output_dir, f"segmentation_heatmap_{heatmap_idx}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_heatmaps} segmentation heatmaps")

# --- Grad-CAM Visualization Function ---
def plot_gradcam(model, val_dataset, output_dir, num_classes=3, num_images=3):
    """Generate Grad-CAM visualizations for validation examples"""
    print("Generating Grad-CAM visualizations...")
    
    # Select random images
    indices = random.sample(range(len(val_dataset)), num_images)
    
    for i, idx in enumerate(indices):
        img, mask = val_dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        # Get the model prediction
        model.eval()
        output = model(input_tensor)
        
        # Generate CAM for each class
        for class_idx in range(num_classes):
            # Reset gradients
            model.zero_grad()
            
            # Create target for specific class
            one_hot = torch.zeros_like(output)
            one_hot[:, class_idx, :, :] = 1
            
            # Compute gradients for the specific class
            output.backward(gradient=one_hot, retain_graph=True)
            
            # Get the gradients and feature maps
            gradients = model.get_activations_gradient()
            feature_maps = model.get_activations()
            
            if gradients is None or feature_maps is None:
                print(f"Gradients or feature maps not available for image {i}, class {class_idx}")
                continue
            
            # Pool the gradients across the channels
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            
            # Weight the feature maps by the gradients
            weighted_feature_maps = torch.zeros_like(feature_maps)
            for j in range(feature_maps.shape[1]):
                weighted_feature_maps[:, j, :, :] = pooled_gradients[j] * feature_maps[:, j, :, :]
            
            # Average across channels and apply ReLU
            cam = torch.mean(weighted_feature_maps, dim=1).squeeze()
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cam = cam.detach().cpu().numpy()
            
            # Resize to input size
            cam = cv2.resize(cam, (patch_size, patch_size))
            cam = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            
            # Convert original image to displayable format
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.uint8(255 * img_np)
            
            # Superimpose heatmap on original image
            superimposed_img = heatmap * 0.4 + img_np * 0.6
            superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
            
            # Create figure
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title("Original Image")
            plt.axis('off')
            
            # Heatmap
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap)
            plt.title(f"Grad-CAM Heatmap\nClass: {class_names[class_idx]}")
            plt.axis('off')
            
            # Superimposed
            plt.subplot(1, 3, 3)
            plt.imshow(superimposed_img)
            plt.title(f"Class Activation Map\nClass: {class_names[class_idx]}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"gradcam_{i}_class_{class_idx}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Saved {num_images*num_classes} Grad-CAM visualizations")

# --- Safe Model Loading ---
def load_model_safe(model, weights_path):
    """Load model weights with safety for PyTorch 2.6+"""
    try:
        # First try with weights_only=True (safe mode)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Safe loading failed: {str(e)}")
        print("Attempting unsafe load (use only if you trust the source)")
        
        # Add safe global for numpy scalar if needed
        from numpy.core.multiarray import scalar
        torch.serialization.add_safe_globals([scalar])
        
        # Try safe loading again
        try:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        except:
            # Fallback to unsafe loading
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Load state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

# --- Main Execution ---
if __name__ == "__main__":
    # Load metrics
    metrics_path = os.path.join(output_dir, "metrics_log.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
    df = pd.read_csv(metrics_path)
    
    # Load model
    model = UNet(in_channels=3, out_channels=3).to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    
    model = load_model_safe(model, weights_path)
    model.eval()
    
    # Load validation dataset
    val_dataset = LMDBPatchDataset(val_lmdb_path)
    
    # Generate plots
    print("\nGenerating plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics-based plots
    plot_metric_comparison("accuracy", classes, df, output_dir)
    plot_metric_comparison("iou", classes, df, output_dir)
    plot_metric_comparison("precision", classes, df, output_dir)
    plot_metric_comparison("recall", classes, df, output_dir)
    plot_metric_comparison("f1-score", classes, df, output_dir)
    
    # Loss and LR plots
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
    
    # Model-based plots
    plot_precision_recall_curves(model, val_dataset, val_dataset, output_dir)
    plot_confusion_matrix(model, val_dataset, output_dir)
    plot_heatmap(model, val_dataset, output_dir, num_heatmaps=5)  # Generate 5 heatmaps
    
    # Grad-CAM visualization
    plot_gradcam(model, val_dataset, output_dir, num_classes, num_images=3)
    
    print("\nAll plots generated in:", os.path.abspath(output_dir))