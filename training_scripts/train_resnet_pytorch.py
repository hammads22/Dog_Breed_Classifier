import os
import time
import requests
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ==========================================
# 1. Configuration & Setup
# ==========================================
DATA_DIR = "DatasetStore"
BATCH_SIZE = 128        # Increased for faster batch processing
EPOCHS = 30
LEARNING_RATE = 0.002   # Scaled up slightly to match the larger batch size
NUM_CLASSES = 120       # Stanford Dogs has 120 breeds

# ==========================================
# 2. Dataset Download Function
# ==========================================
def download_extract():
    print("\n--- [CHECKPOINT 2] Checking Dataset ---")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print("Dataset not found locally. Starting download...")
        img_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
        tar_path = os.path.join(DATA_DIR, "images.tar")
        
        response = requests.get(img_url, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading Images")
        with open(tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
                    
        print("\nExtracting dataset (this might take a minute)...")
        with tarfile.open(tar_path) as tfile:
            tfile.extractall(DATA_DIR)
        print("Download and extraction complete.")
    else:
        print("Dataset already exists locally. Skipping download.")

# ==========================================
# MAIN EXECUTION BLOCK (Windows Safe)
# ==========================================
if __name__ == '__main__':
    print("\n--- [CHECKPOINT 1] Configuration & Hardware Check ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # SPEED BOOST: Enable CuDNN Benchmarking for consistent image sizes
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("CuDNN benchmarking enabled.")

    download_extract()

    # ==========================================
    # 3. Data Augmentation & Loading
    # ==========================================
    print("\n--- [CHECKPOINT 3] Preparing Data Loaders & Augmentation ---")
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_folder = os.path.join(DATA_DIR, 'Images')
    full_dataset = datasets.ImageFolder(root=image_folder, transform=data_transforms)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # SPEED BOOST: pin_memory=True speeds up RAM to GPU transfers
    # WINDOWS FIX: num_workers=0 prevents the multiprocessing crash
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Total training images: {len(train_dataset)}")
    print(f"Total validation images: {len(val_dataset)}")

    # ==========================================
    # 4. Build the Model
    # ==========================================
    print("\n--- [CHECKPOINT 4] Building ResNet50 Model ---")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # SPEED BOOST: Initialize the Automatic Mixed Precision (AMP) Scaler
    scaler = torch.amp.GradScaler('cuda')
    print("Model built and AMP initialized successfully.")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print("\n--- [CHECKPOINT 5] Starting Training Loop ---")
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_loop = tqdm(train_loader, desc="Training  ", leave=False)
        for inputs, labels in train_loop:
            # pin_memory allows us to use non_blocking=True for slightly faster transfers
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # SPEED BOOST: Run the forward pass in 16-bit mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Use the scaler to prevent underflow during the backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            train_loop.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        val_loop = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Validation should also use AMP for speed
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        
        print(f"-> Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"-> Val Loss:   {val_epoch_loss:.4f} | Val Acc:   {val_epoch_acc:.4f}")
        
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), 'Resnet50_PyTorch.pt')
            print("   [+] New best model saved!")

    print("\n--- [CHECKPOINT 6] Training Complete ---")
    time_elapsed = time.time() - start_time
    print(f"Total time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:4f}")