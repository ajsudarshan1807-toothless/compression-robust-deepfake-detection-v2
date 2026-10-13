import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import os
import sys
import random
from PIL import Image
import io

sys.path.append('src')
from models.fusion_model import FusionModelV2

class RandomJPEGCompress(object):
    def __init__(self, min_quality=30, max_quality=95, p=0.5):
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.min_quality, self.max_quality)
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
            # Need to convert back to RGB to ensure no issue with PIL modes
            return img.convert('RGB')
        return img

def train_clip_final():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Adding RandomJPEGCompress and Random Horizontal Flip for Augmentation
    transform = transforms.Compose([
        RandomJPEGCompress(min_quality=20, max_quality=80, p=0.7), # 70% chance to compress heavily
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    class GenericDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, transform):
            self.hf_ds = hf_ds
            self.transform = transform
            
        def __len__(self):
            return len(self.hf_ds)
            
        def __getitem__(self, idx):
            item = self.hf_ds[idx]
            image = item['image'].convert('RGB')
            label = 1 if item['label'] == 0 else 0
            return self.transform(image), torch.tensor([label], dtype=torch.float32)

    class FaceDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, transform):
            self.hf_ds = hf_ds
            self.transform = transform
            
        def __len__(self):
            return len(self.hf_ds)
            
        def __getitem__(self, idx):
            item = self.hf_ds[idx]
            image = item['image'].convert('RGB')
            label = 1 if item['label'] == 0 else 0 
            return self.transform(image), torch.tensor([label], dtype=torch.float32)

    print("Loading Generic Dataset (10,000 images)...")
    ds_generic = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split='train')
    ds_generic = ds_generic.shuffle(seed=42).select(range(10000))
    generic_data = GenericDataset(ds_generic, transform)
    
    print("Loading Specialized Face Dataset (Thien0103)...")
    ds_face = load_dataset("Thien0103/DeepFake_Extracted_Face_Images", split='train')
    ds_face = ds_face.shuffle(seed=42) # CRITICAL FIX: Must shuffle so real faces aren't stuck at the end!
    face_data = FaceDataset(ds_face, transform)
    
    combined_data = torch.utils.data.ConcatDataset([generic_data, face_data])
    loader = DataLoader(combined_data, batch_size=64, shuffle=False)
    
    print(f"Total Combined Images: {len(combined_data)}")
    
    model = FusionModelV2(pretrained_spatial=True, freeze_clip=True).to(device)
    model.eval()
    
    all_features = []
    all_labels = []
    
    print("Extracting features into RAM (Augmented)...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(device)
            spatial_features = model.spatial(imgs, return_features=True)
            frequency_features = model.frequency(imgs, return_features=True)
            fused = torch.cat((spatial_features, frequency_features), dim=1)
            
            all_features.append(fused.cpu())
            all_labels.append(labels)
            
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    # Shuffle the features and labels randomly to mix the generic and face datasets
    torch.manual_seed(42)
    shuffled_indices = torch.randperm(len(features_tensor))
    features_tensor = features_tensor[shuffled_indices]
    labels_tensor = labels_tensor[shuffled_indices]
    
    train_size = int(0.9 * len(features_tensor))
    train_features = features_tensor[:train_size]
    train_labels = labels_tensor[:train_size]
    val_features = features_tensor[train_size:]
    val_labels = labels_tensor[train_size:]
    
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    optimizer = optim.Adam(model.fusion_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    epochs = 15
    print("Training robust classification head (with Frequency Dropout)...")
    for epoch in range(epochs):
        model.fusion_head.train()
        total_loss, correct, total = 0, 0, 0
        
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            
            # Apply Frequency Dropout! 50% chance to zero out the frequency features (last 128 dims)
            if random.random() < 0.5:
                # CLIP spatial is first 768 dims. Frequency is next 128.
                feats[:, 768:] = 0.0

            optimizer.zero_grad()
            outputs = model.fusion_head(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        model.fusion_head.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                
                # During validation, we also evaluate robustness by randomly dropping frequency half the time
                if random.random() < 0.5:
                    feats[:, 768:] = 0.0

                outputs = model.fusion_head(feats)
                val_loss += criterion(outputs, labels).item()
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {correct/total*100:.2f}% | Val Acc: {val_correct/val_total*100:.2f}%")
        
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model_h_clip_robust.pt')
    print("Saved dual-robust model to checkpoints/model_h_clip_robust.pt")

if __name__ == "__main__":
    train_clip_final()
