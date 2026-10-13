import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import os
import sys

sys.path.append('src')
from models.fusion_model import FusionModelV2

def train_clip_combined():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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
            # 0 = AiArtData (Fake), 1 = RealArt (Real)
            # Our convention: 0 = Real, 1 = Fake
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
            # Thien0103 dataset: 'fake' and 'real'
            # Assuming 0 = fake, 1 = real based on standard mapping, let's verify:
            label = 1 if item['label'] == 0 else 0 
            return self.transform(image), torch.tensor([label], dtype=torch.float32)

    print("Loading Generic Dataset (10,000 images)...")
    ds_generic = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split='train')
    ds_generic = ds_generic.shuffle(seed=42).select(range(10000))
    generic_data = GenericDataset(ds_generic, transform)
    
    print("Loading Specialized Face Dataset (Thien0103)...")
    ds_face = load_dataset("Thien0103/DeepFake_Extracted_Face_Images", split='train')
    face_data = FaceDataset(ds_face, transform)
    
    # Combine datasets
    combined_data = torch.utils.data.ConcatDataset([generic_data, face_data])
    loader = DataLoader(combined_data, batch_size=64, shuffle=False)
    
    print(f"Total Combined Images: {len(combined_data)}")
    
    print("Initializing FusionModelV2 for feature extraction...")
    model = FusionModelV2(pretrained_spatial=True, freeze_clip=True).to(device)
    model.eval()
    
    all_features = []
    all_labels = []
    
    print("Extracting features into RAM (This takes ~4 minutes)...")
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
    
    # Split into train/val
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
    print("Training specialized classification head...")
    for epoch in range(epochs):
        model.fusion_head.train()
        total_loss, correct, total = 0, 0, 0
        
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
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
                outputs = model.fusion_head(feats)
                val_loss += criterion(outputs, labels).item()
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {correct/total*100:.2f}% | Val Acc: {val_correct/val_total*100:.2f}%")
        
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model_g_clip_combined.pt')
    print("Saved combined Face+Generic model to checkpoints/model_g_clip_combined.pt")

if __name__ == "__main__":
    train_clip_combined()
