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

def train_clip_10k_fast():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    print("Loading dataset (10,000 images)...")
    dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split='train')
    dataset = dataset.shuffle(seed=42).select(range(10000))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    class HFDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, transform):
            self.hf_ds = hf_ds
            self.transform = transform
            
        def __len__(self):
            return len(self.hf_ds)
            
        def __getitem__(self, idx):
            item = self.hf_ds[idx]
            image = item['image'].convert('RGB')
            # Dataset labels: 0 = AiArtData (Fake), 1 = RealArt (Real)
            # Our convention: 0 = Real, 1 = Fake
            label = 1 if item['label'] == 0 else 0
            return self.transform(image), torch.tensor([label], dtype=torch.float32)

    full_data = HFDataset(dataset, transform)
    loader = DataLoader(full_data, batch_size=64, shuffle=False)
    
    print("Initializing FusionModelV2 (CLIP + Frequency) for embedding extraction...")
    model = FusionModelV2(pretrained_spatial=True, freeze_clip=True).to(device)
    model.eval()
    
    all_features = []
    all_labels = []
    
    print("Extracting features (This takes ~3 minutes but makes training instant!)...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(device)
            # We bypass the fusion_head and manually extract features from spatial and frequency branches
            spatial_features = model.spatial(imgs, return_features=True)
            frequency_features = model.frequency(imgs, return_features=True)
            fused = torch.cat((spatial_features, frequency_features), dim=1)
            
            all_features.append(fused.cpu())
            all_labels.append(labels)
            
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    print(f"Extracted feature shape: {features_tensor.shape}")
    
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
    
    # Now we only train the fusion_head
    optimizer = optim.Adam(model.fusion_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    epochs = 15 # Because it runs instantly on pre-computed features!
    print("Training classifier head on 10k extracted embeddings...")
    for epoch in range(epochs):
        model.fusion_head.train()
        total_loss = 0
        correct = 0
        total = 0
        
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
            
        # Validation
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
                
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/total:.4f}, Train Acc: {correct/total*100:.2f}% | Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_correct/val_total*100:.2f}%")
        
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model_f_clip_10k.pt')
    print("Saved robust 10k CLIP model to checkpoints/model_f_clip_10k.pt")

if __name__ == "__main__":
    train_clip_10k_fast()
